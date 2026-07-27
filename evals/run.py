#!/usr/bin/env python3
"""Evaluation harness.

Runs the golden question set against the real datasets and reports an accuracy
score, per-check detail, latency and token cost. Requires a configured LLM
provider, since it evaluates end-to-end agent behaviour.

    python evals/run.py                    # everything
    python evals/run.py --tag honesty      # one slice
    python evals/run.py --case top_country_revenue
    python evals/run.py --json results.json

Exit code is non-zero when the pass rate falls below ``--threshold``, so this can
gate a release.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent import Agent
from core.cache import ANSWER_CACHE
from core.config import get_settings
from core.engine import DataSession
from core.errors import AnalystError
from core.observability import configure_logging
from evals.cases import GOLDEN_SET, EvalCase
from evals.checks import CheckResult
from evals.ground_truth import COUNTRY_CSV, RETAIL_CSV, compute_all

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass
class CaseOutcome:
    id: str
    question: str
    tags: list[str]
    passed: bool
    checks: list[dict] = field(default_factory=list)
    answer: str = ""
    sql: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


def build_session() -> DataSession:
    if not RETAIL_CSV.exists() or not COUNTRY_CSV.exists():
        raise SystemExit(
            "Real datasets are missing. Run: python scripts/fetch_real_data.py"
        )
    session = DataSession()
    session.add_csv_path(RETAIL_CSV)
    session.add_csv_path(COUNTRY_CSV)
    return session


def run_case(case: EvalCase, truths: dict, *, verbose: bool) -> CaseOutcome:
    # A fresh session per case: conversational state must not leak between cases.
    session = build_session()
    ANSWER_CACHE.clear()
    truth = truths.get(case.truth_key) if case.truth_key else None
    started = time.perf_counter()

    try:
        answer = Agent().answer(session, case.question, use_cache=False)
    except AnalystError as exc:
        session.close()
        return CaseOutcome(
            id=case.id, question=case.question, tags=case.tags, passed=False,
            duration_s=time.perf_counter() - started, error=f"{exc.code}: {exc.message}",
        )

    results: list[CheckResult] = [check.run(answer, truth) for check in case.checks]

    if case.follow_up:
        try:
            follow = Agent().answer(session, case.follow_up, use_cache=False)
            results.extend(
                CheckResult(f"[follow-up] {r.name}", r.passed, r.detail)
                for r in (c.run(follow, truth) for c in case.follow_up_checks)
            )
        except AnalystError as exc:
            results.append(CheckResult("[follow-up] completed", False, str(exc)))

    session.close()
    trace = answer.trace or {}
    outcome = CaseOutcome(
        id=case.id,
        question=case.question,
        tags=case.tags,
        passed=all(r.passed for r in results),
        checks=[asdict(r) for r in results],
        answer=answer.answer_markdown,
        sql=answer.sql_executed,
        duration_s=round(time.perf_counter() - started, 2),
        tokens_in=trace.get("tokens_in", 0),
        tokens_out=trace.get("tokens_out", 0),
    )

    icon = f"{GREEN}PASS{RESET}" if outcome.passed else f"{RED}FAIL{RESET}"
    print(f"  [{icon}] {case.id:<28} {outcome.duration_s:>6.1f}s  {outcome.tokens_in + outcome.tokens_out:>6,} tok")
    for result in results:
        if not result.passed:
            print(f"        {RED}✗{RESET} {result.name} — {result.detail}")
        elif verbose:
            print(f"        {GREEN}✓{RESET} {result.name}")
    if verbose and outcome.answer:
        print(f"{DIM}        answer: {outcome.answer[:220]}{RESET}")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", help="only run cases with this tag")
    parser.add_argument("--case", help="only run this case id")
    parser.add_argument("--json", dest="json_path", help="write full results to this file")
    parser.add_argument("--threshold", type=float, default=0.80, help="minimum pass rate (default 0.80)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    if not settings.llm_configured:
        print(
            f"{RED}No LLM provider configured.{RESET} The harness evaluates end-to-end agent "
            "behaviour, so it needs LLM_PROVIDER and LLM_API_KEY in .env.\n"
            "The deterministic cross-check (pandas vs DuckDB) runs without a key: "
            "pytest tests/test_evals.py",
            file=sys.stderr,
        )
        return 2

    cases = GOLDEN_SET
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if args.case:
        cases = [c for c in cases if c.id == args.case]
    if not cases:
        print("No cases matched.", file=sys.stderr)
        return 2

    print(f"\nAI Data Analyst — evaluation")
    print(f"provider: {settings.llm_provider} · model: {settings.default_model}")
    print(f"data: {RETAIL_CSV.name}, {COUNTRY_CSV.name}")
    print(f"cases: {len(cases)}\n")

    print("Computing ground truth with pandas (independent of the SQL path)…")
    truths = compute_all()
    print(f"  {len(truths)} verified facts\n")

    outcomes = [run_case(case, truths, verbose=args.verbose) for case in cases]

    passed = sum(1 for o in outcomes if o.passed)
    rate = passed / len(outcomes)
    total_checks = sum(len(o.checks) for o in outcomes)
    passed_checks = sum(1 for o in outcomes for c in o.checks if c["passed"])

    print("\n" + "─" * 72)
    print(f"cases   {passed}/{len(outcomes)} passed  ({rate:.0%})")
    print(f"checks  {passed_checks}/{total_checks} passed  ({passed_checks / max(total_checks, 1):.0%})")
    print(f"latency median {sorted(o.duration_s for o in outcomes)[len(outcomes) // 2]:.1f}s"
          f" · total {sum(o.duration_s for o in outcomes):.0f}s")
    print(f"tokens  {sum(o.tokens_in for o in outcomes):,} in · {sum(o.tokens_out for o in outcomes):,} out")

    by_tag: dict[str, list[bool]] = {}
    for outcome in outcomes:
        for tag in outcome.tags:
            by_tag.setdefault(tag, []).append(outcome.passed)
    print("\nby tag:")
    for tag, flags in sorted(by_tag.items()):
        print(f"  {tag:<22} {sum(flags)}/{len(flags)}")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "provider": settings.llm_provider,
                    "model": settings.default_model,
                    "pass_rate": rate,
                    "check_pass_rate": passed_checks / max(total_checks, 1),
                    "ground_truth": {k: {"value": v.value, "derivation": v.derivation} for k, v in truths.items()},
                    "cases": [asdict(o) for o in outcomes],
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {args.json_path}")

    if rate < args.threshold:
        print(f"\n{RED}pass rate {rate:.0%} is below the {args.threshold:.0%} threshold{RESET}")
        return 1
    print(f"\n{GREEN}pass rate {rate:.0%} meets the {args.threshold:.0%} threshold{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
