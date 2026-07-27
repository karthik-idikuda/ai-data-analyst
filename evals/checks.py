"""Assertions used by the evaluation harness.

Grading a natural-language answer is the hard part of an eval framework. Exact
string matching fails on correct answers ("615,519.55" vs "£615.5k" vs
"about 616 thousand"), and asking a model to grade another model is both
expensive and unreliable.

The approach here: extract every number from the answer, normalise unit suffixes,
and check whether the expected value appears within a relative tolerance. That
accepts legitimate rounding while still catching a genuinely wrong figure.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from core.models import AgentAnswer

_NUMBER = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(k|thousand|m|mm|mn|million|bn|billion)?",
    re.IGNORECASE,
)
_MULTIPLIER = {
    None: 1.0, "": 1.0,
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
    "bn": 1e9, "billion": 1e9,
}


def extract_numbers(text: str) -> list[float]:
    """Every number in the text, with k/m/bn suffixes expanded."""
    found: list[float] = []
    for raw, suffix in _NUMBER.findall(text or ""):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        multiplier = _MULTIPLIER.get((suffix or "").lower(), 1.0)
        found.append(value * multiplier)
        if multiplier != 1.0:
            found.append(value)  # also accept the bare figure
    return found


def number_present(text: str, expected: float, *, rel_tolerance: float = 0.01) -> bool:
    """Is ``expected`` present, allowing for rounding and unit abbreviation?"""
    if expected == 0:
        return any(abs(n) < 1e-9 for n in extract_numbers(text))
    scale = abs(expected)
    for candidate in extract_numbers(text):
        if abs(candidate - expected) <= max(scale * rel_tolerance, 0.5):
            return True
        # Accept a figure rounded to thousands/millions, e.g. 615,519.55 -> 616 (k)
        for unit in (1e3, 1e6):
            if scale >= unit and abs(candidate * unit - expected) <= scale * 0.01:
                return True
    return False


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class Check(ABC):
    @abstractmethod
    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult: ...


@dataclass
class ContainsText(Check):
    """A required string must appear in the answer (case-insensitive)."""

    needle: str
    where: str = "answer"

    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        haystack = _haystack(answer, self.where)
        ok = self.needle.lower() in haystack.lower()
        return CheckResult(f"contains '{self.needle}'", ok, "" if ok else f"not found in {self.where}")


@dataclass
class ContainsTruthText(Check):
    """A string taken from the ground truth must appear in the answer."""

    path: str
    where: str = "answer"

    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        expected = str(_dig(truth, self.path))
        haystack = _haystack(answer, self.where)
        ok = expected.lower() in haystack.lower()
        return CheckResult(
            f"{self.path} == '{expected}'", ok, "" if ok else f"'{expected}' missing from {self.where}"
        )


@dataclass
class ContainsTruthNumber(Check):
    """A number taken from the ground truth must appear, within tolerance."""

    path: str
    rel_tolerance: float = 0.01

    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        expected = float(_dig(truth, self.path))
        ok = number_present(answer.answer_markdown, expected, rel_tolerance=self.rel_tolerance)
        return CheckResult(
            f"{self.path} ≈ {expected:,.2f}",
            ok,
            "" if ok else f"expected ≈{expected:,.2f}; answer had {extract_numbers(answer.answer_markdown)[:8]}",
        )


@dataclass
class HasArtifact(Check):
    kind: str

    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        kinds = [a.kind for a in answer.artifacts]
        ok = self.kind in kinds
        return CheckResult(f"produced a {self.kind}", ok, "" if ok else f"artifacts were {kinds}")


@dataclass
class SqlContains(Check):
    """The executed SQL must reference the right columns/constructs.

    Catches the answer that is textually plausible but computed from the wrong
    thing — e.g. summing ``quantity`` for a revenue question.
    """

    fragments: tuple[str, ...]

    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        sql = " ".join(answer.sql_executed).lower()
        missing = [f for f in self.fragments if f.lower() not in sql]
        return CheckResult(
            f"SQL uses {', '.join(self.fragments)}",
            not missing,
            "" if not missing else f"missing: {missing}",
        )


@dataclass
class SqlExecuted(Check):
    """No answer to a factual question is acceptable without a query behind it."""

    minimum: int = 1

    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        count = len(answer.sql_executed)
        ok = count >= self.minimum
        return CheckResult(f"ran ≥{self.minimum} query", ok, "" if ok else "no SQL was executed")


@dataclass
class Rejects(Check):
    """For unanswerable questions: the model must decline, not invent."""

    forbidden: tuple[str, ...] = ()
    admissions: tuple[str, ...] = (
        "cannot", "can't", "no ", "not available", "not present", "does not",
        "doesn't", "unable", "no column", "not in the data", "unavailable",
    )

    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        text = answer.answer_markdown.lower()
        admitted = any(phrase in text for phrase in self.admissions)
        invented = [f for f in self.forbidden if f.lower() in text]
        ok = admitted and not invented
        detail = ""
        if not admitted:
            detail = "did not acknowledge the limitation"
        elif invented:
            detail = f"mentioned fabricated field(s): {invented}"
        return CheckResult("declines unanswerable question", ok, detail)


@dataclass
class ReasoningPresent(Check):
    def run(self, answer: AgentAnswer, truth: Any) -> CheckResult:
        ok = bool(answer.reasoning)
        return CheckResult("explains its reasoning", ok, "" if ok else "reasoning trail was empty")


def _haystack(answer: AgentAnswer, where: str) -> str:
    if where == "answer":
        return answer.answer_markdown
    if where == "sql":
        return " ".join(answer.sql_executed)
    if where == "reasoning":
        return " ".join(answer.reasoning)
    if where == "all":
        return " ".join([answer.answer_markdown, *answer.sql_executed, *answer.reasoning])
    raise ValueError(f"unknown location '{where}'")


def _dig(truth: Any, path: str) -> Any:
    """Resolve a dotted path such as ``top.0.country`` against the truth value."""
    current = truth.value if hasattr(truth, "value") else truth
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            raise KeyError(f"cannot resolve '{part}' in {type(current).__name__}")
    return current
