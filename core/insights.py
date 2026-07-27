"""Business insight generation.

Two clearly separated halves:

``compute_facts``
    Deterministic. Runs real SQL against the loaded data to produce concentration
    ratios, period-over-period movement, top and bottom segments, spread and
    quality figures. Reproducible and testable, and it works with no API key.

``narrate``
    The LLM turns those verified facts into an executive briefing. It is given the
    numbers and told to add none of its own, so the prose cannot drift from the
    data. This path uses no tools, which makes it the natural place for genuine
    token-level streaming.
"""

from __future__ import annotations

from typing import Any, Iterator

import pandas as pd

from .engine import DataSession
from .errors import LLMNotConfiguredError
from .llm import Message, get_provider
from .models import ColumnRole
from .observability import get_logger
from .profile import quality_score
from .prompts import INSIGHTS_SYSTEM

log = get_logger(__name__)


def _safe(session: DataSession, sql: str) -> pd.DataFrame | None:
    try:
        return session.run_dataframe_query(sql, max_rows=200)
    except Exception as exc:  # noqa: BLE001 - a failed fact is skipped, never fatal
        log.info("insights.fact_failed", error=str(exc)[:200])
        return None


def compute_facts(session: DataSession, table: str | None = None) -> dict[str, Any]:
    """Gather verified statistics about a table. No LLM involved."""
    target = table or session.default_table()
    dataset = session.get_dataset(target)
    profile = dataset.profile

    measures = profile.columns_by_role(ColumnRole.MEASURE)
    dims = [c for c in profile.columns_by_role(ColumnRole.DIMENSION) if 1 < c.distinct_count <= 200]
    temporal = profile.columns_by_role(ColumnRole.TEMPORAL)

    facts: dict[str, Any] = {
        "table": profile.table,
        "source_file": profile.source_name,
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "quality": quality_score(profile),
        "columns": [
            {
                "name": c.name,
                "type": c.duckdb_type,
                "role": c.role.value,
                "distinct": c.distinct_count,
                "null_pct": c.null_pct,
            }
            for c in profile.columns
        ],
        "measure_summary": [
            {
                "column": m.name,
                "min": m.min, "max": m.max, "mean": m.mean,
                "median": m.p50, "std": m.std,
                "spread_ratio": round(m.max / m.min, 2) if m.min and m.min > 0 and m.max else None,
            }
            for m in measures[:6]
        ],
        "date_ranges": [
            {"column": t.name, "from": t.min_date, "to": t.max_date} for t in temporal[:3]
        ],
        "segments": [],
        "trend": None,
        "quality_issues": [
            i.message for i in profile.issues if i.severity in ("warning", "error")
        ][:8],
    }

    # ---- concentration by dimension ---------------------------------------
    if measures and dims:
        measure = measures[0].name
        for dim in dims[:3]:
            frame = _safe(
                session,
                f'SELECT "{dim.name}" AS segment, SUM("{measure}") AS total, COUNT(*) AS n '
                f'FROM {target} WHERE "{dim.name}" IS NOT NULL '
                f'GROUP BY 1 ORDER BY total DESC',
            )
            if frame is None or frame.empty:
                continue
            total = float(frame["total"].sum())
            if not total:
                continue
            top = frame.iloc[0]
            bottom = frame.iloc[-1]
            top3_share = round(100 * float(frame["total"].head(3).sum()) / total, 1)
            facts["segments"].append(
                {
                    "dimension": dim.name,
                    "measure": measure,
                    "distinct_segments": int(len(frame)),
                    "grand_total": round(total, 2),
                    "top": {"name": str(top["segment"]), "total": round(float(top["total"]), 2),
                            "share_pct": round(100 * float(top["total"]) / total, 1),
                            "rows": int(top["n"])},
                    "bottom": {"name": str(bottom["segment"]), "total": round(float(bottom["total"]), 2),
                               "share_pct": round(100 * float(bottom["total"]) / total, 1),
                               "rows": int(bottom["n"])},
                    "top3_share_pct": top3_share,
                    "ratio_top_to_bottom": round(float(top["total"]) / float(bottom["total"]), 2)
                    if float(bottom["total"]) else None,
                }
            )

    # ---- period movement ---------------------------------------------------
    if measures and temporal:
        measure, date_col = measures[0].name, temporal[0].name
        frame = _safe(
            session,
            f'SELECT strftime(CAST("{date_col}" AS TIMESTAMP), \'%Y-%m\') AS period, '
            f'SUM("{measure}") AS total FROM {target} '
            f'WHERE "{date_col}" IS NOT NULL GROUP BY 1 ORDER BY 1',
        )
        if frame is not None and len(frame) >= 2:
            values = frame["total"].astype("float64")
            first, last = float(values.iloc[0]), float(values.iloc[-1])
            best = frame.loc[values.idxmax()]
            worst = frame.loc[values.idxmin()]
            change = None
            if first:
                change = round(100 * (last - first) / abs(first), 1)
            latest_vs_prev = None
            if len(values) >= 2 and float(values.iloc[-2]):
                latest_vs_prev = round(
                    100 * (last - float(values.iloc[-2])) / abs(float(values.iloc[-2])), 1
                )
            facts["trend"] = {
                "measure": measure,
                "date_column": date_col,
                "periods": int(len(frame)),
                "first_period": str(frame["period"].iloc[0]),
                "last_period": str(frame["period"].iloc[-1]),
                "first_value": round(first, 2),
                "last_value": round(last, 2),
                "change_first_to_last_pct": change,
                "latest_vs_previous_pct": latest_vs_prev,
                "best_period": {"period": str(best["period"]), "total": round(float(best["total"]), 2)},
                "worst_period": {"period": str(worst["period"]), "total": round(float(worst["total"]), 2)},
                "series": [
                    {"period": str(p), "total": round(float(v), 2)}
                    for p, v in zip(frame["period"], values)
                ][-24:],
            }

    return facts


def _facts_prompt(facts: dict[str, Any]) -> str:
    import json

    return (
        "Verified statistics for this dataset (computed by SQL and profiling, all authoritative):\n\n"
        + json.dumps(facts, indent=2, default=str)[:14_000]
    )


def narrate(facts: dict[str, Any], *, stream: bool = False) -> str | Iterator[str]:
    """Turn verified facts into a briefing. Raises if no provider is configured."""
    provider = get_provider()
    if provider.name == "none":
        raise LLMNotConfiguredError(
            "Insight narration needs an LLM provider.",
            detail="The computed statistics are still available in the dataset panel.",
        )
    messages = [Message(role="user", content=_facts_prompt(facts))]
    if stream:
        return provider.stream(messages, system=INSIGHTS_SYSTEM)
    return provider.chat(messages, system=INSIGHTS_SYSTEM).text


def deterministic_summary(facts: dict[str, Any]) -> str:
    """A no-LLM fallback briefing, so the feature degrades instead of disappearing."""
    lines = [
        f"**{facts['table']}** — {facts['row_count']:,} rows x {facts['column_count']} columns "
        f"(source: {facts['source_file']}).",
        f"Data-quality score {facts['quality']['score']}/100 "
        f"(completeness {facts['quality']['completeness_pct']}%, "
        f"row uniqueness {facts['quality']['uniqueness_pct']}%).",
    ]
    for rng in facts.get("date_ranges", []):
        lines.append(f"`{rng['column']}` spans {rng['from']} to {rng['to']}.")

    for seg in facts.get("segments", [])[:3]:
        lines.append(
            f"By **{seg['dimension']}**: {seg['distinct_segments']} segments, total "
            f"{seg['measure']} {seg['grand_total']:,.2f}. Largest is '{seg['top']['name']}' at "
            f"{seg['top']['total']:,.2f} ({seg['top']['share_pct']}%); smallest is "
            f"'{seg['bottom']['name']}' at {seg['bottom']['total']:,.2f}. "
            f"Top 3 hold {seg['top3_share_pct']}%."
        )

    trend = facts.get("trend")
    if trend:
        direction = "up" if (trend["change_first_to_last_pct"] or 0) >= 0 else "down"
        lines.append(
            f"**{trend['measure']}** across {trend['periods']} months "
            f"({trend['first_period']} to {trend['last_period']}) moved {direction} "
            f"{abs(trend['change_first_to_last_pct'] or 0)}% from first to last period. "
            f"Best month {trend['best_period']['period']} ({trend['best_period']['total']:,.2f}); "
            f"weakest {trend['worst_period']['period']} ({trend['worst_period']['total']:,.2f})."
        )

    if facts.get("quality_issues"):
        lines.append("**Quality flags:** " + " ".join(f"({i + 1}) {m}" for i, m in enumerate(facts["quality_issues"][:5])))

    lines.append("_Computed without an LLM. Configure a provider for a narrative briefing._")
    return "\n\n".join(lines)
