"""Schema inspection, data-quality and data-dictionary search tools."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from ..errors import ToolError
from ..models import Artifact, ColumnRole
from ..profile import quality_score
from ..semantic import describe_dataset, describe_joins
from .base import Tool, ToolOutcome, integer_param, object_schema, string_param

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import DataSession


# --------------------------------------------------------------------------- #
# inspect_schema
# --------------------------------------------------------------------------- #
def _inspect_schema(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    table = (args.get("table") or "").strip()
    if table:
        dataset = session.get_dataset(table)
        text = describe_dataset(dataset.profile)
        artifacts = [
            Artifact(
                kind="profile",
                title=f"Schema: {dataset.table}",
                payload=dataset.profile.model_dump(mode="json"),
            )
        ]
    else:
        blocks = [describe_dataset(ds.profile) for ds in session.datasets.values()]
        joins = describe_joins(session.join_hints)
        text = "\n\n".join(blocks + ([joins] if joins else []))
        artifacts = [
            Artifact(
                kind="profile",
                title=f"Schema: {ds.table}",
                payload=ds.profile.model_dump(mode="json"),
            )
            for ds in session.datasets.values()
        ]
    return ToolOutcome(
        model_text=text,
        artifacts=artifacts,
        reasoning="Re-read the schema and column statistics.",
    )


INSPECT_SCHEMA = Tool(
    name="inspect_schema",
    description=(
        "Re-read the full schema of one or all loaded tables, including column types, roles, "
        "ranges, distinct counts, null rates and sample values. Use when you are unsure whether a "
        "column exists or what values it holds — never guess a column name."
    ),
    parameters=object_schema(
        {"table": string_param("Table name. Omit for every loaded table.")},
    ),
    handler=_inspect_schema,
)


# --------------------------------------------------------------------------- #
# data_quality_report
# --------------------------------------------------------------------------- #
def _data_quality(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    table = (args.get("table") or "").strip() or session.default_table()
    dataset = session.get_dataset(table)
    profile = dataset.profile
    score = quality_score(profile)

    by_severity: dict[str, list[str]] = {"error": [], "warning": [], "info": []}
    for issue in profile.issues:
        by_severity[issue.severity].append(issue.message)

    lines = [
        f"Data-quality report for {profile.table} ({profile.row_count:,} rows x {profile.column_count} columns).",
        f"Overall score {score['score']}/100 — completeness {score['completeness_pct']}%, "
        f"row uniqueness {score['uniqueness_pct']}%.",
        f"{score['null_cells']:,} of {score['total_cells']:,} cells are null. "
        f"{profile.duplicate_row_count} duplicate row(s).",
    ]
    for severity in ("error", "warning", "info"):
        items = by_severity[severity]
        if items:
            lines.append(f"{severity.upper()} ({len(items)}):")
            lines.extend(f"  - {m}" for m in items[:15])
            if len(items) > 15:
                lines.append(f"  - … {len(items) - 15} more")

    artifacts = [
        Artifact(
            kind="quality",
            title=f"Data quality: {profile.table}",
            payload={
                "table": profile.table,
                "score": score,
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "duplicate_row_count": profile.duplicate_row_count,
                "issues": [i.model_dump(mode="json") for i in profile.issues],
                "columns": [
                    {
                        "name": c.name,
                        "type": c.duckdb_type,
                        "role": c.role.value,
                        "null_pct": c.null_pct,
                        "distinct_count": c.distinct_count,
                    }
                    for c in profile.columns
                ],
            },
        )
    ]
    return ToolOutcome(
        model_text="\n".join(lines),
        artifacts=artifacts,
        reasoning=f"Ran data-quality checks on {profile.table}.",
    )


DATA_QUALITY = Tool(
    name="data_quality_report",
    description=(
        "Assess completeness and consistency of a table: null rates, duplicates, constant columns, "
        "inconsistent casing, negative measures, and an overall score. Use when the user asks about "
        "data quality, reliability, missing values, or before trusting a suspicious result."
    ),
    parameters=object_schema(
        {"table": string_param("Table to assess. Omit to use the main table.")},
    ),
    handler=_data_quality,
)


# --------------------------------------------------------------------------- #
# search_columns
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "by", "to", "and", "or", "is", "are",
    "what", "which", "show", "me", "how", "many", "much", "total", "per", "with",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _STOPWORDS}


def _search_columns(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    query = (args.get("query") or "").strip()
    limit = int(args.get("limit") or 12)
    if not query:
        raise ToolError("The 'query' argument is required.")

    terms = _tokens(query)
    if not terms:
        raise ToolError("The query contained no searchable terms.")

    scored: list[tuple[float, str]] = []
    for dataset in session.datasets.values():
        for col in dataset.profile.columns:
            name_tokens = _tokens(col.name)
            score = 2.0 * len(terms & name_tokens)
            for term in terms:
                if any(term in nt or nt in term for nt in name_tokens):
                    score += 0.5
            # Match against actual category values too, so "north" finds `region`.
            value_pool = [str(v["value"]).lower() for v in col.top_values] + [
                s.lower() for s in col.sample_values
            ]
            for term in terms:
                if any(term in v for v in value_pool):
                    score += 1.5
            if score <= 0:
                continue
            detail = f"{dataset.table}.{col.name} :: {col.duckdb_type} [{col.role.value}]"
            if col.role == ColumnRole.MEASURE and col.min is not None:
                detail += f" range {col.min:,.2f}..{col.max:,.2f}"
            elif col.top_values:
                detail += " values: " + ", ".join(f"'{v['value']}'" for v in col.top_values[:6])
            elif col.sample_values:
                detail += " examples: " + ", ".join(f"'{v}'" for v in col.sample_values[:4])
            scored.append((score, detail))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored:
        return ToolOutcome(
            model_text=(
                f"Nothing in the loaded schema matches '{query}'. "
                "Tell the user this data cannot answer that question rather than inventing a column."
            ),
            reasoning=f"Searched the data dictionary for '{query}' — no match.",
        )

    lines = [f"Data-dictionary matches for '{query}' (lexical scoring over names and values):"]
    lines.extend(f"  {i + 1}. {detail}" for i, (_, detail) in enumerate(scored[:limit]))
    return ToolOutcome(
        model_text="\n".join(lines),
        reasoning=f"Searched the data dictionary for '{query}'.",
    )


SEARCH_COLUMNS = Tool(
    name="search_columns",
    description=(
        "Search the data dictionary for columns whose name or values relate to a phrase. Use on wide "
        "or unfamiliar datasets to locate the right column before writing SQL, and to confirm whether "
        "a concept exists in the data at all."
    ),
    parameters=object_schema(
        {
            "query": string_param("Concept to look for, e.g. 'customer country' or 'north'."),
            "limit": integer_param("Maximum matches to return.", minimum=1, maximum=40),
        },
        required=["query"],
    ),
    handler=_search_columns,
)
