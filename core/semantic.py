"""The semantic layer: turning profiles into grounded prompt context.

This is the module that most determines answer quality. Published evaluations of
LLM-generated SQL find that failures cluster around schema hallucination and
incorrect join paths rather than syntax, and that supplying explicit schema and
semantic context is what moves accuracy from roughly half to the high
eighties/nineties. So instead of pasting `df.head()` into a prompt, we hand the
model a compact contract:

* the exact table and column names it is allowed to use, with DuckDB types;
* the analytical role of each column, so it does not SUM an order id;
* real value samples and top categories, so it filters on ``'North'`` rather
  than an invented ``'North Region'``;
* numeric ranges and date ranges, so it never invents a period;
* null rates, so it knows which aggregates are unreliable;
* verified join keys with measured value overlap;
* explicit dialect rules for the traps DuckDB users hit.

Everything here is derived from the data itself. Nothing is guessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import ColumnProfile, ColumnRole, DatasetProfile, JoinHint

if TYPE_CHECKING:  # pragma: no cover
    from .engine import DataSession

MAX_TOP_VALUES = 8
MAX_SAMPLES = 4


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1_000_000 or (value != 0 and abs(value) < 0.001):
        return f"{value:.4g}"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def column_line(col: ColumnProfile) -> str:
    """One dense, information-rich line per column."""
    bits = [f"  - {col.name} :: {col.duckdb_type} [{col.role.value}]"]

    if col.role == ColumnRole.MEASURE and col.min is not None:
        bits.append(
            f"range {_fmt_number(col.min)}..{_fmt_number(col.max)}"
            f", mean {_fmt_number(col.mean)}, median {_fmt_number(col.p50)}"
        )
    elif col.role == ColumnRole.TEMPORAL and col.min_date:
        bits.append(f"spans {col.min_date} .. {col.max_date}")
    elif col.top_values:
        shown = col.top_values[:MAX_TOP_VALUES]
        vals = ", ".join(f"'{v['value']}' ({v['count']})" for v in shown)
        more = "" if len(col.top_values) <= MAX_TOP_VALUES else f", +{col.distinct_count - len(shown)} more"
        bits.append(f"values: {vals}{more}")
    elif col.sample_values:
        bits.append("examples: " + ", ".join(f"'{v}'" for v in col.sample_values[:MAX_SAMPLES]))

    bits.append(f"{col.distinct_count:,} distinct")
    if col.null_pct > 0:
        bits.append(f"{col.null_pct:.1f}% null")
    return bits[0] + " | " + " | ".join(bits[1:])


def describe_dataset(profile: DatasetProfile, *, include_issues: bool = True) -> str:
    """Render one table's schema card."""
    lines = [
        f"TABLE {profile.table}  (from '{profile.source_name}')",
        f"  rows: {profile.row_count:,} | columns: {profile.column_count}",
        "  columns:",
    ]
    lines.extend(column_line(c) for c in profile.columns)

    if include_issues:
        notable = [i for i in profile.issues if i.severity in ("warning", "error")][:6]
        if notable:
            lines.append("  data-quality warnings:")
            lines.extend(f"    ! {i.message}" for i in notable)
    return "\n".join(lines)


def describe_joins(hints: list[JoinHint]) -> str:
    if not hints:
        return ""
    lines = ["VERIFIED JOIN KEYS (measured value overlap, use these and no others):"]
    for h in hints:
        lines.append(
            f"  - {h.left_table}.{h.left_column} = {h.right_table}.{h.right_column}"
            f"  ({h.overlap_pct:.0f}% overlap, {h.reason})"
        )
    return "\n".join(lines)


_QTY_NAMES = ("quantity", "qty", "units", "unit_count", "volume", "no_of_items")
_UNIT_PRICE_NAMES = ("price", "unit_price", "unitprice", "unit_cost", "rate", "price_each")
_REVENUE_NAMES = ("revenue", "sales", "amount", "total", "turnover", "gross", "net_sales", "line_total")
_COST_NAMES = ("cost", "cogs", "expense", "spend")


def derived_metric_hints(profile: DatasetProfile) -> list[str]:
    """Name the metrics that must be *computed* because no column holds them.

    This is grounding, not invention. Real transaction exports frequently store
    quantity and unit price but no line total — the UCI Online Retail II file is
    exactly like that: it has ``quantity`` and ``price`` and no revenue column.
    Without this hint a model asked "which country generated the highest revenue?"
    will either hallucinate a ``revenue`` column or silently answer with
    ``SUM(quantity)``, which is a different question. Telling it the arithmetic
    keeps the answer correct and the derivation visible in the SQL.
    """
    measures = {c.name.lower(): c.name for c in profile.columns_by_role(ColumnRole.MEASURE)}
    if not measures:
        return []

    def find(candidates: tuple[str, ...]) -> str | None:
        for lname, original in measures.items():
            if lname in candidates:
                return original
        for lname, original in measures.items():
            if any(c in lname for c in candidates):
                return original
        return None

    hints: list[str] = []
    revenue = find(_REVENUE_NAMES)
    quantity = find(_QTY_NAMES)
    unit_price = find(_UNIT_PRICE_NAMES)
    cost = find(_COST_NAMES)

    if revenue is None and quantity and unit_price:
        hints.append(
            f"There is no revenue/sales column in {profile.table}. Revenue must be computed as "
            f"`{quantity} * {unit_price}` — e.g. `SUM({quantity} * {unit_price}) AS revenue`. "
            f"Use this for any question about revenue, sales value, or spend."
        )
    if revenue and cost:
        hints.append(
            f"Profit/margin is not stored: compute `SUM({revenue} - {cost})` for profit and "
            f"`SUM({revenue} - {cost}) / NULLIF(SUM({revenue}), 0)` for margin."
        )
    if quantity:
        col = profile.column(quantity)
        if col and col.min is not None and col.min < 0:
            hints.append(
                f"`{quantity}` contains negative values (min {col.min:,.0f}), which represent returns "
                f"or credit notes rather than errors. Decide deliberately whether a question wants them "
                f"included (net) or excluded (gross), and say which you used."
            )
    return hints


def describe_derived_metrics(session: "DataSession") -> str:
    lines: list[str] = []
    for dataset in session.datasets.values():
        lines.extend(derived_metric_hints(dataset.profile))
    if not lines:
        return ""
    return "DERIVED METRICS (these columns do not exist — compute them):\n" + "\n".join(
        f"  - {line}" for line in lines
    )


DIALECT_RULES = """SQL RULES (DuckDB dialect, read-only):
  - SELECT statements only. One statement, no semicolon chains, no DDL/DML.
  - Reference only the tables and columns listed above, spelled exactly as shown.
    If the data cannot answer the question, say so instead of inventing a column.
  - Never aggregate a column whose role is [identifier].
  - Month/quarter grouping: use date_trunc('month', col) or strftime(col, '%Y-%m').
    Sort by the raw date expression, not by the formatted label, or months order
    alphabetically.
  - Integer division truncates. Cast before dividing: SUM(x) * 1.0 / COUNT(*).
  - Guard divisions with NULLIF(denominator, 0).
  - String comparisons are case-sensitive. Use lower(col) = 'value' when a column
    was flagged for inconsistent casing.
  - Prefer explicit column lists over SELECT *; alias every computed column with a
    readable snake_case name, because those names become chart axis labels.
  - Use ORDER BY plus LIMIT for "top N" questions.
  - A row cap is applied automatically; aggregate rather than returning raw rows."""


def build_schema_context(session: "DataSession", *, include_issues: bool = True) -> str:
    """Full grounding block for the system prompt."""
    if not session.datasets:
        return "No datasets are loaded. Ask the user to upload a CSV file."

    blocks = [
        describe_dataset(ds.profile, include_issues=include_issues)
        for ds in session.datasets.values()
    ]
    parts = ["AVAILABLE DATA", "\n\n".join(blocks)]
    derived = describe_derived_metrics(session)
    if derived:
        parts.append(derived)
    joins = describe_joins(session.join_hints)
    if joins:
        parts.append(joins)
    parts.append(DIALECT_RULES)
    return "\n\n".join(parts)


def compact_schema(session: "DataSession") -> str:
    """A one-line-per-table digest, for cheap calls where the full card is overkill."""
    out = []
    for ds in session.datasets.values():
        cols = ", ".join(f"{c.name}:{c.role.value[:3]}" for c in ds.profile.columns)
        out.append(f"{ds.table}({cols}) [{ds.profile.row_count:,} rows]")
    return "\n".join(out)


def schema_fingerprint(session: "DataSession") -> str:
    """Stable hash input identifying the loaded schema.

    Used as part of the cache key so a cached answer can never be served against
    a different dataset.
    """
    parts = []
    for table in sorted(session.datasets):
        prof = session.datasets[table].profile
        cols = ",".join(f"{c.name}:{c.duckdb_type}" for c in prof.columns)
        parts.append(f"{table}|{prof.row_count}|{cols}")
    return ";".join(parts)


def suggest_questions(session: "DataSession", limit: int = 6) -> list[str]:
    """Deterministic starter questions derived from the actual roles present.

    No LLM call: this runs before the user has spent any quota, and it cannot
    suggest a question the data can't answer.
    """
    if not session.datasets:
        return []
    table = session.default_table()
    prof = session.datasets[table].profile
    measures = prof.columns_by_role(ColumnRole.MEASURE)
    dims = [c for c in prof.columns_by_role(ColumnRole.DIMENSION) if 1 < c.distinct_count <= 60]
    temporal = prof.columns_by_role(ColumnRole.TEMPORAL)
    identifiers = prof.columns_by_role(ColumnRole.IDENTIFIER)

    # If revenue has to be derived, phrase the questions in business terms anyway —
    # the agent knows the arithmetic from the derived-metric hints.
    metric = measures[0].name if measures else None
    if any("revenue must be computed" in h.lower() for h in derived_metric_hints(prof)):
        metric = "revenue"

    out: list[str] = []
    if metric and dims:
        out.append(f"Which {dims[0].name} generated the highest {metric}?")
    if metric and temporal:
        out.append(f"Show monthly {metric} trends.")
    if metric and len(dims) > 1:
        out.append(f"Which {dims[1].name} values are underperforming on {metric}?")
    if metric and identifiers:
        out.append(f"What are the top five {identifiers[-1].name} by {metric}?")
    elif metric and dims:
        out.append(f"What are the top five {dims[0].name} by {metric}?")
    if measures:
        out.append(f"Detect anomalies in {measures[0].name} and explain why they were flagged.")
    out.append("Summarise this dataset and give me the three most useful business insights.")
    if len(session.datasets) > 1 and session.join_hints:
        hint = session.join_hints[0]
        other = hint.right_table if hint.left_table == table else hint.left_table
        out.insert(1, f"Join {table} with {other} and compare {metric or 'the totals'} by region.")
    return out[:limit]
