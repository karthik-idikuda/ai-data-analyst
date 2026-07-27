"""Automatic dashboard generation.

Fully deterministic: KPIs and panel definitions are derived from the profiled
column roles and computed with SQL. No LLM is involved, so the dashboard works
with no API key, costs nothing, renders in well under a second, and shows the same
numbers every time.

The layout logic is deliberately simple and explainable:

* **KPIs** — row count, plus each measure's total and average, plus the span of the
  primary date column, plus the cardinality of the leading dimension.
* **Trend panel** — the primary measure aggregated by month over the primary date
  column, when one exists.
* **Ranking panels** — the primary measure by each usable dimension, top 10.
* **Composition panel** — share of the primary measure across the leading
  dimension, when its cardinality is small enough for a pie to be readable.
* **Distribution panel** — a histogram of the primary measure.

Every panel carries the SQL that produced it, so a dashboard is auditable in the
same way a chat answer is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine import DataSession
from .models import ChartSpec, ChartType, ColumnRole, DatasetProfile
from .observability import get_logger
from .semantic import derived_metric_hints

log = get_logger(__name__)

MAX_RANKING_PANELS = 3
MAX_PIE_CATEGORIES = 8
MAX_DIMENSION_CARDINALITY = 200


@dataclass
class Kpi:
    label: str
    value: str
    help: str = ""


@dataclass
class Panel:
    title: str
    spec: ChartSpec
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "spec": self.spec.model_dump(mode="json"),
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "note": self.note,
        }


@dataclass
class Dashboard:
    table: str
    kpis: list[Kpi] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    measure_expression: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "measure_expression": self.measure_expression,
            "kpis": [{"label": k.label, "value": k.value, "help": k.help} for k in self.kpis],
            "panels": [p.to_dict() for p in self.panels],
            "notes": self.notes,
        }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        magnitude = abs(value)
        if magnitude >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.2f}bn"
        if magnitude >= 1_000_000:
            return f"{value / 1_000_000:,.2f}m"
        if magnitude >= 1_000:
            return f"{value:,.0f}"
        if isinstance(value, int) or float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def _primary_measure(profile: DatasetProfile) -> tuple[str, str]:
    """Return (sql_expression, label) for the headline metric.

    When the table stores quantity and unit price but no revenue column — the shape
    of the real UCI retail file — the headline metric is the product of the two.
    Showing 'total quantity' as the headline of a sales dashboard would be
    misleading, so the derived expression is used and labelled as derived.
    """
    hints = " ".join(derived_metric_hints(profile)).lower()
    measures = profile.columns_by_role(ColumnRole.MEASURE)
    if not measures:
        return "COUNT(*)", "records"

    if "revenue must be computed" in hints:
        quantity = next((c.name for c in measures if "quantit" in c.name.lower() or c.name.lower() == "qty"), None)
        price = next((c.name for c in measures if "price" in c.name.lower()), None)
        if quantity and price:
            return f"{quantity} * {price}", "revenue"
    return measures[0].name, measures[0].name


def _usable_dimensions(profile: DatasetProfile) -> list[str]:
    return [
        c.name
        for c in profile.columns_by_role(ColumnRole.DIMENSION)
        if 1 < c.distinct_count <= MAX_DIMENSION_CARDINALITY
    ]


def build(session: DataSession, table: str | None = None) -> Dashboard:
    """Build a dashboard for one table. Executes several small aggregate queries."""
    target = table or session.default_table()
    dataset = session.get_dataset(target)
    profile = dataset.profile

    expression, label = _primary_measure(profile)
    dashboard = Dashboard(table=target, measure_expression=expression)
    if expression != label and expression != "COUNT(*)":
        dashboard.notes.append(
            f"'{label}' is not stored in this table; it is computed as `{expression}`."
        )

    temporal = profile.columns_by_role(ColumnRole.TEMPORAL)
    dimensions = _usable_dimensions(profile)
    date_column = temporal[0].name if temporal else None

    # ---------------------------------------------------------------- KPIs
    dashboard.kpis.append(Kpi("Rows", _fmt(profile.row_count), f"in {target}"))

    if expression != "COUNT(*)":
        try:
            frame = session.run_dataframe_query(
                f"SELECT SUM({expression}) AS total, AVG({expression}) AS average, "
                f"COUNT(*) AS n FROM {target}",
                max_rows=1,
            )
            if not frame.empty:
                dashboard.kpis.append(Kpi(f"Total {label}", _fmt(float(frame.at[0, 'total'] or 0))))
                dashboard.kpis.append(Kpi(f"Average {label}", _fmt(float(frame.at[0, 'average'] or 0)),
                                          "per row"))
        except Exception as exc:  # noqa: BLE001 - a missing KPI must not fail the dashboard
            log.info("dashboard.kpi_failed", error=str(exc)[:200])
            dashboard.notes.append(f"Could not compute totals for {label}.")

    if dimensions:
        first = profile.column(dimensions[0])
        if first:
            dashboard.kpis.append(
                Kpi(f"Distinct {dimensions[0]}", _fmt(first.distinct_count))
            )
    if temporal:
        span = temporal[0]
        dashboard.kpis.append(
            Kpi("Period", f"{(span.min_date or '')[:10]} → {(span.max_date or '')[:10]}",
                f"from {span.name}")
        )

    identifiers = profile.columns_by_role(ColumnRole.IDENTIFIER)
    if identifiers:
        ident = identifiers[-1]
        dashboard.kpis.append(Kpi(f"Distinct {ident.name}", _fmt(ident.distinct_count)))

    # -------------------------------------------------------------- panels
    def add_panel(title: str, sql: str, spec: ChartSpec, note: str = "") -> None:
        try:
            frame = session.run_dataframe_query(sql, max_rows=500)
        except Exception as exc:  # noqa: BLE001
            log.info("dashboard.panel_failed", title=title, error=str(exc)[:200])
            dashboard.notes.append(f"Panel '{title}' could not be computed.")
            return
        if frame.empty:
            return
        from .engine import _jsonable_rows

        dashboard.panels.append(
            Panel(
                title=title,
                spec=spec,
                sql=sql,
                columns=[str(c) for c in frame.columns],
                rows=_jsonable_rows(frame),
                note=note,
            )
        )

    if date_column and expression != "COUNT(*)":
        add_panel(
            f"{label.title()} by month",
            f"SELECT strftime(CAST({date_column} AS TIMESTAMP), '%Y-%m') AS month, "
            f"SUM({expression}) AS {label} FROM {target} "
            f"WHERE {date_column} IS NOT NULL GROUP BY 1 ORDER BY 1",
            ChartSpec(type=ChartType.LINE, x="month", y=label, sort="x_asc",
                      title=f"{label.title()} by month"),
        )

    for dimension in dimensions[:MAX_RANKING_PANELS]:
        add_panel(
            f"Top {dimension} by {label}",
            f"SELECT {dimension}, SUM({expression}) AS {label} FROM {target} "
            f"WHERE {dimension} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
            ChartSpec(type=ChartType.HORIZONTAL_BAR, x=dimension, y=label, sort="y_desc",
                      title=f"Top 10 {dimension} by {label}"),
        )

    if dimensions:
        leading = profile.column(dimensions[0])
        if leading and leading.distinct_count <= MAX_PIE_CATEGORIES:
            add_panel(
                f"{label.title()} share by {dimensions[0]}",
                f"SELECT {dimensions[0]}, SUM({expression}) AS {label} FROM {target} "
                f"WHERE {dimensions[0]} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC",
                ChartSpec(type=ChartType.PIE, x=dimensions[0], y=label,
                          title=f"{label.title()} share by {dimensions[0]}"),
                note="Negative totals cannot be shown as pie slices.",
            )

    measures = profile.columns_by_role(ColumnRole.MEASURE)
    if measures:
        column = measures[0].name
        add_panel(
            f"Distribution of {column}",
            f"SELECT {column} FROM {target} WHERE {column} IS NOT NULL LIMIT 5000",
            ChartSpec(type=ChartType.HISTOGRAM, x=column, title=f"Distribution of {column}"),
            note="Based on the first 5,000 non-null values.",
        )

    if not dashboard.panels:
        dashboard.notes.append(
            "No chartable combination of a dimension, date or measure was found in this table."
        )

    log.info("dashboard.built", table=target, kpis=len(dashboard.kpis), panels=len(dashboard.panels))
    return dashboard
