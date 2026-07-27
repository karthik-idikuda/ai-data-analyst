"""Anomaly detection and forecasting tools.

Both wrap deterministic statistics. The model chooses *when* to run them and
narrates the output; it never produces the numbers.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .. import anomaly as anomaly_mod
from .. import forecast as forecast_mod
from ..errors import ToolError
from ..models import Artifact
from .base import Tool, ToolOutcome, array_param, integer_param, object_schema, string_param

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import DataSession

SENSITIVITY = {
    "low": (3.0, 4.5),      # (IQR k, robust-z threshold) — flags only extreme points
    "medium": (1.5, 3.5),   # Tukey's standard fence
    "high": (1.0, 2.5),     # flags more, expect false positives
}


def _detect_anomalies(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    table = (args.get("table") or "").strip() or session.default_table()
    dataset = session.get_dataset(table)

    columns = args.get("columns")
    if isinstance(columns, str):
        columns = [c.strip() for c in columns.split(",") if c.strip()]
    if columns is not None and not isinstance(columns, list):
        raise ToolError("'columns' must be an array of column names.")

    sensitivity = (args.get("sensitivity") or "medium").strip().lower()
    if sensitivity not in SENSITIVITY:
        raise ToolError(
            f"Unknown sensitivity '{sensitivity}'.",
            detail="Use 'low', 'medium' or 'high'.",
        )
    iqr_k, z_threshold = SENSITIVITY[sensitivity]

    report = anomaly_mod.analyse(
        dataset.frame,
        dataset.profile,
        columns=columns,
        iqr_k=iqr_k,
        z_threshold=z_threshold,
        max_per_method=int(args.get("max_results") or 10),
    )

    lines = [
        f"Anomaly scan of {report.table}: {report.rows_tested:,} rows, "
        f"columns tested: {', '.join(report.columns_tested) or 'none'}.",
        f"Methods that produced findings: {', '.join(m.value for m in report.methods_used) or 'none'} "
        f"(sensitivity '{sensitivity}': IQR k={iqr_k}, robust-z threshold={z_threshold}).",
        f"{len(report.anomalies)} anomaly record(s) found.",
    ]
    for note in report.notes:
        lines.append(f"Note: {note}")
    for i, item in enumerate(report.anomalies[:20], start=1):
        where = f" [{item.label}]" if item.label else ""
        lines.append(f"  {i}. ({item.method.value}) score={item.score}{where} {item.reason}")
    if len(report.anomalies) > 20:
        lines.append(f"  … {len(report.anomalies) - 20} more not listed.")
    lines.append(
        "When you explain these, state the method and threshold that flagged each one, and say plainly "
        "that a statistical outlier is not automatically an error — it may be a legitimate large order."
    )

    artifacts = [
        Artifact(
            kind="anomaly",
            title=f"Anomalies: {report.table}",
            payload=report.model_dump(mode="json"),
        )
    ]
    return ToolOutcome(
        model_text="\n".join(lines),
        artifacts=artifacts,
        reasoning=(
            f"Ran {len(report.methods_used)} statistical method(s) over "
            f"{len(report.columns_tested)} column(s) and found {len(report.anomalies)} anomaly record(s)."
        ),
    )


DETECT_ANOMALIES = Tool(
    name="detect_anomalies",
    description=(
        "Run statistical anomaly detection: Tukey IQR fences, robust z-score (median/MAD), "
        "multivariate Isolation Forest, and STL seasonal-residual analysis on dated series. Returns "
        "each flagged record with the method, the threshold and the observed value. Use whenever the "
        "user asks about anomalies, outliers, unusual values, spikes or suspicious data."
    ),
    parameters=object_schema(
        {
            "table": string_param("Table to scan. Omit to use the main table."),
            "columns": array_param("Numeric columns to test. Omit to test every measure column."),
            "sensitivity": string_param(
                "How aggressive to be. 'medium' is Tukey's standard fence.",
                enum=["low", "medium", "high"],
            ),
            "max_results": integer_param("Maximum findings per method.", minimum=1, maximum=50),
        },
    ),
    handler=_detect_anomalies,
)


def _forecast(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    table = (args.get("table") or "").strip() or session.default_table()
    dataset = session.get_dataset(table)

    date_column = (args.get("date_column") or "").strip()
    value_column = (args.get("value_column") or "").strip()
    if not date_column or not value_column:
        from ..models import ColumnRole

        temporal = dataset.profile.columns_by_role(ColumnRole.TEMPORAL)
        measures = dataset.profile.columns_by_role(ColumnRole.MEASURE)
        if not date_column and temporal:
            date_column = temporal[0].name
        if not value_column and measures:
            value_column = measures[0].name
    if not date_column or not value_column:
        raise ToolError(
            "Forecasting needs a date column and a numeric column.",
            detail=f"'{dataset.table}' does not appear to have both. Check the schema.",
        )

    result = forecast_mod.forecast_series(
        dataset.frame,
        date_column,
        value_column,
        periods=int(args.get("periods") or 6),
        freq=(args.get("freq") or "monthly"),
        agg=(args.get("agg") or "sum"),
        table=dataset.table,
    )

    lines = [
        f"Forecast of {result.value_column} by {result.date_column} on {result.table}.",
        f"Method: {result.method}. History: {len(result.history)} period(s).",
        f"In-sample MAPE: {result.in_sample_mape}%" if result.in_sample_mape is not None
        else "In-sample MAPE: not computable (zeros in history).",
        "Projected periods (95% interval):",
    ]
    for point in result.points:
        lines.append(f"  {point.period}: {point.forecast:,.2f}  [{point.lower:,.2f} … {point.upper:,.2f}]")
    for note in result.notes:
        lines.append(f"Note: {note}")
    lines.append("State the method and the interval when you present this; do not present it as certainty.")

    artifacts = [
        Artifact(
            kind="forecast",
            title=f"Forecast: {result.value_column}",
            payload=result.model_dump(mode="json"),
        )
    ]
    return ToolOutcome(
        model_text="\n".join(lines),
        artifacts=artifacts,
        reasoning=f"Forecast {result.value_column} {len(result.points)} period(s) ahead using {result.method}.",
    )


FORECAST = Tool(
    name="forecast",
    description=(
        "Project a dated numeric series forward. Uses Holt-Winters exponential smoothing when there "
        "are at least two seasonal cycles of history, otherwise an OLS linear trend, and returns 95% "
        "intervals plus in-sample MAPE. Use for questions about next month/quarter, projections or trends ahead."
    ),
    parameters=object_schema(
        {
            "table": string_param("Table to use. Omit for the main table."),
            "date_column": string_param("Date/timestamp column. Omit to use the first temporal column."),
            "value_column": string_param("Numeric column to forecast. Omit to use the first measure."),
            "periods": integer_param("How many periods ahead.", minimum=1, maximum=36),
            "freq": string_param("Resample frequency.", enum=["daily", "weekly", "monthly", "quarterly", "yearly"]),
            "agg": string_param("How to aggregate within each period.", enum=["sum", "mean", "count", "min", "max"]),
        },
    ),
    handler=_forecast,
)
