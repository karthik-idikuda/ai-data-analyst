"""Pydantic models shared by the engine, the API and the UI.

These are the contract. Anything crossing a module boundary is one of these,
which keeps the LLM's free-form output from leaking into the rest of the app:
model output is parsed into a model here, or it is rejected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Profiling / semantic layer
# --------------------------------------------------------------------------- #
class ColumnRole(str, Enum):
    """Inferred analytical role. Drives chart defaults and query hints."""

    DIMENSION = "dimension"
    MEASURE = "measure"
    TEMPORAL = "temporal"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    duckdb_type: str
    role: ColumnRole
    null_count: int
    null_pct: float
    distinct_count: int
    distinct_pct: float
    sample_values: list[str] = Field(default_factory=list)
    # numeric
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    # temporal
    min_date: str | None = None
    max_date: str | None = None
    # categorical
    top_values: list[dict[str, Any]] = Field(default_factory=list)


class DataQualityIssue(BaseModel):
    severity: Literal["info", "warning", "error"]
    kind: str
    column: str | None = None
    message: str


class DatasetProfile(BaseModel):
    table: str
    source_name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    duplicate_row_count: int
    memory_bytes: int
    issues: list[DataQualityIssue] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def column(self, name: str) -> ColumnProfile | None:
        lowered = name.lower()
        for col in self.columns:
            if col.name.lower() == lowered:
                return col
        return None

    def columns_by_role(self, role: ColumnRole) -> list[ColumnProfile]:
        return [c for c in self.columns if c.role == role]


class JoinHint(BaseModel):
    """A candidate relationship between two loaded tables."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    overlap_pct: float
    reason: str


# --------------------------------------------------------------------------- #
# Query execution
# --------------------------------------------------------------------------- #
class QueryResult(BaseModel):
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    duration_ms: float

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def to_markdown(self, limit: int = 25) -> str:
        if not self.rows:
            return "_(query returned no rows)_"
        head = self.rows[:limit]
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "| " + " | ".join("---" for _ in self.columns) + " |",
        ]
        for row in head:
            lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
        if self.row_count > len(head):
            lines.append(f"_… {self.row_count - len(head)} more rows_")
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value).replace("|", "\\|")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
class ChartType(str, Enum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    AREA = "area"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"


class ChartSpec(BaseModel):
    """A declarative chart description.

    The LLM proposes one of these as JSON; we validate it against the actual
    result columns and render it with Plotly. LLM-authored plotting code is
    never executed.
    """

    type: ChartType
    x: str
    y: str | None = None
    series: str | None = None
    title: str = ""
    x_label: str | None = None
    y_label: str | None = None
    sort: Literal["none", "x_asc", "x_desc", "y_asc", "y_desc"] = "none"
    limit: int | None = Field(default=None, ge=1, le=200)
    stacked: bool = False

    @field_validator("title")
    @classmethod
    def _trim_title(cls, v: str) -> str:
        return v.strip()[:160]


# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #
class AnomalyMethod(str, Enum):
    IQR = "iqr"
    ROBUST_Z = "robust_zscore"
    ISOLATION_FOREST = "isolation_forest"
    SEASONAL_RESIDUAL = "seasonal_residual"


class Anomaly(BaseModel):
    method: AnomalyMethod
    column: str
    row_index: int
    value: float | None
    score: float
    direction: Literal["high", "low", "multivariate"]
    threshold_low: float | None = None
    threshold_high: float | None = None
    label: str | None = None
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)


class AnomalyReport(BaseModel):
    table: str
    columns_tested: list[str]
    methods_used: list[AnomalyMethod]
    rows_tested: int
    anomalies: list[Anomaly]
    summary_stats: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
class ForecastPoint(BaseModel):
    period: str
    forecast: float
    lower: float
    upper: float


class ForecastResult(BaseModel):
    table: str
    date_column: str
    value_column: str
    freq: str
    method: str
    history: list[dict[str, Any]]
    points: list[ForecastPoint]
    in_sample_mape: float | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    """Something renderable that a tool produced during a turn."""

    kind: Literal["table", "chart", "anomaly", "forecast", "code", "profile", "quality"]
    title: str
    payload: dict[str, Any]


class AgentAnswer(BaseModel):
    answer_markdown: str
    reasoning: list[str] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    sql_executed: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifacts: list[Artifact] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    sql_executed: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
