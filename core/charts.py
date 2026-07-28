"""Chart spec validation and Plotly rendering.

The model never writes plotting code. It emits a :class:`ChartSpec` (validated
JSON), we check every field against the columns the query actually returned, and
Plotly renders it. Two consequences: no ``exec`` of model output anywhere in the
app, and a chart request for a non-existent column produces a clear error the
agent can repair instead of a traceback.

There is also a deterministic ``auto_spec`` fallback, so a sensible chart appears
even when the model omits one or the LLM is unavailable.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .errors import ToolError
from .models import ChartSpec, ChartType, ColumnRole, DatasetProfile, QueryResult

# No palette or template is set here on purpose. `core` has no opinion about
# presentation: it returns a figure with correct geometry, and the client themes it
# (see ui/theme.py::style_figure). Baking a light-theme palette in here meant the
# per-trace colours won by specificity over the layout colorway, so the dark theme
# could not restyle its own charts.


def _resolve_column(name: str, columns: list[str], *, field: str) -> str:
    """Case-insensitive, then fuzzy, column resolution with an actionable error."""
    if name in columns:
        return name
    lowered = {c.lower(): c for c in columns}
    if name.lower() in lowered:
        return lowered[name.lower()]
    close = get_close_matches(name.lower(), list(lowered), n=1, cutoff=0.82)
    if close:
        return lowered[close[0]]
    raise ToolError(
        f"Chart field '{field}' refers to column '{name}', which is not in the query result.",
        detail=f"Available columns: {', '.join(columns)}",
    )


def validate_spec(spec: ChartSpec, columns: list[str]) -> ChartSpec:
    """Bind a spec to real result columns, enforcing per-chart-type requirements."""
    if not columns:
        raise ToolError("The query returned no columns, so nothing can be charted.")

    resolved = spec.model_copy()
    resolved.x = _resolve_column(spec.x, columns, field="x")
    if spec.y:
        resolved.y = _resolve_column(spec.y, columns, field="y")
    if spec.series:
        resolved.series = _resolve_column(spec.series, columns, field="series")
    if spec.animate_by:
        resolved.animate_by = _resolve_column(spec.animate_by, columns, field="animate_by")

    needs_y = {
        ChartType.BAR, ChartType.HORIZONTAL_BAR, ChartType.LINE, ChartType.AREA,
        ChartType.PIE, ChartType.SCATTER, ChartType.HEATMAP,
    }
    if resolved.type in needs_y and not resolved.y:
        numeric_candidates = [c for c in columns if c != resolved.x]
        if not numeric_candidates:
            raise ToolError(f"A {resolved.type.value} chart needs a 'y' column.")
        resolved.y = numeric_candidates[-1]
    if resolved.type == ChartType.HEATMAP and not resolved.series:
        raise ToolError("A heatmap needs 'x', 'y' and 'series' (the value to colour by).")
    if not resolved.title:
        resolved.title = (
            f"{resolved.y} by {resolved.x}" if resolved.y else f"Distribution of {resolved.x}"
        )
    return resolved


def auto_spec(result: QueryResult, profile: DatasetProfile | None = None) -> ChartSpec | None:
    """Pick a reasonable chart from the result shape alone. No LLM involved."""
    if result.row_count == 0 or len(result.columns) < 2:
        return None

    frame = pd.DataFrame(result.rows, columns=result.columns)
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        return None
    non_numeric = [c for c in frame.columns if c not in numeric]

    # A date-ish label column plus a measure reads as a trend.
    def looks_temporal(col: str) -> bool:
        lname = col.lower()
        if any(k in lname for k in ("date", "month", "year", "quarter", "week", "period", "day")):
            return True
        sample = frame[col].dropna().astype(str).head(20)
        return bool(len(sample)) and sample.str.match(r"^\d{4}([-/]\d{1,2}){0,2}$").mean() > 0.8

    x_candidates = non_numeric or [c for c in frame.columns if c not in numeric[-1:]]
    if not x_candidates:
        return ChartSpec(type=ChartType.HISTOGRAM, x=numeric[0], title=f"Distribution of {numeric[0]}")

    x = x_candidates[0]
    y = numeric[-1] if numeric[-1] != x else numeric[0]

    if looks_temporal(x):
        return ChartSpec(type=ChartType.LINE, x=x, y=y, sort="x_asc", title=f"{y} over {x}")
    if frame[x].nunique() <= 6 and (frame[y] >= 0).all():
        return ChartSpec(type=ChartType.PIE, x=x, y=y, title=f"Share of {y} by {x}")
    if len(numeric) >= 2 and not non_numeric:
        return ChartSpec(type=ChartType.SCATTER, x=numeric[0], y=numeric[1],
                         title=f"{numeric[1]} vs {numeric[0]}")
    horizontal = frame[x].astype(str).str.len().max() > 14 and frame[x].nunique() > 6
    return ChartSpec(
        type=ChartType.HORIZONTAL_BAR if horizontal else ChartType.BAR,
        x=x, y=y, sort="y_desc", limit=25, title=f"{y} by {x}",
    )


def _prepare(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    work = frame.copy()
    if spec.sort != "none" and spec.y:
        column = spec.x if spec.sort.startswith("x") else spec.y
        ascending = spec.sort.endswith("asc")
        if column in work.columns:
            work = work.sort_values(column, ascending=ascending, kind="mergesort")
    if spec.limit:
        # For an animated chart the limit is per frame (e.g. top 10 each year),
        # not a global head() that would drop entire later frames.
        if spec.animate_by and spec.animate_by in work.columns:
            work = (
                work.groupby(spec.animate_by, group_keys=False)
                .apply(lambda g: g.head(spec.limit))
            )
        else:
            work = work.head(spec.limit)
    # Frames must play in order, so sort by the animation column last.
    if spec.animate_by and spec.animate_by in work.columns:
        work = work.sort_values(spec.animate_by, kind="mergesort")
    return work


def build_figure(spec: ChartSpec, frame: pd.DataFrame) -> go.Figure:
    """Render a validated spec. Raises ToolError on any data/spec mismatch."""
    if frame.empty:
        raise ToolError("There are no rows to chart.")
    work = _prepare(frame, spec)
    labels = {}
    if spec.x_label:
        labels[spec.x] = spec.x_label
    if spec.y and spec.y_label:
        labels[spec.y] = spec.y_label

    common: dict[str, Any] = {"title": spec.title, "labels": labels}

    # Animation is only meaningful for the types Plotly Express can frame, and
    # only when the frame column is genuinely present in the query result.
    animate = None
    if spec.animate_by and spec.animate_by in work.columns and spec.type in (
        ChartType.BAR, ChartType.HORIZONTAL_BAR, ChartType.LINE,
        ChartType.AREA, ChartType.SCATTER,
    ):
        animate = spec.animate_by

    if spec.type in (ChartType.BAR, ChartType.HORIZONTAL_BAR):
        horizontal = spec.type == ChartType.HORIZONTAL_BAR
        bar_kwargs: dict[str, Any] = {}
        if animate:
            bar_kwargs["animation_frame"] = animate
            if spec.y and spec.y in work.columns:
                # Lock the value axis across frames so bars don't rescale each step.
                vmax = float(work[spec.y].max()) * 1.05
                bar_kwargs["range_x" if horizontal else "range_y"] = [0, vmax]
        fig = px.bar(
            work,
            x=spec.y if horizontal else spec.x,
            y=spec.x if horizontal else spec.y,
            color=spec.series,
            orientation="h" if horizontal else "v",
            barmode="stack" if spec.stacked else "group",
            **bar_kwargs,
            **common,
        )
        if horizontal:
            fig.update_yaxes(autorange="reversed")
    elif spec.type == ChartType.LINE:
        line_kwargs = {"animation_frame": animate} if animate else {}
        fig = px.line(work, x=spec.x, y=spec.y, color=spec.series, markers=True, **line_kwargs, **common)
    elif spec.type == ChartType.AREA:
        area_kwargs = {"animation_frame": animate} if animate else {}
        fig = px.area(work, x=spec.x, y=spec.y, color=spec.series, **area_kwargs, **common)
    elif spec.type == ChartType.PIE:
        fig = px.pie(work, names=spec.x, values=spec.y, hole=0.35, **common)
    elif spec.type == ChartType.SCATTER:
        scatter_kwargs: dict[str, Any] = {}
        if animate:
            scatter_kwargs["animation_frame"] = animate
            if spec.y and spec.y in work.columns:
                pad_y = (float(work[spec.y].max()) - float(work[spec.y].min())) * 0.05 or 1.0
                scatter_kwargs["range_y"] = [float(work[spec.y].min()) - pad_y, float(work[spec.y].max()) + pad_y]
        fig = px.scatter(work, x=spec.x, y=spec.y, color=spec.series, **scatter_kwargs, **common)
    elif spec.type == ChartType.HISTOGRAM:
        fig = px.histogram(work, x=spec.x, color=spec.series, **common)
    elif spec.type == ChartType.BOX:
        fig = px.box(work, x=spec.series or spec.x, y=spec.y or spec.x, **common)
    elif spec.type == ChartType.HEATMAP:
        pivot = work.pivot_table(index=spec.y, columns=spec.x, values=spec.series, aggfunc="mean")
        fig = px.imshow(pivot, title=spec.title, aspect="auto", color_continuous_scale="Greys")
    else:  # pragma: no cover - ChartType is exhaustive
        raise ToolError(f"Unsupported chart type '{spec.type}'.")

    if animate:
        # Speed up the default (slow) frame transitions for a livelier playback.
        for button in fig.layout.updatemenus or []:
            for step in button.buttons or []:
                if step.args and isinstance(step.args[-1], dict):
                    step.args[-1].setdefault("frame", {})["duration"] = 700
                    step.args[-1].setdefault("transition", {})["duration"] = 300

    fig.update_layout(
        margin=dict(l=50, r=30, t=60, b=50),
        legend_title_text=spec.series or "",
        height=400,
    )
    return fig


def spec_from_dict(raw: dict[str, Any], columns: list[str]) -> ChartSpec:
    """Parse and validate model-supplied JSON into a bound spec."""
    aliases = {
        "chart_type": "type", "kind": "type", "x_axis": "x", "y_axis": "y",
        "group_by": "series", "color": "series", "hue": "series", "label": "x",
        "values": "y", "names": "x", "top_n": "limit",
    }
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        cleaned[aliases.get(key, key)] = value

    if isinstance(cleaned.get("type"), str):
        normalised = cleaned["type"].strip().lower().replace(" ", "_").replace("-", "_")
        synonyms = {
            "barh": "horizontal_bar", "hbar": "horizontal_bar", "bar_h": "horizontal_bar",
            "column": "bar", "donut": "pie", "doughnut": "pie", "timeseries": "line",
            "time_series": "line", "trend": "line", "hist": "histogram",
            "boxplot": "box", "bubble": "scatter",
        }
        cleaned["type"] = synonyms.get(normalised, normalised)

    try:
        spec = ChartSpec.model_validate(cleaned)
    except Exception as exc:  # noqa: BLE001 - pydantic raises ValidationError
        raise ToolError(
            "The chart specification is not valid.",
            detail=f"{exc}"[:400],
        ) from exc
    return validate_spec(spec, columns)
