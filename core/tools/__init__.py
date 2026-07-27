"""Tool registry exposed to the agent."""

from __future__ import annotations

from ..llm.base import ToolSpec
from .analytics import DETECT_ANOMALIES, FORECAST
from .base import Tool, ToolOutcome, ToolRegistry
from .data import DATA_QUALITY, INSPECT_SCHEMA, SEARCH_COLUMNS
from .query import CREATE_CHART, GENERATE_CODE, RUN_PANDAS, RUN_SQL

REGISTRY = ToolRegistry(
    [
        RUN_SQL,
        RUN_PANDAS,
        CREATE_CHART,
        DETECT_ANOMALIES,
        FORECAST,
        INSPECT_SCHEMA,
        DATA_QUALITY,
        SEARCH_COLUMNS,
        GENERATE_CODE,
    ]
)

__all__ = ["REGISTRY", "Tool", "ToolOutcome", "ToolRegistry", "tool_specs"]


def tool_specs() -> list[ToolSpec]:
    """Registry rendered as provider-agnostic function declarations."""
    return [
        ToolSpec(name=t.name, description=t.description, parameters=t.parameters)
        for t in REGISTRY.all()
    ]
