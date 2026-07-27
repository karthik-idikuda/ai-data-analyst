"""Tool contract shared by every agent tool.

A tool is a plain Python callable plus a JSON schema. It receives the session and
validated-ish arguments, and returns a :class:`ToolOutcome` containing:

* ``model_text`` — what the LLM sees next (compact, factual, no formatting fluff);
* ``artifacts`` — what the UI renders (tables, charts, reports);
* ``sql`` — every statement that actually executed, for the audit panel;
* ``reasoning`` — one line explaining what this step established.

Tools raise :class:`core.errors.ToolError` for recoverable problems. The agent
feeds that message back to the model as the tool result so it can repair its own
call, which is what turns a hallucinated column name into a retry instead of a
failed turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from ..models import Artifact

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import DataSession


@dataclass
class ToolOutcome:
    model_text: str
    artifacts: list[Artifact] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)
    reasoning: str | None = None


ToolHandler = Callable[["DataSession", dict[str, Any]], ToolOutcome]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    requires_data: bool = True

    def run(self, session: "DataSession", arguments: dict[str, Any]) -> ToolOutcome:
        return self.handler(session, arguments or {})


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())


def string_param(description: str, *, enum: list[str] | None = None) -> dict[str, Any]:
    param: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        param["enum"] = enum
    return param


def integer_param(description: str, *, minimum: int | None = None, maximum: int | None = None) -> dict[str, Any]:
    param: dict[str, Any] = {"type": "integer", "description": description}
    if minimum is not None:
        param["minimum"] = minimum
    if maximum is not None:
        param["maximum"] = maximum
    return param


def array_param(description: str, item_type: str = "string") -> dict[str, Any]:
    return {"type": "array", "description": description, "items": {"type": item_type}}


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }
