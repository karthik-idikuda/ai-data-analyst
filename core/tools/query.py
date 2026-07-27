"""Query, chart and code-generation tools."""

from __future__ import annotations

import ast
from typing import Any, TYPE_CHECKING

import pandas as pd

from ..charts import auto_spec, spec_from_dict
from ..errors import AnalystError, ToolError
from ..models import Artifact
from ..observability import get_logger
from .base import Tool, ToolOutcome, object_schema, string_param

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import DataSession

log = get_logger(__name__)

# Rows shown to the LLM. The UI always receives the full result set; the model only
# needs enough to describe the shape, and every extra row is resent on every
# subsequent step of the loop.
MAX_MODEL_ROWS = 25


# --------------------------------------------------------------------------- #
# run_sql
# --------------------------------------------------------------------------- #
def _run_sql(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    sql = (args.get("sql") or "").strip()
    purpose = (args.get("purpose") or "").strip()
    if not sql:
        raise ToolError("The 'sql' argument is required.")

    result, guard = session.execute_sql(sql)

    lines = [
        f"Executed SQL over {', '.join(guard.tables)}. "
        f"{result.row_count} row(s) in {result.duration_ms:.0f} ms."
    ]
    if guard.limit_applied:
        lines.append(f"A LIMIT {guard.limit_applied} row cap was applied.")
    for warning in guard.warnings:
        lines.append(warning)
    if result.truncated:
        lines.append("Results hit the row cap — aggregate further if you need totals.")
    lines.append("Result:")
    lines.append(result.to_markdown(limit=MAX_MODEL_ROWS))

    artifacts = [
        Artifact(
            kind="table",
            title=purpose or "Query result",
            payload={
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "truncated": result.truncated,
                "duration_ms": round(result.duration_ms, 1),
                "sql": result.sql,
            },
        )
    ]
    return ToolOutcome(
        model_text="\n".join(lines),
        artifacts=artifacts,
        sql=[result.sql],
        reasoning=purpose or f"Queried {', '.join(guard.tables)} and got {result.row_count} row(s).",
    )


RUN_SQL = Tool(
    name="run_sql",
    description=(
        "Run one read-only DuckDB SELECT statement against the loaded tables and get the rows back. "
        "This is the primary tool: use it for every factual, aggregate, ranking, filtering or "
        "comparison question. Reference only the tables and columns given in the schema."
    ),
    parameters=object_schema(
        {
            "sql": string_param("A single read-only SELECT statement in DuckDB syntax. No semicolons, no DDL/DML."),
            "purpose": string_param("Short description of what this query establishes, e.g. 'revenue by region'."),
        },
        required=["sql"],
    ),
    handler=_run_sql,
)


# --------------------------------------------------------------------------- #
# create_chart
# --------------------------------------------------------------------------- #
def _create_chart(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    sql = (args.get("sql") or "").strip()
    if not sql:
        raise ToolError("The 'sql' argument is required so the chart has data to plot.")

    raw_spec = args.get("chart")
    if isinstance(raw_spec, str):
        from ..llm.base import LLMProvider

        try:
            raw_spec = LLMProvider.parse_json_object(raw_spec)
        except ValueError as exc:
            raise ToolError("The 'chart' argument must be a JSON object.", detail=str(exc)) from exc
    if not isinstance(raw_spec, dict) or not raw_spec:
        raise ToolError(
            "The 'chart' argument must be an object describing the chart.",
            detail='Example: {"type": "bar", "x": "region", "y": "total_revenue", "sort": "y_desc"}',
        )

    result, guard = session.execute_sql(sql)
    if result.row_count == 0:
        raise ToolError("The query returned no rows, so there is nothing to chart.")

    spec = spec_from_dict(raw_spec, result.columns)

    artifacts = [
        Artifact(
            kind="chart",
            title=spec.title,
            payload={
                "spec": spec.model_dump(mode="json"),
                "columns": result.columns,
                "rows": result.rows,
                "sql": result.sql,
            },
        )
    ]
    preview = result.to_markdown(limit=min(MAX_MODEL_ROWS, 20))
    return ToolOutcome(
        model_text=(
            f"Rendered a {spec.type.value} chart titled '{spec.title}' "
            f"(x={spec.x}, y={spec.y}) from {result.row_count} row(s). "
            "The user can see it; describe what it shows rather than repeating every value.\n"
            f"Underlying data:\n{preview}"
        ),
        artifacts=artifacts,
        sql=[result.sql],
        reasoning=f"Charted {spec.y or spec.x} by {spec.x} as a {spec.type.value}.",
    )


CREATE_CHART = Tool(
    name="create_chart",
    description=(
        "Run a SELECT and render its result as a chart. Use whenever a comparison, trend, "
        "distribution or share would be clearer visually. Aggregate in the SQL first — do not "
        "chart thousands of raw rows."
    ),
    parameters=object_schema(
        {
            "sql": string_param("SELECT statement producing exactly the columns the chart needs."),
            "chart": {
                "type": "object",
                "description": "Chart specification bound to the query's output columns.",
                "properties": {
                    "type": string_param(
                        "Chart type.",
                        enum=[
                            "bar", "horizontal_bar", "line", "area", "pie",
                            "scatter", "histogram", "box", "heatmap",
                        ],
                    ),
                    "x": string_param("Column for the x axis (or slice labels for a pie)."),
                    "y": string_param("Column for the y axis (or slice values for a pie)."),
                    "series": string_param("Optional column to split/colour by."),
                    "title": string_param("Chart title."),
                    "sort": string_param(
                        "Row ordering before plotting.",
                        enum=["none", "x_asc", "x_desc", "y_asc", "y_desc"],
                    ),
                    "limit": {"type": "integer", "description": "Keep only the first N rows after sorting.", "minimum": 1, "maximum": 200},
                    "stacked": {"type": "boolean", "description": "Stack bars instead of grouping them."},
                },
                "required": ["type", "x"],
            },
        },
        required=["sql", "chart"],
    ),
    handler=_create_chart,
)


# --------------------------------------------------------------------------- #
# generate_code
# --------------------------------------------------------------------------- #
_BANNED_CODE_TOKENS = (
    "import os", "import sys", "import subprocess", "import shutil", "import socket",
    "import requests", "open(", "eval(", "exec(", "__import__", "compile(",
    "pickle", "globals(", "locals(", "getattr(", "setattr(",
)


def _generate_code(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    """Return reusable code to the user.

    The code is **statically checked and displayed, never executed.** That is the
    whole point: the assignment asks for generated SQL/pandas code, and the safe
    way to provide it is as a reviewable artifact rather than as something the
    application runs on the user's behalf.
    """
    language = (args.get("language") or "pandas").strip().lower()
    code = (args.get("code") or "").strip()
    explanation = (args.get("explanation") or "").strip()
    if not code:
        raise ToolError("The 'code' argument is required.")

    if language in ("sql", "duckdb"):
        from ..guard import format_sql, validate_sql

        guard = validate_sql(code, session.table_names)
        display = format_sql(guard.sql)
        note = "Validated against the guard (read-only, tables checked). Not executed by this tool."
        language = "sql"
    elif language in ("pandas", "python", "py"):
        lowered = code.lower()
        for token in _BANNED_CODE_TOKENS:
            if token in lowered:
                raise ToolError(
                    f"Generated code contains a disallowed construct ('{token}').",
                    detail="Produce plain pandas that reads from an existing DataFrame variable.",
                )
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise ToolError("The generated Python is not syntactically valid.", detail=str(exc)) from exc
        display = code
        note = (
            "Syntax-checked and screened for unsafe constructs. This application never executes "
            "generated Python; copy it into your own notebook to run it."
        )
        language = "python"
    else:
        raise ToolError(f"Unsupported language '{language}'.", detail="Use 'sql' or 'pandas'.")

    artifacts = [
        Artifact(
            kind="code",
            title=f"{'SQL' if language == 'sql' else 'Pandas'} code",
            payload={"language": language, "code": display, "explanation": explanation, "note": note},
        )
    ]
    return ToolOutcome(
        model_text=f"Presented {language} code to the user ({len(display.splitlines())} lines). {note}",
        artifacts=artifacts,
        reasoning=explanation or f"Produced reusable {language} code.",
    )


def _run_pandas(session: "DataSession", args: dict[str, Any]) -> ToolOutcome:
    """Execute a restricted pandas expression against the session's DataFrames."""
    from ..pandas_exec import execute

    code = (args.get("code") or "").strip()
    purpose = (args.get("purpose") or "").strip()
    table = (args.get("table") or "").strip() or None
    if not code:
        raise ToolError("The 'code' argument is required.")
    if table:
        session.get_dataset(table)  # raises a typed error for an unknown table

    frames = {name: dataset.frame for name, dataset in session.datasets.items()}
    result = execute(code, frames, primary=table or session.default_table())

    lines = [
        f"Executed pandas: `{result.code}`",
        f"Returned a {result.result_kind} with {result.row_count} row(s) in {result.duration_ms:.0f} ms.",
    ]
    lines.extend(result.warnings)
    lines.append("Result:")
    lines.append(result.to_markdown(limit=MAX_MODEL_ROWS))

    artifacts = [
        Artifact(
            kind="table",
            title=purpose or "Pandas result",
            payload={
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "truncated": result.truncated,
                "duration_ms": round(result.duration_ms, 1),
                "pandas": result.code,
            },
        ),
        Artifact(
            kind="code",
            title="Pandas code (executed)",
            payload={
                "language": "python",
                "code": result.code,
                "explanation": purpose,
                "note": (
                    "This expression was executed. It passed a restricted-grammar check first: "
                    "a single expression, an AST node allow-list, a pandas method allow-list, "
                    "no dunder access, and an empty builtins namespace."
                ),
            },
        ),
    ]
    return ToolOutcome(
        model_text="\n".join(lines),
        artifacts=artifacts,
        reasoning=purpose or f"Ran pandas `{result.code}` and got {result.row_count} row(s).",
    )


RUN_PANDAS = Tool(
    name="run_pandas",
    description=(
        "Execute a single pandas expression against the loaded DataFrames and get the result back. "
        "`df` is bound to the main table; every table is also available under its own name. "
        "Use this when pandas is genuinely more natural than SQL — rolling windows, pct_change, "
        "string/datetime accessors, describe(), correlations. For plain aggregation, grouping and "
        "ranking prefer run_sql. Only one expression, no assignments or imports; a restricted set "
        "of pandas methods is permitted and anything else is refused with a reason."
    ),
    parameters=object_schema(
        {
            "code": string_param(
                "One pandas expression, e.g. "
                "df.groupby('country')['quantity'].sum().sort_values(ascending=False).head(5)"
            ),
            "purpose": string_param("Short description of what this establishes."),
            "table": string_param("Table to bind to `df`. Omit to use the main table."),
        },
        required=["code"],
    ),
    handler=_run_pandas,
)


GENERATE_CODE = Tool(
    name="generate_code",
    description=(
        "Show the user reusable SQL or pandas code for an analysis. Use when they ask for the code, "
        "the query, or how to reproduce a result. SQL is validated by the safety guard; pandas is "
        "syntax-checked and displayed but never executed."
    ),
    parameters=object_schema(
        {
            "language": string_param("Which language to emit.", enum=["sql", "pandas"]),
            "code": string_param(
                "The code itself. For pandas, assume the table is already a DataFrame named after "
                "the table (e.g. `sales`), and use only pandas/numpy."
            ),
            "explanation": string_param("One or two sentences on what the code does."),
        },
        required=["language", "code"],
    ),
    handler=_generate_code,
)
