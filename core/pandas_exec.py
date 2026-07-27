"""Restricted pandas execution.

The flow this project implements has a Pandas Tool that *runs* the expression the
model writes, not one that merely prints it. Doing that with a bare ``eval`` is how
an LLM app becomes a remote-code-execution hole, so execution here is fenced in
four ways:

1. **Single expression only.** The code is parsed with ``ast.parse(mode="eval")``,
   so statements, assignments, imports, loops, ``with``, comprehension-based file
   access and semicolon chains are syntax errors before anything runs.
2. **Node allow-list.** Only the AST node types needed for data manipulation are
   permitted. Anything else — ``Import``, ``Await``, ``Yield``, walrus, f-strings
   with format specs — is rejected by type.
3. **Attribute allow-list.** Every attribute and method name must appear in
   :data:`ALLOWED_ATTRIBUTES`. Dunder access is rejected outright, which closes the
   classic ``().__class__.__bases__`` sandbox escape. ``eval``, ``query``,
   ``to_csv``, ``to_pickle`` and every ``read_*`` are absent from the list.
4. **Empty namespace.** The expression is compiled and evaluated with
   ``__builtins__`` set to an empty mapping and only the registered DataFrames
   bound as names. There is no ``pd``, no ``open``, no ``__import__``.

On top of that: a wall-clock budget, a cap on returned rows, and a copy of the
DataFrame so a mutating expression cannot corrupt the session's data.

This is a deliberately small language. When something legitimate is refused, the
error names the reason and the agent falls back to SQL, which is the more capable
path anyway.
"""

from __future__ import annotations

import ast
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .errors import ToolError
from .observability import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Grammar
# --------------------------------------------------------------------------- #
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Constant,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.keyword, ast.Starred,
    # arithmetic / comparison / logical operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    # lambdas are needed for .apply(...) and are validated by the same rules
    ast.Lambda, ast.arguments, ast.arg,
)

ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        # selection / shape
        "loc", "iloc", "at", "iat", "columns", "index", "values", "shape", "dtypes",
        "head", "tail", "copy", "T", "size", "empty", "name",
        # grouping / aggregation
        "groupby", "agg", "aggregate", "apply", "transform", "pipe",
        "sum", "mean", "median", "min", "max", "std", "var", "count", "size",
        "nunique", "unique", "value_counts", "quantile", "describe", "mode",
        "prod", "sem", "skew", "kurt", "first", "last", "cumsum", "cumprod",
        "cummax", "cummin", "idxmax", "idxmin", "corr", "cov", "rank",
        # reshaping
        "reset_index", "set_index", "sort_values", "sort_index", "rename",
        "pivot_table", "unstack", "stack", "melt", "explode", "transpose",
        "nlargest", "nsmallest", "drop", "drop_duplicates", "duplicated",
        "to_frame", "squeeze", "assign", "merge", "join", "align",
        # cleaning
        "dropna", "fillna", "isna", "notna", "isnull", "notnull", "astype",
        "round", "abs", "clip", "where", "mask", "replace", "between", "isin",
        "diff", "pct_change", "shift", "resample", "asfreq", "interpolate",
        # accessors
        "str", "dt", "cat",
        # str accessor
        "lower", "upper", "strip", "contains", "startswith", "endswith", "len",
        "split", "replace", "title", "capitalize", "zfill", "cat", "match",
        # dt accessor
        "year", "month", "day", "quarter", "week", "weekday", "dayofweek",
        "hour", "minute", "date", "to_period", "strftime", "days_in_month",
        "month_name", "day_name", "normalize", "floor", "ceil",
        # output helpers
        "tolist", "to_list", "to_dict", "item", "keys", "add", "sub", "mul",
        "div", "truediv", "floordiv", "pow", "radd", "rsub", "rmul", "rdiv",
        "gt", "lt", "ge", "le", "eq", "ne", "all", "any",
    }
)

# Names that must never resolve, even if somehow reachable.
_FORBIDDEN_SUBSTRINGS = ("__", "eval", "exec", "compile", "import", "open", "globals", "locals")

MAX_RESULT_ROWS = 2_000
MAX_CODE_LENGTH = 2_000


@dataclass
class PandasResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    duration_ms: float
    result_kind: str  # "dataframe" | "series" | "scalar"
    code: str
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self, limit: int = 25) -> str:
        if self.result_kind == "scalar" and self.rows:
            return f"**{self.rows[0][0]}**"
        if not self.rows:
            return "_(expression returned no rows)_"
        head = self.rows[:limit]
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "| " + " | ".join("---" for _ in self.columns) + " |",
        ]
        for row in head:
            lines.append(
                "| " + " | ".join("" if v is None else str(v).replace("|", "\\|") for v in row) + " |"
            )
        if self.row_count > len(head):
            lines.append(f"_… {self.row_count - len(head)} more rows_")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_expression(code: str, allowed_names: set[str]) -> ast.Expression:
    """Parse and vet a pandas expression. Raises :class:`ToolError` on rejection."""
    text = (code or "").strip().rstrip(";").strip()
    if not text:
        raise ToolError("The pandas expression is empty.")
    if len(text) > MAX_CODE_LENGTH:
        raise ToolError(f"Expression is longer than {MAX_CODE_LENGTH} characters.")

    lowered = text.lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token in lowered:
            raise ToolError(
                f"Expression rejected: it contains '{token}'.",
                detail="Only plain pandas data manipulation is permitted.",
            )

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(
            "That is not a single valid Python expression.",
            detail=(
                f"{exc.msg} (offset {exc.offset}). Assignments, imports, loops and "
                "multiple statements are not allowed — write one expression that "
                "evaluates to a DataFrame, Series or number."
            ),
        ) from exc

    lambda_params: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambda_params.update(a.arg for a in node.args.args)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ToolError(
                f"Expression rejected: {type(node).__name__} is not permitted.",
                detail="Only pandas selection, filtering, grouping and aggregation are allowed.",
            )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ToolError(
                    f"Expression rejected: attribute '{node.attr}' is not permitted.",
                    detail="Private and dunder attribute access is blocked.",
                )
            if node.attr not in ALLOWED_ATTRIBUTES:
                raise ToolError(
                    f"Expression rejected: '{node.attr}' is not on the allowed pandas method list.",
                    detail=(
                        "Use SQL via run_sql for anything this restricted subset cannot express."
                    ),
                )
        if isinstance(node, ast.Name):
            if node.id in lambda_params:
                continue
            if node.id not in allowed_names:
                raise ToolError(
                    f"Unknown name '{node.id}'.",
                    detail=f"Available DataFrames: {', '.join(sorted(allowed_names)) or '(none)'}",
                )
    return tree


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _normalise(result: Any, code: str, duration_ms: float) -> PandasResult:
    warnings: list[str] = []

    if isinstance(result, pd.DataFrame):
        frame = result
        kind = "dataframe"
    elif isinstance(result, pd.Series):
        frame = result.to_frame(name=result.name or "value")
        kind = "series"
    elif isinstance(result, pd.Index):
        frame = pd.Series(result, name=result.name or "value").to_frame()
        kind = "series"
    elif isinstance(result, (int, float, str, bool, np.integer, np.floating, np.bool_)):
        value = result.item() if hasattr(result, "item") else result
        return PandasResult(
            columns=["result"], rows=[[value]], row_count=1, truncated=False,
            duration_ms=duration_ms, result_kind="scalar", code=code,
        )
    elif result is None:
        raise ToolError(
            "The expression evaluated to None.",
            detail="Return a DataFrame, Series or number rather than calling a method for its side effect.",
        )
    else:
        return PandasResult(
            columns=["result"], rows=[[str(result)[:500]]], row_count=1, truncated=False,
            duration_ms=duration_ms, result_kind="scalar", code=code,
            warnings=[f"Result type {type(result).__name__} was rendered as text."],
        )

    # A named or multi-level index carries meaning; surface it as real columns.
    if frame.index.name is not None or isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()

    total = len(frame)
    truncated = total > MAX_RESULT_ROWS
    if truncated:
        frame = frame.head(MAX_RESULT_ROWS)
        warnings.append(f"Showing the first {MAX_RESULT_ROWS:,} of {total:,} rows.")

    from .engine import _jsonable_rows

    return PandasResult(
        columns=[str(c) for c in frame.columns],
        rows=_jsonable_rows(frame),
        row_count=total,
        truncated=truncated,
        duration_ms=duration_ms,
        result_kind=kind,
        code=code,
        warnings=warnings,
    )


def execute(
    code: str,
    frames: dict[str, pd.DataFrame],
    *,
    timeout_s: float = 20.0,
    primary: str | None = None,
) -> PandasResult:
    """Validate and evaluate a pandas expression against the session's DataFrames.

    ``df`` is bound to the primary table as an alias, because that is what models
    write by default and what every pandas tutorial uses.
    """
    if not frames:
        raise ToolError("No DataFrames are loaded.")

    namespace: dict[str, Any] = {}
    for name, frame in frames.items():
        # Copy so a mutating expression cannot corrupt the session's data.
        namespace[name] = frame.copy(deep=False)
    primary_name = primary or next(iter(frames))
    namespace["df"] = namespace[primary_name]

    tree = validate_expression(code, set(namespace))
    compiled = compile(tree, filename="<pandas-tool>", mode="eval")

    box: dict[str, Any] = {}
    started = time.perf_counter()

    def work() -> None:
        try:
            box["value"] = eval(compiled, {"__builtins__": {}}, namespace)  # noqa: S307
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            box["error"] = exc

    worker = threading.Thread(target=work, daemon=True, name="pandas-tool")
    worker.start()
    worker.join(timeout=timeout_s)
    duration_ms = (time.perf_counter() - started) * 1000

    if worker.is_alive():
        # A daemon thread cannot be killed; it is abandoned and dies with the
        # process. The row cap and the tiny expression grammar make a genuinely
        # long-running expression hard to write in the first place.
        raise ToolError(
            f"The pandas expression exceeded the {timeout_s:.0f}s budget.",
            detail="Aggregate earlier, or use run_sql which has a query planner behind it.",
        )
    if "error" in box:
        exc = box["error"]
        raise ToolError(
            f"The pandas expression failed: {type(exc).__name__}.",
            detail=str(exc)[:400],
        ) from exc

    result = _normalise(box["value"], code.strip(), duration_ms)
    log.info(
        "pandas.executed",
        rows=result.row_count, kind=result.result_kind, duration_ms=round(duration_ms, 1),
    )
    return result
