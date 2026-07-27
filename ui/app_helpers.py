"""Pure presentation helpers.

Kept separate from ``ui/app.py`` so they can be imported and tested without
executing Streamlit page setup. No Streamlit import belongs in this module.
"""

from __future__ import annotations

from typing import Any


def fmt_bound(numeric: float | int | None, temporal: str | None) -> str:
    """Render a column's min/max as text.

    A measure's bound is a number and a date column's bound is a timestamp string.
    Placing both in one DataFrame column untouched makes Arrow reject the entire
    frame — ``Could not convert '2009-12-01 09:28:00' with type str: tried to
    convert to double`` — which takes down the whole schema panel rather than one
    cell. Formatting to text up front keeps the column homogeneous.
    """
    if numeric is None:
        return temporal or ""
    if isinstance(numeric, bool):
        return str(numeric)
    if isinstance(numeric, int):
        return f"{numeric:,}"
    if float(numeric).is_integer() and abs(numeric) < 1e15:
        return f"{int(numeric):,}"
    return f"{numeric:,.4g}"


def truncate(text: Any, limit: int = 80) -> str:
    """Shorten a cell value for display without hiding that it was shortened."""
    value = "" if text is None else str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"
