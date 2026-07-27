"""UI runtime guards and the artifact-rendering data path.

The UI itself is Streamlit, which is awkward to drive headlessly, so these tests
cover the two things that actually broke in practice:

1. The native-library environment guards at the top of ``ui/app.py``. Both exist
   because this app hard-crashed with SIGSEGV under Streamlit; the macOS crash
   report put the faulting frame in pyarrow's bundled mimalloc allocator
   (``mi_thread_init``) reached from ``Table.from_pandas``, which is what
   ``st.dataframe`` calls on Streamlit's per-execution worker thread.

2. That every artifact this app can produce survives the pandas -> Arrow
   conversion ``st.dataframe`` performs. A rendering crash after a correct answer
   is still a broken product, and these payloads carry awkward values: ``None`` in
   float columns, mixed-type object columns, and aggregate rows with ``row_index``
   of -1.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from core import anomaly as anomaly_mod
from core.engine import DataSession
from core.forecast import forecast_series
from core.profile import quality_score
from tests.conftest import requires_real_data

UI_APP = Path(__file__).resolve().parent.parent / "ui" / "app.py"
RETAIL = "online_retail_ii_international"


# --------------------------------------------------------------------------- #
# Native-library guards
# --------------------------------------------------------------------------- #
def test_ui_sets_native_library_guards_before_importing_anything_heavy() -> None:
    """These env vars are read once at library load, so ordering is the whole point.

    Asserted by parsing the module rather than importing it, because importing
    ``ui.app`` would execute Streamlit page setup.
    """
    source = UI_APP.read_text()
    tree = ast.parse(source)

    guard_line: int | None = None
    heavy_import_line: int | None = None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
        ):
            guard_line = node.lineno if guard_line is None else min(guard_line, node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif node.module:
                names = [node.module]
            if any(n.split(".")[0] in {"pandas", "streamlit", "pyarrow", "numpy", "core"} for n in names):
                heavy_import_line = (
                    node.lineno if heavy_import_line is None else min(heavy_import_line, node.lineno)
                )

    assert guard_line is not None, "ui/app.py must set native-library env guards"
    assert heavy_import_line is not None
    assert guard_line < heavy_import_line, (
        "env guards must be set before pandas/streamlit/pyarrow are imported, "
        f"but guards are on line {guard_line} and the first heavy import on {heavy_import_line}"
    )


def test_ui_switches_arrow_off_mimalloc() -> None:
    source = UI_APP.read_text()
    assert "ARROW_DEFAULT_MEMORY_POOL" in source
    assert re.search(r'ARROW_DEFAULT_MEMORY_POOL["\']\s*,\s*["\']system', source), (
        "the Arrow pool must be set to 'system'; mimalloc's per-thread heap init "
        "is what segfaulted under Streamlit"
    )


def test_arrow_honours_the_system_pool_setting() -> None:
    """Confirms the knob we rely on is real in the installed pyarrow."""
    assert pa.system_memory_pool().backend_name == "system"


def test_anomaly_detectors_pin_their_thread_pools() -> None:
    """Two OpenMP runtimes in one process is undefined behaviour."""
    source = (Path(__file__).resolve().parent.parent / "core" / "anomaly.py").read_text()
    assert source.count("threadpool_limits") >= 2, (
        "both the Isolation Forest fit and the STL fit must run under threadpool_limits"
    )


# --------------------------------------------------------------------------- #
# Artifact payloads must survive Arrow conversion
# --------------------------------------------------------------------------- #
def _to_arrow(frame: pd.DataFrame) -> pa.Table:
    """Exactly what st.dataframe does under the hood."""
    return pa.Table.from_pandas(frame)


@requires_real_data
def test_anomaly_table_converts_to_arrow(real_session: DataSession) -> None:
    dataset = real_session.get_dataset(RETAIL)
    report = anomaly_mod.analyse(
        dataset.frame, dataset.profile, columns=["quantity", "price"], max_per_method=10
    )
    payload = report.model_dump(mode="json")

    # Mirrors ui/app.py::render_anomaly.
    frame = pd.DataFrame(
        [
            {
                "method": a["method"],
                "column": a["column"],
                "where": a.get("label") or (f"row {a['row_index']}" if a["row_index"] >= 0 else "aggregate"),
                "value": a.get("value"),
                "score": a["score"],
                "direction": a["direction"],
                "why flagged": a["reason"],
            }
            for a in payload["anomalies"]
        ]
    )
    assert len(frame) > 0
    assert _to_arrow(frame).num_rows == len(frame)


@requires_real_data
def test_query_result_table_converts_to_arrow(real_session: DataSession) -> None:
    """Real rows include NaN customer ids and timestamps rendered as strings."""
    result, _ = real_session.execute_sql(f"SELECT * FROM {RETAIL} LIMIT 200")
    frame = pd.DataFrame(result.rows, columns=result.columns)
    assert _to_arrow(frame).num_rows == 200


@requires_real_data
def test_forecast_tables_convert_to_arrow(real_session: DataSession) -> None:
    dataset = real_session.get_dataset(RETAIL)
    result = forecast_series(
        dataset.frame, "invoicedate", "quantity", periods=6, freq="monthly", table=RETAIL
    )
    payload = result.model_dump(mode="json")
    assert _to_arrow(pd.DataFrame(payload["history"])).num_rows > 0
    assert _to_arrow(pd.DataFrame(payload["points"])).num_rows == 6


@requires_real_data
def test_quality_and_profile_tables_convert_to_arrow(real_session: DataSession) -> None:
    profile = real_session.get_dataset(RETAIL).profile

    issues = pd.DataFrame(
        [
            {"severity": i.severity, "column": i.column or "", "issue": i.message}
            for i in profile.issues
        ]
    )
    assert _to_arrow(issues).num_rows == len(profile.issues)

    # Regression: a measure's min is a float and a date column's min is a
    # timestamp string. Putting both in one column raw makes Arrow reject the whole
    # frame, which took down the schema panel. render_profile formats to text first.
    from ui.app_helpers import fmt_bound

    columns = pd.DataFrame(
        [
            {
                "column": c.name,
                "type": c.duckdb_type,
                "role": c.role.value,
                "distinct": c.distinct_count,
                "null %": c.null_pct,
                "min": fmt_bound(c.min, c.min_date),
                "max": fmt_bound(c.max, c.max_date),
                "examples": ", ".join(c.sample_values[:3]),
            }
            for c in profile.columns
        ]
    )
    assert _to_arrow(columns).num_rows == profile.column_count
    assert columns["min"].map(type).nunique() == 1, "the bound column must be homogeneous"


@requires_real_data
def test_mixed_bounds_would_break_arrow_without_formatting(real_session: DataSession) -> None:
    """Proves the bug the formatter prevents is real, not hypothetical."""
    profile = real_session.get_dataset(RETAIL).profile
    raw = pd.DataFrame(
        [{"min": c.min if c.min is not None else c.min_date} for c in profile.columns]
    )
    with pytest.raises(pa.ArrowInvalid):
        _to_arrow(raw)


@requires_real_data
def test_quality_score_is_json_and_arrow_safe(real_session: DataSession) -> None:
    import json

    for dataset in real_session.datasets.values():
        score = quality_score(dataset.profile)
        json.dumps(score)  # numpy scalars would raise here
        assert isinstance(score["score"], float)


@requires_real_data
def test_empty_result_set_does_not_break_rendering(real_session: DataSession) -> None:
    result, _ = real_session.execute_sql(
        f"SELECT country FROM {RETAIL} WHERE country = 'Atlantis'"
    )
    assert result.row_count == 0
    frame = pd.DataFrame(result.rows, columns=result.columns)
    assert _to_arrow(frame).num_rows == 0
