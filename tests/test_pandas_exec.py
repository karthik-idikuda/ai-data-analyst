"""Restricted pandas executor tests.

This module executes model-written code, so it is the second security boundary in
the project after the SQL guard and gets the same adversarial treatment. The escape
attempts below are the standard ones used against Python sandboxes: dunder walks to
``object.__subclasses__``, builtins lookup, import machinery, file access, and
attribute tricks through pandas objects.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.engine import DataSession
from core.errors import ToolError
from core.pandas_exec import MAX_RESULT_ROWS, execute, validate_expression
from core.tools.query import RUN_PANDAS
from tests.conftest import requires_real_data

RETAIL = "online_retail_ii_international"


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    return {
        "sales": pd.DataFrame(
            {
                "country": ["EIRE", "Germany", "EIRE", "France", "Germany"],
                "quantity": [10, 5, 20, 3, 8],
                "price": [2.0, 3.0, 1.5, 10.0, 2.5],
                "invoice": ["1", "2", "C3", "4", "5"],
                "when": pd.to_datetime(
                    ["2011-01-05", "2011-01-20", "2011-02-11", "2011-02-15", "2011-03-01"]
                ),
            }
        )
    }


# --------------------------------------------------------------------------- #
# Legitimate expressions
# --------------------------------------------------------------------------- #
def test_groupby_sum(frames) -> None:
    result = execute("df.groupby('country')['quantity'].sum()", frames)
    assert result.result_kind == "series"
    records = {row[0]: row[1] for row in result.rows}
    assert records == {"EIRE": 30, "France": 3, "Germany": 13}


def test_the_flows_canonical_expression(frames) -> None:
    """`df.groupby(...)[...].sum()` sorted descending — the example in the spec."""
    result = execute(
        "df.groupby('country')['quantity'].sum().sort_values(ascending=False).head(2)", frames
    )
    assert [row[0] for row in result.rows] == ["EIRE", "Germany"]


def test_derived_column_via_assign(frames) -> None:
    result = execute(
        "df.assign(revenue=df['quantity'] * df['price'])"
        ".groupby('country')['revenue'].sum().sort_values(ascending=False)",
        frames,
    )
    assert result.rows[0][0] == "EIRE"
    assert result.rows[0][1] == pytest.approx(50.0)


def test_scalar_result(frames) -> None:
    result = execute("df['quantity'].sum()", frames)
    assert result.result_kind == "scalar"
    assert result.rows == [[46]]


def test_boolean_filter_and_len(frames) -> None:
    result = execute("df[df['quantity'] > 5]['country'].value_counts()", frames)
    assert result.result_kind == "series"


def test_datetime_accessor(frames) -> None:
    result = execute("df.groupby(df['when'].dt.month)['quantity'].sum()", frames)
    assert len(result.rows) == 3


def test_string_accessor(frames) -> None:
    result = execute("df[df['invoice'].str.startswith('C')]", frames)
    assert result.row_count == 1


def test_lambda_in_apply_is_allowed(frames) -> None:
    result = execute("df['quantity'].apply(lambda v: v * 2).sum()", frames)
    assert result.rows == [[92]]


def test_describe_and_pivot(frames) -> None:
    assert execute("df['price'].describe()", frames).row_count > 0
    result = execute(
        "df.pivot_table(index='country', values='quantity', aggfunc='sum')", frames
    )
    assert result.row_count == 3


def test_named_table_is_addressable_not_only_df(frames) -> None:
    result = execute("sales.groupby('country')['quantity'].sum()", frames)
    assert result.row_count == 3


def test_multiindex_result_is_flattened(frames) -> None:
    result = execute("df.groupby(['country', 'invoice'])['quantity'].sum()", frames)
    assert "country" in result.columns and "invoice" in result.columns


def test_source_frame_is_not_mutated(frames) -> None:
    before = frames["sales"].copy()
    execute("df.fillna(0).drop(columns=['price'])", frames)
    pd.testing.assert_frame_equal(frames["sales"], before)


# --------------------------------------------------------------------------- #
# Sandbox escapes — all must be refused
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "attack",
    [
        # dunder walk to arbitrary classes, the classic escape
        "().__class__.__bases__[0].__subclasses__()",
        "df.__class__.__mro__",
        "df.__dict__",
        "(1).__class__.__base__.__subclasses__()",
        # builtins / import machinery
        "__import__('os').system('id')",
        "__builtins__",
        "eval('1+1')",
        "exec('x=1')",
        "compile('1','','eval')",
        "globals()",
        "locals()",
        # filesystem and process
        "open('/etc/passwd').read()",
        "df.to_csv('/tmp/leak.csv')",
        "df.to_pickle('/tmp/x.pkl')",
        # pandas string-eval surfaces
        "df.query('quantity > 5')",
        "df.eval('quantity * price')",
        # statements rather than expressions
        "import os",
        "x = df",
        "df; print(1)",
        "[x for x in ().__class__.__bases__]",
        "lambda: __import__('os')",
    ],
)
def test_escape_attempts_are_rejected(attack: str, frames) -> None:
    with pytest.raises(ToolError):
        execute(attack, frames)


def test_unknown_name_is_rejected_and_lists_the_real_ones(frames) -> None:
    with pytest.raises(ToolError) as exc:
        execute("other_table.sum()", frames)
    assert "sales" in (exc.value.detail or "")


def test_disallowed_pandas_method_is_named_in_the_error(frames) -> None:
    with pytest.raises(ToolError) as exc:
        execute("df.to_parquet('x')", frames)
    assert "to_parquet" in exc.value.message


def test_empty_and_oversized_code_rejected(frames) -> None:
    with pytest.raises(ToolError):
        execute("   ", frames)
    with pytest.raises(ToolError):
        execute("df" + ".head(1)" * 500, frames)


def test_expression_returning_none_is_rejected(frames) -> None:
    with pytest.raises(ToolError):
        execute("df.rename(columns={'a':'b'}).columns.name", frames)


def test_runtime_error_is_reported_not_raised_raw(frames) -> None:
    with pytest.raises(ToolError) as exc:
        execute("df['does_not_exist'].sum()", frames)
    assert exc.value.detail


def test_validator_rejects_before_any_evaluation(frames) -> None:
    """Validation must be static: nothing runs until the tree is approved."""
    with pytest.raises(ToolError):
        validate_expression("open('/etc/passwd')", set(frames) | {"df"})


def test_no_frames_loaded(frames) -> None:
    with pytest.raises(ToolError):
        execute("df.sum()", {})


# --------------------------------------------------------------------------- #
# Real data through the tool interface
# --------------------------------------------------------------------------- #
@requires_real_data
def test_tool_runs_pandas_on_the_real_dataset(real_session: DataSession) -> None:
    outcome = RUN_PANDAS.run(
        real_session,
        {
            "code": "df.assign(revenue=df['quantity'] * df['price'])"
                    ".groupby('country')['revenue'].sum().sort_values(ascending=False).head(5)",
            "purpose": "revenue by country",
        },
    )
    table = next(a for a in outcome.artifacts if a.kind == "table")
    assert table.payload["row_count"] == 5
    assert table.payload["rows"][0][0] == "EIRE"
    assert table.payload["rows"][0][1] == pytest.approx(615_519.55, rel=1e-6)

    code = next(a for a in outcome.artifacts if a.kind == "code")
    assert "executed" in code.payload["note"].lower()


@requires_real_data
def test_pandas_and_sql_agree_on_the_real_data(real_session: DataSession) -> None:
    """Both execution paths must produce the same number, or one of them is wrong."""
    pandas_outcome = RUN_PANDAS.run(
        real_session,
        {"code": "(df['quantity'] * df['price']).sum()"},
    )
    pandas_total = pandas_outcome.artifacts[0].payload["rows"][0][0]

    result, _ = real_session.execute_sql(f"SELECT SUM(quantity * price) AS total FROM {RETAIL}")
    sql_total = result.rows[0][0]

    assert pandas_total == pytest.approx(sql_total, rel=1e-9)


@requires_real_data
def test_tool_rejects_an_unknown_table(real_session: DataSession) -> None:
    from core.errors import DatasetNotFoundError

    with pytest.raises(DatasetNotFoundError):
        RUN_PANDAS.run(real_session, {"code": "df.head()", "table": "nope"})


@requires_real_data
def test_large_result_is_capped(real_session: DataSession) -> None:
    outcome = RUN_PANDAS.run(real_session, {"code": "df"})
    payload = outcome.artifacts[0].payload
    assert len(payload["rows"]) == MAX_RESULT_ROWS
    assert payload["truncated"] is True
    assert payload["row_count"] == 86_041


@requires_real_data
def test_pandas_result_is_json_serialisable(real_session: DataSession) -> None:
    import json

    outcome = RUN_PANDAS.run(real_session, {"code": "df.head(50)"})
    json.dumps(outcome.artifacts[0].payload)
