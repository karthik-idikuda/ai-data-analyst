"""Profiling, role inference, semantic layer and DuckDB engine tests.

Assertions are made against the real UCI and World Bank files, including the
specific misclassification that a naive heuristic makes on them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.engine import DataSession
from core.errors import DatasetNotFoundError, QueryExecutionError, UnsafeQueryError
from core.ingest import load_csv_bytes
from core.models import ColumnRole
from core.profile import build_profile, infer_role, quality_score
from core.semantic import (
    build_schema_context,
    derived_metric_hints,
    schema_fingerprint,
    suggest_questions,
)
from tests.conftest import requires_real_data


# --------------------------------------------------------------------------- #
# Role inference
# --------------------------------------------------------------------------- #
def test_datetime_is_temporal() -> None:
    series = pd.Series(pd.date_range("2024-01-01", periods=50))
    assert infer_role(series, "order_date", 50) == ColumnRole.TEMPORAL


def test_near_unique_integer_is_an_identifier_not_a_measure() -> None:
    series = pd.Series(range(1000, 1100))
    assert infer_role(series, "order_number", 100) == ColumnRole.IDENTIFIER


def test_named_id_is_an_identifier_even_when_repeated() -> None:
    series = pd.Series([1, 1, 2, 2, 3, 3] * 20)
    assert infer_role(series, "customer_id", 120) == ColumnRole.IDENTIFIER


def test_money_column_is_a_measure() -> None:
    series = pd.Series([10.5, 22.0, 13.25] * 40)
    assert infer_role(series, "revenue", 120) == ColumnRole.MEASURE


def test_low_cardinality_integer_is_a_dimension() -> None:
    series = pd.Series([2019, 2020, 2021, 2022] * 40)
    assert infer_role(series, "year", 160) == ColumnRole.DIMENSION


def test_low_cardinality_text_is_a_dimension() -> None:
    series = pd.Series(["North", "South", "East"] * 50)
    assert infer_role(series, "region", 150) == ColumnRole.DIMENSION


# --------------------------------------------------------------------------- #
# Profiling real data
# --------------------------------------------------------------------------- #
@requires_real_data
def test_real_retail_profile_roles(real_session: DataSession, retail_table: str) -> None:
    profile = real_session.get_dataset(retail_table).profile
    roles = {c.name: c.role for c in profile.columns}

    assert roles["invoicedate"] == ColumnRole.TEMPORAL
    assert roles["quantity"] == ColumnRole.MEASURE
    assert roles["price"] == ColumnRole.MEASURE
    assert roles["country"] == ColumnRole.DIMENSION
    assert roles["customer_id"] == ColumnRole.IDENTIFIER
    # invoice must never be a measure: SUM(invoice) is meaningless.
    assert roles["invoice"] != ColumnRole.MEASURE


@requires_real_data
def test_real_profile_captures_ranges_and_categories(
    real_session: DataSession, retail_table: str
) -> None:
    profile = real_session.get_dataset(retail_table).profile
    quantity = profile.column("quantity")
    country = profile.column("country")

    assert quantity is not None and quantity.min is not None and quantity.min < 0
    assert country is not None and country.top_values
    assert {v["value"] for v in country.top_values} & {"EIRE", "Germany", "France"}


@requires_real_data
def test_real_profile_flags_duplicate_rows(real_session: DataSession, retail_table: str) -> None:
    profile = real_session.get_dataset(retail_table).profile
    assert profile.duplicate_row_count > 0
    assert any(i.kind == "duplicate_rows" for i in profile.issues)


@requires_real_data
def test_quality_score_is_bounded(real_session: DataSession) -> None:
    for dataset in real_session.datasets.values():
        score = quality_score(dataset.profile)
        assert 0 <= score["score"] <= 100
        assert 0 <= score["completeness_pct"] <= 100


def test_small_dataset_is_flagged() -> None:
    loaded = load_csv_bytes(b"a,b\n1,2\n3,4\n", "tiny.csv")
    profile = build_profile(loaded)
    assert any(i.kind == "small_dataset" for i in profile.issues)


def test_constant_column_is_flagged() -> None:
    csv = b"region,value\n" + b"".join(b"North,1\n" for _ in range(30))
    profile = build_profile(load_csv_bytes(csv, "const.csv"))
    assert any(i.kind == "constant_column" for i in profile.issues)


def test_casing_variants_are_flagged() -> None:
    rows = [b"North,1\n", b"north,2\n", b" North ,3\n"] * 12
    profile = build_profile(load_csv_bytes(b"region,value\n" + b"".join(rows), "case.csv"))
    assert any(i.kind == "inconsistent_casing" for i in profile.issues)


# --------------------------------------------------------------------------- #
# Semantic layer
# --------------------------------------------------------------------------- #
@requires_real_data
def test_fact_table_is_chosen_over_the_lookup_table(real_session: DataSession) -> None:
    """Regression: a measure-count heuristic picked the 217-row World Bank file
    over the 86k-row transaction table, because it has more numeric columns."""
    assert real_session.default_table() == "online_retail_ii_international"


@requires_real_data
def test_derived_revenue_hint_is_emitted(real_session: DataSession, retail_table: str) -> None:
    """The real file has quantity and price but no revenue column."""
    profile = real_session.get_dataset(retail_table).profile
    hints = " ".join(derived_metric_hints(profile)).lower()
    assert "revenue" in hints
    assert "quantity" in hints and "price" in hints


@requires_real_data
def test_schema_context_lists_real_columns_and_no_others(real_session: DataSession) -> None:
    context = build_schema_context(real_session)
    for column in ("invoicedate", "stockcode", "world_bank_region", "income_group"):
        assert column in context
    assert "DERIVED METRICS" in context
    assert "VERIFIED JOIN KEYS" in context


@requires_real_data
def test_join_between_the_two_real_sources_is_detected(real_session: DataSession) -> None:
    hints = real_session.join_hints
    assert hints, "country should join the retail data to the World Bank reference"
    top = hints[0]
    assert {top.left_column, top.right_column} == {"country"}
    # 33 of 42 retail countries match by name; the rest genuinely differ (EIRE, USA…).
    assert 50 < top.overlap_pct < 100


@requires_real_data
def test_suggestions_reference_real_columns(real_session: DataSession) -> None:
    suggestions = suggest_questions(real_session)
    assert suggestions
    joined = " ".join(suggestions).lower()
    assert "revenue" in joined
    assert "country" in joined


@requires_real_data
def test_fingerprint_changes_when_data_changes(real_session: DataSession, retail_path) -> None:
    before = schema_fingerprint(real_session)
    other = DataSession()
    other.add_csv_path(retail_path)
    try:
        assert schema_fingerprint(other) != before
    finally:
        other.close()


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@requires_real_data
def test_aggregate_query_on_real_data(real_session: DataSession, retail_table: str) -> None:
    result, guard = real_session.execute_sql(
        f"SELECT country, ROUND(SUM(quantity * price), 2) AS revenue "
        f"FROM {retail_table} GROUP BY country ORDER BY revenue DESC LIMIT 5"
    )
    assert result.row_count == 5
    assert result.columns == ["country", "revenue"]
    assert guard.tables == [retail_table]
    revenues = [row[1] for row in result.rows]
    assert revenues == sorted(revenues, reverse=True)
    assert revenues[0] > 0


@requires_real_data
def test_cross_file_join_executes(real_session: DataSession, retail_table: str) -> None:
    result, guard = real_session.execute_sql(
        f"""
        SELECT w.world_bank_region AS region,
               ROUND(SUM(r.quantity * r.price), 2) AS revenue
        FROM {retail_table} r
        JOIN world_bank_country_profile w ON r.country = w.country
        GROUP BY 1 ORDER BY revenue DESC
        """
    )
    assert result.row_count >= 1
    assert set(guard.tables) == {retail_table, "world_bank_country_profile"}


@requires_real_data
def test_row_cap_is_enforced(real_session: DataSession, retail_table: str) -> None:
    result, guard = real_session.execute_sql(f"SELECT * FROM {retail_table}", max_rows=25)
    assert result.row_count == 25
    assert result.truncated
    assert guard.limit_applied == 25


@requires_real_data
def test_results_are_json_serialisable(real_session: DataSession, retail_table: str) -> None:
    import json

    result, _ = real_session.execute_sql(f"SELECT * FROM {retail_table} LIMIT 20")
    json.dumps(result.model_dump(mode="json"))  # must not raise on Timestamp/int64/NaN


@requires_real_data
def test_engine_external_access_is_locked(real_session: DataSession) -> None:
    """Second line of defence: even if the guard were bypassed, DuckDB itself
    must refuse to read the filesystem."""
    with pytest.raises((QueryExecutionError, UnsafeQueryError, Exception)):
        real_session._con.execute("SELECT * FROM read_csv_auto('/etc/hosts')").fetchall()


@requires_real_data
def test_bad_column_gives_a_typed_error(real_session: DataSession, retail_table: str) -> None:
    with pytest.raises(QueryExecutionError) as exc:
        real_session.execute_sql(f"SELECT no_such_column FROM {retail_table}")
    assert exc.value.detail


def test_query_without_data_is_rejected(empty_session: DataSession) -> None:
    with pytest.raises(UnsafeQueryError):
        empty_session.execute_sql("SELECT * FROM anything")


def test_missing_dataset_raises(empty_session: DataSession) -> None:
    with pytest.raises(DatasetNotFoundError):
        empty_session.get_dataset("nope")
    with pytest.raises(DatasetNotFoundError):
        empty_session.default_table()


@requires_real_data
def test_dataset_can_be_removed(retail_path) -> None:
    session = DataSession()
    session.add_csv_path(retail_path)
    table = session.table_names[0]
    session.remove_dataset(table)
    assert session.table_names == []
    with pytest.raises(UnsafeQueryError):
        session.execute_sql(f"SELECT * FROM {table}")
    session.close()


def test_duplicate_filenames_get_distinct_table_names() -> None:
    session = DataSession()
    session.add_csv_bytes(b"a,b\n1,2\n3,4\n", "same.csv")
    session.add_csv_bytes(b"a,b\n5,6\n7,8\n", "same.csv")
    assert session.table_names == ["same", "same_2"]
    session.close()
