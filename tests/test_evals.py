"""Evaluation-framework tests, plus the pandas-vs-DuckDB cross-check.

These run with no API key. The cross-check is the important one: for every fact
in the golden set, the pandas ground truth and the application's DuckDB query path
must agree. Two independent implementations agreeing is real evidence the numbers
are right; one implementation checking itself is not.
"""

from __future__ import annotations

import pytest

from core.engine import DataSession
from core.models import AgentAnswer, Artifact
from evals import ground_truth as gt
from evals.cases import GOLDEN_SET
from evals.checks import (
    ContainsText,
    ContainsTruthNumber,
    ContainsTruthText,
    HasArtifact,
    Rejects,
    SqlContains,
    SqlExecuted,
    extract_numbers,
    number_present,
)
from tests.conftest import requires_real_data

RETAIL = "online_retail_ii_international"


# --------------------------------------------------------------------------- #
# Number extraction / matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Revenue was 615,519.55 in total", 615519.55),
        ("Revenue was £615.5k", 615500.0),
        ("about 0.62 million", 620000.0),
        ("615519.55", 615519.55),
        ("-1,234.5 units", -1234.5),
    ],
)
def test_extract_numbers_handles_real_answer_formats(text: str, expected: float) -> None:
    assert any(abs(n - expected) < 1 for n in extract_numbers(text))


def test_number_present_allows_rounding() -> None:
    assert number_present("Revenue was £615,520", 615519.55)
    assert number_present("Revenue was about 616k", 615519.55, rel_tolerance=0.01)


def test_number_present_rejects_a_wrong_figure() -> None:
    assert not number_present("Revenue was 415,519.55", 615519.55)
    assert not number_present("no numbers here", 615519.55)


# --------------------------------------------------------------------------- #
# Checks behave correctly
# --------------------------------------------------------------------------- #
def _answer(text: str = "", sql: list[str] | None = None, artifacts=None, reasoning=None) -> AgentAnswer:
    return AgentAnswer(
        answer_markdown=text,
        sql_executed=sql or [],
        artifacts=artifacts or [],
        reasoning=reasoning or [],
    )


def test_contains_truth_text_and_number() -> None:
    truth = gt.Truth(key="k", value={"country": "EIRE", "revenue": 615519.55}, derivation="test")
    answer = _answer("EIRE led with 615,519.55.")
    assert ContainsTruthText("country").run(answer, truth).passed
    assert ContainsTruthNumber("revenue").run(answer, truth).passed
    assert not ContainsTruthText("country").run(_answer("France led."), truth).passed


def test_truth_path_indexes_into_lists() -> None:
    truth = gt.Truth(key="k", value=[{"customer_id": 14646}], derivation="test")
    assert ContainsTruthText("0.customer_id").run(_answer("Customer 14646 is top."), truth).passed


def test_sql_checks() -> None:
    answer = _answer(sql=["SELECT country, SUM(quantity*price) FROM t GROUP BY 1"])
    assert SqlContains(("quantity", "price")).run(answer, None).passed
    assert not SqlContains(("cost",)).run(answer, None).passed
    assert SqlExecuted().run(answer, None).passed
    assert not SqlExecuted().run(_answer("no query"), None).passed


def test_has_artifact() -> None:
    answer = _answer(artifacts=[Artifact(kind="chart", title="t", payload={})])
    assert HasArtifact("chart").run(answer, None).passed
    assert not HasArtifact("anomaly").run(answer, None).passed


def test_rejects_accepts_an_honest_refusal() -> None:
    honest = _answer("This dataset has no cost column, so margin cannot be calculated.")
    assert Rejects(forbidden=("margin column",)).run(honest, None).passed


def test_rejects_catches_a_fabricated_answer() -> None:
    invented = _answer("The profit margin by country was 34%.")
    assert not Rejects(forbidden=("profit margin",)).run(invented, None).passed


def test_rejects_catches_a_silent_non_answer() -> None:
    evasive = _answer("Here is a breakdown by country.")
    assert not Rejects().run(evasive, None).passed


# --------------------------------------------------------------------------- #
# Golden set integrity
# --------------------------------------------------------------------------- #
def test_golden_set_is_well_formed() -> None:
    ids = [c.id for c in GOLDEN_SET]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in GOLDEN_SET:
        assert case.question.strip()
        assert case.checks, f"{case.id} has no checks"
        assert case.tags, f"{case.id} has no tags"


@requires_real_data
def test_every_referenced_truth_key_exists() -> None:
    truths = gt.compute_all()
    for case in GOLDEN_SET:
        if case.truth_key:
            assert case.truth_key in truths, f"{case.id} references unknown truth '{case.truth_key}'"


def test_golden_set_includes_refusal_cases() -> None:
    """An eval suite of only answerable questions rewards guessing."""
    refusals = [c for c in GOLDEN_SET if "refusal" in c.tags]
    assert len(refusals) >= 3


def test_golden_set_covers_the_assignment_examples() -> None:
    examples = [c for c in GOLDEN_SET if "assignment-example" in c.tags]
    assert len(examples) >= 5


# --------------------------------------------------------------------------- #
# The cross-check: pandas ground truth vs the app's DuckDB path
# --------------------------------------------------------------------------- #
@requires_real_data
def test_duckdb_agrees_with_pandas_on_top_country(real_session: DataSession) -> None:
    truth = gt.top_country_by_revenue()
    result, _ = real_session.execute_sql(
        f"SELECT country, SUM(quantity * price) AS revenue FROM {RETAIL} "
        f"GROUP BY country ORDER BY revenue DESC LIMIT 1"
    )
    country, revenue = result.rows[0]
    assert country == truth.value["country"]
    assert revenue == pytest.approx(truth.value["revenue"], rel=1e-6)


@requires_real_data
def test_duckdb_agrees_with_pandas_on_top_five_countries(real_session: DataSession) -> None:
    truth = gt.top_countries_by_revenue(5)
    result, _ = real_session.execute_sql(
        f"SELECT country, SUM(quantity * price) AS revenue FROM {RETAIL} "
        f"GROUP BY country ORDER BY revenue DESC LIMIT 5"
    )
    assert [row[0] for row in result.rows] == [e["country"] for e in truth.value]
    for row, expected in zip(result.rows, truth.value):
        assert row[1] == pytest.approx(expected["revenue"], rel=1e-6)


@requires_real_data
def test_duckdb_agrees_with_pandas_on_top_customers(real_session: DataSession) -> None:
    truth = gt.top_customers_by_revenue(5)
    result, _ = real_session.execute_sql(
        f"SELECT customer_id, SUM(quantity * price) AS revenue FROM {RETAIL} "
        f"WHERE customer_id IS NOT NULL GROUP BY customer_id ORDER BY revenue DESC LIMIT 5"
    )
    assert [int(row[0]) for row in result.rows] == [e["customer_id"] for e in truth.value]


@requires_real_data
def test_duckdb_agrees_with_pandas_on_monthly_series(real_session: DataSession) -> None:
    truth = gt.monthly_revenue()
    result, _ = real_session.execute_sql(
        f"SELECT strftime(invoicedate, '%Y-%m') AS period, SUM(quantity * price) AS revenue "
        f"FROM {RETAIL} GROUP BY 1 ORDER BY revenue DESC LIMIT 1"
    )
    period, revenue = result.rows[0]
    assert period == truth.value["best_period"]
    assert revenue == pytest.approx(truth.value["best_value"], rel=1e-6)


@requires_real_data
def test_duckdb_agrees_with_pandas_on_the_cross_file_join(real_session: DataSession) -> None:
    truth = gt.revenue_by_world_bank_region()
    result, _ = real_session.execute_sql(
        f"""
        SELECT w.world_bank_region AS region, SUM(r.quantity * r.price) AS revenue
        FROM {RETAIL} r
        JOIN world_bank_country_profile w ON r.country = w.country
        GROUP BY 1 ORDER BY revenue DESC LIMIT 1
        """
    )
    region, revenue = result.rows[0]
    assert region == truth.value["top_region"]
    assert revenue == pytest.approx(truth.value["top_revenue"], rel=1e-6)


@requires_real_data
def test_duckdb_agrees_with_pandas_on_returns(real_session: DataSession) -> None:
    truth = gt.returns_summary()
    result, _ = real_session.execute_sql(
        f"SELECT COUNT(*) AS n, SUM(quantity * price) AS value FROM {RETAIL} WHERE quantity < 0"
    )
    count, value = result.rows[0]
    assert count == truth.value["return_rows"]
    assert value == pytest.approx(truth.value["return_value"], rel=1e-6)


@requires_real_data
def test_anomaly_detector_finds_the_true_maximum(real_session: DataSession) -> None:
    """The largest real quantity in the file must be flagged, and the reported
    Tukey fence must match the one pandas computes."""
    from core import anomaly as anomaly_mod

    truth = gt.largest_quantity_outlier()
    dataset = real_session.get_dataset(RETAIL)
    report = anomaly_mod.analyse(dataset.frame, dataset.profile, columns=["quantity"], max_per_method=20)

    values = [a.value for a in report.anomalies if a.column == "quantity"]
    assert float(truth.value["quantity"]) in values

    iqr_flags = [a for a in report.anomalies if a.method.value == "iqr" and a.column == "quantity"]
    assert iqr_flags
    assert iqr_flags[0].threshold_high == pytest.approx(truth.value["upper_fence"], rel=1e-6)
