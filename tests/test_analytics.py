"""Anomaly detection, forecasting, chart and cache tests — on the real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import anomaly as anomaly_mod
from core import forecast as forecast_mod
from core.cache import AnswerCache, make_key
from core.charts import auto_spec, build_figure, spec_from_dict, validate_spec
from core.engine import DataSession
from core.errors import ToolError
from core.insights import compute_facts, deterministic_summary
from core.models import AnomalyMethod, ChartSpec, ChartType, QueryResult
from core.tools.analytics import DETECT_ANOMALIES, FORECAST
from core.tools.data import DATA_QUALITY, SEARCH_COLUMNS
from tests.conftest import requires_real_data

RETAIL = "online_retail_ii_international"


# --------------------------------------------------------------------------- #
# Anomaly detection — statistical correctness
# --------------------------------------------------------------------------- #
def test_iqr_finds_a_planted_outlier() -> None:
    series = pd.Series([10.0] * 50 + [1000.0])
    mask, low, high, stats = anomaly_mod.detect_iqr(series)
    assert mask.iloc[-1]
    assert mask.sum() == 1
    assert high < 1000


def test_robust_z_is_not_fooled_by_the_outlier_it_is_hunting() -> None:
    """A mean/std z-score is dragged by extreme values; median/MAD is not."""
    values = [10.0] * 60 + [5000.0, 5100.0, 5200.0]
    series = pd.Series(values)
    mask, scores, stats = anomaly_mod.detect_robust_z(series)
    assert mask.sum() == 3
    classic_z = (series - series.mean()).abs() / series.std()
    assert classic_z.iloc[-1] < scores.iloc[-1]


def test_constant_column_flags_nothing() -> None:
    series = pd.Series([7.0] * 100)
    mask, _, _ = anomaly_mod.detect_robust_z(series)
    assert not mask.any()


def test_isolation_forest_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    frame.loc[0, ["a", "b"]] = [12.0, -12.0]
    first = anomaly_mod.detect_isolation_forest(frame, ["a", "b"])
    second = anomaly_mod.detect_isolation_forest(frame, ["a", "b"])
    assert first is not None and second is not None
    assert first[0].equals(second[0]), "a fixed seed must give reproducible anomalies"
    assert bool(first[0].iloc[0])


def test_isolation_forest_declines_when_data_is_too_small() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    assert anomaly_mod.detect_isolation_forest(frame, ["a", "b"]) is None


# --------------------------------------------------------------------------- #
# Anomaly detection — real data
# --------------------------------------------------------------------------- #
@requires_real_data
def test_real_anomalies_are_found_and_fully_explained(real_session: DataSession) -> None:
    dataset = real_session.get_dataset(RETAIL)
    report = anomaly_mod.analyse(
        dataset.frame, dataset.profile, columns=["quantity", "price"], max_per_method=5
    )
    assert report.anomalies, "the real file contains genuine extreme orders"
    assert AnomalyMethod.IQR in report.methods_used

    # Each method must justify itself with its own statistic, not a generic phrase.
    signatures = {
        AnomalyMethod.IQR: "Tukey fence",
        AnomalyMethod.ROBUST_Z: "robust z-score",
        AnomalyMethod.ISOLATION_FOREST: "Isolation Forest",
        AnomalyMethod.SEASONAL_RESIDUAL: "STL",
    }
    for item in report.anomalies:
        assert item.reason, "every flag must carry its explanation"
        assert signatures[item.method] in item.reason
        assert item.score >= 0
        assert item.column


@requires_real_data
def test_real_anomaly_thresholds_are_reported_numerically(real_session: DataSession) -> None:
    dataset = real_session.get_dataset(RETAIL)
    report = anomaly_mod.analyse(dataset.frame, dataset.profile, columns=["quantity"])
    iqr = [a for a in report.anomalies if a.method == AnomalyMethod.IQR]
    assert iqr
    flagged = iqr[0]
    assert flagged.threshold_low is not None and flagged.threshold_high is not None
    assert "Q1=" in flagged.reason and "IQR=" in flagged.reason


@requires_real_data
def test_anomaly_detection_is_reproducible(real_session: DataSession) -> None:
    dataset = real_session.get_dataset(RETAIL)
    kwargs = dict(columns=["quantity"], max_per_method=5)
    first = anomaly_mod.analyse(dataset.frame, dataset.profile, **kwargs)
    second = anomaly_mod.analyse(dataset.frame, dataset.profile, **kwargs)
    assert [a.row_index for a in first.anomalies] == [a.row_index for a in second.anomalies]


@requires_real_data
def test_sensitivity_changes_how_much_is_flagged(real_session: DataSession) -> None:
    low = DETECT_ANOMALIES.run(real_session, {"columns": ["quantity"], "sensitivity": "low", "max_results": 50})
    high = DETECT_ANOMALIES.run(real_session, {"columns": ["quantity"], "sensitivity": "high", "max_results": 50})
    low_count = len(low.artifacts[0].payload["anomalies"])
    high_count = len(high.artifacts[0].payload["anomalies"])
    assert high_count >= low_count


@requires_real_data
def test_unknown_anomaly_column_is_reported_not_crashed(real_session: DataSession) -> None:
    outcome = DETECT_ANOMALIES.run(real_session, {"columns": ["not_a_column"]})
    notes = outcome.artifacts[0].payload["notes"] if outcome.artifacts else []
    assert any("unknown" in n.lower() for n in notes) or "no anomaly test" in outcome.model_text.lower()


def test_bad_sensitivity_is_rejected(empty_session: DataSession) -> None:
    empty_session.add_csv_bytes(b"a,b\n1,2\n3,4\n", "t.csv")
    with pytest.raises(ToolError):
        DETECT_ANOMALIES.run(empty_session, {"sensitivity": "extreme"})


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def test_linear_trend_is_recovered_exactly() -> None:
    dates = pd.date_range("2023-01-31", periods=10, freq="ME")
    frame = pd.DataFrame({"d": dates, "v": [100.0 + 10 * i for i in range(10)]})
    result = forecast_mod.forecast_series(frame, "d", "v", periods=3, freq="monthly")
    assert "linear trend" in result.method.lower()
    assert result.points[0].forecast == pytest.approx(200.0, abs=1e-6)
    assert result.in_sample_mape == pytest.approx(0.0, abs=1e-6)
    assert result.points[0].lower <= result.points[0].forecast <= result.points[0].upper


def test_intervals_widen_with_the_horizon() -> None:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2022-01-31", periods=18, freq="ME")
    frame = pd.DataFrame({"d": dates, "v": 100 + np.arange(18) * 5 + rng.normal(0, 8, 18)})
    result = forecast_mod.forecast_series(frame, "d", "v", periods=6)
    widths = [p.upper - p.lower for p in result.points]
    assert widths == sorted(widths)


def test_holt_winters_is_used_with_two_full_cycles() -> None:
    dates = pd.date_range("2021-01-31", periods=30, freq="ME")
    seasonal = np.tile([10, 12, 14, 20, 18, 15, 11, 10, 13, 16, 22, 30], 3)[:30]
    frame = pd.DataFrame({"d": dates, "v": seasonal + np.arange(30) * 0.5})
    result = forecast_mod.forecast_series(frame, "d", "v", periods=6)
    assert "Holt-Winters" in result.method


def test_insufficient_history_is_rejected_clearly() -> None:
    frame = pd.DataFrame({"d": pd.date_range("2024-01-31", periods=3, freq="ME"), "v": [1.0, 2.0, 3.0]})
    with pytest.raises(ToolError) as exc:
        forecast_mod.forecast_series(frame, "d", "v")
    assert "at least 4" in str(exc.value)


def test_forecast_rejects_non_numeric_and_missing_columns() -> None:
    frame = pd.DataFrame({"d": pd.date_range("2024-01-31", periods=6, freq="ME"), "v": list("abcdef")})
    with pytest.raises(ToolError):
        forecast_mod.forecast_series(frame, "d", "v")
    with pytest.raises(ToolError):
        forecast_mod.forecast_series(frame, "nope", "v")


@requires_real_data
def test_real_forecast_uses_seasonality_and_reports_error(real_session: DataSession) -> None:
    outcome = FORECAST.run(
        real_session,
        {"date_column": "invoicedate", "value_column": "quantity", "periods": 4, "freq": "monthly"},
    )
    payload = outcome.artifacts[0].payload
    assert len(payload["points"]) == 4
    assert len(payload["history"]) >= 24
    assert "Holt-Winters" in payload["method"]
    assert payload["in_sample_mape"] is not None
    # Real retail data is noisy; the honest report must say so.
    assert any("MAPE" in n for n in payload["notes"])


@requires_real_data
def test_forecast_infers_columns_when_omitted(real_session: DataSession) -> None:
    outcome = FORECAST.run(real_session, {"periods": 2})
    payload = outcome.artifacts[0].payload
    assert payload["date_column"] == "invoicedate"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _result(columns, rows) -> QueryResult:
    return QueryResult(sql="SELECT 1", columns=columns, rows=rows, row_count=len(rows),
                       truncated=False, duration_ms=1.0)


def test_auto_spec_picks_a_line_chart_for_a_period_column() -> None:
    result = _result(["month", "revenue"], [["2024-01", 10.0], ["2024-02", 20.0], ["2024-03", 15.0]])
    spec = auto_spec(result)
    assert spec is not None and spec.type == ChartType.LINE


def test_auto_spec_picks_a_pie_for_few_categories() -> None:
    result = _result(["country", "revenue"], [["FR", 10.0], ["DE", 20.0], ["ES", 5.0]])
    spec = auto_spec(result)
    assert spec is not None and spec.type == ChartType.PIE


def test_auto_spec_picks_bars_for_many_categories() -> None:
    rows = [[f"country_{i}", float(i)] for i in range(20)]
    spec = auto_spec(_result(["country", "revenue"], rows))
    assert spec is not None and spec.type in (ChartType.BAR, ChartType.HORIZONTAL_BAR)


def test_auto_spec_returns_none_for_unchartable_results() -> None:
    assert auto_spec(_result(["only"], [["x"]])) is None


def test_spec_synonyms_and_aliases_are_normalised() -> None:
    spec = spec_from_dict(
        {"chart_type": "Donut", "x_axis": "country", "values": "revenue"}, ["country", "revenue"]
    )
    assert spec.type == ChartType.PIE
    assert spec.x == "country" and spec.y == "revenue"


def test_spec_resolves_case_and_near_miss_column_names() -> None:
    spec = spec_from_dict({"type": "bar", "x": "COUNTRY", "y": "revenue"}, ["country", "revenue"])
    assert spec.x == "country"


def test_spec_with_unknown_column_is_rejected_with_the_real_options() -> None:
    with pytest.raises(ToolError) as exc:
        spec_from_dict({"type": "bar", "x": "planet", "y": "revenue"}, ["country", "revenue"])
    assert "country" in (exc.value.detail or "")


def test_missing_y_is_inferred_rather_than_failing() -> None:
    spec = validate_spec(ChartSpec(type=ChartType.BAR, x="country"), ["country", "revenue"])
    assert spec.y == "revenue"


def test_heatmap_requires_a_value_column() -> None:
    with pytest.raises(ToolError):
        validate_spec(ChartSpec(type=ChartType.HEATMAP, x="a", y="b"), ["a", "b"])


@pytest.mark.parametrize(
    "chart_type", [ChartType.BAR, ChartType.HORIZONTAL_BAR, ChartType.LINE, ChartType.AREA,
                   ChartType.PIE, ChartType.SCATTER, ChartType.HISTOGRAM, ChartType.BOX]
)
def test_every_chart_type_renders(chart_type: ChartType) -> None:
    frame = pd.DataFrame({"country": ["FR", "DE", "ES", "IT"], "revenue": [10.0, 20.0, 5.0, 8.0]})
    figure = build_figure(ChartSpec(type=chart_type, x="country", y="revenue", title="t"), frame)
    assert figure.data


def test_sort_and_limit_are_applied() -> None:
    frame = pd.DataFrame({"c": list("abcde"), "v": [5.0, 1.0, 4.0, 2.0, 3.0]})
    figure = build_figure(ChartSpec(type=ChartType.BAR, x="c", y="v", sort="y_desc", limit=3), frame)
    assert list(figure.data[0].x) == ["a", "c", "e"]


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(ToolError):
        build_figure(ChartSpec(type=ChartType.BAR, x="c", y="v"), pd.DataFrame({"c": [], "v": []}))


# --------------------------------------------------------------------------- #
# Insights, quality, search
# --------------------------------------------------------------------------- #
@requires_real_data
def test_computed_facts_are_real_and_complete(real_session: DataSession) -> None:
    facts = compute_facts(real_session, RETAIL)
    assert facts["row_count"] == 86_041
    assert facts["segments"], "concentration by dimension should be computed"
    assert facts["trend"] is not None
    assert facts["trend"]["periods"] >= 24
    assert facts["date_ranges"][0]["from"].startswith("2009-12")
    top = facts["segments"][0]["top"]
    assert 0 < top["share_pct"] <= 100


@requires_real_data
def test_deterministic_summary_works_without_an_llm(real_session: DataSession) -> None:
    text = deterministic_summary(compute_facts(real_session, RETAIL))
    assert RETAIL in text
    assert "86,041" in text
    assert "quality" in text.lower()


@requires_real_data
def test_data_quality_tool_reports_real_issues(real_session: DataSession) -> None:
    outcome = DATA_QUALITY.run(real_session, {"table": RETAIL})
    payload = outcome.artifacts[0].payload
    assert payload["duplicate_row_count"] > 0
    assert 0 <= payload["score"]["score"] <= 100
    assert payload["issues"]


@requires_real_data
def test_column_search_finds_real_columns_and_values(real_session: DataSession) -> None:
    by_name = SEARCH_COLUMNS.run(real_session, {"query": "customer country"})
    assert "country" in by_name.model_text
    # 'EIRE' is a real value in the data, not a column name.
    by_value = SEARCH_COLUMNS.run(real_session, {"query": "EIRE"})
    assert "country" in by_value.model_text


@requires_real_data
def test_column_search_admits_when_a_concept_is_absent(real_session: DataSession) -> None:
    outcome = SEARCH_COLUMNS.run(real_session, {"query": "employee headcount pension"})
    assert "nothing" in outcome.model_text.lower() or "no match" in (outcome.reasoning or "").lower()


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def test_cache_key_depends_on_schema_and_question() -> None:
    a = make_key("q", "schema-1")
    assert a == make_key("  Q  ", "schema-1"), "whitespace and case must not matter"
    assert a != make_key("q", "schema-2")
    assert a != make_key("other", "schema-1")
    assert a != make_key("q", "schema-1", history_tail="prior turn")


def test_cache_evicts_least_recently_used() -> None:
    cache = AnswerCache(max_entries=2, ttl_s=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")          # 'a' becomes most recent
    cache.set("c", 3)       # evicts 'b'
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_ttl_zero_expires_immediately() -> None:
    cache = AnswerCache(max_entries=8, ttl_s=0)
    cache.set("k", "v")
    assert cache.get("k") is None


def test_ttl_none_never_expires() -> None:
    cache = AnswerCache(max_entries=8, ttl_s=None)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_cache_stats_track_hit_rate() -> None:
    cache = AnswerCache(max_entries=8, ttl_s=60)
    cache.set("k", "v")
    cache.get("k")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
