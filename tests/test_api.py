"""HTTP layer tests via FastAPI's TestClient.

These assert the contract a client depends on: status codes, typed error bodies,
partial-upload success, guard enforcement over HTTP, SSE framing, and that the
analytics endpoints work with no LLM key configured.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.engine import SESSIONS
from tests.conftest import requires_real_data

client = TestClient(app)


@pytest.fixture
def session_id() -> str:
    response = client.post("/sessions")
    assert response.status_code == 200
    return response.json()["session_id"]


@pytest.fixture
def loaded_session(session_id: str, retail_path, country_path) -> str:
    with open(retail_path, "rb") as retail, open(country_path, "rb") as countries:
        response = client.post(
            f"/sessions/{session_id}/datasets",
            files=[
                ("files", ("online_retail_ii_international.csv", retail, "text/csv")),
                ("files", ("world_bank_country_profile.csv", countries, "text/csv")),
            ],
        )
    assert response.status_code == 200, response.text
    assert len(response.json()["loaded"]) == 2
    return session_id


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
def test_health_reports_configuration() -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "llm" in body and "configured" in body["llm"]
    assert {"run_sql", "run_pandas", "create_chart", "detect_anomalies"} <= set(body["tools"])
    assert body["limits"]["max_result_rows"] > 0


def test_trace_header_is_returned() -> None:
    response = client.get("/health")
    assert response.headers["x-trace-id"]
    assert float(response.headers["x-response-time-ms"]) >= 0


def test_unknown_session_returns_404_with_typed_error() -> None:
    response = client.get("/sessions/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_session_lifecycle(session_id: str) -> None:
    assert client.get(f"/sessions/{session_id}").status_code == 200
    assert client.delete(f"/sessions/{session_id}").json()["status"] == "deleted"
    assert client.get(f"/sessions/{session_id}").status_code == 404


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
@requires_real_data
def test_upload_real_files_returns_profiles_and_suggestions(loaded_session: str) -> None:
    body = client.get(f"/sessions/{loaded_session}/schema").json()
    tables = {t["table"] for t in body["tables"]}
    assert tables == {"online_retail_ii_international", "world_bank_country_profile"}
    assert body["join_hints"], "the two real sources join on country"
    assert body["suggestions"]
    assert "DERIVED METRICS" in body["prompt_context"]


def test_upload_rejects_empty_file(session_id: str) -> None:
    response = client.post(
        f"/sessions/{session_id}/datasets", files=[("files", ("empty.csv", b"", "text/csv"))]
    )
    assert response.status_code == 400
    assert response.json()["failed"][0]["error"]["code"] == "empty_file"


def test_upload_partial_success_keeps_the_good_file(session_id: str) -> None:
    response = client.post(
        f"/sessions/{session_id}/datasets",
        files=[
            ("files", ("good.csv", b"region,value\nNorth,1\nSouth,2\n", "text/csv")),
            ("files", ("bad.csv", b"", "text/csv")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["loaded"]) == 1 and len(body["failed"]) == 1
    assert body["loaded"][0]["table"] == "good"


def test_upload_rejects_wrong_extension(session_id: str) -> None:
    response = client.post(
        f"/sessions/{session_id}/datasets", files=[("files", ("x.xlsx", b"a,b\n1,2\n", "text/csv"))]
    )
    assert response.json()["failed"][0]["error"]["code"] == "unsupported_file"


@requires_real_data
def test_dataset_can_be_removed_over_http(loaded_session: str) -> None:
    response = client.delete(f"/sessions/{loaded_session}/datasets/world_bank_country_profile")
    assert response.status_code == 200
    assert len(response.json()["datasets"]) == 1


# --------------------------------------------------------------------------- #
# SQL endpoint (no LLM needed)
# --------------------------------------------------------------------------- #
@requires_real_data
def test_sql_endpoint_runs_real_aggregate(loaded_session: str) -> None:
    response = client.post(
        f"/sessions/{loaded_session}/sql",
        json={
            "sql": "SELECT country, SUM(quantity * price) AS revenue "
                   "FROM online_retail_ii_international GROUP BY 1 ORDER BY 2 DESC LIMIT 3"
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["row_count"] == 3
    assert body["guard"]["tables"] == ["online_retail_ii_international"]


@requires_real_data
def test_sql_endpoint_blocks_a_destructive_statement(loaded_session: str) -> None:
    response = client.post(
        f"/sessions/{loaded_session}/sql",
        json={"sql": "DROP TABLE online_retail_ii_international"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsafe_query"
    # And the table is still there.
    assert client.get(f"/sessions/{loaded_session}/schema").status_code == 200


@requires_real_data
def test_sql_endpoint_blocks_filesystem_access(loaded_session: str) -> None:
    response = client.post(
        f"/sessions/{loaded_session}/sql", json={"sql": "SELECT * FROM read_csv_auto('/etc/passwd')"}
    )
    assert response.status_code == 422


@requires_real_data
def test_sql_endpoint_applies_the_row_cap(loaded_session: str) -> None:
    response = client.post(
        f"/sessions/{loaded_session}/sql",
        json={"sql": "SELECT * FROM online_retail_ii_international", "max_rows": 10},
    )
    body = response.json()
    assert body["result"]["row_count"] == 10
    assert body["guard"]["limit_applied"] == 10


# --------------------------------------------------------------------------- #
# Analytics endpoints (no LLM needed)
# --------------------------------------------------------------------------- #
@requires_real_data
def test_anomalies_endpoint(loaded_session: str) -> None:
    response = client.post(
        f"/sessions/{loaded_session}/anomalies",
        json={"columns": ["quantity"], "sensitivity": "medium", "max_results": 5},
    )
    assert response.status_code == 200
    report = response.json()["report"]
    assert report["anomalies"]
    assert all(a["reason"] for a in report["anomalies"])


@requires_real_data
def test_forecast_endpoint(loaded_session: str) -> None:
    response = client.post(
        f"/sessions/{loaded_session}/forecast",
        json={"date_column": "invoicedate", "value_column": "quantity", "periods": 3},
    )
    assert response.status_code == 200
    assert len(response.json()["forecast"]["points"]) == 3


@requires_real_data
def test_insights_endpoint_degrades_without_an_llm(loaded_session: str) -> None:
    response = client.get(f"/sessions/{loaded_session}/insights?narrative=false")
    assert response.status_code == 200
    body = response.json()
    assert body["narrated"] is False
    assert body["facts"]["row_count"] == 86_041
    assert "86,041" in body["briefing"]


@requires_real_data
def test_quality_endpoint(loaded_session: str) -> None:
    body = client.get(f"/sessions/{loaded_session}/quality").json()
    retail = body["online_retail_ii_international"]
    assert 0 <= retail["score"]["score"] <= 100
    assert retail["issues"]


@requires_real_data
def test_report_export_in_all_formats(loaded_session: str) -> None:
    markdown = client.get(f"/sessions/{loaded_session}/report?format=markdown")
    assert markdown.status_code == 200
    assert "AI-powered Data Analyst" in markdown.text
    assert "online_retail_ii_international" in markdown.text

    html = client.get(f"/sessions/{loaded_session}/report?format=html")
    assert html.status_code == 200
    assert html.text.startswith("<!DOCTYPE html>")

    xlsx = client.get(f"/sessions/{loaded_session}/report?format=xlsx")
    assert xlsx.status_code == 200
    assert xlsx.content[:2] == b"PK"
    assert "spreadsheetml" in xlsx.headers["content-type"]

    pdf = client.get(f"/sessions/{loaded_session}/report?format=pdf")
    assert pdf.status_code == 200
    assert pdf.content[:5] == b"%PDF-"
    assert pdf.headers["content-type"] == "application/pdf"

    assert client.get(f"/sessions/{loaded_session}/report?format=docx").status_code == 400


@requires_real_data
def test_dashboard_endpoint_needs_no_llm(loaded_session: str) -> None:
    response = client.get(f"/sessions/{loaded_session}/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["table"] == "online_retail_ii_international"
    assert body["measure_expression"] == "quantity * price"
    assert len(body["kpis"]) >= 4
    assert body["panels"]
    for panel in body["panels"]:
        assert panel["sql"].lower().startswith("select")
        assert panel["columns"]


@requires_real_data
def test_dashboard_endpoint_accepts_a_table_parameter(loaded_session: str) -> None:
    body = client.get(
        f"/sessions/{loaded_session}/dashboard?table=world_bank_country_profile"
    ).json()
    assert body["table"] == "world_bank_country_profile"


@requires_real_data
def test_dashboard_endpoint_rejects_an_unknown_table(loaded_session: str) -> None:
    response = client.get(f"/sessions/{loaded_session}/dashboard?table=nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


# --------------------------------------------------------------------------- #
# Chat (no key configured in CI -> typed 503, never a 500)
# --------------------------------------------------------------------------- #
@requires_real_data
def test_chat_without_a_key_returns_a_typed_service_error(loaded_session: str) -> None:
    from core.config import get_settings

    if get_settings().llm_configured:
        pytest.skip("an LLM key is configured; this asserts the unconfigured path")
    response = client.post(f"/sessions/{loaded_session}/chat", json={"question": "revenue by country"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"


@requires_real_data
def test_chat_stream_emits_well_formed_sse(loaded_session: str) -> None:
    with client.stream(
        "POST", f"/sessions/{loaded_session}/chat/stream", json={"question": "revenue by country"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    assert events
    assert events[-1]["type"] == "done"
    assert all("type" in e for e in events)


def test_chat_validation_rejects_an_empty_question(session_id: str) -> None:
    assert client.post(f"/sessions/{session_id}/chat", json={"question": ""}).status_code == 422


@requires_real_data
def test_history_endpoint(loaded_session: str) -> None:
    assert client.get(f"/sessions/{loaded_session}/history").json()["messages"] == []


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_api_key_is_enforced_when_configured(monkeypatch) -> None:
    from core import config

    monkeypatch.setattr(config.get_settings(), "api_key", "s3cret")
    import api.main as api_main

    monkeypatch.setattr(api_main, "settings", config.get_settings())

    assert client.post("/sessions").status_code == 401
    assert client.post("/sessions", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/sessions", headers={"X-API-Key": "s3cret"}).status_code == 200
    # /health stays open on purpose, for container health checks.
    assert client.get("/health").status_code == 200
