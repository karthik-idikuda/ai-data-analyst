#!/usr/bin/env python3
"""End-to-end smoke test of the HTTP API against the real datasets.

Runs the whole deterministic surface in one pass — upload, schema, quality,
guarded SQL, injection rejection, anomalies, forecast, dashboard, insights and all
three export formats — and prints real numbers so the output is verifiable rather
than a row of ticks.

Needs no API key. Uses FastAPI's in-process test client, so no server has to be
running.

    python scripts/api_smoke.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app
from core.observability import configure_logging

DATA = Path(__file__).resolve().parent.parent / "data"
RETAIL_CSV = DATA / "online_retail_ii_international.csv"
COUNTRY_CSV = DATA / "world_bank_country_profile.csv"
RETAIL = "online_retail_ii_international"

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {GREEN}ok{RESET}   {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    else:
        failures.append(label)
        print(f"  {RED}FAIL{RESET} {label}  {detail}")


def main() -> int:
    configure_logging()
    if not RETAIL_CSV.exists() or not COUNTRY_CSV.exists():
        print("Real datasets missing. Run: python scripts/fetch_real_data.py", file=sys.stderr)
        return 2

    client = TestClient(app)

    print("\nhealth")
    health = client.get("/health").json()
    check("service up", health["status"] == "ok", f"v{health['version']}")
    expected_tools = {
        "run_sql", "run_pandas", "create_chart", "detect_anomalies", "forecast",
        "inspect_schema", "data_quality_report", "search_columns", "generate_code",
    }
    check("all tools registered", set(health["tools"]) == expected_tools,
          f"{len(health['tools'])}: " + ", ".join(health["tools"]))
    print(f"  {DIM}llm configured: {health['llm']['configured']} "
          f"({health['llm']['provider']}, {health['llm']['model']}){RESET}")

    print("\nsession + upload")
    sid = client.post("/sessions").json()["session_id"]
    check("session created", bool(sid), sid)

    with open(RETAIL_CSV, "rb") as a, open(COUNTRY_CSV, "rb") as b:
        upload = client.post(
            f"/sessions/{sid}/datasets",
            files=[
                ("files", (RETAIL_CSV.name, a, "text/csv")),
                ("files", (COUNTRY_CSV.name, b, "text/csv")),
            ],
        )
    body = upload.json()
    check("both real files loaded", len(body["loaded"]) == 2)
    for entry in body["loaded"]:
        print(f"  {DIM}{entry['table']}: {entry['rows']:,} rows x {entry['columns']} cols, "
              f"quality {entry['quality']['score']}/100{RESET}")
    retail = next(e for e in body["loaded"] if e["table"] == RETAIL)
    check("real row count", retail["rows"] == 86_041, f"{retail['rows']:,}")

    print("\nvalidation of a broken file")
    bad = client.post(f"/sessions/{sid}/datasets", files=[("files", ("empty.csv", b"", "text/csv"))])
    check("empty file rejected with a typed error",
          bad.status_code == 400 and bad.json()["failed"][0]["error"]["code"] == "empty_file")

    print("\nschema + semantic layer")
    schema = client.get(f"/sessions/{sid}/schema").json()
    context = schema["prompt_context"]
    check("derived-metric hint present", "DERIVED METRICS" in context)
    check("revenue formula stated", "quantity * price" in context)
    check("join key detected", bool(schema["join_hints"]),
          f"{schema['join_hints'][0]['overlap_pct']}% overlap" if schema["join_hints"] else "")
    check("credit notes preserved as text",
          next(c for t in schema["tables"] if t["table"] == RETAIL
               for c in t["columns"] if c["name"] == "invoice")["duckdb_type"] == "VARCHAR")
    print(f"  {DIM}suggestions: {schema['suggestions'][0]}{RESET}")

    print("\nguarded SQL")
    sql = client.post(f"/sessions/{sid}/sql", json={
        "sql": f"SELECT country, SUM(quantity * price) AS revenue FROM {RETAIL} "
               f"GROUP BY 1 ORDER BY 2 DESC LIMIT 3"})
    rows = sql.json()["result"]["rows"]
    check("aggregate query ran", sql.status_code == 200 and len(rows) == 3)
    check("top country is EIRE", rows[0][0] == "EIRE", f"{rows[0][0]} = {rows[0][1]:,.2f}")
    check("matches pandas ground truth", abs(rows[0][1] - 615_519.55) < 1.0,
          f"expected ~615,519.55, got {rows[0][1]:,.2f}")

    join = client.post(f"/sessions/{sid}/sql", json={
        "sql": f"SELECT w.world_bank_region, SUM(r.quantity * r.price) AS revenue "
               f"FROM {RETAIL} r JOIN world_bank_country_profile w ON r.country = w.country "
               f"GROUP BY 1 ORDER BY 2 DESC LIMIT 1"})
    check("cross-file join ran", join.status_code == 200,
          f"top region: {join.json()['result']['rows'][0][0]}")

    print("\nsecurity guard")
    for label, statement in [
        ("DROP rejected", f"DROP TABLE {RETAIL}"),
        ("statement chaining rejected", f"SELECT 1; DROP TABLE {RETAIL}"),
        ("filesystem read rejected", "SELECT * FROM read_csv_auto('/etc/passwd')"),
        ("unknown table rejected", "SELECT * FROM secrets"),
        ("env var read rejected", "SELECT getenv('LLM_API_KEY')"),
    ]:
        response = client.post(f"/sessions/{sid}/sql", json={"sql": statement})
        check(label, response.status_code == 422 and response.json()["error"]["code"] == "unsafe_query")
    check("data survived the attempts",
          client.get(f"/sessions/{sid}/schema").status_code == 200)

    capped = client.post(f"/sessions/{sid}/sql",
                         json={"sql": f"SELECT * FROM {RETAIL}", "max_rows": 10}).json()
    check("row cap enforced", capped["result"]["row_count"] == 10 and capped["guard"]["limit_applied"] == 10)

    print("\npandas execution tool (restricted grammar)")
    from core.engine import SESSIONS
    from core.errors import ToolError
    from core.tools.query import RUN_PANDAS

    live = SESSIONS.get(sid)
    outcome = RUN_PANDAS.run(live, {
        "code": "df.assign(revenue=df['quantity'] * df['price'])"
                ".groupby('country')['revenue'].sum().sort_values(ascending=False).head(3)",
        "purpose": "revenue by country",
    })
    pandas_rows = outcome.artifacts[0].payload["rows"]
    check("pandas expression executed", len(pandas_rows) == 3,
          f"{pandas_rows[0][0]} = {pandas_rows[0][1]:,.2f}")
    check("pandas agrees with SQL", abs(pandas_rows[0][1] - 615_519.55) < 1.0)
    for label, attack in [
        ("dunder escape refused", "().__class__.__bases__[0].__subclasses__()"),
        ("__import__ refused", "__import__('os').system('id')"),
        ("file write refused", "df.to_csv('/tmp/leak.csv')"),
        ("assignment refused", "x = df"),
    ]:
        try:
            RUN_PANDAS.run(live, {"code": attack})
            check(label, False, "expression was NOT refused")
        except ToolError:
            check(label, True)

    print("\ndeterministic analytics")
    anomalies = client.post(f"/sessions/{sid}/anomalies",
                            json={"columns": ["quantity", "price"], "max_results": 5}).json()["report"]
    check("anomalies found", bool(anomalies["anomalies"]), f"{len(anomalies['anomalies'])} findings")
    check("every flag explained", all(a["reason"] for a in anomalies["anomalies"]))
    check("methods reported", len(anomalies["methods_used"]) >= 2, ", ".join(anomalies["methods_used"]))
    print(f"  {DIM}top: {anomalies['anomalies'][0]['reason'][:120]}…{RESET}")

    forecast = client.post(f"/sessions/{sid}/forecast", json={
        "date_column": "invoicedate", "value_column": "quantity", "periods": 3}).json()["forecast"]
    check("forecast produced", len(forecast["points"]) == 3, forecast["method"])
    check("interval widens with horizon",
          (forecast["points"][2]["upper"] - forecast["points"][2]["lower"])
          > (forecast["points"][0]["upper"] - forecast["points"][0]["lower"]))
    check("error reported honestly", forecast["in_sample_mape"] is not None,
          f"MAPE {forecast['in_sample_mape']}%")

    dashboard = client.get(f"/sessions/{sid}/dashboard").json()
    check("dashboard built", bool(dashboard["panels"]),
          f"{len(dashboard['kpis'])} KPIs, {len(dashboard['panels'])} panels")
    check("dashboard uses derived revenue", dashboard["measure_expression"] == "quantity * price")

    insights = client.get(f"/sessions/{sid}/insights?narrative=false").json()
    check("facts computed", insights["facts"]["row_count"] == 86_041,
          f"{insights['facts']['trend']['periods']} monthly periods")
    check("degrades without an LLM", insights["narrated"] is False)

    quality = client.get(f"/sessions/{sid}/quality").json()[RETAIL]
    check("quality issues surfaced", bool(quality["issues"]),
          f"score {quality['score']['score']}/100, {len(quality['issues'])} issues")

    print("\nexports")
    md = client.get(f"/sessions/{sid}/report?format=markdown")
    check("markdown export", md.status_code == 200 and "86,041" in md.text, f"{len(md.text):,} chars")
    html = client.get(f"/sessions/{sid}/report?format=html")
    check("html export", html.status_code == 200 and html.text.startswith("<!DOCTYPE html>"))
    xlsx = client.get(f"/sessions/{sid}/report?format=xlsx")
    check("excel export", xlsx.status_code == 200 and xlsx.content[:2] == b"PK",
          f"{len(xlsx.content):,} bytes")
    pdf = client.get(f"/sessions/{sid}/report?format=pdf")
    check("pdf export", pdf.status_code == 200 and pdf.content[:5] == b"%PDF-",
          f"{len(pdf.content):,} bytes")
    try:
        import pandas as pd

        book = pd.ExcelFile(io.BytesIO(xlsx.content), engine="openpyxl")
        check("excel is a valid workbook", "Overview" in book.sheet_names,
              ", ".join(book.sheet_names[:4]))
    except Exception as exc:  # noqa: BLE001
        check("excel is a valid workbook", False, str(exc))

    print("\nchat")
    stream = client.post(f"/sessions/{sid}/chat/stream", json={"question": "revenue by country"})
    check("SSE endpoint responds", stream.status_code == 200,
          stream.headers.get("content-type", ""))
    if not health["llm"]["configured"]:
        chat = client.post(f"/sessions/{sid}/chat", json={"question": "revenue by country"})
        check("no-key path returns a typed 503", chat.status_code == 503,
              chat.json()["error"]["code"])
    else:
        print(f"  {DIM}skipped live chat to conserve API quota; run `make eval` for that{RESET}")

    client.delete(f"/sessions/{sid}")
    print("\n" + "─" * 68)
    if failures:
        print(f"{RED}{len(failures)} check(s) failed:{RESET} " + "; ".join(failures))
        return 1
    print(f"{GREEN}All checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
