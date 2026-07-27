"""Streamlit UI.

Architecture note: this UI imports `core` directly and runs the engine in-process
rather than calling the FastAPI service over HTTP. Both are first-class consumers
of the same library. In-process avoids serialising DataFrames on every turn and
keeps the demo to a single container; the HTTP API exists for programmatic use and
runs as its own service in docker-compose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Native-library configuration. MUST run before pyarrow/numpy/scipy are imported,
# because these are read once at library load time.
# --------------------------------------------------------------------------- #

# Switch Arrow off its bundled mimalloc allocator.
#
# Not speculative: this app hard-crashed with SIGSEGV under Streamlit, and the
# macOS crash report put the faulting frame in
#   libarrow: mi_thread_init -> mi_heap_main -> KERN_INVALID_ADDRESS at 0x18
# reached from pyarrow.Table.from_pandas (which is what st.dataframe calls).
# Streamlit runs every script execution on a fresh worker thread, and mimalloc
# initialises a thread-local heap on first allocation in each one. The system
# allocator has no per-thread heap setup and is marginally slower on large
# conversions, which is irrelevant at these table sizes.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Cap native thread pools. scikit-learn/SciPy and Arrow each bring an OpenMP
# runtime, and more than one OpenMP runtime per process is undefined behaviour.
# core/anomaly.py also wraps its fits in threadpool_limits(1); this is the
# process-wide belt to that braces.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Allow `streamlit run ui/app.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from core import __version__
from core.agent import Agent
from core.cache import ANSWER_CACHE
from core.charts import build_figure
from core.config import get_settings
from core.dashboard import build as build_dashboard
from core.engine import DataSession
from core.errors import AnalystError
from core.insights import compute_facts, deterministic_summary, narrate
from core.models import Artifact, ChartSpec
from core.observability import configure_logging, get_logger
from core.reports import to_excel, to_html, to_markdown, to_pdf
from core.semantic import suggest_questions
from ui.app_helpers import fmt_bound

configure_logging()
log = get_logger("ui")
settings = get_settings()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container { padding-top: 2rem; max-width: 1400px; }
  [data-testid="stMetricValue"] { font-size: 1.35rem; }
  .stChatMessage { background: transparent; }
  code { font-size: 0.86rem; }
  .small-note { color: #6b7280; font-size: 0.82rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def get_session() -> DataSession:
    if "data_session" not in st.session_state:
        st.session_state.data_session = DataSession()
        st.session_state.uploaded_names = set()
        st.session_state.upload_errors = []
        st.session_state.pending_question = None
    return st.session_state.data_session


session = get_session()


# --------------------------------------------------------------------------- #
# Artifact rendering
# --------------------------------------------------------------------------- #
def render_table(artifact: Artifact) -> None:
    payload = artifact.payload
    frame = pd.DataFrame(payload.get("rows", []), columns=payload.get("columns", []))
    st.caption(
        f"{payload.get('row_count', len(frame)):,} row(s) · "
        f"{payload.get('duration_ms', 0):.0f} ms"
        + (" · row cap reached" if payload.get("truncated") else "")
    )
    st.dataframe(frame, use_container_width=True, hide_index=True, height=min(420, 60 + 35 * len(frame)))
    if len(frame):
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode(),
            file_name="result.csv",
            mime="text/csv",
            key=f"dl_{id(artifact)}",
        )


def render_chart(artifact: Artifact) -> None:
    payload = artifact.payload
    frame = pd.DataFrame(payload.get("rows", []), columns=payload.get("columns", []))
    try:
        spec = ChartSpec.model_validate(payload["spec"])
        st.plotly_chart(build_figure(spec, frame), use_container_width=True)
    except Exception as exc:  # noqa: BLE001 - never let a chart break the answer
        st.warning(f"Could not render this chart: {exc}")
        st.dataframe(frame, use_container_width=True, hide_index=True)


def render_anomaly(artifact: Artifact) -> None:
    payload = artifact.payload
    items = payload.get("anomalies", [])
    cols = st.columns(4)
    cols[0].metric("Findings", len(items))
    cols[1].metric("Rows tested", f"{payload.get('rows_tested', 0):,}")
    cols[2].metric("Columns", len(payload.get("columns_tested", [])))
    cols[3].metric("Methods", len(payload.get("methods_used", [])))

    if payload.get("notes"):
        for note in payload["notes"]:
            st.caption(f"ℹ️ {note}")
    if not items:
        st.info("No anomalies were flagged at this sensitivity.")
        return

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
            for a in items
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True, height=min(420, 60 + 35 * len(frame)))
    st.caption(
        "Every flag above is produced by a named statistical test with an explicit threshold. "
        "A statistical outlier is not proof of an error."
    )


def render_forecast(artifact: Artifact) -> None:
    import plotly.graph_objects as go

    payload = artifact.payload
    history = pd.DataFrame(payload.get("history", []))
    points = pd.DataFrame([p for p in payload.get("points", [])])

    figure = go.Figure()
    if not history.empty:
        figure.add_trace(
            go.Scatter(x=history["period"], y=history["value"], name="Actual",
                       mode="lines+markers", line=dict(color="#2563eb"))
        )
    if not points.empty:
        figure.add_trace(
            go.Scatter(x=points["period"], y=points["upper"], name="Upper 95%",
                       mode="lines", line=dict(width=0), showlegend=False)
        )
        figure.add_trace(
            go.Scatter(x=points["period"], y=points["lower"], name="95% interval",
                       mode="lines", line=dict(width=0), fill="tonexty",
                       fillcolor="rgba(245,158,11,0.20)")
        )
        figure.add_trace(
            go.Scatter(x=points["period"], y=points["forecast"], name="Forecast",
                       mode="lines+markers", line=dict(color="#f59e0b", dash="dash"))
        )
    figure.update_layout(
        template="plotly_white", height=420, title=artifact.title,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        f"Method: {payload.get('method')} · in-sample MAPE: {payload.get('in_sample_mape')}%"
    )
    for note in payload.get("notes", []):
        st.caption(f"ℹ️ {note}")
    if not points.empty:
        st.dataframe(points, use_container_width=True, hide_index=True)


def render_code(artifact: Artifact) -> None:
    payload = artifact.payload
    if payload.get("explanation"):
        st.markdown(payload["explanation"])
    st.code(payload.get("code", ""), language=payload.get("language", "text"))
    st.caption(f"🔒 {payload.get('note', '')}")


def render_quality(artifact: Artifact) -> None:
    payload = artifact.payload
    score = payload.get("score", {})
    cols = st.columns(4)
    cols[0].metric("Quality score", f"{score.get('score', 0)}/100")
    cols[1].metric("Completeness", f"{score.get('completeness_pct', 0)}%")
    cols[2].metric("Row uniqueness", f"{score.get('uniqueness_pct', 0)}%")
    cols[3].metric("Duplicate rows", f"{payload.get('duplicate_row_count', 0):,}")

    issues = payload.get("issues", [])
    if issues:
        frame = pd.DataFrame(
            [{"severity": i["severity"], "column": i.get("column") or "", "issue": i["message"]} for i in issues]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True, height=min(400, 60 + 35 * len(frame)))


def render_profile(artifact: Artifact) -> None:
    payload = artifact.payload
    frame = pd.DataFrame(
        [
            {
                "column": c["name"],
                "type": c["duckdb_type"],
                "role": c["role"],
                "distinct": c["distinct_count"],
                "null %": c["null_pct"],
                "min": fmt_bound(c.get("min"), c.get("min_date")),
                "max": fmt_bound(c.get("max"), c.get("max_date")),
                "examples": ", ".join(c.get("sample_values", [])[:3]),
            }
            for c in payload.get("columns", [])
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True, height=min(460, 60 + 35 * len(frame)))


RENDERERS = {
    "table": render_table,
    "chart": render_chart,
    "anomaly": render_anomaly,
    "forecast": render_forecast,
    "code": render_code,
    "quality": render_quality,
    "profile": render_profile,
}


def render_artifact(artifact: Artifact) -> None:
    renderer = RENDERERS.get(artifact.kind)
    if renderer is None:
        st.json(artifact.payload)
        return
    with st.container(border=True):
        st.markdown(f"**{artifact.title}**")
        renderer(artifact)


def render_audit(message) -> None:  # type: ignore[no-untyped-def]
    """The 'how I got this' panel: real steps, real SQL, real timings."""
    if not (message.reasoning or message.sql_executed or message.trace):
        return
    with st.expander("How I got this"):
        if message.reasoning:
            st.markdown("**Steps taken**")
            for i, line in enumerate(message.reasoning, start=1):
                st.markdown(f"{i}. {line}")
        if message.sql_executed:
            st.markdown("**SQL executed**")
            for sql in message.sql_executed:
                st.code(sql, language="sql")
        trace = message.trace or {}
        if trace:
            st.markdown("**Execution trace**")
            cols = st.columns(4)
            cols[0].metric("Steps", len(trace.get("steps", [])))
            cols[1].metric("Latency", f"{trace.get('duration_ms', 0) / 1000:.1f}s")
            cols[2].metric("Tokens in", f"{trace.get('tokens_in', 0):,}")
            cols[3].metric("Tokens out", f"{trace.get('tokens_out', 0):,}")
            steps = trace.get("steps", [])
            if steps:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "#": s["index"], "kind": s["kind"], "name": s["name"],
                                "ms": s["duration_ms"], "ok": s["ok"], "error": s.get("error") or "",
                            }
                            for s in steps
                        ]
                    ),
                    use_container_width=True, hide_index=True,
                )
            st.caption(f"trace_id `{trace.get('trace_id')}`" + (" · served from cache" if trace.get("cache_hit") else ""))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📊 AI Data Analyst")
    st.caption(f"v{__version__}")

    if settings.llm_configured:
        st.success(f"LLM: **{settings.llm_provider}** · `{settings.default_model}`")
    else:
        st.warning(
            "No LLM configured. Upload, profiling, data-quality checks, SQL, charts, "
            "anomaly detection and forecasting all still work. Set `LLM_PROVIDER` and "
            "`LLM_API_KEY` in `.env` to enable natural-language questions."
        )

    st.divider()
    st.markdown("#### 1. Upload CSV files")
    uploads = st.file_uploader(
        "CSV, TSV or TXT",
        type=["csv", "tsv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads:
        for upload in uploads:
            key = f"{upload.name}:{upload.size}"
            if key in st.session_state.uploaded_names:
                continue
            try:
                dataset = session.add_csv_bytes(upload.getvalue(), upload.name)
                st.session_state.uploaded_names.add(key)
                st.toast(f"Loaded {dataset.table} ({dataset.profile.row_count:,} rows)", icon="✅")
            except AnalystError as exc:
                st.session_state.upload_errors.append((upload.name, exc))

    for name, exc in st.session_state.upload_errors[-4:]:
        st.error(f"**{name}** — {exc.message}" + (f"\n\n{exc.detail}" if exc.detail else ""))

    if session.datasets:
        st.divider()
        st.markdown("#### 2. Loaded data")
        for table, dataset in list(session.datasets.items()):
            quality = dataset.quality
            with st.expander(f"`{table}` · {dataset.profile.row_count:,} rows", expanded=False):
                st.caption(f"from `{dataset.source_name}`")
                cols = st.columns(2)
                cols[0].metric("Quality", f"{quality['score']}/100")
                cols[1].metric("Columns", dataset.profile.column_count)
                warnings = [i for i in dataset.profile.issues if i.severity in ("warning", "error")]
                if warnings:
                    st.markdown("**Flags**")
                    for issue in warnings[:6]:
                        st.caption(f"⚠️ {issue.message}")
                if st.button("Remove", key=f"rm_{table}", use_container_width=True):
                    session.remove_dataset(table)
                    st.rerun()

        if session.join_hints:
            st.markdown("#### Detected joins")
            for hint in session.join_hints[:5]:
                st.caption(
                    f"`{hint.left_table}.{hint.left_column}` ↔ `{hint.right_table}.{hint.right_column}` "
                    f"— {hint.overlap_pct:.0f}% value overlap"
                )

        st.divider()
        st.markdown("#### 3. Export report")
        cols = st.columns(2)
        cols[0].download_button(
            "PDF", to_pdf(session), file_name=f"report-{session.session_id}.pdf",
            mime="application/pdf", use_container_width=True,
        )
        cols[1].download_button(
            "Excel", to_excel(session), file_name=f"report-{session.session_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        cols = st.columns(2)
        cols[0].download_button(
            "Markdown", to_markdown(session), file_name=f"report-{session.session_id}.md",
            mime="text/markdown", use_container_width=True,
        )
        cols[1].download_button(
            "HTML", to_html(session), file_name=f"report-{session.session_id}.html",
            mime="text/html", use_container_width=True,
        )

    st.divider()
    with st.expander("Diagnostics"):
        st.json({"session": session.session_id, "cache": ANSWER_CACHE.stats()})
    if st.button("Reset session", use_container_width=True):
        session.close()
        for key in ("data_session", "uploaded_names", "upload_errors", "pending_question"):
            st.session_state.pop(key, None)
        st.rerun()


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
st.title("AI Data Analyst")

if not session.datasets:
    st.info("Upload one or more CSV files in the sidebar to begin.")
    st.markdown(
        """
#### What this does

Ask questions about your data in plain English. The model writes SQL; a
deterministic guard validates every statement before DuckDB runs it read-only.
Every answer shows the SQL that produced it.

**Included real datasets** (`data/`, fetched by `scripts/fetch_real_data.py`):

| file | source | what it is |
|---|---|---|
| `online_retail_ii_international.csv` | [UCI ML Repository, dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Real transactions from a UK online giftware retailer, Dec 2009 – Dec 2011 |
| `world_bank_country_profile.csv` | [World Bank Open Data](https://data.worldbank.org) | Real region, income group, GDP per capita and population per country |

Upload both to try cross-file analysis.
        """
    )
    st.stop()

tab_chat, tab_dash, tab_insights, tab_sql = st.tabs(
    ["💬 Chat", "📊 Dashboard", "📈 Insights", "🧮 SQL"]
)

# ------------------------------------------------------------------ Chat tab
with tab_chat:
    if not session.history:
        st.markdown("**Try one of these** — generated from the columns actually present in your data:")
        suggestions = suggest_questions(session)
        columns = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if columns[i % 2].button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    for message in session.history:
        with st.chat_message(message.role, avatar="🧑‍💻" if message.role == "user" else "📊"):
            st.markdown(message.content)
            for artifact in message.artifacts:
                render_artifact(artifact)
            if message.role == "assistant":
                render_audit(message)

    typed = st.chat_input("Ask about your data…")
    question = typed or st.session_state.pop("pending_question", None)

    if question:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="📊"):
            status = st.status("Working…", expanded=True)
            answer = None
            try:
                for event in Agent().run(session, question):
                    if event["type"] == "status":
                        status.write(event["message"])
                    elif event["type"] == "step":
                        step = event["step"]
                        icon = "✅" if step["ok"] else "❌"
                        status.write(f"{icon} `{step['name']}` · {step['duration_ms']:.0f} ms")
                    elif event["type"] == "answer":
                        answer = event["answer"]
                status.update(label="Done", state="complete", expanded=False)
            except AnalystError as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.error(f"**{exc.message}**" + (f"\n\n{exc.detail}" if exc.detail else ""))
            except Exception as exc:  # noqa: BLE001
                status.update(label="Failed", state="error", expanded=True)
                log.exception("ui.turn_failed")
                st.error(f"Unexpected error: {type(exc).__name__}: {exc}")

            if answer is not None:
                st.markdown(answer.answer_markdown)
                for artifact in answer.artifacts:
                    render_artifact(artifact)
                st.rerun()

# ------------------------------------------------------------- Dashboard tab
with tab_dash:
    st.markdown(
        "Generated from the profiled column roles and computed with SQL. "
        "No LLM involved, so this costs nothing and shows the same numbers every time."
    )
    dash_table = st.selectbox("Dataset", session.table_names, key="dash_table")
    if st.button("Build dashboard", type="primary", key="build_dash"):
        with st.spinner("Running aggregate queries…"):
            try:
                st.session_state.dashboard = build_dashboard(session, dash_table)
            except AnalystError as exc:
                st.error(f"**{exc.message}**" + (f"\n\n{exc.detail}" if exc.detail else ""))

    dashboard = st.session_state.get("dashboard")
    if dashboard and dashboard.table == dash_table:
        for note in dashboard.notes:
            st.caption(f"ℹ️ {note}")

        kpis = dashboard.kpis
        for start in range(0, len(kpis), 4):
            row = kpis[start : start + 4]
            for column, kpi in zip(st.columns(len(row)), row):
                column.metric(kpi.label, kpi.value, help=kpi.help or None)

        st.divider()
        panels = dashboard.panels
        for index in range(0, len(panels), 2):
            for column, panel in zip(st.columns(2), panels[index : index + 2]):
                with column, st.container(border=True):
                    frame = pd.DataFrame(panel.rows, columns=panel.columns)
                    try:
                        st.plotly_chart(
                            build_figure(panel.spec, frame),
                            use_container_width=True,
                            key=f"dash_{panel.title}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not render '{panel.title}': {exc}")
                        st.dataframe(frame, use_container_width=True, hide_index=True)
                    if panel.note:
                        st.caption(panel.note)
                    with st.expander("SQL"):
                        st.code(panel.sql, language="sql")


# -------------------------------------------------------------- Insights tab
with tab_insights:
    table = st.selectbox("Dataset", session.table_names, key="insight_table")
    if st.button("Generate briefing", type="primary"):
        with st.spinner("Computing verified statistics…"):
            facts = compute_facts(session, table)
        st.session_state.insight_facts = facts

        if settings.llm_configured:
            placeholder = st.empty()
            buffer = ""
            try:
                for chunk in narrate(facts, stream=True):  # type: ignore[union-attr]
                    buffer += chunk
                    placeholder.markdown(buffer)
            except AnalystError as exc:
                st.warning(f"{exc.message} Showing the computed statistics instead.")
                placeholder.markdown(deterministic_summary(facts))
        else:
            st.markdown(deterministic_summary(facts))

    facts = st.session_state.get("insight_facts")
    if facts:
        st.divider()
        cols = st.columns(4)
        cols[0].metric("Rows", f"{facts['row_count']:,}")
        cols[1].metric("Columns", facts["column_count"])
        cols[2].metric("Quality", f"{facts['quality']['score']}/100")
        cols[3].metric("Completeness", f"{facts['quality']['completeness_pct']}%")

        trend = facts.get("trend")
        if trend and trend.get("series"):
            st.markdown(f"**{trend['measure']} by month** (computed by SQL)")
            st.line_chart(pd.DataFrame(trend["series"]).set_index("period"))

        for segment in facts.get("segments", []):
            st.markdown(
                f"**{segment['measure']} by {segment['dimension']}** — "
                f"top: `{segment['top']['name']}` at {segment['top']['total']:,.2f} "
                f"({segment['top']['share_pct']}%); top 3 hold {segment['top3_share_pct']}%"
            )
        with st.expander("Raw computed facts (JSON)"):
            st.json(facts)

# ------------------------------------------------------------------- SQL tab
with tab_sql:
    st.markdown(
        "Run SQL directly. The same guard applies: one read-only `SELECT`, "
        "known tables only, automatic row cap. No LLM involved."
    )
    st.caption("Tables: " + ", ".join(f"`{t}`" for t in session.table_names))
    default_table = session.default_table()
    query = st.text_area(
        "SQL", value=f"SELECT *\nFROM {default_table}\nLIMIT 20", height=160,
        label_visibility="collapsed",
    )
    if st.button("Run query", type="primary"):
        try:
            result, guard = session.execute_sql(query)
            st.success(f"{result.row_count:,} row(s) in {result.duration_ms:.0f} ms")
            if guard.limit_applied:
                st.caption(f"Row cap applied: LIMIT {guard.limit_applied}")
            for warning in guard.warnings:
                st.caption(f"⚠️ {warning}")
            frame = pd.DataFrame(result.rows, columns=result.columns)
            st.dataframe(frame, use_container_width=True, hide_index=True)
            if len(frame):
                st.download_button(
                    "Download CSV", frame.to_csv(index=False).encode(),
                    file_name="query_result.csv", mime="text/csv",
                )
            with st.expander("Statement that actually executed"):
                st.code(result.sql, language="sql")
        except AnalystError as exc:
            st.error(f"**{exc.message}**" + (f"\n\n{exc.detail}" if exc.detail else ""))
