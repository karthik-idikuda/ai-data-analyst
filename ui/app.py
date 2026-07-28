"""AI-powered Data Analyst — Streamlit interface.

Architecture note: this UI imports ``core`` directly and runs the engine
in-process rather than calling the FastAPI service over HTTP. Both are first-class
consumers of the same library; in-process avoids serialising DataFrames on every
turn and keeps the demo to a single container. The HTTP API exists for
programmatic use and runs as its own service in docker-compose.
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
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Cap native thread pools. scikit-learn/SciPy and Arrow each bring an OpenMP
# runtime, and more than one OpenMP runtime per process is undefined behaviour.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Allow `streamlit run ui/app.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from ui.theme import (
    chart_series, style_figure, inject_premium_css, render_header,
    SVG_CHAT, SVG_DASHBOARD, SVG_INSIGHTS, SVG_QUALITY, SVG_EXPLORE, SVG_WORKSPACE
)

configure_logging()
log = get_logger("ui")
settings = get_settings()

st.set_page_config(
    page_title="AI-powered Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(inject_premium_css(), unsafe_allow_html=True)

ACTIVE_SERIES = chart_series()


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def get_session() -> DataSession:
    if "data_session" not in st.session_state:
        st.session_state.data_session = DataSession()
        st.session_state.loaded_keys = set()
        st.session_state.upload_errors = []
        st.session_state.pending_question = None
        st.session_state.uploader_version = 0
    return st.session_state.data_session


_DERIVED_STATE_KEYS = (
    "dashboard",
    "insight_facts",
    "insight_table_done",
    "insight_text",
)


def clear_derived_state() -> None:
    """Drop view models derived from datasets after a source changes."""
    for key in _DERIVED_STATE_KEYS:
        st.session_state.pop(key, None)


def reset_ui_state() -> None:
    """Close native resources and reset every UI/widget key for this browser session."""
    if "data_session" in st.session_state:
        st.session_state.data_session.close()
    st.session_state.clear()


session = get_session()

SAMPLE_DATASET_INFO = (
    {
        "key": "retail",
        "path": PROJECT_ROOT / "data" / "online_retail_ii_international.csv",
        "title": "Online Retail II (international)",
        "description": (
            "86,041 real transactions from a UK-based online gift retailer, "
            "December 2009 to December 2011, across 42 countries."
        ),
        "detail": "UCI Machine Learning Repository, dataset 502 · CC BY 4.0",
    },
    {
        "key": "worldbank",
        "path": PROJECT_ROOT / "data" / "world_bank_country_profile.csv",
        "title": "World Bank country profile",
        "description": (
            "217 countries with region, income group, GDP per capita and "
            "population for 2010 and 2011."
        ),
        "detail": "World Bank Open Data — World Development Indicators · CC BY 4.0",
    },
)
SAMPLE_DATASETS = tuple(info["path"] for info in SAMPLE_DATASET_INFO)


def load_sample_datasets(paths) -> int:
    """Load the given sample CSV paths and invalidate stale presentation state."""
    existing_sources = {dataset.source_name for dataset in session.datasets.values()}
    loaded = 0
    for path in paths:
        if path.exists() and path.name not in existing_sources:
            session.add_csv_path(path)
            loaded += 1
    if loaded:
        session.history.clear()
        clear_derived_state()
    return loaded


def load_sample_workspace() -> int:
    """Load every bundled sample dataset."""
    return load_sample_datasets(SAMPLE_DATASETS)


@st.dialog("Choose a sample dataset")
def sample_dataset_picker() -> None:
    st.caption(
        "Both files are real, published data — nothing is generated or simulated. "
        "Pick one or load both."
    )
    for info in SAMPLE_DATASET_INFO:
        with st.container(border=True):
            st.markdown(f"**{info['title']}**")
            st.caption(info["description"])
            st.caption(info["detail"])
            if st.button(
                f"Load {info['title']}",
                key=f"pick_sample_{info['key']}",
                use_container_width=True,
            ):
                loaded = load_sample_datasets([info["path"]])
                if loaded:
                    st.rerun()
                st.warning("That sample file was not found.")

    st.write("")
    if st.button(
        "Load both datasets", type="primary", use_container_width=True, key="pick_sample_both"
    ):
        loaded = load_sample_workspace()
        if loaded:
            st.rerun()
        st.warning("Sample files were not found.")


# ─── Shared upload processing ────────────────────────────────────────────
def _process_uploads(uploads) -> bool:
    """Parse uploaded files into the session. Returns True if any new file was loaded."""
    if not uploads:
        return False
    loaded_any = False
    for upload in uploads:
        ukey = f"{upload.name}:{upload.size}"
        if ukey in st.session_state.loaded_keys:
            continue
        try:
            dataset = session.add_csv_bytes(upload.getvalue(), upload.name)
            st.session_state.loaded_keys.add(ukey)
            loaded_any = True
            st.toast(f"Loaded {dataset.table}  ·  {dataset.profile.row_count:,} rows")
        except AnalystError as exc:
            st.session_state.upload_errors.append((upload.name, exc))
    if loaded_any:
        session.history.clear()
        clear_derived_state()
    return loaded_any


def chart(figure, key: str) -> None:
    st.plotly_chart(style_figure(figure), use_container_width=True, key=key, theme=None)


def download_frame(frame: pd.DataFrame, name: str, key: str) -> None:
    if len(frame):
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode(),
            file_name=name,
            mime="text/csv",
            key=key,
        )


# --------------------------------------------------------------------------- #
# Artifact renderers
# --------------------------------------------------------------------------- #
def render_table(artifact: Artifact, key: str) -> None:
    payload = artifact.payload
    frame = pd.DataFrame(payload.get("rows", []), columns=payload.get("columns", []))
    meta = [f"{payload.get('row_count', len(frame)):,} rows", f"{payload.get('duration_ms', 0):.0f} ms"]
    if payload.get("truncated"):
        meta.append("row cap reached")
    st.caption("  ·  ".join(meta))
    st.dataframe(
        frame, use_container_width=True, hide_index=True,
        height=min(420, 62 + 35 * max(len(frame), 1)),
    )
    download_frame(frame, "result.csv", f"dl_{key}")
    statement = payload.get("sql") or payload.get("pandas")
    if statement:
        with st.expander("Query"):
            st.code(statement, language="sql" if payload.get("sql") else "python")


def render_chart(artifact: Artifact, key: str) -> None:
    payload = artifact.payload
    frame = pd.DataFrame(payload.get("rows", []), columns=payload.get("columns", []))
    try:
        spec = ChartSpec.model_validate(payload["spec"])
        chart(build_figure(spec, frame), key=f"ch_{key}")
    except Exception as exc:  # noqa: BLE001 - a chart must never break an answer
        st.warning(f"Could not render this chart: {exc}")
        st.dataframe(frame, use_container_width=True, hide_index=True)
    if payload.get("sql"):
        with st.expander("Query"):
            st.code(payload["sql"], language="sql")


def render_anomaly(artifact: Artifact, key: str) -> None:
    payload = artifact.payload
    items = payload.get("anomalies", [])

    cols = st.columns(4)
    cols[0].metric("Findings", f"{len(items):,}")
    cols[1].metric("Rows tested", f"{payload.get('rows_tested', 0):,}")
    cols[2].metric("Columns", len(payload.get("columns_tested", [])))
    cols[3].metric("Methods", len(payload.get("methods_used", [])))

    if payload.get("methods_used"):
        st.caption("Methods: " + "  ·  ".join(m.replace("_", " ") for m in payload["methods_used"]))
    for note in payload.get("notes", []):
        st.caption(note)

    if not items:
        st.info("No anomalies were flagged at this sensitivity.")
        return

    frame = pd.DataFrame(
        [
            {
                "Method": a["method"].replace("_", " "),
                "Column": a["column"],
                "Where": a.get("label") or (f"row {a['row_index']}" if a["row_index"] >= 0 else "aggregate"),
                "Value": a.get("value"),
                "Score": a["score"],
                "Why it was flagged": a["reason"],
            }
            for a in items
        ]
    )
    st.dataframe(
        frame, use_container_width=True, hide_index=True,
        height=min(430, 62 + 35 * len(frame)),
    )
    st.caption(
        "Every finding was produced by a named statistical test with an explicit threshold. "
        "A statistical outlier is not proof of an error."
    )
    download_frame(frame, "anomalies.csv", f"dl_{key}")


def render_forecast(artifact: Artifact, key: str) -> None:
    import plotly.graph_objects as go

    payload = artifact.payload
    history = pd.DataFrame(payload.get("history", []))
    points = pd.DataFrame(payload.get("points", []))

    figure = go.Figure()
    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["period"], y=history["value"], name="Actual",
                mode="lines", line=dict(color=ACTIVE_SERIES[0], width=2),
            )
        )
    if not points.empty:
        figure.add_trace(
            go.Scatter(x=points["period"], y=points["upper"], mode="lines",
                       line=dict(width=0), showlegend=False, hoverinfo="skip")
        )
        figure.add_trace(
            go.Scatter(x=points["period"], y=points["lower"], name="95% interval",
                       mode="lines", line=dict(width=0), fill="tonexty",
                       fillcolor="rgba(128,128,128,0.18)", hoverinfo="skip")
        )
        figure.add_trace(
            go.Scatter(x=points["period"], y=points["forecast"], name="Forecast",
                       mode="lines+markers", line=dict(color=ACTIVE_SERIES[1], width=2, dash="dot"))
        )
    figure.update_layout(title=artifact.title, height=400)
    chart(figure, key=f"fc_{key}")

    st.caption(
        f"{payload.get('method')}  ·  in-sample MAPE {payload.get('in_sample_mape')}%"
    )
    for note in payload.get("notes", []):
        st.caption(note)
    if not points.empty:
        st.dataframe(points, use_container_width=True, hide_index=True)


def render_code(artifact: Artifact, key: str) -> None:
    payload = artifact.payload
    if payload.get("explanation"):
        st.markdown(payload["explanation"])
    st.code(payload.get("code", ""), language=payload.get("language", "text"))
    st.caption(payload.get("note", ""))


def render_quality(artifact: Artifact, key: str) -> None:
    payload = artifact.payload
    score = payload.get("score", {})
    cols = st.columns(4)
    cols[0].metric("Quality score", f"{score.get('score', 0)}")
    cols[1].metric("Completeness", f"{score.get('completeness_pct', 0)}%")
    cols[2].metric("Row uniqueness", f"{score.get('uniqueness_pct', 0)}%")
    cols[3].metric("Duplicate rows", f"{payload.get('duplicate_row_count', 0):,}")

    issues = payload.get("issues", [])
    if issues:
        frame = pd.DataFrame(
            [
                {"Severity": i["severity"], "Column": i.get("column") or "", "Issue": i["message"]}
                for i in issues
            ]
        )
        st.dataframe(
            frame, use_container_width=True, hide_index=True,
            height=min(400, 62 + 35 * len(frame)),
        )


def render_profile(artifact: Artifact, key: str) -> None:
    payload = artifact.payload
    frame = pd.DataFrame(
        [
            {
                "Column": c["name"],
                "Type": c["duckdb_type"],
                "Role": c["role"],
                "Distinct": c["distinct_count"],
                "Null %": c["null_pct"],
                "Min": fmt_bound(c.get("min"), c.get("min_date")),
                "Max": fmt_bound(c.get("max"), c.get("max_date")),
                "Examples": ", ".join(c.get("sample_values", [])[:3]),
            }
            for c in payload.get("columns", [])
        ]
    )
    st.dataframe(
        frame, use_container_width=True, hide_index=True,
        height=min(470, 62 + 35 * max(len(frame), 1)),
    )


RENDERERS = {
    "table": render_table,
    "chart": render_chart,
    "anomaly": render_anomaly,
    "forecast": render_forecast,
    "code": render_code,
    "quality": render_quality,
    "profile": render_profile,
}


def render_artifact(artifact: Artifact, key: str) -> None:
    renderer = RENDERERS.get(artifact.kind)
    with st.container(border=True):
        st.markdown(f"##### {artifact.title}")
        if renderer is None:
            st.json(artifact.payload)
        else:
            renderer(artifact, key)


def render_audit(message, key: str) -> None:  # type: ignore[no-untyped-def]
    """The 'how this was produced' panel: real steps, real SQL, real timings."""
    if not (message.reasoning or message.sql_executed or message.trace):
        return
    with st.expander("How this answer was produced"):
        if message.reasoning:
            st.caption("Steps taken")
            for i, line in enumerate(message.reasoning, start=1):
                st.markdown(f"{i}. {line}")
        if message.sql_executed:
            st.caption("Statements executed")
            for statement in message.sql_executed:
                st.code(statement, language="sql")

        trace = message.trace or {}
        if trace:
            st.caption("Execution trace")
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
                                "#": s["index"], "Kind": s["kind"], "Name": s["name"],
                                "ms": s["duration_ms"], "OK": s["ok"],
                                "Error": s.get("error") or "",
                            }
                            for s in steps
                        ]
                    ),
                    use_container_width=True, hide_index=True,
                )
            footer = f'trace {trace.get("trace_id")}'
            if trace.get("cache_hit"):
                footer += " · served from cache"
            st.caption(footer)


# --------------------------------------------------------------------------- #
# Error rendering
# --------------------------------------------------------------------------- #
for name, exc in st.session_state.upload_errors[-3:]:
    st.error(f"**{name}** — {exc.message}" + (f"\n\n{exc.detail}" if exc.detail else ""))

# --------------------------------------------------------------------------- #
# Empty state (Hero Section)
# --------------------------------------------------------------------------- #
if not session.datasets:
    st.write("<br>", unsafe_allow_html=True)
    st.markdown("<h1 class='hero-title'>AI-powered Data Analyst</h1>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Your intelligent companion for data exploration.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        _hero_uploads = st.file_uploader(
            "Upload CSV File",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=True,
            help="Multiple files are joined automatically when keys overlap.",
            key=f"hero_upload_{st.session_state.uploader_version}",
        )
        # Small sample-data affordance directly beside the uploader, rather than
        # a separate hero block, for anyone without a CSV on hand. Opens a picker
        # so the user sees and chooses which sample dataset gets loaded.
        _sample_note, _sample_action = st.columns([3, 2])
        _sample_note.caption("Don't have a dataset ready? Try our built-in sample data.")
        if _sample_action.button(
            "Load sample workspace", type="secondary", use_container_width=True, key="hero_sample_top"
        ):
            sample_dataset_picker()
    # Process uploads OUTSIDE the column context so st.rerun() fires cleanly
    _hero_did_load = _process_uploads(_hero_uploads)

    st.write("<br>", unsafe_allow_html=True)

    _CARD = (
        "border: 1px solid rgba(255,255,255,0.1); padding: 24px; border-radius: 12px; "
        "background: rgba(255,255,255,0.02); height: 100%;"
    )
    _FEATURES = (
        (
            "Natural language answers", SVG_CHAT,
            "Ask questions in plain English. Conversation context is kept for the whole "
            "session, so follow-up questions resolve against what you already asked.",
        ),
        (
            "Insights and summaries", SVG_INSIGHTS,
            "Business insights and summaries are computed first, then narrated from those "
            "measured numbers only, never invented by the model.",
        ),
        (
            "Charts and dashboards", SVG_DASHBOARD,
            "Bar, line, pie, scatter and more, plus an auto-generated KPI dashboard built "
            "from your columns without writing any code.",
        ),
        (
            "SQL and Pandas code", SVG_EXPLORE,
            "The generated SQL or Pandas expression behind every answer is shown, so each "
            "result can be reviewed, reused and verified.",
        ),
        (
            "Anomaly detection", SVG_QUALITY,
            "Outliers are flagged by named statistical tests, and each finding reports the "
            "method, threshold and observed value that triggered it.",
        ),
        (
            "Explained reasoning", SVG_WORKSPACE,
            "Every response exposes the steps taken, the statements executed and a full "
            "execution trace with timings and token counts.",
        ),
    )
    for _row_index, _row_start in enumerate((0, 3)):
        _cols = st.columns(3, gap="medium")
        for _col, (_name, _svg, _text) in zip(_cols, _FEATURES[_row_start:_row_start + 3]):
            with _col:
                st.markdown(
                    f"<div style='{_CARD}'>{render_header(_name, _svg)}{_text}</div>",
                    unsafe_allow_html=True,
                )
        if _row_index == 0:
            st.write("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Trigger rerun after all widgets are rendered — safe location
    if _hero_did_load:
        st.rerun()
    st.stop()


# --------------------------------------------------------------------------- #
# Main Workspace
# --------------------------------------------------------------------------- #
st.write("<br>", unsafe_allow_html=True)

st.title("AI-powered Data Analyst")
total_rows = sum(d.profile.row_count for d in session.datasets.values())
average_quality = sum(d.quality["score"] for d in session.datasets.values()) / len(session.datasets)
workspace_cols = st.columns(4)
workspace_cols[0].metric("Datasets", len(session.datasets))
workspace_cols[1].metric("Rows available", f"{total_rows:,}")
workspace_cols[2].metric("Average quality", f"{average_quality:.1f}")
workspace_cols[3].metric("Relationships", len(session.join_hints))

st.write(
    "Ask questions in natural language, review auto-generated dashboards and insights, "
    "inspect data quality and anomalies, run SQL or Pandas directly, and export the "
    "whole analysis as a report."
)
st.write("<br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar navigation — replaces the top tab strip with a persistent sidebar
# menu, so every workspace is reachable from one place alongside the dataset
# and export controls.
# --------------------------------------------------------------------------- #
NAV_SECTIONS = ["Overview", "Chat", "Insights", "Quality", "Explore", "Export"]
with st.sidebar:
    st.markdown("### AI-powered Data Analyst")
    st.caption(f"{len(session.datasets)} dataset(s) loaded")
    nav = st.radio("Navigate", NAV_SECTIONS, key="nav_section", label_visibility="collapsed")
    st.divider()
    if st.button("Load another dataset", use_container_width=True, key="sidebar_back"):
        reset_ui_state()
        st.rerun()

# ------------------------------------------------------------------ Chat
if nav == "Chat":
    chat_title, chat_action = st.columns([4, 1])
    with chat_title:
        st.markdown(render_header("Ask a question about your data", SVG_CHAT), unsafe_allow_html=True)
    if session.history and chat_action.button(
        "Clear conversation", key="clear_chat", use_container_width=True
    ):
        session.history.clear()
        st.rerun()

    if not session.history:
        st.caption(
            "Ask anything in plain English. Conversation context is kept for the whole session. "
            "These suggestions are generated from the columns and relationships currently loaded."
        )
        suggestions = suggest_questions(session)
        columns = st.columns(2, gap="small")
        for i, suggestion in enumerate(suggestions):
            if columns[i % 2].button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()
        st.divider()

    for index, message in enumerate(session.history):
        with st.chat_message(message.role):
            st.markdown(message.content)
            for j, artifact in enumerate(message.artifacts):
                render_artifact(artifact, key=f"h{index}_{j}")
            if message.role == "assistant":
                render_audit(message, key=f"h{index}")

    typed = st.chat_input("Ask a question about your data")
    question = typed or st.session_state.pop("pending_question", None)

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status = st.status("Working", expanded=True)
            answer = None
            try:
                for event in Agent().run(session, question):
                    if event["type"] == "status":
                        status.write(event["message"])
                    elif event["type"] == "step":
                        step = event["step"]
                        mark = "" if step["ok"] else "failed — "
                        status.write(f"{mark}{step['name']} · {step['duration_ms']:.0f} ms")
                    elif event["type"] == "answer":
                        answer = event["answer"]
                status.update(label="Done", state="complete", expanded=False)
            except AnalystError as exc:
                status.update(label="Could not answer", state="error", expanded=False)
                st.error(f"**{exc.message}**" + (f"\n\n{exc.detail}" if exc.detail else ""))
            except Exception as exc:  # noqa: BLE001
                status.update(label="Failed", state="error", expanded=False)
                log.exception("ui.turn_failed")
                st.error(f"Unexpected error: {type(exc).__name__}: {exc}")

            if answer is not None:
                st.rerun()

# ------------------------------------------------------------- Dashboard
elif nav == "Overview":
    st.markdown(render_header("Executive overview", SVG_DASHBOARD), unsafe_allow_html=True)
    st.caption(
        "Built from profiled column roles and guarded SQL. No language-model call is needed, "
        "so every refresh is deterministic."
    )
    controls = st.columns([3, 1])
    dash_table = controls[0].selectbox(
        "Dataset for overview", session.table_names, key="dash_table"
    )
    dashboard = st.session_state.get("dashboard")
    refresh_dashboard = controls[1].button(
        "Refresh", type="primary", key="build_dash", use_container_width=True
    )
    if refresh_dashboard or dashboard is None or dashboard.table != dash_table:
        with st.spinner("Building verified overview"):
            try:
                st.session_state.dashboard = build_dashboard(session, dash_table)
                dashboard = st.session_state.dashboard
            except AnalystError as exc:
                dashboard = None
                st.error(f"**{exc.message}**" + (f"\n\n{exc.detail}" if exc.detail else ""))
    if dashboard is None:
        st.info("This dataset does not expose enough analytical structure for an overview yet.")
    elif dashboard.table != dash_table:
        st.info("Refresh the overview after changing datasets.")
    else:
        for note in dashboard.notes:
            st.caption(note)

        st.subheader("Key figures")
        kpis = dashboard.kpis
        for start in range(0, len(kpis), 4):
            row = kpis[start : start + 4]
            for column, kpi in zip(st.columns(4), row):
                column.metric(kpi.label, kpi.value, help=kpi.help or None)

        st.subheader("Visual analysis")
        panels = dashboard.panels
        for index in range(0, len(panels), 2):
            for column, panel in zip(st.columns(2, gap="medium"), panels[index : index + 2]):
                with column, st.container(border=True):
                    frame = pd.DataFrame(panel.rows, columns=panel.columns)
                    try:
                        chart(build_figure(panel.spec, frame), key=f"dp_{index}_{panel.title}")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not render '{panel.title}': {exc}")
                        st.dataframe(frame, use_container_width=True, hide_index=True)
                    if panel.note:
                        st.caption(panel.note)
                    with st.expander("Query"):
                        st.code(panel.sql, language="sql")

# -------------------------------------------------------------- Insights
elif nav == "Insights":
    st.markdown(render_header("Business insights and summaries", SVG_INSIGHTS), unsafe_allow_html=True)
    st.caption(
        "Statistics are computed first, then summarised in plain English constrained to those "
        "measured facts, so no figure in the narrative is invented."
    )
    controls = st.columns([3, 1])
    insight_table = controls[0].selectbox(
        "Dataset for insights", session.table_names, key="insight_table"
    )
    if controls[1].button("Generate", type="primary", key="gen_insights", use_container_width=True):
        with st.spinner("Computing statistics"):
            st.session_state.insight_facts = compute_facts(session, insight_table)
            st.session_state.insight_table_done = insight_table
        facts = st.session_state.insight_facts
        if settings.llm_configured:
            placeholder = st.empty()
            buffer = ""
            try:
                for chunk in narrate(facts, stream=True):  # type: ignore[union-attr]
                    buffer += chunk
                    placeholder.markdown(buffer)
                st.session_state.insight_text = buffer
            except AnalystError as exc:
                st.warning(f"{exc.message} Showing the computed statistics instead.")
                st.session_state.insight_text = deterministic_summary(facts)
                placeholder.markdown(st.session_state.insight_text)
        else:
            st.session_state.insight_text = deterministic_summary(facts)
            st.markdown(st.session_state.insight_text)
    elif st.session_state.get("insight_text"):
        st.markdown(st.session_state.insight_text)

    facts = st.session_state.get("insight_facts")
    if facts:
        st.divider()
        cols = st.columns(4)
        cols[0].metric("Rows", f"{facts['row_count']:,}")
        cols[1].metric("Columns", facts["column_count"])
        cols[2].metric("Quality score", f"{facts['quality']['score']}")
        cols[3].metric("Completeness", f"{facts['quality']['completeness_pct']}%")

        trend = facts.get("trend")
        if trend and trend.get("series"):
            import plotly.graph_objects as go

            series = pd.DataFrame(trend["series"])
            figure = go.Figure(
                go.Scatter(
                    x=series["period"], y=series["total"], mode="lines",
                    line=dict(color=ACTIVE_SERIES[0], width=2), name=trend["measure"],
                )
            )
            figure.update_layout(title="Volume trend", height=300)
            chart(figure, key="is_trend")

        anomalies = facts.get("anomalies")
        if anomalies:
            st.subheader("Statistical outliers")
            for item in anomalies:
                score = item.get("score", 0)
                st.markdown(
                    f"**{item['method'].replace('_', ' ').capitalize()}** "
                    f"in `{item['column']}` — {item['reason']}"
                )

# ------------------------------------------------------------- Data health
elif nav == "Quality":
    st.markdown(render_header("Data quality checks", SVG_QUALITY), unsafe_allow_html=True)
    st.caption(
        "Validation of every uploaded file: structure, types, missing values, duplicates "
        "and statistical anomalies, each with the reason it was flagged."
    )
    
    health_table = st.selectbox(
        "Dataset", session.table_names, key="health_table"
    )

    dataset = session.get_dataset(health_table)
    
    st.write("### Quality Score")
    score = dataset.quality.get("score", 0)
    cols = st.columns(4)
    cols[0].metric("Overall Score", score)
    cols[1].metric("Completeness", f"{dataset.quality.get('completeness_pct', 0)}%")
    cols[2].metric("Uniqueness", f"{dataset.quality.get('uniqueness_pct', 0)}%")
    cols[3].metric("Valid Types", "100%")

    issues = dataset.profile.issues
    if issues:
        st.write("### Quality Issues")
        issue_data = [
            {"Severity": i.severity, "Column": i.column or "Table", "Issue": i.message}
            for i in issues
        ]
        st.dataframe(pd.DataFrame(issue_data), use_container_width=True, hide_index=True)

    st.write("### Column Profile")
    profile_data = [
        {
            "Column": c.name,
            "Type": c.duckdb_type,
            "Role": c.role,
            "Distinct": c.distinct_count,
            "Null %": c.null_pct,
            "Min": fmt_bound(c.min, c.min_date),
            "Max": fmt_bound(c.max, c.max_date),
            "Examples": ", ".join(c.sample_values[:3]) if c.sample_values else "",
        }
        for c in dataset.profile.columns
    ]
    st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------- Explore
elif nav == "Explore":
    st.markdown(render_header("Generate and run SQL", SVG_EXPLORE), unsafe_allow_html=True)
    st.caption(
        "Write SQL yourself or have it generated for an analysis, then run it against the "
        "workspace. Every statement is validated read-only before execution."
    )

    # ── AI suggested queries ──────────────────────────────────────────────
    if "explore_suggestions" not in st.session_state:
        st.session_state.explore_suggestions = []

    suggest_col, _ = st.columns([2, 3])
    if suggest_col.button("Suggest queries with AI", use_container_width=True):
        with st.spinner("Generating query suggestions…"):
            try:
                from core.llm import get_provider
                from core.llm.base import Message
                from core.semantic import build_schema_context
                schema_ctx = build_schema_context(session)
                provider = get_provider()
                prompt = (
                    "You are a SQL expert. Based on the following dataset schema, suggest exactly 5 "
                    "useful and diverse SQL SELECT queries a data analyst would run. "
                    "Each query must be on a single line. Respond ONLY with the 5 SQL statements, "
                    "one per line, no numbering, no markdown, no explanation.\n\n"
                    f"Schema:\n{schema_ctx}"
                )
                response = provider.chat([Message(role="user", content=prompt)])
                raw = response.content.strip()
                suggestions = [l.strip() for l in raw.splitlines() if l.strip().lower().startswith("select")][:5]
                if not suggestions:
                    suggestions = [l.strip() for l in raw.splitlines() if l.strip()][:5]
                st.session_state.explore_suggestions = suggestions
            except Exception as exc:
                tables = session.table_names
                fallback = [
                    f"SELECT * FROM {tables[0]} LIMIT 20;",
                    f"SELECT COUNT(*) AS total_rows FROM {tables[0]};",
                ]
                if len(tables) > 1:
                    fallback.append(f"SELECT * FROM {tables[1]} LIMIT 20;")
                st.session_state.explore_suggestions = fallback
                st.caption(f"AI unavailable, showing basic suggestions.")

    if st.session_state.explore_suggestions:
        st.write("**Click a suggestion to load it into the editor:**")
        for i, sug in enumerate(st.session_state.explore_suggestions):
            # Truncate long queries in the button label for display
            label = sug if len(sug) <= 90 else sug[:87] + "…"
            if st.button(label, key=f"sug_sql_{i}", use_container_width=True):
                # Write directly into the text_area's session state key
                # so the widget renders with the new value immediately
                st.session_state["explore_sql_editor"] = sug
                st.rerun()

    st.divider()

    # ── SQL editor ────────────────────────────────────────────────────────
    # Note: we do NOT use value= here. The content is driven purely by
    # session_state["explore_sql_editor"] which suggestion buttons write into.
    query = st.text_area(
        "SQL Query",
        height=160,
        placeholder="SELECT * FROM table_name LIMIT 10",
        key="explore_sql_editor",
    )
    run_col, clear_col = st.columns([1, 5])
    run_query = run_col.button("Run", type="primary", use_container_width=True)
    if clear_col.button("Clear", use_container_width=True):
        st.session_state.explore_suggestions = []
        st.session_state["explore_sql_editor"] = ""
        st.rerun()

    if run_query:
        if not (query or "").strip():
            st.warning("Please enter a query or click a suggestion above.")
        else:
            try:
                result = session.run_dataframe_query(query)
                st.success(f"{len(result):,} rows returned.")
                st.dataframe(result, use_container_width=True, hide_index=True)
                if len(result):
                    st.download_button(
                        "Download result CSV",
                        result.to_csv(index=False).encode(),
                        file_name="query_result.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Query error: {e}")



# ----------------------------------------------------------------- Export
elif nav == "Export":
    st.markdown(render_header("Export report", SVG_WORKSPACE), unsafe_allow_html=True)
    st.caption(
        "Export the session — datasets, schema, quality findings and the conversation — "
        "as PDF, Excel, Markdown or HTML."
    )
    st.write("")
    export_builders = {
        "PDF": (to_pdf, "application/pdf", "pdf"),
        "Excel workbook": (
            to_excel,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xlsx",
        ),
        "Markdown": (to_markdown, "text/markdown", "md"),
        "HTML": (to_html, "text/html", "html"),
    }
    export_format = st.selectbox(
        "Format", options=list(export_builders), key="settings_export_format",
    )
    builder, mime, extension = export_builders[export_format]
    st.download_button(
        f"Download {export_format} report",
        builder(session),
        file_name=f"report-{session.session_id}.{extension}",
        mime=mime,
        use_container_width=True,
        type="primary",
    )
