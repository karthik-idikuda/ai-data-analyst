"""FastAPI application.

A thin transport layer: it parses requests, checks auth, calls `core`, and maps
typed errors to status codes. No analysis logic lives here, which is why the test
suite and the evaluation harness can exercise the engine without booting a server.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from core import __version__
from core.agent import Agent
from core.cache import ANSWER_CACHE
from core.config import get_settings
from core.dashboard import build as build_dashboard
from core.engine import SESSIONS, DataSession
from core.errors import AnalystError, DatasetNotFoundError, SessionNotFoundError
from core.insights import compute_facts, deterministic_summary, narrate
from core.llm import get_provider
from core.models import AgentAnswer
from core.observability import configure_logging, get_logger, new_trace_id
from core.profile import quality_score
from core.reports import to_excel, to_html, to_markdown, to_pdf
from core.semantic import build_schema_context, suggest_questions
from core.tools import REGISTRY
from core.tools.analytics import DETECT_ANOMALIES, FORECAST

configure_logging()
log = get_logger("api")
settings = get_settings()

app = FastAPI(
    title="AI Data Analyst",
    version=__version__,
    description=(
        "Upload CSV files and query them in natural language. The LLM proposes SQL; "
        "a deterministic guard validates it; DuckDB executes it read-only."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-tenant demo service; tighten before real deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional shared-secret auth.

    Disabled unless ``APP_API_KEY`` is set. Documented rather than silently absent:
    when unset, every endpoint is open, which is fine for local use and not fine
    on a public host.
    """
    expected = settings.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# --------------------------------------------------------------------------- #
# Middleware / error handling
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def add_trace(request: Request, call_next):  # type: ignore[no-untyped-def]
    trace_id = request.headers.get("x-trace-id") or new_trace_id()
    started = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - started) * 1000
    response.headers["x-trace-id"] = trace_id
    response.headers["x-response-time-ms"] = f"{duration:.1f}"
    log.info(
        "http.request",
        trace_id=trace_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration, 1),
    )
    return response


@app.exception_handler(AnalystError)
async def analyst_error_handler(_: Request, exc: AnalystError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    log.exception("http.unhandled")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Something went wrong handling that request.",
                "detail": f"{type(exc).__name__}: {exc}"[:300],
            }
        },
    )


def _session(session_id: str) -> DataSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise SessionNotFoundError(
            f"Session '{session_id}' does not exist or has expired.",
            detail="Create a new session and re-upload your files.",
        )
    return session


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_cache: bool = True


class SqlRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20_000)
    max_rows: int | None = Field(default=None, ge=1, le=100_000)


class AnomalyRequest(BaseModel):
    table: str | None = None
    columns: list[str] | None = None
    sensitivity: str = "medium"
    max_results: int = Field(default=10, ge=1, le=50)


class ForecastRequest(BaseModel):
    table: str | None = None
    date_column: str | None = None
    value_column: str | None = None
    periods: int = Field(default=6, ge=1, le=36)
    freq: str = "monthly"
    agg: str = "sum"


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    provider = get_provider()
    return {
        "status": "ok",
        "version": __version__,
        "llm": {
            "configured": settings.llm_configured,
            "provider": provider.name,
            "model": settings.default_model or None,
        },
        "auth_required": bool(settings.api_key),
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_files_per_session": settings.max_files_per_session,
            "max_result_rows": settings.max_result_rows,
            "query_timeout_s": settings.query_timeout_s,
            "max_agent_steps": settings.max_agent_steps,
        },
        "tools": REGISTRY.names(),
    }


@app.get("/metrics", tags=["meta"], dependencies=[Depends(require_api_key)])
def metrics() -> dict[str, Any]:
    return {"cache": ANSWER_CACHE.stats(), "sessions": len(SESSIONS._sessions)}


# --------------------------------------------------------------------------- #
# Sessions & datasets
# --------------------------------------------------------------------------- #
@app.post("/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)])
def create_session() -> dict[str, Any]:
    session = SESSIONS.create()
    return session.summary()


@app.get("/sessions/{session_id}", tags=["sessions"], dependencies=[Depends(require_api_key)])
def get_session(session_id: str) -> dict[str, Any]:
    return _session(session_id).summary()


@app.delete("/sessions/{session_id}", tags=["sessions"], dependencies=[Depends(require_api_key)])
def delete_session(session_id: str) -> dict[str, str]:
    _session(session_id)
    SESSIONS.delete(session_id)
    return {"status": "deleted"}


@app.post(
    "/sessions/{session_id}/datasets",
    tags=["datasets"],
    dependencies=[Depends(require_api_key)],
)
async def upload_datasets(session_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload one or more CSVs.

    Partial success is a first-class outcome: a bad third file does not discard the
    two that loaded, and each failure is reported with its own typed error.
    """
    session = _session(session_id)
    loaded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for upload in files:
        raw = await upload.read()
        try:
            dataset = session.add_csv_bytes(raw, upload.filename or "upload.csv")
        except AnalystError as exc:
            failed.append({"file": upload.filename, "error": exc.to_dict()})
            continue
        loaded.append(
            {
                "table": dataset.table,
                "source_name": dataset.source_name,
                "rows": dataset.profile.row_count,
                "columns": dataset.profile.column_count,
                "quality": dataset.quality,
                "profile": dataset.profile.model_dump(mode="json"),
            }
        )

    if not loaded and failed:
        return JSONResponse(status_code=400, content={"loaded": [], "failed": failed})
    return {
        "loaded": loaded,
        "failed": failed,
        "session": session.summary(),
        "suggestions": suggest_questions(session),
    }


@app.delete(
    "/sessions/{session_id}/datasets/{table}",
    tags=["datasets"],
    dependencies=[Depends(require_api_key)],
)
def remove_dataset(session_id: str, table: str) -> dict[str, Any]:
    session = _session(session_id)
    session.remove_dataset(table)
    return session.summary()


@app.get(
    "/sessions/{session_id}/schema",
    tags=["datasets"],
    dependencies=[Depends(require_api_key)],
)
def get_schema(session_id: str) -> dict[str, Any]:
    session = _session(session_id)
    return {
        "tables": [d.profile.model_dump(mode="json") for d in session.datasets.values()],
        "join_hints": [h.model_dump() for h in session.join_hints],
        "prompt_context": build_schema_context(session),
        "suggestions": suggest_questions(session),
    }


@app.get(
    "/sessions/{session_id}/quality",
    tags=["datasets"],
    dependencies=[Depends(require_api_key)],
)
def get_quality(session_id: str) -> dict[str, Any]:
    session = _session(session_id)
    if not session.datasets:
        raise DatasetNotFoundError("No datasets loaded.")
    return {
        table: {
            "score": quality_score(dataset.profile),
            "issues": [i.model_dump(mode="json") for i in dataset.profile.issues],
        }
        for table, dataset in session.datasets.items()
    }


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
@app.post(
    "/sessions/{session_id}/chat",
    tags=["chat"],
    response_model=AgentAnswer,
    dependencies=[Depends(require_api_key)],
)
def chat(session_id: str, request: ChatRequest) -> AgentAnswer:
    session = _session(session_id)
    return Agent().answer(session, request.question, use_cache=request.use_cache)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.post(
    "/sessions/{session_id}/chat/stream",
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
def chat_stream(session_id: str, request: ChatRequest) -> StreamingResponse:
    """Server-sent events: real progress as each tool runs, then the answer.

    Step events are emitted the moment they complete, so the client shows genuine
    progress rather than a spinner. Token-level streaming lives on
    ``/insights/stream``, where the call needs no tools and streaming is free.
    """
    session = _session(session_id)

    def generate() -> Iterator[str]:
        agent = Agent()
        try:
            for event in agent.run(session, request.question, use_cache=request.use_cache):
                if event["type"] == "answer":
                    yield _sse({"type": "answer", "answer": event["answer"].model_dump(mode="json")})
                else:
                    yield _sse(event)
        except AnalystError as exc:
            yield _sse({"type": "error", "error": exc.to_dict()})
        except Exception as exc:  # noqa: BLE001
            log.exception("chat_stream.failed")
            yield _sse(
                {
                    "type": "error",
                    "error": {
                        "code": "internal_error",
                        "message": "The request failed while streaming.",
                        "detail": f"{type(exc).__name__}: {exc}"[:300],
                    },
                }
            )
        yield _sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get(
    "/sessions/{session_id}/history",
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
def history(session_id: str) -> dict[str, Any]:
    session = _session(session_id)
    return {"messages": [m.model_dump(mode="json") for m in session.history]}


# --------------------------------------------------------------------------- #
# Direct analytics (work without an LLM key)
# --------------------------------------------------------------------------- #
@app.post("/sessions/{session_id}/sql", tags=["analytics"], dependencies=[Depends(require_api_key)])
def run_sql(session_id: str, request: SqlRequest) -> dict[str, Any]:
    session = _session(session_id)
    result, guard = session.execute_sql(request.sql, max_rows=request.max_rows)
    return {
        "result": result.model_dump(mode="json"),
        "guard": {
            "tables": guard.tables,
            "limit_applied": guard.limit_applied,
            "warnings": guard.warnings,
            "rewritten": guard.sql != guard.original_sql,
        },
    }


@app.post(
    "/sessions/{session_id}/anomalies",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],
)
def anomalies(session_id: str, request: AnomalyRequest) -> dict[str, Any]:
    session = _session(session_id)
    outcome = DETECT_ANOMALIES.run(session, request.model_dump(exclude_none=True))
    return {
        "report": outcome.artifacts[0].payload if outcome.artifacts else {},
        "summary": outcome.model_text,
    }


@app.post(
    "/sessions/{session_id}/forecast",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],
)
def forecast(session_id: str, request: ForecastRequest) -> dict[str, Any]:
    session = _session(session_id)
    outcome = FORECAST.run(session, request.model_dump(exclude_none=True))
    return {
        "forecast": outcome.artifacts[0].payload if outcome.artifacts else {},
        "summary": outcome.model_text,
    }


@app.get(
    "/sessions/{session_id}/insights",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],
)
def insights(session_id: str, table: str | None = None, narrative: bool = True) -> dict[str, Any]:
    session = _session(session_id)
    facts = compute_facts(session, table)
    if not narrative or not settings.llm_configured:
        return {"facts": facts, "briefing": deterministic_summary(facts), "narrated": False}
    try:
        text = narrate(facts)
    except AnalystError as exc:
        return {
            "facts": facts,
            "briefing": deterministic_summary(facts),
            "narrated": False,
            "warning": exc.to_dict(),
        }
    return {"facts": facts, "briefing": text, "narrated": True}


@app.get(
    "/sessions/{session_id}/insights/stream",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],
)
def insights_stream(session_id: str, table: str | None = None) -> StreamingResponse:
    """Token-by-token streamed briefing. No tools, so this is true LLM streaming."""
    session = _session(session_id)
    facts = compute_facts(session, table)

    def generate() -> Iterator[str]:
        yield _sse({"type": "facts", "facts": facts})
        try:
            for chunk in narrate(facts, stream=True):  # type: ignore[union-attr]
                yield _sse({"type": "token", "text": chunk})
        except AnalystError as exc:
            yield _sse({"type": "error", "error": exc.to_dict()})
            yield _sse({"type": "token", "text": deterministic_summary(facts)})
        yield _sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get(
    "/sessions/{session_id}/dashboard",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],
)
def dashboard(session_id: str, table: str | None = None) -> dict[str, Any]:
    """Auto-generated KPIs and chart panels. Deterministic — no LLM, no key needed."""
    session = _session(session_id)
    return build_dashboard(session, table).to_dict()


@app.get(
    "/sessions/{session_id}/report",
    tags=["analytics"],
    dependencies=[Depends(require_api_key)],
)
def report(session_id: str, format: str = "markdown") -> Any:
    session = _session(session_id)
    fmt = format.lower()
    if fmt in ("md", "markdown"):
        return PlainTextResponse(
            to_markdown(session),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="report-{session_id}.md"'},
        )
    if fmt == "html":
        return PlainTextResponse(
            to_html(session),
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="report-{session_id}.html"'},
        )
    if fmt in ("xlsx", "excel"):
        return Response(
            to_excel(session),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="report-{session_id}.xlsx"'},
        )
    if fmt == "pdf":
        return Response(
            to_pdf(session),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="report-{session_id}.pdf"'},
        )
    raise AnalystError(
        f"Unsupported report format '{format}'.",
        detail="Use 'markdown', 'html', 'xlsx' or 'pdf'.",
    )
