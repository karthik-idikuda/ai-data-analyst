"""Structured logging and per-request tracing.

Two things live here:

1. `configure_logging` — structlog wired to emit JSON (prod) or coloured
   key-value lines (local dev).
2. `Trace` — an in-memory record of everything the agent did for one question:
   every LLM call, every tool call, latency, token counts and errors. The trace
   is returned to the UI so a user can audit *how* an answer was produced, and
   it is what makes the "explain your reasoning" requirement verifiable rather
   than something the model merely asserts.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import structlog

from .config import get_settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> Any:
    configure_logging()
    return structlog.get_logger(name)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class TraceStep:
    """One observable unit of work inside an agent turn."""

    index: int
    kind: str  # "llm" | "tool" | "guard" | "cache" | "error"
    name: str
    started_at: float
    duration_ms: float = 0.0
    ok: bool = True
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 1),
            "ok": self.ok,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


@dataclass
class Trace:
    """Collects the steps of a single agent turn."""

    trace_id: str = field(default_factory=new_trace_id)
    question: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    cache_hit: bool = False

    @property
    def duration_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    @property
    def tokens_in(self) -> int:
        return sum(s.tokens_in or 0 for s in self.steps)

    @property
    def tokens_out(self) -> int:
        return sum(s.tokens_out or 0 for s in self.steps)

    @contextmanager
    def step(self, kind: str, name: str, **inputs: Any) -> Iterator[TraceStep]:
        step = TraceStep(
            index=len(self.steps) + 1,
            kind=kind,
            name=name,
            started_at=time.perf_counter(),
            input=inputs,
        )
        self.steps.append(step)
        log = get_logger("trace")
        try:
            yield step
        except Exception as exc:  # noqa: BLE001 - re-raised below
            step.ok = False
            step.error = f"{type(exc).__name__}: {exc}"
            step.duration_ms = (time.perf_counter() - step.started_at) * 1000
            log.warning(
                "step.failed",
                trace_id=self.trace_id,
                kind=kind,
                name=name,
                error=step.error,
                duration_ms=round(step.duration_ms, 1),
            )
            raise
        else:
            step.duration_ms = (time.perf_counter() - step.started_at) * 1000
            log.info(
                "step.ok",
                trace_id=self.trace_id,
                kind=kind,
                name=name,
                duration_ms=round(step.duration_ms, 1),
                tokens_in=step.tokens_in,
                tokens_out=step.tokens_out,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "duration_ms": round(self.duration_ms, 1),
            "cache_hit": self.cache_hit,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "steps": [s.to_dict() for s in self.steps],
        }
