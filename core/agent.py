"""The agent: a bounded plan → act → observe → answer loop.

Written by hand rather than with a framework. The loop is roughly 120 lines, has
no hidden control flow, is trivially unit-testable with a fake provider, and every
transition is recorded in a :class:`~core.observability.Trace`. A graph runtime
earns its keep when you need checkpointed state, resumable branches or
human-in-the-loop pauses; none of those apply to a single-turn analytics question.

Guarantees the loop provides:

* **Bounded** — at most ``MAX_AGENT_STEPS`` LLM round-trips, so a confused model
  cannot bill you indefinitely.
* **Self-repairing** — a ``ToolError`` is returned to the model as the tool result,
  so a hallucinated column name becomes a corrected retry rather than a failure.
* **Loop-breaking** — an identical tool call repeated verbatim is refused, which is
  the most common way tool-calling agents spin.
* **Auditable** — the returned answer carries the executed SQL, the artifacts and
  the full trace. "Explain your reasoning" is answered with the actual steps
  taken, not with a narrative the model invents afterwards.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterator

from .cache import ANSWER_CACHE, make_key
from .config import get_settings
from .engine import DataSession
from .errors import AnalystError, DatasetNotFoundError, LLMNotConfiguredError, ToolError
from .llm import LLMProvider, Message, get_provider
from .models import AgentAnswer, Artifact, ChatMessage
from .observability import Trace, get_logger
from .prompts import analyst_system
from .semantic import build_schema_context, schema_fingerprint
from .tools import REGISTRY, tool_specs

log = get_logger(__name__)


@dataclass
class _TurnState:
    artifacts: list[Artifact]
    reasoning: list[str]
    sql: list[str]
    tool_calls_made: int = 0


class Agent:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or get_provider()
        self.settings = get_settings()

    # ------------------------------------------------------------------ public
    def answer(
        self,
        session: DataSession,
        question: str,
        *,
        use_cache: bool = True,
        record_history: bool = True,
    ) -> AgentAnswer:
        """Answer one question. Synchronous, returns the complete result."""
        events = self.run(
            session, question, use_cache=use_cache, record_history=record_history
        )
        final: AgentAnswer | None = None
        for event in events:
            if event["type"] == "answer":
                final = event["answer"]
        if final is None:  # pragma: no cover - run always emits an answer or raises
            raise AnalystError("The agent produced no answer.")
        return final

    def run(
        self,
        session: DataSession,
        question: str,
        *,
        use_cache: bool = True,
        record_history: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Execute a turn, yielding progress events as they happen.

        Event shapes::

            {"type": "step",   "step": <trace step dict>}
            {"type": "status", "message": str}
            {"type": "answer", "answer": AgentAnswer}
        """
        question = (question or "").strip()
        if not question:
            raise AnalystError("Ask a question about your data.")
        if not session.datasets:
            raise DatasetNotFoundError(
                "No data is loaded yet.",
                detail="Upload at least one CSV file before asking questions.",
            )

        trace = Trace(question=question)
        fingerprint = schema_fingerprint(session)
        history = session.recent_history()
        tail = " | ".join(f"{m.role}:{m.content[:120]}" for m in history[-4:])
        cache_key = make_key(
            question, fingerprint, history_tail=tail, model=self.settings.default_model
        )

        if use_cache:
            cached = ANSWER_CACHE.get(cache_key)
            if cached is not None:
                answer = AgentAnswer.model_validate(cached)
                answer.cache_hit = True
                trace.cache_hit = True
                answer.trace = trace.to_dict()
                yield {"type": "status", "message": "Served from cache."}
                if record_history:
                    self._record(session, question, answer)
                yield {"type": "answer", "answer": answer}
                return

        if isinstance(self.provider, type(get_provider())) and not self.settings.llm_configured:
            raise LLMNotConfiguredError(
                "No LLM provider is configured, so natural-language questions are unavailable.",
                detail=(
                    "Set LLM_PROVIDER and LLM_API_KEY in .env. Upload, profiling, data-quality "
                    "checks, SQL execution, charts and anomaly detection work without a key."
                ),
            )

        system = analyst_system(build_schema_context(session))
        messages = self._seed_messages(history, question)
        state = _TurnState(artifacts=[], reasoning=[], sql=[])
        specs = tool_specs()
        seen_calls: set[str] = set()
        answer_text = ""

        for step_no in range(1, self.settings.max_agent_steps + 1):
            with trace.step("llm", f"completion.{step_no}", messages=len(messages)) as tstep:
                response = self.provider.chat(messages, system=system, tools=specs)
                tstep.tokens_in = response.tokens_in
                tstep.tokens_out = response.tokens_out
                tstep.output = {
                    "tool_calls": [tc.name for tc in response.tool_calls],
                    "text_preview": response.text[:200],
                    "finish_reason": response.finish_reason,
                }
            yield {"type": "step", "step": trace.steps[-1].to_dict()}

            if not response.has_tool_calls:
                answer_text = (response.text or "").strip()
                break

            if response.text.strip():
                state.reasoning.append(response.text.strip())

            messages.append(
                Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )

            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                if signature in seen_calls:
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            tool_name=call.name,
                            content=(
                                "This exact call was already made and returned the same result. "
                                "Do not repeat it — either change the arguments or answer with what you have."
                            ),
                        )
                    )
                    continue
                seen_calls.add(signature)

                yield {"type": "status", "message": f"Running {call.name}…"}
                content = self._execute_tool(session, call.name, call.arguments, state, trace)
                yield {"type": "step", "step": trace.steps[-1].to_dict()}
                messages.append(
                    Message(
                        role="tool", tool_call_id=call.id, tool_name=call.name, content=content
                    )
                )
        else:
            # Step budget exhausted: ask for a final answer with tools withheld,
            # so the user gets whatever the evidence supports instead of nothing.
            log.warning("agent.step_budget_exhausted", trace_id=trace.trace_id)
            messages.append(
                Message(
                    role="user",
                    content=(
                        "You have reached the tool budget for this turn. Answer now using only the "
                        "evidence already gathered, and state explicitly what you could not verify."
                    ),
                )
            )
            with trace.step("llm", "completion.final") as tstep:
                # Tools must still be declared even though we want prose. The
                # conversation already contains functionCall/functionResponse
                # turns, and Gemini rejects a request whose history references
                # tools that the request does not declare (HTTP 400). Any tool
                # call in the reply is ignored — the budget is spent.
                response = self.provider.chat(messages, system=system, tools=specs)
                tstep.tokens_in = response.tokens_in
                tstep.tokens_out = response.tokens_out
            answer_text = (response.text or "").strip()
            yield {"type": "step", "step": trace.steps[-1].to_dict()}
            if not answer_text and response.has_tool_calls:
                answer_text = (
                    "I ran out of query budget for this question before I could finish. "
                    "Here is what I established:\n\n"
                    + "\n".join(f"- {r}" for r in state.reasoning[-5:])
                    + "\n\nAsk a narrower question and I can answer it directly."
                )

        answer_text, why = _split_why(answer_text)
        if why:
            state.reasoning.append(why)
        if not answer_text:
            answer_text = (
                "I could not produce an answer for that. Try rephrasing the question, or ask about "
                "one of the columns listed in the dataset panel."
            )

        answer = AgentAnswer(
            answer_markdown=answer_text,
            reasoning=state.reasoning,
            artifacts=state.artifacts,
            sql_executed=state.sql,
            trace=trace.to_dict(),
            cache_hit=False,
        )
        if use_cache and state.tool_calls_made > 0:
            ANSWER_CACHE.set(cache_key, answer.model_dump(mode="json"))
        if record_history:
            self._record(session, question, answer)

        log.info(
            "agent.turn_complete",
            trace_id=trace.trace_id,
            steps=len(trace.steps),
            tools=state.tool_calls_made,
            tokens_in=trace.tokens_in,
            tokens_out=trace.tokens_out,
            duration_ms=round(trace.duration_ms, 1),
        )
        yield {"type": "answer", "answer": answer}

    # ----------------------------------------------------------------- helpers
    def _seed_messages(self, history: list[ChatMessage], question: str) -> list[Message]:
        """Rolling window of prior turns plus the new question.

        Assistant turns are truncated: the model needs the thread of the
        conversation, not every table it previously printed.
        """
        messages: list[Message] = []
        for msg in history:
            content = msg.content
            if msg.role == "assistant":
                content = content[:900]
                if msg.sql_executed:
                    content += "\n[SQL used: " + " ; ".join(s.replace("\n", " ")[:160] for s in msg.sql_executed[:2]) + "]"
            messages.append(Message(role=msg.role, content=content))
        messages.append(Message(role="user", content=question))
        return messages

    def _execute_tool(
        self,
        session: DataSession,
        name: str,
        arguments: dict[str, Any],
        state: _TurnState,
        trace: Trace,
    ) -> str:
        tool = REGISTRY.get(name)
        if tool is None:
            with trace.step("error", f"tool.{name}", arguments=arguments) as tstep:
                tstep.ok = False
                tstep.error = "unknown tool"
            return (
                f"There is no tool named '{name}'. Available tools: {', '.join(REGISTRY.names())}."
            )

        try:
            with trace.step("tool", name, arguments=_trim(arguments)) as tstep:
                outcome = tool.run(session, arguments)
                tstep.output = {
                    "artifacts": [a.kind for a in outcome.artifacts],
                    "sql": outcome.sql,
                    "text_preview": outcome.model_text[:300],
                }
        except AnalystError as exc:
            # Recoverable: hand the message back so the model can repair its call.
            log.info("agent.tool_error", tool=name, code=exc.code, message=exc.message)
            detail = f" {exc.detail}" if exc.detail else ""
            return f"TOOL ERROR ({exc.code}): {exc.message}{detail}\nFix the problem and try again."
        except Exception as exc:  # noqa: BLE001 - unexpected bug, still recoverable for the turn
            log.exception("agent.tool_crashed", tool=name)
            return f"TOOL ERROR (internal): {type(exc).__name__}: {exc}"

        state.tool_calls_made += 1
        state.artifacts.extend(outcome.artifacts)
        state.sql.extend(outcome.sql)
        if outcome.reasoning:
            state.reasoning.append(outcome.reasoning)
        return outcome.model_text

    @staticmethod
    def _record(session: DataSession, question: str, answer: AgentAnswer) -> None:
        session.add_message(ChatMessage(role="user", content=question))
        session.add_message(
            ChatMessage(
                role="assistant",
                content=answer.answer_markdown,
                artifacts=answer.artifacts,
                reasoning=answer.reasoning,
                sql_executed=answer.sql_executed,
                trace=answer.trace,
            )
        )


# Matches the closing reasoning line in the forms models actually emit:
# `Why: …`, `**Why:** …`, `_Why:_ …`, `> **Why**: …`
_WHY_LINE = re.compile(r"^\s*[>\-*_\s]*\*{0,2}_{0,2}why\*{0,2}_{0,2}\s*:\s*\*{0,2}_{0,2}\s*(.*?)[\s*_]*$", re.IGNORECASE)


def _split_why(text: str) -> tuple[str, str | None]:
    """Separate the trailing ``Why:`` line into the reasoning trail."""
    if not text:
        return "", None
    lines = text.rstrip().splitlines()
    for i in range(len(lines) - 1, max(len(lines) - 4, -1), -1):
        match = _WHY_LINE.match(lines[i])
        if match:
            why = match.group(1).strip()
            remaining = "\n".join(lines[:i] + lines[i + 1 :]).rstrip()
            return remaining, why or None
    return text.rstrip(), None


def _trim(arguments: dict[str, Any], limit: int = 600) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, str) and len(value) > limit:
            out[key] = value[:limit] + "…"
        else:
            out[key] = value
    return out
