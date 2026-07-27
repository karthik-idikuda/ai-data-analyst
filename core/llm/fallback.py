"""Model fallback chain.

Motivation, measured rather than assumed: on the Gemini free tier the quota
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` is **20 requests per day, per
model**. A single multi-step analytical question costs 6–8 requests, so a single
model gives roughly two questions a day. The quota is per model, so trying the
next model when one is exhausted is not a trick — it is the documented shape of
the limit.

This is also a real production pattern independent of free tiers: when the
preferred model is throttled or has an outage, degrading to a smaller model beats
returning an error to the user.

Two rules make it honest:

* Fallback happens **only** on quota/rate-limit errors, never on a bad request or
  a malformed response, which must surface as real bugs.
* The model that actually answered is recorded on the response and in the trace,
  so nobody mistakes a Flash-Lite answer for a frontier-model answer.
"""

from __future__ import annotations

from typing import Iterator

from ..errors import LLMError, LLMRateLimitError
from ..observability import get_logger
from .base import LLMProvider, LLMResponse, Message, ToolSpec

log = get_logger(__name__)


class FallbackProvider(LLMProvider):
    """Wraps one provider and an ordered list of models to try."""

    def __init__(self, provider: LLMProvider, models: list[str]) -> None:
        if not models:
            raise ValueError("a fallback chain needs at least one model")
        self.provider = provider
        self.models = models
        self.name = f"{provider.name}+fallback"
        # Models known to be exhausted for this process; skipped on later calls so
        # every subsequent question does not re-pay the failed first attempt.
        self._exhausted: set[str] = set()

    @property
    def active_model(self) -> str:
        for model in self.models:
            if model not in self._exhausted:
                return model
        return self.models[-1]

    def _chain(self, requested: str | None) -> list[str]:
        if requested:
            ordered = [requested] + [m for m in self.models if m != requested]
        else:
            ordered = list(self.models)
        available = [m for m in ordered if m not in self._exhausted]
        # If everything is marked exhausted, try the whole chain again: a per-minute
        # quota may have recovered since it was marked.
        return available or ordered

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        json_object: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        chain = self._chain(model)
        last: LLMRateLimitError | None = None

        for candidate in chain:
            try:
                response = self.provider.chat(
                    messages,
                    system=system,
                    tools=tools,
                    model=candidate,
                    json_object=json_object,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except LLMRateLimitError as exc:
                last = exc
                self._exhausted.add(candidate)
                remaining = [m for m in chain if m not in self._exhausted]
                log.warning(
                    "llm.fallback",
                    exhausted=candidate,
                    next=remaining[0] if remaining else None,
                    reason=exc.message,
                )
                continue
            if candidate != chain[0]:
                log.info("llm.answered_by_fallback", model=candidate)
            response.model = candidate
            return response

        raise LLMRateLimitError(
            "Every configured model is out of quota.",
            detail=(
                f"Tried: {', '.join(chain)}. "
                + (last.detail or "")
                + " Free-tier Gemini quotas reset daily; add billing or set "
                "LLM_MODEL/LLM_FALLBACK_MODELS to models with remaining quota."
            ),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        chain = self._chain(model)
        for candidate in chain:
            try:
                # Materialise the first chunk here so a quota error surfaces now,
                # while we can still switch models, rather than mid-stream.
                iterator = self.provider.stream(
                    messages,
                    system=system,
                    model=candidate,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                first = next(iterator, None)
            except LLMRateLimitError as exc:
                self._exhausted.add(candidate)
                log.warning("llm.fallback_stream", exhausted=candidate, reason=exc.message)
                continue

            if first is not None:
                yield first
            yield from iterator
            return

        raise LLMRateLimitError(
            "Every configured model is out of quota.",
            detail=f"Tried: {', '.join(chain)}.",
        )
