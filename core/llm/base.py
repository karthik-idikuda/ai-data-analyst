"""Provider-agnostic LLM interface.

Every provider is reached through raw HTTP (httpx) rather than a vendor SDK.
That is deliberate: three SDKs would mean three dependency-upgrade paths and
three sets of breaking changes, and the request bodies involved are small. It
also keeps the tool-calling contract identical across providers, so the agent
code has no provider conditionals in it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list["ToolCallRequest"] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque provider state that must be replayed with the conversation.
    # Gemini 3.x attaches a `thoughtSignature` to every function-call part; if it
    # is not echoed back on the following turn the model loses the reasoning that
    # produced the call, which degrades multi-step tool loops. Other providers
    # leave this unset and ignore it.
    provider_state: str | None = None


@dataclass
class ToolSpec:
    """A JSON-schema described function the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str = ""
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """Minimal surface the agent depends on."""

    name: str = "abstract"

    @abstractmethod
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
        """One completion round-trip."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield text deltas. Tool calling is not used on the streaming path."""

    # -------------------------------------------------------------- utilities
    @staticmethod
    def parse_json_object(text: str) -> dict[str, Any]:
        """Extract a JSON object from model text.

        Models wrap JSON in prose or fences even when told not to, so we degrade
        gracefully: direct parse, then fenced block, then first balanced object.
        """
        candidate = (text or "").strip()
        if not candidate:
            raise ValueError("empty response")
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        if "```" in candidate:
            blocks = candidate.split("```")
            for block in blocks:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        parsed = json.loads(block)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue

        start = candidate.find("{")
        if start >= 0:
            depth, in_string, escape = 0, False, False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(candidate[start : i + 1])
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            break
        raise ValueError(f"no JSON object found in response: {candidate[:200]!r}")


class NullProvider(LLMProvider):
    """Used when no API key is configured.

    It raises a typed, actionable error instead of crashing, which lets the whole
    deterministic half of the app (upload, profiling, quality checks, SQL
    execution, anomaly detection, charts) work with no credentials at all.
    """

    name = "none"

    def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:  # noqa: D102
        from ..errors import LLMNotConfiguredError

        raise LLMNotConfiguredError(
            "No LLM provider is configured.",
            detail="Set LLM_PROVIDER and LLM_API_KEY in your .env file (see .env.example).",
        )

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[str]:  # noqa: D102
        self.chat()
        yield ""  # pragma: no cover
