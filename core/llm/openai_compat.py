"""OpenAI-compatible chat-completions provider (covers OpenAI and Groq).

Groq exposes an OpenAI-compatible surface at ``/openai/v1``, so one client
handles both; only the base URL and default model differ.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import httpx

from ..config import get_settings
from ..errors import LLMError, LLMRateLimitError
from ..observability import get_logger
from .base import LLMProvider, LLMResponse, Message, ToolCallRequest, ToolSpec

log = get_logger(__name__)


class OpenAICompatProvider(LLMProvider):
    name = "openai-compat"
    base_url = "https://api.openai.com/v1"

    def __init__(self, api_key: str, *, base_url: str | None = None, default_model: str = "") -> None:
        self.api_key = api_key
        self.default_model = default_model
        if base_url:
            self.base_url = base_url.rstrip("/")
        self._settings = get_settings()
        self._client = httpx.Client(
            timeout=httpx.Timeout(self._settings.llm_timeout_s),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    # ------------------------------------------------------------- conversion
    @staticmethod
    def _to_wire(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "tool":
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or msg.tool_name or "tool",
                        "content": msg.content,
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                wire.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                wire.append({"role": msg.role, "content": msg.content})
        return wire

    @staticmethod
    def _tools_to_wire(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # --------------------------------------------------------------- requests
    def _post(self, payload: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(1, self._settings.llm_max_retries + 1):
            try:
                if stream:
                    request = self._client.build_request("POST", url, json=payload)
                    response = self._client.send(request, stream=True)
                else:
                    response = self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("llm.transport_error", provider=self.name, attempt=attempt, error=str(exc))
                time.sleep(min(2 ** attempt, 8))
                continue

            if response.status_code == 429:
                retry_after = float(response.headers.get("retry-after", min(2 ** attempt, 20)))
                if attempt < self._settings.llm_max_retries:
                    log.warning("llm.rate_limited", provider=self.name, sleep=retry_after)
                    response.close()
                    time.sleep(retry_after)
                    continue
                body = _safe_body(response)
                raise LLMRateLimitError(
                    "The model provider is rate limiting this key.",
                    detail=body[:400],
                )
            if response.status_code >= 500:
                if attempt < self._settings.llm_max_retries:
                    response.close()
                    time.sleep(min(2 ** attempt, 8))
                    continue
            if response.status_code >= 400:
                body = _safe_body(response)
                raise LLMError(
                    f"Model provider returned HTTP {response.status_code}.",
                    detail=body[:600],
                )
            return response

        raise LLMError(
            "Could not reach the model provider.",
            detail=str(last_error)[:400] if last_error else None,
        )

    # ------------------------------------------------------------------- chat
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
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._to_wire(messages, system),
            "temperature": self._settings.llm_temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = self._tools_to_wire(tools)
            payload["tool_choice"] = "auto"
        if json_object and not tools:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = self._post(payload)
        data = response.json()
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise LLMError("Model returned no choices.", detail=json.dumps(data)[:400]) from exc

        message = choice.get("message") or {}
        calls: list[ToolCallRequest] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__raw": fn.get("arguments")}
            calls.append(
                ToolCallRequest(
                    id=raw.get("id") or fn.get("name", "call"),
                    name=fn.get("name", ""),
                    arguments=args if isinstance(args, dict) else {"value": args},
                )
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=calls,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason") or "",
            model=data.get("model") or payload["model"],
        )

    # ----------------------------------------------------------------- stream
    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._to_wire(messages, system),
            "temperature": self._settings.llm_temperature if temperature is None else temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = self._post(payload, stream=True)
        try:
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk in ("", "[DONE]"):
                    continue
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                for choice in parsed.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta
        finally:
            response.close()

    def close(self) -> None:
        self._client.close()


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    base_url = "https://api.openai.com/v1"


def _safe_body(response: httpx.Response) -> str:
    try:
        if response.is_stream_consumed or not response.is_closed:
            response.read()
        return response.text
    except Exception:  # noqa: BLE001
        return "<unreadable response body>"
