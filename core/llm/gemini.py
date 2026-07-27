"""Google Gemini provider (Generative Language API, v1beta).

Notable differences from the OpenAI shape that this adapter absorbs:

* assistant turns use the role ``model``;
* the system prompt is a separate ``systemInstruction`` field;
* tool calls are ``functionCall`` parts and results are ``functionResponse``
  parts carried on a user turn;
* function parameter schemas must be an OpenAPI 3 subset, so unsupported JSON
  Schema keywords are stripped before sending.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

import httpx

from ..config import get_settings
from ..errors import LLMError, LLMRateLimitError
from ..observability import get_logger
from .base import LLMProvider, LLMResponse, Message, ToolCallRequest, ToolSpec

log = get_logger(__name__)

_ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum", "items",
    "properties", "required", "minimum", "maximum",
}


def _sanitize_schema(schema: Any) -> Any:
    """Strip JSON Schema keywords the Gemini function-declaration parser rejects."""
    if isinstance(schema, dict):
        cleaned: dict[str, Any] = {}
        for key, value in schema.items():
            if key not in _ALLOWED_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {k: _sanitize_schema(v) for k, v in value.items()}
            elif key == "items":
                cleaned[key] = _sanitize_schema(value)
            elif key == "type" and isinstance(value, str):
                cleaned[key] = value.upper()
            else:
                cleaned[key] = value
        if cleaned.get("type") == "OBJECT" and "properties" not in cleaned:
            # Gemini rejects an OBJECT with no properties.
            cleaned["properties"] = {}
        return cleaned
    if isinstance(schema, list):
        return [_sanitize_schema(v) for v in schema]
    return schema


def _is_gemini_3(model: str) -> bool:
    """Gemini 3.x uses a different generation-config contract from 2.x."""
    name = (model or "").lower()
    return name.startswith("gemini-3")


class GeminiProvider(LLMProvider):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, *, default_model: str = "gemini-3.6-flash") -> None:
        self.api_key = api_key
        self.default_model = default_model
        self._settings = get_settings()
        self._client = httpx.Client(
            timeout=httpx.Timeout(self._settings.llm_timeout_s),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )

    # ------------------------------------------------------------- conversion
    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                payload: Any
                try:
                    payload = json.loads(msg.content)
                    if not isinstance(payload, dict):
                        payload = {"result": payload}
                except json.JSONDecodeError:
                    payload = {"result": msg.content}
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.tool_name or "tool",
                                    "response": payload,
                                }
                            }
                        ],
                    }
                )
            elif msg.role == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for call in msg.tool_calls:
                    part: dict[str, Any] = {
                        "functionCall": {"name": call.name, "args": call.arguments}
                    }
                    # Gemini 3.x: replay the thought signature so the model keeps the
                    # reasoning that led to this call across the tool loop.
                    if call.provider_state:
                        part["thoughtSignature"] = call.provider_state
                    parts.append(part)
                if not parts:
                    parts.append({"text": " "})
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": msg.content or " "}]})
        return contents

    # ------------------------------------------------------- generation config
    def _generation_config(
        self, model: str, *, temperature: float | None, max_tokens: int | None
    ) -> dict[str, Any]:
        """Build the per-model generation config.

        Gemini 3.x deprecated ``temperature``/``topP``/``topK`` and replaced them
        with a discrete ``thinkingLevel``. Sending the deprecated fields does not
        currently error, but sending them is meaningless — reasoning depth is what
        actually controls output on these models, so we set that instead and leave
        the deprecated knobs off entirely.
        """
        config: dict[str, Any] = {}
        if _is_gemini_3(model):
            level = (self._settings.llm_thinking_level or "").strip().lower()
            if level in ("minimal", "low", "medium", "high"):
                config["thinkingConfig"] = {"thinkingLevel": level}
        else:
            config["temperature"] = (
                self._settings.llm_temperature if temperature is None else temperature
            )
        if max_tokens:
            config["maxOutputTokens"] = max_tokens
        return config

    # --------------------------------------------------------------- requests
    def _request(self, model: str, body: dict[str, Any], *, stream: bool) -> httpx.Response:
        method = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/models/{model}:{method}"
        params = {"alt": "sse"} if stream else None

        last_error: Exception | None = None
        for attempt in range(1, self._settings.llm_max_retries + 1):
            try:
                if stream:
                    request = self._client.build_request("POST", url, json=body, params=params)
                    response = self._client.send(request, stream=True)
                else:
                    response = self._client.post(url, json=body)
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("llm.transport_error", provider=self.name, attempt=attempt, error=str(exc))
                time.sleep(min(2 ** attempt, 8))
                continue

            if response.status_code == 429:
                body = _safe_body(response)
                retry_after, quota_id = _parse_quota_error(body)
                # A per-day quota cannot be waited out inside a request, so fail
                # fast and let the caller fall back to another model.
                exhausted = "PerDay" in (quota_id or "")
                if not exhausted and attempt < self._settings.llm_max_retries:
                    # Honour the server's own RetryInfo rather than guessing.
                    sleep_for = retry_after if retry_after is not None else min(2 ** attempt, 20)
                    sleep_for = min(sleep_for, self._settings.llm_max_retry_wait_s)
                    log.warning(
                        "llm.rate_limited", provider=self.name, model=model,
                        sleep=round(sleep_for, 1), quota=quota_id,
                    )
                    response.close()
                    time.sleep(sleep_for)
                    continue
                raise LLMRateLimitError(
                    f"Gemini quota exhausted for '{model}'"
                    + (f" ({quota_id})" if quota_id else "")
                    + ".",
                    detail=body[:400],
                )
            if response.status_code >= 500 and attempt < self._settings.llm_max_retries:
                response.close()
                time.sleep(min(2 ** attempt, 8))
                continue
            if response.status_code >= 400:
                raise LLMError(
                    f"Gemini returned HTTP {response.status_code}.",
                    detail=_safe_body(response)[:600],
                )
            return response

        raise LLMError("Could not reach Gemini.", detail=str(last_error)[:400] if last_error else None)

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
        target = model or self.default_model
        generation = self._generation_config(target, temperature=temperature, max_tokens=max_tokens)
        if json_object and not tools:
            generation["responseMimeType"] = "application/json"

        body: dict[str, Any] = {
            "contents": self._to_wire(messages),
            "generationConfig": generation,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": _sanitize_schema(t.parameters),
                        }
                        for t in tools
                    ]
                }
            ]

        response = self._request(target, body, stream=False)
        data = response.json()

        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback") or {}
            raise LLMError(
                "Gemini returned no candidates.",
                detail=json.dumps(feedback)[:400] or "The prompt may have been blocked.",
            )
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []

        text_chunks: list[str] = []
        calls: list[ToolCallRequest] = []
        for i, part in enumerate(parts):
            # Gemini 3.x can return the model's own thought summaries as parts
            # flagged `thought: true`. They are not the answer, so they are skipped.
            if part.get("thought"):
                continue
            if "text" in part:
                text_chunks.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append(
                    ToolCallRequest(
                        id=f"{fc.get('name', 'call')}_{i}",
                        name=fc.get("name", ""),
                        arguments=fc.get("args") or {},
                        provider_state=part.get("thoughtSignature"),
                    )
                )

        usage = data.get("usageMetadata") or {}
        # Thinking tokens are billed as output and are a real cost, so they are
        # counted here rather than quietly omitted from the trace.
        tokens_out = int(usage.get("candidatesTokenCount") or 0) + int(
            usage.get("thoughtsTokenCount") or 0
        )
        return LLMResponse(
            text="".join(text_chunks),
            tool_calls=calls,
            tokens_in=int(usage.get("promptTokenCount") or 0),
            tokens_out=tokens_out,
            finish_reason=candidate.get("finishReason") or "",
            model=target,
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
        target = model or self.default_model
        body: dict[str, Any] = {
            "contents": self._to_wire(messages),
            "generationConfig": self._generation_config(
                target, temperature=temperature, max_tokens=max_tokens
            ),
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        response = self._request(target, body, stream=True)
        try:
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk:
                    continue
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                for candidate in parsed.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        if part.get("thought"):
                            continue  # internal reasoning, not answer text
                        if part.get("text"):
                            yield part["text"]
        finally:
            response.close()

    def close(self) -> None:
        self._client.close()


def _safe_body(response: httpx.Response) -> str:
    try:
        if not response.is_closed:
            response.read()
        return response.text
    except Exception:  # noqa: BLE001
        return "<unreadable response body>"


_RETRY_SECONDS = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _parse_quota_error(body: str) -> tuple[float | None, str | None]:
    """Extract the retry delay and quota id from a 429 body.

    Gemini returns a ``google.rpc.RetryInfo`` detail with the exact wait it wants
    ("Please retry in 43.4s") and a ``QuotaFailure`` naming which quota was hit.
    Guessing an exponential backoff when the server has told you the number is
    both slower and more likely to fail, and the quota id tells us whether waiting
    can possibly help: a *PerDay* quota cannot be waited out.
    """
    retry_after: float | None = None
    quota_id: str | None = None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        payload = {}

    error = payload.get("error") or {}
    for detail in error.get("details") or []:
        kind = detail.get("@type", "")
        if kind.endswith("RetryInfo"):
            raw = str(detail.get("retryDelay") or "").rstrip("s")
            try:
                retry_after = float(raw)
            except ValueError:
                pass
        elif kind.endswith("QuotaFailure"):
            violations = detail.get("violations") or []
            if violations:
                quota_id = violations[0].get("quotaId") or violations[0].get("quotaMetric")

    if retry_after is None:
        match = _RETRY_SECONDS.search(error.get("message", "") or body)
        if match:
            try:
                retry_after = float(match.group(1))
            except ValueError:
                pass
    return retry_after, quota_id
