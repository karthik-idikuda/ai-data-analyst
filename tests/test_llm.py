"""LLM provider-layer tests.

Covers the wire-format translation for each provider and the quota/fallback
behaviour, all without network access. The fixtures are real response bodies
observed from the Gemini API, including the 429 that revealed the free tier's
20-requests-per-day-per-model quota.
"""

from __future__ import annotations

import json

import pytest

from core.config import Settings
from core.errors import LLMNotConfiguredError, LLMRateLimitError
from core.llm.base import LLMProvider, LLMResponse, Message, NullProvider, ToolCallRequest, ToolSpec
from core.llm.fallback import FallbackProvider
from core.llm.gemini import GeminiProvider, _is_gemini_3, _parse_quota_error, _sanitize_schema
from core.llm.openai_compat import OpenAICompatProvider


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"bar","x":"country"}',
        '```json\n{"type":"bar","x":"country"}\n```',
        'Here you go:\n```\n{"type":"bar","x":"country"}\n```\nHope that helps.',
        'Sure! {"type":"bar","x":"country"} — that should work.',
    ],
)
def test_parse_json_object_survives_model_wrapping(raw: str) -> None:
    parsed = LLMProvider.parse_json_object(raw)
    assert parsed["type"] == "bar"
    assert parsed["x"] == "country"


def test_parse_json_object_handles_braces_inside_strings() -> None:
    parsed = LLMProvider.parse_json_object('{"title":"a { weird } title","x":"c"}')
    assert parsed["title"] == "a { weird } title"


@pytest.mark.parametrize("raw", ["", "no json here", "[1,2,3]"])
def test_parse_json_object_rejects_non_objects(raw: str) -> None:
    with pytest.raises(ValueError):
        LLMProvider.parse_json_object(raw)


# --------------------------------------------------------------------------- #
# Gemini specifics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3.5-flash", True),
        ("gemini-3.6-flash", True),
        ("gemini-3.1-pro-preview", True),
        ("gemini-2.5-flash", False),
        ("gpt-4o-mini", False),
    ],
)
def test_gemini_3_detection(model: str, expected: bool) -> None:
    assert _is_gemini_3(model) is expected


def test_gemini_3_uses_thinking_level_and_omits_deprecated_temperature() -> None:
    """Gemini 3.x deprecated temperature/top_p/top_k in favour of thinkingLevel."""
    provider = GeminiProvider("key", default_model="gemini-3.5-flash")
    config = provider._generation_config("gemini-3.5-flash", temperature=0.0, max_tokens=None)
    assert "temperature" not in config
    assert config["thinkingConfig"]["thinkingLevel"] in ("minimal", "low", "medium", "high")


def test_gemini_2_still_uses_temperature() -> None:
    provider = GeminiProvider("key", default_model="gemini-2.5-flash")
    config = provider._generation_config("gemini-2.5-flash", temperature=0.25, max_tokens=512)
    assert config["temperature"] == 0.25
    assert "thinkingConfig" not in config
    assert config["maxOutputTokens"] == 512


def test_thought_signature_is_replayed_on_the_next_turn() -> None:
    """Gemini 3 attaches a thoughtSignature to each function call; dropping it on
    the following request loses the reasoning behind the call."""
    messages = [
        Message(role="user", content="which country?"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCallRequest(id="c1", name="run_sql", arguments={"sql": "SELECT 1"},
                                provider_state="SIGNATURE-ABC")
            ],
        ),
        Message(role="tool", tool_name="run_sql", content='{"rows": 1}'),
    ]
    wire = GeminiProvider._to_wire(messages)
    model_turn = next(c for c in wire if c["role"] == "model")
    part = model_turn["parts"][0]
    assert part["functionCall"]["name"] == "run_sql"
    assert part["thoughtSignature"] == "SIGNATURE-ABC"
    # The tool result becomes a functionResponse on a user turn.
    assert wire[-1]["parts"][0]["functionResponse"]["name"] == "run_sql"


def test_tool_result_that_is_not_json_is_still_wrapped() -> None:
    wire = GeminiProvider._to_wire([Message(role="tool", tool_name="t", content="plain text")])
    assert wire[0]["parts"][0]["functionResponse"]["response"] == {"result": "plain text"}


def test_schema_sanitiser_strips_unsupported_keywords() -> None:
    """Gemini's function-declaration parser accepts only an OpenAPI 3 subset."""
    cleaned = _sanitize_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "sql": {"type": "string", "description": "q", "pattern": "^SELECT"},
                "cols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sql"],
        }
    )
    assert cleaned["type"] == "OBJECT"
    assert "additionalProperties" not in cleaned
    assert "$schema" not in cleaned
    assert "pattern" not in cleaned["properties"]["sql"]
    assert cleaned["properties"]["sql"]["type"] == "STRING"
    assert cleaned["properties"]["cols"]["items"]["type"] == "STRING"
    assert cleaned["required"] == ["sql"]


def test_empty_object_schema_gets_properties() -> None:
    assert _sanitize_schema({"type": "object"})["properties"] == {}


# --------------------------------------------------------------------------- #
# Quota parsing — real 429 bodies
# --------------------------------------------------------------------------- #
PER_DAY_429 = json.dumps(
    {
        "error": {
            "code": 429,
            "message": (
                "You exceeded your current quota.\n* Quota exceeded for metric: "
                "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "limit: 20, model: gemini-3.6-flash\nPlease retry in 43.439048712s."
            ),
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        }
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "43s"},
            ],
        }
    }
)

PER_MINUTE_429 = json.dumps(
    {
        "error": {
            "code": 429,
            "message": "Quota exceeded. Please retry in 7.5s.",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7.5s"},
            ],
        }
    }
)


def test_parses_retry_delay_and_quota_id() -> None:
    delay, quota = _parse_quota_error(PER_DAY_429)
    assert delay == pytest.approx(43.0)
    assert quota == "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert "PerDay" in quota, "a per-day quota cannot be waited out inside a request"


def test_distinguishes_per_minute_from_per_day() -> None:
    delay, quota = _parse_quota_error(PER_MINUTE_429)
    assert delay == pytest.approx(7.5)
    assert "PerDay" not in quota


def test_falls_back_to_the_message_when_details_are_absent() -> None:
    body = json.dumps({"error": {"code": 429, "message": "Please retry in 12.25s."}})
    delay, quota = _parse_quota_error(body)
    assert delay == pytest.approx(12.25)
    assert quota is None


def test_unparsable_body_does_not_raise() -> None:
    assert _parse_quota_error("<html>502 Bad Gateway</html>") == (None, None)


# --------------------------------------------------------------------------- #
# Fallback chain
# --------------------------------------------------------------------------- #
class _StubProvider(LLMProvider):
    """Fails with the given error for named models, succeeds for the rest."""

    name = "stub"

    def __init__(self, exhausted: set[str], error: Exception | None = None) -> None:
        self.exhausted = exhausted
        self.error = error or LLMRateLimitError("out of quota")
        self.attempts: list[str] = []

    def chat(self, messages, *, system=None, tools=None, model=None, json_object=False,
             temperature=None, max_tokens=None) -> LLMResponse:
        self.attempts.append(model or "")
        if model in self.exhausted:
            raise self.error
        return LLMResponse(text=f"answered by {model}", tokens_in=10, tokens_out=5)

    def stream(self, messages, *, system=None, model=None, temperature=None, max_tokens=None):
        self.attempts.append(model or "")
        if model in self.exhausted:
            raise self.error
        yield f"answered by {model}"


CHAIN = ["a", "b", "c"]


def test_uses_the_preferred_model_when_it_works() -> None:
    stub = _StubProvider(exhausted=set())
    response = FallbackProvider(stub, CHAIN).chat([Message(role="user", content="q")])
    assert response.model == "a"
    assert stub.attempts == ["a"]


def test_advances_past_an_exhausted_model() -> None:
    stub = _StubProvider(exhausted={"a"})
    response = FallbackProvider(stub, CHAIN).chat([Message(role="user", content="q")])
    assert response.model == "b", "the answering model must be recorded, not the requested one"
    assert stub.attempts == ["a", "b"]


def test_exhausted_models_are_skipped_on_later_calls() -> None:
    """Otherwise every subsequent question re-pays the failed first attempt."""
    stub = _StubProvider(exhausted={"a"})
    provider = FallbackProvider(stub, CHAIN)
    provider.chat([Message(role="user", content="q1")])
    stub.attempts.clear()
    provider.chat([Message(role="user", content="q2")])
    assert stub.attempts == ["b"]
    assert provider.active_model == "b"


def test_raises_when_the_whole_chain_is_out_of_quota() -> None:
    stub = _StubProvider(exhausted=set(CHAIN))
    with pytest.raises(LLMRateLimitError) as exc:
        FallbackProvider(stub, CHAIN).chat([Message(role="user", content="q")])
    assert "Every configured model" in exc.value.message
    assert "a, b, c" in (exc.value.detail or "")


def test_non_quota_errors_are_never_masked_by_fallback() -> None:
    """A bad request is a bug and must surface, not silently retry on another model."""
    from core.errors import LLMError

    stub = _StubProvider(exhausted={"a"}, error=LLMError("malformed request"))
    with pytest.raises(LLMError) as exc:
        FallbackProvider(stub, CHAIN).chat([Message(role="user", content="q")])
    assert "malformed" in exc.value.message
    assert stub.attempts == ["a"], "must not try further models"


def test_stream_falls_back_before_yielding() -> None:
    stub = _StubProvider(exhausted={"a"})
    chunks = list(FallbackProvider(stub, CHAIN).stream([Message(role="user", content="q")]))
    assert chunks == ["answered by b"]


def test_empty_chain_is_rejected() -> None:
    with pytest.raises(ValueError):
        FallbackProvider(_StubProvider(set()), [])


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_model_chain_is_ordered_and_deduplicated() -> None:
    settings = Settings(
        LLM_PROVIDER="gemini",
        LLM_API_KEY="k",
        LLM_MODEL="gemini-3.5-flash",
        LLM_FALLBACK_MODELS="gemini-3.6-flash, gemini-3.5-flash ,gemini-2.5-flash,",
    )
    assert settings.model_chain == ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-2.5-flash"]


def test_provider_defaults_are_current_models() -> None:
    settings = Settings(LLM_PROVIDER="gemini", LLM_API_KEY="k")
    assert settings.default_model.startswith("gemini-3")
    assert settings.llm_configured is True


def test_missing_key_means_not_configured() -> None:
    assert Settings(LLM_PROVIDER="gemini", LLM_API_KEY="").llm_configured is False
    assert Settings(LLM_PROVIDER="none", LLM_API_KEY="k").llm_configured is False


# --------------------------------------------------------------------------- #
# OpenAI-compatible wire format
# --------------------------------------------------------------------------- #
def test_openai_wire_format_includes_system_and_tool_calls() -> None:
    messages = [
        Message(role="user", content="q"),
        Message(role="assistant", tool_calls=[ToolCallRequest(id="c1", name="run_sql",
                                                              arguments={"sql": "SELECT 1"})]),
        Message(role="tool", tool_call_id="c1", tool_name="run_sql", content="1 row"),
    ]
    wire = OpenAICompatProvider._to_wire(messages, "you are an analyst")
    assert wire[0] == {"role": "system", "content": "you are an analyst"}
    assert wire[2]["tool_calls"][0]["function"]["name"] == "run_sql"
    assert json.loads(wire[2]["tool_calls"][0]["function"]["arguments"]) == {"sql": "SELECT 1"}
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "1 row"}


def test_openai_tool_spec_conversion() -> None:
    spec = ToolSpec(name="run_sql", description="d", parameters={"type": "object", "properties": {}})
    wire = OpenAICompatProvider._tools_to_wire([spec])
    assert wire[0]["type"] == "function"
    assert wire[0]["function"]["name"] == "run_sql"


# --------------------------------------------------------------------------- #
# Null provider
# --------------------------------------------------------------------------- #
def test_null_provider_raises_an_actionable_error() -> None:
    with pytest.raises(LLMNotConfiguredError) as exc:
        NullProvider().chat([Message(role="user", content="q")])
    assert ".env" in (exc.value.detail or "")
