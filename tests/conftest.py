"""Shared fixtures.

The integration fixtures load the **real** committed datasets, so the tests
exercise the same messy data the app is designed for: mixed-type invoice columns,
credit notes, negative quantities, zero prices and duplicated rows. Tiny inline
byte fixtures are used only for ingestion edge cases that the real files do not
contain (bad encodings, duplicate headers, empty files).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from core.cache import ANSWER_CACHE
from core.engine import DataSession
from core.llm.base import LLMProvider, LLMResponse, Message, ToolCallRequest, ToolSpec

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RETAIL_CSV = DATA_DIR / "online_retail_ii_international.csv"
COUNTRY_CSV = DATA_DIR / "world_bank_country_profile.csv"

requires_real_data = pytest.mark.skipif(
    not RETAIL_CSV.exists() or not COUNTRY_CSV.exists(),
    reason="Real datasets missing. Run: python scripts/fetch_real_data.py",
)


@pytest.fixture(scope="session")
def retail_path() -> Path:
    return RETAIL_CSV


@pytest.fixture(scope="session")
def country_path() -> Path:
    return COUNTRY_CSV


@pytest.fixture(scope="session")
def real_session() -> Iterator[DataSession]:
    """Session with both real datasets loaded. Session-scoped: parsing 86k rows
    twenty times would make the suite needlessly slow."""
    if not RETAIL_CSV.exists():
        pytest.skip("Real datasets missing. Run: python scripts/fetch_real_data.py")
    session = DataSession()
    session.add_csv_path(RETAIL_CSV)
    session.add_csv_path(COUNTRY_CSV)
    yield session
    session.close()


@pytest.fixture
def retail_table(real_session: DataSession) -> str:
    return "online_retail_ii_international"


@pytest.fixture
def empty_session() -> Iterator[DataSession]:
    session = DataSession()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    ANSWER_CACHE.clear()
    yield
    ANSWER_CACHE.clear()


# --------------------------------------------------------------------------- #
# Scripted fake LLM
# --------------------------------------------------------------------------- #
@dataclass
class ScriptedTurn:
    """One planned model response: either tool calls or final text."""

    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class FakeProvider(LLMProvider):
    """Deterministic provider driven by a script.

    Lets the agent loop be tested exactly — tool dispatch, error repair, duplicate
    call suppression, step budget — with no network, no cost and no flakiness.
    """

    name = "fake"

    def __init__(self, script: list[ScriptedTurn]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.systems: list[str] = []

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
        self.calls.append(
            {
                "messages": [(m.role, m.content[:200], m.tool_name) for m in messages],
                "tools": [t.name for t in tools or []],
            }
        )
        if system:
            self.systems.append(system)
        if not self.script:
            return LLMResponse(text="No further response scripted.", tokens_in=1, tokens_out=1)

        turn = self.script.pop(0)
        return LLMResponse(
            text=turn.text,
            tool_calls=[
                ToolCallRequest(id=f"call_{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(turn.tool_calls)
            ],
            tokens_in=100,
            tokens_out=50,
            finish_reason="tool_calls" if turn.tool_calls else "stop",
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
        response = self.chat(messages, system=system)
        for word in response.text.split(" "):
            yield word + " "


@pytest.fixture
def fake_provider_factory():
    def build(script: list[ScriptedTurn]) -> FakeProvider:
        return FakeProvider(script)

    return build


def last_tool_message(provider: FakeProvider) -> str:
    """The most recent tool result the fake provider was shown."""
    for call in reversed(provider.calls):
        for role, content, _name in reversed(call["messages"]):
            if role == "tool":
                return content
    return ""
