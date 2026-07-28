"""Agent loop tests.

Driven by a scripted fake provider, so the loop's behaviour is asserted exactly:
tool dispatch, self-repair after a bad column, refusal to repeat an identical
call, step budget, reasoning extraction and cache behaviour. No network, no cost,
no flakiness — but the tools underneath run against the real dataset.
"""

from __future__ import annotations

import pytest

from core.agent import Agent, _split_why
from core.cache import ANSWER_CACHE
from core.engine import DataSession
from core.errors import DatasetNotFoundError
from core.models import AgentAnswer
from tests.conftest import ScriptedTurn, requires_real_data

RETAIL = "online_retail_ii_international"
REVENUE_SQL = (
    f"SELECT country, SUM(quantity * price) AS revenue FROM {RETAIL} "
    f"GROUP BY country ORDER BY revenue DESC LIMIT 5"
)


@requires_real_data
def test_single_tool_turn_produces_answer_sql_and_artifacts(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL, "purpose": "revenue by country"})]),
            ScriptedTurn(text="EIRE leads on revenue.\nWhy: summed quantity*price grouped by country."),
        ]
    )
    answer = Agent(provider).answer(real_session, "Which country generated the highest revenue?")

    assert isinstance(answer, AgentAnswer)
    assert "EIRE" in answer.answer_markdown
    assert "Why:" not in answer.answer_markdown, "the Why line moves into the reasoning trail"
    assert any("summed quantity*price" in r for r in answer.reasoning)
    assert len(answer.sql_executed) == 1
    assert answer.artifacts and answer.artifacts[0].kind == "table"
    assert answer.artifacts[0].payload["row_count"] == 5


@requires_real_data
def test_schema_context_is_in_the_system_prompt(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory([ScriptedTurn(text="done")])
    Agent(provider).answer(real_session, "Tell me about this data")
    system = provider.systems[0]
    assert "invoicedate" in system and "world_bank_region" in system
    assert "DERIVED METRICS" in system


@pytest.mark.parametrize(
    "greeting",
    ["hi", "hey", "Hi!", "hello", "hey there", "thanks", "thank you", "ok cool",
     "what can you do?", "who are you?"],
)
@requires_real_data
def test_small_talk_never_declares_any_tool(
    real_session: DataSession, fake_provider_factory, greeting: str
) -> None:
    """A greeting must not be able to trigger a chart, a schema dump, or any
    other tool call. Regression test for a real bug: telling the model in the
    prompt not to use a tool for a greeting was not reliable — it would reach
    for a different tool (inspect_schema) instead of none. The fix omits tool
    declarations entirely on a detected small-talk turn."""
    provider = fake_provider_factory([ScriptedTurn(text="Hi! Ask me anything about your data.")])
    answer = Agent(provider).answer(real_session, greeting)

    assert provider.calls, "the provider should still be called once for a reply"
    assert provider.calls[0]["tools"] == [], "no tool must be declared on a small-talk turn"
    assert answer.artifacts == []
    assert answer.sql_executed == []
    assert "Why:" not in answer.answer_markdown


@requires_real_data
def test_a_real_question_still_declares_tools(
    real_session: DataSession, fake_provider_factory
) -> None:
    """Guard against the small-talk gate over-firing on genuine questions."""
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL, "purpose": "revenue"})]),
            ScriptedTurn(text="EIRE leads.\nWhy: summed revenue by country."),
        ]
    )
    Agent(provider).answer(real_session, "Which country generated the highest revenue?")
    assert "run_sql" in provider.calls[0]["tools"]


@requires_real_data
def test_agent_repairs_itself_after_a_hallucinated_column(
    real_session: DataSession, fake_provider_factory
) -> None:
    """The model asks for a column that does not exist; the tool error is fed back
    and the corrected second attempt succeeds."""
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": f"SELECT revenue FROM {RETAIL}"})]),
            ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL})]),
            ScriptedTurn(text="Recovered and answered."),
        ]
    )
    answer = Agent(provider).answer(real_session, "revenue by country")

    tool_messages = [
        content
        for call in provider.calls
        for role, content, _ in call["messages"]
        if role == "tool"
    ]
    assert any("TOOL ERROR" in m for m in tool_messages)
    assert len(answer.sql_executed) == 1  # only the successful statement is recorded
    assert "Recovered" in answer.answer_markdown


@requires_real_data
def test_unsafe_sql_is_reported_to_the_model_not_executed(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": f"DROP TABLE {RETAIL}"})]),
            ScriptedTurn(text="I cannot modify the data."),
        ]
    )
    answer = Agent(provider).answer(real_session, "delete everything")
    assert answer.sql_executed == []
    assert RETAIL in real_session.table_names  # table survived


@requires_real_data
def test_identical_repeated_call_is_refused(
    real_session: DataSession, fake_provider_factory
) -> None:
    args = {"sql": REVENUE_SQL, "purpose": "same"}
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", dict(args))]),
            ScriptedTurn(tool_calls=[("run_sql", dict(args))]),
            ScriptedTurn(text="Answering with what I have."),
        ]
    )
    answer = Agent(provider).answer(real_session, "revenue")
    tool_messages = [
        content for call in provider.calls for role, content, _ in call["messages"] if role == "tool"
    ]
    assert any("already made" in m for m in tool_messages)
    assert len(answer.sql_executed) == 1


@requires_real_data
def test_unknown_tool_name_is_handled(real_session: DataSession, fake_provider_factory) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("teleport", {})]),
            ScriptedTurn(text="Used a real tool instead."),
        ]
    )
    answer = Agent(provider).answer(real_session, "do something impossible")
    assert "Used a real tool instead." in answer.answer_markdown


@requires_real_data
def test_step_budget_forces_a_final_answer(
    real_session: DataSession, fake_provider_factory, monkeypatch
) -> None:
    from core import config

    monkeypatch.setattr(config.get_settings(), "max_agent_steps", 2)
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL, "purpose": "a"})]),
            ScriptedTurn(tool_calls=[("run_sql", {"sql": f"SELECT country FROM {RETAIL} LIMIT 3"})]),
            ScriptedTurn(text="Partial answer within budget."),
        ]
    )
    answer = Agent(provider).answer(real_session, "keep going forever")
    assert "Partial answer" in answer.answer_markdown


@requires_real_data
def test_final_call_still_declares_tools(
    real_session: DataSession, fake_provider_factory, monkeypatch
) -> None:
    """Regression, found by the eval harness against the live API.

    Once the history contains functionCall/functionResponse turns, Gemini rejects
    a request that does not declare the tools those turns reference (HTTP 400). The
    budget-exhausted final call must therefore still pass the tool specs.
    """
    from core import config

    monkeypatch.setattr(config.get_settings(), "max_agent_steps", 1)
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL})]),
            ScriptedTurn(text="Final answer after budget."),
        ]
    )
    Agent(provider).answer(real_session, "revenue by country")
    assert provider.calls[-1]["tools"], "the final completion must still declare tools"


@requires_real_data
def test_budget_exhaustion_with_no_prose_still_returns_something_useful(
    real_session: DataSession, fake_provider_factory, monkeypatch
) -> None:
    from core import config

    monkeypatch.setattr(config.get_settings(), "max_agent_steps", 1)
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL, "purpose": "revenue"})]),
            # Model keeps asking for tools instead of answering.
            ScriptedTurn(tool_calls=[("run_sql", {"sql": f"SELECT 1 FROM {RETAIL}"})]),
        ]
    )
    answer = Agent(provider).answer(real_session, "revenue by country")
    assert "ran out of query budget" in answer.answer_markdown
    assert "revenue" in answer.answer_markdown.lower()


@requires_real_data
def test_chart_tool_produces_a_validated_spec(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(
                tool_calls=[
                    (
                        "create_chart",
                        {
                            "sql": REVENUE_SQL,
                            "chart": {"type": "bar", "x": "country", "y": "revenue", "sort": "y_desc"},
                        },
                    )
                ]
            ),
            ScriptedTurn(text="Chart rendered."),
        ]
    )
    answer = Agent(provider).answer(real_session, "chart revenue by country")
    chart = next(a for a in answer.artifacts if a.kind == "chart")
    assert chart.payload["spec"]["type"] == "bar"
    assert chart.payload["spec"]["x"] == "country"
    assert len(chart.payload["rows"]) == 5


@requires_real_data
def test_chart_with_a_nonexistent_column_is_rejected(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(
                tool_calls=[
                    ("create_chart", {"sql": REVENUE_SQL, "chart": {"type": "bar", "x": "profit"}})
                ]
            ),
            ScriptedTurn(text="Could not chart that."),
        ]
    )
    answer = Agent(provider).answer(real_session, "chart profit")
    assert not [a for a in answer.artifacts if a.kind == "chart"]


@requires_real_data
def test_generated_pandas_code_is_never_executed(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(
                tool_calls=[
                    (
                        "generate_code",
                        {
                            "language": "pandas",
                            "code": "result = df.groupby('country')['quantity'].sum()",
                            "explanation": "Totals by country.",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="Here is the code."),
        ]
    )
    answer = Agent(provider).answer(real_session, "give me pandas code")
    code = next(a for a in answer.artifacts if a.kind == "code")
    assert "groupby" in code.payload["code"]
    assert "never executes" in code.payload["note"]


@requires_real_data
def test_dangerous_generated_code_is_blocked(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(
                tool_calls=[
                    ("generate_code", {"language": "pandas", "code": "import os\nos.system('rm -rf /')"})
                ]
            ),
            ScriptedTurn(text="Refused."),
        ]
    )
    answer = Agent(provider).answer(real_session, "delete my disk")
    assert not [a for a in answer.artifacts if a.kind == "code"]


# --------------------------------------------------------------------------- #
# Anomaly invocation policy
# --------------------------------------------------------------------------- #
@requires_real_data
def test_anomaly_tool_is_hidden_and_blocked_without_explicit_user_intent(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(
                tool_calls=[
                    ("detect_anomalies", {"table": RETAIL, "columns": ["quantity"]})
                ]
            ),
            ScriptedTurn(text="I used ordinary analysis instead."),
        ]
    )
    answer = Agent(provider).answer(real_session, "Summarise monthly sales trends")

    advertised = set(provider.calls[0]["tools"])
    assert "detect_anomalies" not in advertised
    assert not [artifact for artifact in answer.artifacts if artifact.kind == "anomaly"]
    tool_messages = [
        content
        for call in provider.calls
        for role, content, _ in call["messages"]
        if role == "tool"
    ]
    assert any("Anomaly detection was not run" in message for message in tool_messages)


@requires_real_data
def test_anomaly_tool_runs_when_user_explicitly_requests_it(
    real_session: DataSession, fake_provider_factory
) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(
                tool_calls=[
                    (
                        "detect_anomalies",
                        {"table": RETAIL, "columns": ["quantity"], "max_results": 1},
                    )
                ]
            ),
            ScriptedTurn(text="The requested anomaly scan is complete."),
        ]
    )
    answer = Agent(provider).answer(real_session, "Detect anomalies in quantity")

    advertised = set(provider.calls[0]["tools"])
    assert "detect_anomalies" in advertised
    assert any(artifact.kind == "anomaly" for artifact in answer.artifacts)


# --------------------------------------------------------------------------- #
# Conversation memory & caching
# --------------------------------------------------------------------------- #
@requires_real_data
def test_history_is_passed_to_the_next_turn(retail_path, fake_provider_factory) -> None:
    session = DataSession()
    session.add_csv_path(retail_path)
    try:
        provider = fake_provider_factory(
            [
                ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL})]),
                ScriptedTurn(text="EIRE is highest."),
                ScriptedTurn(text="Following up on EIRE."),
            ]
        )
        agent = Agent(provider)
        agent.answer(session, "Which country has the highest revenue?")
        agent.answer(session, "And what about the second one?")

        assert len(session.history) == 4
        roles = [m.role for m in session.history]
        assert roles == ["user", "assistant", "user", "assistant"]
        last_call_roles = [role for role, _, _ in provider.calls[-1]["messages"]]
        assert last_call_roles.count("user") >= 2, "prior turns must be replayed"
    finally:
        session.close()


@requires_real_data
def test_identical_question_is_served_from_cache(retail_path, fake_provider_factory) -> None:
    session = DataSession()
    session.add_csv_path(retail_path)
    try:
        provider = fake_provider_factory(
            [
                ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL})]),
                ScriptedTurn(text="Cached answer body."),
            ]
        )
        agent = Agent(provider)
        first = agent.answer(session, "Which country generated the highest revenue?")
        assert first.cache_hit is False

        calls_before = len(provider.calls)
        session.history.clear()  # same question, same schema, same (empty) context
        second = agent.answer(session, "Which country generated the highest revenue?")

        assert second.cache_hit is True
        assert len(provider.calls) == calls_before, "a cache hit must not call the model"
        assert second.answer_markdown == first.answer_markdown
        assert ANSWER_CACHE.stats()["hits"] == 1
    finally:
        session.close()


@requires_real_data
def test_cache_is_not_shared_across_different_data(
    retail_path, country_path, fake_provider_factory
) -> None:
    """The cache key includes a schema fingerprint, so the same question against
    different data must not return the first answer."""
    question = "Summarise the data"

    first_session = DataSession()
    first_session.add_csv_path(retail_path)
    second_session = DataSession()
    second_session.add_csv_path(country_path)
    try:
        provider = fake_provider_factory(
            [
                ScriptedTurn(tool_calls=[("inspect_schema", {})]),
                ScriptedTurn(text="Answer about retail."),
                ScriptedTurn(tool_calls=[("inspect_schema", {})]),
                ScriptedTurn(text="Answer about countries."),
            ]
        )
        agent = Agent(provider)
        first = agent.answer(first_session, question)
        second = agent.answer(second_session, question)
        assert first.answer_markdown != second.answer_markdown
        assert second.cache_hit is False
    finally:
        first_session.close()
        second_session.close()


def test_question_without_data_is_rejected(empty_session: DataSession, fake_provider_factory) -> None:
    with pytest.raises(DatasetNotFoundError):
        Agent(fake_provider_factory([ScriptedTurn(text="hi")])).answer(empty_session, "anything")


@requires_real_data
def test_empty_question_is_rejected(real_session: DataSession, fake_provider_factory) -> None:
    from core.errors import AnalystError

    with pytest.raises(AnalystError):
        Agent(fake_provider_factory([])).answer(real_session, "   ")


@requires_real_data
def test_trace_records_every_step(real_session: DataSession, fake_provider_factory) -> None:
    provider = fake_provider_factory(
        [
            ScriptedTurn(tool_calls=[("run_sql", {"sql": REVENUE_SQL})]),
            ScriptedTurn(text="Done."),
        ]
    )
    answer = Agent(provider).answer(real_session, "revenue by country")
    trace = answer.trace
    kinds = [s["kind"] for s in trace["steps"]]
    assert kinds.count("llm") == 2
    assert kinds.count("tool") == 1
    assert trace["tokens_in"] > 0 and trace["tokens_out"] > 0
    assert trace["trace_id"]
    assert all(s["duration_ms"] >= 0 for s in trace["steps"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected_body,expected_why",
    [
        ("Answer here.\nWhy: because of X.", "Answer here.", "because of X."),
        ("Answer.\n**Why:** bold form.", "Answer.", "bold form."),
        ("Answer.\n_Why:_ italic form.", "Answer.", "italic form."),
        ("Answer.\n> **Why**: quoted form.", "Answer.", "quoted form."),
        ("Answer.\n- Why: bulleted form.", "Answer.", "bulleted form."),
        ("No why line here.", "No why line here.", None),
        ("", "", None),
    ],
)
def test_split_why(text: str, expected_body: str, expected_why: str | None) -> None:
    body, why = _split_why(text)
    assert body == expected_body
    assert why == expected_why
