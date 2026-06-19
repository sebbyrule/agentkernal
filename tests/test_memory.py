"""Memory store tests (Phase 3, design §13).

Stores are tested against canonical ``Message`` objects; the kernel never sees
store-specific shapes.
"""

from __future__ import annotations

from agentkernel.memory import FileMemoryStore, InMemoryMemoryStore
from agentkernel.types import Message, ToolCall, ToolResult
from tests.fakes import FakeProvider, text_response


def _sample_messages() -> list[Message]:
    return [
        Message(role="user", content="hello", token_estimate=2),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall("c1", "echo", {"value": "hi"})],
        ),
        Message(
            role="tool",
            tool_results=[ToolResult("c1", "echo: hi", is_error=False, data={"x": 1})],
        ),
    ]


def test_in_memory_store_round_trips_messages():
    store = InMemoryMemoryStore()
    messages = _sample_messages()
    store.save("s1", messages)
    loaded = store.load("s1")
    assert len(loaded) == len(messages)
    assert loaded[0].content == "hello"
    assert loaded[1].tool_calls[0].name == "echo"
    assert loaded[2].tool_results[0].call_id == "c1"


def test_in_memory_store_is_isolated_by_session_id():
    store = InMemoryMemoryStore()
    store.save("a", [Message(role="user", content="a")])
    store.save("b", [Message(role="user", content="b")])
    assert store.load("a")[0].content == "a"
    assert store.load("b")[0].content == "b"


def test_file_memory_store_round_trips(tmp_path):
    store = FileMemoryStore(tmp_path / "memory")
    messages = _sample_messages()
    store.save("session-2", messages)
    loaded = store.load("session-2")
    assert len(loaded) == len(messages)
    assert loaded[2].tool_results[0].data == {"x": 1}


def test_file_memory_store_returns_empty_for_unknown_session(tmp_path):
    store = FileMemoryStore(tmp_path / "memory")
    assert store.load("does-not-exist") == []


def test_file_memory_store_overwrites_on_save(tmp_path):
    store = FileMemoryStore(tmp_path / "memory")
    store.save("s", [Message(role="user", content="first")])
    store.save("s", [Message(role="user", content="second")])
    assert len(store.load("s")) == 1
    assert store.load("s")[0].content == "second"


def test_agent_loads_memory_into_empty_context(agent_builder):
    store = InMemoryMemoryStore()
    store.save("s1", [Message(role="user", content="remember this")])
    provider = FakeProvider([text_response("ok")])
    agent = agent_builder(provider, memory=store)
    agent.telemetry.session_id = "s1"
    agent.run("go")
    # The provider saw the loaded memory plus the new user message.
    assert any(m.content == "remember this" for m in provider.calls[0])
    assert any(m.content == "go" for m in provider.calls[0])


def test_agent_saves_conversation_after_run(agent_builder):
    store = InMemoryMemoryStore()
    provider = FakeProvider([text_response("answer")])
    agent = agent_builder(provider, memory=store)
    agent.telemetry.session_id = "s2"
    agent.run("question")
    saved = store.load("s2")
    roles = [m.role for m in saved]
    assert "user" in roles
    assert "assistant" in roles


def test_agent_does_not_double_load_when_context_persists(agent_builder):
    store = InMemoryMemoryStore()
    store.save("s1", [Message(role="user", content="seed")])
    provider = FakeProvider(
        [
            text_response("first"),
            text_response("second"),
        ]
    )
    agent = agent_builder(provider, memory=store)
    agent.telemetry.session_id = "s1"
    agent.run("a")
    agent.run("b")
    # The seed was loaded once at the start of the session and remains present;
    # it must not be loaded again (which would duplicate it inside any call).
    assert len(provider.calls) == 2
    for call in provider.calls:
        seed_count_in_call = sum(1 for m in call if m.content == "seed")
        assert seed_count_in_call == 1
