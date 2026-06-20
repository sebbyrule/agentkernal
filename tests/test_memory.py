"""Memory store tests (Phase 3, design \u00a713).

Stores are tested against canonical ``Message`` objects; the kernel never sees
store-specific shapes.
"""

from __future__ import annotations

from agentkernel.memory import (
    FileMemoryStore,
    InMemoryMemoryStore,
    MemoryNotes,
    _normalize_token,
)
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


def test_in_memory_store_delete_and_list_sessions():
    store = InMemoryMemoryStore()
    store.save("a", [Message(role="user", content="a")])
    store.save("b", [Message(role="user", content="b")])
    assert store.list_sessions() == ["a", "b"]
    store.delete("a")
    assert store.list_sessions() == ["b"]
    assert store.load("a") == []


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


def test_file_memory_store_delete_and_list_sessions(tmp_path):
    store = FileMemoryStore(tmp_path / "memory")
    store.save("session-1", [Message(role="user", content="one")])
    store.save("session-2", [Message(role="user", content="two")])
    assert store.list_sessions() == ["session-1", "session-2"]
    store.delete("session-1")
    assert store.list_sessions() == ["session-2"]
    assert store.load("session-1") == []


def test_file_memory_store_excludes_notes_from_session_list(tmp_path):
    store = FileMemoryStore(tmp_path / "memory")
    notes = MemoryNotes(tmp_path / "memory" / "notes.jsonl")
    notes.add("a fact")
    store.save("session-1", [Message(role="user", content="one")])
    assert store.list_sessions() == ["session-1"]


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


# --- Model-controlled notes -------------------------------------------------


def test_memory_notes_assign_incrementing_ids(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    n1 = notes.add("first")
    n2 = notes.add("second")
    assert n1.note_id == 1
    assert n2.note_id == 2


def test_memory_notes_recall_recent_by_default(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("oldest")
    notes.add("middle")
    notes.add("newest")
    recent = notes.recent(2)
    assert [n.text for n in recent] == ["newest", "middle"]


def test_memory_notes_search_finds_related_facts(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("User prefers Python for all scripts")
    notes.add("The project uses pytest for tests")
    notes.add("Lunch was sandwiches")
    results = notes.search("Python scripts")
    assert results[0].text == "User prefers Python for all scripts"


def test_memory_notes_search_stems_plural_and_verb_forms(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("Users prefer tables")
    results = notes.search("preferring table")
    assert len(results) == 1


def test_memory_note_forget_by_id(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("keep me")
    n2 = notes.add("remove me")
    removed = notes.forget(note_id=n2.note_id)
    assert len(removed) == 1
    assert removed[0].text == "remove me"
    assert [n.text for n in notes.all()] == ["keep me"]


def test_memory_note_forget_by_prefix(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("keep me")
    notes.add("user likes dark mode")
    notes.add("user likes markdown")
    removed = notes.forget(text_prefix="user likes")
    assert len(removed) == 2
    assert [n.text for n in notes.all()] == ["keep me"]


def test_memory_note_forget_persists_after_rewrite(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("a")
    notes.add("b")
    notes.forget(note_id=1)
    notes2 = MemoryNotes(notes.path)
    assert [n.text for n in notes2.all()] == ["b"]
    assert notes2._next_id == 3


def test_memory_note_update_rewrites_file(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    n = notes.add("old text", tags=["a"])
    updated = notes.update(n.note_id, "new text", tags=["b"])
    assert updated is not None
    assert updated.text == "new text"
    assert updated.tags == ["b"]
    notes2 = MemoryNotes(notes.path)
    assert notes2.all()[0].text == "new text"


def test_memory_note_update_returns_none_for_missing_id(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    assert notes.update(99, "x") is None


def test_memory_notes_export_to_markdown(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    n = notes.add("fact", tags=["t1", "t2"])
    path = notes.export(tmp_path / "out.md")
    text = path.read_text(encoding="utf-8")
    assert "# Memory Notes" in text
    assert "fact" in text
    assert "t1" in text
    assert f"id={n.note_id}" in text


def test_memory_tool_forget_deletes_note(tmp_path):
    from agentkernel.memory import make_memory_tools

    notes = MemoryNotes(tmp_path / "notes.jsonl")
    tools = {t.name: t for t in make_memory_tools(notes)}
    note = notes.add("to delete")
    result = tools["forget"].handler({"note_id": note.note_id})
    assert "Forgot 1" in result.content
    assert len(notes.all()) == 0


def test_memory_tool_update_memory_changes_note(tmp_path):
    from agentkernel.memory import make_memory_tools

    notes = MemoryNotes(tmp_path / "notes.jsonl")
    tools = {t.name: t for t in make_memory_tools(notes)}
    note = notes.add("old")
    result = tools["update_memory"].handler({"note_id": note.note_id, "text": "new"})
    assert "Updated" in result.content
    assert notes.all()[0].text == "new"


def test_normalize_token_stems_common_suffixes():
    assert _normalize_token("running") == "runn"
    assert _normalize_token("tables") == "table"
    assert _normalize_token("files") == "file"
    assert _normalize_token("has") == "has"  # short / special case
    assert _normalize_token("flying") == "flie"


def test_memory_note_access_count_increments_on_recall(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    note = notes.add("important fact")
    assert note.access_count == 0
    notes.search("important")
    assert note.access_count == 1
    notes.recent(5)
    assert note.access_count == 2


def test_memory_stats_tool_reports_summary(tmp_path):
    from agentkernel.memory import make_memory_tools

    notes = MemoryNotes(tmp_path / "notes.jsonl")
    tools = {t.name: t for t in make_memory_tools(notes)}
    notes.add("first", tags=["a"])
    notes.add("second")
    notes.search("first")
    result = tools["memory_stats"].handler({})
    assert "Total notes: 2" in result.content
    assert "first" in result.content
    assert "(1)" in result.content


def test_session_memory_tools_list_and_delete(tmp_path):
    from agentkernel.memory import make_memory_tools

    store = InMemoryMemoryStore()
    store.save("session-a", [Message(role="user", content="hello")])
    store.save("session-b", [Message(role="user", content="hi")])
    tools = {t.name: t for t in make_memory_tools(MemoryNotes(tmp_path / "notes.jsonl"), store=store)}
    list_result = tools["list_sessions"].handler({})
    assert "session-a" in list_result.content
    assert "session-b" in list_result.content
    tools["delete_session"].handler({"session_id": "session-a"})
    assert store.list_sessions() == ["session-b"]


def test_session_tools_not_provided_without_store(tmp_path):
    from agentkernel.memory import make_memory_tools

    tools = {t.name: t for t in make_memory_tools(MemoryNotes(tmp_path / "notes.jsonl"))}
    assert "list_sessions" not in tools
    assert "delete_session" not in tools


# --- P2: sparse TF-IDF semantic recall ---------------------------------------


def test_tfidf_ranks_rare_terms_higher(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("Python is a programming language used by many teams")
    notes.add("Python snakes are reptiles found in warm climates")
    notes.add("Python and Java are both popular languages")
    results = notes.search("snake reptile")
    assert results[0].text == "Python snakes are reptiles found in warm climates"


def test_deduplicate_memory_merges_identical_notes(tmp_path):
    notes = MemoryNotes(tmp_path / "notes.jsonl")
    notes.add("fact", tags=["a"])
    notes.add("fact", tags=["b"])
    notes.add("other")
    removed = notes.deduplicate()
    assert removed == 1
    texts = [n.text for n in notes.all()]
    assert texts == ["fact", "other"]
    assert sorted(notes.all()[0].tags) == ["a", "b"]


def test_deduplicate_memory_tool_reports_count(tmp_path):
    from agentkernel.memory import make_memory_tools

    notes = MemoryNotes(tmp_path / "notes.jsonl")
    tools = {t.name: t for t in make_memory_tools(notes)}
    notes.add("duplicate")
    notes.add("duplicate")
    result = tools["deduplicate_memory"].handler({})
    assert "1" in result.content
    assert len(notes.all()) == 1


# --- P3: SQLite-backed session memory ----------------------------------------


def test_sqlite_memory_store_roundtrip(tmp_path):
    from agentkernel.memory import SqliteMemoryStore

    db = tmp_path / "memory.db"
    store = SqliteMemoryStore(db)
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    store.save("s1", messages)
    assert store.load("s1") == messages
    assert store.list_sessions() == ["s1"]
    store.delete("s1")
    assert store.load("s1") == []
    assert store.list_sessions() == []
    store.close()


def test_sqlite_memory_store_preserves_tool_data(tmp_path):
    from agentkernel.memory import SqliteMemoryStore

    store = SqliteMemoryStore(tmp_path / "memory.db")
    msg = Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="recall", arguments={"query": "x"})],
    )
    store.save("s1", [msg])
    loaded = store.load("s1")
    assert len(loaded) == 1
    assert loaded[0].tool_calls[0].name == "recall"
    store.close()


def test_sqlite_memory_store_search_sessions(tmp_path):
    from agentkernel.memory import SqliteMemoryStore

    store = SqliteMemoryStore(tmp_path / "memory.db")
    store.save("alpha", [Message(role="user", content="pineapple on pizza")])
    store.save("beta", [Message(role="user", content="cheese on pizza")])
    results = store.search_sessions("pineapple")
    assert results == ["alpha"]
    results = store.search_sessions("pizza")
    assert sorted(results) == ["alpha", "beta"]
    store.close()
