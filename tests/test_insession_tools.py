"""Tests for the in-session tools (design §18.4): todo and clarify."""

from __future__ import annotations

from agentkernel.tools.builtin.clarify import clarify_tool
from agentkernel.tools.builtin.todo import TodoList, todo_tool

# --- todo ---------------------------------------------------------------------

def _todo():
    return todo_tool(TodoList()).handler


def test_todo_add_and_list():
    todo = _todo()
    r = todo({"action": "add", "text": "write the parser"})
    assert not r.is_error and "1. write the parser" in r.content
    assert "[ ]" in r.content  # pending mark


def test_todo_start_and_complete_progress():
    todo = _todo()
    todo({"action": "add", "text": "step one"})
    todo({"action": "add", "text": "step two"})
    todo({"action": "start", "id": 1})
    r = todo({"action": "complete", "id": 1})
    assert "[x] 1. step one" in r.content
    assert "1/2 done" in r.content


def test_todo_add_requires_text():
    todo = _todo()
    assert todo({"action": "add", "text": "  "}).is_error


def test_todo_status_unknown_id_is_error():
    todo = _todo()
    todo({"action": "add", "text": "only one"})
    assert todo({"action": "complete", "id": 99}).is_error


def test_todo_clear():
    todo = _todo()
    todo({"action": "add", "text": "x"})
    assert "Cleared" in todo({"action": "clear"}).content
    assert "empty" in todo({"action": "list"}).content


# --- clarify ------------------------------------------------------------------

def test_clarify_returns_user_answer():
    tool = clarify_tool(input_fn=lambda _p: "use postgres", output_fn=lambda _m: None)
    r = tool.handler({"question": "Which database?"})
    assert not r.is_error and "use postgres" in r.content


def test_clarify_empty_answer_proceeds():
    tool = clarify_tool(input_fn=lambda _p: "  ", output_fn=lambda _m: None)
    r = tool.handler({"question": "Anything?"})
    assert "best judgment" in r.content


def test_clarify_no_stdin_degrades_gracefully():
    def _eof(_prompt):
        raise EOFError

    tool = clarify_tool(input_fn=_eof, output_fn=lambda _m: None)
    r = tool.handler({"question": "Which one?"})
    assert not r.is_error and "No user is available" in r.content


def test_clarify_requires_question():
    tool = clarify_tool(input_fn=lambda _p: "x", output_fn=lambda _m: None)
    assert tool.handler({"question": "  "}).is_error


# --- wiring -------------------------------------------------------------------

def test_flags_register_tools():
    from agentkernel.cli import build_runtime
    from agentkernel.config import Config

    agent, telemetry, clients = build_runtime(Config(enable_todo=True, enable_clarify=True))
    try:
        names = {s.name for s in agent.registry.specs()}
        assert "todo" in names and "clarify" in names
    finally:
        telemetry.close()
        for c in clients:
            c.close()
