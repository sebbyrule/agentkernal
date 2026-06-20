"""Kanban board tests (design §18.3): the Board store, the kanban tool, and the
run_kanban CLI helper."""

from __future__ import annotations

from agentkernel.cli import run_kanban
from agentkernel.config import Config
from agentkernel.kanban import Board
from agentkernel.tools.builtin.kanban_tool import kanban_tool

# --- Board --------------------------------------------------------------------

def _board(tmp_path):
    return Board(tmp_path / "kanban.json")


def test_add_and_list(tmp_path):
    b = _board(tmp_path)
    t = b.add("write the parser")
    assert t.status == "todo" and t.title == "write the parser"
    assert len(b.list()) == 1
    assert b.list(status="done") == []


def test_claim_complete_block_comment(tmp_path):
    b = _board(tmp_path)
    t = b.add("task one")
    assert b.claim(t.id, "worker-1").status == "in_progress"
    assert b.get(t.id).assignee == "worker-1"
    assert b.complete(t.id).status == "done"
    b.comment(t.id, "looks good")
    assert "looks good" in b.get(t.id).notes
    t2 = b.add("task two")
    blocked = b.block(t2.id, "needs API key")
    assert blocked.status == "blocked" and "blocked: needs API key" in blocked.notes


def test_next_todo_pulls_first_unclaimed(tmp_path):
    b = _board(tmp_path)
    a = b.add("first")
    b.add("second")
    b.claim(a.id, "w")  # first is now in_progress
    assert b.next_todo().title == "second"


def test_operations_on_missing_id_return_none(tmp_path):
    b = _board(tmp_path)
    assert b.complete("ghost") is None
    assert b.claim("ghost", "w") is None


# --- the kanban tool ----------------------------------------------------------

def test_tool_add_list_next_complete(tmp_path):
    tool = kanban_tool(_board(tmp_path), worker="alice").handler
    assert "Added" in tool({"action": "add", "title": "do the thing"}).content
    assert "do the thing" in tool({"action": "list"}).content
    claimed = tool({"action": "next"})
    assert "Claimed" in claimed.content
    tid = claimed.content.split()[1].rstrip(":")
    done = tool({"action": "complete", "id": tid})
    assert "[x]" in done.content


def test_tool_errors(tmp_path):
    tool = kanban_tool(_board(tmp_path)).handler
    assert tool({"action": "add", "title": "  "}).is_error
    assert tool({"action": "complete", "id": "nope"}).is_error
    assert tool({"action": "comment", "id": "x"}).is_error  # missing text


def test_tool_next_when_empty(tmp_path):
    tool = kanban_tool(_board(tmp_path)).handler
    assert "No unclaimed tasks" in tool({"action": "next"}).content


# --- CLI ----------------------------------------------------------------------

def test_run_kanban_add_list_complete(tmp_path):
    cfg = Config(kanban_path=str(tmp_path / "kanban.json"))
    out: list[str] = []
    run_kanban(cfg, "add", ["ship", "the", "release"], output_fn=out.append)
    assert "added" in out[0]
    out.clear()
    run_kanban(cfg, "list", [], output_fn=out.append)
    assert "ship the release" in out[0]
    tid = Board(cfg.kanban_path).list()[0].id
    out.clear()
    assert run_kanban(cfg, "complete", [tid], output_fn=out.append) == 0
    assert "completed" in out[0]
    assert Board(cfg.kanban_path).list()[0].status == "done"


def test_run_kanban_remove(tmp_path):
    cfg = Config(kanban_path=str(tmp_path / "kanban.json"))
    board = Board(cfg.kanban_path)
    t = board.add("temp")
    assert run_kanban(cfg, "remove", [t.id], output_fn=lambda _m: None) == 0
    assert board.list() == []


def test_run_kanban_empty(tmp_path):
    cfg = Config(kanban_path=str(tmp_path / "kanban.json"))
    out: list[str] = []
    run_kanban(cfg, "list", [], output_fn=out.append)
    assert "empty" in out[0]
