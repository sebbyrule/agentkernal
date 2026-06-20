"""Checkpoint / rollback tests (design §18.1): the Checkpointer, its use by the
file tools, and the rollback tool."""

from __future__ import annotations

from agentkernel.checkpoint import Checkpointer
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import file_tools
from agentkernel.tools.builtin.checkpoint_tool import rollback_tool
from agentkernel.types import ToolCall

# --- the Checkpointer ---------------------------------------------------------

def test_rollback_restores_modified_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("original", encoding="utf-8")
    cp = Checkpointer()
    cp.record(f)
    f.write_text("changed", encoding="utf-8")
    assert cp.rollback() == 1
    assert f.read_text(encoding="utf-8") == "original"


def test_rollback_deletes_created_file(tmp_path):
    f = tmp_path / "new.txt"
    cp = Checkpointer()
    cp.record(f)  # did not exist at checkpoint time
    f.write_text("created", encoding="utf-8")
    assert cp.rollback() == 1
    assert not f.exists()


def test_record_keeps_earliest_state(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v0", encoding="utf-8")
    cp = Checkpointer()
    cp.record(f)
    f.write_text("v1", encoding="utf-8")
    cp.record(f)  # second record must NOT overwrite the captured v0
    f.write_text("v2", encoding="utf-8")
    cp.rollback()
    assert f.read_text(encoding="utf-8") == "v0"


def test_pending_and_clear(tmp_path):
    cp = Checkpointer()
    assert cp.pending() == 0
    cp.record(tmp_path / "x.txt")
    assert cp.pending() == 1
    cp.rollback()
    assert cp.pending() == 0


# --- file tools record before mutating ----------------------------------------

def _registry(tmp_path, cp):
    reg = ToolRegistry()
    for spec in file_tools(str(tmp_path), checkpointer=cp):
        reg.register(spec)
    reg.register(rollback_tool(cp))
    return reg


def test_write_then_rollback_via_tools(tmp_path):
    (tmp_path / "keep.txt").write_text("before", encoding="utf-8")
    cp = Checkpointer()
    reg = _registry(tmp_path, cp)

    reg.execute(ToolCall("c1", "write_file", {"path": "keep.txt", "content": "after"}))
    reg.execute(ToolCall("c2", "write_file", {"path": "fresh.txt", "content": "new"}))
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "after"
    assert (tmp_path / "fresh.txt").is_file()

    result = reg.execute(ToolCall("c3", "rollback", {}))
    assert not result.is_error and "Rolled back 2" in result.content
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "before"
    assert not (tmp_path / "fresh.txt").exists()


def test_edit_then_rollback_via_tools(tmp_path):
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    cp = Checkpointer()
    reg = _registry(tmp_path, cp)
    reg.execute(ToolCall("c1", "edit_file", {"path": "code.py", "old": "1", "new": "2"}))
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 2\n"
    reg.execute(ToolCall("c2", "rollback", {}))
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 1\n"


def test_rollback_with_nothing_to_undo(tmp_path):
    cp = Checkpointer()
    reg = _registry(tmp_path, cp)
    result = reg.execute(ToolCall("c1", "rollback", {}))
    assert not result.is_error and "Nothing to roll back" in result.content
