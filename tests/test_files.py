"""Builtin file-tool tests (design §6.3): read/write/list round-trips, path
confinement, truncation, and the approval gate on mutating tools."""

from __future__ import annotations

from agentkernel.approval import AutoApprover
from agentkernel.context.truncate import truncate_text
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import file_tools
from agentkernel.types import ToolCall

from tests.fakes import FakeProvider, text_response, tool_call_response


def _registry(tmp_path) -> ToolRegistry:
    reg = ToolRegistry()
    for spec in file_tools(str(tmp_path)):
        reg.register(spec)
    return reg


def test_write_then_read(tmp_path):
    reg = _registry(tmp_path)
    w = reg.execute(ToolCall("c1", "write_file", {"path": "a.txt", "content": "hi"}))
    assert not w.is_error
    r = reg.execute(ToolCall("c2", "read_file", {"path": "a.txt"}))
    assert r.content == "hi" and not r.is_error


def test_read_missing_file_is_error(tmp_path):
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c1", "read_file", {"path": "nope.txt"}))
    assert r.is_error and "Not a file" in r.content


def test_path_escape_rejected(tmp_path):
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c1", "read_file", {"path": "../secret.txt"}))
    assert r.is_error and "escapes working directory" in r.content


def test_list_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.txt").write_text("x")
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c1", "list_dir", {"path": "."}))
    assert "sub/" in r.content and "f.txt" in r.content


def test_truncation_marks_removed_content():
    out = truncate_text("x" * 10_000, max_tokens=100)
    assert "truncated" in out and len(out) < 10_000


def test_write_file_is_gated_and_denial_continues(tmp_path, agent_builder):
    """write_file is mutating + requires_approval; a denial yields an error
    result and the loop continues (design §8.3, §10)."""
    reg = _registry(tmp_path)
    provider = FakeProvider(
        [
            tool_call_response(
                ToolCall("c1", "write_file", {"path": "x.txt", "content": "data"})
            ),
            text_response("understood"),
        ]
    )
    agent = agent_builder(provider, reg, approver=AutoApprover(ask_default=False))
    assert agent.run("write it") == "understood"
    assert not (tmp_path / "x.txt").exists()  # denial prevented the write
    result = provider.calls[1][-1].tool_results[0]
    assert result.is_error and "Denied" in result.content
