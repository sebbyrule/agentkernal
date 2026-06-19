"""Sandbox + bash tests (design §10.3, §6.3): commands are confined to the
working dir, the environment is scrubbed of secrets, timeouts are reported, and
bash is gated through the approver. Uses portable shell commands only."""

from __future__ import annotations

import os

from agentkernel.approval import AutoApprover, LocalSandbox
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import bash_tool
from agentkernel.types import ToolCall, ToolResult

from tests.fakes import FakeProvider, text_response, tool_call_response


def test_local_sandbox_runs_in_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("hi")
    cmd = "dir /b" if os.name == "nt" else "ls"
    code, out, _err = LocalSandbox().run(cmd, cwd=str(tmp_path), timeout=10)
    assert code == 0 and "marker.txt" in out


def test_local_sandbox_reports_nonzero_exit(tmp_path):
    cmd = "exit 3"
    code, _out, _err = LocalSandbox().run(cmd, cwd=str(tmp_path), timeout=10)
    assert code == 3


def test_local_sandbox_scrubs_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("MY_SECRET_TOKEN", "nope")
    monkeypatch.setenv("HARMLESS_VAR", "ok")
    if os.name == "nt":
        cmd = "echo %ANTHROPIC_API_KEY%-%HARMLESS_VAR%"
    else:
        cmd = "echo $ANTHROPIC_API_KEY-$HARMLESS_VAR"
    _code, out, _err = LocalSandbox().run(cmd, cwd=str(tmp_path), timeout=10)
    assert "sk-should-not-leak" not in out
    assert "ok" in out  # non-secret vars survive


def test_local_sandbox_timeout(tmp_path):
    # A command that blocks longer than the timeout, on either platform.
    cmd = "ping -n 10 127.0.0.1 >nul" if os.name == "nt" else "sleep 5"
    code, _out, err = LocalSandbox().run(cmd, cwd=str(tmp_path), timeout=1)
    assert code == 124 and "timed out" in err


def test_bash_tool_output_and_error_flag(tmp_path):
    tool = bash_tool(LocalSandbox(), str(tmp_path), timeout=10)
    ok = tool.handler({"command": "echo hello"})
    assert not ok.is_error and "hello" in ok.content and ok.data["exit_code"] == 0
    bad = tool.handler({"command": "exit 7"})
    assert bad.is_error and bad.data["exit_code"] == 7


def test_bash_is_gated_and_denial_blocks_execution(tmp_path, agent_builder):
    reg = ToolRegistry()
    reg.register(bash_tool(LocalSandbox(), str(tmp_path), timeout=10))
    sentinel = tmp_path / "created.txt"
    provider = FakeProvider(
        [
            tool_call_response(
                ToolCall("c1", "bash", {"command": f"echo x > {sentinel.name}"})
            ),
            text_response("understood"),
        ]
    )
    agent = agent_builder(provider, reg, approver=AutoApprover(ask_default=False))
    assert agent.run("write a file via shell") == "understood"
    assert not sentinel.exists()  # the gate prevented the side effect
    result = provider.calls[1][-1].tool_results[0]
    assert result.is_error and "Denied" in result.content


def test_bash_runs_when_approved(tmp_path, agent_builder):
    reg = ToolRegistry()
    reg.register(bash_tool(LocalSandbox(), str(tmp_path), timeout=10))
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "bash", {"command": "echo hi"})),
            text_response("done"),
        ]
    )
    agent = agent_builder(provider, reg, approver=AutoApprover(ask_default=True))
    assert agent.run("run it") == "done"
    result = provider.calls[1][-1].tool_results[0]
    assert not result.is_error and "hi" in result.content
