"""Telemetry tests (design §12): record schema, redaction default vs verbose,
cost computation, and that the loop appends a record per turn."""

from __future__ import annotations

import json

from agentkernel.context import CompactionEvent
from agentkernel.telemetry import (
    DEFAULT_PRICES,
    JsonlTelemetry,
    ToolOutcome,
    estimate_cost,
)
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import CompletionResponse, Message, ToolCall, ToolResult, Usage
from tests.fakes import FakeProvider, text_response, tool_call_response


def _resp(stop="tool_use") -> CompletionResponse:
    return CompletionResponse(
        message=Message(role="assistant"),
        usage=Usage(input_tokens=1000, output_tokens=200, cache_read_tokens=800),
        stop_reason=stop,
    )


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_has_full_schema(tmp_path):
    tel = JsonlTelemetry(str(tmp_path), "claude-sonnet-4-6", session_id="s1")
    tel.record_turn(
        0,
        _resp(),
        tool_outcomes=[ToolOutcome("bash", {"command": "ls"}, True, False)],
        compaction=CompactionEvent(turns_collapsed=3, tokens_before=100, tokens_after=40),
    )
    tel.close()
    (rec,) = _read(tmp_path / "s1.jsonl")
    assert rec["session_id"] == "s1"
    assert rec["iteration"] == 0
    assert rec["input_tokens"] == 1000 and rec["cache_read_tokens"] == 800
    assert rec["stop_reason"] == "tool_use"
    assert rec["compaction"] == {
        "turns_collapsed": 3,
        "tokens_before": 100,
        "tokens_after": 40,
    }
    assert rec["tool_calls"][0]["name"] == "bash"
    assert rec["tool_calls"][0]["approved"] is True


def test_redaction_default_hides_raw_args(tmp_path):
    tel = JsonlTelemetry(str(tmp_path), "gpt-4o", session_id="s2")
    tel.record_turn(
        0, _resp(),
        tool_outcomes=[ToolOutcome(
            "write_file", {"path": "secret.txt", "content": "sk-123"}, True, False
        )],
    )
    tel.close()
    (rec,) = _read(tmp_path / "s2.jsonl")
    entry = rec["tool_calls"][0]
    assert "arguments" not in entry  # raw args never written by default
    assert "sk-123" not in json.dumps(rec)  # secret value absent everywhere
    assert "args_sha256" in entry and entry["args_len"] > 0


def test_verbose_includes_raw_args(tmp_path):
    tel = JsonlTelemetry(str(tmp_path), "gpt-4o", session_id="s3", verbose=True)
    tel.record_turn(0, _resp(), tool_outcomes=[ToolOutcome("bash", {"command": "ls"}, True, False)])
    tel.close()
    (rec,) = _read(tmp_path / "s3.jsonl")
    assert rec["tool_calls"][0]["arguments"] == {"command": "ls"}


def test_cost_for_known_and_unknown_model():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = estimate_cost("claude-sonnet-4-6", usage, DEFAULT_PRICES)
    assert cost == 3.0 + 15.0  # input + output per 1M tokens
    assert estimate_cost("some-unknown-model", usage, DEFAULT_PRICES) is None


def test_cost_logged_null_for_unknown_model(tmp_path):
    tel = JsonlTelemetry(str(tmp_path), "mystery-model", session_id="s4")
    tel.record_turn(0, _resp())
    tel.close()
    (rec,) = _read(tmp_path / "s4.jsonl")
    assert rec["estimated_cost_usd"] is None
    assert rec["input_tokens"] == 1000  # tokens still recorded


def test_loop_appends_one_record_per_turn(tmp_path, agent_builder):
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            "echo",
            "d",
            {
                "type": "object",
                "properties": {"v": {"type": "string"}},
                "required": ["v"],
                "additionalProperties": False,
            },
            lambda a: ToolResult("", "ok"),
        )
    )
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "echo", {"v": "x"})),
            text_response("done"),
        ]
    )
    tel = JsonlTelemetry(str(tmp_path), "claude-sonnet-4-6", session_id="run")
    # Build an agent that uses the real JSONL telemetry.
    agent = agent_builder(provider, reg)
    agent.telemetry = tel
    agent.run("go")
    tel.close()

    records = _read(tmp_path / "run.jsonl")
    assert len(records) == 2  # one per provider turn (tool turn + final answer)
    assert records[0]["tool_calls"][0]["name"] == "echo"
    assert records[1]["tool_calls"] == []  # final answer turn has no tools
