"""The cacheable prefix (system + tools) must be assembled once per run and
passed unchanged every turn (design §9.3, AGENT.md rule 3)."""

from __future__ import annotations

from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult
from tests.fakes import FakeProvider, text_response, tool_call_response

_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def test_tools_object_is_identical_across_turns(agent_builder):
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "d", _SCHEMA, lambda a: ToolResult("", "ok")))
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "echo", {"value": "x"})),
            tool_call_response(ToolCall("c2", "echo", {"value": "y"})),
            text_response("done"),
        ]
    )
    agent = agent_builder(provider, reg)
    agent.run("go")

    assert len(provider.tool_args) == 3
    first = provider.tool_args[0]
    # Same object every turn — not just equal — so the wire prefix is byte-stable.
    assert all(t is first for t in provider.tool_args)
    assert [s.name for s in first] == ["echo"]
