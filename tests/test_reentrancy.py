"""Re-entrancy (design §7, AGENT.md rule 7): `run` must be safe to call from
inside a tool handler — a tool spawning a sub-agent. This works only because no
module-level mutable state exists and all collaborators are injected."""

from __future__ import annotations

from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult

from tests.fakes import FakeProvider, text_response, tool_call_response

_TASK_SCHEMA = {
    "type": "object",
    "properties": {"task": {"type": "string"}},
    "required": ["task"],
    "additionalProperties": False,
}


def test_tool_handler_spawns_subagent(agent_builder):
    # The spawn tool builds a CHILD agent with its own provider and context
    # (sharing nothing mutable with the parent) and runs it to completion.
    def spawn(args):
        child_provider = FakeProvider([text_response("child says hi")])
        child = agent_builder(child_provider)  # fresh context, fresh everything
        return ToolResult("", child.run(args["task"]))

    registry = ToolRegistry()
    registry.register(ToolSpec("spawn", "Spawn a sub-agent for a task.", _TASK_SCHEMA, spawn))

    parent_provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "spawn", {"task": "do x"})),
            text_response("parent done"),
        ]
    )
    parent = agent_builder(parent_provider, registry)

    assert parent.run("go") == "parent done"
    # The child's answer came back to the parent as the tool result, and the
    # parent's own loop continued normally afterwards.
    tool_msg = parent_provider.calls[1][-1]
    assert tool_msg.tool_results[0].content == "child says hi"
    assert tool_msg.tool_results[0].call_id == "c1"
