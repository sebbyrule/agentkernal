"""Sub-agent spawn tests (design §13). A parent and its children share one
FakeProvider; because the loop executes sequentially, scripted responses are
consumed parent-turn, child-turn(s), parent-turn — proving the seam end to end."""

from __future__ import annotations

from agentkernel.approval import AutoApprover
from agentkernel.config import Config
from agentkernel.subagent import make_spawn_tool
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult

from tests.fakes import FakeProvider, text_response, tool_call_response


def _spawn(provider, *, base_specs=None, max_depth=2) -> ToolSpec:
    return make_spawn_tool(
        provider=provider,
        base_specs=base_specs or [],
        approver=AutoApprover("auto_allow"),
        config=Config(),
        max_depth=max_depth,
    )


def test_spawn_runs_subagent_and_returns_answer(agent_builder):
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "spawn", {"task": "do subtask"})),
            text_response("subtask complete"),  # child's final answer
            text_response("all done"),  # parent's final answer
        ]
    )
    registry = ToolRegistry()
    registry.register(_spawn(provider))
    agent = agent_builder(provider, registry)

    assert agent.run("delegate this") == "all done"
    # calls: [0] parent, [1] child, [2] parent-with-tool-result
    result = provider.calls[2][-1].tool_results[0]
    assert result.call_id == "c1" and "subtask complete" in result.content


def test_spawn_depth_limit_blocks_nested_spawn(agent_builder):
    # max_depth=1 -> the child gets no spawn tool, so its spawn attempt is an
    # unknown-tool error it must recover from.
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "spawn", {"task": "A"})),  # parent
            tool_call_response(ToolCall("c2", "spawn", {"task": "B"})),  # child tries
            text_response("child recovered"),  # child final
            text_response("parent done"),  # parent final
        ]
    )
    registry = ToolRegistry()
    registry.register(_spawn(provider, max_depth=1))
    agent = agent_builder(provider, registry)

    assert agent.run("go") == "parent done"
    child_error = provider.calls[2][-1].tool_results[0]  # child's req after spawn attempt
    assert child_error.is_error and "Unknown tool" in child_error.content


def test_spawn_tool_filter_restricts_child_tools(agent_builder):
    echo = ToolSpec(
        "echo",
        "Echo a value.",
        {
            "type": "object",
            "properties": {"v": {"type": "string"}},
            "required": ["v"],
            "additionalProperties": False,
        },
        lambda a: ToolResult("", f"echo:{a['v']}"),
    )
    provider = FakeProvider(
        [
            tool_call_response(
                ToolCall("c1", "spawn", {"task": "use echo", "tools": ["echo"]})
            ),
            tool_call_response(ToolCall("c2", "echo", {"v": "hi"})),  # child uses echo
            text_response("child used echo"),  # child final
            text_response("parent done"),  # parent final
        ]
    )
    registry = ToolRegistry()
    registry.register(_spawn(provider, base_specs=[echo]))
    agent = agent_builder(provider, registry)

    assert agent.run("go") == "parent done"
    child_echo = provider.calls[2][-1].tool_results[0]  # child's echo result
    assert "echo:hi" in child_echo.content
