"""Loop tests (design §15): single-turn, tool round-trip, parallel calls,
max-iteration guard, and the §8 tool-result pairing contract."""

from __future__ import annotations

from agentkernel.config import Config
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult
from tests.fakes import FakeProvider, text_response, tool_call_response

_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def _echo_tool(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Echo the given value back.",
        parameters=_ECHO_SCHEMA,
        handler=lambda args: ToolResult("", f"echo:{args['value']}"),
    )


def test_single_turn_returns_text(agent_builder):
    provider = FakeProvider([text_response("hello there")])
    agent = agent_builder(provider)
    assert agent.run("hi") == "hello there"
    assert len(provider.calls) == 1


def test_tool_call_round_trip(agent_builder):
    registry = ToolRegistry()
    registry.register(_echo_tool())
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "echo", {"value": "x"})),
            text_response("done"),
        ]
    )
    agent = agent_builder(provider, registry)
    assert agent.run("go") == "done"

    # The second request must contain the tool result paired to call id "c1".
    second_request = provider.calls[1]
    tool_msg = second_request[-1]
    assert tool_msg.role == "tool"
    assert [r.call_id for r in tool_msg.tool_results] == ["c1"]
    assert tool_msg.tool_results[0].content == "echo:x"


def test_parallel_tool_calls_all_answered(agent_builder):
    registry = ToolRegistry()
    registry.register(_echo_tool("a"))
    registry.register(_echo_tool("b"))
    provider = FakeProvider(
        [
            tool_call_response(
                ToolCall("c1", "a", {"value": "1"}),
                ToolCall("c2", "b", {"value": "2"}),
            ),
            text_response("ok"),
        ]
    )
    agent = agent_builder(provider, registry)
    assert agent.run("go") == "ok"

    tool_msg = provider.calls[1][-1]
    # Exactly N results, matching ids, in order (design §8.1/§8.2).
    assert [r.call_id for r in tool_msg.tool_results] == ["c1", "c2"]


def test_max_iteration_guard(agent_builder):
    registry = ToolRegistry()
    registry.register(_echo_tool())
    # Always asks for a tool, never finishes -> guard must fire.
    looping = [
        tool_call_response(ToolCall(f"c{i}", "echo", {"value": "x"}))
        for i in range(10)
    ]
    provider = FakeProvider(looping)
    agent = agent_builder(provider, registry, config=Config(max_iterations=3))
    result = agent.run("go")
    assert "max iterations" in result
    assert len(provider.calls) == 3


def test_assistant_then_tool_message_ordering(agent_builder):
    """No message may be interleaved between an assistant tool call and its
    results (design §8.2)."""
    registry = ToolRegistry()
    registry.register(_echo_tool())
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "echo", {"value": "x"})),
            text_response("done"),
        ]
    )
    agent = agent_builder(provider, registry)
    agent.run("go")

    convo = provider.calls[1]
    # ... user, assistant(tool_calls), tool(results)
    assert convo[-2].role == "assistant" and convo[-2].tool_calls
    assert convo[-1].role == "tool" and convo[-1].tool_results
