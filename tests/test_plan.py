"""Tests for plan mode batch approval."""

from __future__ import annotations

from agentkernel.approval import AutoApprover, CliApprover
from agentkernel.config import Config
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult
from tests.fakes import FakeProvider, text_response, tool_call_response


def _gated_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="gated",
            description="A tool that requires approval.",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: ToolResult("", "done"),
            mutates=True,
        )
    )
    return registry


def test_plan_mode_denied_stops_run(agent_builder):
    registry = _gated_registry()
    provider = FakeProvider([
        tool_call_response(ToolCall(id="c1", name="gated", arguments={})),
        text_response("final"),
    ])
    agent = agent_builder(
        provider,
        registry=registry,
        config=Config(plan_mode=True),
        approver=AutoApprover(ask_default=False),
    )
    result = agent.run("do it")
    assert "Plan denied" in result
    assert len(provider.calls) == 1  # no tool executed, no second completion


def test_plan_mode_approved_executes_plan(agent_builder):
    registry = _gated_registry()
    provider = FakeProvider([
        tool_call_response(ToolCall(id="c1", name="gated", arguments={})),
        text_response("done"),
    ])
    agent = agent_builder(
        provider,
        registry=registry,
        config=Config(plan_mode=True),
        approver=AutoApprover(ask_default=True),
    )
    result = agent.run("do it")
    assert result == "done"
    assert len(provider.calls) == 2


def test_non_plan_mode_uses_per_call_approval(agent_builder):
    registry = _gated_registry()
    provider = FakeProvider([
        tool_call_response(ToolCall(id="c1", name="gated", arguments={})),
        text_response("done"),
    ])
    # With no plan mode, a denied call returns a ToolResult error and the loop
    # may continue to the next scripted provider response.
    agent = agent_builder(
        provider,
        registry=registry,
        config=Config(plan_mode=False),
        approver=AutoApprover(ask_default=False),
    )
    result = agent.run("do it")
    assert result == "done"
    assert len(provider.calls) == 2


def test_cli_plan_approved_executes_calls(agent_builder):
    registry = _gated_registry()
    inputs = iter(["y"])
    provider = FakeProvider([
        tool_call_response(ToolCall(id="c1", name="gated", arguments={})),
        text_response("done"),
    ])
    agent = agent_builder(
        provider,
        registry=registry,
        config=Config(plan_mode=True),
        approver=CliApprover(input_fn=lambda _: next(inputs)),
    )
    result = agent.run("do it")
    assert result == "done"


def test_cli_plan_denied_stops_run(agent_builder):
    registry = _gated_registry()
    inputs = iter(["n"])
    provider = FakeProvider([
        tool_call_response(ToolCall(id="c1", name="gated", arguments={})),
        text_response("ignored"),
    ])
    agent = agent_builder(
        provider,
        registry=registry,
        config=Config(plan_mode=True),
        approver=CliApprover(input_fn=lambda _: next(inputs)),
    )
    result = agent.run("do it")
    assert "Plan denied" in result
    assert len(provider.calls) == 1
