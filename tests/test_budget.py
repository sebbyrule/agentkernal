"""Budget guardrail tests."""

from __future__ import annotations

from agentkernel.budget import BudgetGuard
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import CompletionResponse, Message, ToolCall, ToolResult, Usage
from tests.fakes import FakeProvider, text_response, tool_call_response


def _echo_tool(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Echo.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=lambda args: ToolResult("", f"echo:{args['value']}"),
    )


def test_budget_allows_run_when_not_exceeded(agent_builder):
    guard = BudgetGuard(max_input_tokens=100, max_cost_usd=10.0, model="fake")
    provider = FakeProvider([text_response("ok")])
    agent = agent_builder(provider)
    agent.budget = guard
    assert agent.run("go") == "ok"


def test_budget_input_tokens_exceeded_stops_before_tool_execution(agent_builder):
    """A tool-use turn that would exceed the input-token limit is aborted."""
    registry = ToolRegistry()
    registry.register(_echo_tool())
    provider = FakeProvider(
        [tool_call_response(ToolCall("c1", "echo", {"value": "x"}))]
    )
    agent = agent_builder(provider, registry)
    agent.budget = BudgetGuard(max_input_tokens=1, model="fake")

    result = agent.run("go")

    assert "Stopped: budget exceeded" in result
    assert "input_tokens" in result


def test_budget_cost_exceeded_final_answer_still_returned(agent_builder):
    """If the budget is exceeded by the final-answer turn, the answer is still
    returned (the tokens are already spent)."""
    provider = FakeProvider(
        [
            CompletionResponse(
                message=Message(role="assistant", content="answer"),
                usage=Usage(input_tokens=1_000_000, output_tokens=0),
                stop_reason="end_turn",
            )
        ]
    )
    agent = agent_builder(provider)
    agent.budget = BudgetGuard(max_cost_usd=1.0, model="claude-sonnet-4-6")

    assert agent.run("go") == "answer"


def test_budget_resets_per_run(agent_builder):
    """The same Agent reused across REPL turns must reset cumulative totals."""
    guard = BudgetGuard(max_input_tokens=15, model="fake")
    provider = FakeProvider(
        [
            text_response("first"),
            text_response("second"),
        ]
    )
    agent = agent_builder(provider)
    agent.budget = guard
    assert agent.run("go") == "first"
    # Without reset the cumulative 10+10=20 > 15 would break the second run.
    assert agent.run("go") == "second"
