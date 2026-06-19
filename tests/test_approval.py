"""Approval tests (design §15): policies, the allowlist, CLI prompting, and the
end-to-end denial path where a refused tool yields an error result and the loop
keeps going."""

from __future__ import annotations

from agentkernel.approval import AutoApprover, CliApprover
from agentkernel.approval.policy import decide
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult
from tests.fakes import FakeProvider, text_response, tool_call_response

_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "additionalProperties": True,
}


def _spec(**flags) -> ToolSpec:
    return ToolSpec("bash", "d", _SCHEMA, lambda a: ToolResult("", "ok"), **flags)


def _call(**args) -> ToolCall:
    return ToolCall("c1", "bash", args)


# --- policy decisions -------------------------------------------------------


def test_auto_allow_allows_gated_tool():
    assert decide("auto_allow", _spec(runs_code=True), _call()) == "allow"


def test_deny_mutations_denies_mutating_and_allows_readonly():
    assert decide("deny_mutations", _spec(mutates=True), _call()) == "deny"
    assert decide("deny_mutations", _spec(), _call()) == "allow"  # not gated


def test_always_ask_asks_for_gated_tool():
    assert decide("always_ask", _spec(runs_code=True), _call()) == "ask"


def test_allowlist_skips_the_gate():
    spec = _spec(runs_code=True)
    assert decide("always_ask", spec, _call(command="ls -la"), ["ls*"]) == "allow"
    assert decide("always_ask", spec, _call(command="rm -rf /"), ["ls*"]) == "ask"


def test_non_gated_tool_always_allowed():
    assert decide("deny_mutations", _spec(), _call()) == "allow"


# --- approver implementations ----------------------------------------------


def test_auto_approver_ask_default():
    spec = _spec(runs_code=True)
    assert AutoApprover(ask_default=True).approve(_call(), spec) is True
    assert AutoApprover(ask_default=False).approve(_call(), spec) is False
    assert AutoApprover("auto_allow", ask_default=False).approve(_call(), spec) is True


def test_cli_approver_prompts_and_parses_yes_no():
    spec = _spec(runs_code=True)
    prompts: list[str] = []

    def yes(_prompt):
        prompts.append(_prompt)
        return "y"

    assert CliApprover(input_fn=yes, output_fn=lambda _: None).approve(_call(), spec)
    assert "bash" in prompts[0]  # the pending call is shown to the user

    no = CliApprover(input_fn=lambda _p: "n", output_fn=lambda _: None)
    assert no.approve(_call(), spec) is False


def test_cli_approver_deny_mutations_does_not_prompt():
    asked = {"n": 0}

    def counting_input(_p):
        asked["n"] += 1
        return "y"

    approver = CliApprover(
        "deny_mutations", input_fn=counting_input, output_fn=lambda _: None
    )
    assert approver.approve(_call(), _spec(mutates=True)) is False
    assert asked["n"] == 0  # denied by policy without ever prompting


# --- end-to-end denial path -------------------------------------------------


def test_denied_tool_yields_error_result_and_loop_continues(agent_builder):
    calls = {"n": 0}

    def handler(args):
        calls["n"] += 1
        return ToolResult("", "ran")

    reg = ToolRegistry()
    reg.register(ToolSpec("bash", "d", _SCHEMA, handler, runs_code=True))
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "bash", {"command": "ls"})),
            text_response("ok, I won't run it"),
        ]
    )
    agent = agent_builder(provider, reg, approver=AutoApprover(ask_default=False))
    assert agent.run("go") == "ok, I won't run it"
    assert calls["n"] == 0  # denial prevented execution
    result = provider.calls[1][-1].tool_results[0]
    assert result.is_error and "Denied" in result.content and result.call_id == "c1"
