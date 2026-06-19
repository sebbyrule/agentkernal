"""Registry tests (design §15): schema validation rejects bad args and yields an
error result rather than executing; handler exceptions become error results."""

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


def _spec(handler) -> ToolSpec:
    return ToolSpec("t", "desc", _SCHEMA, handler)


def test_validate_unknown_tool():
    reg = ToolRegistry()
    assert "Unknown tool" in reg.validate(ToolCall("c1", "missing", {}))


def test_validate_rejects_bad_args():
    reg = ToolRegistry()
    reg.register(_spec(lambda a: ToolResult("", "ok")))
    err = reg.validate(ToolCall("c1", "t", {"value": 123}))  # wrong type
    assert err is not None and "Invalid arguments" in err


def test_validate_accepts_good_args():
    reg = ToolRegistry()
    reg.register(_spec(lambda a: ToolResult("", "ok")))
    assert reg.validate(ToolCall("c1", "t", {"value": "x"})) is None


def test_execute_stamps_call_id():
    reg = ToolRegistry()
    reg.register(_spec(lambda a: ToolResult("", "ok")))  # handler omits call_id
    result = reg.execute(ToolCall("c42", "t", {"value": "x"}))
    assert result.call_id == "c42"


def test_handler_exception_becomes_error_result():
    def boom(args):
        raise RuntimeError("kaboom")

    reg = ToolRegistry()
    reg.register(_spec(boom))
    result = reg.execute(ToolCall("c1", "t", {"value": "x"}))
    assert result.is_error and "kaboom" in result.content
    assert result.call_id == "c1"


def test_duplicate_registration_raises():
    reg = ToolRegistry()
    reg.register(_spec(lambda a: ToolResult("", "ok")))
    try:
        reg.register(_spec(lambda a: ToolResult("", "ok")))
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on duplicate registration")


def test_specs_preserve_registration_order():
    reg = ToolRegistry()
    for n in ["zebra", "alpha", "mid"]:
        reg.register(ToolSpec(n, "d", _SCHEMA, lambda a: ToolResult("", "ok")))
    # Stable insertion order, never re-sorted (cacheable prefix, design §9.3).
    assert [s.name for s in reg.specs()] == ["zebra", "alpha", "mid"]


def test_bad_args_yield_error_result_without_executing(agent_builder):
    """End to end: the loop turns a validation failure into an error result and
    keeps going, never invoking the handler."""
    calls = {"n": 0}

    def handler(args):
        calls["n"] += 1
        return ToolResult("", "ran")

    reg = ToolRegistry()
    reg.register(_spec(handler))
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "t", {"value": 999})),  # invalid
            text_response("recovered"),
        ]
    )
    agent = agent_builder(provider, reg)
    assert agent.run("go") == "recovered"
    assert calls["n"] == 0  # handler never executed
    result = provider.calls[1][-1].tool_results[0]
    assert result.is_error and result.call_id == "c1"
