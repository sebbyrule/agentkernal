"""Tests for `smart` approval (design §18.1): the risk judge and its use by the
approvers to auto-approve low-risk gated calls while still prompting on high-risk."""

from __future__ import annotations

from agentkernel.approval import AutoApprover, CliApprover
from agentkernel.approval.risk import RiskJudge
from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolCall
from tests.fakes import FakeProvider, text_response


def _gated_spec() -> ToolSpec:
    return ToolSpec(
        name="bash",
        description="run a shell command",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda _a: None,  # never called by the approver
        runs_code=True,
        mutates=True,
        requires_approval=True,
    )


def _judge(verdict_json: str) -> RiskJudge:
    return RiskJudge(FakeProvider([text_response(verdict_json)]))


# --- the judge ---------------------------------------------------------------

def test_judge_parses_low_and_high():
    call = ToolCall("c", "bash", {"command": "ls"})
    assert _judge('{"risk": "low"}').is_low_risk(call, _gated_spec()) is True
    assert _judge('{"risk": "high"}').is_low_risk(call, _gated_spec()) is False


def test_judge_undecided_on_garbage():
    call = ToolCall("c", "bash", {"command": "ls"})
    assert _judge("no json here").is_low_risk(call, _gated_spec()) is None


def test_judge_returns_none_on_provider_error():
    class _Boom:
        def complete(self, *a, **k):
            raise RuntimeError("unreachable")

    judge = RiskJudge(_Boom())
    assert judge.is_low_risk(ToolCall("c", "bash", {}), _gated_spec()) is None


# --- smart approval via CliApprover ------------------------------------------

def _cli(judge, answers):
    it = iter(answers)
    prompted = []

    def input_fn(prompt):
        prompted.append(prompt)
        return next(it)

    approver = CliApprover(
        "smart", input_fn=input_fn, output_fn=lambda _m: None, risk_judge=judge
    )
    return approver, prompted


def test_smart_auto_approves_low_risk_without_prompting():
    approver, prompted = _cli(_judge('{"risk": "low"}'), answers=[])
    assert approver.approve(ToolCall("c", "bash", {"command": "ls"}), _gated_spec()) is True
    assert prompted == []  # never asked the human


def test_smart_prompts_on_high_risk():
    approver, prompted = _cli(_judge('{"risk": "high"}'), answers=["n"])
    assert approver.approve(ToolCall("c", "bash", {"command": "rm -rf /"}), _gated_spec()) is False
    assert len(prompted) == 1  # fell through to the human


def test_smart_prompts_when_judge_undecided():
    approver, prompted = _cli(_judge("garbage"), answers=["y"])
    assert approver.approve(ToolCall("c", "bash", {}), _gated_spec()) is True
    assert len(prompted) == 1


def test_smart_prompts_when_no_judge_configured():
    prompted = []
    approver = CliApprover(
        "smart", input_fn=lambda p: (prompted.append(p) or "y"), output_fn=lambda _m: None
    )
    assert approver.approve(ToolCall("c", "bash", {}), _gated_spec()) is True
    assert len(prompted) == 1


# --- smart approval via AutoApprover (non-interactive) -----------------------

def test_auto_approver_smart_uses_judge():
    spec = _gated_spec()
    low = AutoApprover("smart", ask_default=False, risk_judge=_judge('{"risk": "low"}'))
    high = AutoApprover("smart", ask_default=False, risk_judge=_judge('{"risk": "high"}'))
    assert low.approve(ToolCall("c", "bash", {}), spec) is True
    assert high.approve(ToolCall("c", "bash", {}), spec) is False  # falls to ask_default
