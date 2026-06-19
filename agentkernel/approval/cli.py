"""Approver implementations (design §10.2).

Both apply the shared policy in ``policy.py``. ``CliApprover`` prompts the
terminal when the policy says ``ask``; ``AutoApprover`` never prompts (for tests
and non-interactive runs) and resolves ``ask`` to a fixed default.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from agentkernel.approval.policy import decide
from agentkernel.types import ToolCall

if TYPE_CHECKING:
    from agentkernel.tools import ToolSpec


def _summarize(call: ToolCall) -> str:
    """One-line, side-effect-free description of a pending call for the prompt."""
    try:
        args = json.dumps(call.arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        args = str(call.arguments)
    return f"{call.name}({args})"


class AutoApprover:
    """Non-interactive approver. Applies policy; resolves ``ask`` to ``ask_default``.

    Defaults (no args) allow everything, which is what the offline test agents
    rely on. Pass ``ask_default=False`` to exercise the denial path.
    """

    def __init__(
        self,
        policy: str = "always_ask",
        *,
        allowlist: list[str] | None = None,
        ask_default: bool = True,
    ) -> None:
        self._policy = policy
        self._allowlist = allowlist or []
        self._ask_default = ask_default

    def approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        decision = decide(self._policy, spec, call, self._allowlist)
        if decision == "allow":
            return True
        if decision == "deny":
            return False
        return self._ask_default


class CliApprover:
    """Interactive approver: prints the pending call and reads y/n when the
    policy requires asking. ``input_fn``/``output_fn`` are injectable for tests."""

    def __init__(
        self,
        policy: str = "always_ask",
        *,
        allowlist: list[str] | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._policy = policy
        self._allowlist = allowlist or []
        self._input = input_fn
        self._output = output_fn

    def approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        decision = decide(self._policy, spec, call, self._allowlist)
        if decision == "allow":
            return True
        if decision == "deny":
            self._output(f"Denied by policy: {_summarize(call)}")
            return False
        answer = self._input(f"Approve {_summarize(call)}? [y/N] ").strip().lower()
        return answer in ("y", "yes")
