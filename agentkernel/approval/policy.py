"""Approval-policy decision, shared by every Approver (design §10.2).

Policies: ``always_ask`` (default), ``auto_allow``, ``deny_mutations``, and
``smart``. An optional allowlist of patterns (matched against the tool name and,
for shell tools, the command) skips the gate. The loop only consults an approver
for gated tools, but ``decide`` stays safe for non-gated ones too.

``smart`` resolves here to ``ask`` (the safe default); the approver, which has a
provider, may consult a risk judge before that prompt and auto-approve low-risk
calls. ``decide`` is pure and has no model, so the judging lives in the approver.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, Literal

from agentkernel.types import ToolCall

if TYPE_CHECKING:
    from agentkernel.tools import ToolSpec

Decision = Literal["allow", "deny", "ask"]


def _allowlisted(call: ToolCall, allowlist: list[str]) -> bool:
    if not allowlist:
        return False
    targets = [call.name]
    command = call.arguments.get("command")
    if isinstance(command, str):
        targets.append(command)
    for pattern in allowlist:
        for target in targets:
            if target == pattern or target.startswith(pattern) or fnmatch.fnmatch(
                target, pattern
            ):
                return True
    return False


def decide(
    policy: str,
    spec: ToolSpec,
    call: ToolCall,
    allowlist: list[str] | None = None,
) -> Decision:
    """Resolve a policy to allow / deny / ask for this call."""
    if not spec.gated:
        return "allow"
    if _allowlisted(call, allowlist or []):
        return "allow"
    if policy == "auto_allow":
        return "allow"
    if policy == "deny_mutations":
        return "deny" if (spec.mutates or spec.runs_code) else "allow"
    # always_ask (default and unknown-policy fallback)
    return "ask"
