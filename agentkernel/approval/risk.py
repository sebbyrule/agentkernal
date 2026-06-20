"""Risk judge for the ``smart`` approval mode (design §18.1).

A small auxiliary model classifies a pending gated tool call as low- or high-risk
so the approver can auto-approve the boring ones (reading a file, listing a dir)
and prompt only on the dangerous ones (``rm -rf``, overwriting config, anything
irreversible). It is intentionally conservative: any parse failure or provider
error returns ``None`` so the approver falls back to asking — a judge that can't
decide must never silently approve.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from agentkernel.types import Message

if TYPE_CHECKING:
    from agentkernel.providers import Provider
    from agentkernel.tools import ToolSpec
    from agentkernel.types import ToolCall

_SYSTEM = (
    "You are a security gate for an autonomous coding agent. Decide whether a "
    "pending tool call is safe to auto-approve without a human. Treat as HIGH "
    "risk anything destructive or irreversible: deleting or overwriting data, "
    "force-resetting version control, modifying system or global config, "
    "installing software, sending data over the network, or running shell "
    "commands with broad/ambiguous scope. Treat as LOW risk read-only or easily "
    "reversible, narrowly-scoped actions. When unsure, choose high. Respond with "
    'ONLY a JSON object: {"risk": "low" | "high", "reason": "<short>"}.'
)


def _build_prompt(call: ToolCall, spec: ToolSpec | None) -> str:
    flags = []
    if spec is not None:
        if spec.mutates:
            flags.append("mutates")
        if spec.runs_code:
            flags.append("runs_code")
    try:
        args = json.dumps(call.arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        args = str(call.arguments)
    return (
        f"Tool: {call.name}\n"
        f"Flags: {', '.join(flags) or 'none'}\n"
        f"Arguments: {args}\n\n"
        "Classify the risk of running this call."
    )


def _parse_risk(text: str) -> bool | None:
    """Return True (low risk), False (high risk), or None (undecided)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    risk = str(data.get("risk", "")).strip().lower()
    if risk == "low":
        return True
    if risk == "high":
        return False
    return None


class RiskJudge:
    """Classifies a pending tool call's risk with a cheap auxiliary model."""

    def __init__(self, provider: Provider, *, max_tokens: int = 256) -> None:
        self._provider = provider
        self._max_tokens = max_tokens

    def is_low_risk(self, call: ToolCall, spec: ToolSpec | None) -> bool | None:
        """True if safe to auto-approve, False if it should be asked, None if the
        judge could not decide (provider error or unparseable reply)."""
        try:
            resp = self._provider.complete(
                [Message(role="user", content=_build_prompt(call, spec))],
                [],
                max_tokens=self._max_tokens,
                system=_SYSTEM,
            )
        except Exception:  # noqa: BLE001 - a judge failure must fall back to asking
            return None
        return _parse_risk(resp.message.content)
