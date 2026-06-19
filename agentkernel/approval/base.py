"""Approver protocol (design §10.1).

The loop consults the approver before executing any gated tool (one whose
``requires_approval``, ``mutates``, or ``runs_code`` flag is set). A denial
produces a ``ToolResult(is_error=True)``; it never raises. The Sandbox protocol
and approval policies land in M3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from agentkernel.types import ToolCall

if TYPE_CHECKING:
    from agentkernel.tools import ToolSpec


class Approver(Protocol):
    def approve(self, call: ToolCall, spec: ToolSpec) -> bool: ...


class Sandbox(Protocol):
    """Execution boundary for ``runs_code`` tools (design §10.3).

    ``run`` executes a command confined to ``cwd`` and returns
    ``(exit_code, stdout, stderr)``. v1 ships ``LocalSandbox``; the target is a
    container-per-project ``DockerSandbox`` swappable behind this protocol.
    """

    def run(self, command: str, *, cwd: str, timeout: int) -> tuple[int, str, str]: ...
