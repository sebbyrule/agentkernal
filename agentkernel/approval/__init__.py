"""Approval gate and execution boundary (design §10)."""

from agentkernel.approval.base import Approver, Sandbox
from agentkernel.approval.cli import AutoApprover, CliApprover
from agentkernel.approval.policy import decide
from agentkernel.approval.sandbox import (
    DockerSandbox,
    LocalSandbox,
    SandboxError,
    make_sandbox,
)

__all__ = [
    "Approver",
    "Sandbox",
    "AutoApprover",
    "CliApprover",
    "LocalSandbox",
    "DockerSandbox",
    "SandboxError",
    "make_sandbox",
    "decide",
]
