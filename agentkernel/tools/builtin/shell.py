"""The bash tool (design §6.3, §10.3).

Flagged ``runs_code`` + ``mutates`` + ``requires_approval``, so the loop gates
it through the approver and it executes only inside the injected ``Sandbox``,
confined to the working directory. Output is assembled here; the loop applies
the shared §8.4 truncation before it enters context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

if TYPE_CHECKING:
    from agentkernel.approval import Sandbox

_BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to run in the working directory.",
        }
    },
    "required": ["command"],
    "additionalProperties": False,
}


def bash_tool(sandbox: "Sandbox", working_dir: str = ".", *, timeout: int = 60) -> ToolSpec:
    """Build the bash tool bound to ``sandbox`` and ``working_dir``."""

    def bash(args: dict) -> ToolResult:
        command = args["command"]
        exit_code, stdout, stderr = sandbox.run(
            command, cwd=working_dir, timeout=timeout
        )
        sections = []
        if stdout:
            sections.append(stdout.rstrip("\n"))
        if stderr:
            sections.append(f"[stderr]\n{stderr.rstrip(chr(10))}")
        if exit_code != 0:
            sections.append(f"[exit code {exit_code}]")
        content = "\n".join(sections) if sections else "(no output)"
        return ToolResult(
            "",
            content,
            is_error=exit_code != 0,
            data={"exit_code": exit_code},
        )

    return ToolSpec(
        name="bash",
        description=(
            "Run a shell command in the working directory and return its output. "
            "Use for builds, tests, git, and filesystem operations."
        ),
        parameters=_BASH_SCHEMA,
        handler=bash,
        requires_approval=True,
        mutates=True,
        runs_code=True,
        category="shell",
    )
