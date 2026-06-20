"""The `rollback` tool — restore files to their pre-edit checkpoint (§18.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

if TYPE_CHECKING:
    from agentkernel.checkpoint import Checkpointer


def rollback_tool(checkpointer: Checkpointer) -> ToolSpec:
    """Build a `rollback` tool bound to ``checkpointer``."""

    def rollback(_args: dict) -> ToolResult:
        if checkpointer.pending() == 0:
            return ToolResult("", "Nothing to roll back — no files have been modified.")
        n = checkpointer.rollback()
        return ToolResult("", f"Rolled back {n} file(s) to their pre-edit state.")

    return ToolSpec(
        name="rollback",
        description=(
            "Undo all file changes made this session, restoring every file the "
            "file tools modified (and deleting any they created) to its state at "
            "the start. Use this to recover after a wrong edit."
        ),
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=rollback,
        category="files",
    )
