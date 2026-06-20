"""Builtin tools the kernel ships (design §6.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentkernel.tools.builtin.files import file_tools
from agentkernel.tools.builtin.search import search_tools
from agentkernel.tools.builtin.shell import bash_tool

if TYPE_CHECKING:
    from agentkernel.approval import Sandbox
    from agentkernel.checkpoint import Checkpointer
    from agentkernel.tools.base import ToolSpec

__all__ = ["file_tools", "search_tools", "bash_tool", "default_tools"]


def default_tools(
    sandbox: Sandbox,
    working_dir: str = ".",
    *,
    max_result_tokens: int = 4096,
    bash_timeout: int = 60,
    checkpointer: Checkpointer | None = None,
) -> list[ToolSpec]:
    """The full builtin toolset: file + search tools + bash, bound to one working dir.

    When ``checkpointer`` is set, the file tools record pre-edit state for
    rollback; the ``rollback`` tool itself is registered by the runtime builder.
    """
    tools = file_tools(
        working_dir, max_result_tokens=max_result_tokens, checkpointer=checkpointer
    )
    tools += search_tools(working_dir, max_result_tokens=max_result_tokens)
    tools.append(bash_tool(sandbox, working_dir, timeout=bash_timeout))
    return tools
