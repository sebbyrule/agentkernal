"""Sub-agent delegation via a ``spawn`` tool (design §13).

Re-entrancy (a tool handler calling ``Agent.run``) is already a kernel property;
this exposes it to the *model* as an ordinary registered tool. The handler builds
a child ``Agent`` with its own fresh context — optionally a focused system prompt
and a subset of tools — runs the subtask, and returns the child's final answer.

A depth limit prevents unbounded recursion: each child receives a ``spawn`` tool
with one less depth budget, and at ``max_depth`` the child gets no ``spawn`` at
all. Because the budget is captured per construction (not a shared counter), this
stays correct under the loop's re-entrancy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentkernel.agent import Agent
from agentkernel.context import ContextManager
from agentkernel.profiles import Profile
from agentkernel.telemetry import NullTelemetry
from agentkernel.tools.base import ToolRegistry, ToolSpec
from agentkernel.types import ToolResult

if TYPE_CHECKING:
    from agentkernel.approval import Approver
    from agentkernel.config import Config
    from agentkernel.providers import Provider

_SPAWN_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The self-contained subtask for the sub-agent to complete.",
        },
        "system": {
            "type": "string",
            "description": "Optional system prompt focusing the sub-agent.",
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional subset of tool names the sub-agent may use.",
        },
    },
    "required": ["task"],
    "additionalProperties": False,
}


def make_spawn_tool(
    *,
    provider: Provider,
    base_specs: list[ToolSpec],
    approver: Approver,
    config: Config,
    max_depth: int = 2,
    depth: int = 1,
) -> ToolSpec:
    """Build a ``spawn`` tool. ``base_specs`` are the tools a child may use
    (must NOT include a spawn tool — deeper spawns are added here recursively)."""

    def handler(arguments: dict[str, Any]) -> ToolResult:
        task = arguments["task"]
        wanted = arguments.get("tools")
        specs = base_specs
        if wanted:
            allowed = set(wanted)
            specs = [s for s in base_specs if s.name in allowed]

        child_registry = ToolRegistry()
        for spec in specs:
            child_registry.register(spec)
        # Give the child its own (shallower) spawn ability until the limit.
        if depth < max_depth:
            child_registry.register(
                make_spawn_tool(
                    provider=provider,
                    base_specs=base_specs,
                    approver=approver,
                    config=config,
                    max_depth=max_depth,
                    depth=depth + 1,
                )
            )

        context = ContextManager(
            budget=provider.context_window - config.output_reserve,
            keep_recent_turns=config.keep_recent_turns,
        )
        child = Agent(
            provider, child_registry, context, approver, NullTelemetry(), config
        )
        system = arguments.get("system")
        profile = Profile(name="subagent", system_prompt=system) if system else None
        try:
            answer = child.run(task, profile=profile)
        except Exception as exc:  # noqa: BLE001 - a child failure is a tool result
            return ToolResult("", f"sub-agent error: {exc}", is_error=True)
        return ToolResult("", answer, data={"depth": depth})

    return ToolSpec(
        name="spawn",
        description=(
            "Delegate a self-contained subtask to a focused sub-agent and return "
            "its final answer. Optionally restrict the sub-agent's tools or give "
            "it a system prompt. Use this to isolate a side investigation or to "
            "parallelize independent work."
        ),
        parameters=_SPAWN_SCHEMA,
        handler=handler,
        category="agent",
    )
