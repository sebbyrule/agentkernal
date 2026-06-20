"""The agent loop (design §7).

This reads like the pseudocode in the design doc on purpose: no clever
metaprogramming, no provider-specific branching. The loop sends the context
window plus tools to the provider, executes any tool calls through the registry
(gating mutations through the approver), and appends every result back, paired
to its call id (design §8), until the model produces a final answer.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from agentkernel.budget import BudgetGuard
from agentkernel.context.truncate import truncate_text
from agentkernel.telemetry import ToolOutcome
from agentkernel.types import Message, ToolResult

if TYPE_CHECKING:
    from agentkernel.approval import Approver
    from agentkernel.config import Config
    from agentkernel.context import ContextManager
    from agentkernel.memory import MemoryNotes, MemoryStore
    from agentkernel.providers import Provider
    from agentkernel.skills import ContextSource
    from agentkernel.telemetry import Telemetry
    from agentkernel.tools import ToolRegistry, ToolSpec


class Agent:
    """Orchestrates one conversation. All collaborators are injected (no global
    state), so ``run`` is re-entrant: a tool handler may construct another Agent
    and call ``run`` to spawn a sub-agent (design §7, §13)."""

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        context: ContextManager,
        approver: Approver,
        telemetry: Telemetry,
        config: Config,
        budget: BudgetGuard | None = None,
        memory: MemoryStore | None = None,
        notes: MemoryNotes | None = None,
        context_source: ContextSource | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.context = context
        self.approver = approver
        self.telemetry = telemetry
        self.config = config
        self.budget = budget
        self.memory = memory
        self.notes = notes
        self.context_source = context_source

    def run(self, user_input: str, *, profile: Any | None = None) -> str:
        """Drive the loop until a final answer or the max-iteration guard.

        ``profile`` (design §13, Phase 5) is accepted but, in the kernel, only
        ``tool_filter`` / ``system_prompt`` are honored if trivially present.
        """
        session_id = getattr(self.telemetry, "session_id", str(uuid.uuid4()))

        # Pre-run memory load (Phase 3). Only load when context is empty so a
        # persistent REPL session does not replay the same stored turns twice.
        if self.memory is not None and not self.context.messages():
            for message in self.memory.load(session_id):
                self.context.add(message)

        self.context.add(
            Message(role="user", content=self._prepare_user_message(user_input))
        )

        # Assemble the cacheable prefix ONCE per run and reuse the same objects
        # every turn. Re-building or re-sorting these per turn would silently
        # destroy prompt-cache hit-rate (design §9.3, AGENT.md rule 3).
        tools = self._tools_for(profile)
        system = self._system_for(profile)

        if self.budget is not None:
            self.budget.reset()

        for iteration in range(self.config.max_iterations):
            messages = self.context.window()  # compacted to budget in M2

            resp = self.provider.complete(
                messages,
                tools,
                max_tokens=self.config.max_output_tokens,
                system=system,
            )
            self.context.add(resp.message)
            compaction = self.context.take_compaction()

            if self.budget is not None:
                self.budget.add(resp.usage)
                exceeded, reason = self.budget.exceeded()
                if exceeded:
                    # The token spend already happened; record it, then either
                    # return the final answer (if we have one) or stop early.
                    self.telemetry.record_turn(iteration, resp, compaction=compaction)
                    if not resp.message.tool_calls:
                        self._persist_memory(session_id)
                        return resp.message.content
                    self._persist_memory(session_id)
                    return f"Stopped: budget exceeded ({reason})."

            if not resp.message.tool_calls:
                self.telemetry.record_turn(iteration, resp, compaction=compaction)
                self._persist_memory(session_id)
                return resp.message.content  # final answer

            results: list[ToolResult] = []
            outcomes: list[ToolOutcome] = []
            for call in resp.message.tool_calls:  # may be >1 (parallel tool use)
                err = self.registry.validate(call)
                if err:
                    results.append(ToolResult(call.id, err, is_error=True))
                    outcomes.append(ToolOutcome(call.name, call.arguments, None, True))
                    continue
                spec = self.registry.spec(call.name)
                if self._needs_approval(spec) and not self.approver.approve(call, spec):
                    results.append(
                        ToolResult(call.id, "Denied by user.", is_error=True)
                    )
                    outcomes.append(ToolOutcome(call.name, call.arguments, False, True))
                    continue
                result = self.registry.execute(call)
                results.append(result)
                outcomes.append(
                    ToolOutcome(call.name, call.arguments, True, result.is_error)
                )

            # Cap every result before it enters context (design §8.4). This is
            # the single truncation point for all tools — builtin and future
            # ones — sharing the §9 mechanism. Structured `data` is left intact.
            for r in results:
                r.content = truncate_text(r.content, self.config.max_tool_result_tokens)

            self.telemetry.record_turn(
                iteration, resp, tool_outcomes=outcomes, compaction=compaction
            )

            # One tool-role message carries every result, paired to its call id.
            # The adapter fans this out to the provider's shape (design §8.1).
            self.context.add(Message(role="tool", tool_results=results))

        self._persist_memory(session_id)
        return "Stopped: reached max iterations without a final answer."

    # --- memory helper ------------------------------------------------------

    def _persist_memory(self, session_id: str) -> None:
        if self.memory is not None:
            self.memory.save(session_id, self.context.messages())

    def _prepare_user_message(self, user_input: str) -> str:
        """Augment the user input with relevant long-term memory when configured.

        This keeps memory at the model's fingertips for the current turn without
        changing the stable system-prompt prefix.
        """
        if not user_input or not self.notes or not getattr(
            self.config, "memory_auto_context", False
        ):
            return user_input
        limit = getattr(self.config, "memory_auto_context_limit", 3)
        notes = self.notes.search(user_input, limit=limit)
        if not notes:
            return user_input
        lines = ["Relevant long-term memory:"]
        for n in notes:
            lines.append(f"- {n.text}")
        lines.append("---")
        lines.append(user_input)
        return "\n".join(lines)

    # --- profile seams (design §13, Phase 5) -------------------------------

    def _tools_for(self, profile: Any | None) -> list[ToolSpec]:
        """The tool set for this run. Stable across turns to keep the prefix
        cacheable (design §9.3): assembled from the registry's registration
        order and not re-sorted."""
        specs = self.registry.specs()
        tool_filter = getattr(profile, "tool_filter", None)
        if tool_filter is not None:
            allowed = set(tool_filter)
            specs = [s for s in specs if s.name in allowed]
        return specs

    def _system_for(self, profile: Any | None) -> str | None:
        """Combine profile system prompt and active skill additions.

        The cacheable prefix stays stable because tools and system Prompt are
        assembled once per run.
        """
        parts: list[str] = []
        profile_prompt = getattr(profile, "system_prompt", None)
        if profile_prompt:
            parts.append(profile_prompt)
        if self.context_source is not None:
            parts.extend(self.context_source.system_additions())
        if not parts:
            return None
        return "\n\n".join(parts)

    @staticmethod
    def _needs_approval(spec: ToolSpec | None) -> bool:
        return bool(spec and spec.gated)
