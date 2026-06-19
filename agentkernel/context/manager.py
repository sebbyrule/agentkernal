"""Conversation context: accounting, budget, and compaction (design §9).

The system prompt and tool definitions are NOT in this message list — they live
in the cacheable prefix (§9.3) and are passed to the provider separately. So
compaction here operates only on user/assistant/tool messages and can never
drop the system prompt.

Compaction collapses the oldest completed turns into one synthetic assistant
summary, always keeping the most recent ``keep_recent_turns`` turns verbatim and
never splitting an assistant tool-call from its tool results (§9.2).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from agentkernel.context.truncate import CHARS_PER_TOKEN
from agentkernel.types import Message

# A summarizer turns a list of (old) messages into one summary string. The
# default is the deterministic structural fallback below; a model-based
# summarizer can be injected here (design §9.2).
Summarizer = Callable[[list[Message]], str]


@dataclass
class CompactionEvent:
    """Telemetry for one compaction pass (design §9.2)."""

    turns_collapsed: int
    tokens_before: int
    tokens_after: int


def estimate_tokens(message: Message) -> int:
    """Conservative chars/4 estimate for one message (design §9.1)."""
    chars = len(message.content or "")
    for tc in message.tool_calls:
        chars += len(tc.name) + len(json.dumps(tc.arguments))
    for r in message.tool_results:
        chars += len(r.content or "")
    return max(1, chars // CHARS_PER_TOKEN)


def structural_summary(messages: list[Message]) -> str:
    """Deterministic, offline summary: message counts, tools used, files touched."""
    users = sum(1 for m in messages if m.role == "user")
    assistants = sum(1 for m in messages if m.role == "assistant")
    tool_names: list[str] = []
    paths: set[str] = set()
    for m in messages:
        for tc in m.tool_calls:
            tool_names.append(tc.name)
            p = tc.arguments.get("path")
            if isinstance(p, str):
                paths.add(p)
    parts = [f"{users} user and {assistants} assistant message(s) were exchanged"]
    if tool_names:
        counts = Counter(tool_names)
        parts.append("tools used: " + ", ".join(f"{n}×{c}" for n, c in counts.items()))
    if paths:
        parts.append("files touched: " + ", ".join(sorted(paths)))
    return "; ".join(parts) + "."


class ContextManager:
    def __init__(
        self,
        *,
        budget: int | None = None,
        keep_recent_turns: int = 6,
        summarizer: Optional[Summarizer] = None,
        estimator: Callable[[Message], int] = estimate_tokens,
    ) -> None:
        # budget=None means unlimited (no compaction) — used by tests that don't
        # exercise context limits.
        self._messages: list[Message] = []
        self._budget = budget
        self._keep_recent_turns = keep_recent_turns
        # TODO(owner): a model-based summarizer (config.summarizer_model) can be
        # injected here; until then the deterministic structural fallback is used.
        self._summarize: Summarizer = summarizer or structural_summary
        self._estimate = estimator
        self._pending_compaction: CompactionEvent | None = None

    def add(self, message: Message) -> None:
        if message.token_estimate is None:
            message.token_estimate = self._estimate(message)
        self._messages.append(message)

    def messages(self) -> list[Message]:
        """The full stored history."""
        return list(self._messages)

    def window(self) -> list[Message]:
        """Messages to send this turn, compacted in place if over budget (§9.2)."""
        if self._budget is not None and self._total() > self._budget:
            self._compact()
        return list(self._messages)

    def take_compaction(self) -> CompactionEvent | None:
        """Return and clear the most recent compaction event (for telemetry)."""
        event, self._pending_compaction = self._pending_compaction, None
        return event

    # --- internals ---------------------------------------------------------

    def _total(self) -> int:
        return sum(m.token_estimate or 0 for m in self._messages)

    def _group_turns(self) -> list[list[Message]]:
        """Group messages into atomic units. An assistant tool-call message is
        bound to the tool-result message that answers it so compaction can never
        split an open pair (§9.2)."""
        groups: list[list[Message]] = []
        i = 0
        n = len(self._messages)
        while i < n:
            m = self._messages[i]
            if (
                m.role == "assistant"
                and m.tool_calls
                and i + 1 < n
                and self._messages[i + 1].role == "tool"
            ):
                groups.append([m, self._messages[i + 1]])
                i += 2
            else:
                groups.append([m])
                i += 1
        return groups

    def _compact(self) -> None:
        groups = self._group_turns()
        if len(groups) <= self._keep_recent_turns:
            return  # nothing old enough to compact; keep recent turns verbatim

        old = groups[: -self._keep_recent_turns]
        recent = groups[-self._keep_recent_turns :]
        old_messages = [m for g in old for m in g]

        tokens_before = self._total()
        summary = Message(
            role="assistant",
            content="Earlier in this session: " + self._summarize(old_messages),
        )
        summary.token_estimate = self._estimate(summary)
        recent_messages = [m for g in recent for m in g]

        self._messages = [summary, *recent_messages]
        self._pending_compaction = CompactionEvent(
            turns_collapsed=len(old),
            tokens_before=tokens_before,
            tokens_after=self._total(),
        )
