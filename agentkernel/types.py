"""Canonical, provider-independent data types (design §4).

These types are the lingua franca of the kernel. Nothing outside a provider
adapter speaks a provider's native format: Anthropic content blocks and OpenAI
``tool_calls`` arrays are translated to and from these types inside
``agentkernel/providers/*`` and never appear in the loop, registry, or context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A model request to invoke a tool. ``id`` is unique within a run."""

    id: str
    name: str
    arguments: dict[str, Any]  # already parsed from JSON by the adapter


@dataclass
class ToolResult:
    """The outcome of a tool call. ``call_id`` pairs back to ``ToolCall.id``.

    A failure (validation error, approval denial, handler exception, or the
    tool's own error) is reported with ``is_error=True`` rather than raised, so
    the loop continues and the model can recover (design §8.3).
    """

    call_id: str
    content: str  # text shown to the model
    is_error: bool = False
    data: dict | None = None  # structured payload for kernel use; not model-visible


@dataclass
class Message:
    """One conversational turn in canonical form.

    A single assistant turn may carry both ``content`` text and one or more
    ``tool_calls``. A tool-role turn carries ``tool_results`` (one per call from
    the preceding assistant turn).
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant turns only
    tool_results: list[ToolResult] = field(default_factory=list)  # tool turns only
    # Bookkeeping:
    cacheable: bool = False  # marks a stable prefix boundary (design §9.3)
    token_estimate: int | None = None


@dataclass
class Usage:
    """Token accounting for one completion, including cache read/write."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class CompletionResponse:
    """A provider's reply, normalized. ``raw`` is for debugging only."""

    message: Message  # the assistant message (text and/or tool_calls)
    usage: Usage
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | provider-specific
    raw: Any = None  # untouched provider response; never inspected by the loop
