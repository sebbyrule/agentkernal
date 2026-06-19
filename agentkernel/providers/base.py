"""The Provider protocol (design §5.1).

An adapter translates the canonical message/tool types to a provider's wire
format, calls the API, and translates the reply back into one
``CompletionResponse``. No provider-specific object escapes an adapter except
inside ``CompletionResponse.raw``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from agentkernel.types import CompletionResponse, Message

if TYPE_CHECKING:
    from agentkernel.tools import ToolSpec


class Provider(Protocol):
    name: str
    context_window: int  # total token capacity of the selected model

    def complete(
        self,
        messages: list[Message],
        tools: list["ToolSpec"],
        *,
        max_tokens: int,
        temperature: float = 1.0,
        system: str | None = None,
    ) -> CompletionResponse: ...
