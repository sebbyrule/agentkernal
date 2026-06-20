"""Anthropic Messages API adapter (design §5, §8.1, §9.3).

Wire shape: assistant ``tool_use`` content blocks; all tool results for a turn
go in a single ``user`` message of ``tool_result`` blocks keyed by
``tool_use_id``. The stable prefix (system + tool defs) carries
``cache_control: ephemeral`` on its final element so Anthropic serves it from
cache. No Anthropic dict escapes this module except inside ``CompletionResponse.raw``.
"""

from __future__ import annotations

from typing import Any

from agentkernel.providers._http import ProviderError, post_json_pooled
from agentkernel.providers.credentials import CredentialPool
from agentkernel.tools import ToolSpec
from agentkernel.types import CompletionResponse, Message, ToolCall, Usage

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_CONTEXT_WINDOW = 200_000
_EPHEMERAL = {"type": "ephemeral"}


# --- translation: canonical -> wire (pure, offline-testable) ---------------


def render_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Render tool specs to Anthropic's schema, caching the prefix at the last
    tool. Order is preserved (never re-sorted) so the prefix stays byte-stable."""
    wire: list[dict[str, Any]] = [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]
    if wire:
        wire[-1]["cache_control"] = _EPHEMERAL  # prefix boundary (design §9.3)
    return wire


_THINKING_BUDGET = {"low": 1024, "medium": 4096, "high": 8192}


def thinking_config(reasoning: str | None, max_tokens: int) -> dict[str, Any] | None:
    """Map a reasoning level to an extended-thinking block, or None.

    The budget is capped below ``max_tokens`` (thinking must leave room for the
    reply); if there isn't enough room, thinking is skipped rather than erroring.
    Thinking blocks in the response are already ignored by ``parse_response``.
    """
    if not reasoning:
        return None
    headroom = max_tokens - 1024
    if headroom < 1024:
        return None
    budget = min(_THINKING_BUDGET.get(reasoning, 4096), headroom)
    return {"type": "enabled", "budget_tokens": budget}


def render_system(system: str | None) -> list[dict[str, Any]] | None:
    """System prompt as a cached text block, or None when absent."""
    if not system:
        return None
    return [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]


def render_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in m.tool_calls
                )
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "assistant", "content": m.content})
        elif m.role == "tool":
            # All results for the turn in one user message (design §8.1).
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.call_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        }
                        for r in m.tool_results
                    ],
                }
            )
        # role == "system" is delivered via the `system` param, not as a message.
    return out


# --- translation: wire -> canonical ----------------------------------------


def parse_response(data: dict[str, Any]) -> CompletionResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            args = block.get("input")
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=args if isinstance(args, dict) else {},
                )
            )
    u = data.get("usage", {})
    usage = Usage(
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read_tokens=u.get("cache_read_input_tokens", 0),
        cache_write_tokens=u.get("cache_creation_input_tokens", 0),
    )
    return CompletionResponse(
        message=Message(
            role="assistant", content="".join(text_parts), tool_calls=tool_calls
        ),
        usage=usage,
        stop_reason=data.get("stop_reason", ""),
        raw=data,
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self._pool = (
            CredentialPool([api_key]) if api_key
            else CredentialPool.from_env("ANTHROPIC_API_KEY")
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int,
        temperature: float = 1.0,
        system: str | None = None,
        reasoning: str | None = None,
    ) -> CompletionResponse:
        if self._pool.current() is None:
            raise ProviderError("ANTHROPIC_API_KEY is not set in the environment")
        thinking = thinking_config(reasoning, max_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # Extended thinking requires temperature 1; otherwise honor the caller.
            "temperature": 1.0 if thinking else temperature,
            "messages": render_messages(messages),
        }
        if thinking is not None:
            payload["thinking"] = thinking
        if tools:
            payload["tools"] = render_tools(tools)
        sys_blocks = render_system(system)
        if sys_blocks is not None:
            payload["system"] = sys_blocks

        def header_for_key(key: str | None) -> dict[str, str]:
            return {
                "x-api-key": key or "",
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            }

        return parse_response(
            post_json_pooled(
                API_URL, header_for_key=header_for_key, payload=payload, pool=self._pool
            )
        )
