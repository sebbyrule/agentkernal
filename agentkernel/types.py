"""Canonical, provider-independent data types (design §4).

These types are the lingua franca of the kernel. Nothing outside a provider
adapter speaks a provider's native format: Anthropic content blocks and OpenAI
``tool_calls`` arrays are translated to and from these types inside
``agentkernel/providers/*`` and never appear in the loop, registry, or context.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ImageContent:
    """An image attached to a message (design §18.6).

    ``kind="base64"`` carries raw image bytes, base64-encoded in ``data`` with an
    explicit ``media_type``. ``kind="url"`` carries a URL in ``data`` that the
    provider fetches itself. Adapters translate this to each provider's wire
    format; providers that cannot accept images ignore it (see
    ``Provider.supports_images``), so attaching an image never breaks a text-only
    run — it is simply not sent.
    """

    data: str  # base64 bytes (kind="base64") or a URL (kind="url")
    media_type: str = "image/png"
    kind: Literal["base64", "url"] = "base64"

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data, "media_type": self.media_type, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageContent:
        return cls(
            data=data["data"],
            media_type=data.get("media_type", "image/png"),
            kind=data.get("kind", "base64"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> ImageContent:
        """Load and base64-encode a local image file, inferring its media type."""
        p = Path(path)
        media_type = mimetypes.guess_type(p.name)[0] or "image/png"
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
        return cls(data=encoded, media_type=media_type, kind="base64")

    @classmethod
    def from_url(cls, url: str) -> ImageContent:
        return cls(data=url, kind="url")

    def as_data_uri(self) -> str:
        """An ``data:`` URI for OpenAI-style ``image_url`` parts."""
        if self.kind == "url":
            return self.data
        return f"data:{self.media_type};base64,{self.data}"


@dataclass
class ToolCall:
    """A model request to invoke a tool. ``id`` is unique within a run."""

    id: str
    name: str
    arguments: dict[str, Any]  # already parsed from JSON by the adapter

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(id=data["id"], name=data["name"], arguments=data.get("arguments", {}))


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "content": self.content,
            "is_error": self.is_error,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        return cls(
            call_id=data["call_id"],
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
            data=data.get("data"),
        )


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
    images: list[ImageContent] = field(default_factory=list)  # user turns (design §18.6)
    # Bookkeeping:
    cacheable: bool = False  # marks a stable prefix boundary (design §9.3)
    token_estimate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (e.g., for persistence in memory stores)."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "images": [img.to_dict() for img in self.images],
            "cacheable": self.cacheable,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Reconstruct a Message from `to_dict()` output."""
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            tool_results=[ToolResult.from_dict(tr) for tr in data.get("tool_results", [])],
            images=[ImageContent.from_dict(img) for img in data.get("images", [])],
            cacheable=data.get("cacheable", False),
            token_estimate=data.get("token_estimate"),
        )

    def __hash__(self) -> int:
        # Messages are mutable, but a stable hash is useful for in-memory store keys.
        return id(self)


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
