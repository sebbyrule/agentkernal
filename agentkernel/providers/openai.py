"""OpenAI Chat Completions adapter (design §5, §8.1).

Wire shape: assistant ``tool_calls`` array (arguments are JSON *strings*); each
tool result is its own ``role: "tool"`` message keyed by ``tool_call_id``.
OpenAI caches the prefix automatically, so there are no explicit cache markers —
``cache_read_tokens`` is read back from ``usage.prompt_tokens_details``.
"""

from __future__ import annotations

import json
from typing import Any

from agentkernel.providers._http import ProviderError, post_json_pooled
from agentkernel.providers.credentials import CredentialPool
from agentkernel.tools import ToolSpec
from agentkernel.types import CompletionResponse, Message, ToolCall, Usage

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CONTEXT_WINDOW = 128_000

_STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


# --- translation: canonical -> wire (pure, offline-testable) ---------------


def render_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def render_messages(
    messages: list[Message], system: str | None = None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            # One message per result, keyed by tool_call_id (design §8.1).
            out.extend(
                {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                for r in m.tool_results
            )
        # role == "system" messages are delivered via the `system` param.
    return out


# --- translation: wire -> canonical ----------------------------------------


def parse_response(data: dict[str, Any]) -> CompletionResponse:
    choice = data["choices"][0]
    msg = choice.get("message", {})
    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}  # malformed JSON surfaces as a validation error in §6
        tool_calls.append(
            ToolCall(id=tc["id"], name=fn.get("name", ""), arguments=args)
        )
    u = data.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    usage = Usage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
        cache_read_tokens=cached,
    )
    finish = choice.get("finish_reason", "")
    return CompletionResponse(
        message=Message(
            role="assistant",
            content=msg.get("content") or "",
            tool_calls=tool_calls,
        ),
        usage=usage,
        stop_reason=_STOP_REASONS.get(finish, finish),
        raw=data,
    )


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        name: str = "openai",
        require_key: bool = True,
        env_key: str = "OPENAI_API_KEY",
        send_reasoning: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.context_window = context_window
        self._base_url = base_url.rstrip("/")
        self._require_key = require_key
        self._send_reasoning = send_reasoning
        self._pool = (
            CredentialPool([api_key]) if api_key else CredentialPool.from_env(env_key)
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
        if self._require_key and self._pool.current() is None:
            raise ProviderError(f"API key for provider {self.name!r} is not set")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": render_messages(messages, system),
        }
        # reasoning_effort is honored by OpenAI reasoning models; only sent when a
        # profile asks for it, and never for local endpoints that may reject it.
        if reasoning and self._send_reasoning:
            payload["reasoning_effort"] = reasoning
        if tools:
            payload["tools"] = render_tools(tools)

        def header_for_key(key: str | None) -> dict[str, str]:
            headers = {"content-type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            return headers

        url = f"{self._base_url}/chat/completions"
        return parse_response(
            post_json_pooled(
                url, header_for_key=header_for_key, payload=payload, pool=self._pool
            )
        )
