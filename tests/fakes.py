"""Test fakes (design §15). The FakeProvider is the most important fixture: it
drives the full loop deterministically with zero network access."""

from __future__ import annotations

from agentkernel.types import CompletionResponse, Message, ToolCall, Usage


class FakeProvider:
    """Returns a scripted sequence of CompletionResponses, one per ``complete``.

    Each ``complete`` call records the messages it was handed (in ``.calls``) so
    tests can assert on what the loop sent — e.g. the §8 tool-result pairing.
    """

    name = "fake"
    context_window = 200_000

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[list[Message]] = []
        self.tool_args: list[object] = []  # the tools list passed each turn
        self.system_args: list[str | None] = []

    def complete(
        self,
        messages,
        tools,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        system: str | None = None,
    ) -> CompletionResponse:
        # Snapshot the conversation as the loop sent it this turn.
        self.calls.append(list(messages))
        self.tool_args.append(tools)
        self.system_args.append(system)
        if self._index >= len(self._responses):
            raise AssertionError("FakeProvider ran out of scripted responses")
        resp = self._responses[self._index]
        self._index += 1
        return resp


def text_response(text: str, *, stop_reason: str = "end_turn") -> CompletionResponse:
    """A final assistant answer with no tool calls."""
    return CompletionResponse(
        message=Message(role="assistant", content=text),
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason=stop_reason,
    )


def tool_call_response(*calls: ToolCall, text: str = "") -> CompletionResponse:
    """An assistant turn requesting one or more tool calls (design §7)."""
    return CompletionResponse(
        message=Message(role="assistant", content=text, tool_calls=list(calls)),
        usage=Usage(input_tokens=12, output_tokens=8),
        stop_reason="tool_use",
    )
