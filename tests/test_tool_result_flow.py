"""The §8 tool-result pairing contract, asserted at the adapter level for both
Anthropic and OpenAI translation functions. Runs fully offline (design §15)."""

from __future__ import annotations

from agentkernel.providers import anthropic, openai
from agentkernel.types import Message, ToolCall, ToolResult


def _turn() -> list[Message]:
    """A turn with two tool calls and their two results, in canonical form."""
    return [
        Message(role="user", content="do two things"),
        Message(
            role="assistant",
            content="working",
            tool_calls=[
                ToolCall("c1", "a", {"x": 1}),
                ToolCall("c2", "b", {"y": 2}),
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult("c1", "res-1"),
                ToolResult("c2", "res-2", is_error=True),
            ],
        ),
    ]


def test_anthropic_pairs_all_results_in_one_user_message():
    wire = anthropic.render_messages(_turn())
    # assistant tool_use blocks carry the ids...
    assistant = wire[1]
    tool_use_ids = [b["id"] for b in assistant["content"] if b["type"] == "tool_use"]
    assert tool_use_ids == ["c1", "c2"]
    # ...and ALL results come back in a single user message, paired + ordered.
    result_msg = wire[2]
    assert result_msg["role"] == "user"
    blocks = result_msg["content"]
    assert [b["tool_use_id"] for b in blocks] == ["c1", "c2"]
    assert blocks[1]["is_error"] is True


def test_openai_pairs_each_result_in_its_own_tool_message():
    wire = openai.render_messages(_turn())
    # assistant tool_calls array carries the ids (arguments are JSON strings)...
    assistant = next(m for m in wire if m["role"] == "assistant")
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["c1", "c2"]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    # ...and there is exactly one role:tool message per result, in order.
    tool_msgs = [m for m in wire if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


def test_openai_system_is_first_message():
    wire = openai.render_messages([Message(role="user", content="hi")], system="SYS")
    assert wire[0] == {"role": "system", "content": "SYS"}
