"""Context tests (design §15): compaction triggers at budget, preserves the
recent turns, never splits an open tool pair, marks truncated content, and the
end-to-end loop keeps running across a compaction."""

from __future__ import annotations

from agentkernel.context import ContextManager, estimate_tokens, structural_summary
from agentkernel.context.truncate import truncate_text
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import Message, ToolCall, ToolResult

from tests.fakes import FakeProvider, text_response, tool_call_response


def _user(text: str) -> Message:
    return Message(role="user", content=text)


def test_no_compaction_under_budget():
    cm = ContextManager(budget=1_000, keep_recent_turns=2)
    for i in range(5):
        cm.add(_user(f"m{i}"))
    assert len(cm.window()) == 5
    assert cm.take_compaction() is None


def test_compaction_triggers_and_preserves_recent():
    cm = ContextManager(budget=50, keep_recent_turns=2)
    # Each message ~50 tokens (200 chars / 4); 5 of them blow the budget.
    for i in range(5):
        cm.add(_user("x" * 200 + f"#{i}"))
    window = cm.window()

    # Oldest 3 collapsed into one summary; recent 2 kept verbatim.
    assert window[0].role == "assistant"
    assert window[0].content.startswith("Earlier in this session:")
    assert [m.content[-2:] for m in window[1:]] == ["#3", "#4"]

    event = cm.take_compaction()
    assert event is not None
    assert event.turns_collapsed == 3
    assert event.tokens_after < event.tokens_before


def test_compaction_never_splits_open_tool_pair():
    cm = ContextManager(budget=40, keep_recent_turns=1)
    # An assistant tool-call bound to its tool-result must stay together.
    cm.add(_user("x" * 400))  # bulky, will be compacted away
    cm.add(
        Message(
            role="assistant",
            content="y" * 200,
            tool_calls=[ToolCall("c1", "bash", {"command": "ls"})],
        )
    )
    cm.add(Message(role="tool", tool_results=[ToolResult("c1", "out")]))
    window = cm.window()

    # The summary replaces the first user message; the assistant+tool pair (one
    # "turn") is preserved intact and in order.
    assert window[0].content.startswith("Earlier in this session:")
    assert window[1].role == "assistant" and window[1].tool_calls
    assert window[2].role == "tool" and window[2].tool_results[0].call_id == "c1"


def test_keeps_recent_when_nothing_old_enough():
    cm = ContextManager(budget=1, keep_recent_turns=3)
    for i in range(2):  # fewer turns than keep_recent_turns
        cm.add(_user("x" * 400))
    # Over budget but nothing older than the kept window -> returned as-is.
    assert len(cm.window()) == 2
    assert cm.take_compaction() is None


def test_structural_summary_lists_tools_and_files():
    msgs = [
        Message(role="user", content="read it"),
        Message(
            role="assistant",
            tool_calls=[ToolCall("c1", "read_file", {"path": "a.txt"})],
        ),
        Message(role="tool", tool_results=[ToolResult("c1", "data")]),
    ]
    s = structural_summary(msgs)
    assert "read_file" in s and "a.txt" in s


def test_estimate_tokens_counts_all_parts():
    m = Message(
        role="assistant",
        content="hello",
        tool_calls=[ToolCall("c1", "bash", {"command": "echo hi"})],
    )
    assert estimate_tokens(m) >= 1


def test_truncate_marks_removed_content():
    out = truncate_text("a" * 8000, max_tokens=50)
    assert "truncated" in out and len(out) < 8000
    assert truncate_text("short", max_tokens=50) == "short"


def test_loop_continues_across_compaction(agent_builder):
    """A session that exceeds budget compacts mid-run and still completes
    (design §16, M2 done-when)."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "blob",
            "returns a large blob",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda a: ToolResult("", "z" * 600),
        )
    )
    # Tiny budget forces compaction between turns; tool keeps producing bulk.
    context = ContextManager(budget=60, keep_recent_turns=2)
    provider = FakeProvider(
        [
            tool_call_response(ToolCall("c1", "blob", {})),
            tool_call_response(ToolCall("c2", "blob", {})),
            tool_call_response(ToolCall("c3", "blob", {})),
            text_response("finished"),
        ]
    )
    agent = agent_builder(provider, registry, context=context)
    assert agent.run("go") == "finished"
    # A compaction happened, yet the run produced its final answer.
    assert len(provider.calls) == 4
