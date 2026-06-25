"""Parallel tool execution within one assistant turn (design §7).

Approval/validation runs sequentially (ordered prompts); the approved handlers
run concurrently when config.tool_concurrency > 1, with results paired back by
index so the §8 ordering contract still holds."""

from __future__ import annotations

import threading
import time

from agentkernel.config import Config
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult
from tests.conftest import build_agent
from tests.fakes import FakeProvider, text_response, tool_call_response

_SCHEMA = {
    "type": "object",
    "properties": {"v": {"type": "string"}},
    "required": ["v"],
    "additionalProperties": False,
}


def _sleep_tool(name: str, seconds: float, seen: list) -> ToolSpec:
    def handler(args):
        seen.append(threading.get_ident())
        time.sleep(seconds)
        return ToolResult("", f"{name}:{args['v']}")

    return ToolSpec(name, "sleep then echo", _SCHEMA, handler)


def test_parallel_tool_calls_run_concurrently_and_keep_order():
    seen: list[int] = []
    reg = ToolRegistry()
    reg.register(_sleep_tool("a", 0.2, seen))
    reg.register(_sleep_tool("b", 0.2, seen))
    reg.register(_sleep_tool("c", 0.2, seen))
    provider = FakeProvider(
        [
            tool_call_response(
                ToolCall("c1", "a", {"v": "1"}),
                ToolCall("c2", "b", {"v": "2"}),
                ToolCall("c3", "c", {"v": "3"}),
            ),
            text_response("done"),
        ]
    )
    agent = build_agent(provider, reg, config=Config(tool_concurrency=4))

    start = time.perf_counter()
    assert agent.run("go") == "done"
    elapsed = time.perf_counter() - start

    # Three 0.2s tools in parallel finish well under the 0.6s sequential total.
    assert elapsed < 0.5
    assert len(set(seen)) >= 2  # ran on multiple threads
    # Results are still paired and ordered to the calls (design §8).
    tool_msg = provider.calls[1][-1]
    assert [r.call_id for r in tool_msg.tool_results] == ["c1", "c2", "c3"]
    assert [r.content for r in tool_msg.tool_results] == ["a:1", "b:2", "c:3"]


def test_sequential_when_concurrency_is_one():
    seen: list[int] = []
    reg = ToolRegistry()
    reg.register(_sleep_tool("a", 0.05, seen))
    reg.register(_sleep_tool("b", 0.05, seen))
    provider = FakeProvider(
        [
            tool_call_response(
                ToolCall("c1", "a", {"v": "1"}), ToolCall("c2", "b", {"v": "2"})
            ),
            text_response("ok"),
        ]
    )
    agent = build_agent(provider, reg, config=Config(tool_concurrency=1))
    assert agent.run("go") == "ok"
    assert len(set(seen)) == 1  # all on the calling thread
