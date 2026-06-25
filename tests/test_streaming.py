"""Streaming tests (design §7: streaming must not change the loop contract).

Offline: the SSE accumulators are fed fabricated event dicts and the result is
run through the existing parse_response, so the streamed CompletionResponse is
identical to the non-streaming one."""

from __future__ import annotations

from agentkernel.providers import anthropic, openai
from tests.fakes import FakeProvider, text_response


def test_openai_accumulate_text_and_tool_calls():
    events = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "echo", "arguments": '{"v":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"hi"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {
            "prompt_tokens": 10, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 3}}},
    ]
    deltas: list[str] = []
    resp = openai.parse_response(openai.accumulate_stream(events, deltas.append))
    assert deltas == ["Hel", "lo"]
    assert resp.message.content == "Hello"
    assert resp.message.tool_calls[0].name == "echo"
    assert resp.message.tool_calls[0].arguments == {"v": "hi"}
    assert resp.stop_reason == "tool_use"
    assert resp.usage.input_tokens == 10 and resp.usage.cache_read_tokens == 3


def test_anthropic_accumulate_text_and_tool_use():
    events = [
        {"type": "message_start", "message": {"usage": {
            "input_tokens": 20, "cache_read_input_tokens": 5}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hi "}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "there"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "t1", "name": "echo"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"v":'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '"x"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    deltas: list[str] = []
    resp = anthropic.parse_response(anthropic.accumulate_stream(events, deltas.append))
    assert "".join(deltas) == "Hi there"
    assert resp.message.content == "Hi there"
    assert resp.message.tool_calls[0].name == "echo"
    assert resp.message.tool_calls[0].arguments == {"v": "x"}
    assert resp.stop_reason == "tool_use"
    assert resp.usage.input_tokens == 20
    assert resp.usage.cache_read_tokens == 5
    assert resp.usage.output_tokens == 7


def test_loop_forwards_streamed_text_to_on_text(agent_builder):
    provider = FakeProvider([text_response("streamed answer")])
    agent = agent_builder(provider)
    chunks: list[str] = []
    answer = agent.run("hi", on_text=chunks.append)
    assert answer == "streamed answer"
    assert "".join(chunks) == "streamed answer"
