"""Adapter response-parsing, cache-marker, and usage tests. Offline: every test
feeds the translation functions hand-built wire dicts (design §15)."""

from __future__ import annotations

from agentkernel.providers import anthropic, openai
from agentkernel.tools import ToolSpec
from agentkernel.types import ToolResult

_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _specs(*names: str) -> list[ToolSpec]:
    return [ToolSpec(n, f"desc {n}", _SCHEMA, lambda a: ToolResult("", "ok")) for n in names]


# --- Anthropic --------------------------------------------------------------


def test_anthropic_cache_marker_on_last_tool_only():
    wire = anthropic.render_tools(_specs("a", "b", "c"))
    assert "cache_control" not in wire[0]
    assert "cache_control" not in wire[1]
    assert wire[2]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_render_tools_preserves_order():
    wire = anthropic.render_tools(_specs("zebra", "alpha"))
    assert [t["name"] for t in wire] == ["zebra", "alpha"]  # never re-sorted


def test_anthropic_system_block_is_cached():
    blocks = anthropic.render_system("you are helpful")
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert anthropic.render_system(None) is None


def test_anthropic_parse_mixed_text_and_tool_use():
    data = {
        "content": [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 10,
        },
    }
    resp = anthropic.parse_response(data)
    assert resp.message.content == "let me check"
    assert resp.message.tool_calls[0].arguments == {"path": "x"}
    assert resp.stop_reason == "tool_use"
    assert resp.usage.cache_read_tokens == 80
    assert resp.usage.cache_write_tokens == 10
    assert resp.raw is data  # provider payload retained for debugging only


def test_anthropic_parse_handles_non_dict_input():
    data = {"content": [{"type": "tool_use", "id": "t1", "name": "f", "input": None}]}
    resp = anthropic.parse_response(data)
    assert resp.message.tool_calls[0].arguments == {}


# --- OpenAI -----------------------------------------------------------------


def test_openai_parse_tool_calls_and_cached_tokens():
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 128},
        },
    }
    resp = openai.parse_response(data)
    assert resp.message.content == ""
    assert resp.message.tool_calls[0].arguments == {"command": "ls"}
    assert resp.stop_reason == "tool_use"  # mapped from "tool_calls"
    assert resp.usage.cache_read_tokens == 128


def test_openai_malformed_arguments_become_empty_dict():
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{not valid json"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    resp = openai.parse_response(data)
    # Empty args let the registry's schema validation surface the error (design §5.2).
    assert resp.message.tool_calls[0].arguments == {}


def test_openai_stop_reason_mapping():
    def _resp(reason):
        return openai.parse_response(
            {"choices": [{"message": {"content": "hi"}, "finish_reason": reason}]}
        )

    assert _resp("stop").stop_reason == "end_turn"
    assert _resp("length").stop_reason == "max_tokens"
    assert _resp("content_filter").stop_reason == "content_filter"  # passthrough
