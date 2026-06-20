"""Reasoning-effort run parameter (design §18.5): the Anthropic thinking mapping,
the per-provider payload wiring, and end-to-end plumbing from the profile."""

from __future__ import annotations

from agentkernel.profiles import Profile
from agentkernel.providers.anthropic import AnthropicProvider, thinking_config
from agentkernel.providers.local import LocalProvider
from agentkernel.providers.openai import OpenAIProvider
from agentkernel.types import Message
from tests.conftest import build_agent
from tests.fakes import FakeProvider, text_response

# --- Anthropic thinking mapping ----------------------------------------------

def test_thinking_config_levels():
    assert thinking_config("low", 8000)["budget_tokens"] == 1024
    assert thinking_config("medium", 8000)["budget_tokens"] == 4096
    assert thinking_config("high", 8000)["budget_tokens"] == 6976  # capped to max-1024


def test_thinking_config_none_and_too_small():
    assert thinking_config(None, 8000) is None
    assert thinking_config("high", 1500) is None  # no headroom -> skip rather than error


# --- payload wiring (transport monkeypatched, offline) -----------------------

def _capture(monkeypatch, module, response):
    box = {}

    def fake_pooled(url, *, header_for_key, payload, pool, **kw):
        box["payload"] = payload
        return response

    monkeypatch.setattr(f"agentkernel.providers.{module}.post_json_pooled", fake_pooled)
    return box


_OPENAI_RESP = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}
_ANTHROPIC_RESP = {
    "content": [{"type": "text", "text": "hi"}], "usage": {}, "stop_reason": "end_turn",
}


def test_openai_sends_reasoning_effort(monkeypatch):
    box = _capture(monkeypatch, "openai", _OPENAI_RESP)
    OpenAIProvider("gpt-5", api_key="k").complete(
        [Message(role="user", content="hi")], [], max_tokens=100, reasoning="high"
    )
    assert box["payload"]["reasoning_effort"] == "high"


def test_openai_omits_reasoning_when_absent(monkeypatch):
    box = _capture(monkeypatch, "openai", _OPENAI_RESP)
    OpenAIProvider("gpt-4o", api_key="k").complete(
        [Message(role="user", content="hi")], [], max_tokens=100
    )
    assert "reasoning_effort" not in box["payload"]


def test_local_never_sends_reasoning(monkeypatch):
    box = _capture(monkeypatch, "openai", _OPENAI_RESP)  # Local inherits OpenAI.complete
    LocalProvider("m", api_key="k").complete(
        [Message(role="user", content="hi")], [], max_tokens=100, reasoning="high"
    )
    assert "reasoning_effort" not in box["payload"]


def test_anthropic_sends_thinking(monkeypatch):
    box = _capture(monkeypatch, "anthropic", _ANTHROPIC_RESP)
    AnthropicProvider("claude-sonnet-4-6", api_key="k").complete(
        [Message(role="user", content="hi")], [], max_tokens=8000, reasoning="high"
    )
    assert box["payload"]["thinking"]["type"] == "enabled"
    assert box["payload"]["temperature"] == 1.0  # thinking forces temperature 1


# --- profile -> agent -> provider plumbing -----------------------------------

def test_profile_reasoning_reaches_provider():
    provider = FakeProvider([text_response("done")])
    agent = build_agent(provider)
    agent.run("go", profile=Profile(name="deep", reasoning="high"))
    assert provider.reasoning_args == ["high"]


def test_no_profile_means_no_reasoning():
    provider = FakeProvider([text_response("done")])
    agent = build_agent(provider)
    agent.run("go")
    assert provider.reasoning_args == [None]
