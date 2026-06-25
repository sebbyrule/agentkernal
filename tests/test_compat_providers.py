"""Named OpenAI-compatible adapters: OpenRouter, DeepSeek, Gemini (§18.5).

These are thin OpenAIProvider subclasses, so the test pins their defaults and the
make_provider wiring rather than re-testing the shared wire translation.
"""

from __future__ import annotations

import pytest

from agentkernel.config import Config
from agentkernel.providers import (
    DeepSeekProvider,
    GeminiProvider,
    OpenRouterProvider,
    ProviderError,
    make_provider,
)


def test_openrouter_defaults():
    p = OpenRouterProvider("anthropic/claude-3.5-sonnet", api_key="k")
    assert p.name == "openrouter"
    assert p._base_url == "https://openrouter.ai/api/v1"
    assert p.supports_images is True


def test_deepseek_defaults_text_only():
    p = DeepSeekProvider("deepseek-chat", api_key="k")
    assert p.name == "deepseek"
    assert p._base_url == "https://api.deepseek.com/v1"
    assert p.supports_images is False


def test_gemini_defaults_multimodal():
    p = GeminiProvider("gemini-2.0-flash", api_key="k")
    assert p.name == "gemini"
    assert p._base_url.endswith("/v1beta/openai")
    assert p.supports_images is True
    assert p.context_window == 1_000_000


@pytest.mark.parametrize(
    ("provider", "expected_name"),
    [("openrouter", "openrouter"), ("deepseek", "deepseek"), ("gemini", "gemini")],
)
def test_make_provider_builds_named_compat_adapters(provider, expected_name, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = make_provider(Config(provider=provider, model="m"))
    assert p.name == expected_name


def test_make_provider_honors_base_url_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    p = make_provider(
        Config(provider="openrouter", model="m", base_url="http://proxy/v1")
    )
    assert p._base_url == "http://proxy/v1"


def test_make_provider_rejects_unknown():
    with pytest.raises(ProviderError):
        make_provider(Config(provider="nope", model="m"))
