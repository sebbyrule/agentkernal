"""Named adapters for OpenAI-compatible hosted providers (design §18.5).

OpenRouter, DeepSeek, and Google Gemini all expose an OpenAI Chat Completions
shape, so each is a thin ``OpenAIProvider`` subclass that only pins a default
``base_url``, env var, and capability — no new wire translation. The kernel's
generic ``local`` adapter could reach these too; these named variants exist so a
user can write ``provider = "openrouter"`` with sensible defaults instead of
hand-wiring ``base_url`` and the key env var.

Keys come from the environment only (never config/traces), like every adapter.
"""

from __future__ import annotations

from agentkernel.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter (https://openrouter.ai) — routes to many models via one key."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        context_window: int = 128_000,
        supports_images: bool = True,
    ) -> None:
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            context_window=context_window,
            name="openrouter",
            env_key="OPENROUTER_API_KEY",
            send_reasoning=False,  # routed models vary; don't assume reasoning_effort
            supports_images=supports_images,
        )


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek (https://api.deepseek.com) — OpenAI-compatible chat models."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.deepseek.com/v1",
        api_key: str | None = None,
        context_window: int = 64_000,
        supports_images: bool = False,  # deepseek-chat/-reasoner are text-only
    ) -> None:
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            context_window=context_window,
            name="deepseek",
            env_key="DEEPSEEK_API_KEY",
            send_reasoning=False,
            supports_images=supports_images,
        )


class GeminiProvider(OpenAIProvider):
    """Google Gemini via its OpenAI-compatibility endpoint (multimodal)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        api_key: str | None = None,
        context_window: int = 1_000_000,
        supports_images: bool = True,
    ) -> None:
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            context_window=context_window,
            name="gemini",
            env_key="GEMINI_API_KEY",
            send_reasoning=False,
            supports_images=supports_images,
        )
