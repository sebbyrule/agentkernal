"""Local / OpenAI-compatible endpoint adapter (design §5.2).

Same wire shape as OpenAI (Ollama, vLLM, LM Studio, …) with a configurable
``base_url`` and optional auth. No prompt caching is assumed, so the turn-2
cache check in M1 does not apply to this provider.
"""

from __future__ import annotations

from agentkernel.providers.openai import OpenAIProvider

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama default
DEFAULT_CONTEXT_WINDOW = 8192


class LocalProvider(OpenAIProvider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            context_window=context_window,
            name="local",
            require_key=False,  # local endpoints commonly need no key
            env_key="LOCAL_API_KEY",
        )
