"""Provider abstraction and adapters (design §5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentkernel.providers._http import ProviderError
from agentkernel.providers.anthropic import AnthropicProvider
from agentkernel.providers.base import Provider
from agentkernel.providers.local import LocalProvider
from agentkernel.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from agentkernel.config import Config

__all__ = [
    "Provider",
    "ProviderError",
    "AnthropicProvider",
    "OpenAIProvider",
    "LocalProvider",
    "make_provider",
]


def make_provider(config: "Config") -> Provider:
    """Construct the adapter named by ``config.provider``. Keys come from env."""
    if config.provider == "anthropic":
        return AnthropicProvider(config.model)
    if config.provider == "openai":
        return OpenAIProvider(config.model)
    if config.provider == "local":
        kwargs = {} if config.base_url is None else {"base_url": config.base_url}
        return LocalProvider(config.model, **kwargs)
    raise ProviderError(f"unknown provider: {config.provider!r}")
