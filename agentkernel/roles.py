"""Auxiliary model roles (design §18.5).

Beside the main loop the kernel makes several cheaper/secondary model calls:
compaction **summaries**, the `smart`-approval risk **classifier**, memory
**curation**, and eval **judging**. Each may use a different (usually cheaper)
model. Rather than repeat the ``<role>_model or summarizer_model or model``
fallback at every call site, the chains live here as named roles, so adding or
retargeting an auxiliary model is a one-line change.

The per-role config fields (``summarizer_model``, ``approval_judge_model``,
``memory_curator_model``, ``judge_model``) are unchanged — this is the router
that reads them, not a new schema.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from agentkernel.providers import make_provider

if TYPE_CHECKING:
    from agentkernel.config import Config
    from agentkernel.providers import Provider

# role -> config fields tried in order before falling back to config.model.
ROLE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "summarize": ("summarizer_model",),
    "classify": ("approval_judge_model", "summarizer_model"),
    "curate": ("memory_curator_model", "summarizer_model"),
    "judge": ("judge_model",),
}


def role_model(config: Config, role: str) -> str:
    """The model id for an auxiliary ``role``, honoring its fallback chain.

    Always resolves to a usable model: an unset chain falls back to
    ``config.model``.
    """
    for field in ROLE_FALLBACKS[role]:
        value = getattr(config, field, None)
        if value:
            return value
    return config.model


def provider_with_model(config: Config, model: str) -> Provider:
    """A provider built from ``config`` but bound to ``model``."""
    return make_provider(replace(config, model=model))


def provider_for_role(config: Config, role: str) -> Provider:
    """A provider for an auxiliary ``role`` (its resolved model, same backend)."""
    return provider_with_model(config, role_model(config, role))
