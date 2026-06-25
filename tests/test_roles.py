"""Auxiliary model-role router (design §18.5).

Centralizes the ``<role>_model or summarizer_model or model`` fallback chains
that used to be duplicated at each call site (compaction, risk classifier,
curation, eval judge).
"""

from __future__ import annotations

from agentkernel.config import Config
from agentkernel.roles import ROLE_FALLBACKS, provider_for_role, role_model


def test_every_role_falls_back_to_main_model_when_unset():
    cfg = Config(model="main-model")
    for role in ROLE_FALLBACKS:
        assert role_model(cfg, role) == "main-model"


def test_summarize_role_prefers_summarizer_model():
    cfg = Config(model="main", summarizer_model="cheap")
    assert role_model(cfg, "summarize") == "cheap"


def test_classify_role_chain():
    # approval_judge_model wins; else summarizer_model; else model.
    assert role_model(Config(model="m"), "classify") == "m"
    assert role_model(Config(model="m", summarizer_model="s"), "classify") == "s"
    assert (
        role_model(Config(model="m", summarizer_model="s", approval_judge_model="j"), "classify")
        == "j"
    )


def test_curate_role_chain():
    assert role_model(Config(model="m", summarizer_model="s"), "curate") == "s"
    assert (
        role_model(Config(model="m", summarizer_model="s", memory_curator_model="c"), "curate")
        == "c"
    )


def test_judge_role_uses_judge_model_then_main():
    assert role_model(Config(model="m"), "judge") == "m"
    assert role_model(Config(model="m", judge_model="jm"), "judge") == "jm"


def test_provider_for_role_binds_resolved_model():
    cfg = Config(provider="anthropic", model="main", summarizer_model="cheap")
    provider = provider_for_role(cfg, "summarize")
    assert provider.model == "cheap"
    # A role that falls through uses the main model.
    assert provider_for_role(cfg, "judge").model == "main"
