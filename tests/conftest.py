"""Shared fixtures and builders for the offline test suite."""

from __future__ import annotations

import pytest

from agentkernel.agent import Agent
from agentkernel.approval import AutoApprover
from agentkernel.config import Config
from agentkernel.context import ContextManager
from agentkernel.telemetry import NullTelemetry
from agentkernel.tools import ToolRegistry


def build_agent(
    provider, registry=None, *, config=None, approver=None, context=None, memory=None
) -> Agent:
    """Wire an Agent with default offline collaborators around ``provider``."""
    config = config or Config()
    if context is None:
        budget = provider.context_window - config.output_reserve
        context = ContextManager(
            budget=budget, keep_recent_turns=config.keep_recent_turns
        )
    return Agent(
        provider=provider,
        registry=registry or ToolRegistry(),
        context=context,
        approver=approver or AutoApprover(),
        telemetry=NullTelemetry(),
        config=config,
        memory=memory,
    )


@pytest.fixture
def agent_builder():
    return build_agent
