"""Profile tests (Phase 5, design §13).

Profiles let a run be parameterized by system prompt, tool filter, model
override, and rubric. The kernel currently applies system_prompt and
tool_filter; model_override and rubric are stored as future seams.
"""

from __future__ import annotations

from agentkernel.profiles import Profile, load_profile
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolCall, ToolResult
from tests.fakes import FakeProvider, text_response


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=lambda args: ToolResult("", f"echo:{args['value']}"),
    )


def test_profile_sets_system_prompt(agent_builder):
    provider = FakeProvider([text_response("ok")])
    agent = agent_builder(provider)
    profile = Profile(name="coder", system_prompt="You are a precise coder.")
    agent.run("go", profile=profile)
    assert provider.system_args == ["You are a precise coder."]


def test_profile_filters_tools(agent_builder):
    registry = ToolRegistry()
    registry.register(_echo_tool())
    provider = FakeProvider([text_response("no tools needed")])
    agent = agent_builder(provider, registry)
    profile = Profile(name="limited", tool_filter=[])
    # With an empty tool_filter, no tools are offered to the provider.
    agent.run("go", profile=profile)
    assert agent.registry.specs()  # registry still has echo
    assert provider.tool_args[0] == []


def test_load_profile_from_toml(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "reviewer.toml").write_text(
        'system_prompt = "Review the code carefully."\n'
        'tool_filter = ["read_file"]\n'
        'model_override = "claude-haiku-4-5"\n'
        'rubric = "Correctness, clarity, performance."\n'
    )
    profile = load_profile("reviewer", search_dirs=[profile_dir])
    assert profile is not None
    assert profile.name == "reviewer"
    assert profile.system_prompt == "Review the code carefully."
    assert profile.tool_filter == ["read_file"]
    assert profile.model_override == "claude-haiku-4-5"
    assert "Correctness" in profile.rubric


def test_load_profile_missing_returns_none(tmp_path):
    assert load_profile("missing", search_dirs=[tmp_path]) is None
