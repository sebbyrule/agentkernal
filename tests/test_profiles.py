"""Profile loading and profile-scoped configuration tests (Phase 5)."""

from __future__ import annotations

from agentkernel.profiles import Profile, load_profile


def test_load_profile_reads_all_fields(tmp_path):
    path = tmp_path / "coder.toml"
    path.write_text(
        '''
name = "coder"
system_prompt = "You are a senior engineer."
tool_filter = ["read_file", "write_file"]
model_override = "claude-opus-4"
rubric = "Code compiles and has tests."
'''.strip(),
        encoding="utf-8",
    )
    profile = load_profile("coder", search_dirs=[tmp_path])
    assert isinstance(profile, Profile)
    assert profile.name == "coder"
    assert profile.system_prompt == "You are a senior engineer."
    assert profile.tool_filter == ["read_file", "write_file"]
    assert profile.model_override == "claude-opus-4"
    assert profile.rubric == "Code compiles and has tests."


def test_load_profile_returns_none_when_missing(tmp_path):
    assert load_profile("missing", search_dirs=[tmp_path]) is None
