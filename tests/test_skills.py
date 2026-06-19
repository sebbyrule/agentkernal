"""Skills / AGENTS.md loading tests (Phase 4, design §13).

Skills are reusable system-prompt fragments discovered from files. The kernel
sees them only as context additions.
"""

from __future__ import annotations

from pathlib import Path

from agentkernel.skills import DirectorySkillStore
from tests.fakes import FakeProvider, text_response


def test_skill_store_loads_markdown_body_as_prompt(tmp_path):
    (tmp_path / "reviewer.md").write_text("Always review code carefully.")
    store = DirectorySkillStore(tmp_path)
    skill = store.get("reviewer")
    assert skill is not None
    assert skill.system_prompt == "Always review code carefully."


def test_skill_store_loads_toml_frontmatter_from_markdown(tmp_path):
    (tmp_path / "coder.md").write_text(
        "---\n"
        "name = \"pythonista\"\n"
        "---\n\n"
        "Write idiomatic Python."
    )
    store = DirectorySkillStore(tmp_path, active_skills=["pythonista"])
    assert "pythonista" in store.available_skills()
    assert store.system_additions() == ["Write idiomatic Python."]


def test_skill_store_loads_toml_file(tmp_path):
    (tmp_path / "tester.toml").write_text(
        'name = "qa"\n'
        'system_prompt = "Never trust a unit test."\n'
    )
    store = DirectorySkillStore(tmp_path, active_skills=["qa"])
    assert store.system_additions() == ["Never trust a unit test."]


def test_skill_store_returns_no_additions_when_nothing_active(tmp_path):
    (tmp_path / "a.md").write_text("be concise")
    store = DirectorySkillStore(tmp_path)
    assert store.system_additions() == []


def test_skill_store_toggle_activation(tmp_path):
    (tmp_path / "s.md").write_text("S")
    store = DirectorySkillStore(tmp_path)
    assert store.activate("s") is True
    assert store.system_additions() == ["S"]
    assert store.activate("s") is False
    assert store.system_additions() == []


def test_agent_applies_skill_context_source(agent_builder):
    store = DirectorySkillStore.__new__(DirectorySkillStore)
    from agentkernel.skills import Skill
    store._skills = {"concise": Skill(name="concise", system_prompt="Reply in one word.")}
    store.active_skills = {"concise"}
    provider = FakeProvider([text_response("hi")])
    agent = agent_builder(provider)
    agent.context_source = store
    agent.run("hello")
    assert provider.system_args[0] == "Reply in one word."


def test_agent_combines_profile_and_skill_prompts(agent_builder):
    from agentkernel.profiles import Profile
    from agentkernel.skills import Skill

    store = DirectorySkillStore.__new__(DirectorySkillStore)
    store._skills = {"skill1": Skill(name="skill1", system_prompt="Skill text.")}
    store.active_skills = {"skill1"}
    provider = FakeProvider([text_response("ok")])
    agent = agent_builder(provider)
    agent.context_source = store
    profile = Profile(name="test", system_prompt="Profile text.")
    agent.run("go", profile=profile)
    system = provider.system_args[0]
    assert "Profile text." in system
    assert "Skill text." in system
