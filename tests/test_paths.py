"""Tests for running the agent outside its project folder: home/project
resolution and the 'global brain, project sessions' path policy."""

from __future__ import annotations

from pathlib import Path

from agentkernel.config import resolve_config
from agentkernel.paths import agent_home, anchor_path, find_project_root


def test_agent_home_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTKERNEL_HOME", str(tmp_path / "home"))
    assert agent_home() == tmp_path / "home"
    monkeypatch.delenv("AGENTKERNEL_HOME")
    assert agent_home() == Path.home() / ".agentkernel"


def test_find_project_root_walks_up(tmp_path):
    root = tmp_path / "proj"
    (root / ".agentkernel").mkdir(parents=True)
    sub = root / "a" / "b"
    sub.mkdir(parents=True)
    assert find_project_root(sub) == root.resolve()


def test_find_project_root_falls_back_to_start(tmp_path):
    lonely = tmp_path / "nowhere"
    lonely.mkdir()
    assert find_project_root(lonely) == lonely.resolve()


def test_anchor_path_respects_absolute(tmp_path):
    assert anchor_path("rel/x", base=tmp_path) == str((tmp_path / "rel/x").resolve())
    abs_path = tmp_path / "abs"
    assert anchor_path(str(abs_path), base=Path("/other")) == str(abs_path)


def test_global_brain_project_sessions_split(monkeypatch, tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    (project / "agentkernel.toml").write_text("provider = 'local'\n", encoding="utf-8")
    monkeypatch.setenv("AGENTKERNEL_HOME", str(home))

    config, project_cfg = resolve_config(cwd=project, env={"AGENTKERNEL_HOME": str(home)})

    assert config.provider == "local"
    assert project_cfg == project / "agentkernel.toml"
    # working dir is the project root
    assert config.working_dir == str(project.resolve())
    # global brain -> under home
    assert config.memory_notes_path.startswith(str(home.resolve()))
    assert config.skills_dir.startswith(str(home.resolve()))
    assert config.graph_path.startswith(str(home.resolve()))
    # project sessions -> under the project
    assert config.log_dir.startswith(str(project.resolve()))
    assert config.kanban_path.startswith(str(project.resolve()))
    assert config.memory_dir.startswith(str(project.resolve()))


def test_global_config_layered_under_project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        "provider = 'local'\nmodel = 'global-model'\n", encoding="utf-8"
    )
    project = tmp_path / "p"
    project.mkdir()
    (project / "agentkernel.toml").write_text("model = 'project-model'\n", encoding="utf-8")
    env = {"AGENTKERNEL_HOME": str(home)}

    config, _ = resolve_config(cwd=project, env=env)
    assert config.provider == "local"  # from global
    assert config.model == "project-model"  # project overrides global


def test_customized_relative_path_anchors_to_project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "p"
    project.mkdir()
    # A project that customizes a normally-global path keeps it project-local.
    (project / "agentkernel.toml").write_text('skills_dir = "myskills"\n', encoding="utf-8")
    env = {"AGENTKERNEL_HOME": str(home)}
    config, _ = resolve_config(cwd=project, env=env)
    assert config.skills_dir == str((project / "myskills").resolve())


def test_explicit_config_arg_used_directly(tmp_path):
    cfg = tmp_path / "custom.toml"
    cfg.write_text("model = 'explicit'\n", encoding="utf-8")
    config, project_cfg = resolve_config(str(cfg), cwd=tmp_path, env={})
    assert config.model == "explicit"
    assert project_cfg == cfg
