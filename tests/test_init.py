"""Tests for `agentkernel init` (scaffold a starter config)."""

from __future__ import annotations

from agentkernel.cli import run_init


def test_init_creates_project_config(tmp_path):
    out: list[str] = []
    assert run_init(target_dir=str(tmp_path), output_fn=out.append) == 0
    cfg = tmp_path / "agentkernel.toml"
    assert cfg.is_file()
    assert "provider" in cfg.read_text(encoding="utf-8")
    assert any("created" in line for line in out)


def test_init_refuses_overwrite_without_force(tmp_path):
    (tmp_path / "agentkernel.toml").write_text("existing", encoding="utf-8")
    out: list[str] = []
    assert run_init(target_dir=str(tmp_path), output_fn=out.append) == 1
    assert any("exists" in line for line in out)
    assert (tmp_path / "agentkernel.toml").read_text(encoding="utf-8") == "existing"


def test_init_force_overwrites(tmp_path):
    (tmp_path / "agentkernel.toml").write_text("old", encoding="utf-8")
    assert run_init(target_dir=str(tmp_path), force=True, output_fn=lambda _l: None) == 0
    assert "provider" in (tmp_path / "agentkernel.toml").read_text(encoding="utf-8")


def test_init_global_writes_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTKERNEL_HOME", str(tmp_path / "home"))
    out: list[str] = []
    assert run_init(global_config=True, output_fn=out.append) == 0
    assert (tmp_path / "home" / "config.toml").is_file()
