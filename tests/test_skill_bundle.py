"""Shareable skill bundles: pack / install round-trip (design §18.8).

A bundle is a zip with SKILL.md (and any resources) at the root — the same shape
SkillLibrary discovers, so a packed skill installs straight back into a library.
"""

from __future__ import annotations

import zipfile

import pytest

from agentkernel.skills import (
    SKILL_BUNDLE_SUFFIX,
    SkillLibrary,
    install_skill,
    pack_skill,
)


def _folder_skill(skills_dir, name="code-review"):
    d = skills_dir / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: code-review\ndescription: Review a diff\n---\nDo the review.",
        encoding="utf-8",
    )
    (d / "checklist.md").write_text("- correctness\n- tests", encoding="utf-8")
    return d


def test_pack_folder_skill_round_trips(tmp_path):
    src = tmp_path / "skills"
    _folder_skill(src)
    archive = pack_skill(src, "code-review", out_path=tmp_path / "cr.skill.zip")
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert names == {"SKILL.md", "checklist.md"}

    # Install into a fresh library and confirm discovery + the resource.
    dest = tmp_path / "other-skills"
    dest.mkdir()
    name = install_skill(archive, dest)
    assert name == "code-review"
    assert (dest / "code-review" / "checklist.md").is_file()
    assert SkillLibrary(dest).get("code-review").description == "Review a diff"


def test_pack_default_output_name(tmp_path, monkeypatch):
    src = tmp_path / "skills"
    _folder_skill(src, "triage")
    monkeypatch.chdir(tmp_path)  # default out path is relative to cwd
    archive = pack_skill(src, "triage")
    assert archive.name == f"triage{SKILL_BUNDLE_SUFFIX}"


def test_pack_loose_md_skill(tmp_path):
    src = tmp_path / "skills"
    src.mkdir()
    (src / "summarize.md").write_text(
        "---\nname: summarize\ndescription: TL;DR\n---\nBe concise.", encoding="utf-8"
    )
    archive = pack_skill(src, "summarize", out_path=tmp_path / "s.skill.zip")
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["SKILL.md"]


def test_pack_unknown_skill_raises(tmp_path):
    (tmp_path / "skills").mkdir()
    with pytest.raises(FileNotFoundError):
        pack_skill(tmp_path / "skills", "ghost")


def test_install_rejects_existing_without_force(tmp_path):
    src = tmp_path / "skills"
    _folder_skill(src)
    archive = pack_skill(src, "code-review", out_path=tmp_path / "cr.skill.zip")
    dest = tmp_path / "lib"
    install_skill(archive, dest)
    with pytest.raises(FileExistsError):
        install_skill(archive, dest)
    # --force overwrites.
    assert install_skill(archive, dest, force=True) == "code-review"


def test_install_rejects_bundle_without_skill_md(tmp_path):
    bad = tmp_path / "bad.skill.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("notes.txt", "hi")
    with pytest.raises(ValueError, match="not a skill bundle"):
        install_skill(bad, tmp_path / "lib")


def test_run_skill_cli_pack_list_install(tmp_path):
    from agentkernel.cli import run_skill
    from agentkernel.config import Config

    src = tmp_path / "skills"
    _folder_skill(src)
    cfg = Config(skills_dir=str(src))
    out: list[str] = []

    assert run_skill(cfg, "list", None, output_fn=out.append) == 0
    assert any("code-review" in line for line in out)

    archive = tmp_path / "cr.skill.zip"
    assert run_skill(cfg, "pack", "code-review", out_path=str(archive), output_fn=print) == 0
    assert archive.is_file()

    dest = tmp_path / "lib"
    cfg2 = Config(skills_dir=str(dest))
    assert run_skill(cfg2, "install", str(archive), output_fn=print) == 0
    assert (dest / "code-review" / "SKILL.md").is_file()

    # Unknown action is a clean error, not a crash.
    assert run_skill(cfg, "frobnicate", None, output_fn=out.append) == 1


def test_install_rejects_zip_slip(tmp_path):
    evil = tmp_path / "evil.skill.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("SKILL.md", "x")
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(ValueError, match="unsafe path"):
        install_skill(evil, tmp_path / "lib")
