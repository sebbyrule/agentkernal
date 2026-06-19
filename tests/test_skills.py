"""Skills tests (Anthropic-style, design §13 Phase 4).

Skills are SKILL.md folders (or loose .md/.toml), surfaced progressively: a
name+description catalog is always in the prefix, full bodies load on demand via
the use_skill tool, and a pinned (active) skill's body also joins the prefix.
"""

from __future__ import annotations

from agentkernel.skills import SkillLibrary, _split_frontmatter, make_skill_tool


def _skill_md(directory, name, desc, body):
    d = directory / name
    d.mkdir()
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\n{body}", encoding="utf-8"
    )
    return d


def test_loads_skill_md_folder_with_resources(tmp_path):
    d = _skill_md(tmp_path, "pdf", "Work with PDF files", "Use pdfplumber to extract text.")
    (d / "helper.py").write_text("print('x')")
    lib = SkillLibrary(tmp_path)
    skill = lib.get("pdf")
    assert skill.description == "Work with PDF files"
    assert "pdfplumber" in skill.body
    assert any(r.endswith("helper.py") for r in skill.resources)


def test_loads_loose_md_and_toml(tmp_path):
    (tmp_path / "a.md").write_text("Just instructions, no frontmatter.")
    (tmp_path / "b.toml").write_text(
        'name = "qa"\ndescription = "QA skill"\nsystem_prompt = "Never trust a test."\n'
    )
    lib = SkillLibrary(tmp_path)
    assert lib.get("a").body == "Just instructions, no frontmatter."
    assert lib.get("qa").body == "Never trust a test."


def test_split_frontmatter_yaml_subset():
    meta, body = _split_frontmatter(
        '---\nname: x\ndescription: "hi there"\ntools:\n  - read\n  - write\n---\nBody here'
    )
    assert meta["name"] == "x" and meta["description"] == "hi there"
    assert meta["tools"] == ["read", "write"]
    assert body == "Body here"
    assert _split_frontmatter("no frontmatter") == ({}, "no frontmatter")


def test_catalog_always_present_and_pin_adds_body(tmp_path):
    _skill_md(tmp_path, "terse", "Be brief", "Answer in one word.")
    lib = SkillLibrary(tmp_path)

    adds = lib.system_additions()
    assert adds and adds[0].startswith("# Available skills")
    assert "terse: Be brief" in adds[0]  # catalog discloses name + description
    assert all("Answer in one word." not in a for a in adds[1:])  # body not yet shown

    lib.activate("terse")
    assert any(a == "Answer in one word." for a in lib.system_additions())


def test_no_skills_means_no_additions(tmp_path):
    assert SkillLibrary(tmp_path).system_additions() == []


def test_use_returns_body_and_resources(tmp_path):
    d = _skill_md(tmp_path, "pdf", "PDF", "Extract with pdfplumber.")
    (d / "reference.md").write_text("ref")
    lib = SkillLibrary(tmp_path)
    out = lib.use("pdf")
    assert "Extract with pdfplumber." in out and "reference.md" in out
    assert "Unknown skill" in lib.use("nope")


def test_use_skill_tool(tmp_path):
    _skill_md(tmp_path, "x", "X skill", "Do the X thing.")
    tool = make_skill_tool(SkillLibrary(tmp_path))
    ok = tool.handler({"name": "x"})
    assert not ok.is_error and "Do the X thing." in ok.content
    assert tool.handler({"name": "missing"}).is_error


def test_activate_toggle(tmp_path):
    _skill_md(tmp_path, "s", "S desc", "S body")
    lib = SkillLibrary(tmp_path)
    assert lib.activate("s") is True
    assert lib.activate("s") is False
    assert lib.activate("unknown") is False
