"""Tests for `agentkernel new` scaffolding (run_new) and that the bundled
skills / profiles / loops actually load through their real loaders."""

from __future__ import annotations

from pathlib import Path

from agentkernel.cli import _find_templates_dir, run_new

_REPO = Path(__file__).resolve().parent.parent


def _templates() -> Path:
    return _REPO / "templates"


def test_new_loop_substitutes_name(tmp_path):
    out: list[str] = []
    rc = run_new(
        "loop", "my-loop", project_root=tmp_path, templates_dir=_templates(),
        output_fn=out.append,
    )
    assert rc == 0
    dest = tmp_path / "loops" / "my-loop.toml"
    assert dest.is_file()
    assert 'name = "my-loop"' in dest.read_text(encoding="utf-8")


def test_new_skill_creates_bundle_dir(tmp_path):
    rc = run_new("skill", "my-skill", project_root=tmp_path, templates_dir=_templates(),
                 output_fn=lambda _m: None)
    assert rc == 0
    dest = tmp_path / "skills" / "my-skill" / "SKILL.md"
    assert dest.is_file()
    assert "name: my-skill" in dest.read_text(encoding="utf-8")


def test_new_profile_and_eval(tmp_path):
    assert run_new("profile", "p1", project_root=tmp_path, templates_dir=_templates(),
                   output_fn=lambda _m: None) == 0
    assert (tmp_path / "profiles" / "p1.toml").is_file()
    assert run_new("eval", "e1", project_root=tmp_path, templates_dir=_templates(),
                   output_fn=lambda _m: None) == 0
    assert (tmp_path / "evals" / "e1.toml").is_file()


def test_new_rejects_bad_names(tmp_path):
    for bad in ("../evil", "a/b", ".hidden"):
        out: list[str] = []
        rc = run_new("loop", bad, project_root=tmp_path, templates_dir=_templates(),
                     output_fn=out.append)
        assert rc == 1
        assert "invalid name" in out[0]


def test_new_refuses_overwrite_without_force(tmp_path):
    kw = dict(project_root=tmp_path, templates_dir=_templates(), output_fn=lambda _m: None)
    assert run_new("loop", "dup", **kw) == 0
    assert run_new("loop", "dup", **kw) == 1  # exists
    out: list[str] = []
    assert run_new("loop", "dup", project_root=tmp_path, templates_dir=_templates(),
                   force=True, output_fn=out.append) == 0
    assert "created" in out[0]


def test_unknown_kind_is_rejected(tmp_path):
    rc = run_new("widget", "x", project_root=tmp_path, templates_dir=_templates(),
                 output_fn=lambda _m: None)
    assert rc == 1


def test_find_templates_dir_walks_up():
    # From inside the repo, the templates/ dir is discoverable.
    assert _find_templates_dir(_REPO / "agentkernel") == _REPO / "templates"


# --- the bundled libraries actually load -------------------------------------

def test_bundled_skills_load():
    from agentkernel.skills import SkillLibrary

    lib = SkillLibrary(_REPO / "skills")
    names = {s.name for s in lib._skills.values()}
    assert {"code-review", "debug-triage", "write-tests", "refactor"} <= names
    for skill in lib._skills.values():
        assert skill.description.strip()  # every skill has a trigger description


def test_bundled_profiles_load():
    from agentkernel.profiles import load_profile

    for name in ("reviewer", "researcher", "coder", "planner", "safe"):
        prof = load_profile(name, search_dirs=[_REPO / "profiles"])
        assert prof is not None and prof.system_prompt
    # planner has no tools; reviewer is read-only.
    assert load_profile("planner", search_dirs=[_REPO / "profiles"]).tool_filter == []
    assert "write_file" not in (
        load_profile("reviewer", search_dirs=[_REPO / "profiles"]).tool_filter or []
    )


def test_bundled_loops_load():
    from agentkernel.loops import load_loop

    for f in (_REPO / "loops").glob("*.toml"):
        loop = load_loop(f)
        assert loop.prompt.strip()
        assert loop.max_iterations >= 1
