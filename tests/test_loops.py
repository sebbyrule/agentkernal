"""Loop-engineering tests (design §13, Phase 4+)."""

from __future__ import annotations

from pathlib import Path

from agentkernel.loops import Loop, LoopRunner, load_loop, loop_from_skill
from agentkernel.skills import SkillLibrary
from tests.conftest import build_agent
from tests.fakes import FakeProvider, text_response


def test_loop_stops_on_success_streak_without_check():
    provider = FakeProvider(
        [text_response("a"), text_response("b"), text_response("c")]
    )
    runner = LoopRunner(lambda: build_agent(provider))
    loop = Loop(
        name="echo",
        prompt="say something",
        max_iterations=5,
        success_streak=2,
    )
    result = runner.run(loop)
    assert result.succeeded is True
    assert result.count == 2
    assert result.iterations[0].answer == "a"


def test_loop_resets_streak_when_check_fails():
    provider = FakeProvider(
        [text_response("n"), text_response("n"), text_response("y")]
    )

    class _ToggleSandbox:
        def __init__(self):
            self._i = 0

        def run(self, cmd, *, cwd, timeout):
            self._i += 1
            return (0 if self._i >= 3 else 1), "", ""

    runner = LoopRunner(
        lambda: build_agent(provider),
        sandbox=_ToggleSandbox(),
    )
    loop = Loop(
        name="fix",
        prompt="fix it",
        max_iterations=5,
        success_check="./check.sh",
        success_streak=1,
    )
    result = runner.run(loop)
    assert result.succeeded is True
    assert result.count == 3
    assert result.iterations[0].check_passed is False
    assert result.iterations[-1].check_passed is True


def test_loop_exhausts_iterations_without_success():
    provider = FakeProvider([text_response("x")] * 5)
    runner = LoopRunner(lambda: build_agent(provider))
    loop = Loop(
        name="never",
        prompt="never succeed",
        max_iterations=3,
        success_streak=5,
    )
    result = runner.run(loop)
    assert result.succeeded is False
    assert result.count == 3


def test_load_loop_from_toml(tmp_path):
    toml = tmp_path / "loop.toml"
    toml.write_text(
        '''
name = "ci"
prompt = "Keep fixing until tests pass."
max_iterations = 10
success_check = "uv run pytest -q"
success_streak = 1
'''.strip(),
        encoding="utf-8",
    )
    loop = load_loop(toml)
    assert loop.name == "ci"
    assert "tests pass" in loop.prompt
    assert loop.max_iterations == 10
    assert loop.success_check == "uv run pytest -q"


def test_loop_from_skill_uses_skill_body(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "terse").mkdir()
    (skill_dir / "terse" / "SKILL.md").write_text(
        "---\nname: terse\ndescription: Be brief\n---\nAnswer in one word.",
        encoding="utf-8",
    )
    loop = loop_from_skill(SkillLibrary(skill_dir), "terse")
    assert loop is not None
    assert loop.name == "terse"
    assert loop.prompt == "Answer in one word."


def test_loop_from_skill_missing_returns_none(tmp_path):
    lib = SkillLibrary(tmp_path)
    assert loop_from_skill(lib, "missing") is None
