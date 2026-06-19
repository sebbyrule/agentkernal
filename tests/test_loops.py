"""Loop-engineering tests (action -> check -> iterate -> stop). Offline: a fake
sandbox scripts the success-check exit codes; a shared FakeProvider drives the
per-iteration agent answers."""

from __future__ import annotations

from agentkernel.loops import Loop, LoopRunner, load_loop, loop_from_skill
from agentkernel.skills import SkillLibrary

from tests.conftest import build_agent
from tests.fakes import FakeProvider, text_response


class _FakeSandbox:
    def __init__(self, codes):
        self._codes = list(codes)
        self.calls: list[str] = []

    def run(self, command, *, cwd, timeout):
        self.calls.append(command)
        return (self._codes.pop(0) if self._codes else 0), "", ""

    def close(self):
        pass


def _factory(provider):
    return lambda: build_agent(provider)


def test_loop_stops_when_check_passes():
    provider = FakeProvider([text_response("a"), text_response("b"), text_response("c")])
    sandbox = _FakeSandbox([1, 1, 0])  # fail, fail, pass
    runner = LoopRunner(_factory(provider), sandbox=sandbox)
    loop = Loop("fix", "fix the build", success_check="pytest", success_streak=1, max_iterations=5)
    result = runner.run(loop)
    assert result.succeeded and result.count == 3
    assert sandbox.calls == ["pytest", "pytest", "pytest"]


def test_loop_streak_requires_consecutive_passes():
    provider = FakeProvider([text_response("x")] * 5)
    sandbox = _FakeSandbox([0, 1, 0, 0, 0])  # pass, fail(reset), pass, pass, pass
    runner = LoopRunner(_factory(provider), sandbox=sandbox)
    loop = Loop("streak", "work", success_check="t", success_streak=3, max_iterations=5)
    result = runner.run(loop)
    assert result.succeeded and result.count == 5  # only hits 3-in-a-row on the last


def test_loop_stops_without_success_after_max():
    provider = FakeProvider([text_response("x")] * 3)
    sandbox = _FakeSandbox([1, 1, 1])
    runner = LoopRunner(_factory(provider), sandbox=sandbox)
    loop = Loop("nope", "w", success_check="t", success_streak=1, max_iterations=3)
    result = runner.run(loop)
    assert not result.succeeded and result.count == 3


def test_loop_without_check_counts_iterations_toward_streak():
    provider = FakeProvider([text_response("x")] * 5)
    runner = LoopRunner(_factory(provider))  # no sandbox, no check
    loop = Loop("n", "w", success_check=None, success_streak=2, max_iterations=5)
    result = runner.run(loop)
    assert result.succeeded and result.count == 2


def test_load_loop_toml(tmp_path):
    f = tmp_path / "loop.toml"
    f.write_text(
        'name = "docs"\nprompt = "sweep the docs"\nmax_iterations = 3\n'
        'success_check = "pytest -q"\nsuccess_streak = 2\n'
    )
    loop = load_loop(f)
    assert loop.name == "docs" and loop.prompt == "sweep the docs"
    assert loop.max_iterations == 3 and loop.success_streak == 2
    assert loop.success_check == "pytest -q"


def test_loop_from_skill(tmp_path):
    (tmp_path / "sweep.md").write_text("Sweep the docs and open a PR.")
    lib = SkillLibrary(tmp_path)
    loop = loop_from_skill(lib, "sweep", max_iterations=2)
    assert loop is not None
    assert loop.prompt == "Sweep the docs and open a PR." and loop.max_iterations == 2
    assert loop_from_skill(lib, "missing") is None
