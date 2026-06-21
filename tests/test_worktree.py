"""Worktree isolation tests (design §18.3): the WorktreeManager (injected runner
+ a real-git integration test) and spawn's worktree=true path."""

from __future__ import annotations

import subprocess

import pytest

from agentkernel.approval import AutoApprover
from agentkernel.config import Config
from agentkernel.subagent import make_spawn_tool
from agentkernel.tools import ToolRegistry
from agentkernel.types import ToolCall
from agentkernel.worktree import WorktreeManager, git_available
from tests.fakes import FakeProvider, text_response

# --- WorktreeManager command construction (offline) --------------------------

class _RecordRunner:
    def __init__(self, results=None):
        self.calls: list[list[str]] = []
        self._results = results or {}

    def __call__(self, args):
        self.calls.append(args)
        # match on the first arg or a tuple key
        for key, value in self._results.items():
            if args[: len(key)] == list(key):
                return value
        return (0, "", "")


def test_is_git_repo_parses_output():
    runner = _RecordRunner({("rev-parse",): (0, "true\n", "")})
    assert WorktreeManager(".", runner=runner).is_git_repo() is True

    runner2 = _RecordRunner({("rev-parse",): (128, "", "not a repo")})
    assert WorktreeManager(".", runner=runner2).is_git_repo() is False


def test_create_builds_worktree_add_command():
    runner = _RecordRunner()
    path, branch = WorktreeManager(".", runner=runner).create()
    add = runner.calls[0]
    assert add[:3] == ["worktree", "add", "-b"]
    assert branch in add and str(path) in add


def test_create_raises_on_failure():
    from agentkernel.worktree import WorktreeError

    runner = _RecordRunner({("worktree", "add"): (1, "", "fatal: boom")})
    with pytest.raises(WorktreeError):
        WorktreeManager(".", runner=runner).create()


def test_has_changes_and_remove_commands():
    runner = _RecordRunner({("-C",): (0, " M file.py\n", "")})
    wm = WorktreeManager(".", runner=runner)
    assert wm.has_changes("/tmp/wt") is True
    wm.remove("/tmp/wt")
    assert runner.calls[-1][:2] == ["worktree", "remove"]


# --- real git integration -----------------------------------------------------

@pytest.mark.skipif(not git_available(), reason="git not installed")
def test_real_worktree_lifecycle(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "a.txt").write_text("hello", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")

    wm = WorktreeManager(str(repo))
    assert wm.is_git_repo()
    path, branch = wm.create()
    try:
        assert path.is_dir() and (path / "a.txt").is_file()
        assert wm.has_changes(path) is False
        (path / "new.txt").write_text("x", encoding="utf-8")
        assert wm.has_changes(path) is True
    finally:
        wm.remove(path)
    assert not path.exists()


# --- spawn worktree=true path -------------------------------------------------

def test_spawn_worktree_uses_factory_and_removes_clean_tree(monkeypatch, tmp_path):
    wt_path = tmp_path / "wt"

    class _FakeWM:
        removed: list = []

        def __init__(self, repo):
            pass

        def is_git_repo(self):
            return True

        def create(self):
            return wt_path, "agentkernel/wt"

        def has_changes(self, path):
            return False  # clean -> should be removed

        def remove(self, path):
            _FakeWM.removed.append(str(path))

    monkeypatch.setattr("agentkernel.worktree.WorktreeManager", _FakeWM)

    factory_dirs: list[str] = []

    def factory(working_dir):
        factory_dirs.append(working_dir)
        return []  # child gets a minimal toolset

    provider = FakeProvider([text_response("child finished")])
    spawn = make_spawn_tool(
        provider=provider,
        base_specs=[],
        approver=AutoApprover("auto_allow"),
        config=Config(working_dir=str(tmp_path)),
        tool_factory=factory,
    )
    result = spawn.handler({"task": "edit code", "worktree": True})
    assert "child finished" in result.content
    assert "worktree removed" in result.content
    assert factory_dirs == [str(wt_path)]  # tools rebound to the worktree
    assert str(wt_path) in _FakeWM.removed


def test_spawn_worktree_without_factory_runs_normally():
    provider = FakeProvider([text_response("done")])
    registry = ToolRegistry()
    spawn = make_spawn_tool(
        provider=provider,
        base_specs=[],
        approver=AutoApprover("auto_allow"),
        config=Config(),
        tool_factory=None,  # no factory -> worktree request ignored, no error
    )
    registry.register(spawn)
    result = registry.execute(ToolCall("c", "spawn", {"task": "x", "worktree": True}))
    assert not result.is_error and "done" in result.content
