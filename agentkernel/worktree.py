"""Git worktree isolation for sub-agents (design §18.3).

When a spawned child edits code, running it in a throwaway ``git worktree`` keeps
parallel children from colliding on the same files. This is a thin wrapper over
the ``git worktree`` CLI: create a worktree on a fresh branch, ask whether it has
changes, and remove it. No git library dependency — just the CLI behind an
injectable runner so the command construction is testable offline.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

Runner = Callable[[list[str]], "tuple[int, str, str]"]


class WorktreeError(RuntimeError):
    """A git worktree operation failed."""


def git_available() -> bool:
    return shutil.which("git") is not None


class WorktreeManager:
    """Create and clean up git worktrees for a repository."""

    def __init__(self, repo_dir: str = ".", *, runner: Runner | None = None) -> None:
        self._repo = Path(repo_dir).resolve()
        self._run = runner or self._default_runner

    def _default_runner(self, args: list[str]) -> tuple[int, str, str]:
        proc = subprocess.run(
            ["git", *args], cwd=self._repo, capture_output=True, text=True
        )
        return proc.returncode, proc.stdout, proc.stderr

    def is_git_repo(self) -> bool:
        code, out, _ = self._run(["rev-parse", "--is-inside-work-tree"])
        return code == 0 and out.strip() == "true"

    def create(self, *, prefix: str = "ak") -> tuple[Path, str]:
        """Add a worktree on a new branch at a temp path. Returns (path, branch)."""
        name = f"{prefix}-{uuid.uuid4().hex[:8]}"
        path = Path(tempfile.gettempdir()) / name
        branch = f"agentkernel/{name}"
        code, _out, err = self._run(["worktree", "add", "-b", branch, str(path)])
        if code != 0:
            raise WorktreeError(f"git worktree add failed: {err.strip() or 'unknown error'}")
        return path, branch

    def has_changes(self, path: Path | str) -> bool:
        """True if the worktree at ``path`` has uncommitted changes."""
        code, out, _ = self._run(["-C", str(path), "status", "--porcelain"])
        return code == 0 and bool(out.strip())

    def remove(self, path: Path | str) -> None:
        """Remove a worktree (force, since it may have untracked files)."""
        self._run(["worktree", "remove", "--force", str(path)])
