"""Execution boundary implementations (design §10.3).

``LocalSandbox`` runs commands as a subprocess confined to a working directory,
with a timeout and a scrubbed environment (secrets removed). The target is
``DockerSandbox`` (one container per project); it is intentionally left as a
stub so ``bash``'s handler never has to change when it lands.
"""

from __future__ import annotations

import os
import signal
import subprocess

# Substrings that mark an environment variable as a secret to scrub before
# handing the environment to a subprocess.
_SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def _scrubbed_env() -> dict[str, str]:
    """A copy of the environment with secret-looking variables removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(marker in k.upper() for marker in _SECRET_MARKERS)
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process *and its children* so a timeout actually stops the work.

    ``shell=True`` means ``proc`` is the shell; the real command is a child, so
    killing only ``proc`` would orphan it and leave the output pipe open. We put
    the process in its own group/session and tear the whole group down.
    """
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            proc.kill()
    else:
        # taskkill /T kills the cmd.exe child tree; then kill the shell itself.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
        proc.kill()


class LocalSandbox:
    """Subprocess execution confined to ``cwd``, with a real timeout (design §10.3)."""

    def run(self, command: str, *, cwd: str, timeout: int) -> tuple[int, str, str]:
        kwargs: dict = dict(
            shell=True,
            cwd=cwd,
            env=_scrubbed_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Isolate the process group/tree so _kill_tree can stop it on timeout.
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(command, **kwargs)
        try:
            out, err = proc.communicate(timeout=timeout)
            return proc.returncode, out, err
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            out, err = proc.communicate()  # drain pipes after the tree is dead
            return 124, out or "", (err or "") + f"\n[timed out after {timeout}s]"


class DockerSandbox:  # pragma: no cover - stub for a later phase
    """TODO: container-per-project sandbox. Not implemented in v1 (design §10.3)."""

    def run(self, command: str, *, cwd: str, timeout: int) -> tuple[int, str, str]:
        raise NotImplementedError("DockerSandbox is not implemented yet")
