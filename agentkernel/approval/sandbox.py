"""Execution boundary implementations (design §10.3).

``LocalSandbox`` runs commands as a subprocess confined to a working directory.
``DockerSandbox`` runs them inside a per-project container — real isolation
(separate filesystem, no host network by default, resource limits) — behind the
same ``Sandbox`` protocol, so ``bash``'s handler never changes.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path

# A command runner: takes an argv list + timeout, returns (exit, stdout, stderr).
# Injectable so DockerSandbox is testable offline without a Docker daemon.
CommandRunner = Callable[[list[str], int], "tuple[int, str, str]"]

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


class SandboxError(RuntimeError):
    """A sandbox lifecycle fault (e.g. the container could not be started)."""


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

    def close(self) -> None:
        """No persistent resources to release (kept for the Sandbox protocol)."""


def _subprocess_runner(argv: list[str], timeout: int) -> tuple[int, str, str]:
    """Default ``CommandRunner``: run an argv list, capture output, honor timeout."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out, err + "\n[timed out]"


class DockerSandbox:
    """Runs commands inside one long-lived container per project (design §10.3).

    The host working directory is bind-mounted at ``workdir`` inside the
    container; the container starts lazily on the first ``run`` and is removed by
    ``close``. By default it has no network and bounded memory/CPU/PIDs, so a
    command cannot reach the host filesystem, the network, or exhaust resources —
    the isolation ``LocalSandbox`` lacks.

    The Docker CLI is invoked through an injectable ``runner`` so the argv
    construction and lifecycle are unit-testable without a Docker daemon.
    """

    def __init__(
        self,
        working_dir: str = ".",
        *,
        image: str = "python:3.12-slim",
        network: str = "none",
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 256,
        workdir: str = "/workspace",
        container_name: str | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self._host_dir = str(Path(working_dir).resolve())
        self._image = image
        self._network = network
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._workdir = workdir
        self._name = container_name or f"agentkernel-{uuid.uuid4().hex[:12]}"
        self._run = runner or _subprocess_runner
        self._started = False

    # --- docker argv construction (pure; the unit-test surface) ------------

    def _start_args(self) -> list[str]:
        return [
            "docker", "run", "-d", "--rm", "--name", self._name,
            "--network", self._network,
            "--memory", self._memory,
            "--cpus", self._cpus,
            "--pids-limit", str(self._pids_limit),
            "--security-opt", "no-new-privileges",
            "-v", f"{self._host_dir}:{self._workdir}",
            "-w", self._workdir,
            self._image, "sleep", "infinity",
        ]

    def _exec_args(self, command: str, timeout: int) -> list[str]:
        # Wrap in the container's `timeout` so the in-container process is
        # actually killed, not just the host-side `docker exec`.
        inner = f"timeout {timeout} sh -c {shlex.quote(command)}"
        return ["docker", "exec", "-w", self._workdir, self._name, "sh", "-c", inner]

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        code, out, err = self._run(self._start_args(), 120)
        if code != 0:
            raise SandboxError(f"could not start container: {(err or out).strip()}")
        self._started = True

    def run(self, command: str, *, cwd: str, timeout: int) -> tuple[int, str, str]:
        try:
            self.start()
        except (SandboxError, FileNotFoundError, OSError) as exc:
            # Surface as a non-zero result so bash turns it into an error result
            # the model can react to (design §8.3) rather than crashing the loop.
            return 127, "", f"docker sandbox unavailable: {exc}"
        # Allow a little host-side slack beyond the in-container timeout.
        return self._run(self._exec_args(command, timeout), timeout + 5)

    def close(self) -> None:
        if not self._started:
            return
        self._run(["docker", "rm", "-f", self._name], 30)
        self._started = False


def make_sandbox(
    sandbox: str, working_dir: str, *, image: str = "python:3.12-slim", network: str = "none"
):
    """Build the configured sandbox. ``"docker"`` -> isolated container; else local."""
    if sandbox == "docker":
        return DockerSandbox(working_dir, image=image, network=network)
    return LocalSandbox()
