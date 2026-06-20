"""DockerSandbox tests (design §10.3). Offline: a fake CommandRunner records the
docker argv and returns canned output, so the argv construction and lifecycle are
verified without a Docker daemon. A live test runs only when docker is present."""

from __future__ import annotations

import shutil

import pytest

from agentkernel.approval import DockerSandbox, LocalSandbox, make_sandbox
from agentkernel.approval.sandbox import SandboxError


class _FakeRunner:
    """Records each argv and returns scripted (exit, out, err) results."""

    def __init__(self, results=None):
        self.calls: list[list[str]] = []
        self._results = list(results or [])

    def __call__(self, argv, timeout):
        self.calls.append(argv)
        if self._results:
            return self._results.pop(0)
        return 0, "", ""


def _sandbox(runner, **kw):
    return DockerSandbox("/proj", runner=runner, container_name="ak-test", **kw)


def test_start_args_have_isolation_flags():
    sb = _sandbox(_FakeRunner(), network="none", memory="256m", cpus="0.5")
    args = sb._start_args()
    assert args[:3] == ["docker", "run", "-d"]
    assert "--network" in args and args[args.index("--network") + 1] == "none"
    assert "--memory" in args and args[args.index("--memory") + 1] == "256m"
    assert "--cpus" in args and args[args.index("--cpus") + 1] == "0.5"
    assert "--pids-limit" in args
    assert args[
        args.index("--security-opt") : args.index("--security-opt") + 2
    ] == ["--security-opt", "no-new-privileges"]
    # The host dir is bind-mounted at the container workdir.
    assert "-v" in args and args[args.index("-v") + 1].endswith(":/workspace")
    assert args[-3:] == ["python:3.12-slim", "sleep", "infinity"]


def test_exec_args_wrap_command_with_timeout():
    sb = _sandbox(_FakeRunner())
    args = sb._exec_args("echo hi && ls", timeout=30)
    assert args[:2] == ["docker", "exec"]
    assert args[-3] == "sh" and args[-2] == "-c"
    inner = args[-1]
    assert inner.startswith("timeout 30 sh -c ")
    assert "echo hi && ls" in inner  # original command preserved (quoted)


def test_run_starts_container_once_then_execs():
    runner = _FakeRunner([(0, "", ""), (0, "hello\n", "")])
    sb = _sandbox(runner)
    code, out, err = sb.run("echo hello", cwd="/proj", timeout=10)
    assert code == 0 and out == "hello\n"
    assert runner.calls[0][:2] == ["docker", "run"]  # started once
    assert runner.calls[1][:2] == ["docker", "exec"]
    # A second run reuses the container (no new `docker run`).
    sb.run("echo again", cwd="/proj", timeout=10)
    assert sum(1 for c in runner.calls if c[:2] == ["docker", "run"]) == 1


def test_run_reports_error_when_start_fails():
    runner = _FakeRunner([(1, "", "Cannot connect to the Docker daemon")])
    sb = _sandbox(runner)
    code, _out, err = sb.run("echo hi", cwd="/proj", timeout=10)
    assert code == 127 and "docker sandbox unavailable" in err
    assert "Docker daemon" in err


def test_run_reports_error_when_docker_missing():
    def missing(_argv, _timeout):
        raise FileNotFoundError("docker")

    sb = _sandbox(missing)
    code, _out, err = sb.run("echo hi", cwd="/proj", timeout=10)
    assert code == 127 and "unavailable" in err


def test_start_raises_sandbox_error_on_nonzero():
    sb = _sandbox(_FakeRunner([(1, "", "boom")]))
    with pytest.raises(SandboxError):
        sb.start()


def test_close_removes_container_only_when_started():
    runner = _FakeRunner([(0, "", ""), (0, "", "")])
    sb = _sandbox(runner)
    sb.close()  # never started -> no docker call
    assert runner.calls == []
    sb.start()
    sb.close()
    assert runner.calls[-1][:3] == ["docker", "rm", "-f"]


def test_make_sandbox_selects_implementation():
    assert isinstance(make_sandbox("local", "."), LocalSandbox)
    assert isinstance(make_sandbox("docker", "."), DockerSandbox)
    LocalSandbox().close()  # no-op, must not raise


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_live_docker_roundtrip(tmp_path):
    """Real container: write a file on the host, read it from inside the
    container via the bind mount, and confirm isolation removes it on close."""
    (tmp_path / "hello.txt").write_text("from host")
    sb = DockerSandbox(str(tmp_path), image="alpine", network="none")
    try:
        # Pull may be needed; allow time. Skip if the daemon can't pull offline.
        code, out, err = sb.run("cat hello.txt", cwd=str(tmp_path), timeout=60)
        if code == 127:
            pytest.skip(f"docker unavailable at runtime: {err}")
        assert code == 0 and "from host" in out
        # No host network is reachable from inside (network=none).
        net_code, _o, _e = sb.run(
            "wget -q -T 3 -O - http://example.com || echo BLOCKED",
            cwd=str(tmp_path),
            timeout=30,
        )
        assert "BLOCKED" in (_o + _e) or net_code != 0
    finally:
        sb.close()
