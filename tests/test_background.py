"""Background-run tests (design §18.2): run_background spawns a detached process
and reports the output path. The spawn is injected so no real process launches."""

from __future__ import annotations

from agentkernel.cli import run_background


class _FakeProc:
    pid = 4242


def test_background_spawns_and_reports(tmp_path):
    captured = {}

    def fake_spawn(argv, *, stdout):
        captured["argv"] = argv
        captured["stdout_name"] = getattr(stdout, "name", None)
        return _FakeProc()

    out: list[str] = []
    rc = run_background(
        "summarize the repo",
        config_path="my.toml",
        log_dir=str(tmp_path / "traces"),
        spawn=fake_spawn,
        output_fn=out.append,
    )
    assert rc == 0
    # argv re-invokes the CLI as a module with the prompt and config.
    argv = captured["argv"]
    assert argv[1:3] == ["-m", "agentkernel"]
    assert "run" in argv and "summarize the repo" in argv
    assert "--config" in argv and "my.toml" in argv
    # an output file was created under <log_dir>/../background/
    out_file = captured["stdout_name"]
    assert out_file is not None and "background" in out_file.replace("\\", "/")
    assert any("pid 4242" in line for line in out)
    assert any(".out" in line for line in out)


def test_background_empty_prompt_is_error(tmp_path):
    out: list[str] = []
    rc = run_background("   ", log_dir=str(tmp_path / "t"), spawn=lambda *a, **k: _FakeProc(),
                        output_fn=out.append)
    assert rc == 1
    assert "nothing to run" in out[0]


def test_background_creates_output_dir(tmp_path):
    log_dir = tmp_path / "traces"
    run_background("go", log_dir=str(log_dir), spawn=lambda *a, **k: _FakeProc(),
                   output_fn=lambda _m: None)
    assert (tmp_path / "background").is_dir()
