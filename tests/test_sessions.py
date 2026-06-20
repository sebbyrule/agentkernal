"""Session list/show/delete and resume wiring (design §18.2)."""

from __future__ import annotations

from agentkernel.cli import build_runtime, run_sessions
from agentkernel.config import Config
from agentkernel.memory import make_memory_store
from agentkernel.types import Message


def _config_with_session(tmp_path, sid="sess-1", messages=None):
    mem_dir = tmp_path / "mem"
    store = make_memory_store("file", str(mem_dir))
    store.save(sid, messages or [
        Message(role="user", content="first question here"),
        Message(role="assistant", content="an answer"),
    ])
    return Config(
        memory_store="file", memory_dir=str(mem_dir), log_dir=str(tmp_path / "traces")
    )


def test_sessions_list_shows_id_and_preview(tmp_path):
    cfg = _config_with_session(tmp_path)
    out: list[str] = []
    assert run_sessions(cfg, "list", None, output_fn=out.append) == 0
    body = "\n".join(out)
    assert "sess-1" in body and "first question here" in body and "2 msgs" in body


def test_sessions_show_prints_transcript(tmp_path):
    cfg = _config_with_session(tmp_path)
    out: list[str] = []
    run_sessions(cfg, "show", "sess-1", output_fn=out.append)
    assert any("user: first question here" in line for line in out)
    assert any("assistant: an answer" in line for line in out)


def test_sessions_show_unknown_id(tmp_path):
    cfg = _config_with_session(tmp_path)
    out: list[str] = []
    assert run_sessions(cfg, "show", "nope", output_fn=out.append) == 1


def test_sessions_delete_removes_session(tmp_path):
    cfg = _config_with_session(tmp_path)
    assert run_sessions(cfg, "delete", "sess-1", output_fn=lambda _m: None) == 0
    out: list[str] = []
    run_sessions(cfg, "list", None, output_fn=out.append)
    assert "no saved sessions" in out[0]


def test_sessions_list_empty(tmp_path):
    cfg = Config(memory_store="file", memory_dir=str(tmp_path / "m"),
                 log_dir=str(tmp_path / "t"))
    out: list[str] = []
    run_sessions(cfg, "list", None, output_fn=out.append)
    assert "no saved sessions" in out[0]


def test_resume_threads_session_id_into_telemetry(tmp_path):
    cfg = Config(provider="local", log_dir=str(tmp_path / "traces"))
    agent, telemetry, clients = build_runtime(cfg, session_id="resume-me")
    try:
        assert telemetry.session_id == "resume-me"
        # The trace file is named after the (resumed) session id.
        assert telemetry.path.name == "resume-me.jsonl"
    finally:
        telemetry.close()
        for c in clients:
            c.close()


def test_resumed_agent_loads_prior_transcript(tmp_path):
    """The agent's pre-run memory load uses telemetry.session_id, so resuming a
    session id replays that transcript into the new run's context."""
    from tests.conftest import build_agent
    from tests.fakes import FakeProvider, text_response

    mem = make_memory_store("memory")
    mem.save("S", [Message(role="user", content="remember the magic word is plum")])
    provider = FakeProvider([text_response("ok")])
    agent = build_agent(provider, memory=mem)
    agent.telemetry.session_id = "S"  # simulate resuming session S
    agent.run("what was it?")
    first_request = provider.calls[0]
    assert any("magic word is plum" in (m.content or "") for m in first_request)
