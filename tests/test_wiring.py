"""Integration wiring tests: build_runtime exposes skills (Phase 4) and the
knowledge graph (Phase 6); the `improve` path (Phase 7) is reachable; and the
REPL skill slash commands work. All offline (the local/anthropic providers do
not connect until a completion, which these tests never trigger)."""

from __future__ import annotations

from agentkernel.cli import build_runtime, repl, run_improve
from agentkernel.config import Config
from agentkernel.skills import DirectorySkillStore

from tests.fakes import FakeProvider, text_response


class _ScriptedInput:
    def __init__(self, lines):
        self._lines = list(lines)

    def __call__(self, _prompt):
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def _cfg(tmp_path, **kw) -> Config:
    return Config(provider="anthropic", log_dir=str(tmp_path / "traces"), **kw)


def _teardown(telemetry, clients):
    telemetry.close()
    for c in clients:
        c.close()


# --- knowledge graph (Phase 6) ---------------------------------------------


def test_graph_tools_registered_when_enabled(tmp_path):
    cfg = _cfg(tmp_path, enable_graph=True, graph_path=str(tmp_path / "g.jsonl"))
    agent, telemetry, clients = build_runtime(cfg)
    try:
        names = {s.name for s in agent.registry.specs()}
        assert {"graph_add", "graph_query"} <= names
    finally:
        _teardown(telemetry, clients)


def test_graph_tools_absent_by_default(tmp_path):
    agent, telemetry, clients = build_runtime(_cfg(tmp_path))
    try:
        assert "graph_add" not in {s.name for s in agent.registry.specs()}
    finally:
        _teardown(telemetry, clients)


# --- skills (Phase 4) ------------------------------------------------------


def test_build_runtime_injects_memory_tools_when_enabled(tmp_path):
    cfg = _cfg(
        tmp_path,
        enable_memory_tools=True,
        memory_notes_path=str(tmp_path / "notes.jsonl"),
    )
    agent, telemetry, clients = build_runtime(cfg)
    try:
        names = {s.name for s in agent.registry.specs()}
        assert {"remember", "recall", "forget", "update_memory"} <= names
        # Session tools depend on a MemoryStore being configured.
        assert "list_sessions" not in names
    finally:
        _teardown(telemetry, clients)


def test_build_runtime_injects_session_search_when_sqlite(tmp_path):
    cfg = _cfg(
        tmp_path,
        memory_store="sqlite",
        memory_dir=str(tmp_path / "mem"),
        enable_memory_tools=True,
        memory_notes_path=str(tmp_path / "notes.jsonl"),
    )
    agent, telemetry, clients = build_runtime(cfg)
    try:
        names = {s.name for s in agent.registry.specs()}
        assert "search_sessions" in names
        assert "list_sessions" in names
    finally:
        _teardown(telemetry, clients)


def test_build_runtime_injects_skill_catalog_and_pin(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "terse.md").write_text("Answer in one sentence.")
    cfg = _cfg(tmp_path, skills_dir=str(skills), skills=["terse"])
    agent, telemetry, clients = build_runtime(cfg)
    try:
        adds = agent.context_source.system_additions()
        assert adds[0].startswith("# Available skills")  # catalog always present
        assert any("Answer in one sentence." in a for a in adds)  # pinned body shown
        # The use_skill tool is registered so bodies can also load on demand.
        assert "use_skill" in {s.name for s in agent.registry.specs()}
    finally:
        _teardown(telemetry, clients)


def test_build_runtime_wires_memory_notes_for_repl(tmp_path):
    cfg = _cfg(
        tmp_path,
        enable_memory_tools=True,
        memory_notes_path=str(tmp_path / "notes.jsonl"),
    )
    agent, telemetry, clients = build_runtime(cfg)
    try:
        assert agent.notes is not None
        agent.notes.add("first fact", tags=["demo"])
        agent.notes.add("second fact")
        out: list[str] = []
        repl(
            agent,
            config=cfg,
            input_fn=_ScriptedInput(
                ["/memory list", "/memory delete 2", "/memory export", "exit"]
            ),
            output_fn=out.append,
        )
        assert any("[1] first fact [tags: demo]" in line for line in out)
        assert any("[2] second fact" in line for line in out)
        assert any("deleted note 2" in line.lower() for line in out)
        assert any("exported" in line.lower() for line in out)
        assert len(agent.notes.all()) == 1
    finally:
        _teardown(telemetry, clients)


def test_repl_skill_toggle_activates_system_addition(agent_builder, tmp_path):
    (tmp_path / "s.md").write_text("Be terse.")
    provider = FakeProvider([text_response("ok")])
    agent = agent_builder(provider)
    agent.context_source = DirectorySkillStore(tmp_path)
    repl(
        agent,
        input_fn=_ScriptedInput(["/skill s", "go", "exit"]),
        output_fn=lambda _line: None,
    )
    # Pinning the skill put its body into the run's system prompt (with catalog).
    assert "Be terse." in provider.system_args[0]


def test_repl_skills_lists_available(agent_builder, tmp_path):
    (tmp_path / "alpha.md").write_text("A")
    provider = FakeProvider([])
    agent = agent_builder(provider)
    agent.context_source = DirectorySkillStore(tmp_path)
    out: list[str] = []
    repl(agent, input_fn=_ScriptedInput(["/skills", "exit"]), output_fn=out.append)
    assert any("alpha" in line for line in out)


# --- sub-agent spawn (design §13) ------------------------------------------


def test_spawn_tool_registered_when_enabled(tmp_path):
    cfg = _cfg(tmp_path, enable_spawn=True)
    agent, telemetry, clients = build_runtime(cfg)
    try:
        assert "spawn" in {s.name for s in agent.registry.specs()}
    finally:
        _teardown(telemetry, clients)


def test_spawn_tool_absent_by_default(tmp_path):
    agent, telemetry, clients = build_runtime(_cfg(tmp_path))
    try:
        assert "spawn" not in {s.name for s in agent.registry.specs()}
    finally:
        _teardown(telemetry, clients)


# --- self-improvement (Phase 7) --------------------------------------------


def test_run_improve_reports_no_trace(tmp_path):
    # Empty log_dir -> nothing to analyze; returns 1 without calling the model.
    out: list[str] = []
    assert run_improve(_cfg(tmp_path), output_fn=out.append) == 1
    assert any("no trace" in line for line in out)
