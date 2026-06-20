"""REPL and CLI tests (design §16, M4).

Tests run offline against a scripted provider."""

from __future__ import annotations

from agentkernel.cli import repl, run_once
from agentkernel.providers import ProviderError
from agentkernel.tools import ToolRegistry, ToolSpec
from agentkernel.types import ToolResult
from tests.fakes import FakeProvider, text_response
from agentkernel.skills import SkillLibrary


class _ScriptedInput:
    """Yields queued lines, then raises EOFError like a closed stdin."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __call__(self, _prompt):
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo the given value back.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=lambda args: ToolResult("", f"echo:{args['value']}"),
    )


def test_repl_prints_answer_then_exits(agent_builder):
    provider = FakeProvider([text_response("hello!")])
    agent = agent_builder(provider)
    out: list[str] = []
    code = repl(agent, input_fn=_ScriptedInput(["hi", "exit"]), output_fn=out.append)
    assert code == 0
    assert "hello!" in out


def test_repl_skips_blank_lines(agent_builder):
    provider = FakeProvider([text_response("answer")])
    agent = agent_builder(provider)
    out: list[str] = []
    repl(agent, input_fn=_ScriptedInput(["", "   ", "real question"]), output_fn=out.append)
    assert "answer" in out
    assert len(provider.calls) == 1  # blank lines never reached the model


def test_repl_handles_eof(agent_builder):
    provider = FakeProvider([])
    agent = agent_builder(provider)
    out: list[str] = []
    assert repl(agent, input_fn=_ScriptedInput([]), output_fn=out.append) == 0


def test_repl_surfaces_provider_error_without_crashing(agent_builder):
    class _BoomProvider(FakeProvider):
        def complete(self, *a, **k):
            raise ProviderError("ANTHROPIC_API_KEY is not set")

    agent = agent_builder(_BoomProvider([]))
    out: list[str] = []
    repl(agent, input_fn=_ScriptedInput(["hi", "exit"]), output_fn=out.append)
    assert any("provider error" in line for line in out)


def test_run_once_prints_answer(agent_builder):
    provider = FakeProvider([text_response("hello!")])
    agent = agent_builder(provider)
    out: list[str] = []
    assert run_once(agent, "hi", output_fn=out.append) == 0
    assert "hello!" in out


def test_repl_slash_exit(agent_builder):
    provider = FakeProvider([])
    agent = agent_builder(provider)
    out: list[str] = []
    code = repl(agent, input_fn=_ScriptedInput(["/exit"]), output_fn=out.append)
    assert code == 0
    assert not any("hello" in line for line in out)


def test_repl_slash_clear_clears_context(agent_builder):
    provider = FakeProvider([text_response("ok")])
    agent = agent_builder(provider)
    repl(
        agent,
        input_fn=_ScriptedInput(["/clear", "go", "exit"]),
        output_fn=lambda _line: None,
    )
    # Context was reset; only the "go" turn reaches the provider.
    assert len(provider.calls) == 1
    assert provider.calls[0][-1].role == "user"
    assert provider.calls[0][-1].content == "go"


def test_repl_slash_system_sets_prompt(agent_builder):
    provider = FakeProvider([text_response("ok")])
    agent = agent_builder(provider)
    repl(
        agent,
        input_fn=_ScriptedInput(["/system be helpful", "go", "exit"]),
        output_fn=lambda _line: None,
    )
    assert provider.system_args == ["be helpful"]


def test_repl_slash_tools_lists_tools(agent_builder):
    registry = ToolRegistry()
    registry.register(_echo_tool())
    provider = FakeProvider([])
    agent = agent_builder(provider, registry=registry)
    out: list[str] = []
    repl(
        agent,
        input_fn=_ScriptedInput(["/tools", "exit"]),
        output_fn=out.append,
    )
    assert any("echo" in line for line in out)


def test_repl_unknown_slash_command(agent_builder):
    provider = FakeProvider([])
    agent = agent_builder(provider)
    out: list[str] = []
    repl(
        agent,
        input_fn=_ScriptedInput(["/nope", "exit"]),
        output_fn=out.append,
    )
    assert any("unknown command" in line for line in out)


def test_repl_slash_skill_toggles_active_skill(tmp_path, agent_builder):
    provider = FakeProvider([text_response("ok")])
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "terse").mkdir()
    (skill_dir / "terse" / "SKILL.md").write_text(
        "---\nname: terse\n---\nAlways answer in one word.", encoding="utf-8"
    )
    source = SkillLibrary(skill_dir)
    agent = agent_builder(provider, context_source=source)
    out: list[str] = []
    repl(
        agent,
        input_fn=_ScriptedInput(["/skill terse", "go", "exit"]),
        output_fn=out.append,
    )
    assert any("[skill terse: on]" in line for line in out)
    # The next turn's system prompt included the pinned skill body.
    assert provider.system_args and "Always answer in one word." in provider.system_args[-1]

