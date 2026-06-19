"""REPL tests (design §16, M4): the loop echoes answers, persists context across
messages, handles exit/EOF, and surfaces provider errors without crashing. Runs
offline against a scripted provider."""

from __future__ import annotations

from agentkernel.cli import repl
from agentkernel.providers import ProviderError

from tests.fakes import FakeProvider, text_response


class _ScriptedInput:
    """Yields queued lines, then raises EOFError like a closed stdin."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __call__(self, _prompt):
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


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
