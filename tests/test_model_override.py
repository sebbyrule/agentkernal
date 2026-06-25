"""profile.model_override: a run uses the override model (design §13, Phase 5)."""

from __future__ import annotations

from types import SimpleNamespace

from agentkernel.providers import OpenAIProvider
from agentkernel.types import CompletionResponse, Message, Usage
from tests.conftest import build_agent


def test_with_model_clones_and_shares_credentials():
    base = OpenAIProvider("model-a", api_key="k", base_url="http://x/v1")
    clone = base.with_model("model-b")
    assert clone.model == "model-b" and base.model == "model-a"
    assert clone._base_url == base._base_url
    assert clone._pool is base._pool  # shares credentials, no env re-read


class _RecordingProvider:
    name = "rec"
    context_window = 100_000

    def __init__(self, model: str = "base", *, log: list[str] | None = None) -> None:
        self.model = model
        self.used: list[str] = log if log is not None else []

    def with_model(self, model: str) -> _RecordingProvider:
        return _RecordingProvider(model, log=self.used)

    def complete(self, messages, tools, *, max_tokens, temperature=1.0,
                 system=None, reasoning=None, on_text=None) -> CompletionResponse:
        self.used.append(self.model)
        return CompletionResponse(
            Message(role="assistant", content="ok"), Usage(), "end_turn"
        )


def test_run_uses_override_model_then_default():
    provider = _RecordingProvider("base")
    agent = build_agent(provider)
    agent.run("hi", profile=SimpleNamespace(model_override="big-model"))
    agent.run("hi", profile=None)
    assert provider.used == ["big-model", "base"]
