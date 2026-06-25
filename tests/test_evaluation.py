"""Evaluator / eval-harness tests (design §13, Phase 5). Offline: separate
FakeProviders drive the agent answers and the judge scores."""

from __future__ import annotations

from agentkernel.evaluation import (
    EvalCase,
    Evaluator,
    _parse_score,
    load_eval_suite,
)
from tests.conftest import build_agent
from tests.fakes import FakeProvider, text_response


def test_parse_score_normalizes_scales_and_thresholds():
    s, p, r = _parse_score('{"score": 80, "pass": true, "reasoning": "ok"}', 0.6)
    assert s == 0.8 and p is True and r == "ok"

    # 0-1 scale, no explicit pass -> derived from threshold.
    s, p, _ = _parse_score('{"score": 0.5}', 0.6)
    assert s == 0.5 and p is False

    # JSON embedded in prose is still extracted.
    s, p, _ = _parse_score('Here is my verdict: {"score": 70} . done', 0.6)
    assert s == 0.7 and p is True

    # Unparseable -> a non-answer never silently passes.
    s, p, _ = _parse_score("looks fine to me", 0.6)
    assert s == 0.0 and p is False


def test_evaluator_runs_cases_and_aggregates():
    agent_provider = FakeProvider([text_response("4 files"), text_response("no idea")])
    judge = FakeProvider(
        [
            text_response('{"score": 95, "pass": true, "reasoning": "correct"}'),
            text_response('{"score": 10, "pass": false, "reasoning": "wrong"}'),
        ]
    )
    evaluator = Evaluator(lambda: build_agent(agent_provider), judge)
    summary = evaluator.run_suite(
        [EvalCase("a", "count files"), EvalCase("b", "count files")]
    )
    assert summary.total == 2 and summary.passed == 1
    assert summary.results[0].passed and not summary.results[1].passed
    assert summary.pass_rate == 0.5
    assert 0.0 < summary.mean_score < 1.0


def test_evaluator_unparseable_judge_fails_case():
    agent_provider = FakeProvider([text_response("an answer")])
    judge = FakeProvider([text_response("I think it's fine (no JSON)")])
    evaluator = Evaluator(lambda: build_agent(agent_provider), judge)
    summary = evaluator.run_suite([EvalCase("x", "do a task")])
    assert summary.passed == 0 and summary.results[0].score == 0.0


def test_load_eval_suite(tmp_path):
    suite = tmp_path / "suite.toml"
    suite.write_text(
        'rubric = "be correct"\n'
        "[[cases]]\n"
        'name = "c1"\n'
        'prompt = "do x"\n'
        "[[cases]]\n"
        'name = "c2"\n'
        'prompt = "do y"\n'
        'rubric = "be precise"\n'
    )
    default_rubric, cases = load_eval_suite(suite)
    assert default_rubric == "be correct" and len(cases) == 2
    assert cases[0].rubric is None and cases[1].rubric == "be precise"


def test_run_eval_empty_suite_returns_1(tmp_path):
    from agentkernel.cli import run_eval
    from agentkernel.config import Config

    suite = tmp_path / "empty.toml"
    suite.write_text('rubric = "x"\n')  # no cases
    out: list[str] = []
    code = run_eval(
        Config(provider="anthropic", log_dir=str(tmp_path / "t")),
        str(suite),
        output_fn=out.append,
    )
    assert code == 1 and any("no cases" in line for line in out)


def test_evaluator_uses_default_rubric_for_case_without_rubric():
    agent_provider = FakeProvider([text_response("42")])
    judge = FakeProvider(
        [text_response('{"score": 100, "pass": true, "reasoning": "ok"}')]
    )
    evaluator = Evaluator(
        lambda: build_agent(agent_provider),
        judge,
        default_rubric="custom default rubric",
    )
    summary = evaluator.run_suite([EvalCase("x", "what is it?")])
    assert summary.passed == 1
    judge_prompt = judge.calls[-1][0].content
    assert "custom default rubric" in judge_prompt


def test_eval_summary_to_dict_serializes_results():
    from agentkernel.evaluation import EvalResult, EvalSummary

    summary = EvalSummary(
        [
            EvalResult("a", "answer", 0.9, True, "good", '{"score": 90}'),
            EvalResult("b", "bad", 0.2, False, "missing"),
        ]
    )
    data = summary.to_dict()
    assert data["total"] == 2
    assert data["passed"] == 1
    assert data["pass_rate"] == 0.5
    assert data["results"][0]["name"] == "a"
    assert data["results"][0]["raw_judge"] == '{"score": 90}'


def test_run_eval_filters_cases_and_writes_report(tmp_path, monkeypatch):
    from agentkernel.cli import run_eval
    from agentkernel.config import Config
    from agentkernel.tools import ToolRegistry

    suite = tmp_path / "suite.toml"
    suite.write_text(
        'rubric = "default"\n'
        "[[cases]]\n"
        'name = "add"\n'
        'prompt = "1+1"\n'
        "[[cases]]\n"
        'name = "mult"\n'
        'prompt = "2*2"\n',
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    agent_provider = FakeProvider([text_response("2"), text_response("4")])
    judge_provider = FakeProvider(
        [text_response('{"score": 100, "pass": true, "reasoning": "ok"}')]
    )

    class _FakeAgent:
        provider = agent_provider
        registry = ToolRegistry()
        context_source = None

        def run(self, prompt: str) -> str:
            resp = agent_provider.complete([], [], max_tokens=10)
            return resp.message.content

    class _FakeTelemetry:
        path = tmp_path / "trace.jsonl"

        def close(self):
            pass

    import agentkernel.cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "build_runtime",
        lambda config, **kw: (_FakeAgent(), _FakeTelemetry(), []),
    )
    # The eval judge is built via the role router; patch where it resolves.
    import agentkernel.roles as roles_mod

    monkeypatch.setattr(roles_mod, "make_provider", lambda config, **kw: judge_provider)
    class _FakeSandbox:
        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "make_sandbox", lambda *a, **k: _FakeSandbox())

    out: list[str] = []
    code = run_eval(
        Config(
            provider="fake",
            judge_model="judge",
            log_dir=str(tmp_path / "t"),
        ),
        str(suite),
        case_filter=["add"],
        output_path=str(report_path),
        output_fn=out.append,
    )
    assert code == 0
    assert report_path.exists()
    import json

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["total"] == 1
    assert data["results"][0]["name"] == "add"

