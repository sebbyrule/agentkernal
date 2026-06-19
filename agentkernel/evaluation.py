"""Evaluator / eval harness (design §13, Phase 5).

"An evaluator is a profile whose final output is a structured score." This runs
the agent on each case, then asks a judge model to score the answer against a
rubric, producing a structured ``EvalResult``. A suite aggregates into pass-rate
and mean score — useful for regression tests and model comparison, and as signal
for the self-improvement loop.

Built entirely on the kernel (an Agent + a provider); the loop is untouched.
Judging is best-effort: an unparseable judge reply scores 0 rather than raising.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agentkernel.types import Message

if TYPE_CHECKING:
    from agentkernel.agent import Agent
    from agentkernel.providers import Provider

# A factory that returns a FRESH agent per case (independent context).
AgentFactory = Callable[[], "Agent"]

_JUDGE_SYSTEM = (
    "You are a strict evaluator. Score how well an agent's answer satisfies the "
    "rubric for the given task. Respond with ONLY a JSON object: "
    '{"score": <0-100 integer>, "pass": <true|false>, "reasoning": "<one sentence>"}.'
)
_DEFAULT_RUBRIC = "The answer is correct, complete, and directly addresses the task."


@dataclass
class EvalCase:
    name: str
    prompt: str
    rubric: str | None = None  # overrides the suite/default rubric


@dataclass
class EvalResult:
    name: str
    answer: str
    score: float  # normalized 0.0–1.0
    passed: bool
    reasoning: str
    raw_judge: str = ""


@dataclass
class EvalSummary:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / self.total if self.total else 0.0


def _parse_score(text: str, pass_threshold: float) -> tuple[float, bool, str]:
    """Extract ``(score 0-1, passed, reasoning)`` from a judge reply.

    Tolerant of prose around the JSON; if nothing parseable is found, the case
    scores 0 and fails (a non-answer should never silently pass)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return 0.0, False, "could not parse judge output"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0, False, "could not parse judge output"

    raw = data.get("score", 0)
    try:
        score = float(raw)
    except (TypeError, ValueError):
        score = 0.0
    if score > 1.0:  # judges return 0-100; normalize to 0-1
        score = score / 100.0
    score = max(0.0, min(1.0, score))

    passed = data.get("pass")
    if not isinstance(passed, bool):
        passed = score >= pass_threshold
    return score, passed, str(data.get("reasoning", ""))


class Evaluator:
    """Runs cases through an agent and scores answers with a judge provider."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        judge: "Provider",
        *,
        default_rubric: str = _DEFAULT_RUBRIC,
        pass_threshold: float = 0.6,
        judge_max_tokens: int = 512,
    ) -> None:
        self._agent_factory = agent_factory
        self._judge = judge
        self._default_rubric = default_rubric
        self._pass_threshold = pass_threshold
        self._judge_max_tokens = judge_max_tokens

    def evaluate_case(self, case: EvalCase) -> EvalResult:
        answer = self._agent_factory().run(case.prompt)
        rubric = case.rubric or self._default_rubric
        raw = self._score(case.prompt, rubric, answer)
        score, passed, reasoning = _parse_score(raw, self._pass_threshold)
        return EvalResult(case.name, answer, score, passed, reasoning, raw)

    def run_suite(self, cases: list[EvalCase]) -> EvalSummary:
        return EvalSummary([self.evaluate_case(c) for c in cases])

    def _score(self, prompt: str, rubric: str, answer: str) -> str:
        judge_prompt = (
            f"Task:\n{prompt}\n\nRubric:\n{rubric}\n\nAgent answer:\n{answer}\n\n"
            "Score the answer against the rubric."
        )
        resp = self._judge.complete(
            [Message(role="user", content=judge_prompt)],
            [],
            max_tokens=self._judge_max_tokens,
            temperature=0.0,
            system=_JUDGE_SYSTEM,
        )
        return resp.message.content.strip()


def load_eval_suite(path: str | Path) -> tuple[str, list[EvalCase]]:
    """Load ``(default_rubric, cases)`` from a TOML suite file.

    Format::

        rubric = "default rubric for all cases"   # optional
        [[cases]]
        name = "..."
        prompt = "..."
        rubric = "..."                            # optional per-case override
    """
    with Path(path).open("rb") as fh:
        data = tomllib.load(fh)
    default_rubric = data.get("rubric", _DEFAULT_RUBRIC)
    cases = [
        EvalCase(name=c["name"], prompt=c["prompt"], rubric=c.get("rubric"))
        for c in data.get("cases", [])
    ]
    return default_rubric, cases
