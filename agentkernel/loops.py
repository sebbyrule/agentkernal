"""Loop engineering: repeatable workflows with built-in stopping conditions.

A "loop" (cf. the Forward Future loop-library) is a workflow the agent runs
repeatedly until a condition is met — the **action → evaluation → iteration →
stopping condition** pattern. This is an *outer* loop around ``Agent.run``: it
re-invokes the agent on the loop's prompt, optionally checks success with a
shell command, and stops on a streak of successes or when iterations run out.

Loops are defined in TOML or sourced from a skill body, so they compose with the
skills system. The runner builds a fresh agent per iteration (independent
context) via the injected factory; the success check runs in the injected
sandbox, so a loop can verify its own work (tests pass, build is green, …).
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentkernel.agent import Agent
    from agentkernel.approval import Sandbox
    from agentkernel.skills import SkillLibrary

AgentFactory = Callable[[], "Agent"]


@dataclass
class Loop:
    """A repeatable workflow with a stopping condition."""

    name: str
    prompt: str  # the workflow instructions handed to the agent each iteration
    description: str = ""
    max_iterations: int = 5
    success_check: str | None = None  # shell command; exit 0 == success
    success_streak: int = 1  # require this many consecutive successes to stop
    cwd: str = "."
    check_timeout: int = 120


@dataclass
class LoopIteration:
    index: int
    answer: str
    check_passed: bool | None  # None when there is no success_check


@dataclass
class LoopResult:
    name: str
    iterations: list[LoopIteration] = field(default_factory=list)
    succeeded: bool = False

    @property
    def count(self) -> int:
        return len(self.iterations)


class LoopRunner:
    """Runs a :class:`Loop` until its stopping condition (or max iterations)."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        sandbox: Sandbox | None = None,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._sandbox = sandbox
        self._emit = output_fn or (lambda _msg: None)

    def run(self, loop: Loop) -> LoopResult:
        result = LoopResult(name=loop.name)
        streak = 0
        for i in range(loop.max_iterations):
            answer = self._agent_factory().run(loop.prompt)
            passed = self._check(loop)
            result.iterations.append(LoopIteration(i, answer, passed))
            status = "ok" if passed else ("fail" if passed is False else "done")
            self._emit(f"  iteration {i + 1}/{loop.max_iterations}: {status}")

            if passed is False:
                streak = 0
                continue
            # passed is True or None (no check) — both count toward the streak.
            streak += 1
            if streak >= loop.success_streak:
                result.succeeded = True
                return result
        result.succeeded = streak >= loop.success_streak
        return result

    def _check(self, loop: Loop) -> bool | None:
        if not loop.success_check:
            return None  # no programmatic check; the workflow itself decides
        if self._sandbox is None:
            return None
        code, _out, _err = self._sandbox.run(
            loop.success_check, cwd=loop.cwd, timeout=loop.check_timeout
        )
        return code == 0


def load_loop(path: str | Path) -> Loop:
    """Load a loop from a TOML file (keys mirror :class:`Loop` fields)."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Loop(
        name=data.get("name", Path(path).stem),
        prompt=data["prompt"],
        description=data.get("description", ""),
        max_iterations=int(data.get("max_iterations", 5)),
        success_check=data.get("success_check"),
        success_streak=int(data.get("success_streak", 1)),
        cwd=data.get("cwd", "."),
        check_timeout=int(data.get("check_timeout", 120)),
    )


def loop_from_skill(
    library: SkillLibrary,
    name: str,
    *,
    max_iterations: int = 5,
    success_check: str | None = None,
    success_streak: int = 1,
    cwd: str = ".",
) -> Loop | None:
    """Build a loop whose prompt is a skill's body, composing the two systems."""
    skill = library.get(name)
    if skill is None:
        return None
    return Loop(
        name=name,
        prompt=skill.body,
        description=skill.description,
        max_iterations=max_iterations,
        success_check=success_check,
        success_streak=success_streak,
        cwd=cwd,
    )
