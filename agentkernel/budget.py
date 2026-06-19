"""Run-level budget guardrails (design §11 extension).

A ``BudgetGuard`` tracks cumulative usage for one ``Agent.run`` call and returns a
stop reason if the configured cost or token ceiling is exceeded. It is re-set at
the start of every run so the guard is per-run, not global across a REPL session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentkernel.telemetry import DEFAULT_PRICES, Price, estimate_cost
from agentkernel.types import Usage


@dataclass
class BudgetGuard:
    """Simple guardrail against runaway spend or token usage.

    ``max_cost_usd`` and ``max_input_tokens`` are checked after each provider
    completion. If a limit is exceeded, the loop stops before executing any
    further tool calls; a final-answer turn is still returned because its tokens
    have already been spent.
    """

    max_cost_usd: float | None = None
    max_input_tokens: int | None = None
    model: str = "unknown"
    prices: dict[str, Price] = field(default_factory=lambda: DEFAULT_PRICES.copy())
    _total: Usage = field(default_factory=Usage)

    def reset(self) -> None:
        self._total = Usage()

    def add(self, usage: Usage) -> None:
        self._total.input_tokens += usage.input_tokens
        self._total.output_tokens += usage.output_tokens
        self._total.cache_read_tokens += usage.cache_read_tokens
        self._total.cache_write_tokens += usage.cache_write_tokens

    def exceeded(self) -> tuple[bool, str]:
        """Return ``(True, reason)`` if a budget has been exceeded."""
        if self.max_input_tokens is not None and self._total.input_tokens > self.max_input_tokens:
            return True, (
                f"input_tokens {self._total.input_tokens:n} > limit {self.max_input_tokens:n}"
            )
        if self.max_cost_usd is not None:
            cost = estimate_cost(self.model, self._total, self.prices)
            if cost is not None and cost > self.max_cost_usd:
                return True, f"cost ${cost:.6f} > limit ${self.max_cost_usd:.2f}"
        return False, ""

    @property
    def total_usage(self) -> Usage:
        return Usage(
            input_tokens=self._total.input_tokens,
            output_tokens=self._total.output_tokens,
            cache_read_tokens=self._total.cache_read_tokens,
            cache_write_tokens=self._total.cache_write_tokens,
        )

    @property
    def total_cost(self) -> float | None:
        return estimate_cost(self.model, self._total, self.prices)
