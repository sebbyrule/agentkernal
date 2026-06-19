"""Per-turn progress display for the REPL.

Wraps a ``JsonlTelemetry`` so every recorded turn is also printed as a concise,
one-line status line. The progress wrapper tracks cumulative usage and cost for
the current REPL session without changing the underlying telemetry interface or
the agent loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from agentkernel.budget import BudgetGuard
from agentkernel.context import CompactionEvent
from agentkernel.telemetry import JsonlTelemetry, ToolOutcome, estimate_cost
from agentkernel.types import CompletionResponse, Usage


@dataclass
class ProgressTelemetry:
    """Prints a status line for each turn and keeps cumulative totals."""

    telemetry: JsonlTelemetry
    output_fn: Callable[[str], None]
    _cumulative: BudgetGuard = field(init=False)

    def __post_init__(self) -> None:
        self._cumulative = BudgetGuard(
            model=self.telemetry.model,
            prices=self.telemetry.prices.copy(),
        )

    @property
    def path(self) -> str:
        return str(self.telemetry.path)

    @property
    def cumulative_cost(self) -> float | None:
        return self._cumulative.total_cost

    @property
    def cumulative_usage(self) -> Usage:
        return self._cumulative.total_usage

    def record_turn(
        self,
        iteration: int,
        response: CompletionResponse,
        *,
        tool_outcomes: Sequence[ToolOutcome] = (),
        compaction: CompactionEvent | None = None,
    ) -> None:
        self._cumulative.add(response.usage)
        self.output_fn(_format_line(iteration, response, tool_outcomes, self._cumulative))
        self.telemetry.record_turn(
            iteration,
            response,
            tool_outcomes=tool_outcomes,
            compaction=compaction,
        )

    def close(self) -> None:
        self.telemetry.close()


def _format_line(
    iteration: int,
    response: CompletionResponse,
    tool_outcomes: Sequence[ToolOutcome],
    cumulative: BudgetGuard,
) -> str:
    usage = response.usage
    parts = [f"[{iteration}]"]
    if response.message.tool_calls:
        names = [o.name for o in tool_outcomes]
        parts.append(f"tool_use: {', '.join(names) if names else '...'}")
    else:
        parts.append(response.stop_reason)
    parts.append(f"in={usage.input_tokens} out={usage.output_tokens}")
    if usage.cache_read_tokens or usage.cache_write_tokens:
        parts.append(f"cache={usage.cache_read_tokens}/{usage.cache_write_tokens}")
    turn_cost = estimate_cost(cumulative.model, usage, cumulative.prices)
    if turn_cost is not None:
        parts.append(f"this=${turn_cost:.6f}")
    total_cost = cumulative.total_cost
    if total_cost is not None:
        parts.append(f"total=${total_cost:.6f}")
    return " ".join(parts)
