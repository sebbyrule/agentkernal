"""Per-turn telemetry (design §12).

Writes one JSONL file per session under ``config.log_dir``. The record schema
(below) is a stable interface for later phases — cost dashboards and the Phase-7
self-improvement loop read it — so treat it as a contract, not throwaway logs.

Redaction is the default: tool arguments are logged as a hash + length, never
raw, and file contents never enter a record at all. ``verbose=True`` (the
``--verbose-trace`` flag) includes raw arguments for local debugging only.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agentkernel.context import CompactionEvent
    from agentkernel.types import CompletionResponse


@dataclass
class ToolOutcome:
    """What happened to one tool call this turn (for the trace)."""

    name: str
    arguments: dict[str, Any]
    approved: bool | None  # True executed, False denied, None never reached the gate
    is_error: bool


@dataclass
class Price:
    """USD per million tokens for one model."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


# TODO(owner): verify these seed prices against current provider pricing.
DEFAULT_PRICES: dict[str, Price] = {
    "claude-opus-4-8": Price(15.0, 75.0, 1.50, 18.75),
    "claude-sonnet-4-6": Price(3.0, 15.0, 0.30, 3.75),
    "claude-haiku-4-5-20251001": Price(1.0, 5.0, 0.10, 1.25),
    "gpt-4o": Price(2.5, 10.0, 1.25, 0.0),
    "gpt-4o-mini": Price(0.15, 0.60, 0.075, 0.0),
}


def estimate_cost(model: str, usage, prices: dict[str, Price]) -> float | None:
    """Cost in USD, or None for an unknown model (tokens still logged)."""
    price = prices.get(model)
    if price is None:
        return None
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    )
    return round(total / 1_000_000, 6)


class Telemetry(Protocol):
    """Minimal telemetry interface for the agent loop."""

    @property
    def model(self) -> str:
        """Model name used for cost estimates."""
        ...

    @property
    def prices(self) -> dict[str, Price]:
        """Price table used for cost estimates."""
        ...

    def record_turn(
        self,
        iteration: int,
        response: CompletionResponse,
        *,
        tool_outcomes: Sequence[ToolOutcome] = (),
        compaction: CompactionEvent | None = None,
    ) -> None:
        ...


class NullTelemetry:
    """Records nothing. Used where tracing is not configured (and by tests)."""

    @property
    def model(self) -> str:
        return "null"

    @property
    def prices(self) -> dict[str, Price]:
        return DEFAULT_PRICES.copy()

    def record_turn(
        self,
        iteration: int,
        response: CompletionResponse,
        *,
        tool_outcomes: Sequence[ToolOutcome] = (),
        compaction: CompactionEvent | None = None,
    ) -> None:
        return None


def _redact(outcome: ToolOutcome, verbose: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": outcome.name,
        "approved": outcome.approved,
        "is_error": outcome.is_error,
    }
    serialized = json.dumps(outcome.arguments, sort_keys=True, default=str)
    if verbose:
        entry["arguments"] = outcome.arguments  # raw, local debugging only
    else:
        entry["args_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()[:12]
        entry["args_len"] = len(serialized)
    return entry


class JsonlTelemetry:
    """Appends one JSON object per turn to ``<log_dir>/<session_id>.jsonl``."""

    def __init__(
        self,
        log_dir: str,
        model: str,
        *,
        session_id: str | None = None,
        prices: dict[str, Price] | None = None,
        verbose: bool = False,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._model = model
        self._prices = DEFAULT_PRICES if prices is None else prices
        self._verbose = verbose
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.session_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    @property
    def model(self) -> str:
        return self._model

    @property
    def prices(self) -> dict[str, Price]:
        return self._prices

    def record_turn(
        self,
        iteration: int,
        response: CompletionResponse,
        *,
        tool_outcomes: Sequence[ToolOutcome] = (),
        compaction: CompactionEvent | None = None,
    ) -> None:
        usage = response.usage
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "iteration": iteration,
            "model": self._model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "estimated_cost_usd": estimate_cost(self._model, usage, self._prices),
            "tool_calls": [_redact(o, self._verbose) for o in tool_outcomes],
            "stop_reason": response.stop_reason,
            "compaction": None
            if compaction is None
            else {
                "turns_collapsed": compaction.turns_collapsed,
                "tokens_before": compaction.tokens_before,
                "tokens_after": compaction.tokens_after,
            },
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
