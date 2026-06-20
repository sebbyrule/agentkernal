"""Usage insights from session traces (design §18.7).

Aggregates the JSONL telemetry under ``log_dir`` — the stable per-turn schema
from §12 — into a usage/cost/tool-frequency report. Pure reading: it never calls
a provider, so it works offline and costs nothing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass
class ModelStats:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class ToolStats:
    calls: int = 0
    errors: int = 0


@dataclass
class Insights:
    sessions: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost: float = 0.0
    compactions: int = 0
    models: dict[str, ModelStats] = field(default_factory=lambda: defaultdict(ModelStats))
    tools: dict[str, ToolStats] = field(default_factory=lambda: defaultdict(ToolStats))
    models_without_price: set[str] = field(default_factory=set)


def _within(ts: str, cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    try:
        return datetime.fromisoformat(ts) >= cutoff
    except (ValueError, TypeError):
        return True  # undated records are kept rather than silently dropped


def aggregate_traces(log_dir: str | Path, *, days: int | None = None) -> Insights:
    """Aggregate every ``*.jsonl`` trace under ``log_dir`` into one ``Insights``."""
    directory = Path(log_dir)
    cutoff = datetime.now(UTC) - timedelta(days=days) if days else None
    ins = Insights()
    if not directory.is_dir():
        return ins

    for trace in sorted(directory.glob("*.jsonl")):
        counted_session = False
        for line in trace.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _within(rec.get("ts", ""), cutoff):
                continue
            if not counted_session:
                ins.sessions += 1
                counted_session = True

            ins.turns += 1
            ins.input_tokens += rec.get("input_tokens", 0)
            ins.output_tokens += rec.get("output_tokens", 0)
            ins.cache_read_tokens += rec.get("cache_read_tokens", 0)
            ins.cache_write_tokens += rec.get("cache_write_tokens", 0)
            if rec.get("compaction"):
                ins.compactions += 1

            model = rec.get("model", "unknown")
            ms = ins.models[model]
            ms.turns += 1
            ms.input_tokens += rec.get("input_tokens", 0)
            ms.output_tokens += rec.get("output_tokens", 0)

            cost = rec.get("estimated_cost_usd")
            if cost is None:
                ins.models_without_price.add(model)
            else:
                ins.total_cost += cost
                ms.cost += cost

            for call in rec.get("tool_calls", []) or []:
                ts_ = ins.tools[call.get("name", "?")]
                ts_.calls += 1
                if call.get("is_error"):
                    ts_.errors += 1
    return ins


def format_insights(ins: Insights, *, days: int | None = None) -> str:
    """Render an ``Insights`` as a readable text report."""
    scope = f" (last {days} day(s))" if days else ""
    if ins.turns == 0:
        return f"No trace records found{scope}."

    lines = [f"Usage insights{scope}", ""]
    lines.append(f"  sessions:  {ins.sessions}")
    lines.append(f"  turns:     {ins.turns}")
    lines.append(
        f"  tokens:    in={ins.input_tokens:,} out={ins.output_tokens:,} "
        f"cache_read={ins.cache_read_tokens:,} cache_write={ins.cache_write_tokens:,}"
    )
    cost_note = "" if not ins.models_without_price else (
        f"  (excludes {len(ins.models_without_price)} model(s) with no price table)"
    )
    lines.append(f"  est. cost: ${ins.total_cost:.4f}{cost_note}")
    lines.append(f"  compactions: {ins.compactions}")

    lines.append("")
    lines.append("By model:")
    for name, ms in sorted(ins.models.items(), key=lambda kv: kv[1].turns, reverse=True):
        lines.append(
            f"  {name}: {ms.turns} turns, in={ms.input_tokens:,} out={ms.output_tokens:,}, "
            f"${ms.cost:.4f}"
        )

    if ins.tools:
        lines.append("")
        lines.append("Tool usage (most used first):")
        for name, ts_ in sorted(ins.tools.items(), key=lambda kv: kv[1].calls, reverse=True):
            err = f", {ts_.errors} error(s)" if ts_.errors else ""
            lines.append(f"  {name}: {ts_.calls} call(s){err}")
    return "\n".join(lines)
