"""Self-improvement seam (design §13, Phase 7).

The kernel records structured telemetry from turn one. ``SelfImprover`` reads a
session trace, asks the configured provider to suggest one concise rule or
system-prompt addition, and writes the result as a markdown note in
``.agentkernel/improvements``. It is intentionally lightweight — enough to close
the loop, with room for a future richer analyzer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentkernel.providers import Provider


_REFLECTION_SYSTEM_PROMPT = (
    "You are a self-improvement analyst for an agent kernel. "
    "Given a session trace, propose one concise rule, instruction, or "
    "system-prompt addition that would improve future runs. "
    "Return only the rule text followed by a brief rationale."
)


@dataclass
class Improvement:
    suggestion: str
    rule: str
    trace_path: str
    output_path: str | None = None


def _load_trace(path: Path) -> list[dict[str, Any]]:
    """Load the JSONL trace written by ``JsonlTelemetry``."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _summarize_trace(records: list[dict[str, Any]]) -> str:
    """Build a compact textual summary suitable for an LLM prompt."""
    lines: list[str] = []
    for i, record in enumerate(records):
        lines.append(f"--- turn {i} ---")
        lines.append(f"model: {record.get('model', 'unknown')}")
        lines.append(f"stop_reason: {record.get('stop_reason', 'unknown')}")
        for call in record.get("tool_calls", []):
            lines.append(f"tool: {call.get('name')} approved={call.get('approved')} error={call.get('is_error')}")
        # Note: the redacted JSONL trace (design §12) does not carry assistant
        # text or raw tool args, so reflection works from the structural signal
        # — tools used, errors, stop reasons, and token/cost figures.
        cost = record.get("estimated_cost_usd")
        if cost is not None:
            lines.append(f"cost_usd: {cost}")
    return "\n".join(lines)


class SelfImprover:
    """Reflect on a completed session and emit a proposed improvement note."""

    def __init__(self, provider: Provider, output_dir: str | Path) -> None:
        self.provider = provider
        self.output_dir = Path(output_dir)

    def analyze_trace(self, trace_path: str | Path) -> Improvement:
        trace_path = Path(trace_path)
        records = _load_trace(trace_path)
        summary = _summarize_trace(records)
        prompt = (
            f"Session trace summary:\n{summary}\n\n"
            "Propose one concise improvement rule for the agent. "
            "Start with the rule, then a one-line rationale."
        )

        from agentkernel.types import Message

        messages = [Message(role="user", content=prompt)]
        response = self.provider.complete(
            messages,
            [],
            max_tokens=1024,
            system=_REFLECTION_SYSTEM_PROMPT,
        )
        suggestion = response.message.content.strip()
        rule = suggestion.splitlines()[0] if suggestion else suggestion

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = self.output_dir / f"improvement-{timestamp}.md"
        output_path.write_text(
            f"---\n"
            f"type: improvement\n"
            f"trace: {trace_path}\n"
            f"timestamp: {timestamp}\n"
            f"---\n\n"
            f"{suggestion}\n",
            encoding="utf-8",
        )

        return Improvement(
            suggestion=suggestion,
            rule=rule,
            trace_path=str(trace_path),
            output_path=str(output_path),
        )

    def latest_trace(self, log_dir: str | Path) -> Path | None:
        """Return the most recent ``*.jsonl`` trace in ``log_dir``."""
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            return None
        traces = sorted(
            (p for p in log_dir.iterdir() if p.suffix == ".jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return traces[0] if traces else None
