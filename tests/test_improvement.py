"""Self-improvement tests (Phase 7, design §13).

``SelfImprover`` reflects on a structured session trace and emits a proposed
improvement rule. Tests use a scripted provider so no network calls are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentkernel.improvement import SelfImprover
from tests.fakes import FakeProvider, text_response


def _write_trace(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_improver_analyzes_trace_and_writes_file(tmp_path):
    trace = tmp_path / "session.jsonl"
    _write_trace(
        trace,
        {
            "model": "claude",
            "stop_reason": "tool_use",
            "tool_calls": [{"name": "bash", "approved": True, "is_error": True}],
            "assistant_message": "",
        },
        {
            "model": "claude",
            "stop_reason": "end_turn",
            "tool_calls": [],
            "assistant_message": "done",
        },
    )
    provider = FakeProvider([text_response("Validate bash commands before running them.")])
    improver = SelfImprover(provider, tmp_path / "improvements")
    improvement = improver.analyze_trace(trace)
    assert improvement.rule == "Validate bash commands before running them."
    assert improvement.output_path is not None
    assert Path(improvement.output_path).is_file()
    content = Path(improvement.output_path).read_text(encoding="utf-8")
    assert "Validate bash commands before running them." in content


def test_improver_picks_latest_trace(tmp_path):
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    _write_trace(old, {"model": "a", "stop_reason": "end_turn", "tool_calls": [], "assistant_message": ""})
    _write_trace(new, {"model": "b", "stop_reason": "end_turn", "tool_calls": [], "assistant_message": ""})
    # Ensure mtime ordering even on fast filesystems.
    old.touch()
    new.touch()
    provider = FakeProvider([text_response("foo")])
    improver = SelfImprover(provider, tmp_path / "improvements")
    assert improver.latest_trace(tmp_path).name == "new.jsonl"


def test_improver_returns_none_for_empty_log_dir(tmp_path):
    provider = FakeProvider([text_response("foo")])
    improver = SelfImprover(provider, tmp_path / "improvements")
    assert improver.latest_trace(tmp_path) is None
