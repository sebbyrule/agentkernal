"""Tests for §18.7 observability/DX: insights aggregation, doctor checks, and
plugin tool discovery."""

from __future__ import annotations

import json

from agentkernel.config import Config
from agentkernel.doctor import has_failures, run_checks
from agentkernel.insights import aggregate_traces, format_insights
from agentkernel.plugins import load_plugin_tools
from agentkernel.tools.base import ToolSpec

# --- insights -----------------------------------------------------------------

def _trace(tmp_path, name, records):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_insights_aggregates_tokens_cost_and_tools(tmp_path):
    _trace(tmp_path, "s1", [
        {"ts": "2026-06-01T10:00:00+00:00", "model": "claude-sonnet-4-6",
         "input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 0,
         "cache_write_tokens": 0, "estimated_cost_usd": 0.001,
         "tool_calls": [{"name": "read_file", "is_error": False},
                        {"name": "bash", "is_error": True}], "compaction": None},
        {"ts": "2026-06-01T10:01:00+00:00", "model": "claude-sonnet-4-6",
         "input_tokens": 50, "output_tokens": 10, "cache_read_tokens": 0,
         "cache_write_tokens": 0, "estimated_cost_usd": 0.0005,
         "tool_calls": [{"name": "read_file", "is_error": False}],
         "compaction": {"turns_collapsed": 2}},
    ])
    ins = aggregate_traces(tmp_path)
    assert ins.sessions == 1
    assert ins.turns == 2
    assert ins.input_tokens == 150 and ins.output_tokens == 30
    assert abs(ins.total_cost - 0.0015) < 1e-9
    assert ins.compactions == 1
    assert ins.tools["read_file"].calls == 2 and ins.tools["read_file"].errors == 0
    assert ins.tools["bash"].calls == 1 and ins.tools["bash"].errors == 1
    assert ins.models["claude-sonnet-4-6"].turns == 2


def test_insights_unknown_model_cost_excluded(tmp_path):
    _trace(tmp_path, "s", [
        {"ts": "2026-06-01T10:00:00+00:00", "model": "mystery",
         "input_tokens": 1, "output_tokens": 1, "estimated_cost_usd": None,
         "tool_calls": [], "compaction": None},
    ])
    ins = aggregate_traces(tmp_path)
    assert "mystery" in ins.models_without_price
    assert ins.total_cost == 0.0


def test_insights_days_filter(tmp_path):
    from datetime import UTC, datetime, timedelta

    old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    recent = datetime.now(UTC).isoformat()
    _trace(tmp_path, "s", [
        {"ts": old, "model": "m", "input_tokens": 1, "output_tokens": 1,
         "estimated_cost_usd": 0, "tool_calls": [], "compaction": None},
        {"ts": recent, "model": "m", "input_tokens": 2, "output_tokens": 2,
         "estimated_cost_usd": 0, "tool_calls": [], "compaction": None},
    ])
    ins = aggregate_traces(tmp_path, days=3)
    assert ins.turns == 1 and ins.input_tokens == 2


def test_insights_empty_dir(tmp_path):
    assert "No trace records" in format_insights(aggregate_traces(tmp_path))


# --- doctor -------------------------------------------------------------------

def test_doctor_local_provider_with_base_url_ok():
    checks = run_checks(Config(provider="local", base_url="http://x:1234/v1"), env={})
    by_name = {c.name: c for c in checks}
    assert by_name["provider: local"].status == "ok"
    assert by_name["python"].status == "ok"
    assert not has_failures([c for c in checks if c.name != "provider: local"])


def test_doctor_missing_api_key_fails():
    checks = run_checks(Config(provider="anthropic"), env={})  # no key in env
    provider = next(c for c in checks if c.name == "provider: anthropic")
    assert provider.status == "fail"
    assert has_failures(checks)


def test_doctor_api_key_present_ok():
    checks = run_checks(Config(provider="openai"), env={"OPENAI_API_KEY": "x"})
    provider = next(c for c in checks if c.name == "provider: openai")
    assert provider.status == "ok"


def test_doctor_semantic_search_warns_without_key():
    checks = run_checks(
        Config(provider="local", base_url="u", semantic_search=True), env={}
    )
    sem = next(c for c in checks if c.name == "semantic search")
    assert sem.status == "warn"


# --- plugins ------------------------------------------------------------------

_PLUGIN_TOOLS_FN = '''
from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

def tools(working_dir="."):
    return [ToolSpec(
        name="plugin_echo",
        description="echo",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda a: ToolResult("", "hi"),
    )]
'''

_PLUGIN_TABLE = '''
from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

TOOLS = [ToolSpec(
    name="plugin_table",
    description="t",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=lambda a: ToolResult("", "t"),
)]
'''


def test_plugin_discovery_loads_tools(tmp_path):
    (tmp_path / "a.py").write_text(_PLUGIN_TOOLS_FN, encoding="utf-8")
    (tmp_path / "b.py").write_text(_PLUGIN_TABLE, encoding="utf-8")
    (tmp_path / "_skip.py").write_text("raise RuntimeError('should be skipped')", encoding="utf-8")
    specs = load_plugin_tools(tmp_path)
    names = {s.name for s in specs}
    assert names == {"plugin_echo", "plugin_table"}
    assert all(isinstance(s, ToolSpec) for s in specs)


def test_plugin_bad_module_is_reported_not_raised(tmp_path):
    (tmp_path / "broken.py").write_text("import nonexistent_module_xyz", encoding="utf-8")
    errors = []
    specs = load_plugin_tools(tmp_path, on_error=lambda p, e: errors.append(p.name))
    assert specs == []
    assert errors == ["broken.py"]


def test_plugin_dir_absent_returns_empty(tmp_path):
    assert load_plugin_tools(tmp_path / "nope") == []
