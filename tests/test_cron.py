"""Scheduled-job tests (design §18.2): interval parsing, the job store, due
logic, and the run_cron CLI helper with an injected runner (offline)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentkernel.cli import run_cron
from agentkernel.config import Config
from agentkernel.cron import CronJob, JobStore, parse_interval, run_due_jobs

# --- parsing ------------------------------------------------------------------

def test_parse_interval_units():
    assert parse_interval("30s") == timedelta(seconds=30)
    assert parse_interval("15m") == timedelta(minutes=15)
    assert parse_interval("2h") == timedelta(hours=2)
    assert parse_interval("1d") == timedelta(days=1)
    assert parse_interval("1w") == timedelta(weeks=1)


def test_parse_interval_invalid():
    for bad in ("", "5x", "abc", "h", "-3m"):
        with pytest.raises(ValueError):
            parse_interval(bad)


# --- due logic ----------------------------------------------------------------

def test_never_run_is_due():
    job = CronJob(id="a", schedule="1h", prompt="p")
    assert job.is_due(datetime.now(UTC)) is True


def test_not_due_before_interval_elapses():
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    job = CronJob(id="a", schedule="1h", prompt="p", last_run=now.isoformat())
    assert job.is_due(now + timedelta(minutes=30)) is False
    assert job.is_due(now + timedelta(hours=1)) is True


def test_disabled_never_due():
    job = CronJob(id="a", schedule="1h", prompt="p", enabled=False)
    assert job.is_due(datetime.now(UTC)) is False


# --- store --------------------------------------------------------------------

def _store(tmp_path):
    return JobStore(tmp_path / "cron.json")


def test_add_list_remove(tmp_path):
    store = _store(tmp_path)
    job = store.add("30m", "check the deploy")
    assert len(store.list()) == 1
    assert store.get(job.id).prompt == "check the deploy"
    assert store.remove(job.id) is True
    assert store.list() == []
    assert store.remove("ghost") is False


def test_add_rejects_bad_schedule(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).add("nope", "p")


def test_mark_run_updates_last_run(tmp_path):
    store = _store(tmp_path)
    job = store.add("1h", "p")
    assert store.get(job.id).last_run is None
    store.mark_run(job.id)
    assert store.get(job.id).last_run is not None


def test_due_jobs_filters(tmp_path):
    store = _store(tmp_path)
    j1 = store.add("1h", "due now")  # never run -> due
    store.add("1h", "not due")
    store.mark_run(store.list()[1].id)  # second job just ran -> not due
    due = store.due_jobs()
    assert [j.id for j in due] == [j1.id]


# --- run_due_jobs + run_cron --------------------------------------------------

def test_run_due_jobs_runs_and_marks(tmp_path):
    store = _store(tmp_path)
    store.add("1h", "first")
    store.add("1h", "second")
    ran = []
    results = run_due_jobs(store, lambda p: ran.append(p) or f"ok:{p}")
    assert set(ran) == {"first", "second"}
    assert len(results) == 2
    # both now marked run -> nothing due on the next tick
    assert run_due_jobs(store, lambda p: "x") == []


def test_run_due_jobs_records_error_and_continues(tmp_path):
    store = _store(tmp_path)
    store.add("1h", "boom")

    def boom(_p):
        raise RuntimeError("kaboom")

    results = run_due_jobs(store, boom)
    assert results and "[error]" in results[0][1]
    assert store.list()[0].last_run is not None  # marked so it won't re-fire


def test_run_cron_add_list_tick(tmp_path):
    cfg = Config(cron_path=str(tmp_path / "cron.json"))
    out: list[str] = []
    assert run_cron(cfg, "add", ["30m", "do", "the", "thing"], output_fn=out.append) == 0
    out.clear()
    run_cron(cfg, "list", [], output_fn=out.append)
    assert "every 30m" in out[0] and "do the thing" in out[0]
    out.clear()
    run_cron(cfg, "tick", [], output_fn=out.append, run_fn=lambda p: f"ran: {p}")
    assert any("ran: do the thing" in line for line in out)


def test_run_cron_tick_nothing_due(tmp_path):
    cfg = Config(cron_path=str(tmp_path / "cron.json"))
    out: list[str] = []
    run_cron(cfg, "tick", [], output_fn=out.append, run_fn=lambda p: "x")
    assert out == ["(nothing due)"]
