"""Scheduled jobs (design §18.2).

A durable, dependency-light scheduler: jobs live in a JSON file and run on a
fixed interval. There is no long-running daemon — ``agentkernel cron tick`` runs
whatever is due once and exits, so an OS scheduler (cron, systemd timer, Windows
Task Scheduler) can drive it. ``cron run <id>`` runs one job immediately.

Schedules are interval strings (``30s``, ``15m``, ``2h``, ``1d``, ``1w``). Full
5-field cron expressions are intentionally left for later — the interval covers
the common "every N" case with a fraction of the surface area.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$")


def parse_interval(schedule: str) -> timedelta:
    """Parse an interval like ``30m`` / ``2h`` into a timedelta. Raises ValueError."""
    match = _INTERVAL_RE.match(schedule or "")
    if not match:
        raise ValueError(
            f"invalid schedule {schedule!r}; use forms like '30s', '15m', '2h', '1d', '1w'"
        )
    return timedelta(seconds=int(match.group(1)) * _UNIT_SECONDS[match.group(2)])


@dataclass
class CronJob:
    id: str
    schedule: str
    prompt: str
    enabled: bool = True
    last_run: str | None = None  # ISO-8601, UTC
    created: str = ""

    def next_run(self) -> datetime | None:
        """When this job is next due, or None if it has never run (due now)."""
        if self.last_run is None:
            return None
        try:
            return datetime.fromisoformat(self.last_run) + parse_interval(self.schedule)
        except ValueError:
            return None

    def is_due(self, now: datetime) -> bool:
        if not self.enabled:
            return False
        nxt = self.next_run()
        return nxt is None or now >= nxt


class JobStore:
    """JSON-backed store of cron jobs."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _read(self) -> list[CronJob]:
        if not self._path.is_file():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [CronJob(**j) for j in raw if isinstance(j, dict)]

    def _write(self, jobs: list[CronJob]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(j) for j in jobs], indent=2), encoding="utf-8"
        )

    def list(self) -> list[CronJob]:
        return self._read()

    def get(self, job_id: str) -> CronJob | None:
        return next((j for j in self._read() if j.id == job_id), None)

    def add(self, schedule: str, prompt: str, *, enabled: bool = True) -> CronJob:
        parse_interval(schedule)  # validate up front
        job = CronJob(
            id=uuid.uuid4().hex[:8],
            schedule=schedule.strip(),
            prompt=prompt,
            enabled=enabled,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        jobs = self._read()
        jobs.append(job)
        self._write(jobs)
        return job

    def remove(self, job_id: str) -> bool:
        jobs = self._read()
        kept = [j for j in jobs if j.id != job_id]
        if len(kept) == len(jobs):
            return False
        self._write(kept)
        return True

    def set_enabled(self, job_id: str, enabled: bool) -> bool:
        return self._update(job_id, lambda j: setattr(j, "enabled", enabled))

    def mark_run(self, job_id: str, when: datetime | None = None) -> bool:
        stamp = (when or datetime.now(UTC)).isoformat(timespec="seconds")
        return self._update(job_id, lambda j: setattr(j, "last_run", stamp))

    def _update(self, job_id: str, mutate) -> bool:
        jobs = self._read()
        found = False
        for j in jobs:
            if j.id == job_id:
                mutate(j)
                found = True
        if found:
            self._write(jobs)
        return found

    def due_jobs(self, now: datetime | None = None) -> list[CronJob]:
        now = now or datetime.now(UTC)
        return [j for j in self._read() if j.is_due(now)]


def run_due_jobs(store: JobStore, run_fn, *, now: datetime | None = None) -> list[tuple[str, str]]:
    """Run every due job via ``run_fn(prompt) -> str``; mark each run. Returns
    ``(job_id, result)`` pairs. A job whose run raises is recorded as run anyway
    (with the error as its result) so one bad job doesn't re-fire every tick."""
    now = now or datetime.now(UTC)
    results: list[tuple[str, str]] = []
    for job in store.due_jobs(now):
        try:
            result = run_fn(job.prompt)
        except Exception as exc:  # noqa: BLE001 - record and continue to next job
            result = f"[error] {exc}"
        store.mark_run(job.id, now)
        results.append((job.id, result))
    return results
