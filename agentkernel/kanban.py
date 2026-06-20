"""A lightweight work-queue board for multi-agent coordination (design §18.3).

A durable JSON board of tasks that a long mission can fan out across — the parent
(or a human) files tasks, and workers (often spawned sub-agents) claim, work, and
complete or block them. Deliberately "lite": one JSON file, whole-file
read-modify-write, no daemon or dispatcher. For heavy multi-worker contention a
real SQLite board (cf. Hermes) would be the next step; this covers the common
case with a fraction of the surface area.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

STATUSES = ("todo", "in_progress", "done", "blocked")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Task:
    id: str
    title: str
    status: str = "todo"
    assignee: str | None = None
    notes: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""


class Board:
    """JSON-backed kanban board. Each mutation rewrites the whole file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    # --- persistence ------------------------------------------------------
    def _read(self) -> list[Task]:
        if not self._path.is_file():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [Task(**t) for t in raw if isinstance(t, dict)]

    def _write(self, tasks: list[Task]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(t) for t in tasks], indent=2), encoding="utf-8"
        )

    def _update(self, task_id: str, mutate) -> Task | None:
        tasks = self._read()
        for t in tasks:
            if t.id == task_id:
                mutate(t)
                t.updated = _now()
                self._write(tasks)
                return t
        return None

    # --- operations -------------------------------------------------------
    def list(self, status: str | None = None) -> list[Task]:
        tasks = self._read()
        return [t for t in tasks if status is None or t.status == status]

    def get(self, task_id: str) -> Task | None:
        return next((t for t in self._read() if t.id == task_id), None)

    def add(self, title: str) -> Task:
        task = Task(id=uuid.uuid4().hex[:8], title=title.strip(), created=_now(), updated=_now())
        tasks = self._read()
        tasks.append(task)
        self._write(tasks)
        return task

    def claim(self, task_id: str, assignee: str) -> Task | None:
        def _claim(t: Task) -> None:
            t.assignee = assignee
            t.status = "in_progress"
        return self._update(task_id, _claim)

    def complete(self, task_id: str) -> Task | None:
        return self._update(task_id, lambda t: setattr(t, "status", "done"))

    def block(self, task_id: str, reason: str) -> Task | None:
        def _block(t: Task) -> None:
            t.status = "blocked"
            if reason:
                t.notes.append(f"blocked: {reason}")
        return self._update(task_id, _block)

    def comment(self, task_id: str, text: str) -> Task | None:
        return self._update(task_id, lambda t: t.notes.append(text))

    def next_todo(self) -> Task | None:
        """The first unclaimed task, for a worker pulling work off the board."""
        return next((t for t in self._read() if t.status == "todo"), None)


_MARKS = {"todo": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]"}


def render_task(t: Task) -> str:
    mark = _MARKS.get(t.status, "[ ]")
    who = f" @{t.assignee}" if t.assignee else ""
    return f"{mark} {t.id}{who}  {t.title}"
