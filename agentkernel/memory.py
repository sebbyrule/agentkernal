"""Persistent memory seam (Phase 3, design §13).

A ``MemoryStore`` loads relevant prior context before a run and saves the
conversation after a run. It is deliberately minimal: the kernel only defines the
interface; concrete stores decide what to persist and how to recall it.

All stores operate on canonical ``Message`` objects so the loop never learns
where memory came from.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agentkernel.types import Message


class MemoryStore(Protocol):
    """Pluggable memory: load before a run, save after it."""

    def load(self, session_id: str) -> list[Message]:
        """Return messages to inject before the current run."""
        ...

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        """Persist the messages from the just-finished run."""
        ...


@dataclass
class InMemoryMemoryStore:
    """Volatile memory for tests and ephemeral sessions."""

    _data: dict[str, list[Message]] = field(default_factory=dict)

    def load(self, session_id: str) -> list[Message]:
        return list(self._data.get(session_id, []))

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        self._data[session_id] = list(messages)


@dataclass
class FileMemoryStore:
    """Append-only JSONL memory on disk.

    Each line is one serialized ``Message``. Saving rewrites the file so the
    persisted view always matches the in-memory context for the session.
    """

    directory: str | Path

    def __post_init__(self) -> None:
        self._dir = Path(self.directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> list[Message]:
        path = self._path(session_id)
        if not path.is_file():
            return []
        messages: list[Message] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(Message.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue  # corrupted line; skip rather than crash
        return messages

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        path = self._path(session_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for message in messages:
                fh.write(json.dumps(message.to_dict()) + "\n")

    def _path(self, session_id: str) -> Path:
        # Sanitize session_id enough for a filename; UUIDs are the normal input.
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_.")
        return self._dir / f"{safe}.jsonl"


def make_memory_store(kind: str | None, directory: str | Path | None = None) -> MemoryStore | None:
    """Factory for the built-in memory stores."""
    if kind == "file":
        return FileMemoryStore(directory or ".agentkernel/memory")
    if kind == "memory":
        return InMemoryMemoryStore()
    return None
