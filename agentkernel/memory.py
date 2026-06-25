"""Persistent memory seam (Phase 3, design \u00a713).

A ``MemoryStore`` loads relevant prior context before a run and saves the
conversation after a run. It is deliberately minimal: the kernel only defines the
interface; concrete stores decide what to persist and how to recall it.

All stores operate on canonical ``Message`` objects so the loop never learns
where memory came from.

This module also exposes a model-controlled ``NoteStore``: discrete facts the
*model* chooses to write and read on demand (``remember`` / ``recall`` tools).
The default backend is an append-only JSONL notebook; a SQLite-backed store is
available for unified, full-text-searchable storage.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
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

    def delete(self, session_id: str) -> None:
        """Remove any persisted messages for ``session_id``."""
        ...

    def list_sessions(self) -> list[str]:
        """Return known session ids (for management / cleanup)."""
        ...


class NoteStore(Protocol):
    """Pluggable notebook of discrete facts the model reads and writes."""

    def add(
        self, text: str, *, tags: Sequence[str] | None = None, scope: str | None = None
    ) -> MemoryNote:
        ...

    def all(self) -> list[MemoryNote]:
        ...

    def recent(self, limit: int = 5) -> list[MemoryNote]:
        ...

    def search(self, query: str, *, limit: int = 5) -> list[MemoryNote]:
        ...

    def forget(
        self, *, note_id: int | None = None, text_prefix: str | None = None
    ) -> list[MemoryNote]:
        ...

    def update(
        self, note_id: int, text: str, *, tags: Sequence[str] | None = None
    ) -> MemoryNote | None:
        ...

    def deduplicate(self) -> int:
        ...

    def export(self, destination: str | Path) -> Path:
        ...


@dataclass
class InMemoryMemoryStore:
    """Volatile memory for tests and ephemeral sessions."""

    _data: dict[str, list[Message]] = field(default_factory=dict)

    def load(self, session_id: str) -> list[Message]:
        return list(self._data.get(session_id, []))

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        self._data[session_id] = list(messages)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return sorted(self._data)


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

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.is_file():
            path.unlink()

    def list_sessions(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        sessions: list[str] = []
        for path in self._dir.iterdir():
            if path.suffix == ".jsonl" and path.name != "notes.jsonl":
                sessions.append(path.stem)
        return sorted(sessions)

    def _path(self, session_id: str) -> Path:
        # Sanitize session_id enough for a filename; UUIDs are the normal input.
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_.")
        return self._dir / f"{safe}.jsonl"


@dataclass
class SqliteMemoryStore:
    """SQLite-backed session memory.

    Messages are stored relationally with optional FTS5 content search across
    session transcripts. ``sqlite3`` is part of the Python stdlib, so this adds
    no external dependency. If FTS5 is unavailable in the local build, the store
    falls back to relational storage and search methods use a LIKE fallback.
    """

    path: str | Path

    def __post_init__(self) -> None:
        self._path = Path(self.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._fts_enabled: bool | None = None
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                position INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_position
                ON messages(session_id, position);
            """
        )
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content)"
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
        conn.commit()

    def load(self, session_id: str) -> list[Message]:
        rows = self._connection().execute(
            """
            SELECT payload_json
            FROM messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        messages: list[Message] = []
        for row in rows:
            try:
                messages.append(Message.from_dict(json.loads(row["payload_json"])))
            except (json.JSONDecodeError, KeyError):
                continue  # skip corrupt records rather than crash
        return messages

    def save(self, session_id: str, messages: Sequence[Message]) -> None:
        conn = self._connection()
        with conn:
            existing_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM messages WHERE session_id = ?", (session_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            if self._fts_enabled:
                for mid in existing_ids:
                    conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (mid,))
            conn.execute(
                "INSERT OR REPLACE INTO sessions(session_id) VALUES (?)", (session_id,)
            )
            for position, message in enumerate(messages):
                cursor = conn.execute(
                    """
                    INSERT INTO messages
                        (session_id, content, payload_json, position)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        message.content,
                        json.dumps(message.to_dict()),
                        position,
                    ),
                )
                if self._fts_enabled:
                    conn.execute(
                        "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                        (cursor.lastrowid, message.content),
                    )

    def delete(self, session_id: str) -> None:
        conn = self._connection()
        with conn:
            existing_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM messages WHERE session_id = ?", (session_id,)
                ).fetchall()
            ]
            if self._fts_enabled:
                for mid in existing_ids:
                    conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (mid,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def list_sessions(self) -> list[str]:
        rows = self._connection().execute(
            "SELECT session_id FROM sessions ORDER BY session_id"
        ).fetchall()
        return [r["session_id"] for r in rows]

    def search_sessions(self, query: str, limit: int = 10) -> list[str]:
        """Return session_ids whose messages match ``query``.

        Uses FTS5 MATCH when available; otherwise falls back to substring search
        on message contents.
        """
        query = query.strip()
        if not query or not self._has_messages():
            return []
        conn = self._connection()
        if self._fts_enabled:
            try:
                rows = conn.execute(
                    """
                    SELECT DISTINCT m.session_id
                    FROM messages_fts f
                    JOIN messages m ON f.rowid = m.id
                    WHERE f MATCH ?
                    ORDER BY m.session_id
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
                return [r["session_id"] for r in rows]
            except sqlite3.OperationalError:
                pass  # malformed FTS5 query; fall through to LIKE
        like = f"%{query}%"
        rows = conn.execute(
            """
            SELECT DISTINCT session_id
            FROM messages
            WHERE content LIKE ?
            ORDER BY session_id
            LIMIT ?
            """,
            (like, limit),
        ).fetchall()
        return [r["session_id"] for r in rows]

    def _has_messages(self) -> bool:
        return (
            self._connection()
            .execute("SELECT 1 FROM messages LIMIT 1")
            .fetchone()
            is not None
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def make_memory_store(kind: str | None, directory: str | Path | None = None) -> MemoryStore | None:
    """Factory for the built-in memory stores."""
    if kind == "file":
        return FileMemoryStore(directory or ".agentkernel/memory")
    if kind == "sqlite":
        path = Path(directory or ".agentkernel/memory") / "memory.db"
        return SqliteMemoryStore(path)
    if kind == "memory":
        return InMemoryMemoryStore()
    return None


# --- Token normalization for keyword search ---------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_token(token: str) -> str:
    """Simple English-lite stemmer for better keyword recall.

    Handles plurals, possessives, and the most common verb suffixes. This is a
    dependency-free approximation; it intentionally keeps false positives low
    for short tokens and avoids normalizing away useful distinctions.
    """
    if len(token) <= 3:
        return token
    if token.endswith("'s"):
        token = token[:-2]
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("ses") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        token = token[:-1]
    if token.endswith("ying") and len(token) > 5:
        token = token[:-4] + "ie"
    elif token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ied") and len(token) > 5:
        token = token[:-3] + "y"
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    return token


def _tokens(text: str) -> set[str]:
    return {_normalize_token(t) for t in _TOKEN_RE.findall(text.lower())}


# --- Model-controlled memory: remember / recall / forget tools ---------------
#
# Distinct from MemoryStore (which auto-loads/saves a session transcript): this
# is a persistent notebook of discrete facts the *model* chooses to write and
# read on demand, exposed as ordinary tools (the "everything is a tool"
# principle). The notebook is append-only JSONL and shared across sessions, so a
# fact remembered in one session is recallable in the next.


@dataclass
class MemoryNote:
    """One remembered fact."""

    text: str
    tags: list[str] = field(default_factory=list)
    created: str = ""
    note_id: int = 0
    accessed: str = ""  # ISO timestamp of last recall/update (P1 metadata)
    access_count: int = 0  # how many times the note has been recalled (P1)
    scope: str = ""  # namespace; "" = global, recallable from any scope

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tags": self.tags,
            "created": self.created,
            "note_id": self.note_id,
            "accessed": self.accessed,
            "access_count": self.access_count,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryNote:
        return cls(
            text=data.get("text", ""),
            tags=list(data.get("tags", [])),
            created=data.get("created", ""),
            note_id=data.get("note_id", 0),
            accessed=data.get("accessed", ""),
            access_count=data.get("access_count", 0),
            scope=data.get("scope", ""),
        )


@dataclass(frozen=True)
class RecallWeighting:
    """Recency- and importance-aware re-ranking of recalled notes.

    Relevance stays the gate: each note's base relevance score is multiplied by
    ``1 + recency_weight*recency + importance_weight*importance`` and the set is
    re-sorted. A note the query does not match (base 0) therefore never
    surfaces, but among relevant notes a newer or more-often-recalled one is
    nudged ahead. ``recency`` decays exponentially with the note's age
    (``half_life_days`` per halving, measured from its ``created`` time);
    ``importance`` rises with the recall ``access_count`` that the stores
    already track but never used for ranking.

    Recency is measured from ``created`` rather than last access on purpose:
    ``search`` touches every candidate (updating ``accessed`` to *now*) before
    ranking, so an access-based recency would read as fresh for all of them.
    "Often recalled" is captured by the separate importance term instead.

    With both weights ``0`` (the default) the weighting is *inactive*: ``rerank``
    returns the notes in exactly the order the caller supplied, so existing
    keyword/TF-IDF/semantic ordering is unchanged unless a user opts in.
    """

    recency_weight: float = 0.0
    importance_weight: float = 0.0
    half_life_days: float = 30.0

    @property
    def active(self) -> bool:
        return bool(self.recency_weight or self.importance_weight)

    def _recency(self, note: MemoryNote, now: datetime) -> float:
        ts = note.created
        if not ts or self.half_life_days <= 0:
            return 0.0
        try:
            when = datetime.fromisoformat(ts)
        except ValueError:
            return 0.0
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        age_days = max(0.0, (now - when).total_seconds() / 86400.0)
        return 0.5 ** (age_days / self.half_life_days)

    @staticmethod
    def _importance(note: MemoryNote) -> float:
        # Diminishing returns: 0 recalls -> 0.0, many recalls -> approaches 1.0.
        return 1.0 - 1.0 / (1.0 + max(0, note.access_count))

    def rerank(
        self,
        scored: Sequence[tuple[float, MemoryNote]],
        *,
        limit: int,
        now: datetime | None = None,
    ) -> list[MemoryNote]:
        """Re-order ``(base_score, note)`` pairs and return the top ``limit`` notes.

        ``scored`` must already be in the caller's preferred order so the
        inactive path is a faithful identity. The active path re-sorts by the
        weighted score (stable, so the supplied order breaks ties).
        """
        pairs = list(scored)
        if not self.active:
            return [note for _base, note in pairs[:limit]]
        now = now or datetime.now(UTC)
        weighted: list[tuple[float, MemoryNote]] = []
        for base, note in pairs:
            multiplier = (
                1.0
                + self.recency_weight * self._recency(note, now)
                + self.importance_weight * self._importance(note)
            )
            weighted.append((base * multiplier, note))
        weighted.sort(key=lambda item: item[0], reverse=True)
        return [note for _score, note in weighted[:limit]]


def _scope_visible(note_scope: str, active: str | None) -> bool:
    """Whether a note in ``note_scope`` is visible under the ``active`` namespace.

    ``active is None`` means scoping is disabled — every note is visible. A
    global note (``scope == ""``) is always visible; otherwise the note's scope
    must equal the active namespace.
    """
    return active is None or note_scope == "" or note_scope == active


class JsonlNoteStore:
    """Append-only notebook of facts with keyword + recency recall.

    Persistence is a single JSONL file (one note per line). Recall scores notes
    by token overlap with the query and breaks ties toward more recent notes; an
    empty query returns the most recent notes.

    Notes are assigned stable IDs so they can be listed, updated, or forgotten
    from the REPL or by the model via ordinary tools.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        weighting: RecallWeighting | None = None,
        scope: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._weighting = weighting or RecallWeighting()
        self._scope = scope  # None = scoping disabled
        self._notes: list[MemoryNote] = []
        self._next_id: int = 1
        if self.path.is_file():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    note = MemoryNote.from_dict(json.loads(line))
                except (json.JSONDecodeError, AttributeError):
                    continue  # skip a corrupt line rather than crash
                self._notes.append(note)
                if note.note_id >= self._next_id:
                    self._next_id = note.note_id + 1

    def _append_line(self, note: MemoryNote) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(note.to_dict()) + "\n")

    def _rewrite(self) -> None:
        """Rewrite the file after a deletion or update."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for note in self._notes:
                fh.write(json.dumps(note.to_dict()) + "\n")

    def _touch(self, note: MemoryNote) -> None:
        """Record that a note was recalled. Updated in memory; a subsequent
        rewrite will persist the new access metadata."""
        note.access_count += 1
        note.accessed = datetime.now(UTC).isoformat()

    def add(
        self,
        text: str,
        tags: Sequence[str] | None = None,
        *,
        note_id: int | None = None,
        scope: str | None = None,
    ) -> MemoryNote:
        note = MemoryNote(
            text=text.strip(),
            tags=[str(t) for t in (tags or [])],
            created=datetime.now(UTC).isoformat(),
            note_id=note_id if note_id is not None else self._next_id,
            scope=scope if scope is not None else (self._scope or ""),
        )
        if note.note_id >= self._next_id:
            self._next_id = note.note_id + 1
        self._notes.append(note)
        self._append_line(note)
        return note

    def all(self) -> list[MemoryNote]:
        return list(self._notes)

    def recent(self, limit: int = 5) -> list[MemoryNote]:
        visible = [n for n in self._notes if _scope_visible(n.scope, self._scope)]
        results = visible[-limit:][::-1]  # newest first
        for note in results:
            self._touch(note)
        return results

    def _tfidf_vectors(self) -> tuple[dict[str, float], list[dict[str, float]]]:
        """Compute sparse TF-IDF vectors for all notes.

        Returns ``(idf, vectors)`` where each vector maps normalized token to
        its TF-IDF weight. This is a dependency-free semantic approximation;
        it ranks notes by cosine similarity rather than raw keyword overlap.
        """
        df: dict[str, int] = {}
        doc_tokens: list[set[str]] = []
        for note in self._notes:
            tokens = _tokens(note.text) | {_normalize_token(t) for t in note.tags}
            doc_tokens.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        num_docs = len(self._notes)
        idf = {t: math.log((num_docs + 1) / (df[t] + 1)) + 1 for t in df}
        vectors: list[dict[str, float]] = []
        for tokens in doc_tokens:
            total = len(tokens) or 1
            vectors.append({t: (1 / total) * idf[t] for t in tokens})
        return idf, vectors

    def search(self, query: str, limit: int = 5) -> list[MemoryNote]:
        terms = _tokens(query)
        if not terms:
            return self.recent(limit)
        idf, vectors = self._tfidf_vectors()
        query_total = len(terms) or 1
        query_vec = {t: (1 / query_total) * idf.get(t, 0) for t in terms}
        query_norm = math.sqrt(sum(v * v for v in query_vec.values())) or 1.0
        scored: list[tuple[float, int, MemoryNote]] = []
        for index, (note, vec) in enumerate(zip(self._notes, vectors, strict=True)):
            if not vec or not _scope_visible(note.scope, self._scope):
                continue
            dot = sum(query_vec.get(t, 0) * vec.get(t, 0) for t in terms)
            note_norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            similarity = dot / (query_norm * note_norm)
            if similarity > 0:
                self._touch(note)
                scored.append((similarity, index, note))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        pairs = [(score, note) for score, _index, note in scored]
        return self._weighting.rerank(pairs, limit=limit)

    def deduplicate(self) -> int:
        """Merge notes with identical text, combining tags and keeping the oldest id.

        Returns the number of notes removed.
        """
        seen: dict[str, MemoryNote] = {}
        kept: list[MemoryNote] = []
        removed = 0
        for note in self._notes:
            text = note.text.strip().lower()
            if text in seen:
                existing = seen[text]
                existing.tags = sorted(set(existing.tags) | set(note.tags))
                if note.access_count:
                    existing.access_count += note.access_count
                removed += 1
            else:
                seen[text] = note
                kept.append(note)
        if removed:
            self._notes = kept
            self._rewrite()
        return removed

    def forget(
        self, *, note_id: int | None = None, text_prefix: str | None = None
    ) -> list[MemoryNote]:
        """Remove notes matching ``note_id`` or whose text starts with ``text_prefix``.

        Returns the notes that were removed.
        """
        removed: list[MemoryNote] = []
        remaining: list[MemoryNote] = []
        prefix = (text_prefix or "").strip().lower()
        for note in self._notes:
            if (note_id is not None and note.note_id == note_id) or (
                prefix and note.text.lower().startswith(prefix)
            ):
                removed.append(note)
            else:
                remaining.append(note)
        self._notes = remaining
        if removed:
            self._rewrite()
        return removed

    def update(
        self, note_id: int, text: str, tags: Sequence[str] | None = None
    ) -> MemoryNote | None:
        """Replace the text/tags of an existing note in place."""
        for note in self._notes:
            if note.note_id == note_id:
                note.text = text.strip()
                note.tags = [str(t) for t in (tags or [])]
                note.accessed = datetime.now(UTC).isoformat()
                self._rewrite()
                return note
        return None

    def export(self, destination: str | Path) -> Path:
        """Write a human-readable markdown summary of all notes."""
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = ["# Memory Notes\n"]
        for note in self._notes:
            tag_part = f" *[{', '.join(note.tags)}]*" if note.tags else ""
            access_part = f", accessed {note.access_count} time(s)" if note.access_count else ""
            lines.append(
                f"- {note.text}{tag_part}  (id={note.note_id}, {note.created}{access_part})\n"
            )
        dest.write_text("".join(lines), encoding="utf-8")
        return dest


# Backward-compatible alias for code that directly constructs the JSONL notebook.
MemoryNotes = JsonlNoteStore


def make_note_store(
    path: str | Path,
    *,
    weighting: RecallWeighting | None = None,
    scope: str | None = None,
) -> NoteStore:
    """Select a note store backend based on the path extension.

    Paths ending in ``.db``/``.sqlite``/``.sqlite3`` become a SQLite-backed
    store; anything else uses the original JSONL notebook. ``weighting`` tunes
    recency/importance-aware recall (inactive by default); ``scope`` restricts
    recall to the global + named namespace (``None`` disables scoping).
    """
    p = Path(path)
    if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        return SqliteNoteStore(p, weighting=weighting, scope=scope)
    return JsonlNoteStore(p, weighting=weighting, scope=scope)


def make_memory_tools(notes: NoteStore, store: MemoryStore | None = None) -> list:
    """Build the memory-management tools over a notebook and optional session store.

    ``remember`` is ungated: writes go only to the dedicated notebook file, so
    the model can manage its own memory autonomously (cf. Anthropic's memory
    tool) without an approval prompt per fact.
    """
    from agentkernel.tools.base import ToolSpec
    from agentkernel.types import ToolResult

    def remember(arguments: dict) -> ToolResult:
        text = arguments["text"]
        # `global` forces the empty (universal) scope so the fact is recalled from
        # every namespace, even when this run is scoped to one project; otherwise
        # the note inherits the store's active scope.
        scope = "" if arguments.get("global") else None
        note = notes.add(text, tags=arguments.get("tags"), scope=scope)
        suffix = f" [tags: {', '.join(note.tags)}]" if note.tags else ""
        scope_note = " (global)" if note.scope == "" and arguments.get("global") else ""
        return ToolResult("", f"Remembered{scope_note}: {note.text}{suffix}")

    def recall(arguments: dict) -> ToolResult:
        query = arguments.get("query", "") or ""
        limit = int(arguments.get("limit", 5))
        results = notes.search(query, limit=limit) if query else notes.recent(limit)
        if not results:
            return ToolResult("", "(no relevant memories)")
        lines = [
            f"- [{n.note_id}] {n.text}" + (f"  [tags: {', '.join(n.tags)}]" if n.tags else "")
            for n in results
        ]
        return ToolResult("", "\n".join(lines))

    def forget(arguments: dict) -> ToolResult:
        note_id = arguments.get("note_id")
        if note_id is not None:
            note_id = int(note_id)
        removed = notes.forget(note_id=note_id, text_prefix=arguments.get("text_prefix", ""))
        if not removed:
            return ToolResult("", "(no matching memories)")
        return ToolResult("", f"Forgot {len(removed)} memory(s).")

    def update_memory(arguments: dict) -> ToolResult:
        note_id = int(arguments["note_id"])
        note = notes.update(note_id, arguments["text"], tags=arguments.get("tags"))
        if note is None:
            return ToolResult("", f"No note with id={note_id}.", is_error=True)
        return ToolResult("", f"Updated note {note_id}.")

    def memory_stats(arguments: dict) -> ToolResult:
        total = len(notes.all())
        if not total:
            return ToolResult("", "No memory notes stored yet.")
        by_access = sorted(notes.all(), key=lambda n: n.access_count, reverse=True)[:5]
        lines = [f"Total notes: {total}"]
        if by_access and by_access[0].access_count:
            lines.append("Most recalled:")
            lines.extend(f"  [{n.note_id}] {n.text} ({n.access_count})" for n in by_access)
        newest = notes.recent(1)[0] if notes.all() else None
        if newest:
            lines.append(f"Newest note: [{newest.note_id}] {newest.text} ({newest.created})")
        return ToolResult("", "\n".join(lines))

    def deduplicate_memory(arguments: dict) -> ToolResult:
        removed = notes.deduplicate()
        return ToolResult(
            "", f"Removed {removed} duplicate note(s). {len(notes.all())} unique note(s) remain."
        )

    tools = [
        ToolSpec(
            name="remember",
            description=(
                "Save a durable fact to long-term memory (persists across "
                "sessions). Use for stable user preferences, project facts, and "
                "decisions worth recalling later — not transient chatter. Set "
                "global=true for a fact true everywhere (e.g. a user preference) "
                "so it is recalled from every project, not just this one."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The fact to remember."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional keywords to aid later recall.",
                    },
                    "global": {
                        "type": "boolean",
                        "description": (
                            "Save as a universal fact recalled from every project "
                            "scope (default: inherit the current scope)."
                        ),
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=remember,
            category="memory",
        ),
        ToolSpec(
            name="recall",
            description=(
                "Search long-term memory for relevant facts. Provide a query to "
                "find related notes, or omit it for the most recent ones. Note IDs "
                "are shown so you can update or forget them later."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "limit": {"type": "integer", "description": "Max notes to return."},
                },
                "additionalProperties": False,
            },
            handler=recall,
            category="memory",
        ),
        ToolSpec(
            name="forget",
            description=(
                "Remove one or more durable facts from long-term memory. Match by "
                "exact note_id (preferred) or by deleting every note whose text "
                "starts with text_prefix."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "integer",
                        "description": "Exact id of the note to remove.",
                    },
                    "text_prefix": {
                        "type": "string",
                        "description": "Remove notes whose text starts with this string.",
                    },
                },
                "additionalProperties": False,
            },
            handler=forget,
            category="memory",
        ),
        ToolSpec(
            name="update_memory",
            description=(
                "Replace the text and optional tags of an existing memory note "
                "by its note_id. Use when a fact changes rather than deleting and "
                "re-adding it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "integer",
                        "description": "Exact id of the note to update.",
                    },
                    "text": {"type": "string", "description": "New note text."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional replacement tags.",
                    },
                },
                "required": ["note_id", "text"],
                "additionalProperties": False,
            },
            handler=update_memory,
            category="memory",
        ),
        ToolSpec(
            name="memory_stats",
            description=(
                "Show summary statistics about the long-term memory notebook: "
                "total notes, most-recalled facts, and the newest note."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=memory_stats,
            category="memory",
        ),
        ToolSpec(
            name="deduplicate_memory",
            description=(
                "Merge duplicate notes (identical text) by combining their tags "
                "and access counts. Call this when the notebook feels cluttered "
                "or the user asks to clean up redundant facts."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=deduplicate_memory,
            category="memory",
        ),
    ]

    if store is not None:
        def list_sessions(arguments: dict) -> ToolResult:
            sessions = store.list_sessions()
            if not sessions:
                return ToolResult("", "(no saved sessions)")
            return ToolResult("", "Saved session IDs:\n" + "\n".join(f"- {s}" for s in sessions))

        def delete_session(arguments: dict) -> ToolResult:
            session_id = arguments["session_id"]
            store.delete(session_id)
            return ToolResult("", f"Deleted session {session_id}.")

        tools.extend([
            ToolSpec(
                name="list_sessions",
                description=(
                    "List IDs of previously persisted conversation sessions. Use "
                    "this when the user asks about history from another session."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=list_sessions,
                category="memory",
            ),
            ToolSpec(
                name="delete_session",
                description=(
                    "Delete a previously persisted conversation session by its "
                    "session_id. This is permanent: the transcript will not be "
                    "loaded in future runs."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID to delete."},
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
                handler=delete_session,
                category="memory",
            ),
        ])

        if hasattr(store, "search_sessions"):
            def search_sessions(arguments: dict) -> ToolResult:
                query = arguments.get("query", "") or ""
                if not query:
                    return ToolResult("", "usage: provide a query", is_error=True)
                limit = int(arguments.get("limit", 10))
                results = store.search_sessions(query, limit=limit)  # type: ignore[attr-defined]
                if not results:
                    return ToolResult("", "(no matching sessions)")
                return ToolResult("", "Matching sessions:\n" + "\n".join(f"- {s}" for s in results))

            tools.append(
                ToolSpec(
                    name="search_sessions",
                    description=(
                        "Search saved conversation sessions for those containing "
                        "messages that match the query. Uses full-text search "
                        "when the underlying store supports it."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Words to search for in session messages.",
                            },
                            "limit": {"type": "integer", "description": "Max sessions to return."},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=search_sessions,
                    category="memory",
                )
            )

    if hasattr(notes, "reindex_embeddings"):
        def reindex_memory(arguments: dict) -> ToolResult:
            count = notes.reindex_embeddings()
            return ToolResult("", f"Reindexed {count} note(s) for semantic search.")

        tools.append(
            ToolSpec(
                name="reindex_memory",
                description=(
                    "Recompute missing dense embeddings for semantic note recall. "
                    "Use this after enabling semantic_search or restoring a notebook."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=reindex_memory,
                category="memory",
            )
        )

    return tools


class SqliteNoteStore:
    """SQLite-backed notebook with full-text recall.

    Uses the same ``MemoryNote`` model as ``JsonlNoteStore`` but persists in a
    relational table. An optional FTS5 index is created for fast text search;
    builds without FTS5 fall back to substring search.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        weighting: RecallWeighting | None = None,
        scope: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._weighting = weighting or RecallWeighting()
        self._scope = scope  # None = scoping disabled
        self._conn: sqlite3.Connection | None = None
        self._fts_enabled: bool | None = None
        self._ensure_schema()

    def _scope_clause(self) -> tuple[str, tuple]:
        """SQL fragment + params restricting rows to the visible namespace.

        Returns ``("", ())`` when scoping is disabled. Callers splice the
        fragment after an existing ``WHERE`` with ``AND``.
        """
        if self._scope is None:
            return "", ()
        return "(scope = '' OR scope = ?)", (self._scope,)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                created TEXT NOT NULL,
                accessed TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                scope TEXT NOT NULL DEFAULT ''
            );
            """
        )
        # Migrate notebooks created before the scope column existed.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(notes)").fetchall()}
        if "scope" not in cols:
            conn.execute("ALTER TABLE notes ADD COLUMN scope TEXT NOT NULL DEFAULT ''")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(text)"
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
        conn.commit()

    def _row_to_note(self, row: sqlite3.Row) -> MemoryNote:
        return MemoryNote(
            text=row["text"],
            tags=json.loads(row["tags_json"]),
            created=row["created"],
            note_id=row["note_id"],
            accessed=row["accessed"] or "",
            access_count=row["access_count"],
            scope=row["scope"],
        )

    def add(
        self,
        text: str,
        *,
        tags: Sequence[str] | None = None,
        scope: str | None = None,
    ) -> MemoryNote:
        created = datetime.now(UTC).isoformat()
        note_scope = scope if scope is not None else (self._scope or "")
        conn = self._connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO notes (text, tags_json, created, accessed, access_count, scope)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    text.strip(),
                    json.dumps([str(t) for t in (tags or [])]),
                    created,
                    "",
                    0,
                    note_scope,
                ),
            )
            note_id = cursor.lastrowid or 0
            if self._fts_enabled:
                conn.execute(
                    "INSERT INTO notes_fts(rowid, text) VALUES (?, ?)",
                    (note_id, text.strip()),
                )
        note = MemoryNote(
            text=text.strip(),
            tags=[str(t) for t in (tags or [])],
            created=created,
            note_id=note_id,
            accessed="",
            access_count=0,
            scope=note_scope,
        )
        return note

    def all(self) -> list[MemoryNote]:
        rows = self._connection().execute(
            "SELECT * FROM notes ORDER BY note_id"
        ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def recent(self, limit: int = 5) -> list[MemoryNote]:
        clause, params = self._scope_clause()
        where = f"WHERE {clause} " if clause else ""
        rows = self._connection().execute(
            f"SELECT * FROM notes {where}ORDER BY note_id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        notes = [self._row_to_note(r) for r in rows]
        for note in notes:
            self._touch(note)
        return notes

    def search(self, query: str, *, limit: int = 5) -> list[MemoryNote]:
        query = query.strip()
        if not query:
            return self.recent(limit)
        conn = self._connection()
        clause, sparams = self._scope_clause()
        rows: list[sqlite3.Row] = []
        if self._fts_enabled:
            scope_and = f"AND {clause.replace('scope', 'n.scope')} " if clause else ""
            try:
                rows = conn.execute(
                    f"""
                    SELECT n.*
                    FROM notes_fts f
                    JOIN notes n ON f.rowid = n.note_id
                    WHERE f MATCH ? {scope_and}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, *sparams, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            like = f"%{query}%"
            scope_and = f"AND {clause} " if clause else ""
            rows = conn.execute(
                f"""
                SELECT * FROM notes
                WHERE text LIKE ? {scope_and}
                ORDER BY note_id DESC
                LIMIT ?
                """,
                (like, *sparams, limit),
            ).fetchall()
        notes = [self._row_to_note(r) for r in rows]
        for note in notes:
            self._touch(note)
        # Rows arrive in relevance order (FTS rank / recency); give them
        # decreasing positional base scores so the weighting can re-rank them.
        pairs = [(float(len(notes) - i), note) for i, note in enumerate(notes)]
        return self._weighting.rerank(pairs, limit=limit)

    def forget(
        self, *, note_id: int | None = None, text_prefix: str | None = None
    ) -> list[MemoryNote]:
        if note_id is None and not text_prefix:
            return []
        removed: list[MemoryNote] = []
        conn = self._connection()
        with conn:
            if note_id is not None:
                rows = conn.execute(
                    "SELECT * FROM notes WHERE note_id = ?", (note_id,)
                ).fetchall()
                removed = [self._row_to_note(r) for r in rows]
                self._delete_by_ids([r["note_id"] for r in rows])
            elif text_prefix:
                prefix = text_prefix.strip().lower()
                rows = conn.execute(
                    "SELECT * FROM notes WHERE LOWER(text) LIKE ?",
                    (f"{prefix}%",),
                ).fetchall()
                removed = [self._row_to_note(r) for r in rows]
                self._delete_by_ids([r["note_id"] for r in rows])
        return removed

    def _delete_by_ids(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        conn = self._connection()
        with conn:
            if self._fts_enabled:
                conn.execute(
                    f"DELETE FROM notes_fts WHERE rowid IN ({placeholders})",
                    tuple(ids),
                )
            conn.execute(
                f"DELETE FROM notes WHERE note_id IN ({placeholders})",
                tuple(ids),
            )

    def update(
        self, note_id: int, text: str, *, tags: Sequence[str] | None = None
    ) -> MemoryNote | None:
        accessed = datetime.now(UTC).isoformat()
        conn = self._connection()
        with conn:
            existing = conn.execute(
                "SELECT * FROM notes WHERE note_id = ?", (note_id,)
            ).fetchone()
            if existing is None:
                return None
            if self._fts_enabled:
                conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (note_id,))
            conn.execute(
                """
                UPDATE notes
                SET text = ?, tags_json = ?, accessed = ?,
                    access_count = access_count + 1
                WHERE note_id = ?
                """,
                (
                    text.strip(),
                    json.dumps([str(t) for t in (tags or [])]),
                    accessed,
                    note_id,
                ),
            )
            if self._fts_enabled:
                conn.execute(
                    "INSERT INTO notes_fts(rowid, text) VALUES (?, ?)",
                    (note_id, text.strip()),
                )
        row = conn.execute(
            "SELECT * FROM notes WHERE note_id = ?", (note_id,)
        ).fetchone()
        return self._row_to_note(row) if row is not None else None

    def deduplicate(self) -> int:
        conn = self._connection()
        with conn:
            rows = conn.execute(
                "SELECT * FROM notes ORDER BY note_id"
            ).fetchall()
            seen: dict[str, MemoryNote] = {}
            ids_to_remove: list[int] = []
            for row in rows:
                note = self._row_to_note(row)
                text = note.text.strip().lower()
                if text in seen:
                    existing = seen[text]
                    existing.tags = sorted(set(existing.tags) | set(note.tags))
                    if note.access_count:
                        existing.access_count += note.access_count
                    ids_to_remove.append(note.note_id)
                    conn.execute(
                        "UPDATE notes SET tags_json = ?, access_count = ? WHERE note_id = ?",
                        (
                            json.dumps(existing.tags),
                            existing.access_count,
                            existing.note_id,
                        ),
                    )
                else:
                    seen[text] = note
        if ids_to_remove:
            self._delete_by_ids(ids_to_remove)
        return len(ids_to_remove)

    def export(self, destination: str | Path) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = ["# Memory Notes\n"]
        for note in self.all():
            tag_part = f" *[{', '.join(note.tags)}]*" if note.tags else ""
            access_part = (
                f", accessed {note.access_count} time(s)" if note.access_count else ""
            )
            lines.append(
                f"- {note.text}{tag_part}  (id={note.note_id}, {note.created}{access_part})\n"
            )
        dest.write_text("".join(lines), encoding="utf-8")
        return dest

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _touch(self, note: MemoryNote) -> None:
        note.access_count += 1
        note.accessed = datetime.now(UTC).isoformat()
        with self._connection():
            self._connection().execute(
                "UPDATE notes SET access_count = ?, accessed = ? WHERE note_id = ?",
                (note.access_count, note.accessed, note.note_id),
            )
