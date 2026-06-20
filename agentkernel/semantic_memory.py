"""Semantic note search over SQLite notebooks.

This is a thin subclass of ``SqliteNoteStore`` so the JSONL / SQLite split and
all existing memory-tool behavior stay intact. When an ``EmbeddingProvider`` is
configured, each note stores a dense vector and recall is re-ranked by cosine
similarity rather than only token overlap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from agentkernel.embeddings import EmbeddingProvider, cosine_similarity
from agentkernel.memory import MemoryNote, SqliteNoteStore


class SemanticSqliteNoteStore(SqliteNoteStore):
    """SQLite notebook that also stores dense embeddings for semantic ranking.

    Keyword and full-text search still retrieve candidates; dense similarity
    refines their order. Notes created before the provider existed can be
    backfilled with ``reindex_embeddings()``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._embedding_provider = embedding_provider
        # Parent creates the notes table and optional FTS5 index.
        super().__init__(path)
        self._ensure_embedding_schema()

    def _ensure_embedding_schema(self) -> None:
        conn = self._connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS note_embeddings (
                note_id INTEGER PRIMARY KEY,
                embedding_json TEXT NOT NULL
            )
            """
        )
        conn.commit()

    def _upsert_embedding(self, note_id: int, text: str) -> None:
        if not text or self._embedding_provider is None:
            return
        vector = self._embedding_provider.embed([text])[0]
        if not vector:
            return
        with self._connection():
            self._connection().execute(
                """
                INSERT INTO note_embeddings (note_id, embedding_json)
                VALUES (?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    embedding_json = excluded.embedding_json
                """,
                (note_id, json.dumps(vector)),
            )

    def add(self, text: str, *, tags: Sequence[str] | None = None) -> MemoryNote:
        note = super().add(text, tags=tags)
        self._upsert_embedding(note.note_id, note.text)
        return note

    def update(
        self,
        note_id: int,
        text: str,
        *,
        tags: Sequence[str] | None = None,
    ) -> MemoryNote | None:
        note = super().update(note_id, text, tags=tags)
        if note is not None:
            self._upsert_embedding(note.note_id, note.text)
        return note

    def search(self, query: str, *, limit: int = 5) -> list[MemoryNote]:
        query = query.strip()
        if not query:
            return super().search(query, limit=limit)
        if self._embedding_provider is None:
            return super().search(query, limit=limit)

        # Dense ranking over the whole notebook. For typical personal notebooks
        # this is small; swap in an approximate vector index if scale demands it.
        query_vec = self._embedding_provider.embed([query])[0]
        candidates = self.all()
        if not candidates:
            return []

        ids = [note.note_id for note in candidates]
        vectors = self._load_embeddings(ids)

        scored: list[tuple[float, int, MemoryNote]] = []
        for note in candidates:
            vec = vectors.get(note.note_id)
            if vec and query_vec:
                similarity = cosine_similarity(query_vec, vec)
            else:
                # Notes without embeddings fall below any scored note.
                similarity = -1.0 if vectors else 0.0
            scored.append((similarity, note.note_id, note))

        # Highest similarity first; tie-break by note_id for stability.
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        ranked = [note for _, _, note in scored[:limit]]
        for note in ranked:
            self._touch(note)
        return ranked

    def _load_embeddings(self, note_ids: list[int]) -> dict[int, list[float]]:
        if not note_ids:
            return {}
        placeholders = ",".join("?" for _ in note_ids)
        rows = self._connection().execute(
            f"""
            SELECT note_id, embedding_json
            FROM note_embeddings
            WHERE note_id IN ({placeholders})
            """,
            tuple(note_ids),
        ).fetchall()
        return {row["note_id"]: json.loads(row["embedding_json"]) for row in rows}

    def reindex_embeddings(self) -> int:
        """Compute and store embeddings for all notes that do not have one yet."""
        rows = self._connection().execute(
            """
            SELECT n.note_id, n.text
            FROM notes n
            LEFT JOIN note_embeddings e ON n.note_id = e.note_id
            WHERE e.note_id IS NULL AND n.text IS NOT NULL AND n.text != ''
            """
        ).fetchall()
        if not rows:
            return 0
        vectors = self._embedding_provider.embed([r["text"] for r in rows])
        count = 0
        with self._connection():
            for row, vec in zip(rows, vectors):
                if vec:
                    self._connection().execute(
                        "INSERT INTO note_embeddings (note_id, embedding_json) VALUES (?, ?)",
                        (row["note_id"], json.dumps(vec)),
                    )
                    count += 1
        return count

    def forget(
        self,
        *,
        note_id: int | None = None,
        text_prefix: str | None = None,
    ) -> list[MemoryNote]:
        removed = super().forget(note_id=note_id, text_prefix=text_prefix)
        if removed:
            ids = [note.note_id for note in removed]
            placeholders = ",".join("?" for _ in ids)
            with self._connection():
                self._connection().execute(
                    f"DELETE FROM note_embeddings WHERE note_id IN ({placeholders})",
                    tuple(ids),
                )
        return removed
