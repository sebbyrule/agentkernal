"""Semantic note search over SQLite notebooks.

This is a thin subclass of ``SqliteNoteStore`` so the JSONL / SQLite split and
all existing memory-tool behavior stay intact. When an ``EmbeddingProvider`` is
configured, each note stores a dense vector and recall is re-ranked by cosine
similarity rather than only token overlap.

For large notebooks the optional LSH index in ``semantic_index`` prunes the
candidate set before the dense comparison, avoiding a full linear scan.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from agentkernel.embeddings import EmbeddingProvider, cosine_similarity
from agentkernel.memory import (
    MemoryNote,
    RecallWeighting,
    SqliteNoteStore,
    _scope_visible,
)
from agentkernel.semantic_index import LSHIndex


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
        lsh_bits: int | None = None,
        weighting: RecallWeighting | None = None,
        scope: str | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._lsh_bits = lsh_bits
        self._lsh_index: LSHIndex | None = None
        # Parent creates the notes table and optional FTS5 index.
        super().__init__(path, weighting=weighting, scope=scope)
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

    def _ensure_lsh_index(self, sample_vector: list[float] | None = None) -> None:
        """Create the LSH index once we know the vector dimension."""
        if self._lsh_index is not None or not self._lsh_bits:
            return

        dim = len(sample_vector) if sample_vector else None
        if dim is None:
            rows = self._connection().execute(
                "SELECT embedding_json FROM note_embeddings LIMIT 1"
            ).fetchall()
            if rows:
                dim = len(json.loads(rows[0]["embedding_json"]))
        if dim is None:
            return  # no embeddings yet; will initialize on first note

        self._lsh_index = LSHIndex(
            dim=dim,
            num_bits=self._lsh_bits,
            conn=self._connection,
            seed=0,
        )
        # Backfill buckets for any existing embeddings (e.g. after a schema
        # change or when opening an older notebook).
        count = self._connection().execute(
            "SELECT COUNT(*) FROM lsh_buckets"
        ).fetchone()[0]
        if count == 0:
            rows = self._connection().execute(
                "SELECT note_id, embedding_json FROM note_embeddings"
            ).fetchall()
            for row in rows:
                self._lsh_index.upsert(row["note_id"], json.loads(row["embedding_json"]))

    def _upsert_embedding(self, note_id: int, text: str) -> None:
        if not text or self._embedding_provider is None:
            return
        vector = self._embedding_provider.embed([text])[0]
        if not vector:
            return
        self._ensure_lsh_index(vector)
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
        if self._lsh_index is not None:
            self._lsh_index.upsert(note_id, vector)

    def add(
        self,
        text: str,
        *,
        tags: Sequence[str] | None = None,
        scope: str | None = None,
    ) -> MemoryNote:
        note = super().add(text, tags=tags, scope=scope)
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

        query_vec = self._embedding_provider.embed([query])[0]
        candidates = self._candidates(query_vec, limit=limit)
        candidates = [n for n in candidates if _scope_visible(n.scope, self._scope)]
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
        pairs = [(similarity, note) for similarity, _nid, note in scored]
        ranked = self._weighting.rerank(pairs, limit=limit)
        for note in ranked:
            self._touch(note)
        return ranked

    def _candidates(
        self, query_vec: list[float], *, limit: int
    ) -> list[MemoryNote]:
        """Return notes to score for ``query_vec``.

        If an LSH index is active, use it to narrow the set; otherwise fall
        back to scanning every note. When the pruned set is too small we also
        fall back to avoid missing neighbors due to hash collisions.
        """
        all_notes = self.all()
        if not self._lsh_bits or not all_notes:
            return all_notes

        self._ensure_lsh_index()
        if self._lsh_index is None:
            return all_notes

        buckets = self._lsh_index.query_buckets(query_vec)
        candidate_ids = self._lsh_index.candidate_ids(buckets)
        # LSH is only a speedup; if the bucket is empty or tiny, scan the table
        # so accuracy does not suffer on small notebooks or unlucky hashes.
        if len(candidate_ids) < max(limit * 2, 8):
            return all_notes

        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self._connection().execute(
            f"""
            SELECT * FROM notes
            WHERE note_id IN ({placeholders})
            ORDER BY note_id
            """,
            tuple(candidate_ids),
        ).fetchall()
        return [self._row_to_note(r) for r in rows]

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
            for row, vec in zip(rows, vectors, strict=True):
                if vec:
                    self._ensure_lsh_index(vec)
                    self._connection().execute(
                        "INSERT INTO note_embeddings (note_id, embedding_json) VALUES (?, ?)",
                        (row["note_id"], json.dumps(vec)),
                    )
                    if self._lsh_index is not None:
                        self._lsh_index.upsert(row["note_id"], vec)
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
                if self._lsh_index is not None:
                    for nid in ids:
                        self._lsh_index.remove(nid)
        return removed
