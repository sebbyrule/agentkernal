"""Approximate nearest-neighbor helpers for semantic memory.

These use only the Python standard library so the kernel keeps its stdlib-only
constraint. The default path is a brute-force cosine scan; when scale demands it,
a small random-projection LSH index can prune the candidate set.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Callable


class LSHIndex:
    """Random-projection locality-sensitive hash index for dense vectors.

    Each vector is projected onto ``num_bits`` random hyperplanes; the sign of
    each projection becomes one bit of an integer bucket key. Queries fetch the
    exact bucket plus all buckets one bit away, which dramatically improves
    recall without a full linear scan.

    The hyperplanes are persisted in the same SQLite database as the vectors so
    the index is stable across process restarts.
    """

    def __init__(
        self,
        dim: int,
        num_bits: int,
        conn: Callable[[], sqlite3.Connection],
        *,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.num_bits = num_bits
        self._conn = conn
        self._seed = seed
        self._hyperplanes = self._ensure_hyperplanes()

    def _ensure_hyperplanes(self) -> list[list[float]]:
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS lsh_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS lsh_buckets ("
            "note_id INTEGER PRIMARY KEY, bucket INTEGER NOT NULL"
            ")"
        )

        bits_row = conn.execute(
            'SELECT value FROM lsh_meta WHERE key = "bits"'
        ).fetchone()
        seed_row = conn.execute(
            'SELECT value FROM lsh_meta WHERE key = "seed"'
        ).fetchone()
        planes_row = conn.execute(
            'SELECT value FROM lsh_meta WHERE key = "hyperplanes"'
        ).fetchone()

        if bits_row and seed_row and planes_row:
            stored_bits = int(bits_row["value"])
            stored_seed = int(seed_row["value"])
            planes = json.loads(planes_row["value"])
            if (
                stored_bits == self.num_bits
                and stored_seed == self._seed
                and len(planes) == self.num_bits
                and all(len(p) == self.dim for p in planes)
            ):
                return planes

        rng = random.Random(self._seed)
        planes = [
            [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
            for _ in range(self.num_bits)
        ]
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO lsh_meta (key, value) VALUES (?, ?)",
                ("bits", str(self.num_bits)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO lsh_meta (key, value) VALUES (?, ?)",
                ("seed", str(self._seed)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO lsh_meta (key, value) VALUES (?, ?)",
                ("hyperplanes", json.dumps(planes)),
            )
            conn.execute("DELETE FROM lsh_buckets")
        return planes

    def hash(self, vector: list[float]) -> int:
        """Return the integer bucket for ``vector``."""
        bucket = 0
        for bit, plane in enumerate(self._hyperplanes):
            dot = sum(v * p for v, p in zip(vector, plane, strict=True))
            if dot >= 0:
                bucket |= 1 << bit
        return bucket

    def query_buckets(self, vector: list[float]) -> list[int]:
        """Return the query bucket and all one-bit neighbors."""
        base = self.hash(vector)
        buckets = [base]
        for bit in range(self.num_bits):
            buckets.append(base ^ (1 << bit))
        return buckets

    def upsert(self, note_id: int, vector: list[float]) -> None:
        """Store/update the bucket for ``note_id``."""
        with self._conn():
            self._conn().execute(
                "INSERT OR REPLACE INTO lsh_buckets (note_id, bucket) VALUES (?, ?)",
                (note_id, self.hash(vector)),
            )

    def remove(self, note_id: int) -> None:
        with self._conn():
            self._conn().execute(
                "DELETE FROM lsh_buckets WHERE note_id = ?", (note_id,)
            )

    def candidate_ids(self, buckets: list[int]) -> list[int]:
        """Return note ids whose bucket is in ``buckets``."""
        if not buckets:
            return []
        placeholders = ",".join("?" for _ in buckets)
        rows = self._conn().execute(
            f"""
            SELECT note_id FROM lsh_buckets
            WHERE bucket IN ({placeholders})
            """,
            tuple(buckets),
        ).fetchall()
        return [row["note_id"] for row in rows]
