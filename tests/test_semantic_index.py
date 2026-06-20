"""Tests for the approximate LSH vector index used by semantic memory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentkernel.semantic_index import LSHIndex


def _tmp_db(tmp_path: Path) -> tuple[sqlite3.Connection, callable]:
    path = tmp_path / "lsh.db"
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    def factory():
        return conn

    return conn, factory


def test_hash_consistency_and_query_neighbors(tmp_path):
    conn, factory = _tmp_db(tmp_path)
    index = LSHIndex(dim=4, num_bits=8, conn=factory, seed=7)

    vec = [1.0, 0.0, 0.0, 0.0]
    bucket = index.hash(vec)
    buckets = index.query_buckets(vec)
    assert len(buckets) == 9  # exact bucket + 8 one-bit neighbors
    assert bucket in buckets
    assert len(set(buckets)) == 9


def test_upsert_and_candidate_lookup(tmp_path):
    conn, factory = _tmp_db(tmp_path)
    index = LSHIndex(dim=3, num_bits=6, conn=factory, seed=0)

    vectors = {
        1: [1.0, 0.0, 0.0],
        2: [0.9, 0.1, 0.0],
        3: [0.0, 1.0, 0.0],
        4: [0.0, 0.0, 1.0],
    }
    for note_id, vec in vectors.items():
        index.upsert(note_id, vec)

    candidates = index.candidate_ids(index.query_buckets([1.0, 0.0, 0.0]))
    # The exact/neighbor buckets should contain the nearby notes.
    assert 1 in candidates
    # Sanity: all four notes are stored somewhere.
    all_rows = conn.execute("SELECT note_id FROM lsh_buckets").fetchall()
    assert {r["note_id"] for r in all_rows} == {1, 2, 3, 4}


def test_remove_deletes_bucket(tmp_path):
    conn, factory = _tmp_db(tmp_path)
    index = LSHIndex(dim=2, num_bits=4, conn=factory, seed=0)
    index.upsert(1, [1.0, 0.0])
    index.upsert(2, [0.0, 1.0])
    index.remove(1)

    rows = conn.execute("SELECT note_id FROM lsh_buckets").fetchall()
    assert [r["note_id"] for r in rows] == [2]


def test_hyperplanes_persisted_in_meta(tmp_path):
    conn, factory = _tmp_db(tmp_path)
    LSHIndex(dim=5, num_bits=10, conn=factory, seed=42)  # persists meta on init

    meta = {
        row["key"]: row["value"]
        for row in conn.execute('SELECT key, value FROM lsh_meta').fetchall()
    }
    assert meta["bits"] == "10"
    assert meta["seed"] == "42"
    planes = json.loads(meta["hyperplanes"])
    assert len(planes) == 10
    assert all(len(p) == 5 for p in planes)

    # Reopening with matching dim/bits reuses stored planes.
    index2 = LSHIndex(dim=5, num_bits=10, conn=factory, seed=42)
    assert index2._hyperplanes == planes
