"""Recency- and importance-weighted note recall (design §18.2).

The weighting re-ranks recalled notes using the ``created``/``accessed`` and
``access_count`` metadata the stores already track. It is opt-in: with the
default zero weights it must be a faithful no-op so existing keyword/TF-IDF and
semantic ordering is unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agentkernel.memory import (
    JsonlNoteStore,
    MemoryNote,
    RecallWeighting,
    SqliteNoteStore,
    make_note_store,
)


def _iso(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


# --- the pure re-ranker ------------------------------------------------------


def test_inactive_weighting_preserves_supplied_order():
    w = RecallWeighting()  # both weights 0
    assert not w.active
    a = MemoryNote(text="a", note_id=1, created=_iso(100))
    b = MemoryNote(text="b", note_id=2, created=_iso(1))
    # Supplied with equal base scores in a deliberate order; must be preserved.
    ranked = w.rerank([(0.5, a), (0.5, b)], limit=5)
    assert [n.note_id for n in ranked] == [1, 2]


def test_inactive_weighting_truncates_to_limit():
    w = RecallWeighting()
    notes = [MemoryNote(text=str(i), note_id=i) for i in range(5)]
    ranked = w.rerank([(1.0, n) for n in notes], limit=2)
    assert [n.note_id for n in ranked] == [0, 1]


def test_recency_weight_breaks_ties_toward_newer_notes():
    w = RecallWeighting(recency_weight=1.0)
    old = MemoryNote(text="old", note_id=1, created=_iso(120))
    new = MemoryNote(text="new", note_id=2, created=_iso(1))
    # Equal base relevance -> the newer note should win once recency is applied.
    ranked = w.rerank([(0.5, old), (0.5, new)], limit=5)
    assert [n.note_id for n in ranked] == [2, 1]


def test_importance_weight_breaks_ties_toward_recalled_notes():
    w = RecallWeighting(importance_weight=1.0)
    cold = MemoryNote(text="cold", note_id=1, access_count=0)
    hot = MemoryNote(text="hot", note_id=2, access_count=25)
    ranked = w.rerank([(0.5, cold), (0.5, hot)], limit=5)
    assert [n.note_id for n in ranked] == [2, 1]


def test_relevance_still_gates_weighting():
    # A zero-base (irrelevant) note must never overtake a relevant one no matter
    # how new or how often recalled it is.
    w = RecallWeighting(recency_weight=5.0, importance_weight=5.0)
    irrelevant = MemoryNote(text="x", note_id=1, created=_iso(0), access_count=99)
    relevant = MemoryNote(text="y", note_id=2, created=_iso(365))
    ranked = w.rerank([(0.0, irrelevant), (0.8, relevant)], limit=5)
    assert ranked[0].note_id == 2


def test_half_life_controls_decay_strength():
    now = datetime.now(UTC)
    short = RecallWeighting(recency_weight=1.0, half_life_days=1.0)
    long = RecallWeighting(recency_weight=1.0, half_life_days=365.0)
    note = MemoryNote(text="n", note_id=1, created=(now - timedelta(days=7)).isoformat())
    # A 7-day-old note keeps almost all of its recency under a long half-life
    # but almost none under a 1-day half-life.
    assert short._recency(note, now) < 0.05
    assert long._recency(note, now) > 0.9


def test_recency_uses_created_not_last_access():
    # Recency is creation-age based: a freshly accessed but old note still reads
    # as old, because search touches every candidate before ranking.
    now = datetime.now(UTC)
    w = RecallWeighting(recency_weight=1.0, half_life_days=10.0)
    note = MemoryNote(
        text="n",
        note_id=1,
        created=(now - timedelta(days=100)).isoformat(),
        accessed=now.isoformat(),
    )
    assert w._recency(note, now) < 0.01


def test_malformed_timestamp_does_not_crash():
    w = RecallWeighting(recency_weight=1.0)
    note = MemoryNote(text="n", note_id=1, created="not-a-date")
    assert w._recency(note, datetime.now(UTC)) == 0.0


# --- backend integration -----------------------------------------------------


def test_jsonl_store_default_is_unweighted(tmp_path):
    notes = make_note_store(tmp_path / "notes.jsonl")
    assert isinstance(notes, JsonlNoteStore)
    notes.add("the cat sat on the mat")
    notes.add("the cat chased the mouse")
    # Both relevant to "cat"; default weighting leaves TF-IDF order intact.
    results = notes.search("cat")
    assert {n.text for n in results} == {
        "the cat sat on the mat",
        "the cat chased the mouse",
    }


def test_jsonl_store_recency_weight_surfaces_newer_note(tmp_path):
    weighting = RecallWeighting(recency_weight=4.0, half_life_days=1.0)
    notes = JsonlNoteStore(tmp_path / "notes.jsonl", weighting=weighting)
    # Identical text -> identical base relevance, so recency alone decides.
    older = notes.add("python programming language")
    newer = notes.add("python programming language")
    older.created = _iso(90)
    newer.created = _iso(0)
    results = notes.search("python programming language")
    assert results[0].note_id == newer.note_id


def test_sqlite_store_importance_weight_surfaces_recalled_note(tmp_path):
    weighting = RecallWeighting(importance_weight=6.0)
    notes = SqliteNoteStore(tmp_path / "notes.db", weighting=weighting)
    a = notes.add("deploy uses the staging cluster")
    b = notes.add("deploy uses the production cluster")
    # Give b a high accumulated recall count so importance breaks the tie.
    conn = notes._connection()
    with conn:
        conn.execute(
            "UPDATE notes SET access_count = ? WHERE note_id = ?", (20, b.note_id)
        )
    # Single-token query so it survives both the FTS and the LIKE fallback path.
    results = notes.search("cluster")
    assert results[0].note_id == b.note_id
    assert a.note_id in {n.note_id for n in results}
