"""Tests for semantic note search with a deterministic fake embedding provider."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentkernel.embeddings import EmbeddingError, OpenAIEmbeddingProvider
from agentkernel.memory import SqliteNoteStore, make_memory_tools
from agentkernel.semantic_memory import SemanticSqliteNoteStore


class _FakeEmbeddingProvider:
    """Deterministic 3-D vectors for a few animal/vehicle words."""

    _vectors = {
        "cat": [1.0, 0.0, 0.0],
        "kitten": [0.9, 0.1, 0.0],
        "feline": [0.95, 0.05, 0.0],
        "dog": [0.0, 1.0, 0.0],
        "puppy": [0.1, 0.9, 0.0],
        "car": [0.0, 0.0, 1.0],
        "automobile": [0.05, 0.0, 0.95],
    }
    _dim = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            words = [w for w in re.findall(r"\w+", text.lower())]
            vecs = [self._vectors[w] for w in words if w in self._vectors]
            if vecs:
                avg = [sum(v[i] for v in vecs) / len(vecs) for i in range(self._dim)]
            else:
                avg = [0.0] * self._dim
            result.append(avg)
        return result


def _store(tmp_path: Path, path: Path | None = None) -> SemanticSqliteNoteStore:
    target = path if path is not None else tmp_path / "semantic.db"
    return SemanticSqliteNoteStore(target, embedding_provider=_FakeEmbeddingProvider())


def test_semantic_search_ranks_by_similarity(tmp_path):
    store = _store(tmp_path)
    store.add("cat fact")
    store.add("dog fact")
    results = store.search("kitten", limit=2)
    assert [n.text for n in results] == ["cat fact", "dog fact"]


def test_reindex_embeddings_backfills_existing_notes(tmp_path):
    # Create an ordinary SQLite notebook, add notes, then promote it to semantic
    # and backfill embeddings.
    path = tmp_path / "promoted.db"
    plain = SqliteNoteStore(path)
    plain.add("cat fact")
    plain.add("dog fact")
    plain.close()

    semantic = _store(tmp_path, path)
    assert semantic.reindex_embeddings() == 2
    results = semantic.search("kitten", limit=2)
    assert [n.text for n in results] == ["cat fact", "dog fact"]
    assert semantic.reindex_embeddings() == 0  # idempotent


def test_keyword_fallback_without_provider(tmp_path):
    plain = SqliteNoteStore(tmp_path / "plain.db")
    plain.add("cat fact")
    plain.add("dog fact")
    results = plain.search("kitten")
    # Without embeddings, keyword/FTS search still returns something usable
    # (in this case the literal substring "kitten" returns nothing; the point is
    # that the class still functions).
    assert results == []


def test_forget_removes_embeddings(tmp_path):
    store = _store(tmp_path)
    note = store.add("cat fact")
    store.search("kitten")  # ensure embedding was stored
    store.forget(note_id=note.note_id)
    rows = store._connection().execute("SELECT COUNT(*) AS c FROM note_embeddings").fetchone()
    assert rows["c"] == 0


def test_update_recomputes_embedding(tmp_path):
    store = _store(tmp_path)
    note = store.add("cat fact")
    store.update(note.note_id, "dog fact")
    results = store.search("puppy", limit=1)
    assert [n.text for n in results] == ["dog fact"]


def test_reindex_memory_tool_registered_for_semantic_store(tmp_path):
    store = _store(tmp_path)
    tools = {t.name: t for t in make_memory_tools(store)}
    assert "reindex_memory" in tools


def test_reindex_memory_tool_absent_for_plain_store(tmp_path):
    plain = SqliteNoteStore(tmp_path / "plain.db")
    tools = {t.name: t for t in make_memory_tools(plain)}
    assert "reindex_memory" not in tools


def test_reindex_memory_tool_backfills_embeddings(tmp_path):
    path = tmp_path / "promoted.db"
    plain = SqliteNoteStore(path)
    plain.add("cat fact")
    plain.close()

    store = _store(tmp_path, path)
    tools = {t.name: t for t in make_memory_tools(store)}
    result = tools["reindex_memory"].handler({})
    assert result.is_error is False
    assert "Reindexed 1 note(s)" in result.content


def test_openai_provider_requires_endpoint_for_anthropic():
    from agentkernel.config import Config

    cfg = Config(provider="anthropic")
    with pytest.raises(EmbeddingError) as exc_info:
        OpenAIEmbeddingProvider.from_config(cfg)
    assert "embedding_base_url" in str(exc_info.value)


def test_openai_provider_infers_openai_endpoint():
    from agentkernel.config import Config

    cfg = Config(provider="openai", embedding_model="text-embedding-3-small")
    provider = OpenAIEmbeddingProvider.from_config(cfg)
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider.model == "text-embedding-3-small"


def test_lsh_index_returns_same_top_results_as_flat_search(tmp_path):
    store = SemanticSqliteNoteStore(
        tmp_path / "lsh.db",
        embedding_provider=_FakeEmbeddingProvider(),
        lsh_bits=4,
    )
    for _ in range(6):
        store.add("cat fact")
    for _ in range(6):
        store.add("dog fact")
    results = store.search("kitten", limit=2)
    assert all("cat" in n.text for n in results)


def test_lsh_bucket_metadata_survives_reopen(tmp_path):
    path = tmp_path / "lsh.db"
    store = SemanticSqliteNoteStore(
        path,
        embedding_provider=_FakeEmbeddingProvider(),
        lsh_bits=4,
    )
    store.add("cat fact")
    store.close()

    reopened = SemanticSqliteNoteStore(
        path,
        embedding_provider=_FakeEmbeddingProvider(),
        lsh_bits=4,
    )
    rows = reopened._connection().execute(
        "SELECT COUNT(*) AS c FROM lsh_buckets"
    ).fetchone()
    assert rows["c"] == 1
    results = reopened.search("kitten", limit=1)
    assert results and "cat" in results[0].text

