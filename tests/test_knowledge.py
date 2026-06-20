"""Tests for the file-backed knowledge graph (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentkernel.knowledge import Fact, KnowledgeGraph, make_graph_tools


@pytest.fixture
def graph(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(tmp_path / "graph.jsonl")


def test_add_and_query(graph):
    graph.add("Alice", "knows", "Bob")
    results = graph.query(subject="Alice")
    assert len(results) == 1
    assert results[0].object == "Bob"


def test_add_strips_whitespace(graph):
    fact = graph.add("  Alice  ", "  knows  ", "  Bob  ")
    assert fact == Fact("Alice", "knows", "Bob")
    assert graph.query(subject="Alice")[0].object == "Bob"


def test_deduplicates_exact_triples(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Alice", "knows", "Bob")
    assert len(graph._facts) == 1


def test_query_by_object_and_predicate(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Alice", "works_at", "Acme")
    assert [f.predicate for f in graph.query(predicate="knows")] == ["knows"]


def test_query_like_substring(graph):
    graph.add("Alice", "works_at", "Acme Corp")
    graph.add("Bob", "works_at", "Acme Corp")
    graph.add("Charlie", "knows", "Alice")
    assert len(graph.query(like="acme")) == 2
    assert len(graph.query(like="charlie")) == 1


def test_query_exact_and_like_combined(graph):
    graph.add("Alice", "knows", "Bob", source="chat")
    graph.add("Alice", "knows", "Charlie")
    results = graph.query(subject="Alice", like="bob")
    assert len(results) == 1
    assert results[0].object == "Bob"


def test_neighbors_outgoing(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Alice", "works_at", "Acme")
    graph.add("Charlie", "knows", "Alice")
    out = graph.neighbors("Alice")
    assert len(out["outgoing"]) == 2
    assert out["incoming"] == []


def test_neighbors_incoming(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Charlie", "knows", "Alice")
    inn = graph.neighbors("Alice", direction="in")
    assert len(inn["incoming"]) == 1
    assert inn["incoming"][0]["subject"] == "Charlie"
    assert inn["outgoing"] == []


def test_neighbors_both_directions(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Charlie", "knows", "Alice")
    both = graph.neighbors("Alice", direction="both")
    assert len(both["outgoing"]) == 1
    assert len(both["incoming"]) == 1


def test_neighbors_filtered_by_predicate(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Alice", "works_at", "Acme")
    out = graph.neighbors("Alice", predicate="knows")
    assert len(out["outgoing"]) == 1
    assert out["outgoing"][0]["object"] == "Bob"


def test_shortest_path(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Bob", "knows", "Charlie")
    graph.add("Charlie", "works_at", "Acme")
    trail = graph.find_path("Alice", "Acme")
    assert len(trail) == 3
    assert trail[0] == Fact("Alice", "knows", "Bob")
    assert trail[2] == Fact("Charlie", "works_at", "Acme")


def test_path_not_found(graph):
    graph.add("Alice", "knows", "Bob")
    assert graph.find_path("Alice", "Charlie") == []


def test_path_respects_max_depth(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Bob", "knows", "Charlie")
    graph.add("Charlie", "knows", "Dana")
    assert graph.find_path("Alice", "Dana", max_depth=2) == []
    assert len(graph.find_path("Alice", "Dana", max_depth=3)) == 3


def test_stats(graph):
    graph.add("Alice", "knows", "Bob")
    graph.add("Alice", "works_at", "Acme")
    assert graph.stats() == {
        "facts": 2,
        "entities": 3,
        "subjects": 1,
        "objects": 2,
        "predicates": 2,
    }


def test_persistence(tmp_path):
    path = tmp_path / "graph.jsonl"
    g1 = KnowledgeGraph(path)
    g1.add("A", "b", "C")
    g2 = KnowledgeGraph(path)
    assert len(g2._facts) == 1
    assert g2._facts[0] == Fact("A", "b", "C")


def test_tools(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.jsonl")
    tools = {t.name: t for t in make_graph_tools(graph)}

    # add
    result = tools["graph_add"].handler(
        {"subject": "Alice", "predicate": "knows", "object": "Bob"}
    )
    assert result.is_error is False
    assert len(graph._facts) == 1

    # add with missing field -> error
    bad = tools["graph_add"].handler({"subject": "Alice"})
    assert bad.is_error and "Missing required fields" in bad.content

    # query exact
    query = tools["graph_query"].handler({"subject": "Alice"})
    assert not query.is_error and "Bob" in query.content

    # query like
    like = tools["graph_query"].handler({"like": "ali"})
    assert not like.is_error and "Alice" in like.content

    # neighbors
    nb = tools["graph_neighbors"].handler({"entity": "Alice"})
    assert not nb.is_error and "Bob" in nb.content

    # path
    graph.add("Bob", "works_at", "Acme")
    path = tools["graph_path"].handler({"from": "Alice", "to": "Acme"})
    assert not path.is_error and "works_at" in path.content

    # stats
    stats = tools["graph_stats"].handler({})
    assert not stats.is_error and "facts" in stats.content
