"""Knowledge graph tests (Phase 6, design §13).

The graph is exposed as ordinary registered tools so the kernel does not need
special knowledge-graph state.
"""

from __future__ import annotations

import json

from agentkernel.knowledge import KnowledgeGraph, make_graph_tools


def test_knowledge_graph_adds_and_queries(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.jsonl")
    graph.add("agentkernel", "implements", "agent loop", source="tests")
    graph.add("agentkernel", "uses", "tool registry")
    results = graph.query(subject="agentkernel")
    assert len(results) == 2


def test_knowledge_graph_query_filters(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.jsonl")
    graph.add("a", "b", "c")
    graph.add("a", "b", "d")
    graph.add("x", "y", "z")
    assert len(graph.query(predicate="b")) == 2
    assert len(graph.query(object="z")) == 1
    assert len(graph.query(subject="a", object="d")) == 1


def test_knowledge_graph_persists_to_file(tmp_path):
    path = tmp_path / "graph.jsonl"
    graph = KnowledgeGraph(path)
    graph.add("python", "is", "snake")
    del graph
    reloaded = KnowledgeGraph(path)
    assert len(reloaded.query()) == 1
    assert reloaded.query()[0].object == "snake"


def test_graph_add_tool_adds_fact(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.jsonl")
    add_tool, query_tool = make_graph_tools(graph)
    result = add_tool.handler(
        {"subject": "test", "predicate": "runs", "object": "tool"}
    )
    assert not result.is_error
    assert len(graph.query()) == 1


def test_graph_query_tool_returns_json(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.jsonl")
    graph.add("test", "has", "value")
    add_tool, query_tool = make_graph_tools(graph)
    result = query_tool.handler({"subject": "test"})
    data = json.loads(result.content)
    assert len(data) == 1
    assert data[0]["object"] == "value"


def test_graph_add_tool_reports_missing_fields(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.jsonl")
    add_tool, _ = make_graph_tools(graph)
    result = add_tool.handler({"subject": "x", "predicate": "y"})
    assert result.is_error
    assert "object" in result.content
