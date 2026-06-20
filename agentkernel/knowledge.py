"""Knowledge graph seam (design §13, Phase 6).

A tiny, file-backed triple store. It is exposed to the loop as ordinary
registered tools so the kernel itself does not need any special state for it.

The feature set is intentionally minimal but no longer a stub: exact and
substring queries, one-hop neighbors, shortest-path traversal, and basic stats
are exposed as ordinary tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str
    source: str | None = None

    def matches(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        like: str | None = None,
    ) -> bool:
        if subject is not None and self.subject != subject:
            return False
        if predicate is not None and self.predicate != predicate:
            return False
        if object is not None and self.object != object:
            return False
        if like is not None:
            needle = like.lower()
            haystack = f"{self.subject} {self.predicate} {self.object}".lower()
            if needle not in haystack:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source,
        }


class KnowledgeGraph:
    """Append-only triple store backed by a JSONL file.

    Exact (subject, predicate, object) triples are deduplicated on add so the
    graph cannot grow without bound when the model repeats a fact. Queries are
    exact by default; pass ``like`` for a case-insensitive substring search.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path(".agentkernel/graph.jsonl")
        self._facts: list[Fact] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        facts: list[Fact] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                facts.append(
                    Fact(
                        subject=data.get("subject", "").strip(),
                        predicate=data.get("predicate", "").strip(),
                        object=data.get("object", "").strip(),
                        source=data.get("source"),
                    )
                )
        self._facts = facts

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for fact in self._facts:
                handle.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")

    def add(
        self,
        subject: str,
        predicate: str,
        object: str,
        source: str | None = None,
    ) -> Fact:
        """Add a fact. Exact (subject, predicate, object) duplicates are ignored."""
        fact = Fact(subject.strip(), predicate.strip(), object.strip(), source)
        for existing in self._facts:
            if existing == fact:
                return existing
        self._facts.append(fact)
        self._save()
        return fact

    def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        like: str | None = None,
    ) -> list[Fact]:
        """Return facts matching all provided exact filters and optional substring."""
        return [
            fact
            for fact in self._facts
            if fact.matches(
                subject=subject, predicate=predicate, object=object, like=like
            )
        ]

    def neighbors(
        self,
        entity: str,
        *,
        predicate: str | None = None,
        direction: str = "out",
    ) -> dict[str, Any]:
        """Return one-hop neighbors of ``entity``.

        ``direction`` is one of ``out`` (subject -> object), ``in`` (object ->
        subject), or ``both``. Both ``incoming`` and ``outgoing`` keys are always
        present so callers have a stable JSON shape.
        """
        entity = entity.strip()
        outgoing: list[dict[str, Any]] = []
        incoming: list[dict[str, Any]] = []

        for fact in self._facts:
            if predicate is not None and fact.predicate != predicate:
                continue
            if direction in ("out", "both") and fact.subject == entity:
                outgoing.append({"predicate": fact.predicate, "object": fact.object})
            if direction in ("in", "both") and fact.object == entity:
                incoming.append({"predicate": fact.predicate, "subject": fact.subject})

        result = {"entity": entity, "outgoing": outgoing, "incoming": incoming}
        return result

    def find_path(
        self,
        from_subject: str,
        to_entity: str,
        *,
        max_depth: int = 5,
    ) -> list[Fact]:
        """Shortest undirected path between two entities, up to ``max_depth`` hops."""
        start = from_subject.strip()
        goal = to_entity.strip()
        if start == goal:
            return []

        # BFS over an undirected view of the graph.
        visited: set[str] = {start}
        queue: list[tuple[str, list[Fact]]] = [(start, [])]

        while queue:
            current, trail = queue.pop(0)
            if len(trail) >= max_depth:
                continue
            for fact in self._facts:
                if fact.subject == current:
                    next_node = fact.object
                elif fact.object == current:
                    next_node = fact.subject
                else:
                    continue
                if next_node in visited:
                    continue
                new_trail = trail + [fact]
                if next_node == goal:
                    return new_trail
                visited.add(next_node)
                queue.append((next_node, new_trail))

        return []

    def stats(self) -> dict[str, int]:
        """Basic graph metrics."""
        subjects = {f.subject for f in self._facts}
        objects = {f.object for f in self._facts}
        predicates = {f.predicate for f in self._facts}
        return {
            "facts": len(self._facts),
            "entities": len(subjects | objects),
            "subjects": len(subjects),
            "objects": len(objects),
            "predicates": len(predicates),
        }

    def to_dicts(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self._facts]


def make_graph_tools(graph: KnowledgeGraph) -> list[Any]:
    """Return ToolSpec instances for the knowledge-graph tools.

    Importing ``ToolSpec`` here avoids a circular import at module load time.
    """
    from agentkernel.tools import ToolSpec
    from agentkernel.types import ToolResult

    def _missing_fields(
        arguments: dict[str, Any], required: set[str]
    ) -> ToolResult | None:
        """Return an error ToolResult if any required field is missing or empty."""
        missing = required - arguments.keys()
        if missing:
            return ToolResult(
                "",
                f"Missing required fields: {sorted(missing)}",
                is_error=True,
            )
        empty_required = [
            k for k in required if not str(arguments.get(k, "")).strip()
        ]
        if empty_required:
            return ToolResult(
                "",
                f"Empty required fields: {sorted(empty_required)}",
                is_error=True,
            )
        return None

    def _add(arguments: dict[str, Any]) -> ToolResult:
        err = _missing_fields(arguments, {"subject", "predicate", "object"})
        if err is not None:
            return err
        graph.add(
            arguments["subject"],
            arguments["predicate"],
            arguments["object"],
            arguments.get("source"),
        )
        return ToolResult("", "Fact added.")

    def _query(arguments: dict[str, Any]) -> ToolResult:
        results = graph.query(
            subject=arguments.get("subject"),
            predicate=arguments.get("predicate"),
            object=arguments.get("object"),
            like=arguments.get("like"),
        )
        return ToolResult(
            "",
            json.dumps([f.to_dict() for f in results], ensure_ascii=False),
        )

    def _neighbors(arguments: dict[str, Any]) -> ToolResult:
        err = _missing_fields(arguments, {"entity"})
        if err is not None:
            return err
        result = graph.neighbors(
            arguments["entity"],
            predicate=arguments.get("predicate"),
            direction=arguments.get("direction", "out"),
        )
        return ToolResult("", json.dumps(result, ensure_ascii=False))

    def _path(arguments: dict[str, Any]) -> ToolResult:
        err = _missing_fields(arguments, {"from", "to"})
        if err is not None:
            return err
        trail = graph.find_path(
            arguments["from"],
            arguments["to"],
            max_depth=int(arguments.get("max_depth", 5)),
        )
        return ToolResult(
            "", json.dumps([f.to_dict() for f in trail], ensure_ascii=False)
        )

    def _stats(arguments: dict[str, Any]) -> ToolResult:
        return ToolResult("", json.dumps(graph.stats(), ensure_ascii=False))

    return [
        ToolSpec(
            name="graph_add",
            description=(
                "Add a fact to the knowledge graph as (subject, predicate, object). "
                "Exact duplicates are ignored."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Entity"},
                    "predicate": {"type": "string", "description": "Relationship"},
                    "object": {"type": "string", "description": "Entity or value"},
                    "source": {"type": "string", "description": "Optional source note"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
            handler=_add,
            mutates=True,
        ),
        ToolSpec(
            name="graph_query",
            description=(
                "Query facts in the knowledge graph. Any field may be omitted to match all. "
                "Use `like` for a case-insensitive substring search across subjects, predicates, or objects."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "like": {"type": "string", "description": "Case-insensitive substring"},
                },
                "additionalProperties": False,
            },
            handler=_query,
        ),
        ToolSpec(
            name="graph_neighbors",
            description=(
                "List one-hop neighbors of an entity. direction may be 'out' (default), "
                "'in', or 'both'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "predicate": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                        "default": "out",
                    },
                },
                "required": ["entity"],
                "additionalProperties": False,
            },
            handler=_neighbors,
        ),
        ToolSpec(
            name="graph_path",
            description=(
                "Find the shortest undirected path between two entities, up to max_depth hops."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 5},
                },
                "required": ["from", "to"],
                "additionalProperties": False,
            },
            handler=_path,
        ),
        ToolSpec(
            name="graph_stats",
            description="Return counts of facts, entities, and predicates in the graph.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_stats,
        ),
    ]
