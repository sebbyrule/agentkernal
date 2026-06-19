"""Knowledge graph seam (design §13, Phase 6).

A tiny, file-backed triple store. It is exposed to the loop as ordinary
registered tools so the kernel itself does not need any special state for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    source: str | None = None


class KnowledgeGraph:
    """Append-only triple store backed by a JSONL file.

    Each line is ``{"subject": ..., "predicate": ..., "object": ...,
    "source": ...}``. Reads load the whole graph into memory; this is
    intentionally primitive — Phase 6 is a seam for later specialized stores.
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
                        subject=data.get("subject", ""),
                        predicate=data.get("predicate", ""),
                        object=data.get("object", ""),
                        source=data.get("source"),
                    )
                )
        self._facts = facts

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for fact in self._facts:
                handle.write(
                    json.dumps(
                        {
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "source": fact.source,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def add(self, subject: str, predicate: str, object: str, source: str | None = None) -> Fact:
        fact = Fact(subject, predicate, object, source)
        self._facts.append(fact)
        self._save()
        return fact

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[Fact]:
        results: list[Fact] = []
        for fact in self._facts:
            if subject is not None and fact.subject != subject:
                continue
            if predicate is not None and fact.predicate != predicate:
                continue
            if object is not None and fact.object != object:
                continue
            results.append(fact)
        return results

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "source": f.source,
            }
            for f in self._facts
        ]


def make_graph_tools(graph: KnowledgeGraph) -> list[Any]:
    """Return ToolSpec instances for ``graph_add`` and ``graph_query``.

    Importing ``ToolSpec`` here avoids a circular import at module load time.
    """
    from agentkernel.tools import ToolSpec
    from agentkernel.types import ToolResult

    def _add(arguments: dict[str, Any]) -> ToolResult:
        required = {"subject", "predicate", "object"}
        missing = required - arguments.keys()
        if missing:
            return ToolResult(
                "",
                f"Missing required fields: {sorted(missing)}",
                is_error=True,
            )
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
        )
        return ToolResult("", json.dumps([{"subject": f.subject, "predicate": f.predicate, "object": f.object, "source": f.source} for f in results], ensure_ascii=False))

    return [
        ToolSpec(
            name="graph_add",
            description="Add a fact to the knowledge graph as (subject, predicate, object).",
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Entity"},
                    "predicate": {"type": "string", "description": "Relationship"},
                    "object": {"type": "string", "description": "Entity or value"},
                    "source": {"type": "string", "description": "Optional source note"},
                },
                "required": ["subject", "predicate", "object"],
            },
            handler=_add,
            mutates=True,
        ),
        ToolSpec(
            name="graph_query",
            description="Query facts in the knowledge graph. Any field may be omitted to match all.",
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
            },
            handler=_query,
        ),
    ]
