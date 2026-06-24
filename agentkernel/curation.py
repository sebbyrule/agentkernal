"""Self-curating memory: extract durable facts from a transcript, and
consolidate the notebook with an LLM (design §13, Phase 3).

The storage/recall machinery (``NoteStore``, semantic search, dedup) is mature;
what was missing is *populating and cleaning* it without the model having to call
``remember`` in the moment. This module adds two harness operations on top of the
kernel (an Agent is not required — just a ``NoteStore`` and a ``Provider``):

* ``extract(messages)`` — distil a finished conversation into candidate facts,
  skipping ones that duplicate (by token overlap) what is already stored.
* ``consolidate()`` — ask the model to merge related notes and supersede
  outdated ones, then rebuild the notebook from the cleaned set.

Both are best-effort: an unparseable model reply leaves memory unchanged rather
than raising or destroying notes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentkernel.memory import MemoryNote, _tokens
from agentkernel.types import Message

if TYPE_CHECKING:
    from agentkernel.memory import NoteStore
    from agentkernel.providers import Provider

_EXTRACT_SYSTEM = (
    "You curate an AI agent's long-term memory. From a conversation, extract only "
    "DURABLE facts worth remembering across future sessions: stable user "
    "preferences, project facts and conventions, decisions, and constraints. "
    "Exclude transient task details, pleasantries, and anything ephemeral. "
    'Respond with ONLY a JSON array of objects {"text": "...", "tags": ["..."]}. '
    "Return [] if nothing is worth remembering."
)

_CONSOLIDATE_SYSTEM = (
    "You consolidate an AI agent's long-term memory notes. Produce a cleaner set: "
    "merge notes that say the same or closely related things, remove redundancy, "
    "and when two notes conflict keep only the most recent/true statement. Do not "
    "drop any distinct information. "
    'Respond with ONLY a JSON array of objects {"text": "...", "tags": ["..."]}.'
)


@dataclass
class ExtractionResult:
    added: list[MemoryNote] = field(default_factory=list)
    skipped_duplicates: int = 0


@dataclass
class ConsolidationResult:
    before: int
    after: int
    notes: list[MemoryNote] = field(default_factory=list)

    @property
    def removed(self) -> int:
        return max(self.before - self.after, 0)


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array of objects from a model reply, tolerating prose."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [item for item in data if isinstance(item, dict)]


def _render_transcript(messages: list[Message], max_chars: int) -> str:
    lines: list[str] = []
    for m in messages:
        if m.content and m.role in ("user", "assistant"):
            lines.append(f"{m.role}: {m.content}")
        for tc in m.tool_calls:
            lines.append(f"assistant called {tc.name}")
        for r in m.tool_results:
            snippet = r.content if len(r.content) < 200 else r.content[:200] + "…"
            lines.append(f"tool[{'error' if r.is_error else 'ok'}]: {snippet}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… [transcript truncated]"
    return text


class MemoryCurator:
    """Populates and cleans a ``NoteStore`` with the help of a model."""

    def __init__(
        self,
        notes: "NoteStore",
        provider: "Provider",
        *,
        max_tokens: int = 1024,
        dedup_threshold: float = 0.8,
        transcript_chars: int = 16000,
    ) -> None:
        self._notes = notes
        self._provider = provider
        self._max_tokens = max_tokens
        self._dedup_threshold = dedup_threshold
        self._transcript_chars = transcript_chars

    def extract(self, messages: list[Message]) -> ExtractionResult:
        transcript = _render_transcript(messages, self._transcript_chars)
        if not transcript.strip():
            return ExtractionResult()
        candidates = self._ask(
            _EXTRACT_SYSTEM,
            f"Conversation:\n{transcript}\n\nExtract the durable facts.",
        )
        existing = self._notes.all()
        result = ExtractionResult()
        for cand in candidates:
            text = str(cand.get("text", "")).strip()
            if not text:
                continue
            if self._is_duplicate(text, existing):
                result.skipped_duplicates += 1
                continue
            note = self._notes.add(text, tags=cand.get("tags") or [])
            existing.append(note)  # dedup later candidates against it too
            result.added.append(note)
        return result

    def consolidate(self) -> ConsolidationResult:
        existing = self._notes.all()
        if len(existing) < 2:
            return ConsolidationResult(len(existing), len(existing), existing)
        listing = "\n".join(
            f"{n.note_id}. {n.text}"
            + (f"  [tags: {', '.join(n.tags)}]" if n.tags else "")
            for n in existing
        )
        cleaned = [
            c for c in self._ask(_CONSOLIDATE_SYSTEM, f"Current memory notes:\n{listing}")
            if str(c.get("text", "")).strip()
        ]
        if not cleaned:
            return ConsolidationResult(len(existing), len(existing), existing)  # no-op
        # Rebuild from the consolidated set using the store's public API so all
        # backends (JSONL / SQLite / semantic) stay consistent.
        for note in existing:
            self._notes.forget(note_id=note.note_id)
        new_notes = [
            self._notes.add(str(c["text"]).strip(), tags=c.get("tags") or [])
            for c in cleaned
        ]
        return ConsolidationResult(len(existing), len(new_notes), new_notes)

    # --- internals ---------------------------------------------------------

    def _ask(self, system: str, user: str) -> list[dict]:
        resp = self._provider.complete(
            [Message(role="user", content=user)],
            [],
            max_tokens=self._max_tokens,
            temperature=0.0,
            system=system,
        )
        return _parse_json_array(resp.message.content)

    def _is_duplicate(self, text: str, existing: list[MemoryNote]) -> bool:
        terms = _tokens(text)
        if not terms:
            return False
        for note in existing:
            other = _tokens(note.text)
            if not other:
                continue
            jaccard = len(terms & other) / len(terms | other)
            if jaccard >= self._dedup_threshold:
                return True
        return False
