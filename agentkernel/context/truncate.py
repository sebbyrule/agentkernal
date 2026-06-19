"""Shared truncation (design §8.4 / §9).

Both large tool results and context management use this single mechanism so the
marker and policy stay consistent. M0 needs it for ``read_file``; M2's context
manager reuses it. Truncation keeps the head and tail and replaces the middle
with a marker, which is more useful to the model than a hard cut.
"""

from __future__ import annotations

# Rough chars-per-token heuristic for budgeting; only needs to be conservative.
CHARS_PER_TOKEN = 4


def truncate_text(text: str, max_tokens: int, *, hint: str = "use a narrower query") -> str:
    """Truncate ``text`` to roughly ``max_tokens``, marking the removed middle.

    Returns the text unchanged if it already fits. The marker states how many
    bytes were dropped so the model knows the result is partial.
    """
    max_chars = max(max_tokens * CHARS_PER_TOKEN, 0)
    if len(text) <= max_chars or max_chars == 0:
        return text

    marker_template = "\n… [truncated {n} bytes; {hint}] …\n"
    # Reserve room for the marker, then split the budget between head and tail.
    marker = marker_template.format(n=0, hint=hint)
    body_chars = max(max_chars - len(marker), 0)
    head_chars = body_chars // 2
    tail_chars = body_chars - head_chars

    head = text[:head_chars]
    tail = text[len(text) - tail_chars :] if tail_chars else ""
    dropped = len(text) - len(head) - len(tail)
    return head + marker_template.format(n=dropped, hint=hint) + tail
