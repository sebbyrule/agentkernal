"""Filesystem checkpoints (design §18.1).

When enabled, the builtin file tools record a file's contents *before* they first
modify it this session. The ``rollback`` tool then restores every recorded file
to that original state — undoing the agent's edits in one step, including files it
created (which are deleted on rollback). This makes a destructive run reversible
without trusting the model to clean up after itself.

Backups are held per session (in memory); the first time a path is touched its
original bytes are captured, so repeated edits to the same file still roll back to
the pre-run state, not the previous edit.
"""

from __future__ import annotations

from pathlib import Path


class Checkpointer:
    """Records pre-modification file state and restores it on rollback."""

    def __init__(self) -> None:
        # path -> original bytes, or None if the file did not exist yet.
        self._original: dict[Path, bytes | None] = {}

    def record(self, path: Path) -> None:
        """Capture ``path``'s current state, once, before it is first modified."""
        key = path.resolve()
        if key in self._original:
            return  # already captured the pre-run state; keep the earliest
        self._original[key] = key.read_bytes() if key.is_file() else None

    def pending(self) -> int:
        """How many files have a recorded checkpoint."""
        return len(self._original)

    def rollback(self) -> int:
        """Restore every recorded file to its captured state. Returns the count."""
        restored = 0
        for path, content in self._original.items():
            if content is None:
                # The file did not exist at checkpoint time → remove it.
                if path.is_file():
                    path.unlink()
                    restored += 1
            else:
                path.write_bytes(content)
                restored += 1
        self._original.clear()
        return restored
