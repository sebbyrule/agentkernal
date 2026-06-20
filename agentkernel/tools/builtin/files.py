"""Filesystem tools: read_file, write_file, list_dir, edit_file (design §6.3).

All paths are confined to the configured working directory: ``..`` escapes and
absolute paths outside the root are rejected with an error result (never a
raise). ``read_file`` truncates large files via the shared §8.4/§9 mechanism.
"""

from __future__ import annotations

from pathlib import Path

from agentkernel.context.truncate import truncate_text
from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult


def resolve_within(root: Path, path: str) -> Path:
    """Resolve ``path`` under ``root``, or raise ValueError if it escapes.

    Shared by every working-dir-confined tool so the containment rule lives in
    one place (design §6.3, §10.3).
    """
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes working directory: {path!r}")
    return candidate


def file_tools(working_dir: str = ".", *, max_result_tokens: int = 4096) -> list[ToolSpec]:
    """Build the file toolset bound to ``working_dir``.

    Binding the root (and result cap) here keeps handlers pure functions of
    their arguments — they never reach for global config (AGENT.md, design §7).
    """
    root = Path(working_dir).resolve()

    def _resolve(path: str) -> Path:
        return resolve_within(root, path)

    def read_file(args: dict) -> ToolResult:
        path = args["path"]
        try:
            target = _resolve(path)
        except ValueError as exc:
            return ToolResult("", str(exc), is_error=True)
        if not target.is_file():
            return ToolResult("", f"Not a file: {path!r}", is_error=True)
        text = target.read_text(encoding="utf-8", errors="replace")
        return ToolResult("", truncate_text(text, max_result_tokens))

    def write_file(args: dict) -> ToolResult:
        path = args["path"]
        content = args["content"]
        try:
            target = _resolve(path)
        except ValueError as exc:
            return ToolResult("", str(exc), is_error=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult("", f"Wrote {len(content)} bytes to {path}")

    def list_dir(args: dict) -> ToolResult:
        path = args.get("path", ".")
        try:
            target = _resolve(path)
        except ValueError as exc:
            return ToolResult("", str(exc), is_error=True)
        if not target.is_dir():
            return ToolResult("", f"Not a directory: {path!r}", is_error=True)
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
        )
        listing = "\n".join(entries) if entries else "(empty)"
        return ToolResult("", truncate_text(listing, max_result_tokens))

    def edit_file(args: dict) -> ToolResult:
        path = args["path"]
        old = args["old"]
        new = args["new"]
        replace_all = bool(args.get("replace_all", False))
        try:
            target = _resolve(path)
        except ValueError as exc:
            return ToolResult("", str(exc), is_error=True)
        if not target.is_file():
            return ToolResult("", f"Not a file: {path!r}", is_error=True)
        if old == new:
            return ToolResult("", "`old` and `new` are identical; nothing to do.", is_error=True)
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult("", f"`old` text not found in {path!r}.", is_error=True)
        if count > 1 and not replace_all:
            return ToolResult(
                "",
                f"`old` text is not unique in {path!r} ({count} occurrences); "
                "pass replace_all=true or include more surrounding context.",
                is_error=True,
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        target.write_text(updated, encoding="utf-8")
        replaced = count if replace_all else 1
        return ToolResult("", f"Replaced {replaced} occurrence(s) in {path}.")

    return [
        ToolSpec(
            name="read_file",
            description=(
                "Read a UTF-8 text file within the working directory. Returns the "
                "file contents; large files are truncated with a marker."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the working directory.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=read_file,
            category="files",
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Write (creating or overwriting) a UTF-8 text file within the "
                "working directory. Parent directories are created as needed. To "
                "change part of an existing file, prefer edit_file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=write_file,
            mutates=True,
            requires_approval=True,
            category="files",
        ),
        ToolSpec(
            name="list_dir",
            description="List the entries of a directory within the working directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the working directory.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=list_dir,
            category="files",
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Replace an exact substring in a text file within the working "
                "directory — the surgical alternative to rewriting the whole file "
                "with write_file. `old` must match the file byte-for-byte and be "
                "unique unless replace_all is true. Fails (without writing) if "
                "`old` is missing or ambiguous, so include enough surrounding "
                "context to pin down the one spot you mean."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to edit, relative to the working dir.",
                    },
                    "old": {
                        "type": "string",
                        "description": (
                            "Exact text to find; add surrounding lines to make it unique."
                        ),
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Replace every occurrence instead of needing a unique match."
                        ),
                    },
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
            handler=edit_file,
            mutates=True,
            requires_approval=True,
            category="files",
        ),
    ]
