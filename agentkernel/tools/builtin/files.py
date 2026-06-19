"""Filesystem tools: read_file, write_file, list_dir (design §6.3).

All paths are confined to the configured working directory: ``..`` escapes and
absolute paths outside the root are rejected with an error result (never a
raise). ``read_file`` truncates large files via the shared §8.4/§9 mechanism.
"""

from __future__ import annotations

from pathlib import Path

from agentkernel.context.truncate import truncate_text
from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult


def file_tools(working_dir: str = ".", *, max_result_tokens: int = 4096) -> list[ToolSpec]:
    """Build the file toolset bound to ``working_dir``.

    Binding the root (and result cap) here keeps handlers pure functions of
    their arguments — they never reach for global config (AGENT.md, design §7).
    """
    root = Path(working_dir).resolve()

    def _resolve(path: str) -> Path:
        """Resolve ``path`` under the root, or raise ValueError if it escapes."""
        candidate = (root / path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"path escapes working directory: {path!r}")
        return candidate

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
                "working directory. Parent directories are created as needed."
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
    ]
