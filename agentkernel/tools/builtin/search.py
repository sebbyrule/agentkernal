"""Read-only discovery tools: find_files, search_text, file_info (design §6.3).

These let the model locate and inspect code without shelling out to `bash`, so
they work the same on every platform and inside the no-network Docker sandbox.
Like the file tools, every path is confined to the working directory and every
failure is returned as an error result, never raised (AGENT.md, design §8.3).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from agentkernel.context.truncate import truncate_text
from agentkernel.tools.base import ToolSpec
from agentkernel.tools.builtin.files import resolve_within
from agentkernel.types import ToolResult

# Directories that are never worth walking for code search.
_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".agentkernel", ".pytest_cache"}


def _is_probably_binary(data: bytes) -> bool:
    """A cheap heuristic: a NUL byte in the first chunk means "not text"."""
    return b"\x00" in data[:1024]


def _iter_files(base: Path, pattern: str):
    """Yield files under ``base`` matching ``pattern``, skipping noise dirs."""
    for p in sorted(base.glob(pattern)):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        yield p


def search_tools(working_dir: str = ".", *, max_result_tokens: int = 4096) -> list[ToolSpec]:
    """Build the read-only discovery toolset bound to ``working_dir``."""
    root = Path(working_dir).resolve()

    def _rel(p: Path) -> str:
        return p.relative_to(root).as_posix()

    def find_files(args: dict) -> ToolResult:
        pattern = args["pattern"]
        sub = args.get("path", ".")
        try:
            base = resolve_within(root, sub)
        except ValueError as exc:
            return ToolResult("", str(exc), is_error=True)
        if not base.is_dir():
            return ToolResult("", f"Not a directory: {sub!r}", is_error=True)
        matches = []
        for p in sorted(base.glob(pattern)):
            if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
                continue
            matches.append(_rel(p) + ("/" if p.is_dir() else ""))
        if not matches:
            return ToolResult("", f"(no files match {pattern!r})")
        return ToolResult("", truncate_text("\n".join(matches), max_result_tokens))

    def search_text(args: dict) -> ToolResult:
        pattern = args["pattern"]
        glob = args.get("glob", "**/*")
        sub = args.get("path", ".")
        max_results = int(args.get("max_results", 100))
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return ToolResult("", f"Invalid regex {pattern!r}: {exc}", is_error=True)
        try:
            base = resolve_within(root, sub)
        except ValueError as exc:
            return ToolResult("", str(exc), is_error=True)
        if not base.is_dir():
            return ToolResult("", f"Not a directory: {sub!r}", is_error=True)

        hits: list[str] = []
        truncated = False
        for f in _iter_files(base, glob):
            try:
                raw = f.read_bytes()
            except OSError:
                continue
            if _is_probably_binary(raw):
                continue
            rel = _rel(f)
            for lineno, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(hits) >= max_results:
                        truncated = True
                        break
            if truncated:
                break
        if not hits:
            return ToolResult("", f"(no matches for {pattern!r})")
        body = "\n".join(hits)
        if truncated:
            body += f"\n... (stopped at {max_results} matches)"
        return ToolResult("", truncate_text(body, max_result_tokens))

    def file_info(args: dict) -> ToolResult:
        path = args["path"]
        try:
            target = resolve_within(root, path)
        except ValueError as exc:
            return ToolResult("", str(exc), is_error=True)
        if not target.exists():
            return ToolResult("", f"No such path: {path!r}", is_error=True)
        st = target.stat()
        modified = datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(timespec="seconds")
        kind = "directory" if target.is_dir() else "file"
        lines = [
            f"path: {_rel(target)}",
            f"type: {kind}",
            f"size: {st.st_size} bytes",
            f"modified: {modified}",
        ]
        if target.is_dir():
            lines.append(f"entries: {sum(1 for _ in target.iterdir())}")
        elif not _is_probably_binary(target.read_bytes()[:1024]):
            line_count = target.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            lines.append(f"lines: {line_count}")
        else:
            lines.append("content: binary")
        return ToolResult("", "\n".join(lines))

    return [
        ToolSpec(
            name="find_files",
            description=(
                "Find files and directories by glob pattern within the working "
                "directory — your first move when you know roughly what a file is "
                "named but not where it lives. Supports `**` for recursive match "
                "(e.g. '**/*.py', 'src/**/test_*.py'). Returns matching paths; "
                "noise dirs like .git and __pycache__ are skipped."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py' or 'docs/*.md'.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Subdirectory to search under (default: the working dir root)."
                        ),
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=find_files,
            category="search",
        ),
        ToolSpec(
            name="search_text",
            description=(
                "Search file contents by regular expression within the working "
                "directory (a built-in grep) — use it to locate where a symbol, "
                "string, or pattern is defined or used. Returns 'path:line: text' "
                "for each match. Filter the files searched with `glob` and narrow "
                "with `path`; binary files and noise dirs are skipped."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regular expression to match against each line.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Restrict to files matching this glob (default '**/*').",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Subdirectory to search under (default: the working dir root)."
                        ),
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive match.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Stop after this many matches (default 100).",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=search_text,
            category="search",
        ),
        ToolSpec(
            name="file_info",
            description=(
                "Report metadata for a path within the working directory: whether "
                "it is a file or directory, its size, last-modified time, and line "
                "count (for text). Cheaper than reading a whole file when you only "
                "need to know it exists or how big it is."
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
            handler=file_info,
            category="search",
        ),
    ]
