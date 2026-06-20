"""Skills, Anthropic-style (design §13, Phase 4).

A skill is a reusable bundle of instructions the model can apply. This follows
the Anthropic Agent Skills shape: a ``SKILL.md`` with YAML frontmatter
(``name`` + ``description``) and a markdown body, optionally alongside bundled
scripts/resources in the same folder.

Disclosure is **progressive** (hybrid): only each skill's name + description
sits in the cacheable prefix (a lightweight catalog, always present), and the
model loads a skill's full body + file listing on demand via the ``use_skill``
tool. A skill may additionally be *pinned* (``active``) so its full body is
injected into the prefix up front.

Discovery accepts three layouts so existing skills keep working:
  * ``<dir>/<skill>/SKILL.md`` — Anthropic-style folder (bundled files allowed)
  * ``<dir>/<skill>.md`` — a loose markdown file (optional ``---`` frontmatter)
  * ``<dir>/<skill>.toml`` — frontmatter-only (``name`` + ``system_prompt``)
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Skill:
    """A named skill: a description (for disclosure) + a body (the instructions)."""

    name: str
    description: str
    body: str
    source: Path | None = None
    resources: tuple[str, ...] = ()  # bundled file paths, relative to cwd

    @property
    def system_prompt(self) -> str:  # back-compat alias
        return self.body


@runtime_checkable
class ContextSource(Protocol):
    """Anything that contributes extra system-prompt text per run."""

    def system_additions(self) -> list[str]: ...


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split a ``---`` delimited frontmatter block from the body.

    Returns ``(metadata, body)``. The frontmatter is parsed as a small YAML
    subset (flat ``key: value`` plus ``- item`` lists) — enough for SKILL.md,
    without taking a YAML dependency.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    # Find the closing '---' after the opening one.
    close = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if close is None:
        return {}, text
    meta = _parse_yaml_subset(lines[1:close])
    body = "\n".join(lines[close + 1 :]).strip()
    return meta, body


def _parse_yaml_subset(lines: Sequence[str]) -> dict[str, object]:
    meta: dict[str, object] = {}
    list_key: str | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if list_key is not None and raw.lstrip().startswith("- "):
            if not isinstance(meta.get(list_key), list):
                meta[list_key] = []  # convert the empty-scalar placeholder to a list
            meta[list_key].append(_unquote(raw.lstrip()[2:].strip()))  # type: ignore[union-attr]
            continue
        list_key = None
        key, sep, value = raw.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value == "":
            list_key = key  # a block list may follow
            meta[key] = ""
        else:
            meta[key] = _unquote(value)
    return meta


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


class SkillLibrary:
    """Discovers skills in a directory and exposes them as a ContextSource.

    ``active_skills`` are *pinned*: their full bodies join the prefix in addition
    to the always-present catalog.
    """

    def __init__(self, directory: str | Path, active_skills: Sequence[str] | None = None) -> None:
        self.directory = Path(directory) if directory else Path("skills")
        self.active_skills: set[str] = set(active_skills or [])
        self._skills: dict[str, Skill] = {}
        if self.directory.is_dir():
            self._load()

    # --- discovery ---------------------------------------------------------

    def _load(self) -> None:
        for path in sorted(self.directory.iterdir()):
            skill: Skill | None = None
            if path.is_dir() and (path / "SKILL.md").is_file():
                skill = self._load_skill_md(path / "SKILL.md", bundle_dir=path)
            elif path.suffix.lower() == ".md":
                skill = self._load_skill_md(path, bundle_dir=None)
            elif path.suffix.lower() == ".toml":
                skill = self._load_toml(path)
            if skill is not None:
                self._skills[skill.name] = skill

    def _load_skill_md(self, path: Path, *, bundle_dir: Path | None) -> Skill | None:
        text = path.read_text(encoding="utf-8")
        meta, body = _split_frontmatter(text)
        name = str(
            meta.get("name") or path.stem
            if not bundle_dir
            else meta.get("name") or bundle_dir.name
        )
        description = str(meta.get("description") or _first_line(body))
        resources: tuple[str, ...] = ()
        if bundle_dir is not None:
            resources = tuple(
                _relpath(p)
                for p in sorted(bundle_dir.rglob("*"))
                if p.is_file() and p.name != "SKILL.md"
            )
        return Skill(
            name=name, description=description, body=body, source=path, resources=resources
        )

    @staticmethod
    def _load_toml(path: Path) -> Skill | None:
        try:
            values = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            return None
        body = str(values.get("system_prompt", ""))
        return Skill(
            name=str(values.get("name", path.stem)),
            description=str(values.get("description", _first_line(body))),
            body=body,
            source=path,
        )

    # --- queries -----------------------------------------------------------

    def available_skills(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def describe(self) -> list[tuple[str, str]]:
        return [(s.name, s.description) for s in (self._skills[n] for n in sorted(self._skills))]

    # --- activation (pinning) ---------------------------------------------

    def set_active(self, names: Sequence[str]) -> None:
        self.active_skills = set(names)

    def activate(self, name: str) -> bool:
        """Toggle a skill's pin. Returns the resulting pinned state."""
        if name in self.active_skills:
            self.active_skills.discard(name)
            return False
        if name in self._skills:
            self.active_skills.add(name)
            return True
        return False

    # --- disclosure --------------------------------------------------------

    def catalog_text(self) -> str:
        entries = "\n".join(f"- {name}: {desc}" for name, desc in self.describe())
        return (
            "# Available skills\n"
            "You have skills you can load on demand. When a task matches a "
            "skill's purpose, call the `use_skill` tool with its name to load "
            "the full instructions.\n\n" + entries
        )

    def system_additions(self) -> list[str]:
        if not self._skills:
            return []
        additions = [self.catalog_text()]
        for name in sorted(self.active_skills):
            skill = self._skills.get(name)
            if skill and skill.body:
                additions.append(skill.body)
        return additions

    def use(self, name: str) -> str:
        """Full disclosure for one skill (the ``use_skill`` tool result)."""
        skill = self._skills.get(name)
        if skill is None:
            return f"Unknown skill: {name!r}. Available: {', '.join(self.available_skills())}"
        out = [f"# Skill: {skill.name}\n", skill.body]
        if skill.resources:
            out.append("\nBundled files (read with read_file):")
            out.extend(f"  {r}" for r in skill.resources)
        return "\n".join(out)


# Backward-compatible alias used elsewhere in the kernel.
DirectorySkillStore = SkillLibrary
SkillStore = SkillLibrary


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("# ").strip()
        if stripped:
            return stripped
    return ""


def _relpath(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:  # pragma: no cover - different drive on Windows
        return str(path)


def make_skill_tool(library: SkillLibrary):
    """Build the ``use_skill`` tool: load a skill's full body on demand."""
    from agentkernel.tools.base import ToolSpec
    from agentkernel.types import ToolResult

    def handler(arguments: dict) -> ToolResult:
        name = arguments["name"]
        text = library.use(name)
        is_error = text.startswith("Unknown skill:")
        return ToolResult("", text, is_error=is_error)

    return ToolSpec(
        name="use_skill",
        description=(
            "Load the full instructions for one of your available skills by name. "
            "Call this when a task matches a skill listed in 'Available skills'."
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The skill name."}},
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=handler,
        category="skills",
    )
