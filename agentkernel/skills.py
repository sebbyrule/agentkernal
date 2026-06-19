"""Skills / AGENTS.md loading seam (design §13, Phase 4).

A ``Skill`` is a reusable system-prompt fragment stored in a markdown or TOML
file. A ``SkillStore`` discovers skills in a directory and returns their
system-prompt text as context additions. This lets agent behavior be extended
without changing the kernel loop.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, Set, runtime_checkable


@dataclass(frozen=True)
class Skill:
    """A skill is a named system-prompt fragment backed by a file.

    The name is taken from frontmatter or from the file stem. The
    ``system_prompt`` is the text the model sees.
    """

    name: str
    system_prompt: str
    source: Path | None = None


@runtime_checkable
class ContextSource(Protocol):
    """Anything that contributes extra system-prompt text per run."""

    def system_additions(self) -> list[str]: ...


class DirectorySkillStore:
    """Loads skills from ``*.md`` and ``*.toml`` files in a directory.

    Markdown files may contain an optional TOML frontmatter block delimited by
    ``---``. If no frontmatter is present, the whole file body becomes the
    system prompt. TOML files are treated as frontmatter directly.

    Only skills whose names are in ``active_skills`` are returned by
    ``system_additions``. When ``active_skills`` is empty, no additions are
    returned.
    """

    def __init__(
        self,
        directory: str | Path,
        active_skills: Sequence[str] | None = None,
    ) -> None:
        self.directory = Path(directory) if directory else Path("skills")
        self.active_skills: Set[str] = set(active_skills or [])
        self._skills: dict[str, Skill] = {}
        if self.directory.exists():
            self._load()

    def _load(self) -> None:
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.iterdir()):
            if path.suffix.lower() == ".toml":
                skill = self._load_toml(path)
            elif path.suffix.lower() == ".md":
                skill = self._load_markdown(path)
            else:
                continue
            if skill is not None:
                self._skills[skill.name] = skill

    @staticmethod
    def _load_toml(path: Path) -> Skill | None:
        try:
            values = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return Skill(
            name=values.get("name", path.stem),
            system_prompt=values.get("system_prompt", ""),
            source=path,
        )

    @staticmethod
    def _load_markdown(path: Path) -> Skill | None:
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            try:
                _, front, body = text.split("---", 2)
                values = tomllib.loads(front.strip())
            except Exception:
                values = {}
                body = text
            system_prompt = values.get("system_prompt", body.strip())
            name = values.get("name", path.stem)
        else:
            name = path.stem
            system_prompt = text.strip()
        return Skill(name=name, system_prompt=system_prompt, source=path)

    def available_skills(self) -> list[str]:
        """Return all discovered skill names in stable order."""
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def set_active(self, names: Sequence[str]) -> None:
        """Replace the active skill set."""
        self.active_skills = set(names)

    def activate(self, name: str) -> bool:
        """Toggle a skill on or off. Returns final active state."""
        if name in self.active_skills:
            self.active_skills.discard(name)
            return False
        if name in self._skills:
            self.active_skills.add(name)
            return True
        return False

    def system_additions(self) -> list[str]:
        """Return system prompts for currently active skills."""
        if not self.active_skills:
            return []
        additions: list[str] = []
        for name in sorted(self.active_skills):
            skill = self._skills.get(name)
            if skill and skill.system_prompt:
                additions.append(skill.system_prompt)
        return additions


# Backward-compatible alias
SkillStore = DirectorySkillStore
