"""Profile seam (Phase 5, design §13).

A ``Profile`` parameterizes a run: system prompt, tool filter, optional model
override, and an optional rubric for evaluation. The kernel honors
``system_prompt`` and ``tool_filter``; ``model_override`` and ``rubric`` are
extension points for later phases / the CLI.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Profile:
    """A parameterization of one run (design §13)."""

    name: str
    system_prompt: str | None = None
    tool_filter: list[str] | None = None
    model_override: str | None = None
    rubric: str | None = None
    reasoning: str | None = None  # "low" | "medium" | "high" (§18.5); ignored where unsupported


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept snake_case keys plus the older-style aliases in TOML files."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        out[key] = value
    return out


def load_profile(name: str, *, search_dirs: Sequence[Path] | None = None) -> Profile | None:
    """Load a profile by name from the first matching TOML file.

    Searches, in order:
      - ``<search_dir>/<name>.toml`` for each directory in ``search_dirs``
      - ``profiles/<name>.toml``
      - ``.agentkernel/profiles/<name>.toml``

    Returns ``None`` if no file is found.
    """
    if search_dirs is None:
        search_dirs = []
    search_dirs = [
        *search_dirs,
        Path("profiles"),
        Path(".agentkernel/profiles"),
    ]

    for directory in search_dirs:
        path = Path(directory) / f"{name}.toml"
        if path.is_file():
            with path.open("rb") as fh:
                data = tomllib.load(fh)
            values = _normalize(data)
            return Profile(
                name=name,
                system_prompt=values.get("system_prompt"),
                tool_filter=values.get("tool_filter"),
                model_override=values.get("model_override"),
                rubric=values.get("rubric"),
                reasoning=values.get("reasoning"),
            )
    return None
