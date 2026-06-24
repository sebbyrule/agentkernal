"""Location policy for running the agent outside its own project folder.

Two roots:

* **Agent home** — the user-global "brain & library" (memory notebook, knowledge
  graph, skills, profiles, improvements, scheduled jobs). Shared across every
  project. ``$AGENTKERNEL_HOME`` overrides the default ``~/.agentkernel``.
* **Project root** — the directory the agent operates on, found by walking up
  from the target directory for a marker (``agentkernel.toml`` / ``.agentkernel``
  / ``.git``). Project-local state (session traces, kanban board, checkpoints)
  lives under ``<project>/.agentkernel``.

``anchor_path`` resolves a configured path against the right root, leaving
absolute paths untouched, so "global brain, project sessions" works no matter
where you launch ``agentkernel``.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV = "AGENTKERNEL_HOME"
DEFAULT_HOME_DIRNAME = ".agentkernel"
_CONFIG_MARKERS = ("agentkernel.toml", ".agentkernel")
_FALLBACK_MARKERS = (".git",)


def agent_home(env: dict[str, str] | None = None) -> Path:
    """The user-global agent home (``$AGENTKERNEL_HOME`` or ``~/.agentkernel``)."""
    source = os.environ if env is None else env
    override = source.get(HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_HOME_DIRNAME


def find_project_root(start: str | os.PathLike[str] = ".") -> Path:
    """Walk up from ``start`` to the nearest project root.

    Prefers a directory containing ``agentkernel.toml`` or ``.agentkernel``; falls
    back to a ``.git`` root; otherwise returns ``start`` itself.
    """
    here = Path(start).expanduser().resolve()
    if here.is_file():
        here = here.parent
    candidates = [here, *here.parents]
    for directory in candidates:
        if any((directory / marker).exists() for marker in _CONFIG_MARKERS):
            return directory
    for directory in candidates:
        if any((directory / marker).exists() for marker in _FALLBACK_MARKERS):
            return directory
    return here


def find_project_config(project_root: Path) -> Path | None:
    """The project's ``agentkernel.toml`` if present."""
    candidate = project_root / "agentkernel.toml"
    return candidate if candidate.is_file() else None


def global_config_path(home: Path | None = None) -> Path:
    """The user-global config file (``<home>/config.toml``)."""
    return (home or agent_home()) / "config.toml"


def anchor_path(value: str | os.PathLike[str], *, base: Path) -> str:
    """Resolve ``value`` against ``base``; absolute and ``~`` paths are honored."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())
