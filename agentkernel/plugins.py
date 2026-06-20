"""Plugin tool discovery (design §18.7).

Loads user-authored tools from a ``plugins/`` directory and registers them
exactly like builtins — the same seam MCP uses (§13). A plugin is a ``.py`` file
that exposes either a ``tools()`` callable returning ``list[ToolSpec]`` (optionally
taking ``working_dir``) or a module-level ``TOOLS`` list.

SECURITY: importing a plugin executes its module code. Discovery is opt-in
(``enable_plugins``), and only files you place in ``plugins_dir`` are loaded.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from agentkernel.tools.base import ToolSpec


def _import_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"agentkernel_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_tools_fn(fn, working_dir: str) -> list[ToolSpec]:
    """Call a plugin's ``tools`` entrypoint, passing working_dir if it accepts it."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    result = fn(working_dir) if params else fn()
    return [s for s in (result or []) if isinstance(s, ToolSpec)]


def _extract(module, working_dir: str) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    fn = getattr(module, "tools", None)
    if callable(fn):
        specs.extend(_call_tools_fn(fn, working_dir))
    table = getattr(module, "TOOLS", None)
    if isinstance(table, (list, tuple)):
        specs.extend(s for s in table if isinstance(s, ToolSpec))
    return specs


def load_plugin_tools(
    plugins_dir: str | Path,
    *,
    working_dir: str = ".",
    on_error=None,
) -> list[ToolSpec]:
    """Discover and return tools from every ``*.py`` in ``plugins_dir``.

    A module that fails to import is skipped; ``on_error(path, exc)`` is called if
    provided so the caller can warn. Files starting with ``_`` are ignored.
    """
    directory = Path(plugins_dir)
    specs: list[ToolSpec] = []
    if not directory.is_dir():
        return specs
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _import_file(path)
            if module is not None:
                specs.extend(_extract(module, working_dir))
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not crash startup
            if on_error is not None:
                on_error(path, exc)
    return specs
