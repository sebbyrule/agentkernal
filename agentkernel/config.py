"""Configuration (design §11).

Precedence: explicit constructor args > environment (``AGENTKERNEL_*``) >
config file (``agentkernel.toml``) > defaults. API keys come **only** from the
environment (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, …) and are never read
from, or written to, the config file or traces.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "AGENTKERNEL_"
DEFAULT_CONFIG_FILE = "agentkernel.toml"


@dataclass
class Config:
    provider: str = "anthropic"  # "anthropic" | "openai" | "local"
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None  # for local/OpenAI-compatible endpoints
    max_output_tokens: int = 4096
    output_reserve: int = 8192  # budget headroom for the reply
    max_iterations: int = 25
    keep_recent_turns: int = 6
    max_tool_result_tokens: int = 4096
    approval_policy: str = "always_ask"  # always_ask | auto_allow | deny_mutations | smart
    approval_judge_model: str | None = None  # model for `smart` risk judging (§18.1)
    redact_tool_output: bool = True  # scrub secret-looking strings from tool results (§18.1)
    checkpoints: bool = False  # back up files before edits; expose a `rollback` tool (§18.1)
    enable_todo: bool = False  # register the in-session `todo` planning tool (§18.4)
    enable_clarify: bool = False  # register the `clarify` ask-the-user tool (§18.4)
    enable_plugins: bool = False  # auto-load tools from plugins_dir (§18.7)
    plugins_dir: str = "plugins"  # directory of plugin tool modules
    cron_path: str = ".agentkernel/cron.json"  # scheduled-job store (§18.2)
    enable_kanban: bool = False  # register the `kanban` work-board tool (§18.3)
    kanban_path: str = ".agentkernel/kanban.json"  # shared work-board store
    approval_allowlist: list[str] = field(default_factory=list)  # patterns that skip the gate
    plan_mode: bool = False              # batch-approve the whole tool plan at once
    sandbox: str = "local"  # "local" | "docker" (design §10.3)
    sandbox_image: str = "python:3.12-slim"  # image for the docker sandbox
    sandbox_network: str = "none"  # docker container network ("none" | "bridge" | …)
    working_dir: str = "."
    summarizer_model: str | None = None  # cheap model for compaction; None -> structural
    log_dir: str = ".agentkernel/traces"
    mcp_log_dir: str = ".agentkernel/mcp-logs"  # per-server stderr logs
    max_cost_usd: float | None = None  # stop if cumulative cost exceeds this
    max_input_tokens_per_run: int | None = None  # stop if input tokens exceed this
    profile: str | None = None  # active profile name (Phase 5)
    profile_dir: str = "profiles"
    memory_store: str | None = None  # "file" | "sqlite" | "memory" | None (Phase 3)
    memory_dir: str | None = None
    enable_memory_tools: bool = False  # register remember/recall/forget tools (Phase 3)
    memory_notes_path: str = ".agentkernel/memory/notes.jsonl"  # the notebook file
    memory_auto_context: bool = False  # auto-inject recalled notes into user message
    memory_auto_context_limit: int = 3  # max notes per auto-recall
    memory_store_budget: int | None = None  # max tokens to persist per session
    memory_curator_model: str | None = None  # cheap model for extract/consolidate; None -> summarizer_model/model
    semantic_search: bool = False  # rank note recall with dense embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None  # optional truncation (OpenAI only)
    embedding_base_url: str | None = None  # OpenAI-compatible endpoint
    embedding_api_key_env: str = "OPENAI_API_KEY"  # env var holding API key
    skills_dir: str = "skills"  # Phase 4
    skills: list[str] = field(default_factory=list)  # active skill names (Phase 4)
    enable_graph: bool = False  # register graph_add/graph_query tools (Phase 6)
    graph_path: str = ".agentkernel/graph.jsonl"  # Phase 6
    improvements_dir: str = ".agentkernel/improvements"  # Phase 7
    enable_spawn: bool = False  # register the sub-agent `spawn` tool (design §13)
    spawn_max_depth: int = 2  # recursion limit for nested spawn
    judge_model: str | None = None  # model used to score evals; None -> `model`
    eval_threshold: float = 0.6  # pass/fail score cutoff for evals
    eval_rubric: str | None = None  # default rubric for eval runs
    semantic_search_lsh_bits: int | None = None  # LSH index bits for large notebooks

    @classmethod
    def load(
        cls,
        config_file: str | os.PathLike[str] = DEFAULT_CONFIG_FILE,
        *,
        env: dict[str, str] | None = None,
        **overrides: Any,
    ) -> Config:
        """Build a Config from defaults < file < environment < explicit overrides."""
        env = os.environ if env is None else env
        values: dict[str, Any] = {}

        path = Path(config_file)
        if path.is_file():
            with path.open("rb") as fh:
                file_data = tomllib.load(fh)
            values.update({k: v for k, v in file_data.items() if k in _FIELD_TYPES})

        for name, typ in _FIELD_TYPES.items():
            raw = env.get(ENV_PREFIX + name.upper())
            if raw is not None:
                values[name] = _coerce(raw, typ)

        values.update({k: v for k, v in overrides.items() if k in _FIELD_TYPES})
        return cls(**values)


_FIELD_TYPES: dict[str, Any] = {f.name: f.type for f in fields(Config)}

# State that is the agent's user-global "brain & library" vs per-project.
# Defaults anchor to the appropriate root (see paths.py); a value customized in
# project config anchors to the project instead, so per-project overrides win.
_GLOBAL_PATH_FIELDS = (
    "memory_notes_path", "graph_path", "skills_dir", "profile_dir",
    "improvements_dir", "cron_path",
)
_PROJECT_PATH_FIELDS = (
    "log_dir", "mcp_log_dir", "kanban_path", "plugins_dir", "memory_dir",
)


def _read_config_fields(path: Path) -> dict[str, Any]:
    """Read recognized fields from a TOML config file."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return {k: v for k, v in data.items() if k in _FIELD_TYPES}


def resolve_config(
    config_arg: str | os.PathLike[str] | None = None,
    *,
    cwd: str | os.PathLike[str] = ".",
    env: dict[str, str] | None = None,
    **overrides: Any,
) -> tuple[Config, Path | None]:
    """Resolve config for running anywhere (global brain, project sessions).

    Precedence: explicit ``config_arg`` (single file) **or** layered
    ``<home>/config.toml`` < ``<project>/agentkernel.toml``, then env, then
    ``overrides``. State paths are anchored to the agent home (global fields) or
    the project root (project fields); ``working_dir`` defaults to the project
    root. Returns ``(config, project_config_path)``.
    """
    from agentkernel.paths import (
        agent_home,
        anchor_path,
        find_project_config,
        find_project_root,
        global_config_path,
    )

    env = os.environ if env is None else env
    home = agent_home(env)
    project_root = find_project_root(cwd)
    values: dict[str, Any] = {}
    project_config_path: Path | None = None

    if config_arg is not None:
        explicit = Path(config_arg)
        if explicit.is_file():
            values.update(_read_config_fields(explicit))
            project_config_path = explicit
    else:
        gpath = global_config_path(home)
        if gpath.is_file():
            values.update(_read_config_fields(gpath))
        ppath = find_project_config(project_root)
        if ppath is not None:
            values.update(_read_config_fields(ppath))
            project_config_path = ppath

    for name, typ in _FIELD_TYPES.items():
        raw = env.get(ENV_PREFIX + name.upper())
        if raw is not None:
            values[name] = _coerce(raw, typ)
    values.update({k: v for k, v in overrides.items() if k in _FIELD_TYPES})

    config = Config(**values)
    defaults = Config()

    if config.working_dir in (".", "", None):
        config.working_dir = str(project_root)
    else:
        config.working_dir = anchor_path(config.working_dir, base=project_root)

    for name in _GLOBAL_PATH_FIELDS:
        value = getattr(config, name)
        if value is None:
            continue
        base = home if value == getattr(defaults, name) else project_root
        setattr(config, name, anchor_path(value, base=base))

    for name in _PROJECT_PATH_FIELDS:
        value = getattr(config, name)
        if value is None:
            if name == "memory_dir":
                config.memory_dir = anchor_path(".agentkernel/memory", base=project_root)
            continue
        setattr(config, name, anchor_path(value, base=project_root))

    return config, project_config_path


def _coerce(raw: str, typ: Any) -> Any:
    """Coerce an environment string to the field's declared type."""
    # Field types are stringified annotations under ``from __future__ import``.
    if typ == "bool":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if typ == "int":
        return int(raw)
    if typ == "float":
        return float(raw)
    if typ == "list[str]":
        return [s.strip() for s in raw.split(",") if s.strip()]
    if typ in ("str | None", "int | None", "float | None"):
        if raw == "":
            return None
        if "int" in typ:
            return int(raw)
        if "float" in typ:
            return float(raw)
        return raw
    return raw
