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
    approval_policy: str = "always_ask"  # always_ask | auto_allow | deny_mutations
    approval_allowlist: list[str] = field(default_factory=list)  # patterns that skip the gate
    working_dir: str = "."
    summarizer_model: str | None = None  # cheap model for compaction; None -> structural
    log_dir: str = ".agentkernel/traces"
    max_cost_usd: float | None = None  # stop if cumulative cost exceeds this
    max_input_tokens_per_run: int | None = None  # stop if input tokens exceed this
    profile: str | None = None  # active profile name (Phase 5)
    profile_dir: str = "profiles"
    memory_store: str | None = None  # "file" | "memory" | None (Phase 3)
    memory_dir: str | None = None
    skills_dir: str = "skills"  # Phase 4
    skills: list[str] = field(default_factory=list)  # active skill names (Phase 4)
    enable_graph: bool = False  # register graph_add/graph_query tools (Phase 6)
    graph_path: str = ".agentkernel/graph.jsonl"  # Phase 6
    improvements_dir: str = ".agentkernel/improvements"  # Phase 7

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
