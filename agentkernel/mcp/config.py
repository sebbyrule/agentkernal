"""MCP server declarations (Phase 2).

Servers are declared in the config TOML as an array of tables:

    [[mcp_servers]]
    name = "filesystem"
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
    timeout = 60

    [[mcp_servers]]
    name = "git"
    command = "uvx"
    args = ["mcp-server-git"]
    env = { GIT_AUTHOR_NAME = "agent" }

This loader is kept separate from ``Config`` because the scalar config loader
(``config.py``) only handles flat values, while server definitions are nested.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MCPServerConfig:
    """How to launch and connect to one MCP server over stdio."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    timeout: float | None = None  # seconds; falls back to MCPClient default (30)


def load_mcp_servers(config_file: str | Path) -> list[MCPServerConfig]:
    """Read ``[[mcp_servers]]`` from the config TOML. Missing file → no servers."""
    path = Path(config_file)
    if not path.is_file():
        return []
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    servers: list[MCPServerConfig] = []
    for entry in data.get("mcp_servers", []):
        servers.append(
            MCPServerConfig(
                name=entry["name"],
                command=entry["command"],
                args=list(entry.get("args", [])),
                env=entry.get("env"),
                cwd=entry.get("cwd"),
                timeout=entry.get("timeout"),
            )
        )
    return servers
