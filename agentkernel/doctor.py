"""Environment health check (design §18.7).

`agentkernel doctor` runs a set of fast, network-free checks — Python version,
required and optional dependencies, provider credentials, sandbox availability,
and writable paths — and prints a checklist. It exits non-zero if any check
fails, so it doubles as a setup smoke test. Checks are pure functions of the
config + environment, so they're deterministic and offline-testable.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass

from agentkernel.config import Config

# ASCII marks: plain print() to a Windows cp1252 console can't encode unicode glyphs.
_MARK = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}

# provider -> the env var that holds its key (local needs no key).
_PROVIDER_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


@dataclass
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str = ""


def _have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_checks(config: Config, *, env: dict[str, str] | None = None) -> list[Check]:
    """Return the list of health checks for ``config`` and ``env``."""
    env = os.environ if env is None else env
    checks: list[Check] = []

    # Python version.
    v = sys.version_info
    checks.append(
        Check(
            "python",
            "ok" if v >= (3, 11) else "fail",
            f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 11) else " (need >= 3.11)"),
        )
    )

    # Required dependencies.
    for dep in ("jsonschema", "httpx"):
        checks.append(
            Check(f"dependency: {dep}", "ok" if _have_module(dep) else "fail")
        )

    # Provider credentials.
    provider = config.provider
    if provider == "local":
        checks.append(
            Check(
                "provider: local",
                "ok" if config.base_url else "warn",
                config.base_url or "no base_url set - local endpoint won't be reachable",
            )
        )
    else:
        key_env = _PROVIDER_KEY_ENV.get(provider)
        if key_env is None:
            checks.append(Check(f"provider: {provider}", "warn", "unknown provider"))
        else:
            present = bool(env.get(key_env))
            checks.append(
                Check(
                    f"provider: {provider}",
                    "ok" if present else "fail",
                    f"{key_env} is set" if present else f"{key_env} is not set",
                )
            )

    # Sandbox.
    if config.sandbox == "docker":
        have = shutil.which("docker") is not None
        checks.append(
            Check("sandbox: docker", "ok" if have else "fail",
                  "docker CLI found" if have else "docker CLI not on PATH")
        )
    else:
        checks.append(Check("sandbox: local", "ok"))

    # Semantic search needs an embedding key.
    if config.semantic_search:
        present = bool(env.get(config.embedding_api_key_env))
        checks.append(
            Check(
                "semantic search",
                "ok" if present else "warn",
                f"{config.embedding_api_key_env} "
                + ("is set" if present else "is not set - recall will error"),
            )
        )

    # TUI backend on Windows.
    if sys.platform == "win32":
        checks.append(
            Check(
                "tui: curses",
                "ok" if _have_module("curses") else "warn",
                "available" if _have_module("curses")
                else "windows-curses not installed - `agentkernel tui` won't run",
            )
        )

    # Log dir writable.
    try:
        from pathlib import Path

        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        checks.append(Check("log dir writable", "ok", config.log_dir))
    except OSError as exc:
        checks.append(Check("log dir writable", "fail", f"{config.log_dir}: {exc}"))

    return checks


def format_checks(checks: list[Check]) -> str:
    lines = ["agentkernel doctor", ""]
    for c in checks:
        detail = f" - {c.detail}" if c.detail else ""
        lines.append(f"  {_MARK[c.status]} {c.name}{detail}")
    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")
    lines.append("")
    lines.append(f"{len(checks)} checks: {fails} failed, {warns} warnings.")
    return "\n".join(lines)


def has_failures(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)
