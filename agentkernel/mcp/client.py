"""A minimal MCP client over JSON-RPC 2.0 stdio (Phase 2).

Messages are newline-delimited JSON objects exchanged with the server
subprocess's stdin/stdout. A background reader thread drains stdout into a
queue so requests can wait with a timeout (a misbehaving server must never hang
the agent loop).

This client supports exactly what tool discovery and invocation need:
``initialize`` → ``notifications/initialized`` → ``tools/list`` / ``tools/call``.
Server-initiated requests (sampling, roots) are ignored in v1.

When ``log_dir`` is provided, each server's stderr is captured to a file
``<log_dir>/<server-name>.log`` for post-mortem debugging.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from agentkernel import __version__
from agentkernel.mcp.config import MCPServerConfig

PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    """An MCP transport or protocol fault (connection, timeout, JSON-RPC error)."""


class MCPClient:
    """Connects to one MCP server over stdio and issues JSON-RPC requests."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        timeout: float = 30.0,
        log_dir: str | None = None,
    ) -> None:
        self._config = config
        self._timeout = timeout
        self._log_dir = log_dir
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._incoming: queue.Queue[dict[str, Any]] = queue.Queue()
        self._next_id = 0
        self.server_info: dict[str, Any] | None = None
        self.stderr_log_path: str | None = None

    # --- lifecycle ---------------------------------------------------------

    def connect(self) -> MCPClient:
        if self._proc is not None:
            raise MCPError(f"MCP client {self._config.name!r} is already connected")
        env = {**os.environ, **(self._config.env or {})}

        stderr_target: int = subprocess.DEVNULL
        if self._log_dir:
            log_path = Path(self._log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.stderr_log_path = str(log_path / f"{self._config.name}.log")
            stderr_target = subprocess.PIPE

        try:
            self._proc = subprocess.Popen(
                [self._config.command, *self._config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_target,
                text=True,
                bufsize=1,
                env=env,
                cwd=self._config.cwd,
            )
        except (OSError, ValueError) as exc:
            raise MCPError(f"failed to launch MCP server {self._config.name!r}: {exc}")

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        if self.stderr_log_path is not None and self._proc.stderr is not None:
            self._stderr_reader = threading.Thread(
                target=self._read_loop_stderr,
                args=(self.stderr_log_path, self._proc.stderr),
                daemon=True,
            )
            self._stderr_reader.start()

        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agentkernel", "version": __version__},
            },
        )
        self.server_info = result.get("serverInfo")
        self._notify("notifications/initialized", {})
        return self

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best effort shutdown
            proc.kill()
        # Give the stderr logger a moment to drain and flush.
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=1.0)

    # --- MCP operations ----------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        return self._request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    # --- JSON-RPC plumbing -------------------------------------------------

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._incoming.put(json.loads(line))
            except json.JSONDecodeError:
                continue  # ignore non-JSON noise on stdout

    def _read_loop_stderr(self, log_path: str, stream) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            for line in stream:
                fh.write(line)
                fh.flush()

    def _send(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.poll() is not None:
            raise MCPError(f"MCP server {self._config.name!r} is not running")
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"timed out waiting for response to {method!r}")
            try:
                msg = self._incoming.get(timeout=remaining)
            except queue.Empty:
                raise MCPError(f"timed out waiting for response to {method!r}")
            # Skip server-initiated requests/notifications and stale responses.
            if msg.get("method") is not None or msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg["error"]
                raise MCPError(
                    f"{method!r} failed: {err.get('message')} (code {err.get('code')})"
                )
            return msg.get("result", {})
