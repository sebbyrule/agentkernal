"""Adapt MCP tools into the kernel's ``ToolSpec`` (Phase 2, design §13).

Each discovered MCP tool becomes a ``ToolSpec`` whose handler issues a
``tools/call`` and maps the MCP result back to a canonical ``ToolResult``. The
registry and loop are untouched — this module is the entire MCP seam.

Gating: MCP tools can do anything, so they default to ``requires_approval``.
A tool that advertises ``annotations.readOnlyHint`` is treated as read-only and
left ungated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentkernel.mcp.client import MCPClient, MCPError
from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

if TYPE_CHECKING:
    from agentkernel.mcp.config import MCPServerConfig
    from agentkernel.tools import ToolRegistry


def _result_to_tool_result(result: dict[str, Any]) -> ToolResult:
    """Flatten an MCP call result's content blocks into a ToolResult."""
    text_parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if block.get("type") == "text"
    ]
    content = "\n".join(p for p in text_parts if p) or "(no text content)"
    return ToolResult(
        "",
        content,
        is_error=bool(result.get("isError", False)),
        data={"mcp_result": result},
    )


def _make_handler(client: MCPClient, name: str):
    def handler(arguments: dict[str, Any]) -> ToolResult:
        try:
            result = client.call_tool(name, arguments)
        except MCPError as exc:
            # Transport/protocol faults become error results, not raises, so the
            # model can recover (design §8.3).
            return ToolResult("", f"MCP error: {exc}", is_error=True)
        return _result_to_tool_result(result)

    return handler


def mcp_tool_specs(client: MCPClient) -> list[ToolSpec]:
    """Discover the client's tools and wrap each as a ToolSpec."""
    specs: list[ToolSpec] = []
    for tool in client.list_tools():
        annotations = tool.get("annotations") or {}
        read_only = bool(annotations.get("readOnlyHint", False))
        specs.append(
            ToolSpec(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get("inputSchema") or {"type": "object"},
                handler=_make_handler(client, tool["name"]),
                requires_approval=not read_only,
                mutates=not read_only,
                category="mcp",
            )
        )
    return specs


def register_mcp_servers(
    registry: ToolRegistry, servers: list[MCPServerConfig]
) -> list[MCPClient]:
    """Connect each server and register its tools. Returns the open clients so
    the caller can close them. On any failure, already-opened clients are
    closed before the error propagates."""
    clients: list[MCPClient] = []
    try:
        for server in servers:
            client = MCPClient(server, timeout=server.timeout or 30.0).connect()
            clients.append(client)
            for spec in mcp_tool_specs(client):
                registry.register(spec)
    except Exception:
        for client in clients:
            client.close()
        raise
    return clients
