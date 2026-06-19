"""MCP client (Phase 2, design §13).

An MCP client connects to a server, discovers its tools, and registers each as
an ordinary ``ToolSpec`` whose handler issues an MCP ``tools/call``. This is the
test of whether §6 is right: **no loop or registry change is required** — an
MCP-backed tool and a native builtin register identically.

Hand-written over JSON-RPC 2.0 stdio, consistent with the kernel's
dependency-light, no-frameworks stance (no ``mcp`` SDK dependency).
"""

from agentkernel.mcp.client import MCPClient, MCPError
from agentkernel.mcp.config import MCPServerConfig, load_mcp_servers
from agentkernel.mcp.tools import mcp_tool_specs, register_mcp_servers

__all__ = [
    "MCPClient",
    "MCPError",
    "MCPServerConfig",
    "load_mcp_servers",
    "mcp_tool_specs",
    "register_mcp_servers",
]
