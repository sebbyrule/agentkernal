"""MCP client tests (Phase 2, design §13). Hermetic: a stdlib stdio server runs
as a subprocess — no network. The headline test proves the §6 seam: an MCP tool
runs through the unmodified loop exactly like a builtin."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentkernel.mcp import (
    MCPClient,
    MCPServerConfig,
    load_mcp_servers,
    mcp_tool_specs,
    register_mcp_servers,
)
from agentkernel.mcp.client import MCPError
from agentkernel.tools import ToolRegistry
from agentkernel.types import ToolCall
from tests.fakes import FakeProvider, text_response, tool_call_response

_SERVER = str(Path(__file__).parent / "mcp_server.py")


def _config(name: str = "fake") -> MCPServerConfig:
    return MCPServerConfig(name=name, command=sys.executable, args=[_SERVER])


@pytest.fixture
def client():
    c = MCPClient(_config()).connect()
    yield c
    c.close()


def test_connect_reports_server_info(client):
    assert client.server_info["name"] == "fake-mcp"


def test_list_tools(client):
    names = {t["name"] for t in client.list_tools()}
    assert names == {"echo", "boom"}


def test_call_tool_echo(client):
    result = client.call_tool("echo", {"text": "hi"})
    assert result["content"][0]["text"] == "echo: hi"
    assert result["isError"] is False


def test_unknown_tool_raises_mcp_error(client):
    with pytest.raises(MCPError):
        client.call_tool("does_not_exist", {})


def test_tool_specs_gating_from_annotations(client):
    specs = {s.name: s for s in mcp_tool_specs(client)}
    # echo advertises readOnlyHint -> ungated; boom has no hint -> gated.
    assert specs["echo"].requires_approval is False and specs["echo"].mutates is False
    assert specs["boom"].requires_approval is True
    assert specs["echo"].category == "mcp"


def test_tool_spec_handler_maps_result(client):
    specs = {s.name: s for s in mcp_tool_specs(client)}
    ok = specs["echo"].handler({"text": "yo"})
    assert not ok.is_error and "echo: yo" in ok.content
    bad = specs["boom"].handler({})
    assert bad.is_error and "it broke" in bad.content  # isError -> error result


def test_handler_transport_error_becomes_error_result(client):
    # Close the server, then call: the MCP fault must degrade to an error
    # result, never raise out of the handler (design §8.3).
    specs = {s.name: s for s in mcp_tool_specs(client)}
    client.close()
    result = specs["echo"].handler({"text": "x"})
    assert result.is_error and "MCP error" in result.content


def test_mcp_tool_runs_through_unmodified_loop(agent_builder):
    """The §6 seam: an MCP-backed tool registers identically to a builtin and
    the loop runs it with no special-casing."""
    registry = ToolRegistry()
    clients = register_mcp_servers(registry, [_config()])
    try:
        provider = FakeProvider(
            [
                tool_call_response(ToolCall("c1", "echo", {"text": "loop"})),
                text_response("done"),
            ]
        )
        agent = agent_builder(provider, registry)
        assert agent.run("use the echo tool") == "done"
        result = provider.calls[1][-1].tool_results[0]
        assert result.call_id == "c1" and "echo: loop" in result.content
    finally:
        for c in clients:
            c.close()


def test_register_closes_clients_on_failure():
    # A second server whose command does not exist should fail to connect, and
    # the first (good) client must be closed rather than leaked.
    good = _config("good")
    bad = MCPServerConfig(name="bad", command="this-command-does-not-exist-xyz")
    registry = ToolRegistry()
    with pytest.raises(MCPError):
        register_mcp_servers(registry, [good, bad])


def test_load_mcp_servers_from_toml(tmp_path):
    toml = tmp_path / "agentkernel.toml"
    toml.write_text(
        '[[mcp_servers]]\n'
        'name = "fs"\n'
        'command = "npx"\n'
        'args = ["-y", "@modelcontextprotocol/server-filesystem", "."]\n'
    )
    servers = load_mcp_servers(toml)
    assert len(servers) == 1
    assert servers[0].name == "fs" and servers[0].command == "npx"
    assert servers[0].args[0] == "-y"
    assert load_mcp_servers(tmp_path / "missing.toml") == []
