"""A minimal stdio MCP server for tests (pure stdlib, runs as a subprocess).

Implements just enough of the protocol to exercise MCPClient: initialize,
tools/list (an ``echo`` read-only tool and a ``boom`` tool that errors), and
tools/call. No network — communication is over stdin/stdout only.
"""

import json
import sys


def _send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


_TOOLS = [
    {
        "name": "echo",
        "description": "Echo text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "boom",
        "description": "Always returns an error result.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp", "version": "0.1"},
                },
            })
        elif method == "notifications/initialized":
            pass  # notification: no response
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": _TOOLS}})
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "echo":
                _send({
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "content": [{"type": "text", "text": "echo: " + str(args.get("text", ""))}],
                        "isError": False,
                    },
                })
            elif name == "boom":
                _send({
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "content": [{"type": "text", "text": "it broke"}],
                        "isError": True,
                    },
                })
            else:
                _send({
                    "jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": "unknown tool"},
                })
        elif mid is not None:
            _send({
                "jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": "method not found"},
            })


if __name__ == "__main__":
    main()
