"""Template for a custom tool module (agentkernel conventions).

A tool is a ToolSpec: a name, a model-facing description (write it like a prompt),
a JSON-Schema parameter object, and a handler. Build tools with a factory function
so any dependencies (a working dir, a client, a config value) are bound once and
the handler stays a pure function of its arguments — never reaching for globals.

Conventions to follow:
- Return a ToolResult; NEVER raise. Turn failures into ToolResult(is_error=True).
  (The registry also catches stray exceptions, but handling them yourself gives a
  clearer message to the model.)
- Set additionalProperties: False and list required fields in the schema.
- Flag mutations: pass requires_approval/mutates/runs_code so the loop gates them.

Register these with your runtime, e.g. in build_runtime or a plugin loader:

    from your_module import my_tools
    for spec in my_tools(working_dir="."):
        registry.register(spec)
"""

from __future__ import annotations

from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult


def my_tools(working_dir: str = ".") -> list[ToolSpec]:
    """Build the toolset, binding any dependencies here."""

    def greet(args: dict) -> ToolResult:
        # args has already been validated against the schema below.
        name = args["name"]
        excited = bool(args.get("excited", False))
        if not name.strip():
            return ToolResult("", "name must not be empty", is_error=True)
        greeting = f"Hello, {name}{'!' if excited else '.'}"
        return ToolResult("", greeting)

    return [
        ToolSpec(
            name="greet",
            description=(
                "Return a friendly greeting for the given name. Use when the user "
                "asks to greet or welcome someone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Who to greet."},
                    "excited": {
                        "type": "boolean",
                        "description": "Use an exclamation mark.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=greet,
            category="custom",
            # For a tool that writes files / runs commands, gate it:
            #   mutates=True, requires_approval=True   (or runs_code=True)
        ),
    ]
