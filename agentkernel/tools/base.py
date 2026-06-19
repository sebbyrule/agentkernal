"""Tool definitions and the registry (design §6).

The registry is agnostic about a tool's origin: a native builtin and (later) an
MCP-backed tool register identically. This is the Phase-2 seam — nothing here is
special-cased per origin.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jsonschema

from agentkernel.types import ToolCall, ToolResult


@dataclass
class ToolSpec:
    """A registered tool. ``parameters`` is a JSON Schema (draft 2020-12) object.

    Flags drive the approval/sandbox gate (design §10): any of ``requires_approval``,
    ``mutates``, or ``runs_code`` causes the loop to consult the ``Approver`` before
    executing; ``runs_code`` additionally routes execution through the ``Sandbox``.
    """

    name: str
    description: str  # model-facing; write it like a prompt
    parameters: dict[str, Any]  # JSON Schema (draft 2020-12) object
    handler: Callable[[dict], ToolResult]
    requires_approval: bool = False
    mutates: bool = False
    runs_code: bool = False
    category: str = "general"

    @property
    def gated(self) -> bool:
        """True if this tool must pass the approver before executing."""
        return self.requires_approval or self.mutates or self.runs_code


class ToolRegistry:
    """Holds tool specs and dispatches calls. See design §6.2.

    Spec ordering is insertion order and is never re-sorted — the spec list is
    part of the cacheable prefix (design §9.3), so reordering it between turns
    would destroy prompt-cache hit-rate.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name!r}")
        self._specs[spec.name] = spec

    def spec(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def specs(self) -> list[ToolSpec]:
        """All specs in stable registration order (for the provider prefix)."""
        return list(self._specs.values())

    def validate(self, call: ToolCall) -> Optional[str]:
        """Validate ``call.arguments`` against the tool's schema.

        Returns an error string on failure (unknown tool or schema violation) or
        ``None`` if the call is valid. The loop turns a non-None result into a
        ``ToolResult(is_error=True)`` *instead of* executing, so the model can
        correct itself.
        """
        spec = self._specs.get(call.name)
        if spec is None:
            return f"Unknown tool: {call.name!r}"
        try:
            jsonschema.validate(call.arguments, spec.parameters)
        except jsonschema.ValidationError as exc:
            return f"Invalid arguments for {call.name!r}: {exc.message}"
        return None

    def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch to the handler. A handler exception becomes an error result.

        Handlers receive only ``call.arguments`` and so cannot know the call id;
        the registry stamps ``call_id`` onto the returned result here, which keeps
        the §8 pairing contract the registry's responsibility, not the handler's.
        """
        spec = self._specs.get(call.name)
        if spec is None:  # pragma: no cover - validate() runs first in the loop
            return ToolResult(call.id, f"Unknown tool: {call.name!r}", is_error=True)
        try:
            result = spec.handler(call.arguments)
        except Exception as exc:  # noqa: BLE001 - errors become results, not raises
            summary = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
            return ToolResult(call.id, summary, is_error=True)
        result.call_id = call.id
        return result
