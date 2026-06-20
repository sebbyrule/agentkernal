"""The `clarify` tool (design §18.4).

Lets the model ask the user a single focused question mid-run instead of guessing,
routed through the same terminal input channel the approver uses. In a
non-interactive run (no stdin), it degrades gracefully: the model is told no one
is available and to proceed with its best judgment, rather than blocking.
"""

from __future__ import annotations

from collections.abc import Callable

from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

_NO_ANSWER = "No user is available to answer; proceed with your best judgment."


def clarify_tool(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> ToolSpec:
    """Build the `clarify` tool over a terminal input/output channel."""

    def clarify(args: dict) -> ToolResult:
        question = (args.get("question") or "").strip()
        if not question:
            return ToolResult("", "clarify requires a `question`.", is_error=True)
        output_fn(f"\n[clarify] {question}")
        try:
            answer = input_fn("> your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            return ToolResult("", _NO_ANSWER)
        if not answer:
            return ToolResult("", "(no answer given) Proceed with your best judgment.")
        return ToolResult("", f"User answered: {answer}")

    return ToolSpec(
        name="clarify",
        description=(
            "Ask the user one focused question when a requirement is genuinely "
            "ambiguous and guessing would risk doing the wrong thing. Use "
            "sparingly — prefer reasonable defaults. Returns the user's answer, or "
            "tells you to proceed if no one is available."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The single, specific question to ask the user.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        handler=clarify,
        category="interaction",
    )
