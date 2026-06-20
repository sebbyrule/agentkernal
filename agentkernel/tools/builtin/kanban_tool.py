"""The `kanban` tool (design §18.3).

Exposes a shared work-queue board to the model so a long mission — or several
cooperating sub-agents — can file, claim, and complete tasks. The board is the
durable JSON store in ``kanban.py``; the tool is a thin, single-entry dispatcher
over it, bound to one board (and an optional worker identity) by the factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentkernel.kanban import render_task
from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

if TYPE_CHECKING:
    from agentkernel.kanban import Board


def _board_view(board: Board) -> str:
    tasks = board.list()
    if not tasks:
        return "(board is empty)"
    open_n = sum(1 for t in tasks if t.status in ("todo", "in_progress"))
    lines = [f"Board ({open_n} open / {len(tasks)} total):"]
    lines += [f"  {render_task(t)}" for t in tasks]
    return "\n".join(lines)


def kanban_tool(board: Board, *, worker: str = "agent") -> ToolSpec:
    """Build the `kanban` tool over ``board``. ``worker`` labels who claims tasks."""

    def kanban(args: dict) -> ToolResult:
        action = args["action"]

        if action == "list":
            return ToolResult("", _board_view(board))
        if action == "add":
            title = (args.get("title") or "").strip()
            if not title:
                return ToolResult("", "add requires `title`.", is_error=True)
            task = board.add(title)
            return ToolResult("", f"Added {task.id}: {task.title}\n\n{_board_view(board)}")
        if action == "next":
            task = board.next_todo()
            if task is None:
                return ToolResult("", "No unclaimed tasks on the board.")
            board.claim(task.id, worker)
            return ToolResult("", f"Claimed {task.id}: {task.title}")

        # The remaining actions target a specific task id.
        task_id = args.get("id")
        if not task_id:
            return ToolResult("", f"{action} requires `id`.", is_error=True)
        if action == "claim":
            result = board.claim(task_id, args.get("assignee") or worker)
        elif action == "complete":
            result = board.complete(task_id)
        elif action == "block":
            result = board.block(task_id, args.get("reason") or "")
        elif action == "comment":
            text = (args.get("text") or "").strip()
            if not text:
                return ToolResult("", "comment requires `text`.", is_error=True)
            result = board.comment(task_id, text)
        else:  # pragma: no cover - schema restricts the enum
            return ToolResult("", f"unknown action {action!r}", is_error=True)

        if result is None:
            return ToolResult("", f"No task with id={task_id}.", is_error=True)
        return ToolResult("", f"{render_task(result)}")

    return ToolSpec(
        name="kanban",
        description=(
            "Coordinate work on a shared task board. Use it to break a large job "
            "into tasks and track them, or — as a worker — to pull and finish work. "
            "Actions: list; add (title); next (claim the next todo); claim/complete/"
            "block/comment (id, plus reason/text where relevant)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "next", "claim", "complete", "block", "comment"],
                },
                "title": {"type": "string", "description": "Task title (for add)."},
                "id": {"type": "string", "description": "Task id (claim/complete/block/comment)."},
                "assignee": {"type": "string", "description": "Who claims it (default: you)."},
                "reason": {"type": "string", "description": "Why it's blocked (for block)."},
                "text": {"type": "string", "description": "Comment text (for comment)."},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=kanban,
        category="coordination",
    )
