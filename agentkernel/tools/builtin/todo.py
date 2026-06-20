"""In-session todo tool (design §18.4).

A lightweight task list the model maintains while working through a multi-step
job, so its plan stays legible to both the model and the user. State is held in
memory for the life of one runtime (a session), bound into the handler by the
factory — no global state, no persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentkernel.tools.base import ToolSpec
from agentkernel.types import ToolResult

_STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}


@dataclass
class TodoItem:
    id: int
    text: str
    status: str = "pending"  # pending | in_progress | done


@dataclass
class TodoList:
    """The session's task list. Bound into the todo tool's handler."""

    _items: list[TodoItem] = field(default_factory=list)
    _next_id: int = 1

    def add(self, text: str) -> TodoItem:
        item = TodoItem(self._next_id, text.strip())
        self._items.append(item)
        self._next_id += 1
        return item

    def set_status(self, item_id: int, status: str) -> TodoItem | None:
        for item in self._items:
            if item.id == item_id:
                item.status = status
                return item
        return None

    def clear(self) -> None:
        self._items.clear()

    def render(self) -> str:
        if not self._items:
            return "(todo list is empty)"
        done = sum(1 for i in self._items if i.status == "done")
        lines = [f"Todos ({done}/{len(self._items)} done):"]
        lines += [f"  {_STATUS_MARK[i.status]} {i.id}. {i.text}" for i in self._items]
        return "\n".join(lines)


def todo_tool(todo_list: TodoList) -> ToolSpec:
    """Build the `todo` tool over a session task list."""

    def todo(args: dict) -> ToolResult:
        action = args["action"]
        if action == "add":
            text = (args.get("text") or "").strip()
            if not text:
                return ToolResult("", "add requires non-empty `text`.", is_error=True)
            todo_list.add(text)
            return ToolResult("", todo_list.render())
        if action in ("start", "complete"):
            if "id" not in args:
                return ToolResult("", f"{action} requires `id`.", is_error=True)
            status = "in_progress" if action == "start" else "done"
            if todo_list.set_status(int(args["id"]), status) is None:
                return ToolResult("", f"No todo with id={args['id']}.", is_error=True)
            return ToolResult("", todo_list.render())
        if action == "clear":
            todo_list.clear()
            return ToolResult("", "Cleared the todo list.")
        # action == "list"
        return ToolResult("", todo_list.render())

    return ToolSpec(
        name="todo",
        description=(
            "Maintain a short task list for the current job so your plan stays "
            "visible. Use it for multi-step work: add the steps, mark one `start` "
            "as you begin it and `complete` when done, and `list` to review. "
            "Actions: add (needs text), start/complete (need id), list, clear."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "start", "complete", "list", "clear"],
                    "description": "The operation to perform.",
                },
                "text": {"type": "string", "description": "Task text (for add)."},
                "id": {"type": "integer", "description": "Task id (for start/complete)."},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=todo,
        category="planning",
    )
