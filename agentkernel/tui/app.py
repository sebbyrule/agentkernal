"""Curses TUI application — chat panes, input, status bar, and background agent."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

# curses is imported lazily inside TuiApp.run() so the module can be imported
# on platforms where curses is unavailable (e.g., Windows without windows-curses).
# Helper functions like _wrap_text do not depend on self._c.

if TYPE_CHECKING:
    from agentkernel.config import Config

# ── constants ────────────────────────────────────────────────────────────────

_SPINNER = "|/-\\"
_SEND_KEY = ord("\n")  # Enter sends the current input
_QUIT_KEYS = {27, ord("q")}  # Esc or 'q' to quit (Esc is 27)
_SCROLL_UP = {339, 259}  # KEY_PPAGE, KEY_UP
_SCROLL_DOWN = {338, 258}  # KEY_NPAGE, KEY_DOWN

_COLOR_USER = 1
_COLOR_ASSISTANT = 2
_COLOR_TOOL = 3
_COLOR_SYSTEM = 4
_COLOR_STATUS = 5
_COLOR_INPUT_BORDER = 6
_COLOR_SPINNER = 7


# ── message model ────────────────────────────────────────────────────────────

@dataclass
class _DisplayMessage:
    role: str  # "user", "assistant", "tool"
    content: str
    iteration: int = 0


# ── application ──────────────────────────────────────────────────────────────

class TuiApp:
    """Manages the curses screen, input, and background agent execution."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._messages: list[_DisplayMessage] = []
        self._scroll_offset = 0  # lines scrolled back (0 = bottom)
        self._input_text = ""
        self._cursor_pos = 0
        self._status = "Ready. Type a message and press Enter."
        self._running = True

        # Background agent state
        self._agent_thread: threading.Thread | None = None
        self._agent_result: str | None = None
        self._agent_error: str | None = None
        self._agent_done = threading.Event()
        self._spinner_idx = 0

        # Agent components — built lazily on first message
        self._agent = None
        self._telemetry = None
        self._mcp_clients: list = []

        # curses objects set in run()
        self._stdscr: object | None = None
        self._chat_pad: object | None = None
        self._input_win: object | None = None
        self._status_win: object | None = None
        self._max_y = 0
        self._max_x = 0

    # ── public entry point ────────────────────────────────────────────────

    def run(self, stdscr) -> int:
        import curses as _c
        self._c = _c  # lazy curses import for cross-platform compat
        """Main event loop. ``self._c.wrapper`` passes the stdscr."""
        self._stdscr = stdscr
        self._init_colors()
        self._c.curs_set(1)  # show cursor in input area
        stdscr.timeout(80)   # 80 ms getch timeout → polls ~12 fps, no spin

        self._resize()
        self._dirty = True    # force initial draw

        while self._running:
            dirty = self._dirty
            self._dirty = False
            self._poll_agent()
            if dirty or self._dirty:
                self._draw()
            self._handle_input()

        self._cleanup()
        return 0

    # ── drawing ────────────────────────────────────────────────────────────

    def _init_colors(self) -> None:
        self._c.start_color()
        self._c.use_default_colors()
        self._c.init_pair(_COLOR_USER, self._c.COLOR_CYAN, -1)
        self._c.init_pair(_COLOR_ASSISTANT, self._c.COLOR_GREEN, -1)
        self._c.init_pair(_COLOR_TOOL, self._c.COLOR_YELLOW, -1)
        self._c.init_pair(_COLOR_SYSTEM, self._c.COLOR_MAGENTA, -1)
        self._c.init_pair(_COLOR_STATUS, self._c.COLOR_BLACK, self._c.COLOR_WHITE)
        self._c.init_pair(_COLOR_INPUT_BORDER, self._c.COLOR_BLUE, -1)
        self._c.init_pair(_COLOR_SPINNER, self._c.COLOR_YELLOW, -1)

    def _resize(self) -> None:
        """Recompute sub-window geometries after a terminal resize."""
        self._max_y, self._max_x = self._stdscr.getmaxyx()

        input_height = 4  # border + 2 content lines
        status_height = 1
        chat_height = self._max_y - input_height - status_height

        # Chat pad — virtual scrollable area, viewport into the main screen
        self._chat_pad = self._c.newpad(max(chat_height * 2, 1024), self._max_x)
        self._chat_pad.scrollok(True)
        self._chat_vp_height = chat_height
        self._chat_vp_width = self._max_x
        self._chat_vp_y = 0
        self._chat_vp_x = 0

        # Input window (bottom, above status)
        self._input_win = self._c.newwin(
            input_height, self._max_x,
            chat_height, 0,
        )
        self._input_win.keypad(True)

        # Status bar (very bottom)
        self._status_win = self._c.newwin(
            status_height, self._max_x,
            self._max_y - 1, 0,
        )

    def _draw(self) -> None:
        """Full redraw of all panes."""
        if self._stdscr is None:
            return
        self._stdscr.erase()

        self._draw_chat()
        self._draw_input()
        self._draw_status()

        self._stdscr.refresh()

    def _draw_chat(self) -> None:
        if self._chat_pad is None:
            return

        self._chat_pad.erase()
        y = 0
        color_role = {
            "user": _COLOR_USER,
            "assistant": _COLOR_ASSISTANT,
            "tool": _COLOR_TOOL,
        }

        # Render messages onto the virtual pad
        for msg in self._messages:
            prefix = {"user": "▶ You", "assistant": "■ Agent", "tool": "⚙ Tool"}.get(msg.role, "?")
            color = color_role.get(msg.role, _COLOR_SYSTEM)

            # Header line
            with contextlib.suppress(Exception):
                self._chat_pad.addstr(
                    y, 0, f" {prefix} ", self._c.color_pair(color) | self._c.A_BOLD
                )
            y += 1

            # Content lines (word-wrapped)
            for line in self._wrap_text(msg.content, self._max_x - 4):
                with contextlib.suppress(Exception):
                    self._chat_pad.addstr(y, 2, line[: self._max_x - 2])
                y += 1
            y += 1  # blank line between messages

        # Compute scroll bounds
        total_lines = y
        viewport_lines = self._chat_vp_height
        max_scroll = max(0, total_lines - viewport_lines)
        self._scroll_offset = min(self._scroll_offset, max_scroll)

        # Display the viewport
        with contextlib.suppress(Exception):
            self._chat_pad.refresh(
                self._scroll_offset, 0,
                self._chat_vp_y, self._chat_vp_x,
                self._chat_vp_y + viewport_lines - 1, self._chat_vp_x + self._chat_vp_width - 1,
            )

    def _draw_input(self) -> None:
        if self._input_win is None:
            return

        self._input_win.erase()
        h, w = self._input_win.getmaxyx()

        # Border
        self._input_win.border(0)
        with contextlib.suppress(Exception):
            self._input_win.addstr(
                0, 2, " Message (Enter=send, Esc=quit) ",
                self._c.color_pair(_COLOR_INPUT_BORDER),
            )

        # Show input text with cursor
        text = self._input_text
        cursor = min(self._cursor_pos, len(text))
        for row in range(min(2, h - 2)):
            line_start = row * (w - 4)
            line_text = text[line_start : line_start + w - 4]
            with contextlib.suppress(Exception):
                self._input_win.addstr(1 + row, 2, line_text)

        # Position cursor
        cursor_y = 1 + (cursor // (w - 4))
        cursor_x = 2 + (cursor % (w - 4))
        if cursor_y < h - 1:
            with contextlib.suppress(Exception):
                self._input_win.move(cursor_y, cursor_x)
        self._input_win.refresh()

    def _draw_status(self) -> None:
        if self._status_win is None:
            return

        self._status_win.erase()
        with contextlib.suppress(Exception):
            self._status_win.bkgd(" ", self._c.color_pair(_COLOR_STATUS))

        status = self._status[: self._max_x - 1]
        with contextlib.suppress(Exception):
            self._status_win.addstr(
                0, 0, status.ljust(self._max_x - 1), self._c.color_pair(_COLOR_STATUS)
            )
        self._status_win.refresh()

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_text(text: str, width: int) -> list[str]:
        """Word-wrap text to the given width."""
        lines: list[str] = []
        for paragraph in text.split("\n"):
            if not paragraph:
                lines.append("")
                continue
            words = paragraph.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= width:
                    current = f"{current} {word}".strip()
                else:
                    if current:
                        lines.append(current)
                    # If a single word is too long, hard-break it
                    while len(word) > width:
                        lines.append(word[:width])
                        word = word[width:]
                    current = word
            if current:
                lines.append(current)
        return lines

    def _append_message(self, role: str, content: str) -> None:
        self._messages.append(_DisplayMessage(role=role, content=content))
        # Auto-scroll to bottom
        self._scroll_offset = 999_999

    # ── input handling ───────────────────────────────────────────────────

    def _handle_input(self) -> None:
        """Read one key from the input window and dispatch."""
        if self._input_win is None:
            return

        try:
            key = self._input_win.getch()
        except Exception:
            return

        if key == -1:
            return  # no input (timeout)

        self._dirty = True  # something changed

        if key in _QUIT_KEYS:
            # Warn once, then allow quit on next Esc.
            if (
                self._agent_thread
                and self._agent_thread.is_alive()
                and not getattr(self, "_quit_warned", False)
            ):
                self._quit_warned = True
                self._status = "Agent is running. Press Esc again to force quit."
                self._dirty = True
                return
            self._running = False
            return

        if key == _SEND_KEY:
            if not self._agent_done.is_set():
                return  # agent still running
            self._submit()
            return

        # Scrolling when chat pad has focus (use Ctrl+Up/Down or Page keys)
        if key in _SCROLL_UP:
            self._scroll_offset = max(0, self._scroll_offset - 3)
            return
        if key in _SCROLL_DOWN:
            self._scroll_offset += 3
            return

        # Text editing
        if key in (self._c.KEY_BACKSPACE, 127, 8):
            if self._cursor_pos > 0:
                self._input_text = (
                    self._input_text[: self._cursor_pos - 1] + self._input_text[self._cursor_pos :]
                )
                self._cursor_pos -= 1
        elif key == self._c.KEY_DC:
            if self._cursor_pos < len(self._input_text):
                self._input_text = (
                    self._input_text[: self._cursor_pos] + self._input_text[self._cursor_pos + 1 :]
                )
        elif key == self._c.KEY_LEFT:
            self._cursor_pos = max(0, self._cursor_pos - 1)
        elif key == self._c.KEY_RIGHT:
            self._cursor_pos = min(len(self._input_text), self._cursor_pos + 1)
        elif key == self._c.KEY_HOME:
            self._cursor_pos = 0
        elif key == self._c.KEY_END:
            self._cursor_pos = len(self._input_text)
        elif 32 <= key <= 126:
            ch = chr(key)
            self._input_text = (
                self._input_text[: self._cursor_pos] + ch + self._input_text[self._cursor_pos :]
            )
            self._cursor_pos += 1

    # ── agent dispatch ────────────────────────────────────────────────────

    def _submit(self) -> None:
        text = self._input_text.strip()
        if not text:
            return

        self._append_message("user", text)
        self._input_text = ""
        self._cursor_pos = 0

        # Spawn background agent
        self._agent_result = None
        self._agent_error = None
        self._agent_done.clear()
        self._status = "Thinking..."
        self._spinner_idx = 0

        self._quit_warned = False
        self._agent_thread = threading.Thread(target=self._run_agent, args=(text,), daemon=True)
        self._agent_thread.start()

    def _run_agent(self, user_input: str) -> None:
        try:
            from agentkernel.cli import build_runtime

            agent, telemetry, mcp_clients = build_runtime(self._config)
            self._telemetry = telemetry
            self._mcp_clients = mcp_clients
            result = agent.run(user_input)
            self._agent_result = result
        except Exception as exc:
            self._agent_error = str(exc)
        finally:
            self._agent_done.set()

    def _poll_agent(self) -> None:
        """Check whether the background agent has finished and collect its result."""
        if self._agent_done.is_set() and self._agent_thread is not None:
            self._agent_thread.join(timeout=0.1)

            if self._agent_error:
                self._append_message("assistant", f"[Error] {self._agent_error}")
                self._status = f"Error: {self._agent_error[:60]}"
            elif self._agent_result is not None:
                self._append_message("assistant", self._agent_result)
                self._status = "Ready"
            else:
                self._status = "Ready"

            self._agent_thread = None
            self._dirty = True
            self._agent_done.clear()
            self._scroll_offset = 999_999  # auto-scroll

            # Clean up MCP clients and telemetry
            if self._telemetry is not None:
                with contextlib.suppress(Exception):
                    self._telemetry.close()
            for client in self._mcp_clients:
                with contextlib.suppress(Exception):
                    client.close()
            self._telemetry = None
            self._mcp_clients = []

        elif self._agent_thread is not None and self._agent_thread.is_alive():
            # Update spinner
            self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
            self._status = f" {_SPINNER[self._spinner_idx]} Thinking..."

    def _cleanup(self) -> None:
        """Join any running agent thread and close resources."""
        if self._agent_thread and self._agent_thread.is_alive():
            self._agent_done.set()
            self._agent_thread.join(timeout=2.0)
        if self._telemetry is not None:
            with contextlib.suppress(Exception):
                self._telemetry.close()
        for client in self._mcp_clients:
            with contextlib.suppress(Exception):
                client.close()
