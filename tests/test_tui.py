"""Tests for the TUI module (non-curses logic only).

Curses rendering is inherently visual and platform-dependent; these tests
cover the message buffer, text wrapping, input handling logic, and the clean
entry-point error path.
"""

from __future__ import annotations

import sys

# ── message buffer tests ────────────────────────────────────────────────────

def test_display_message_model():
    from agentkernel.tui.app import _DisplayMessage

    msg = _DisplayMessage(role="assistant", content="hello world", iteration=0)
    assert msg.role == "assistant"
    assert msg.content == "hello world"
    assert msg.iteration == 0


def test_wrap_text_simple():
    from agentkernel.tui.app import TuiApp

    wrapped = TuiApp._wrap_text("hello world", width=6)
    assert wrapped == ["hello", "world"]


def test_wrap_text_long_word():
    from agentkernel.tui.app import TuiApp

    wrapped = TuiApp._wrap_text("abcdefghij", width=3)
    assert wrapped == ["abc", "def", "ghi", "j"]


def test_wrap_text_paragraphs():
    from agentkernel.tui.app import TuiApp

    wrapped = TuiApp._wrap_text("line one\n\nline two", width=20)
    assert wrapped == ["line one", "", "line two"]


def test_wrap_text_unicode():
    from agentkernel.tui.app import TuiApp

    wrapped = TuiApp._wrap_text("héllo wörld", width=10)
    assert wrapped == ["héllo", "wörld"]


# ── entry-point error path ──────────────────────────────────────────────────

def test_run_tui_graceful_import_error(monkeypatch):
    """run_tui returns 1 with a helpful message when curses is unavailable."""
    try:
        import curses  # noqa: F401 — probe availability
        curses_available = True
    except ImportError:
        curses_available = False

    from agentkernel.config import Config
    from agentkernel.tui import run_tui

    if curses_available:
        # Simulate unavailability by hiding the real module.
        saved = sys.modules.pop("curses", None)
        try:
            sys.modules["curses"] = None
            code = run_tui(Config())
            assert code == 1
        finally:
            if saved is not None:
                sys.modules["curses"] = saved
    else:
        # Already unavailable — just call through.
        code = run_tui(Config())
        assert code == 1


def test_tui_module_exports_run_tui():
    from agentkernel.tui import run_tui

    assert callable(run_tui)


# ── draw composition (no-flicker regression) ────────────────────────────────

class _FakeWin:
    def __init__(self, size=(4, 40)):
        self._size = size
        self.calls: list[str] = []

    def _rec(self, name):
        self.calls.append(name)

    def erase(self):
        self._rec("erase")

    def noutrefresh(self, *a):
        self._rec("noutrefresh")

    def refresh(self, *a):
        self._rec("refresh")

    def addstr(self, *a, **k):
        self._rec("addstr")

    def border(self, *a):
        self._rec("border")

    def move(self, *a):
        self._rec("move")

    def bkgd(self, *a):
        self._rec("bkgd")

    def getmaxyx(self):
        return self._size


class _FakeCurses:
    A_BOLD = 1

    def __init__(self):
        self.doupdate_calls = 0

    def color_pair(self, n):
        return 0

    def doupdate(self):
        self.doupdate_calls += 1


# ── slash commands & branding ───────────────────────────────────────────────

def _app():
    from agentkernel.config import Config
    from agentkernel.tui.app import TuiApp

    return TuiApp(Config(provider="local", model="demo-model"))


def test_startup_shows_welcome_system_message():
    app = _app()
    assert app._messages[0].role == "system"
    assert "Welcome to agentkernel" in app._messages[0].content


def test_command_help_appends_help():
    app = _app()
    before = len(app._messages)
    app._handle_command("/help")
    assert len(app._messages) == before + 1
    assert app._messages[-1].role == "system"
    assert "/tools" in app._messages[-1].content


def test_command_clear_resets_to_welcome():
    from agentkernel.tui.app import _DisplayMessage

    app = _app()
    app._messages.append(_DisplayMessage("user", "hi"))
    app._handle_command("/clear")
    assert len(app._messages) == 1
    assert app._messages[0].role == "system"
    assert "Welcome" in app._messages[0].content


def test_command_exit_stops_running():
    app = _app()
    assert app._running
    app._handle_command("/exit")
    assert app._running is False


def test_command_model_reports_config():
    app = _app()
    app._handle_command("/model")
    assert "demo-model" in app._messages[-1].content


def test_command_tools_before_any_run():
    app = _app()
    app._handle_command("/tools")
    assert "Send a message first" in app._messages[-1].content

    app._tool_lines = ["  read_file: read a file"]
    app._handle_command("/tools")
    assert "read_file" in app._messages[-1].content


def test_command_unknown_is_reported():
    app = _app()
    app._handle_command("/bogus")
    assert "Unknown command" in app._messages[-1].content


def test_submit_routes_slash_to_command_not_agent():
    """A slash command must be handled locally, never spawned as an agent run."""
    app = _app()
    app._input_text = "/help"
    app._cursor_pos = len(app._input_text)
    app._submit()
    assert app._agent_thread is None  # no background run started
    assert app._input_text == ""  # input cleared
    assert "/tools" in app._messages[-1].content


def test_can_submit_again_after_run_completes():
    """Regression: the send gate must reopen after a run finishes. _agent_done
    doubles as the 'ready to submit' flag; clearing it on completion previously
    blocked every message after the first."""
    app = _app()

    class _DoneThread:
        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    app._agent_thread = _DoneThread()
    app._agent_result = "the answer"
    app._agent_done.set()  # _run_agent's `finally` sets this when the run ends

    app._poll_agent()

    assert app._agent_thread is None
    assert app._agent_done.is_set()  # gate open → a second message can be sent
    assert app._messages[-1].content == "the answer"


def test_draw_is_double_buffered_and_does_not_flicker():
    """Regression: panes must be staged with noutrefresh and flushed by a single
    doupdate. A per-pane refresh (or refreshing stdscr on top of the panes) blanks
    what was just drawn and makes the screen flicker."""
    from agentkernel.config import Config
    from agentkernel.tui.app import TuiApp, _DisplayMessage

    app = TuiApp(Config())
    fc = _FakeCurses()
    app._c = fc
    app._stdscr = _FakeWin((24, 40))
    app._title_win = _FakeWin((1, 40))
    app._chat_pad = _FakeWin((48, 40))
    app._input_win = _FakeWin((4, 40))
    app._status_win = _FakeWin((1, 40))
    app._max_x = 40
    app._max_y = 24
    app._chat_vp_height = 18
    app._chat_vp_width = 40
    app._chat_vp_y = 1
    app._chat_vp_x = 0
    app._scroll_offset = 0
    app._status = "Ready"
    app._input_text = "hi"
    app._cursor_pos = 2
    app._messages = [_DisplayMessage("user", "hello")]

    app._draw()

    assert fc.doupdate_calls == 1  # exactly one physical update per frame
    panes = (app._stdscr, app._title_win, app._chat_pad, app._input_win, app._status_win)
    for win in panes:
        assert "noutrefresh" in win.calls
        assert "refresh" not in win.calls
