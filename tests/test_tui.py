"""Tests for the TUI module (non-curses logic only).

Curses rendering is inherently visual and platform-dependent; these tests
cover the message buffer, text wrapping, input handling logic, and the
clean entry-point error path.
"""

from __future__ import annotations

import sys
from pathlib import Path


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
    """When curses is unavailable, run_tui prints a helpful message and returns 1."""
    monkeypatch.setitem(sys.modules, "curses", None)

    # Remove curses from sys.modules so the import inside run_tui fails.
    saved = sys.modules.pop("curses", None)
    try:
        from agentkernel.tui import run_tui
        from agentkernel.config import Config

        # Simulate import failure
        code = run_tui(Config())
        assert code == 1
    finally:
        if saved is not None:
            sys.modules["curses"] = saved


def test_tui_module_exports_run_tui():
    from agentkernel.tui import run_tui

    assert callable(run_tui)