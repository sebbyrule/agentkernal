"""Terminal UI for agentkernel — a curses-based interactive chat interface.

Provides a split-pane terminal UI with:
- Scrollable chat history (color-coded by role)
- Multi-line input area
- Status bar (model, tokens, cost)
- Background agent execution with live status indicator

Usage:
    from agentkernel.tui import run_tui
    run_tui(config)  # or: uv run agentkernel tui

On Windows, ``pip install windows-curses`` first.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentkernel.config import Config


def run_tui(config: Config) -> int:
    """Entry point: initialise curses and run the TUI event loop.

    Returns 0 on clean exit, 1 on error.
    """
    try:
        import curses
    except ImportError:
        print(
            "The TUI requires the `curses` module, which is not available.\n"
            "On Windows, install it with:  pip install windows-curses\n"
            "On Unix, it should be included with your Python installation.",
            file=sys.stderr,
        )
        return 1

    from agentkernel.tui.app import TuiApp

    try:
        app = TuiApp(config)
        return curses.wrapper(app.run)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"TUI error: {exc}", file=sys.stderr)
        return 1
