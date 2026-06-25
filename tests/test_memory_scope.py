"""Per-namespace note scoping (global brain + project-scoped facts).

Scoping is opt-in: with ``scope=None`` (the default) every note is visible, so
existing single-notebook behavior is unchanged. With an active namespace, recall
returns global notes (``scope == ""``) plus notes in that namespace, and new
notes are stamped with it.
"""

from __future__ import annotations

import sqlite3

from agentkernel.cli import _resolve_memory_scope
from agentkernel.config import Config
from agentkernel.memory import (
    JsonlNoteStore,
    SqliteNoteStore,
    _scope_visible,
    make_note_store,
)


def test_scope_visible_rules():
    # Scoping disabled -> everything visible.
    assert _scope_visible("anything", None)
    # Global notes visible under any active scope.
    assert _scope_visible("", "proj")
    # Matching namespace visible; mismatching hidden.
    assert _scope_visible("proj", "proj")
    assert not _scope_visible("other", "proj")


def test_jsonl_default_scope_disabled_shows_all(tmp_path):
    notes = JsonlNoteStore(tmp_path / "n.jsonl")  # scope=None
    a = notes.add("uses postgres")
    assert a.scope == ""  # stamped global when scoping is off
    # A note from a different scope (written elsewhere) is still visible.
    notes._notes.append(type(a)(text="other-scoped", note_id=99, scope="other"))
    texts = {n.text for n in notes.search("postgres")} | {
        n.text for n in notes.recent(10)
    }
    assert "uses postgres" in texts


def test_jsonl_active_scope_filters_recall(tmp_path):
    proj = JsonlNoteStore(tmp_path / "n.jsonl", scope="proj")
    proj.add("project uses redis")
    # Simulate a note written under a different project and a global note.
    proj._notes.append(
        type(proj._notes[0])(text="other uses redis", note_id=50, scope="other")
    )
    proj._notes.append(
        type(proj._notes[0])(text="global uses redis", note_id=51, scope="")
    )
    found = {n.text for n in proj.search("redis")}
    assert "project uses redis" in found
    assert "global uses redis" in found
    assert "other uses redis" not in found


def test_jsonl_add_stamps_active_scope(tmp_path):
    proj = JsonlNoteStore(tmp_path / "n.jsonl", scope="proj")
    note = proj.add("scoped fact")
    assert note.scope == "proj"
    # Persisted and reloaded with the scope intact.
    reopened = JsonlNoteStore(tmp_path / "n.jsonl", scope="proj")
    assert reopened.all()[0].scope == "proj"


def test_sqlite_active_scope_filters_recall(tmp_path):
    path = tmp_path / "n.db"
    proj = SqliteNoteStore(path, scope="proj")
    proj.add("project deploy target")
    other = SqliteNoteStore(path, scope="other")
    other.add("other deploy target")
    glob = SqliteNoteStore(path, scope=None)
    glob.add("shared deploy target")  # written global (scoping off -> scope "")

    results = {n.text for n in proj.search("deploy")}
    assert "project deploy target" in results
    assert "shared deploy target" in results
    assert "other deploy target" not in results


def test_sqlite_recent_respects_scope(tmp_path):
    path = tmp_path / "n.db"
    SqliteNoteStore(path, scope="a").add("alpha note")
    SqliteNoteStore(path, scope="b").add("beta note")
    a = SqliteNoteStore(path, scope="a")
    texts = {n.text for n in a.recent(10)}
    assert texts == {"alpha note"}


def test_sqlite_migrates_legacy_notebook_without_scope_column(tmp_path):
    path = tmp_path / "legacy.db"
    # Build a notebook with the pre-scope schema and one row.
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            created TEXT NOT NULL,
            accessed TEXT,
            access_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO notes (text, tags_json, created) VALUES (?, '[]', ?)",
        ("legacy fact", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    # Opening it migrates the schema; the legacy note is treated as global.
    store = SqliteNoteStore(path, scope="proj")
    legacy = store.all()[0]
    assert legacy.scope == ""
    assert legacy.text in {n.text for n in store.search("legacy")}


def test_make_note_store_passes_scope_through(tmp_path):
    store = make_note_store(tmp_path / "n.db", scope="proj")
    note = store.add("hello")
    assert note.scope == "proj"


def test_resolve_memory_scope():
    assert _resolve_memory_scope(Config(memory_scope=None)) is None
    assert _resolve_memory_scope(Config(memory_scope="")) is None
    assert _resolve_memory_scope(Config(memory_scope="myteam")) == "myteam"
    auto = _resolve_memory_scope(Config(memory_scope="auto", working_dir="/tmp/proj-x"))
    assert auto == "proj-x"
