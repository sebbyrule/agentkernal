"""Tests for the read-only discovery tools (find_files, search_text, file_info)
and the edit_file mutation: matching, confinement, and error-as-result behavior."""

from __future__ import annotations

from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import file_tools, search_tools
from agentkernel.types import ToolCall


def _registry(tmp_path) -> ToolRegistry:
    reg = ToolRegistry()
    for spec in file_tools(str(tmp_path)) + search_tools(str(tmp_path)):
        reg.register(spec)
    return reg


def _seed(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("VALUE = 42\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Title\nhello world\n", encoding="utf-8")
    skip = tmp_path / "__pycache__"
    skip.mkdir()
    (skip / "junk.py").write_text("hello\n", encoding="utf-8")


# --- find_files ----------------------------------------------------------------

def test_find_files_recursive_glob(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "find_files", {"pattern": "**/*.py"}))
    assert not r.is_error
    assert "src/app.py" in r.content and "src/util.py" in r.content
    # noise dirs are skipped
    assert "__pycache__" not in r.content


def test_find_files_no_match(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "find_files", {"pattern": "*.rs"}))
    assert not r.is_error and "no files match" in r.content


# --- search_text ---------------------------------------------------------------

def test_search_text_finds_matches_with_locations(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "search_text", {"pattern": r"hello"}))
    assert not r.is_error
    assert "README.md:2:" in r.content
    assert "__pycache__" not in r.content  # skipped dir


def test_search_text_glob_filter_and_ignore_case(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(
        ToolCall("c", "search_text", {"pattern": "VALUE", "glob": "**/*.py", "ignore_case": True})
    )
    assert not r.is_error and "src/util.py:1:" in r.content
    assert "README.md" not in r.content


def test_search_text_invalid_regex_is_error(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "search_text", {"pattern": "(unclosed"}))
    assert r.is_error and "Invalid regex" in r.content


def test_search_text_respects_max_results(tmp_path):
    (tmp_path / "many.txt").write_text("\n".join("hit" for _ in range(50)), encoding="utf-8")
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "search_text", {"pattern": "hit", "max_results": 5}))
    assert not r.is_error
    assert r.content.count("many.txt:") == 5
    assert "stopped at 5 matches" in r.content


# --- file_info -----------------------------------------------------------------

def test_file_info_reports_lines_for_text(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "file_info", {"path": "src/app.py"}))
    assert not r.is_error
    assert "type: file" in r.content and "lines:" in r.content


def test_file_info_missing_is_error(tmp_path):
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "file_info", {"path": "nope.txt"}))
    assert r.is_error and "No such path" in r.content


def test_search_tool_path_escape_is_error(tmp_path):
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "find_files", {"pattern": "*", "path": "../.."}))
    assert r.is_error and "escapes working directory" in r.content


# --- edit_file -----------------------------------------------------------------

def test_edit_file_replaces_unique_match(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(
        ToolCall("c", "edit_file", {"path": "src/util.py", "old": "42", "new": "99"})
    )
    assert not r.is_error and "Replaced 1" in r.content
    assert (tmp_path / "src" / "util.py").read_text(encoding="utf-8") == "VALUE = 99\n"


def test_edit_file_ambiguous_match_refuses(tmp_path):
    (tmp_path / "d.txt").write_text("x\nx\n", encoding="utf-8")
    reg = _registry(tmp_path)
    r = reg.execute(ToolCall("c", "edit_file", {"path": "d.txt", "old": "x", "new": "y"}))
    assert r.is_error and "not unique" in r.content
    # file is untouched on the error path
    assert (tmp_path / "d.txt").read_text(encoding="utf-8") == "x\nx\n"


def test_edit_file_replace_all(tmp_path):
    (tmp_path / "d.txt").write_text("x\nx\n", encoding="utf-8")
    reg = _registry(tmp_path)
    r = reg.execute(
        ToolCall("c", "edit_file", {"path": "d.txt", "old": "x", "new": "y", "replace_all": True})
    )
    assert not r.is_error and "Replaced 2" in r.content
    assert (tmp_path / "d.txt").read_text(encoding="utf-8") == "y\ny\n"


def test_edit_file_missing_old_is_error(tmp_path):
    _seed(tmp_path)
    reg = _registry(tmp_path)
    r = reg.execute(
        ToolCall("c", "edit_file", {"path": "README.md", "old": "absent", "new": "z"})
    )
    assert r.is_error and "not found" in r.content


def test_edit_file_is_gated():
    specs = {s.name: s for s in file_tools(".")}
    assert specs["edit_file"].gated  # mutates + requires_approval
