"""Self-curating memory tests (design §13, Phase 3). Offline: a real JSONL note
store plus a scripted FakeProvider for the extraction/consolidation model."""

from __future__ import annotations

from agentkernel.curation import MemoryCurator, _parse_json_array
from agentkernel.memory import JsonlNoteStore
from agentkernel.types import Message
from tests.fakes import FakeProvider, text_response


def _notes(tmp_path):
    return JsonlNoteStore(tmp_path / "notes.jsonl")


def _convo():
    return [
        Message(role="user", content="I prefer tabs and we use uv for everything."),
        Message(role="assistant", content="Got it."),
    ]


def test_parse_json_array_tolerant():
    assert _parse_json_array('[{"text": "a"}]') == [{"text": "a"}]
    assert _parse_json_array('verdict: [{"text": "a"}] done') == [{"text": "a"}]
    assert _parse_json_array("not json at all") == []
    assert _parse_json_array('[1, 2, {"text": "x"}]') == [{"text": "x"}]  # non-dicts dropped


def test_extract_adds_durable_facts(tmp_path):
    notes = _notes(tmp_path)
    reply = '[{"text": "User prefers tabs", "tags": ["prefs"]}, {"text": "Project uses uv"}]'
    provider = FakeProvider([text_response(reply)])
    result = MemoryCurator(notes, provider).extract(_convo())
    assert len(result.added) == 2
    assert {n.text for n in notes.all()} == {"User prefers tabs", "Project uses uv"}
    assert notes.all()[0].tags == ["prefs"]


def test_extract_skips_duplicates(tmp_path):
    notes = _notes(tmp_path)
    notes.add("User prefers tabs over spaces")
    reply = '[{"text": "User prefers tabs over spaces"}, {"text": "Project uses uv"}]'
    provider = FakeProvider([text_response(reply)])
    result = MemoryCurator(notes, provider).extract(_convo())
    assert result.skipped_duplicates == 1
    assert [n.text for n in result.added] == ["Project uses uv"]


def test_extract_unparseable_reply_changes_nothing(tmp_path):
    notes = _notes(tmp_path)
    provider = FakeProvider([text_response("I could not find any durable facts.")])
    result = MemoryCurator(notes, provider).extract(_convo())
    assert result.added == [] and notes.all() == []


def test_consolidate_merges_and_rebuilds(tmp_path):
    notes = _notes(tmp_path)
    notes.add("User likes Python")
    notes.add("User enjoys python programming")
    notes.add("Project uses uv")
    provider = FakeProvider(
        [text_response('[{"text": "User likes Python programming"}, {"text": "Project uses uv"}]')]
    )
    result = MemoryCurator(notes, provider).consolidate()
    assert result.before == 3 and result.after == 2 and result.removed == 1
    texts = {n.text for n in notes.all()}
    assert texts == {"User likes Python programming", "Project uses uv"}


def test_consolidate_noop_below_two_notes(tmp_path):
    notes = _notes(tmp_path)
    notes.add("only one fact")
    # Provider has no scripted responses; consolidate must not call the model.
    result = MemoryCurator(notes, FakeProvider([])).consolidate()
    assert result.before == 1 and result.after == 1


def test_consolidate_keeps_notes_on_unparseable_reply(tmp_path):
    notes = _notes(tmp_path)
    notes.add("fact one")
    notes.add("fact two")
    provider = FakeProvider([text_response("sorry, no JSON")])
    result = MemoryCurator(notes, provider).consolidate()
    assert result.after == 2 and len(notes.all()) == 2  # unchanged, not destroyed


def test_run_memory_consolidate_empty_notebook(tmp_path):
    from agentkernel.cli import run_memory
    from agentkernel.config import Config

    cfg = Config(
        provider="anthropic",
        memory_notes_path=str(tmp_path / "notes.jsonl"),
        log_dir=str(tmp_path / "t"),
    )
    out: list[str] = []
    # Empty notebook -> consolidate is a no-op and never calls the model.
    assert run_memory(cfg, "consolidate", output_fn=out.append) == 0
    assert any("0 -> 0" in line for line in out)
