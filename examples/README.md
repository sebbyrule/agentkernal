# Testing the agent yourself

This folder gives you a self-contained way to exercise the agent and its builtin
tools by hand. There are two pieces:

- **`playground/`** — a tiny project (a calculator with a deliberate bug, a small
  inventory module, and a notes file with TODOs) for the agent to read, search,
  and edit.
- **`evals/builtin-tools.toml`** — a scored eval suite that runs a handful of
  tasks against the playground automatically.

Everything works offline against your configured provider (e.g. the local
LM Studio endpoint in `agentkernel.toml`).

## Builtin tools at a glance

| Tool | What it does | Gated? |
|---|---|---|
| `read_file` | read a text file under the working dir | no |
| `list_dir` | list a directory | no |
| `find_files` | glob for files (`**/*.py`), skips noise dirs | no |
| `search_text` | regex grep over file contents → `path:line: text` | no |
| `file_info` | size / type / mtime / line count for a path | no |
| `write_file` | create or overwrite a file | **yes** |
| `edit_file` | exact-substring replace in a file | **yes** |
| `bash` | run a shell command in the sandbox | **yes** |

Gated tools prompt for approval in the REPL/TUI (policy `always_ask`). For
hands-off runs use `--memory`-style env overrides or set
`approval_policy = "auto_allow"`.

## Try it interactively

Start a session (REPL or the full-screen TUI):

```bash
uv run agentkernel            # REPL
uv run agentkernel tui        # curses UI
```

Then paste any of these prompts. Each is chosen to make the model reach for a
specific builtin tool:

1. **find_files** — *"List every Python file under examples/playground."*
2. **search_text** — *"Search examples/playground for every TODO and show me the file and line of each."*
3. **file_info** — *"How big is examples/playground/inventory.py and how many lines does it have?"*
4. **read_file + reasoning** — *"Read examples/playground/inventory.py and tell me which SKUs are out of stock and the total dollar value of the inventory."*
5. **spot a bug** — *"There's a bug in examples/playground/calc.py. Find it and quote the exact line."*
6. **edit_file (gated)** — *"Fix the bug in examples/playground/calc.py using edit_file."* — you'll be asked to approve the write; say yes, then verify with `git diff`.
7. **bash (gated)** — *"Run the test for calc with `python -c \"import examples.playground.calc as c; print(c.subtract(5,3))\"` and tell me if it's correct."*

> Tip: after task 6 edits the file, run `git checkout examples/playground/calc.py`
> to restore the deliberate bug so you can try again.

## Try it as a scored eval suite

The eval harness runs each case through the agent and asks a judge model to
score the answer against a rubric (0–1, pass/fail), then prints a pass-rate.

```bash
# all cases
uv run agentkernel eval --suite examples/evals/builtin-tools.toml

# write a machine-readable report
uv run agentkernel eval --suite examples/evals/builtin-tools.toml -o report.json

# run a subset by name glob
uv run agentkernel eval --suite examples/evals/builtin-tools.toml --case "*bug*"
```

All five cases are read-only, so the suite runs unattended (no approval prompts).
Use `--judge-model <id>` to score with a different/stronger model than the one
answering, if your endpoint has one loaded.

## One-shot, non-interactive

```bash
uv run agentkernel run "Find every TODO under examples/playground and list each with its file and line."
```
