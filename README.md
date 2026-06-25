# agentkernel

A minimal, dependency-light **kernel for a general-purpose AI agent**. The kernel runs the agent loop — send a conversation plus tool definitions to a language model, parse out any tool calls, execute them through a registry, feed the results back, repeat until the model produces a final answer — and nothing more.

It is **provider-agnostic**, **tool-agnostic**, and **fully testable without network access**. Everything a user might call a "feature" (web search, file editing, project memory, profiles) is built *on top of* this kernel as a tool, a piece of injected context, or a run parameter — never inside it.

> Dependencies: `jsonschema` + `httpx`. No agent frameworks (no LangChain, LlamaIndex, CrewAI, …). The loop is the product.

---

## Why this exists

Most "agent frameworks" hide the one thing that actually matters — the loop — behind layers of abstraction. agentkernel keeps the loop small, explicit, and readable, and pushes everything else to the edges:

- **Everything is a tool, a context injection, or a run parameter.** If a proposed addition isn't one of those three, it doesn't belong in the kernel.
- **One canonical message format.** Provider quirks (Anthropic content blocks vs. OpenAI `tool_calls` arrays) are normalized inside adapters and never leak into the loop or the registry.
- **Errors become tool results, not exceptions.** A failing tool returns a `ToolResult(is_error=True)`; the loop keeps going so the model can recover. Only unrecoverable kernel faults raise.
- **The cacheable prefix is stable.** System prompt + tool definitions are assembled once per run and never reordered, so prompt caching actually hits — treated as a correctness property, not an optimization.
- **Every mutation is gated.** Tools that write files or run shell pass through an approver, and code runs inside a sandbox confined to the working directory.
- **Telemetry from turn one.** Every turn records tokens (including cache read/write), tool calls, stop reason, and cost — redacted by default.

---

## Installation

Requires **Python 3.11+**.

**As a CLI tool** (recommended for everyday use — puts `agentkernel` on your `PATH`):

```bash
uv tool install agentkernel-cli  # or: pipx install agentkernel-cli  /  pip install agentkernel-cli
agentkernel init                 # scaffold ./agentkernel.toml (or: agentkernel init --global)
agentkernel run "summarize the failing tests"
```

**For development** (working on agentkernel itself), with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev      # install runtime + dev (pytest, ruff) dependencies
uv run pytest
```

API keys are read **only** from the environment — never from config files or traces:

```bash
export ANTHROPIC_API_KEY=***     # for provider = "anthropic"
export OPENAI_API_KEY=***        # for provider = "openai" / embeddings
# local/OpenAI-compatible endpoints (Ollama, vLLM) usually need no key
# Credential pool: give several keys and the provider rotates on rate limits —
# comma-separate (ANTHROPIC_API_KEY="k1,k2") or number them (ANTHROPIC_API_KEY_1, _2).
```

### Running anywhere

`agentkernel` works from any directory, not just its own repo:

```bash
agentkernel -C ~/code/my-app run "summarize the failing tests"
```

Config is discovered in layers — **explicit `--config`** overrides everything;
otherwise the user-global **`~/.agentkernel/config.toml`** is the base and the
nearest project **`agentkernel.toml`** (found by walking up from the target
directory) overrides it, then `AGENTKERNEL_*` env vars, then defaults. Set
`AGENTKERNEL_HOME` to relocate the global home.

State follows a **global brain, project sessions** policy:

| Lives in `~/.agentkernel/` (global) | Lives in `<project>/.agentkernel/` (per-project) |
|---|---|
| memory notebook, knowledge graph, skills, profiles, improvements, cron jobs | session traces, kanban board, checkpoints, the session memory store |

So your skills, long-term memory, and scheduled jobs are shared across every
project, while each project keeps its own transcripts and work board. `-C PATH`
(like `git -C`) points the agent at a project from elsewhere; an absolute path in
config is always honored as-is, and a path customized in a project's
`agentkernel.toml` stays project-local.

## Quick start

```bash
uv run agentkernel                            # interactive REPL (default)
uv run agentkernel tui                        # full-screen curses terminal UI
uv run agentkernel run "your prompt"          # single non-interactive run, prints the answer
uv run agentkernel run --file task.md         # single run from a prompt file
uv run agentkernel run --background "..."      # detached run; output goes to a file
uv run agentkernel improve                    # reflect on the latest trace, write a rule note
uv run agentkernel eval --suite s.toml        # run an eval suite, score answers with a judge
uv run agentkernel eval --suite s.toml -o report.json  # ...and write a JSON report
uv run agentkernel loop --file l.toml         # run a workflow loop until its stopping condition
uv run agentkernel insights --days 30         # aggregate session traces into a usage/cost report
uv run agentkernel doctor                     # check config, deps, credentials, sandbox
uv run agentkernel sessions list              # list saved sessions (needs a memory store)
uv run agentkernel --resume <id> run "..."    # resume a saved session by id
uv run agentkernel cron add 1h "check CI"     # schedule a job; `cron tick` runs what's due
uv run agentkernel cron tick                  # run all due jobs once (drive from OS scheduler)
uv run agentkernel kanban add "ship release"  # file a task on the shared work board
uv run agentkernel kanban list                # inspect the work board
uv run agentkernel new skill my-skill         # scaffold a skill/profile/loop/eval from a template
uv run agentkernel --profile reviewer run "review src/"  # run with a bundled profile
uv run agentkernel --skill code-review repl   # start REPL with a skill pinned
uv run agentkernel --model o3-mini run "hi"   # override the model for one run
uv run agentkernel --help                     # options
uv run pytest                                 # full test suite, offline
```

The REPL keeps conversational context across messages, prints a one-line progress
status per turn, and writes a per-session JSONL trace. It supports slash commands:

```
$ uv run agentkernel
[session trace: .agentkernel/traces/<session-id>.jsonl]
agentkernel REPL - type your message and press enter. Commands: /exit, /clear,
/system, /profile, /skills, /skill, /tools, /trace, /cost, /memory, /improve.
> summarize the files in this directory
```

| Command | Effect |
|---|---|
| `/clear` | reset the conversation context |
| `/system [text]` | set (or clear) the system prompt for following turns |
| `/profile [name]` | show, or load, a profile from `profile_dir` |
| `/skills` | list discovered skills (`*` = active) |
| `/skill <name>` | toggle a skill on/off |
| `/tools` | list registered tools (builtin + MCP + graph) |
| `/trace` / `/cost` | show the trace path / cumulative session cost |
| `/memory [list [limit] \| delete <note_id> \| export [path] \| reindex]` | manage the notebook |
| `/improve [trace-path]` | reflect on the current (or chosen) trace and write an improvement |
| `/exit` | leave |

### Terminal UI

`uv run agentkernel tui` launches a full-screen [curses](agentkernel/tui) interface over the same runtime: a scrollable, color-coded chat history, a multi-line input area, and a status bar, with the agent running on a background thread so the UI stays responsive. Type and press **Enter** to send, **PgUp/PgDn** (or arrows) to scroll, **Esc**/**q** to quit. It reads the same `agentkernel.toml`, so any configured memory, skills, and MCP servers are active. On Windows the `windows-curses` backend is installed automatically; on Unix `curses` ships with Python.

### Using the kernel as a library

```python
from agentkernel.config import Config
from agentkernel.cli import build_runtime

config = Config(provider="anthropic", model="claude-sonnet-4-6")
agent, telemetry, mcp_clients = build_runtime(config)
try:
    print(agent.run("List the Python files here and count the lines in each."))
finally:
    telemetry.close()
    for client in mcp_clients:
        client.close()
```

`build_runtime` wires a provider, the builtin tools inside a `LocalSandbox`, a `CliApprover`, JSONL telemetry, and any configured MCP servers / skills / knowledge-graph / memory tools into an `Agent`. You can also assemble these yourself — every collaborator is injected, nothing is global.

---

## Configuration

Configuration loads from `agentkernel.toml` (see [`agentkernel.toml.example`](agentkernel.toml.example)) with this precedence:

> explicit constructor args **>** `AGENTKERNEL_*` environment variables **>** `agentkernel.toml` **>** defaults
> CLI flags (`--model`, `--profile`, `--skill`, `--memory`) override the file.

| Key | Default | Meaning |
|---|---|---|
| `provider` | `anthropic` | `anthropic` \| `openai` \| `local` |
| `model` | `claude-sonnet-4-6` | model id for the selected provider |
| `base_url` | `None` | endpoint for `provider = "local"` |
| `max_output_tokens` | `4096` | reply token cap |
| `output_reserve` | `8192` | budget headroom reserved for the reply |
| `max_iterations` | `25` | loop guard against runaway sessions |
| `keep_recent_turns` | `6` | turns kept verbatim during compaction |
| `max_tool_result_tokens` | `4096` | per-result truncation cap |
| `approval_policy` | `always_ask` | `always_ask` \| `auto_allow` \| `deny_mutations` \| `smart` |
| `approval_allowlist` | `[]` | patterns that skip the approval prompt |
| `approval_judge_model` | `None` | model that judges call risk under `smart` (defaults to `summarizer_model`, then `model`) |
| `redact_tool_output` | `True` | scrub secret-looking strings from tool results before they enter context/traces |
| `checkpoints` | `False` | back up files before edits and register a `rollback` tool to undo them |
| `enable_todo` / `enable_clarify` | `False` | register the in-session `todo` planning tool / the `clarify` ask-the-user tool |
| `enable_plugins` / `plugins_dir` | `False` / `plugins` | auto-load user tool modules from `plugins_dir` (executes their code) |
| `enable_kanban` / `kanban_path` | `False` / `.agentkernel/kanban.json` | register the `kanban` shared work-board tool for multi-agent coordination |
| `working_dir` | `.` | root that file/shell tools are confined to |
| `summarizer_model` | `None` | cheap model for compaction (`None` → structural fallback) |
| `log_dir` | `.agentkernel/traces` | where session traces are written |
| `max_cost_usd` | `None` | per-run cost ceiling; the run stops when exceeded |
| `max_input_tokens_per_run` | `None` | per-run input-token ceiling |
| `profile` / `profile_dir` | `None` / `profiles` | active profile name and where profiles are loaded from |
| `memory_store` / `memory_notes_path` | `None` / `.agentkernel/memory/notes.jsonl` | `file` \| `memory` \| `sqlite`; notebook directory/path |
| `enable_memory_tools` | `False` | register `remember`/`recall`/`forget` tools |
| `memory_auto_context` / `memory_auto_context_limit` | `False` / `3` | auto-inject recalled notes before each user message |
| `memory_store_budget` | `None` | summarize older turns before persisting memory |
| `memory_scope` | `None` | recall namespace: `auto` (project dir name), a literal name, or off; recall returns global + active-namespace notes |
| `memory_recency_weight` / `memory_importance_weight` | `0.0` / `0.0` | opt-in recall re-ranking: boost notes by recency of creation and by recall frequency (`0` = relevance only) |
| `memory_half_life_days` | `30.0` | days for a note's recency score to halve |
| `memory_curator_model` | `None` | cheap model for `memory extract`/`consolidate` (falls back to `summarizer_model`/`model`) |
| `semantic_search` | `False` | rank note recall with dense embeddings (SQLite only) |
| `semantic_search_lsh_bits` | `None` | approximate vector index bits; omit for brute force |
| `embedding_model` | `text-embedding-3-small` | OpenAI-compatible embedding model |
| `embedding_dimensions` | `None` | optional truncation (OpenAI only) |
| `embedding_base_url` | `None` | OpenAI-compatible embedding endpoint |
| `embedding_api_key_env` | `OPENAI_API_KEY` | env var holding the embedding API key |
| `skills_dir` / `skills` | `skills` / `[]` | skill source directory and the initially-active skill names |
| `enable_graph` / `graph_path` | `False` / `.agentkernel/graph.jsonl` | register `graph_*` tools backed by this file |
| `mcp_log_dir` | `mcp_logs/` | one stderr log file per configured MCP server |
| `improvements_dir` | `.agentkernel/improvements` | where `improve` writes reflection notes |
| `sandbox` / `sandbox_image` / `sandbox_network` | `local` / `python:3.12-slim` / `none` | execution boundary: `local` or `docker`, plus the container image and network |
| `enable_spawn` / `spawn_max_depth` | `False` / `2` | register the `spawn` sub-agent tool and bound its recursion |
| `judge_model` / `eval_threshold` / `eval_rubric` | `None` / `0.6` / `None` | model that scores evals (defaults to `model`), the pass cutoff, and a default rubric |

MCP servers are declared separately as `[[mcp_servers]]` tables (see [MCP](#mcp-mcp) below). Each server supports an optional `timeout` (request seconds) and emits its stderr to `mcp_log_dir/<name>.log`.

---

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │                   Agent                      │
                │  (the loop; orchestrates everything below)   │
                └───┬───────────┬───────────┬──────────┬───────┘
                    │           │           │          │
          ┌─────────▼──┐  ┌─────▼─────┐ ┌───▼──────┐ ┌─▼──────────┐
          │  Provider  │  │   Tool    │ │ Context  │ │  Approver  │
          │  (adapter) │  │ Registry  │ │ Manager  │ │  + Sandbox │
          └─────┬──────┘  └─────┬─────┘ └──────────┘ └────────────┘
                │               │
        Anthropic/OpenAI/   builtin tools
        local endpoint      (files, shell)

  Cross-cutting: Config (injected), Telemetry (records every turn)
```

**One turn:**

1. The `ContextManager` returns the message window within budget (compacting if needed).
2. The provider adapter translates canonical → wire, calls the API, and translates the reply back to a canonical `CompletionResponse`.
3. The assistant message is appended. No tool calls → the run ends and returns the final text.
4. For each tool call: validate args → check approval → execute → produce a `ToolResult`.
5. All results are appended as one tool-role message, paired to their call ids.
6. Telemetry records the turn. Loop.

### Canonical types ([`types.py`](agentkernel/types.py))

`Message`, `ToolCall`, `ToolResult`, `Usage`, `CompletionResponse` — stdlib dataclasses that are the lingua franca of the kernel. Nothing outside a provider adapter speaks a provider's native format.

### Providers ([`providers/`](agentkernel/providers))

Hand-written `httpx` adapters for **Anthropic** (Messages API), **OpenAI** (Chat Completions), and **local** (OpenAI-compatible: Ollama, vLLM, LM Studio). Each adapter:

- translates canonical messages/tools to the provider's exact wire shape and back,
- handles the **tool-result pairing** fan-out (Anthropic: all results in one `user` message of `tool_result` blocks; OpenAI: one `role:"tool"` message per result),
- reports cache read/write token counts where available,
- applies cache markers on the stable prefix (Anthropic `cache_control: ephemeral`).

Translation is implemented as **pure functions** separate from the HTTP call, which is what makes adapter behavior testable offline. The adapters share one `httpx` transport ([`providers/_http.py`](agentkernel/providers/_http.py)) that retries transient failures (timeouts and `429`/`5xx`), honoring a server `Retry-After` header (bounded) when present, and raises `ProviderError` only once retries are exhausted.

### Tool system ([`tools/`](agentkernel/tools))

A `ToolSpec` carries a JSON-Schema parameter definition, a handler, and flags (`requires_approval`, `mutates`, `runs_code`). The `ToolRegistry` validates arguments against the schema (validation failures become error results, not executions) and dispatches. Builtin tools:

| Tool | Flags |
|---|---|
| `read_file(path)` | read-only |
| `list_dir(path)` | read-only |
| `find_files(pattern, path?)` | read-only — glob search (`**/*.py`), skips noise dirs |
| `search_text(pattern, glob?, …)` | read-only — regex grep → `path:line: text` |
| `file_info(path)` | read-only — size / type / mtime / line count |
| `write_file(path, content)` | mutates, requires approval |
| `edit_file(path, old, new, replace_all?)` | mutates, requires approval — exact-substring replace |
| `rollback()` | restores files to their pre-edit state (only when `checkpoints = true`) |
| `bash(command)` | runs code, mutates, requires approval |

File and search tools confine paths to the working directory (rejecting `..` escapes and absolute paths outside the root); `bash` runs inside the sandbox boundary. See [`examples/`](examples) for a playground project, ready-to-paste prompts, and a scored eval suite that exercise these tools.

### Context management ([`context/`](agentkernel/context))

Per-message token accounting, a budget (`provider.context_window − output_reserve`), and **compaction**: when the budget is exceeded, the oldest completed turns collapse into one synthetic summary while the most recent turns are kept verbatim. Compaction never splits an open tool-call/result pair, and the system prompt can never be lost (it lives in the cacheable prefix, not the message list). The summarizer is pluggable; the default is a deterministic structural summary.

### Approval & sandbox ([`approval/`](agentkernel/approval))

`Approver` implementations (`CliApprover`, `AutoApprover`) apply a shared policy. Two execution boundaries sit behind the `Sandbox` protocol:

- **`LocalSandbox`** (default) — a subprocess confined to the working directory, with a scrubbed environment and a **real** timeout that kills the whole process tree. Convenient, but cwd-scoped, not a security jail (a command using absolute paths can still reach the host).
- **`DockerSandbox`** (`sandbox = "docker"`) — one long-lived container per project. The working directory is bind-mounted; by default there is **no network**, a separate filesystem, and bounded memory/CPU/PIDs, so a command can't reach the host or the network. The Docker CLI is driven through an injectable runner, so the argv/lifecycle are unit-tested without a daemon. Use this to run untrusted tasks.

### Telemetry ([`telemetry.py`](agentkernel/telemetry.py))

One JSONL file per session. Each turn records tokens (input/output/cache), estimated cost (from a per-model price table; unknown models log tokens with `null` cost), tool-call outcomes, stop reason, and any compaction event. **Redaction is the default** — tool arguments are logged as a hash + length, never raw; file contents never enter a record. `--verbose-trace` opts into raw arguments for local debugging.

### MCP ([`mcp/`](agentkernel/mcp))

A hand-written [Model Context Protocol](https://modelcontextprotocol.io) client (JSON-RPC 2.0 over stdio — no SDK dependency) connects to MCP servers, discovers their tools, and registers each as an ordinary `ToolSpec`. The registry and loop are **completely unchanged** — an MCP-backed tool and a native builtin register identically. Read-only tools (advertising `readOnlyHint`) skip the approval gate; everything else is gated by default. A transport or protocol fault becomes an error result, never a raise.

Each server gets its own stderr log file under `mcp_log_dir` for easy debugging, and an optional `timeout` controls per-request patience:

```toml
[[mcp_servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
timeout = 30
```

On Windows, point `command` at the actual executable (e.g. `npx.cmd`) since the client launches the process directly without a shell.

### Higher-level capabilities (built on the kernel)

These are implemented on top of the kernel using the three primitives — a tool, a context injection, or a run parameter — never by changing the loop:

- **Profiles** ([`profiles.py`](agentkernel/profiles.py)) — a run parameter `(system_prompt, tool_filter, model_override, rubric)` loaded from `profiles/<name>.toml`. The loop honors `system_prompt` and `tool_filter`; CLI `--profile` sets the active profile, and a profile's `model_override` or `rubric` override the defaults.
- **Skills** ([`skills.py`](agentkernel/skills.py)) — [Anthropic-style](https://github.com/anthropics/skills) `SKILL.md` folders (YAML frontmatter `name`/`description` + body + bundled files) discovered from `skills_dir`, with **progressive disclosure**: only a name+description catalog sits in the (stable, assembled-once) prefix; the model loads a skill's full body + file listing on demand via the `use_skill` tool. A skill can also be *pinned* (`skills = [...]`, `--skill <name>`, or `/skill <name>`) to force its body into the prefix. Loose `.md`/`.toml` skills still work.
- **Memory** ([`memory.py`](agentkernel/memory.py), [`semantic_memory.py`](agentkernel/semantic_memory.py)) — a `MemoryStore` loaded before a run and saved after; ships with in-memory, JSONL, and SQLite/FTS5 stores. Enable with `memory_store`. The SQLite notebook supports optional `semantic_search` via an OpenAI-compatible embedding endpoint, cosine-ranked recall, and a standard-library-only approximate LSH index (`semantic_search_lsh_bits`) for large notebooks. The `reindex_memory` tool backfills embeddings when a notebook is promoted to semantic recall. Recall can optionally be re-ranked by note age and recall frequency (`memory_recency_weight`/`memory_importance_weight`/`memory_half_life_days`) using the `access_count` metadata the notebook already tracks; with the default zero weights, ordering is pure relevance. A single notebook can also be partitioned into namespaces with `memory_scope` (e.g. `auto` for a per-project namespace): recall returns global notes plus the active namespace's, and new notes are stamped with it — keeping one "global brain" file while letting project-specific facts stay project-local.
- **Knowledge graph** ([`knowledge.py`](agentkernel/knowledge.py)) — a file-backed triple store exposed purely as `graph_add`, `graph_query`, `graph_neighbors`, `graph_path`, and `graph_stats` tools (`enable_graph = true`). The kernel keeps no graph state.
- **Loops** ([`loops.py`](agentkernel/loops.py)) — [loop-engineering](https://signals.forwardfuture.ai/loop-library/) workflows: `agentkernel loop` re-runs the agent on a loop's prompt until a stopping condition (a success shell-check and/or an N-in-a-row streak), following **action → check → iterate → stop**. Loops load from TOML or from a skill body (`--skill`), and the success check runs in the sandbox so a loop can verify its own work (e.g. "fix until `pytest` is green").
- **Self-improvement** ([`improvement.py`](agentkernel/improvement.py)) — `agentkernel improve` or the REPL's `/improve` reads a session trace and asks the model for one concrete rule, written to `improvements_dir`. This is why telemetry exists from turn one.
- **Sub-agents** ([`subagent.py`](agentkernel/subagent.py)) — `enable_spawn = true` registers a `spawn` tool so the model can delegate a self-contained subtask to a focused child `Agent` (own context, optional system prompt and tool subset), depth-limited by `spawn_max_depth`. Built on the loop's re-entrancy; no loop change.
- **Evaluators** ([`evaluation.py`](agentkernel/evaluation.py)) — `agentkernel eval --suite suite.toml` runs each case through the agent, then a judge model scores the answer against a rubric (0–1, pass/fail). Use `--case <glob>` to filter cases and `--output/-o report.json` to write a machine-readable report. Aggregates to pass-rate and mean score; exits non-zero unless every case passes, so it doubles as a CI gate and a way to compare models.
- **Budget guard** ([`budget.py`](agentkernel/budget.py)) — per-run cost/token ceilings (`max_cost_usd`, `max_input_tokens_per_run`) that stop a run cleanly.

---

## Project layout

```
agentkernel/
  types.py              # canonical data types
  config.py             # configuration loading + layered discovery
  paths.py              # agent home / project root resolution (run anywhere)
  telemetry.py          # JSONL traces + cost table
  agent.py              # the loop
  providers/            # base protocol + anthropic / openai / local adapters
  tools/                # ToolSpec, ToolRegistry, builtin file & shell tools
  context/              # accounting, compaction, shared truncation
  approval/             # Approver, Sandbox, policies
  mcp/                  # MCP stdio client; registers remote tools as ToolSpecs
  budget.py             # per-run cost/token guard
  progress.py           # per-turn REPL status lines
  profiles.py           # run-parameter profiles (Phase 5)
  skills.py             # Anthropic-style SKILL.md skills (progressive disclosure)
  memory.py             # pre/post-run MemoryStore and notebook backends (Phase 3)
  semantic_memory.py    # dense embeddings + cosine-ranked recall over SQLite
  semantic_index.py     # standard-library LSH approximate vector index
  embeddings.py         # OpenAI-compatible embedding provider protocol
  knowledge.py          # triple store exposed as tools (Phase 6)
  improvement.py        # trace -> improvement rule (Phase 7)
  subagent.py           # spawn tool: delegate to a child Agent
  evaluation.py         # eval harness: judge-scored runs
  loops.py              # loop-engineering runner (run-until-condition)
  cli.py                # REPL + run/improve/eval/loop/tui/new entry points
  tui/                  # curses interactive terminal UI (agentkernel tui)
skills/                 # bundled starter skills (auto-discovered)
profiles/               # bundled run profiles (reviewer, coder, researcher, …)
loops/                  # bundled loop workflows (until-tests-pass, …)
templates/              # annotated skeletons + `agentkernel new` scaffolding
examples/               # playground project, eval suite, sample skill/loop
tests/                  # offline suite (FakeProvider-driven)
```

### Bundled content

The skills, profiles, and loop machinery ships with a starter library so it's
useful out of the box, plus templates to author your own:

- **[`skills/`](skills)** — `code-review`, `debug-triage`, `write-tests`, `refactor`, `commit-and-pr`, `security-review`. Discovered automatically; pin one with `--skill <name>` or load on demand via `use_skill`.
- **[`profiles/`](profiles)** — `reviewer` (read-only + rubric), `coder`, `researcher` (mutations denied), `planner` (plan-only), `safe` (minimal). Run with `--profile <name>`.
- **[`loops/`](loops)** — `until-tests-pass`, `until-lint-clean`, `until-typecheck-clean`, `until-build-succeeds`, `review-and-fix`. Run with `loop --file loops/<name>.toml`.
- **[`templates/`](templates)** — annotated skeletons for each, plus `agentkernel new skill|profile|loop|eval <name>` to scaffold a fresh one with the name filled in.

---

## Testing

```bash
uv run pytest
```

The suite is **fully offline** — a `FakeProvider` returns scripted responses, so the loop, registry, compaction, approval path, and adapter translation are all exercised with zero network calls. No test makes a network request.

---

## The seam principle

The kernel proves its design by adding every capability through one of three primitives — a tool, a context injection, or a run parameter — **without changing the loop or the registry**:

- **MCP** — an MCP client registers each remote tool as a `ToolSpec` (a tool).
- **Knowledge graph** — `graph_add`/`graph_query` are ordinary registered tools.
- **Skills** — a `ContextSource` contributes system-prompt text (a context injection).
- **Memory** — pre-run load and post-run save hooks around `run`, plus optional recall injected before each user message (context injection). A model-controlled notebook (`remember`/`recall`/…) and **self-curation** ([curation.py](agentkernel/curation.py)): `agentkernel memory extract` distils a session transcript into durable facts (deduped against existing notes), and `agentkernel memory consolidate` has the model merge related notes and supersede outdated ones. Schedule it via `cron` to keep memory tidy automatically.
- **Profiles** — `run()` accepts a `profile` parameter (a run parameter).
- **Sub-agents** — the `spawn` tool builds a child `Agent` from inside a handler (a tool, on top of re-entrancy).
- **Self-improvement** — reads the telemetry the kernel has emitted since turn one.
- **Evaluators** — a harness that runs the agent and judge-scores the output; no kernel change.

Each lands at the edge. The loop in [`agent.py`](agentkernel/agent.py) still reads like the design's pseudocode.

---

## Scope & contributing

This repository is the **kernel only**. Contributions should preserve the design principles above: keep the kernel small, keep provider details inside adapters, return errors as results, keep the cacheable prefix stable, gate mutations, and never log secrets or raw file contents. New features belong on top of the kernel as tools, context injections, or run parameters — not inside the loop.
