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

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev      # install runtime + dev (pytest) dependencies
```

API keys are read **only** from the environment — never from config files or traces:

```bash
export ANTHROPIC_API_KEY=...     # for provider = "anthropic"
export OPENAI_API_KEY=...        # for provider = "openai"
# local/OpenAI-compatible endpoints (Ollama, vLLM) usually need no key
```

## Quick start

```bash
uv run agentkernel                      # interactive REPL (default)
uv run agentkernel run "your prompt"    # single non-interactive run, prints the answer
uv run agentkernel run --file task.md   # single run from a prompt file
uv run agentkernel improve              # reflect on the latest trace, write a rule note
uv run agentkernel --help               # options
uv run pytest                           # full test suite, offline
```

The REPL keeps conversational context across messages, prints a one-line progress
status per turn, and writes a per-session JSONL trace. It supports slash commands:

```
$ uv run agentkernel
[session trace: .agentkernel/traces/<session-id>.jsonl]
agentkernel REPL - type your message and press enter. Commands: /exit, /clear,
/system, /profile, /skills, /skill, /tools, /trace, /cost.
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
| `/exit` | leave |

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

`build_runtime` wires a provider, the builtin tools inside a `LocalSandbox`, a `CliApprover`, JSONL telemetry, and any configured MCP servers / skills / knowledge-graph tools into an `Agent`. You can also assemble these yourself — every collaborator is injected, nothing is global.

---

## Configuration

Configuration loads from `agentkernel.toml` (see [`agentkernel.toml.example`](agentkernel.toml.example)) with this precedence:

> explicit constructor args **>** `AGENTKERNEL_*` environment variables **>** `agentkernel.toml` **>** defaults

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
| `approval_policy` | `always_ask` | `always_ask` \| `auto_allow` \| `deny_mutations` |
| `approval_allowlist` | `[]` | patterns that skip the approval prompt |
| `working_dir` | `.` | root that file/shell tools are confined to |
| `summarizer_model` | `None` | cheap model for compaction (`None` → structural fallback) |
| `log_dir` | `.agentkernel/traces` | where session traces are written |
| `max_cost_usd` | `None` | per-run cost ceiling; the run stops when exceeded |
| `max_input_tokens_per_run` | `None` | per-run input-token ceiling |
| `profile` / `profile_dir` | `None` / `profiles` | active profile name and where profiles are loaded from |
| `memory_store` / `memory_dir` | `None` / `None` | `file` or `memory` store, and its directory |
| `skills_dir` / `skills` | `skills` / `[]` | skill source directory and the initially-active skill names |
| `enable_graph` / `graph_path` | `False` / `.agentkernel/graph.jsonl` | register `graph_add`/`graph_query` tools backed by this file |
| `improvements_dir` | `.agentkernel/improvements` | where `improve` writes reflection notes |

MCP servers are declared separately as `[[mcp_servers]]` tables (see [MCP](#mcp-mcp) below).

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

Translation is implemented as **pure functions** separate from the HTTP call, which is what makes adapter behavior testable offline.

### Tool system ([`tools/`](agentkernel/tools))

A `ToolSpec` carries a JSON-Schema parameter definition, a handler, and flags (`requires_approval`, `mutates`, `runs_code`). The `ToolRegistry` validates arguments against the schema (validation failures become error results, not executions) and dispatches. Builtin tools:

| Tool | Flags |
|---|---|
| `read_file(path)` | read-only |
| `list_dir(path)` | read-only |
| `write_file(path, content)` | mutates, requires approval |
| `bash(command)` | runs code, mutates, requires approval |

File tools confine paths to the working directory (rejecting `..` escapes and absolute paths outside the root); `bash` runs inside the sandbox boundary.

### Context management ([`context/`](agentkernel/context))

Per-message token accounting, a budget (`provider.context_window − output_reserve`), and **compaction**: when the budget is exceeded, the oldest completed turns collapse into one synthetic summary while the most recent turns are kept verbatim. Compaction never splits an open tool-call/result pair, and the system prompt can never be lost (it lives in the cacheable prefix, not the message list). The summarizer is pluggable; the default is a deterministic structural summary.

### Approval & sandbox ([`approval/`](agentkernel/approval))

`Approver` implementations (`CliApprover`, `AutoApprover`) apply a shared policy. `LocalSandbox` runs shell commands as a subprocess confined to the working directory, with a scrubbed environment (secrets removed) and a **real** timeout that kills the whole process tree. A `DockerSandbox` is left as a swappable stub.

### Telemetry ([`telemetry.py`](agentkernel/telemetry.py))

One JSONL file per session. Each turn records tokens (input/output/cache), estimated cost (from a per-model price table; unknown models log tokens with `null` cost), tool-call outcomes, stop reason, and any compaction event. **Redaction is the default** — tool arguments are logged as a hash + length, never raw; file contents never enter a record. `--verbose-trace` opts into raw arguments for local debugging.

### MCP ([`mcp/`](agentkernel/mcp))

A hand-written [Model Context Protocol](https://modelcontextprotocol.io) client (JSON-RPC 2.0 over stdio — no SDK dependency) connects to MCP servers, discovers their tools, and registers each as an ordinary `ToolSpec`. The registry and loop are **completely unchanged** — an MCP-backed tool and a native builtin register identically. Read-only tools (advertising `readOnlyHint`) skip the approval gate; everything else is gated by default. A transport or protocol fault becomes an error result, never a raise.

Declare servers in `agentkernel.toml`:

```toml
[[mcp_servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

On Windows, point `command` at the actual executable (e.g. `npx.cmd`) since the client launches the process directly without a shell.

### Higher-level capabilities (built on the kernel)

These are implemented on top of the kernel using the three primitives — a tool, a context injection, or a run parameter — never by changing the loop:

- **Profiles** ([`profiles.py`](agentkernel/profiles.py)) — a run parameter `(system_prompt, tool_filter, model_override, rubric)` loaded from `profiles/<name>.toml`. The loop honors `system_prompt` and `tool_filter`.
- **Skills** ([`skills.py`](agentkernel/skills.py)) — markdown/TOML system-prompt fragments discovered from `skills_dir`; active skills contribute to the (still stable, assembled-once) prefix via a `ContextSource`. Toggle them live with `/skill`.
- **Memory** ([`memory.py`](agentkernel/memory.py)) — a `MemoryStore` loaded before a run and saved after; ships in-memory and JSONL-file stores. Enable with `memory_store`.
- **Knowledge graph** ([`knowledge.py`](agentkernel/knowledge.py)) — a file-backed triple store exposed purely as `graph_add`/`graph_query` tools (`enable_graph = true`). The kernel keeps no graph state.
- **Self-improvement** ([`improvement.py`](agentkernel/improvement.py)) — `agentkernel improve` reads a session trace and asks the model for one concrete rule, written to `improvements_dir`. This is why telemetry exists from turn one.
- **Budget guard** ([`budget.py`](agentkernel/budget.py)) — per-run cost/token ceilings (`max_cost_usd`, `max_input_tokens_per_run`) that stop a run cleanly.

---

## Project layout

```
agentkernel/
  types.py              # canonical data types
  config.py             # configuration loading
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
  skills.py             # system-prompt fragments via ContextSource (Phase 4)
  memory.py             # pre/post-run MemoryStore (Phase 3)
  knowledge.py          # triple store exposed as tools (Phase 6)
  improvement.py        # trace -> improvement rule (Phase 7)
  cli.py                # REPL + run/improve entry points
tests/                  # offline suite (FakeProvider-driven)
```

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
- **Memory** — pre-run load and post-run save hooks around `run` (context injection).
- **Profiles** — `run()` accepts a `profile` parameter (a run parameter).
- **Sub-agents** — enabled by the loop's re-entrancy; a tool handler can spawn a child `Agent`.
- **Self-improvement** — reads the telemetry the kernel has emitted since turn one.

Each lands at the edge. The loop in [`agent.py`](agentkernel/agent.py) still reads like the design's pseudocode.

---

## Scope & contributing

This repository is the **kernel only**. Contributions should preserve the design principles above: keep the kernel small, keep provider details inside adapters, return errors as results, keep the cacheable prefix stable, gate mutations, and never log secrets or raw file contents. New features belong on top of the kernel as tools, context injections, or run parameters — not inside the loop.
