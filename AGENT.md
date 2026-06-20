# AGENT.md

Persistent build instructions for Claude Code. Read this before every task. The full specification is in **`agent-kernel-design.md`** — this file tells you *how to work*; that file tells you *what to build*.

## Project

**agentkernel** — a minimal, dependency-light kernel for a general-purpose AI agent. The kernel runs the agent loop (model → tool calls → results → repeat) and nothing more. All planned Phase seams (MCP, skills, profiles, memory, knowledge graph, self-improvement) are now integrated behind clean extension interfaces.

Build strictly to `agent-kernel-design.md`. If something is unspecified, follow the design principles below and leave a `# TODO(owner):` note rather than inventing scope.

## Stack

- **Python 3.11+**, managed with **uv**.
- Dependencies kept minimal. Allowed in the kernel: `httpx` (or official provider SDKs) for API calls, `jsonschema` for tool-arg validation, `tomli`/stdlib `tomllib` for config. The summarizer and cost table need nothing extra.
- **No agent frameworks.** Do not add LangChain, LlamaIndex, CrewAI, Autogen, or similar. The loop is the product; a framework would hide it and defeat the point. If you think you need one, you've misread the scope.
- Tests with **pytest**. No network calls in tests.

## Hard architecture rules

These come from the design doc's principles and are non-negotiable:

1. **One canonical message format** (`agentkernel/types.py`). Provider-specific shapes (Anthropic content blocks, OpenAI `tool_calls`) are translated inside `providers/*` adapters and never appear in the loop, the registry, or the context manager.
2. **Errors become tool results, not exceptions.** Validation failures, approval denials, and handler exceptions return `ToolResult(is_error=True, ...)` so the loop continues and the model can recover. Only kernel faults (provider unreachable after retries, invalid config) raise.
3. **The cacheable prefix is stable.** System prompt + tool definitions are assembled once per run and never reordered between turns. Re-sorting tools silently destroys prompt-cache hit-rate. Treat prefix stability as a correctness property, not an optimization.
4. **Every mutation is gated.** Any tool with `mutates`, `runs_code`, or `requires_approval` passes through the `Approver` before executing. Shell/code runs inside the `Sandbox` boundary, confined to the working directory.
5. **Tool-result pairing is exact.** Every `ToolCall.id` is answered by exactly one `ToolResult.call_id` in the next request, in order, with no message interleaved between an assistant tool call and its results. See design §8 — this is the bug that fails silently.
6. **Telemetry on every turn.** Record tokens (including cache read/write), tool calls, stop reason, and cost. Redact secrets and file contents by default. This trace is a stable interface for later phases — don't treat it as throwaway logging.
7. **No global mutable state.** Inject `provider`, `registry`, `context`, `approver`, `telemetry`, `config` into `Agent`. The loop must be re-entrant (a tool may spawn a sub-agent).

## Coding conventions

- Type-annotate everything. Public protocols and dataclasses get short docstrings stating the contract.
- Small modules, one responsibility each (mirror the layout in design §14). Don't merge files to save lines.
- Prefer `Protocol` + dependency injection over inheritance for `Provider`, `Approver`, `Sandbox`.
- Keep handlers pure where possible; side effects (fs, subprocess, network) live behind the tool/sandbox boundary.
- No clever metaprogramming in the loop. It should read like the pseudocode in design §7.

## Build order

Implement milestones **M0 → M4 in order** (design §16). Each is independently testable; do not jump ahead.

- **M0** — canonical types, `ToolRegistry`, the loop, `FakeProvider`, and the file tools. **Build `FakeProvider` first** — it's how every later test drives the loop offline.
- **M1** — Anthropic, OpenAI, and local adapters + the stable cacheable prefix.
- **M2** — context accounting, budget, compaction, shared truncation.
- **M3** — `bash`, `CliApprover`/`AutoApprover`, `LocalSandbox`, approval policies.
- **M4** — JSONL telemetry, cost table, the REPL entry point.
- **M5** (Phase 2/3/5 seams) — MCP client, `MemoryStore`, and `Profile` integration.
- **M6** (Phase 4/6/7 seams) — Skills / AGENTS.md loading, knowledge-graph tools, and self-improvement reflection.

All documented extension seams are implemented.

## Definition of done (every milestone)

- New code has pytest coverage; the relevant tests in design §15 pass.
- `uv run pytest` passes with **no network access**.
- No provider-specific type escapes `providers/`.
- No new dependency outside the allowed list without a `# TODO(owner): dependency request` note explaining why.
- The cacheable prefix is still assembled once and unreordered.

## How to run

```bash
uv sync
uv run pytest                 # full suite, offline
uv run agentkernel            # start the interactive REPL (M4+)
uv run agentkernel tui        # full-screen curses terminal UI over the same runtime
```

Configuration loads from `agentkernel.toml` (+ `AGENTKERNEL_*` env overrides). API keys come **only** from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) — never read them from, or write them to, the config file or traces.

## Do NOT

- Do not add a web UI or any framework. Terminal-only (the CLI/REPL and the curses TUI are fine); keep all logic UI-independent so any front end is a thin layer over the same runtime.
- Do not hardcode provider-specific behavior in `agent.py`, `tools/`, or `context/`.
- Do not let a tool failure raise out of the loop.
- Do not log raw tool arguments, file contents, or secrets.
- Do not reorder or rebuild the system-prompt/tool prefix per turn.
- Do not expand scope to "make it nicer." Build to the spec; note ideas as `# TODO(owner):`.

## When unsure

Resolve ambiguity in favor of the design principles (smaller kernel, canonical types, errors-as-results, stable prefix, gated mutations, telemetry). If a real decision is needed that the spec doesn't cover, stop and leave a `# TODO(owner):` with the question rather than guessing at scope.
