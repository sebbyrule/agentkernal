# Agent Kernel — Software Design Document

**Status:** Draft v1 — implementation spec
**Audience:** Claude Code (implementer) and the project owner
**Scope:** The agent *kernel* only. This is Phase 0 plus the cost-control primitives that must live in the kernel from day one. Everything past the kernel (MCP, skills, profiles, memory, the knowledge graph, self-improvement) is explicitly out of scope here and is designed for only as an extension seam.

Project working name: **agentkernel** (rename freely; this name is used for the package throughout).

---

## 1. Purpose and scope

### 1.1 What this kernel is

A minimal, dependency-light core that runs the agent loop: it sends a conversation plus tool definitions to a language model, parses any tool calls out of the response, executes them through a registry, feeds the results back, and repeats until the model produces a final answer. It is provider-agnostic, tool-agnostic, and testable without network access.

Everything a user might call a "feature" — web search, file editing, image generation, project memory, profiles — is built *on top of* this kernel as a tool, a piece of injected context, or a parameterization of a run. None of those belong inside the kernel.

### 1.2 In scope (build this)

1. Canonical, provider-independent data types for messages, tool calls, tool results, and usage.
2. A provider abstraction with adapters for Anthropic, OpenAI, and a local/OpenAI-compatible endpoint (Ollama/vLLM).
3. A tool system: tool definitions with JSON-Schema parameters, a registry, argument validation, and dispatch.
4. The agent loop, including parallel tool calls, a max-iteration guard, and re-entrancy (a tool may spawn a nested run).
5. Context management: token accounting, a context budget, and compaction when the budget is approached.
6. Prompt/response caching support: a stable cacheable prefix and provider cache markers.
7. An approval gate and a pluggable sandbox/execution boundary for mutating and code-running tools.
8. Configuration loading (file + environment).
9. Structured per-turn telemetry written as a session trace.
10. A simple interactive CLI (REPL) entry point.

### 1.3 Out of scope (do NOT build; leave seams only)

| Capability | Belongs to | Seam to leave |
|---|---|---|
| MCP client | Phase 2 | External tools register into the same `ToolRegistry` |
| SKILL.md / AGENTS.md loading | Phase 4 | A context-injection hook before the loop |
| Profiles & evaluators | Phase 5 | The loop accepts a `Profile` parameter |
| Memory system | Phase 3 | Pre-run load hook + post-run save hook |
| Knowledge graph (graphify) | Phase 6 | Just another registered tool/MCP server |
| Self-improvement loop | Phase 7 | Requires the telemetry built here first |
| Web UI | Not planned | CLI only; keep all logic UI-independent |

Each seam is specified in §13. Implement the interface, not the feature. *(Status: the kernel is complete and every seam in this table has since been built on top of it — see §13. The rule that mattered held: none of them required changing the loop.)*

---

## 2. Design principles

These are binding. When a decision is ambiguous, resolve it in favor of the principle.

1. **The kernel stays small and dependency-light.** No agent frameworks (no LangChain, LlamaIndex, etc.). The loop is the product; frameworks would hide it.
2. **Everything is a tool, a context injection, or a run parameter.** If a proposed addition isn't one of those three, it probably doesn't belong in the kernel.
3. **One canonical message format.** Provider quirks (Anthropic content blocks vs. OpenAI `tool_calls` arrays) are normalized inside adapters and never leak into the loop or the registry.
4. **Errors become tool results, not exceptions.** A tool that fails returns a `ToolResult` with `is_error=True`. The loop keeps going so the model can recover. Only unrecoverable kernel faults raise.
5. **Testable without a network.** A `FakeProvider` returns scripted responses. The full loop, registry, compaction, and approval path are unit-testable offline.
6. **The cacheable prefix is stable.** System prompt + tool definitions are assembled once and not reordered between turns, so prompt caching actually hits.
7. **Every mutation is gated.** Tools that write files, run shell, or hit the network pass through the `Approver` before executing.
8. **Telemetry from turn one.** Every turn logs messages, tool calls, results, token usage, and cost. This is the substrate for later cost analysis and self-improvement; it is not optional polish.

---

## 3. High-level architecture

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
        local endpoint      (files, shell, …)

  Cross-cutting: Config (injected), Telemetry (records every turn)
```

**Data flow for one turn:**

1. Agent asks `ContextManager` for the message list within budget (compacting if needed).
2. Agent calls `Provider.complete(messages, tools, ...)`.
3. Provider adapter translates canonical → wire format, calls the API, translates the response back to a canonical `CompletionResponse` (text + tool calls + usage).
4. Agent appends the assistant message. If there are no tool calls, the run ends and the final text is returned.
5. For each tool call: validate args → check approval → execute via registry → produce a `ToolResult`.
6. All results are appended as tool-result messages (paired to their call ids).
7. Telemetry records the whole turn. Loop back to step 1.

---

## 4. Canonical data types

Defined in `agentkernel/types.py`. Use stdlib `dataclasses` + `typing`. These types are the lingua franca; nothing outside an adapter speaks a provider's native format.

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class ToolCall:
    id: str                     # unique within the run; provider id or generated
    name: str
    arguments: dict[str, Any]   # already parsed from JSON

@dataclass
class ToolResult:
    call_id: str                # pairs back to ToolCall.id
    content: str                # text shown to the model
    is_error: bool = False
    data: Optional[dict] = None # optional structured payload (not sent to model unless serialized)

@dataclass
class Message:
    role: Role
    content: str = ""                          # plain text portion
    tool_calls: list[ToolCall] = field(default_factory=list)   # assistant turns only
    tool_results: list[ToolResult] = field(default_factory=list)  # tool turns only
    # Bookkeeping:
    cacheable: bool = False     # marks a stable prefix boundary (see §9.3)
    token_estimate: Optional[int] = None

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

@dataclass
class CompletionResponse:
    message: Message            # the assistant message (text and/or tool_calls)
    usage: Usage
    stop_reason: str            # "end_turn" | "tool_use" | "max_tokens" | provider-specific
    raw: Any = None             # untouched provider response, for debugging only
```

Design notes:
- A single assistant turn can contain **both** text and one or more `tool_calls`. Handle the mixed case.
- `data` on `ToolResult` lets a tool return structured output for kernel use (e.g., a future memory hook) without forcing it into the model-visible `content`.

---

## 5. Provider abstraction

Defined in `agentkernel/providers/`.

### 5.1 The protocol

```python
from typing import Protocol
from agentkernel.types import Message, CompletionResponse
from agentkernel.tools import ToolSpec

class Provider(Protocol):
    name: str
    context_window: int          # total token capacity of the selected model

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int,
        temperature: float = 1.0,
        system: str | None = None,
    ) -> CompletionResponse: ...
```

### 5.2 Adapter responsibilities

Each adapter (`anthropic.py`, `openai.py`, `local.py`) must:

1. **Translate canonical → wire.** Render `Message` lists and `ToolSpec` lists into the provider's exact schema.
   - Anthropic: assistant `tool_use` content blocks; tool results as `user` messages containing `tool_result` blocks keyed by `tool_use_id`.
   - OpenAI: assistant `tool_calls` array; tool results as `role: "tool"` messages keyed by `tool_call_id`.
   - local: OpenAI-compatible; same as OpenAI but configurable `base_url` and optional auth.
2. **Translate wire → canonical.** Parse the response into one `CompletionResponse`, parsing tool-call argument JSON into `dict` (handle malformed JSON by returning a tool-call with empty args and letting validation in §6 surface the error).
3. **Report usage**, including cache read/write token counts where the provider exposes them.
4. **Apply cache markers** on the stable prefix (§9.3). For Anthropic, set `cache_control` breakpoints on the system block and the last tool definition. For providers without explicit caching, this is a no-op.
5. **Never leak.** No provider-specific dict ever escapes the adapter except inside `CompletionResponse.raw`.

### 5.3 Implementation latitude

The owner previously discussed LiteLLM as a way to avoid hand-writing every provider. That is acceptable **behind this interface**: an adapter may delegate to LiteLLM internally, but the `Provider` protocol and the canonical types remain the contract. Do not let a third-party request/response object pass into the loop.

---

## 6. Tool system

Defined in `agentkernel/tools/`.

### 6.1 Tool definition

```python
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class ToolSpec:
    name: str
    description: str                 # model-facing; write it like a prompt
    parameters: dict[str, Any]       # JSON Schema (draft 2020-12) object
    handler: Callable[[dict], "ToolResult"]
    requires_approval: bool = False  # gate before execution (see §10)
    mutates: bool = False            # writes files / external state
    runs_code: bool = False          # executes shell or arbitrary code → sandbox + approval
    category: str = "general"
```

### 6.2 Registry

```python
class ToolRegistry:
    def register(self, spec: ToolSpec) -> None: ...
    def specs(self) -> list[ToolSpec]: ...                  # for the provider
    def validate(self, call: ToolCall) -> Optional[str]:    # returns error string or None
        ...                                                 # JSON-Schema validate call.arguments
    def execute(self, call: ToolCall) -> ToolResult: ...    # dispatch by name
```

Rules:
- `validate` checks `call.arguments` against the tool's `parameters` schema. On failure, the loop returns a `ToolResult(is_error=True, content="<validation error>")` to the model **instead of** executing — the model can correct itself.
- `execute` dispatches to the handler. A handler raising an exception is caught and converted to `ToolResult(is_error=True, content=<traceback summary>)`.
- The registry is agnostic about a tool's origin. A native tool and (later) an MCP-backed tool register identically. This is the seam for Phase 2 — do not special-case anything.

Use the `jsonschema` package for validation. This is the one validation dependency the kernel may take.

### 6.3 Builtin tools (kernel ships these)

In `agentkernel/tools/builtin/`:

| Tool | File | Flags |
|---|---|---|
| `read_file(path)` | `files.py` | read-only |
| `write_file(path, content)` | `files.py` | `mutates`, `requires_approval` |
| `list_dir(path)` | `files.py` | read-only |
| `bash(command)` | `shell.py` | `runs_code`, `mutates`, `requires_approval` |

`read_file`/`list_dir`/`write_file` confine paths to the configured working directory (reject `..` escapes and absolute paths outside the root). `bash` runs inside the sandbox boundary (§10.3). Large file reads are truncated with a clear marker; the truncation policy is shared with §9.

---

## 7. The agent loop

Defined in `agentkernel/agent.py`.

```python
class Agent:
    def __init__(self, provider, registry, context, approver, telemetry, config):
        ...

    def run(self, user_input: str, *, profile: "Profile | None" = None) -> str:
        self.context.add(Message(role="user", content=user_input))
        for iteration in range(self.config.max_iterations):
            messages = self.context.window()          # compacted to budget (§9)
            tools = self._tools_for(profile)          # full set, or a subset if a profile is given
            system = self._system_for(profile)

            resp = self.provider.complete(
                messages, tools,
                max_tokens=self.config.max_output_tokens,
                system=system,
            )
            self.context.add(resp.message)
            self.telemetry.record_turn(iteration, resp, messages)

            if not resp.message.tool_calls:
                return resp.message.content           # final answer

            results: list[ToolResult] = []
            for call in resp.message.tool_calls:      # may be >1 (parallel tool use)
                err = self.registry.validate(call)
                if err:
                    results.append(ToolResult(call.id, err, is_error=True))
                    continue
                spec = self.registry.spec(call.name)
                if self._needs_approval(spec) and not self.approver.approve(call, spec):
                    results.append(ToolResult(call.id, "Denied by user.", is_error=True))
                    continue
                results.append(self.registry.execute(call))

            self.context.add(Message(role="tool", tool_results=results))

        return "Stopped: reached max iterations without a final answer."
```

Requirements:
- **Parallel tool calls.** One assistant turn may request several tools. Execute all (sequentially is fine for v1; the structure must allow concurrency later), collect every result, and append them together before the next `complete` call. Never call the model again with a tool call left unanswered — every `ToolCall.id` must have a matching `ToolResult.call_id` in the next request (see §8).
- **Max-iteration guard.** `config.max_iterations` (default 25) prevents infinite loops. Returning the guard message is a normal, logged outcome.
- **Re-entrancy.** `run` must be safe to call from within a tool handler (a tool spawning a sub-agent). Keep all state on the instance/arguments; no module-level mutable state. The sub-agent gets its own `Agent` with its own context but may share the provider and registry.
- **Streaming** is not required for v1. If added, it must not change the loop's contract — accumulate the stream into the same `CompletionResponse`.

---

## 8. Tool result flow (the contract that breaks silently if wrong)

This deserves its own section because mis-pairing results is the most common and most confusing agent bug.

### 8.1 The pairing rule

Every `ToolCall` emitted by the assistant **must** be answered by exactly one `ToolResult` with a matching id, and those results **must** be present in the very next request to the model. Providers reject or misbehave if a tool call is left dangling.

- Anthropic: the assistant message contains `tool_use` blocks, each with an `id`. The reply must be a `user` message whose content is one `tool_result` block per call, each carrying `tool_use_id` equal to that `id`. **All** results for a turn go in a single user message.
- OpenAI: the assistant message contains a `tool_calls` array, each with an `id`. The reply is **one `role: "tool"` message per result**, each with `tool_call_id` equal to that `id`.

The adapter (§5.2) is responsible for this fan-out. The loop always hands the adapter a single canonical tool-role `Message` carrying a list of `ToolResult`; the adapter renders it to whichever shape the provider needs. The loop must not know the difference.

### 8.2 Ordering

Append the assistant message first, then the tool-role message, in that order, every time. Do not interleave a user message between a tool call and its results.

### 8.3 Errors are results

A validation failure, an approval denial, a handler exception, or a tool's own failure all produce `ToolResult(is_error=True, content=<explanation>)`. They are returned to the model, not raised. The model decides whether to retry, change approach, or report the failure to the user. The only things that raise out of the loop are kernel faults (provider unreachable after retries, config invalid).

### 8.4 Large results

Tool output can be huge (a file, a command's stdout). Before a result enters the context:
- Truncate to a configured byte/token cap (`config.max_tool_result_tokens`, default ~4k tokens).
- Replace the removed middle with a marker: `… [truncated N bytes; use a narrower query] …`.
- Keep the structured `data` payload intact (it isn't sent to the model) for any kernel-side consumer.

This truncation is the same mechanism context management uses (§9) and should be shared code.

---

## 9. Context management

Defined in `agentkernel/context/`.

### 9.1 Accounting

Track an estimated token count per `Message`. Prefer real counts from the provider's `usage` when available; otherwise estimate (a cheap heuristic such as chars/4 is acceptable for budgeting — it only needs to be conservative). Store the estimate on `Message.token_estimate`.

### 9.2 Budget and compaction

```
budget = provider.context_window - config.output_reserve   # reserve room for the reply
```

`ContextManager.window()` returns the messages to send. If the running total exceeds `budget`:

1. Always keep, verbatim: the system prompt, the tool definitions (these live in the cacheable prefix, not the message list), and the most recent `config.keep_recent_turns` turns.
2. Compact the oldest non-preserved turns into a single synthetic `assistant` summary message: "Earlier in this session: …". Generate the summary with a cheap model call (configurable model, may differ from the main one) or, as a fallback, a deterministic structural summary (list of tool calls made + files touched). Make the summarizer pluggable.
3. Never compact across a still-open tool-call/result pair. Compaction operates on completed turns only.

Compaction events are logged to telemetry (how many turns/tokens were collapsed).

### 9.3 Caching (the biggest cost lever — build it in v1)

The stable prefix = system prompt + tool definitions. It must be byte-identical across turns so the provider can serve it from cache:

- Assemble the system prompt and tool list once per run and reuse the same object; do not re-sort tools or re-render descriptions per turn.
- The Anthropic adapter sets `cache_control: {type: "ephemeral"}` on the final system block and the final tool definition, marking the prefix boundary.
- Mark the boundary in canonical form with `Message.cacheable` / a prefix flag so adapters that support caching can act and others can ignore it.
- Telemetry records `cache_read_tokens` vs `input_tokens` so cache hit-rate is observable from day one.

---

## 10. Approval and sandbox

Defined in `agentkernel/approval/`.

### 10.1 Approver protocol

```python
class Approver(Protocol):
    def approve(self, call: ToolCall, spec: ToolSpec) -> bool: ...
```

The loop invokes the approver before executing any tool where `requires_approval`, `mutates`, or `runs_code` is true (policy configurable — see §10.2).

### 10.2 Policies

`config.approval_policy` ∈ `{ "always_ask", "auto_allow", "deny_mutations" }`, plus an optional allowlist of command/tool patterns that skip the prompt. Default for v1: `always_ask` for anything `runs_code` or `mutates`, auto-allow read-only tools.

Ship two implementations: `CliApprover` (prints the tool name + arguments, reads y/n from the terminal) and `AutoApprover` (for tests and non-interactive runs, decides from policy without prompting).

### 10.3 Execution boundary

Tools with `runs_code=True` execute inside a `Sandbox`:

```python
class Sandbox(Protocol):
    def run(self, command: str, *, cwd: str, timeout: int) -> tuple[int, str, str]: ...
    # returns (exit_code, stdout, stderr)
```

v1 ships `LocalSandbox` (subprocess confined to the working directory, with a timeout and environment scrubbing). The **target** is `DockerSandbox` (one container per project) — design `Sandbox` so it can be swapped without touching `bash`'s handler. Do not implement Docker now; just leave the interface and a `# TODO: DockerSandbox` stub.

---

## 11. Configuration

Defined in `agentkernel/config.py`.

```python
@dataclass
class Config:
    provider: str = "anthropic"        # "anthropic" | "openai" | "local"
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None        # for local/OpenAI-compatible endpoints
    max_output_tokens: int = 4096
    output_reserve: int = 8192         # budget headroom for the reply
    max_iterations: int = 25
    keep_recent_turns: int = 6
    max_tool_result_tokens: int = 4096
    approval_policy: str = "always_ask"
    working_dir: str = "."
    summarizer_model: str | None = None  # cheap model for compaction; None → structural fallback
    log_dir: str = ".agentkernel/traces"
```

The fields above are the kernel core. The implemented `Config` extends them with options for the §13 seams — sandbox (`sandbox`, `sandbox_image`, `sandbox_network`), budget ceilings (`max_cost_usd`, `max_input_tokens_per_run`), memory (`memory_store`, `enable_memory_tools`, `memory_auto_context`, `semantic_search`, `embedding_*`), skills (`skills_dir`, `skills`), profiles, the knowledge graph, sub-agents (`enable_spawn`), and evals. See [`agentkernel.toml.example`](agentkernel.toml.example) for the full set. Every new field follows the same precedence and coercion rules.

Precedence: explicit constructor args > environment variables (`AGENTKERNEL_*`) > config file (`agentkernel.toml`) > defaults. API keys come **only** from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) and are never written to the config file or the trace.

---

## 12. Telemetry

Defined in `agentkernel/telemetry.py`. Writes one JSONL file per session under `config.log_dir`.

Each turn record:

```json
{
  "ts": "ISO-8601",
  "session_id": "uuid",
  "iteration": 3,
  "model": "claude-sonnet-4-6",
  "input_tokens": 1840,
  "output_tokens": 220,
  "cache_read_tokens": 1600,
  "cache_write_tokens": 0,
  "estimated_cost_usd": 0.0091,
  "tool_calls": [{"name": "bash", "approved": true, "is_error": false}],
  "stop_reason": "tool_use",
  "compaction": null
}
```

- Redact tool arguments and file contents that may contain secrets (log names and a hash/length, not raw values) — make redaction the default, with an opt-in `--verbose-trace` for local debugging.
- Cost is computed from a small per-model price table in config; if a model is unknown, log tokens and leave cost null.
- This trace is the foundation for later cost dashboards and the Phase 7 self-improvement loop. Treat its schema as a stable interface.

---

## 13. Extension seams (now implemented)

These interfaces were specified as seams so later phases could plug in without reshaping the kernel. **All of them are now implemented** — each landed at the edge (a tool, a context injection, or a run parameter) with the loop in `agent.py` unchanged. The seam interfaces are the proof that §6/§7/§9 were right.

- **MCP (Phase 2 — `mcp/`):** a hand-written JSON-RPC-over-stdio client discovers remote tools and registers each as a `ToolSpec` (its `handler` issues the MCP call). No loop or registry change required. Read-only tools (`readOnlyHint`) skip the approval gate.
- **Skills / AGENTS.md (Phase 4 — `skills.py`):** a `ContextSource` with `system_additions() -> list[str]` and `available_skills() -> list[str]`, consulted when assembling the system prompt. Progressive disclosure: only a name+description catalog sits in the cacheable prefix; the model loads a skill's body on demand via `use_skill`. Prefix stability (§9.3) is preserved — skill text is assembled once per run.
- **Profiles & evaluators (Phase 5 — `profiles.py`, `evaluation.py`):** `Profile = (name, system_prompt, tool_filter, model_override, rubric)`, loaded from `profiles/<name>.toml`; the loop honors `system_prompt`/`tool_filter`. An evaluator (`agentkernel eval`) runs cases through the agent and judge-scores each answer against a rubric.
- **Memory (Phase 3 — `memory.py`, `semantic_memory.py`):** `MemoryStore` with `load(session_id) -> list[Message]` (pre-run) and `save(session_id, trace)` (post-run); in-memory, JSONL, and SQLite/FTS5 backends. A separate `NoteStore` powers the `remember`/`recall`/`forget`/`update_memory` tools, with optional dense-embedding semantic recall (`semantic_search`) and a stdlib LSH index. Optional recall is injected before each user message (`memory_auto_context`). The note tools call the store through keyword-only `add`/`search`/`update` so they work identically across the JSONL and SQLite/semantic backends.
- **Knowledge graph (Phase 6 — `knowledge.py`):** a file-backed triple store exposed only as `graph_*` tools; the kernel keeps no graph state.
- **Self-improvement (Phase 7 — `improvement.py`):** `agentkernel improve` reads a session trace and asks the model for one concrete rule — built on the telemetry emitted since turn one.
- **Sub-agents (`subagent.py`):** `enable_spawn` registers a `spawn` tool that constructs a depth-limited child `Agent` and calls `run`, built purely on the loop's re-entrancy (§7).
- **Loops (`loops.py`):** `agentkernel loop` re-runs the agent on a prompt until a stopping condition (a sandboxed success check and/or an N-in-a-row streak).

---

## 14. Module layout

The kernel core is the §4–§12 set; everything below the dividing comment is an
extension seam from §13, built on the kernel without changing the loop.

```
agentkernel/
  __init__.py
  types.py              # §4 canonical types
  config.py             # §11
  telemetry.py          # §12
  agent.py              # §7 the loop
  budget.py             # per-run cost/token guard
  progress.py           # per-turn REPL status lines
  providers/
    base.py             # §5 Provider protocol
    _http.py            # shared httpx transport (retries + Retry-After)
    anthropic.py
    openai.py
    local.py
  tools/
    base.py             # §6 ToolSpec, ToolRegistry
    builtin/
      files.py
      shell.py
  context/
    manager.py          # §9 accounting + compaction
    truncate.py         # shared truncation (§8.4 / §9)
  approval/
    base.py             # §10 Approver, Sandbox protocols
    cli.py              # CliApprover, AutoApprover
    policy.py           # approval policies
    sandbox.py          # LocalSandbox + DockerSandbox
  cli.py                # REPL + run/improve/eval/loop/tui entry points
  # --- §13 extension seams (on top of the kernel) ---
  mcp/                  # MCP stdio client; registers remote tools as ToolSpecs
  skills.py             # Anthropic-style SKILL.md skills (progressive disclosure)
  profiles.py           # run-parameter profiles
  memory.py             # MemoryStore + JSONL/SQLite notebooks; remember/recall tools
  semantic_memory.py    # dense embeddings + cosine-ranked recall over SQLite
  semantic_index.py     # stdlib LSH approximate vector index
  embeddings.py         # OpenAI-compatible embedding provider
  knowledge.py          # triple store exposed as graph_* tools
  improvement.py        # trace -> improvement rule
  subagent.py           # spawn tool: delegate to a child Agent
  evaluation.py         # eval harness: judge-scored runs
  loops.py              # loop-engineering runner (run-until-condition)
  tui/                  # curses interactive terminal UI (agentkernel tui)
tests/
  fakes.py              # FakeProvider, scripted responses
  test_loop.py, test_registry.py, test_context.py,
  test_tool_result_flow.py, test_approval.py, test_http.py, … (offline suite)
pyproject.toml
agentkernel.toml.example
```

---

## 15. Testing strategy

- **FakeProvider** (`tests/fakes.py`): constructed with a scripted list of `CompletionResponse`s. Lets every test drive the loop deterministically with zero network. This is the most important test fixture — build it first.
- **Loop tests:** single-turn (no tools) returns text; one tool call → result → final answer; multiple parallel tool calls all answered; max-iteration guard fires.
- **Tool-result-flow test:** assert that for a turn with N tool calls, the next request contains exactly N results with matching ids and correct ordering, for both the Anthropic and OpenAI adapters (adapter-level, can run offline against the translation functions).
- **Registry tests:** schema validation rejects bad args and returns an error result rather than executing; handler exceptions become error results.
- **Context tests:** compaction triggers at budget, preserves system + recent turns, never splits an open tool pair; truncation marks removed content.
- **Approval tests:** `AutoApprover` honors policy; denial yields an error result and the loop continues.
- **No test may make a network call.** CI runs `pytest` offline.

---

## 16. Milestones (within the kernel)

| Milestone | Delivers | Done when |
|---|---|---|
| **M0 — Skeleton** | types, registry, loop, `FakeProvider`, `read_file`/`write_file`/`list_dir` | Loop runs a scripted multi-tool session end to end in a test |
| **M1 — Real providers + cache** | Anthropic + OpenAI + local adapters; stable cacheable prefix | A real session works against one live provider; cache_read_tokens > 0 on turn 2 |
| **M2 — Context** | accounting, budget, compaction, shared truncation | A session that exceeds budget compacts and continues without losing the system prompt |
| **M3 — Approval + sandbox** | `bash`, `CliApprover`, `LocalSandbox`, policies | Shell runs are gated and confined; denial path tested |
| **M4 — Telemetry + CLI** | JSONL traces, cost table, REPL | `agentkernel` starts a chat; each turn appends a redacted trace record |

Ship M0–M4 in order. Each is independently testable. Do not start any out-of-scope phase until M4 is green.

**Status:** the kernel (M0–M4) is complete, and every §13 extension seam has since been
integrated on top of it (M5: MCP, memory, profiles; M6: skills, knowledge graph,
self-improvement), plus sub-agents, evaluators, loops, a budget guard, and a curses
TUI. The whole suite runs offline.

---

## 17. Non-goals and reminders

- No agent framework dependency. The loop is hand-written and small.
- No provider-specific types outside `providers/`.
- No feature from §1.3. Leave the seam; move on.
- Secrets only from env; never logged raw; never written to config or traces.
- Keep the cacheable prefix stable — re-sorting tools silently destroys cache hit-rate and is hard to notice without the telemetry in §12.

---

## 18. Roadmap

Candidate features, gathered partly by studying mature terminal agents (e.g.
Hermes Agent, Claude Code) and filtered through this kernel's one test: **every
item must land as a tool, a context injection, or a run parameter — never as a
change to the loop in `agent.py`.** Anything that fails that test (a messaging
gateway, a voice stack) belongs in a separate package that *consumes* the kernel,
not inside it; those are listed last so the boundary stays explicit. Ordering
within each group is rough priority. Nothing here is committed.

### 18.1 Safety & trust (highest leverage, smallest surface)

| Idea | Seam | Notes |
|---|---|---|
| **`smart` approval mode** ✅ | approver | **Done** (`approval/risk.py`, `approval_policy = "smart"`). A `RiskJudge` (cheap model, defaults to `summarizer_model` then `model`) classifies each gated call; the approver auto-approves low-risk ones and prompts on high-risk. Conservative: any judge error or unparseable reply falls back to asking. |
| **Secret redaction of tool *output*** ✅ | result post-processing | **Done** (`redaction.py`, `redact_tool_output` config, on by default). Scrubs known token formats (provider key prefixes, PEM blocks, `Authorization` headers, labelled `secret=…` assignments) from tool results at the single §8.4 processing point — before truncation, so a secret can't be split past the cap — and thus before they reach context or traces. Stdlib-only. |
| **Filesystem checkpoints + `rollback`** | tool + run param | Snapshot the working dir (git stash-like, or a copy) before a batch of mutations; a `rollback` tool / `--checkpoints` flag restores it. Makes destructive runs reversible without trusting the model's own cleanup. |

### 18.2 Durable & scheduled execution

| Idea | Seam | Notes |
|---|---|---|
| **Scheduled runs (cron)** | external driver | A small durable scheduler that invokes `Agent.run` on a cron/interval and writes the result to a trace. Pure orchestration around the existing entry point — a `agentkernel cron` CLI, no loop change. |
| **Background / detached runs** | external driver | `agentkernel run --background "…"` that fire-and-forgets a run and records completion. Pairs naturally with cron. |
| **Session store + resume** | memory seam | Promote the JSONL transcript store into addressable sessions: `sessions list/show/resume <id>`. Builds directly on the Phase-3 `MemoryStore`; resume = load that session's messages before the loop. |

### 18.3 Multi-agent

| Idea | Seam | Notes |
|---|---|---|
| **Git worktree isolation for `spawn`** | sub-agent tool | When a spawned child edits code, run it in a throwaway `git worktree` so parallel children don't collide. Extends the existing depth-limited `spawn` (§13) — no new concept, just an isolation flag. |
| **Work-queue (kanban-lite)** | tool + driver | A durable SQLite board of tasks that worker sub-agents claim, complete, or block. Lets a long mission fan out across many bounded runs. Heavier; only worthwhile once `spawn` + sessions land. |

### 18.4 In-session tools (cheap, high-utility)

| Idea | Seam | Notes |
|---|---|---|
| **`todo` tool** | tool | In-session task list the model maintains (add/complete/list). Keeps multi-step plans legible to both the model and the user; pure in-memory state, trivially testable. |
| **`clarify` tool** | tool (over the approver/input seam) | Lets the model ask the user a focused question mid-run instead of guessing, routed through the same CLI input channel the approver uses. No-op (auto-skip) in non-interactive `run`. |

### 18.5 Provider & model layer

| Idea | Seam | Notes |
|---|---|---|
| **Credential pools / key rotation** | provider | Accept a list of keys per provider; rotate on `429`/exhaustion (the `_http` transport already classifies these). Config-only, invisible to the loop. |
| **Reasoning-effort run parameter** | run param | Plumb a `reasoning` level through `run(profile=…)` to providers that support it; ignored by those that don't. |
| **Model router for auxiliary work** | config | Generalize today's single `summarizer_model` into named roles (summarize / judge / classify-risk) so compaction, evals, and `smart` approval can each pick a cheap model. |
| **More first-class adapters** | provider | The OpenAI-compatible `local` adapter already covers many endpoints; add thin named adapters (Gemini, OpenRouter, DeepSeek) only where the wire shape genuinely differs. |

### 18.6 Multimodality

| Idea | Seam | Notes |
|---|---|---|
| **Image input in canonical messages** | canonical types | Extend `Message.content` to allow typed content parts (text + image refs) and teach each adapter to translate them. The biggest single change here — it touches §4 types and every adapter — but still not the loop. Gate behind a capability flag so text-only providers are unaffected. |

### 18.7 Observability & DX

| Idea | Seam | Notes |
|---|---|---|
| **`insights` command** | reads telemetry | Aggregate the JSONL traces into a usage/cost/tool-frequency report. Builds entirely on the trace schema we already treat as stable (§12). |
| **`doctor` command** | standalone | Health check: config validity, provider reachability, sandbox availability, optional deps. Pure diagnostics. |
| **Plugin discovery seam** | tool registration | Auto-import user tool modules from a `plugins/` dir (a top-level `register()` call), mirroring how MCP and builtins already register identically. |
| **Shell completions** | CLI | `agentkernel completion bash`/`zsh`. Trivial DX polish. |

### 18.8 Bundled content & templates (assets, not engine work) — ✅ done

The skills (§13, Phase 4) and profiles (§13, Phase 5) *machinery* is built and
tested, but almost nothing ships through it: there is a single example skill
(`examples/skills/code-review/`), no `profiles/` directory at all (only the
in-code `default`), and no scaffolding for authoring more. The engine works; the
library is empty. This group fills it. None of it touches the kernel — skills are
context injections, profiles are run parameters, templates are inert files.

**Status (implemented):** ships `skills/` (code-review, debug-triage, write-tests,
refactor, commit-and-pr, security-review), `profiles/` (reviewer, researcher,
coder, planner, safe), `loops/` (until-tests-pass, until-lint-clean,
until-typecheck-clean, until-build-succeeds, review-and-fix), `templates/`
(annotated skeletons for each + an MCP block + a tool module), and the
`agentkernel new <skill|profile|loop|eval> <name>` scaffolding command. Each
bundled asset is covered by a load test; the shareable-bundle-format row remains
future work.

| Idea | Seam | Notes |
|---|---|---|
| **A starter skills library** | context injection | Ship a curated set of `SKILL.md` skills in the default `skills/` dir so they're discovered out of the box, not just an example. Candidates: `code-review` (promote the example), `debug-triage`, `write-tests`, `refactor`, `commit-and-pr` (Conventional Commits + PR body), `document`, `research-summarize`, `security-review`. Each stays small with progressive disclosure — name+description in the prefix, body loaded on demand. |
| **A set of named profiles** | run parameter | Populate `profiles/` with ready-to-use parameterizations beyond `default`. Candidates: `reviewer` (read-only tool filter + review system prompt + rubric), `coder` (full tools), `researcher` (read/search/web tools, mutations denied), `planner` (no tools, plan-only), `safe` (minimal locked-down toolset). Each is a small TOML — `(system_prompt, tool_filter, model_override, rubric)`. |
| **A starter loops library** | external driver | Like skills and profiles, the loop runner (`loops.py`, `agentkernel loop`) is built and tested but ships only one example (`examples/loops/until-tests-pass.toml`). Add a curated `loops/` set following action → check → iterate → stop, each with a sandboxed `success_check` and/or `success_streak`: `until-tests-pass` (promote the example), `until-lint-clean`, `until-typecheck-clean`, `until-build-succeeds`, `review-and-fix` (run a review, apply fixes, repeat until clean). Pure TOML over the existing runner — no engine change. |
| **Reusable templates** | DX / scaffolding | A `templates/` directory of annotated skeletons for the things users author repeatedly: `SKILL.md`, a profile TOML, an eval suite, a loop TOML, an `[[mcp_servers]]` block, and a builtin-style tool module. Each is a copy-paste starting point with inline comments explaining every field. |
| **`new` scaffolding command** | CLI / DX | `agentkernel new skill <name>` / `new profile <name>` / `new eval <name>` / `new loop <name>` copies the matching template into place with the name filled in. Turns the templates above into a one-liner; pure file generation, no loop involvement. |
| **A shareable bundle format** | packaging | Optional: a convention for packaging a skill (its `SKILL.md` + bundled files) as a single archive so skills can be shared/installed between projects, mirroring how MCP servers are declared once and reused. Only worth doing once the library above exists and people want to trade skills. |

### 18.9 Explicitly out of the kernel (separate packages)

These are valuable but violate "terminal-only, UI-independent, dependency-light."
They should live in their own packages built *on* the kernel, exactly as §13's
seams predicted — listed here so the boundary is a deliberate choice, not an
oversight.

- **Messaging gateway** (Telegram/Discord/Slack/etc.) — a long-running service that maps platform events to `Agent.run`. Big dependency surface; belongs outside.
- **Voice (STT/TTS)** — an I/O layer around the CLI, not a kernel concern.
- **Curator (skill lifecycle)** — background maintenance of agent-authored skills. Plausible as a sidecar to the skills layer, but it's a service with its own state and schedule, so it sits beside the kernel rather than in it.
