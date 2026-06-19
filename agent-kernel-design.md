# Agent Kernel — Software Design Document

**Status:** Draft v1 — implementation spec
**Audience:** Claude Code (implementer) and the project owner
**Scope:** The agent *kernel* only. This is Phase 0 plus the cost-control primitives that must live in the kernel from day one. Everything past the kernel (MCP, skills, profiles, memory, the knowledge graph, self-improvement) is explicitly out of scope here and is designed for only as an extension seam.

Project working name: **agentcore** (rename freely; this name is used for the package throughout).

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

Each seam is specified in §13. Implement the interface, not the feature.

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

Defined in `agentcore/types.py`. Use stdlib `dataclasses` + `typing`. These types are the lingua franca; nothing outside an adapter speaks a provider's native format.

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

Defined in `agentcore/providers/`.

### 5.1 The protocol

```python
from typing import Protocol
from agentcore.types import Message, CompletionResponse
from agentcore.tools import ToolSpec

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

Defined in `agentcore/tools/`.

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

In `agentcore/tools/builtin/`:

| Tool | File | Flags |
|---|---|---|
| `read_file(path)` | `files.py` | read-only |
| `write_file(path, content)` | `files.py` | `mutates`, `requires_approval` |
| `list_dir(path)` | `files.py` | read-only |
| `bash(command)` | `shell.py` | `runs_code`, `mutates`, `requires_approval` |

`read_file`/`list_dir`/`write_file` confine paths to the configured working directory (reject `..` escapes and absolute paths outside the root). `bash` runs inside the sandbox boundary (§10.3). Large file reads are truncated with a clear marker; the truncation policy is shared with §9.

---

## 7. The agent loop

Defined in `agentcore/agent.py`.

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

Defined in `agentcore/context/`.

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

Defined in `agentcore/approval/`.

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

Defined in `agentcore/config.py`.

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
    log_dir: str = ".agentcore/traces"
```

Precedence: explicit constructor args > environment variables (`AGENTCORE_*`) > config file (`agentcore.toml`) > defaults. API keys come **only** from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) and are never written to the config file or the trace.

---

## 12. Telemetry

Defined in `agentcore/telemetry.py`. Writes one JSONL file per session under `config.log_dir`.

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

## 13. Extension seams (design only — do not implement)

Leave these interfaces in place so later phases plug in without reshaping the kernel.

- **MCP (Phase 2):** an MCP client discovers remote tools and registers each as a `ToolSpec` (its `handler` issues the MCP call). No loop or registry change required — that's the test of whether §6 is right.
- **Skills / AGENTS.md (Phase 4):** a `ContextSource` interface with `system_additions() -> str` and `available_skills() -> list[str]`, consulted when assembling the system prompt. Must not disturb prefix stability (§9.3) — skill text joins the cacheable prefix, assembled once per run.
- **Profiles (Phase 5):** `Profile = (name, system_prompt, tool_filter, model_override, rubric)`. The loop already accepts `profile` in `run()`; v1 ignores everything but may apply `tool_filter`/`system_prompt` if trivially present. An evaluator is a profile whose final output is a structured score.
- **Memory (Phase 3):** `MemoryStore` with `load(context) -> list[Message]` (pre-run) and `save(session_trace)` (post-run). The Agent calls these around `run` if a store is configured.
- **Sub-agents:** already enabled by re-entrancy (§7). A future `spawn` tool constructs a child `Agent` and calls `run`.

---

## 14. Module layout

```
agentcore/
  __init__.py
  types.py              # §4 canonical types
  config.py             # §11
  telemetry.py          # §12
  agent.py              # §7 the loop
  providers/
    base.py             # §5 Provider protocol
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
    sandbox.py          # LocalSandbox (+ DockerSandbox TODO)
  cli.py                # REPL entry point
tests/
  fakes.py              # FakeProvider, scripted responses
  test_loop.py
  test_registry.py
  test_context.py
  test_tool_result_flow.py
  test_approval.py
pyproject.toml
agentcore.toml.example
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
| **M4 — Telemetry + CLI** | JSONL traces, cost table, REPL | `agentcore` starts a chat; each turn appends a redacted trace record |

Ship M0–M4 in order. Each is independently testable. Do not start any out-of-scope phase until M4 is green.

---

## 17. Non-goals and reminders

- No agent framework dependency. The loop is hand-written and small.
- No provider-specific types outside `providers/`.
- No feature from §1.3. Leave the seam; move on.
- Secrets only from env; never logged raw; never written to config or traces.
- Keep the cacheable prefix stable — re-sorting tools silently destroys cache hit-rate and is hard to notice without the telemetry in §12.
