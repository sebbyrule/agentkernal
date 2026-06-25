"""CLI entry point (design §16, M4).

Wires a live provider, the builtin tools inside a ``LocalSandbox``, a
``CliApprover``, and JSONL telemetry into an ``Agent``.

Modes:
- ``agentkernel`` / ``agentkernel repl`` — interactive REPL with per-turn progress
  and slash commands.
- ``agentkernel run "prompt"`` / ``agentkernel run --file prompt.md`` — single
  non-interactive run, prints the final answer, and exits.

The wiring (``build_runtime``) and the loop (``repl`` / ``run_once``) are
separated so the loop can be tested offline with a scripted provider.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from agentkernel.agent import Agent
from agentkernel.approval import AutoApprover, CliApprover, Sandbox, make_sandbox
from agentkernel.budget import BudgetGuard
from agentkernel.config import Config, resolve_config
from agentkernel.context import ContextManager, ModelSummarizer
from agentkernel.embeddings import EmbeddingError, OpenAIEmbeddingProvider
from agentkernel.knowledge import KnowledgeGraph, make_graph_tools
from agentkernel.mcp import MCPClient, MCPError, load_mcp_servers, register_mcp_servers
from agentkernel.mcp.config import MCPServerConfig
from agentkernel.memory import (
    MemoryStore,
    NoteStore,
    RecallWeighting,
    make_memory_store,
    make_memory_tools,
    make_note_store,
)
from agentkernel.paths import agent_home, global_config_path
from agentkernel.profiles import Profile, load_profile
from agentkernel.progress import ProgressTelemetry
from agentkernel.providers import ProviderError, make_provider
from agentkernel.semantic_memory import SemanticSqliteNoteStore
from agentkernel.skills import DirectorySkillStore, make_skill_tool
from agentkernel.subagent import make_spawn_tool
from agentkernel.telemetry import JsonlTelemetry, NullTelemetry
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import default_tools

_BANNER = (
    "agentkernel REPL - type your message and press enter. Commands: /exit, "
    "/clear, /system, /profile, /skills, /skill, /tools, /trace, /cost, /memory, /improve."
)
_PROMPT = "> "
_EXIT_WORDS = {"exit", "quit", ":q"}


def _make_configured_note_store(config: Config) -> NoteStore:
    """Build the note store named by config (semantic SQLite or JSONL notebook).

    Shared by build_runtime (memory tools) and run_memory (curation), so the
    notebook backend is selected identically in both.
    """
    weighting = RecallWeighting(
        recency_weight=config.memory_recency_weight,
        importance_weight=config.memory_importance_weight,
        half_life_days=config.memory_half_life_days,
    )
    scope = _resolve_memory_scope(config)
    if config.semantic_search:
        try:
            emb_provider = OpenAIEmbeddingProvider.from_config(config)
            notes_path = Path(config.memory_notes_path)
            if notes_path.suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
                notes_path = notes_path.parent / (notes_path.stem + ".semantic.db")
            return SemanticSqliteNoteStore(
                notes_path,
                embedding_provider=emb_provider,
                lsh_bits=config.semantic_search_lsh_bits,
                weighting=weighting,
                scope=scope,
            )
        except EmbeddingError as exc:
            print(f"Warning: semantic search disabled: {exc}", file=sys.stderr)
    return make_note_store(config.memory_notes_path, weighting=weighting, scope=scope)


def _resolve_memory_scope(config: Config) -> str | None:
    """Resolve ``config.memory_scope`` to the active namespace (or ``None`` = off).

    ``"auto"`` derives a stable name from the project directory; an empty value
    disables scoping; any other string is used literally.
    """
    raw = (config.memory_scope or "").strip()
    if not raw:
        return None
    if raw.lower() == "auto":
        return Path(config.working_dir or ".").resolve().name or None
    return raw


def build_runtime(
    config: Config,
    *,
    mcp_servers: list[MCPServerConfig] | None = None,
    verbose: bool = False,
    budget: BudgetGuard | None = None,
    memory: MemoryStore | None = None,
    sandbox: Sandbox | None = None,
    session_id: str | None = None,
) -> tuple[Agent, JsonlTelemetry, list[MCPClient]]:
    """Construct an Agent, its telemetry, and any MCP clients from config.

    Keys come from env. MCP-discovered tools register into the same registry as
    the builtins — the loop never learns they came from elsewhere (design §13).
    Pass ``sandbox`` to own its lifecycle (e.g. close a DockerSandbox container);
    otherwise one is built from config.
    """
    provider = make_provider(config)
    if sandbox is None:
        sandbox = make_sandbox(
            config.sandbox,
            config.working_dir,
            image=config.sandbox_image,
            network=config.sandbox_network,
        )
    # Filesystem checkpoints (§18.1): when enabled, file tools back up files
    # before editing and a `rollback` tool restores them.
    checkpointer = None
    if config.checkpoints:
        from agentkernel.checkpoint import Checkpointer
        from agentkernel.tools.builtin.checkpoint_tool import rollback_tool

        checkpointer = Checkpointer()

    registry = ToolRegistry()
    for spec in default_tools(
        sandbox,
        config.working_dir,
        max_result_tokens=config.max_tool_result_tokens,
        checkpointer=checkpointer,
    ):
        registry.register(spec)
    if checkpointer is not None:
        registry.register(rollback_tool(checkpointer))

    # In-session tools (§18.4): a planning todo list and a clarify-the-user tool.
    if config.enable_todo:
        from agentkernel.tools.builtin.todo import TodoList, todo_tool

        registry.register(todo_tool(TodoList()))
    if config.enable_clarify:
        from agentkernel.tools.builtin.clarify import clarify_tool

        registry.register(clarify_tool())
    if config.enable_kanban:
        from agentkernel.kanban import Board
        from agentkernel.tools.builtin.kanban_tool import kanban_tool

        registry.register(kanban_tool(Board(config.kanban_path)))

    # Plugin tools (§18.7): user-authored tools auto-loaded from plugins_dir.
    if config.enable_plugins:
        from agentkernel.plugins import load_plugin_tools

        def _warn_plugin(path, exc):
            print(f"[plugin load failed: {path.name}] {exc}", file=sys.stderr)

        for spec in load_plugin_tools(
            config.plugins_dir, working_dir=config.working_dir, on_error=_warn_plugin
        ):
            if registry.spec(spec.name) is None:
                registry.register(spec)
            else:
                print(f"[plugin tool skipped: {spec.name!r} already registered]", file=sys.stderr)

    mcp_clients = register_mcp_servers(
        registry, list(mcp_servers or []), log_dir=config.mcp_log_dir
    )

    # Phase 6: expose the knowledge graph as ordinary tools when enabled.
    if config.enable_graph:
        for spec in make_graph_tools(KnowledgeGraph(config.graph_path)):
            registry.register(spec)

    # Phase 3: session transcript memory. Use the injected store if provided,
    # otherwise honor config.memory_store (file/sqlite/memory). Notes are
    # independent and always live in a JSONL notebook at memory_notes_path.
    if memory is None:
        memory = make_memory_store(
            config.memory_store,
            config.memory_dir or ".agentkernel/memory",
        )

    notes: NoteStore | None = None
    if config.enable_memory_tools:
        notes = _make_configured_note_store(config)
        for spec in make_memory_tools(notes, store=memory):
            registry.register(spec)

    # Phase 4: skills contribute a progressive-disclosure catalog via the
    # context source; the model loads a skill's full body on demand with the
    # use_skill tool (registered only when skills exist).
    context_source = DirectorySkillStore(config.skills_dir, active_skills=config.skills)
    if context_source.available_skills():
        registry.register(make_skill_tool(context_source))

    budget_for_context = provider.context_window - config.output_reserve
    summarizer = None
    if config.summarizer_model:
        summarizer = ModelSummarizer(
            make_provider(replace(config, model=config.summarizer_model))
        )
    context = ContextManager(
        budget=budget_for_context,
        keep_recent_turns=config.keep_recent_turns,
        summarizer=summarizer,
    )
    # `smart` approval consults a cheap risk judge before prompting (§18.1).
    risk_judge = None
    if config.approval_policy == "smart":
        from agentkernel.approval.risk import RiskJudge

        judge_model = config.approval_judge_model or config.summarizer_model or config.model
        risk_judge = RiskJudge(make_provider(replace(config, model=judge_model)))
    approver = CliApprover(
        config.approval_policy,
        allowlist=config.approval_allowlist,
        risk_judge=risk_judge,
    )

    # Sub-agent delegation (design §13): the model can spawn focused children.
    # base_specs snapshots the tools BEFORE spawn so spawn isn't self-recursive
    # except through the explicit, depth-limited spawn tools it creates.
    if config.enable_spawn:
        base_specs = registry.specs()

        def _tool_factory(working_dir: str):
            # Rebuild the builtin toolset bound to a worktree dir (§18.3), with its
            # own sandbox so the child's file/shell tools are isolated there.
            wt_sandbox = make_sandbox(
                config.sandbox, working_dir,
                image=config.sandbox_image, network=config.sandbox_network,
            )
            return default_tools(
                wt_sandbox, working_dir,
                max_result_tokens=config.max_tool_result_tokens,
            )

        registry.register(
            make_spawn_tool(
                provider=provider,
                base_specs=base_specs,
                approver=approver,
                config=config,
                max_depth=config.spawn_max_depth,
                tool_factory=_tool_factory,
            )
        )

    # A resumed session reuses its id, so telemetry appends to the same trace and
    # the agent's pre-run memory load (§7) pulls that session's transcript.
    telemetry = JsonlTelemetry(
        config.log_dir, config.model, verbose=verbose, session_id=session_id
    )
    agent = Agent(
        provider,
        registry,
        context,
        approver,
        telemetry,
        config,
        budget=budget,
        memory=memory,
        notes=notes,
        context_source=context_source,
    )
    return agent, telemetry, mcp_clients


def _handle_slash(
    line: str,
    agent: Agent,
    profile: Profile,
    config: Config,
    output_fn: Callable[[str], None],
) -> bool:
    """Process a REPL slash command. Returns True if the line was handled."""
    parts = line.split(None, 1)
    cmd = parts[0][1:]
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q"):
        return False  # signal to exit the loop

    if cmd == "clear":
        agent.context.clear()
        output_fn("[context cleared]")
        return True

    if cmd == "system":
        if not arg:
            output_fn("[system prompt cleared]")
            profile.system_prompt = None
        else:
            profile.system_prompt = arg
            output_fn(f"[system prompt set: {arg[:60]!r}]")
        return True

    if cmd == "profile":
        if not arg:
            output_fn(f"[active profile: {profile.name}]")
            return True
        loaded = load_profile(
            arg,
            search_dirs=[Path(config.profile_dir)] if config.profile_dir else [],
        )
        if loaded is None:
            output_fn(f"[profile not found: {arg}]")
            return True
        profile.name = loaded.name
        profile.system_prompt = loaded.system_prompt
        profile.tool_filter = loaded.tool_filter
        profile.model_override = loaded.model_override
        profile.rubric = loaded.rubric
        output_fn(f"[profile loaded: {loaded.name}]")
        return True

    if cmd == "skills":
        source = agent.context_source
        available = source.available_skills() if source is not None else []
        if not available:
            output_fn("(no skills found)")
            return True
        active = getattr(source, "active_skills", set())
        for name in available:
            output_fn(f"  [{'*' if name in active else ' '}] {name}")
        return True

    if cmd == "skill":
        source = agent.context_source
        if source is None or not hasattr(source, "activate"):
            output_fn("(no skill store)")
            return True
        if not arg:
            output_fn("usage: /skill <name>")
            return True
        state = source.activate(arg)
        output_fn(f"[skill {arg}: {'on' if state else 'off'}]")
        return True

    if cmd == "tools":
        specs = agent.registry.specs()
        if not specs:
            output_fn("(no tools registered)")
        for spec in specs:
            desc = spec.description.splitlines()[0] if spec.description else ""
            output_fn(f"  {spec.name}: {desc}")
        return True

    if cmd == "trace":
        telemetry = agent.telemetry
        path = getattr(telemetry, "path", str(getattr(telemetry, "path", "unknown")))
        output_fn(f"[session trace: {path}]")
        return True

    if cmd == "cost":
        telemetry = agent.telemetry
        total = getattr(telemetry, "cumulative_cost", None)
        usage = getattr(telemetry, "cumulative_usage", None)
        if total is not None:
            output_fn(
                f"[session cost: ${total:.6f} | in={usage.input_tokens} "
                f"out={usage.output_tokens}]"
            )
        else:
            output_fn("[session cost: not tracked]")
        return True

    if cmd == "memory":
        notes = getattr(agent, "notes", None)
        if notes is None:
            output_fn("(memory tools are not enabled)")
            return True
        subparts = arg.split(None, 1)
        sub = subparts[0].lower() if subparts else "list"
        subarg = subparts[1] if len(subparts) > 1 else ""
        if sub == "list":
            limit = int(subarg) if subarg.isdigit() else 20
            all_notes = notes.recent(limit)
            if not all_notes:
                output_fn("(no memory notes)")
                return True
            for n in all_notes:
                tag_part = f" [tags: {', '.join(n.tags)}]" if n.tags else ""
                output_fn(f"  [{n.note_id}] {n.text}{tag_part}")
            return True
        if sub == "delete":
            if not subarg or not subarg.isdigit():
                output_fn("usage: /memory delete <note_id>")
                return True
            removed = notes.forget(note_id=int(subarg))
            if removed:
                output_fn(f"[deleted note {subarg}]")
            else:
                output_fn(f"[note {subarg} not found]")
            return True
        if sub == "export":
            dest = subarg or str(Path(notes.path).with_suffix(".md"))
            path = notes.export(dest)
            output_fn(f"[exported {len(notes.all())} notes to {path}]")
            return True
        if sub == "reindex":
            if hasattr(notes, "reindex_embeddings"):
                count = notes.reindex_embeddings()
                output_fn(f"[reindexed {count} note(s)]")
            else:
                output_fn("(semantic search is not enabled for this notebook)")
            return True
        output_fn("usage: /memory [list [limit]|delete <note_id>|export [path]|reindex]")
        return True

    if cmd == "improve":
        trace_arg = arg.strip() if arg else ""
        trace_path: str | None = trace_arg or None
        if not trace_path:
            telemetry = agent.telemetry
            trace_path = getattr(telemetry, "path", None)
            if trace_path:
                trace_path = str(trace_path)
        try:
            return run_improve(config, trace=trace_path, output_fn=output_fn)
        except Exception as exc:
            output_fn(f"[improve error] {exc}")
            return True

    output_fn(f"[unknown command: /{cmd}]")
    return True


def repl(
    agent: Agent,
    *,
    config: Config | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    stream_fn: Callable[[str], None] | None = None,
) -> int:
    """Read-eval-print chat over one Agent (context persists across messages).

    When ``stream_fn`` is set (and config.stream), model text is written to it as
    it arrives and the final answer is not re-printed."""
    cfg = config or agent.config
    streaming = stream_fn is not None and getattr(cfg, "stream", True)
    output_fn(_BANNER)
    profile = Profile(name="default")
    while True:
        try:
            line = input_fn(_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")  # newline after ^D / ^C
            break
        if not line:
            continue
        if line.lower() in _EXIT_WORDS:
            break
        if line.startswith("/"):
            if not _handle_slash(line, agent, profile, cfg, output_fn):
                break
            continue
        streamed = {"any": False}

        def on_text(text: str, _s=streamed) -> None:
            _s["any"] = True
            stream_fn(text)  # type: ignore[misc]

        try:
            answer = agent.run(
                line, profile=profile, on_text=on_text if streaming else None
            )
        except ProviderError as exc:
            output_fn(f"[provider error] {exc}")
            continue
        if streaming and streamed["any"]:
            stream_fn("\n")  # type: ignore[misc]
        else:
            output_fn(answer)
    return 0


def run_once(
    agent: Agent,
    prompt: str,
    *,
    profile: Profile | None = None,
    output_fn: Callable[[str], None] = print,
    stream_fn: Callable[[str], None] | None = None,
    config: Config | None = None,
) -> int:
    """Execute a single non-interactive turn and print (or stream) the answer."""
    cfg = config or agent.config
    streaming = stream_fn is not None and getattr(cfg, "stream", True)
    streamed = {"any": False}

    def on_text(text: str) -> None:
        streamed["any"] = True
        stream_fn(text)  # type: ignore[misc]

    try:
        answer = agent.run(
            prompt, profile=profile, on_text=on_text if streaming else None
        )
    except ProviderError as exc:
        output_fn(f"[provider error] {exc}")
        return 1
    if streaming and streamed["any"]:
        stream_fn("\n")  # type: ignore[misc]
    else:
        output_fn(answer)
    return 0


# kind -> (template filename, destination path relative to project root).
# {name} in the destination is filled with the asset name.
_NEW_KINDS: dict[str, tuple[str, str]] = {
    "skill": ("SKILL.md", "skills/{name}/SKILL.md"),
    "profile": ("profile.toml", "profiles/{name}.toml"),
    "loop": ("loop.toml", "loops/{name}.toml"),
    "eval": ("eval-suite.toml", "evals/{name}.toml"),
}


def _find_templates_dir(start: Path | None = None) -> Path | None:
    """Locate the templates/ directory: nearest one walking up from ``start``,
    else the copy bundled inside the installed package (so `new` works after a
    global install, not just from a checkout)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / "templates"
        if candidate.is_dir():
            return candidate
    bundled = Path(__file__).parent / "templates"
    return bundled if bundled.is_dir() else None


def run_new(
    kind: str,
    name: str,
    *,
    force: bool = False,
    templates_dir: Path | None = None,
    project_root: Path | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Scaffold a skill/profile/loop/eval from a template (§18.8 roadmap)."""
    if kind not in _NEW_KINDS:
        output_fn(f"[unknown kind: {kind}] choose one of {', '.join(_NEW_KINDS)}")
        return 1
    if not name or any(sep in name for sep in ("/", "\\", "..")) or name.startswith("."):
        output_fn(f"[invalid name: {name!r}] use a simple kebab-case name")
        return 1

    templates = templates_dir or _find_templates_dir()
    if templates is None:
        output_fn("[no templates/ directory found] run this inside an agentkernel project")
        return 1
    template_file, dest_pattern = _NEW_KINDS[kind]
    template_path = templates / template_file
    if not template_path.is_file():
        output_fn(f"[template missing: {template_path}]")
        return 1

    root = project_root or templates.parent
    dest = root / dest_pattern.format(name=name)
    if dest.exists() and not force:
        output_fn(f"[exists: {dest}] pass --force to overwrite")
        return 1

    content = template_path.read_text(encoding="utf-8").replace("{{name}}", name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    output_fn(f"[created {kind}: {dest}]")
    return 0


_PROJECT_CONFIG_TEMPLATE = """\
# agentkernel project config. Global defaults live in ~/.agentkernel/config.toml
# (or $AGENTKERNEL_HOME); keys here override them for this project.
# API keys come ONLY from the environment, never this file.

provider = "anthropic"            # "anthropic" | "openai" | "local"
model = "claude-sonnet-4-6"
# base_url = "http://localhost:1234/v1"   # for provider = "local" (LM Studio, Ollama, vLLM)

approval_policy = "always_ask"    # always_ask | auto_allow | deny_mutations | smart

# Opt into the higher-level capabilities you want:
# enable_memory_tools = true       # remember/recall long-term facts
# enable_spawn = true              # let the model delegate to sub-agents
# skills = ["code-review"]         # pin skills from skills_dir
"""

_GLOBAL_CONFIG_TEMPLATE = """\
# agentkernel user-global config (applies to every project unless overridden by
# a project agentkernel.toml). API keys come ONLY from the environment.

provider = "anthropic"
model = "claude-sonnet-4-6"
approval_policy = "always_ask"
"""


def run_init(
    *,
    target_dir: str = ".",
    global_config: bool = False,
    force: bool = False,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Scaffold a starter config: a project ``agentkernel.toml`` or the global one."""
    if global_config:
        dest = global_config_path(agent_home())
        template = _GLOBAL_CONFIG_TEMPLATE
    else:
        dest = Path(target_dir).resolve() / "agentkernel.toml"
        template = _PROJECT_CONFIG_TEMPLATE
    if dest.exists() and not force:
        output_fn(f"[exists: {dest}] pass --force to overwrite")
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(template, encoding="utf-8")
    output_fn(f"[created {dest}]")
    output_fn(
        "Set provider/model and export your API key env var, then run: "
        'agentkernel run "..."'
    )
    return 0


def run_sessions(
    config: Config,
    action: str,
    session_id: str | None,
    *,
    output_fn: Callable[[str], None] = print,
) -> int:
    """List, show, or delete saved sessions (§18.2). Resume one with --resume."""
    memory_dir = config.memory_dir or str(Path(config.log_dir).parent / "memory")
    store = make_memory_store(config.memory_store or "file", memory_dir)
    if store is None:
        output_fn("[no memory store configured]")
        return 1

    if action == "list":
        ids = store.list_sessions()
        if not ids:
            output_fn("(no saved sessions)")
            return 0
        for sid in ids:
            messages = store.load(sid)
            first_user = next(
                (m.content for m in messages if m.role == "user" and m.content), ""
            )
            preview = (first_user[:60] + "…") if len(first_user) > 60 else first_user
            output_fn(f"  {sid}  ({len(messages)} msgs)  {preview}")
        output_fn("\nResume one with:  agentkernel --resume <id>")
        return 0

    if not session_id:
        output_fn(f"usage: agentkernel sessions {action} <session_id>")
        return 1

    if action == "show":
        messages = store.load(session_id)
        if not messages:
            output_fn(f"[no session {session_id!r}]")
            return 1
        for m in messages:
            text = m.content or (
                f"[{len(m.tool_results)} tool result(s)]" if m.tool_results else ""
            )
            output_fn(f"{m.role}: {text}")
        return 0

    # action == "delete"
    store.delete(session_id)
    output_fn(f"[deleted session {session_id}]")
    return 0


def run_background(
    prompt: str,
    *,
    config_path: str | None = None,
    cwd: str | None = None,
    log_dir: str = ".agentkernel/traces",
    spawn=None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Launch ``agentkernel run <prompt>`` as a detached process (§18.2).

    Output is redirected to a file under ``<log_dir>/../background/``. The child
    is fully detached so it survives this process exiting. ``spawn`` is injectable
    for tests; by default it is a platform-appropriate ``subprocess.Popen``.
    """
    import subprocess
    import sys
    import uuid

    if not prompt.strip():
        output_fn("[background] nothing to run (empty prompt)")
        return 1

    out_dir = Path(log_dir).parent / "background"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex[:8]}.out"
    argv = [sys.executable, "-m", "agentkernel"]
    if config_path:
        argv += ["--config", config_path]
    if cwd:
        argv += ["-C", cwd]  # let the detached child resolve config from the project
    argv += ["run", prompt]

    def _default_spawn(args, *, stdout):
        kwargs = {
            "stdout": stdout,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(args, **kwargs)

    launcher = spawn or _default_spawn
    handle = out_path.open("w", encoding="utf-8")
    try:
        proc = launcher(argv, stdout=handle)
    finally:
        # The child holds its own copy of the fd; we can close ours.
        handle.close()
    pid = getattr(proc, "pid", "?")
    output_fn(f"[background] started (pid {pid}); output -> {out_path}")
    return 0


def run_kanban(
    config: Config,
    action: str,
    rest: list[str],
    *,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Inspect and manage the shared work board from the CLI (§18.3)."""
    from agentkernel.kanban import Board, render_task

    board = Board(config.kanban_path)

    if action == "list":
        tasks = board.list()
        if not tasks:
            output_fn("(board is empty)")
            return 0
        for t in tasks:
            output_fn(f"  {render_task(t)}")
        return 0
    if action == "add":
        if not rest:
            output_fn('usage: agentkernel kanban add "<title>"')
            return 1
        task = board.add(" ".join(rest))
        output_fn(f"[added {task.id}: {task.title}]")
        return 0

    if not rest:
        output_fn(f"usage: agentkernel kanban {action} <task_id>")
        return 1
    task_id = rest[0]
    if action == "show":
        task = board.get(task_id)
        if task is None:
            output_fn(f"[no task {task_id}]")
            return 1
        output_fn(render_task(task))
        for note in task.notes:
            output_fn(f"    - {note}")
        return 0
    if action == "complete":
        ok = board.complete(task_id) is not None
    elif action == "block":
        ok = board.block(task_id, " ".join(rest[1:])) is not None
    else:  # remove
        tasks = [t for t in board.list() if t.id != task_id]
        ok = len(tasks) != len(board.list())
        if ok:
            board._write(tasks)
    output_fn(f"[{action}d {task_id}]" if ok else f"[no task {task_id}]")
    return 0 if ok else 1


def _cron_run_one(config: Config, prompt: str) -> str:
    """Run one cron job's prompt through a fresh runtime and return the answer."""
    sandbox = make_sandbox(
        config.sandbox, config.working_dir,
        image=config.sandbox_image, network=config.sandbox_network,
    )
    try:
        agent, telemetry, clients = build_runtime(config, sandbox=sandbox)
        try:
            return agent.run(prompt)
        finally:
            telemetry.close()
            for c in clients:
                c.close()
    finally:
        sandbox.close()


def run_cron(
    config: Config,
    action: str,
    rest: list[str],
    *,
    output_fn: Callable[[str], None] = print,
    run_fn: Callable[[str], str] | None = None,
) -> int:
    """Manage and run scheduled jobs (§18.2)."""
    from agentkernel.cron import JobStore, run_due_jobs

    store = JobStore(config.cron_path)
    runner = run_fn or (lambda prompt: _cron_run_one(config, prompt))

    if action == "list":
        jobs = store.list()
        if not jobs:
            output_fn("(no scheduled jobs)")
            return 0
        for j in jobs:
            state = "" if j.enabled else " [disabled]"
            last = j.last_run or "never"
            preview = (j.prompt[:50] + "…") if len(j.prompt) > 50 else j.prompt
            output_fn(f"  {j.id}  every {j.schedule}{state}  (last: {last})  {preview}")
        return 0

    if action == "add":
        if len(rest) < 2:
            output_fn('usage: agentkernel cron add <schedule> "<prompt>"')
            return 1
        schedule, prompt = rest[0], " ".join(rest[1:])
        try:
            job = store.add(schedule, prompt)
        except ValueError as exc:
            output_fn(f"[invalid schedule] {exc}")
            return 1
        output_fn(f"[added job {job.id}: every {job.schedule}]")
        return 0

    if action in ("remove", "run"):
        if not rest:
            output_fn(f"usage: agentkernel cron {action} <job_id>")
            return 1
        job_id = rest[0]
        if action == "remove":
            ok = store.remove(job_id)
            output_fn(f"[removed {job_id}]" if ok else f"[no job {job_id}]")
            return 0 if ok else 1
        job = store.get(job_id)
        if job is None:
            output_fn(f"[no job {job_id}]")
            return 1
        result = runner(job.prompt)
        store.mark_run(job_id)
        output_fn(result)
        return 0

    # action == "tick": run everything due once.
    results = run_due_jobs(store, runner)
    if not results:
        output_fn("(nothing due)")
        return 0
    for job_id, result in results:
        output_fn(f"[{job_id}] {result}")
    return 0


def run_memory(
    config: Config,
    action: str,
    *,
    session: str | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Curate long-term memory: extract facts from a session, or consolidate.

    These are best-effort harness operations over the configured note store; no
    sandbox, MCP, or tools are needed.
    """
    from agentkernel.curation import MemoryCurator

    notes = _make_configured_note_store(config)
    curator_model = config.memory_curator_model or config.summarizer_model or config.model
    provider = make_provider(replace(config, model=curator_model))
    curator = MemoryCurator(notes, provider)

    if action == "consolidate":
        result = curator.consolidate()
        output_fn(
            f"Consolidated memory: {result.before} -> {result.after} note(s) "
            f"({result.removed} merged/removed)."
        )
        return 0

    if action == "extract":
        memory_dir = config.memory_dir or ".agentkernel/memory"
        store = make_memory_store(config.memory_store or "file", memory_dir)
        target = session
        if target is None:
            sessions = store.list_sessions()
            if not sessions:
                output_fn("[no saved sessions to extract from]")
                return 1
            if len(sessions) > 1:
                output_fn(
                    "[multiple sessions; pass --session <id>. Available: "
                    + ", ".join(sessions)
                    + "]"
                )
                return 1
            target = sessions[0]
        messages = store.load(target)
        if not messages:
            output_fn(f"[no messages in session {target}]")
            return 1
        result = curator.extract(messages)
        output_fn(
            f"Extracted {len(result.added)} new fact(s) from session {target} "
            f"({result.skipped_duplicates} duplicate(s) skipped)."
        )
        for note in result.added:
            output_fn(f"  + {note.text}")
        return 0

    output_fn(f"[unknown memory action: {action}]")
    return 1


def run_improve(
    config: Config,
    *,
    trace: str | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Reflect on a session trace and write an improvement note (Phase 7)."""
    from agentkernel.improvement import SelfImprover

    improver = SelfImprover(make_provider(config), config.improvements_dir)
    trace_path = trace or improver.latest_trace(config.log_dir)
    if trace_path is None:
        output_fn(f"[no trace found in {config.log_dir}]")
        return 1
    try:
        improvement = improver.analyze_trace(trace_path)
    except ProviderError as exc:
        output_fn(f"[provider error] {exc}")
        return 1
    output_fn(improvement.suggestion)
    output_fn(f"[improvement written to {improvement.output_path}]")
    return 0


def run_eval(
    config: Config,
    suite_path: str,
    *,
    judge_model: str | None = None,
    output_path: str | None = None,
    case_filter: list[str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run an eval suite: agent answers each case, a judge scores it (Phase 5).

    Returns 0 only if every case passes, so it doubles as a CI gate.
    """
    from agentkernel.evaluation import Evaluator, load_eval_suite

    default_rubric, cases = load_eval_suite(suite_path)
    effective_default = config.eval_rubric or default_rubric
    if not cases:
        output_fn("[no cases in suite]")
        return 1

    if case_filter:
        case_filter = list(dict.fromkeys(case_filter))
        cases = [
            c
            for c in cases
            if any(fnmatch.fnmatchcase(c.name, pat) for pat in case_filter)
        ]
        if not cases:
            output_fn(f"[no cases matched filter: {case_filter!r}]")
            return 1

    sandbox = make_sandbox(
        config.sandbox, config.working_dir,
        image=config.sandbox_image, network=config.sandbox_network,
    )
    base_agent, telemetry, mcp_clients = build_runtime(config, sandbox=sandbox)

    def agent_factory() -> Agent:
        context = ContextManager(
            budget=base_agent.provider.context_window - config.output_reserve,
            keep_recent_turns=config.keep_recent_turns,
        )
        return Agent(
            base_agent.provider,
            base_agent.registry,
            context,
            AutoApprover(config.approval_policy),  # non-interactive during eval
            NullTelemetry(),
            config,
            context_source=base_agent.context_source,
        )

    judge_model = judge_model or config.judge_model
    judge = (
        make_provider(replace(config, model=judge_model))
        if judge_model
        else base_agent.provider
    )
    evaluator = Evaluator(
        agent_factory, judge,
        default_rubric=effective_default, pass_threshold=config.eval_threshold,
    )
    try:
        summary = evaluator.run_suite(cases)
    finally:
        telemetry.close()
        for client in mcp_clients:
            client.close()
        sandbox.close()

    for result in summary.results:
        mark = "PASS" if result.passed else "FAIL"
        output_fn(f"  [{mark}] {result.name}  score={result.score:.2f}  {result.reasoning}")
    output_fn(
        f"{summary.passed}/{summary.total} passed  "
        f"pass_rate={summary.pass_rate:.0%}  mean_score={summary.mean_score:.2f}"
    )
    if output_path:
        Path(output_path).write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )
        output_fn(f"[report written to {output_path}]")
    return 0 if summary.passed == summary.total else 1


def run_loop(
    config: Config,
    *,
    loop_file: str | None = None,
    skill: str | None = None,
    max_iterations: int | None = None,
    check: str | None = None,
    streak: int | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run a loop-engineering workflow until its stopping condition (Phase 4+).

    Returns 0 if the loop succeeded (reached its success streak), else 1.
    """
    from agentkernel.loops import LoopRunner, load_loop, loop_from_skill

    sandbox = make_sandbox(
        config.sandbox, config.working_dir,
        image=config.sandbox_image, network=config.sandbox_network,
    )
    base_agent, telemetry, mcp_clients = build_runtime(config, sandbox=sandbox)

    if loop_file:
        loop = load_loop(loop_file)
        if max_iterations is not None:
            loop.max_iterations = max_iterations
        if check is not None:
            loop.success_check = check
        if streak is not None:
            loop.success_streak = streak
    elif skill:
        loop = loop_from_skill(
            base_agent.context_source, skill,
            max_iterations=max_iterations or 5,
            success_check=check,
            success_streak=streak or 1,
            cwd=config.working_dir,
        )
        if loop is None:
            telemetry.close()
            for c in mcp_clients:
                c.close()
            sandbox.close()
            output_fn(f"[skill not found: {skill}]")
            return 1
    else:
        telemetry.close()
        for c in mcp_clients:
            c.close()
        sandbox.close()
        output_fn("[loop requires --file or --skill]")
        return 1

    def agent_factory() -> Agent:
        context = ContextManager(
            budget=base_agent.provider.context_window - config.output_reserve,
            keep_recent_turns=config.keep_recent_turns,
        )
        return Agent(
            base_agent.provider, base_agent.registry, context,
            AutoApprover(config.approval_policy), NullTelemetry(), config,
            context_source=base_agent.context_source,
        )

    output_fn(f"[loop: {loop.name} — max {loop.max_iterations} iterations]")
    runner = LoopRunner(agent_factory, sandbox=sandbox, output_fn=output_fn)
    try:
        result = runner.run(loop)
    finally:
        telemetry.close()
        for c in mcp_clients:
            c.close()
        sandbox.close()

    verdict = "SUCCEEDED" if result.succeeded else "stopped without success"
    output_fn(f"{loop.name}: {verdict} after {result.count} iteration(s).")
    return 0 if result.succeeded else 1


def _read_prompt_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"could not read prompt file: {exc}")


def _active_profile(config: Config, args: argparse.Namespace) -> Profile | None:
    name = args.profile or config.profile
    if not name:
        return None
    profile_dir = Path(config.profile_dir) if config.profile_dir else None
    search_dirs = [profile_dir] if profile_dir else None
    return load_profile(name, search_dirs=search_dirs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentkernel", description="Agent kernel CLI")
    parser.add_argument(
        "--config",
        default=None,
        help="explicit TOML config file (skips global/project discovery)",
    )
    parser.add_argument(
        "-C",
        "--cwd",
        default=None,
        help="run as if launched from this directory (sets the project root)",
    )
    parser.add_argument(
        "--verbose-trace",
        action="store_true",
        help="log raw tool arguments to the trace (local debugging only)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable per-turn progress lines in run/repl modes",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="disable live token streaming (print the answer when complete)",
    )
    parser.add_argument(
        "--profile",
        help="active profile name (overrides config.profile)",
    )
    parser.add_argument(
        "--model",
        help="model override for this session (overrides config.model and profile.model_override)",
    )
    parser.add_argument(
        "--memory",
        choices=("file", "memory"),
        help="enable a built-in memory store (overrides config.memory_store)",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume a saved session by id (run/repl); requires a memory store",
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="activate a skill for this session (repeatable)",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("repl", help="interactive REPL")
    subparsers.add_parser("tui", help="interactive curses-based terminal UI")
    run_parser = subparsers.add_parser("run", help="single non-interactive run")
    run_parser.add_argument("prompt", nargs="?", help="text prompt")
    run_parser.add_argument("--file", help="path to a file containing the prompt")
    run_parser.add_argument(
        "--background",
        action="store_true",
        help="run detached in the background; output goes to a file",
    )
    improve_parser = subparsers.add_parser(
        "improve", help="reflect on a session trace and write an improvement note"
    )
    improve_parser.add_argument(
        "--trace", help="trace file to analyze (default: latest in log_dir)"
    )
    eval_parser = subparsers.add_parser(
        "eval", help="run an eval suite and score the answers with a judge model"
    )
    eval_parser.add_argument("--suite", required=True, help="path to a TOML eval suite")
    eval_parser.add_argument(
        "--judge-model", help="model to score answers (default: config.judge_model)"
    )
    eval_parser.add_argument(
        "--output", "-o", help="write a JSON evaluation report to this path"
    )
    eval_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run only matching case names/globs (repeatable)",
    )
    loop_parser = subparsers.add_parser(
        "loop", help="run a repeatable workflow loop with a stopping condition"
    )
    loop_parser.add_argument("--file", help="path to a loop TOML file")
    loop_parser.add_argument("--skill", help="use a skill's body as the loop prompt")
    loop_parser.add_argument("--max-iterations", type=int, help="iteration cap")
    loop_parser.add_argument("--check", help="success shell command (exit 0 = success)")
    loop_parser.add_argument(
        "--streak", type=int, help="consecutive successes required to stop"
    )
    insights_parser = subparsers.add_parser(
        "insights", help="aggregate session traces into a usage/cost report"
    )
    insights_parser.add_argument(
        "--days", type=int, help="only include records from the last N days"
    )
    subparsers.add_parser("doctor", help="check config, dependencies, and credentials")
    memory_parser = subparsers.add_parser(
        "memory", help="curate long-term memory (extract facts, consolidate)"
    )
    memory_parser.add_argument(
        "action", choices=("extract", "consolidate"), help="what to do"
    )
    memory_parser.add_argument(
        "--session", help="session id to extract from (default: the only session)"
    )
    sessions_parser = subparsers.add_parser(
        "sessions", help="list, show, or delete saved sessions"
    )
    sessions_parser.add_argument(
        "action", choices=("list", "show", "delete"), help="what to do"
    )
    sessions_parser.add_argument("session_id", nargs="?", help="session id (show/delete)")
    cron_parser = subparsers.add_parser(
        "cron", help="manage scheduled jobs (list/add/remove/run/tick)"
    )
    cron_parser.add_argument(
        "action", choices=("list", "add", "remove", "run", "tick")
    )
    cron_parser.add_argument(
        "rest", nargs="*",
        help="add: <schedule> <prompt...>; remove/run: <job_id>",
    )
    kanban_parser = subparsers.add_parser(
        "kanban", help="manage the shared work board (list/add/show/complete/remove)"
    )
    kanban_parser.add_argument(
        "action", choices=("list", "add", "show", "complete", "block", "remove")
    )
    kanban_parser.add_argument(
        "rest", nargs="*", help="add: <title...>; show/complete/block/remove: <task_id>"
    )
    init_parser = subparsers.add_parser(
        "init", help="scaffold a starter agentkernel.toml (project or --global)"
    )
    init_parser.add_argument(
        "--global", dest="global_config", action="store_true",
        help="write the user-global ~/.agentkernel/config.toml instead of a project file",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="overwrite if the config already exists"
    )
    new_parser = subparsers.add_parser(
        "new", help="scaffold a skill, profile, loop, or eval suite from a template"
    )
    new_parser.add_argument(
        "kind", choices=("skill", "profile", "loop", "eval"), help="what to create"
    )
    new_parser.add_argument("name", help="name for the new asset (kebab-case)")
    new_parser.add_argument(
        "--force", action="store_true", help="overwrite if the target already exists"
    )

    args = parser.parse_args(argv)
    command = getattr(args, "command", None) or "repl"

    if command == "run" and not args.prompt and not args.file:
        run_parser.error("the following arguments are required: prompt or --file")

    # `init` and `new` scaffold files; they need no provider/config/runtime.
    if command == "init":
        return run_init(
            target_dir=args.cwd or ".",
            global_config=getattr(args, "global_config", False),
            force=args.force,
        )
    if command == "new":
        return run_new(args.kind, args.name, force=args.force)

    config, project_config_path = resolve_config(args.config, cwd=args.cwd or ".")
    # The concrete config path to hand to subprocesses / MCP discovery.
    effective_config_path = args.config or (
        str(project_config_path) if project_config_path else None
    )
    if args.skill:
        config.skills = list(dict.fromkeys(config.skills + args.skill))
    if args.model:
        config.model = args.model

    # `insights` and `doctor` read config but need no provider/runtime (§18.7).
    if command == "insights":
        from agentkernel.insights import aggregate_traces, format_insights

        days = getattr(args, "days", None)
        print(format_insights(aggregate_traces(config.log_dir, days=days), days=days))
        return 0
    if command == "doctor":
        from agentkernel.doctor import format_checks, has_failures, run_checks

        checks = run_checks(config)
        print(format_checks(checks))
        return 1 if has_failures(checks) else 0
    if command == "sessions":
        return run_sessions(config, args.action, getattr(args, "session_id", None))
    if command == "memory":
        return run_memory(config, args.action, session=getattr(args, "session", None))
    if command == "cron":
        return run_cron(config, args.action, args.rest)
    if command == "kanban":
        return run_kanban(config, args.action, args.rest)
    if command == "run" and getattr(args, "background", False):
        prompt = _read_prompt_file(args.file) if args.file else (args.prompt or "")
        return run_background(
            prompt,
            config_path=effective_config_path,
            cwd=config.working_dir,
            log_dir=config.log_dir,
        )

    # Load profile early so its model_override and rubric feed into config for
    # every command (run, repl, eval, loop, improve).
    active_profile = _active_profile(config, args)
    if active_profile is not None:
        if not args.model and active_profile.model_override:
            config.model = active_profile.model_override
        if active_profile.rubric and config.eval_rubric is None:
            config.eval_rubric = active_profile.rubric

    # 'improve' is a self-contained tool: it needs only a provider, not the full
    # runtime (no MCP servers, sandbox, or session trace file).
    if command == "improve":
        return run_improve(config, trace=getattr(args, "trace", None))

    if command == "eval":
        return run_eval(
            config,
            args.suite,
            judge_model=getattr(args, "judge_model", None),
            output_path=getattr(args, "output", None),
            case_filter=args.case or None,
        )

    if command == "tui":
        from agentkernel.tui import run_tui
        return run_tui(config)

    if command == "loop":
        return run_loop(
            config,
            loop_file=args.file,
            skill=args.skill,
            max_iterations=getattr(args, "max_iterations", None),
            check=args.check,
            streak=args.streak,
        )

    # Merge MCP servers from the global config and the project config (or just
    # the explicit file), so servers can be declared once for all projects.
    if args.config:
        mcp_servers = load_mcp_servers(args.config)
    else:
        mcp_servers = []
        gpath = global_config_path(agent_home())
        if gpath.is_file():
            mcp_servers += load_mcp_servers(gpath)
        if project_config_path is not None:
            mcp_servers += load_mcp_servers(project_config_path)
    budget = BudgetGuard(
        max_cost_usd=config.max_cost_usd,
        max_input_tokens=config.max_input_tokens_per_run,
        model=config.model,
    )

    memory_kind = args.memory or config.memory_store
    memory_dir = config.memory_dir or str(Path(config.log_dir).parent / "memory")
    memory = make_memory_store(memory_kind, memory_dir)

    resume_id = getattr(args, "resume", None)
    if resume_id and memory is None:
        print(
            "[warning] --resume needs a memory store; enable one with --memory file "
            "or memory_store in config. Starting a fresh session.",
            file=sys.stderr,
        )
        resume_id = None

    sandbox = make_sandbox(
        config.sandbox,
        config.working_dir,
        image=config.sandbox_image,
        network=config.sandbox_network,
    )
    try:
        agent, telemetry, mcp_clients = build_runtime(
            config,
            mcp_servers=mcp_servers,
            verbose=args.verbose_trace,
            budget=budget,
            memory=memory,
            sandbox=sandbox,
            session_id=resume_id,
        )
    except (ProviderError, MCPError) as exc:
        sandbox.close()
        print(f"[startup error] {exc}")
        return 1

    profile = active_profile

    # Live streaming writes model text to stdout as it arrives. When on, skip the
    # per-turn progress lines so they don't interleave with the streamed text.
    streaming = getattr(config, "stream", True) and not args.no_stream

    def _stdout_stream(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    stream_fn = _stdout_stream if streaming else None

    if not args.no_progress and not streaming:
        telemetry = ProgressTelemetry(telemetry, output_fn=print)
        agent.telemetry = telemetry

    try:
        print(f"[session trace: {telemetry.path}]")
        if config.sandbox == "docker":
            print(
                f"[sandbox: docker image={config.sandbox_image} "
                f"network={config.sandbox_network}]"
            )
        if memory_kind:
            print(f"[memory: {memory_kind} @ {memory_dir}]")
        if mcp_clients:
            print(f"[connected MCP servers: {', '.join(s.name for s in mcp_servers)}]")

        if command == "run":
            prompt = args.prompt
            if args.file:
                prompt = _read_prompt_file(args.file)
            return run_once(
                agent, prompt or "", profile=profile, stream_fn=stream_fn, config=config
            )
        return repl(agent, config=config, stream_fn=stream_fn)
    finally:
        telemetry.close()
        for client in mcp_clients:
            client.close()
        sandbox.close()


if __name__ == "__main__":
    raise SystemExit(main())
