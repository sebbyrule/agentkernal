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
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from agentkernel.agent import Agent
from agentkernel.approval import AutoApprover, CliApprover, Sandbox, make_sandbox
from agentkernel.budget import BudgetGuard
from agentkernel.config import Config
from agentkernel.context import ContextManager, ModelSummarizer
from agentkernel.mcp import MCPClient, MCPError, load_mcp_servers, register_mcp_servers
from agentkernel.mcp.config import MCPServerConfig
from agentkernel.knowledge import KnowledgeGraph, make_graph_tools
from agentkernel.memory import (
    NoteStore,
    MemoryStore,
    make_memory_store,
    make_memory_tools,
    make_note_store,
)
from agentkernel.embeddings import EmbeddingError, OpenAIEmbeddingProvider
from agentkernel.semantic_memory import SemanticSqliteNoteStore
from agentkernel.profiles import Profile, load_profile
from agentkernel.progress import ProgressTelemetry
from agentkernel.providers import ProviderError, make_provider
from agentkernel.skills import DirectorySkillStore, make_skill_tool
from agentkernel.subagent import make_spawn_tool
from agentkernel.telemetry import JsonlTelemetry, NullTelemetry
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import default_tools

_BANNER = (
    "agentkernel REPL - type your message and press enter. Commands: /exit, "
    "/clear, /system, /profile, /skills, /skill, /tools, /trace, /cost, /memory."
)
_PROMPT = "> "
_EXIT_WORDS = {"exit", "quit", ":q"}


def build_runtime(
    config: Config,
    *,
    mcp_servers: list[MCPServerConfig] | None = None,
    verbose: bool = False,
    budget: BudgetGuard | None = None,
    memory: MemoryStore | None = None,
    sandbox: Sandbox | None = None,
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
    registry = ToolRegistry()
    for spec in default_tools(
        sandbox,
        config.working_dir,
        max_result_tokens=config.max_tool_result_tokens,
    ):
        registry.register(spec)
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
        if config.semantic_search:
            try:
                emb_provider = OpenAIEmbeddingProvider.from_config(config)
                notes_path = Path(config.memory_notes_path)
                if notes_path.suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
                    notes_path = notes_path.parent / (notes_path.stem + ".semantic.db")
                notes = SemanticSqliteNoteStore(notes_path, embedding_provider=emb_provider)
            except EmbeddingError as exc:
                print(f"Warning: semantic search disabled: {exc}", file=sys.stderr)
                notes = make_note_store(config.memory_notes_path)
        else:
            notes = make_note_store(config.memory_notes_path)
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
    approver = CliApprover(config.approval_policy, allowlist=config.approval_allowlist)

    # Sub-agent delegation (design §13): the model can spawn focused children.
    # base_specs snapshots the tools BEFORE spawn so spawn isn't self-recursive
    # except through the explicit, depth-limited spawn tools it creates.
    if config.enable_spawn:
        base_specs = registry.specs()
        registry.register(
            make_spawn_tool(
                provider=provider,
                base_specs=base_specs,
                approver=approver,
                config=config,
                max_depth=config.spawn_max_depth,
            )
        )

    telemetry = JsonlTelemetry(config.log_dir, config.model, verbose=verbose)
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

    output_fn(f"[unknown command: /{cmd}]")
    return True


def repl(
    agent: Agent,
    *,
    config: Config | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Read-eval-print chat over one Agent (context persists across messages)."""
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
            if not _handle_slash(line, agent, profile, config or agent.config, output_fn):
                break
            continue
        try:
            answer = agent.run(line, profile=profile)
        except ProviderError as exc:
            output_fn(f"[provider error] {exc}")
            continue
        output_fn(answer)
    return 0


def run_once(
    agent: Agent,
    prompt: str,
    *,
    profile: Profile | None = None,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Execute a single non-interactive turn and print the final answer."""
    try:
        answer = agent.run(prompt, profile=profile)
    except ProviderError as exc:
        output_fn(f"[provider error] {exc}")
        return 1
    output_fn(answer)
    return 0


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
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run an eval suite: agent answers each case, a judge scores it (Phase 5).

    Returns 0 only if every case passes, so it doubles as a CI gate.
    """
    from agentkernel.evaluation import Evaluator, load_eval_suite

    default_rubric, cases = load_eval_suite(suite_path)
    if not cases:
        output_fn("[no cases in suite]")
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
        default_rubric=default_rubric, pass_threshold=config.eval_threshold,
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
        "--config", default="agentkernel.toml", help="path to the TOML config file"
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
        "--profile",
        help="active profile name (overrides config.profile)",
    )
    parser.add_argument(
        "--memory",
        choices=("file", "memory"),
        help="enable a built-in memory store (overrides config.memory_store)",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("repl", help="interactive REPL")
    run_parser = subparsers.add_parser("run", help="single non-interactive run")
    run_parser.add_argument("prompt", nargs="?", help="text prompt")
    run_parser.add_argument("--file", help="path to a file containing the prompt")
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

    args = parser.parse_args(argv)
    command = getattr(args, "command", None) or "repl"

    if command == "run" and not args.prompt and not args.file:
        run_parser.error("the following arguments are required: prompt or --file")

    config = Config.load(args.config)

    # 'improve' is a self-contained tool: it needs only a provider, not the full
    # runtime (no MCP servers, sandbox, or session trace file).
    if command == "improve":
        return run_improve(config, trace=getattr(args, "trace", None))

    if command == "eval":
        return run_eval(config, args.suite, judge_model=getattr(args, "judge_model", None))

    if command == "loop":
        return run_loop(
            config,
            loop_file=args.file,
            skill=args.skill,
            max_iterations=getattr(args, "max_iterations", None),
            check=args.check,
            streak=args.streak,
        )

    mcp_servers = load_mcp_servers(args.config)
    budget = BudgetGuard(
        max_cost_usd=config.max_cost_usd,
        max_input_tokens=config.max_input_tokens_per_run,
        model=config.model,
    )

    memory_kind = args.memory or config.memory_store
    memory_dir = config.memory_dir or str(Path(config.log_dir).parent / "memory")
    memory = make_memory_store(memory_kind, memory_dir)

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
        )
    except (ProviderError, MCPError) as exc:
        sandbox.close()
        print(f"[startup error] {exc}")
        return 1

    profile = _active_profile(config, args)

    if not args.no_progress:
        telemetry = ProgressTelemetry(telemetry, output_fn=print)
        agent.telemetry = telemetry

    try:
        print(f"[session trace: {telemetry.path}]")
        if config.sandbox == "docker":
            print(f"[sandbox: docker image={config.sandbox_image} network={config.sandbox_network}]")
        if memory_kind:
            print(f"[memory: {memory_kind} @ {memory_dir}]")
        if mcp_clients:
            print(f"[connected MCP servers: {', '.join(s.name for s in mcp_servers)}]")

        if command == "run":
            prompt = args.prompt
            if args.file:
                prompt = _read_prompt_file(args.file)
            return run_once(agent, prompt or "", profile=profile)
        return repl(agent, config=config)
    finally:
        telemetry.close()
        for client in mcp_clients:
            client.close()
        sandbox.close()


if __name__ == "__main__":
    raise SystemExit(main())
