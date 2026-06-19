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
from agentkernel.approval import CliApprover, LocalSandbox
from agentkernel.budget import BudgetGuard
from agentkernel.config import Config
from agentkernel.context import ContextManager, ModelSummarizer
from agentkernel.mcp import MCPClient, MCPError, load_mcp_servers, register_mcp_servers
from agentkernel.mcp.config import MCPServerConfig
from agentkernel.knowledge import KnowledgeGraph, make_graph_tools
from agentkernel.memory import MemoryStore, make_memory_store
from agentkernel.progress import ProgressTelemetry
from agentkernel.profiles import Profile, load_profile
from agentkernel.providers import ProviderError, make_provider
from agentkernel.skills import DirectorySkillStore
from agentkernel.telemetry import JsonlTelemetry, NullTelemetry
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import default_tools

_BANNER = (
    "agentkernel REPL - type your message and press enter. Commands: /exit, "
    "/clear, /system, /profile, /skills, /skill, /tools, /trace, /cost."
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
) -> tuple[Agent, JsonlTelemetry, list[MCPClient]]:
    """Construct an Agent, its telemetry, and any MCP clients from config.

    Keys come from env. MCP-discovered tools register into the same registry as
    the builtins — the loop never learns they came from elsewhere (design §13).
    """
    provider = make_provider(config)
    registry = ToolRegistry()
    for spec in default_tools(
        LocalSandbox(),
        config.working_dir,
        max_result_tokens=config.max_tool_result_tokens,
    ):
        registry.register(spec)
    mcp_clients = register_mcp_servers(registry, list(mcp_servers or []))

    # Phase 6: expose the knowledge graph as ordinary tools when enabled.
    if config.enable_graph:
        for spec in make_graph_tools(KnowledgeGraph(config.graph_path)):
            registry.register(spec)

    # Phase 4: skills contribute system-prompt text via the context source.
    # The store is harmless when the directory is absent or no skills are active.
    context_source = DirectorySkillStore(config.skills_dir, active_skills=config.skills)

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

    args = parser.parse_args(argv)
    command = getattr(args, "command", None) or "repl"

    if command == "run" and not args.prompt and not args.file:
        run_parser.error("the following arguments are required: prompt or --file")

    config = Config.load(args.config)

    # 'improve' is a self-contained tool: it needs only a provider, not the full
    # runtime (no MCP servers, sandbox, or session trace file).
    if command == "improve":
        return run_improve(config, trace=getattr(args, "trace", None))

    mcp_servers = load_mcp_servers(args.config)
    budget = BudgetGuard(
        max_cost_usd=config.max_cost_usd,
        max_input_tokens=config.max_input_tokens_per_run,
        model=config.model,
    )

    memory_kind = args.memory or config.memory_store
    memory_dir = config.memory_dir or str(Path(config.log_dir).parent / "memory")
    memory = make_memory_store(memory_kind, memory_dir)

    try:
        agent, telemetry, mcp_clients = build_runtime(
            config,
            mcp_servers=mcp_servers,
            verbose=args.verbose_trace,
            budget=budget,
            memory=memory,
        )
    except (ProviderError, MCPError) as exc:
        print(f"[startup error] {exc}")
        return 1

    profile = _active_profile(config, args)

    if not args.no_progress:
        telemetry = ProgressTelemetry(telemetry, output_fn=print)
        agent.telemetry = telemetry

    try:
        print(f"[session trace: {telemetry.path}]")
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


if __name__ == "__main__":
    raise SystemExit(main())
