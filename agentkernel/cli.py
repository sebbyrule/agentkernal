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
from types import SimpleNamespace

from agentkernel.agent import Agent
from agentkernel.approval import CliApprover, LocalSandbox
from agentkernel.budget import BudgetGuard
from agentkernel.config import Config
from agentkernel.context import ContextManager, ModelSummarizer
from agentkernel.mcp import MCPClient, MCPError, load_mcp_servers, register_mcp_servers
from agentkernel.mcp.config import MCPServerConfig
from agentkernel.progress import ProgressTelemetry
from agentkernel.providers import ProviderError, make_provider
from agentkernel.telemetry import JsonlTelemetry, NullTelemetry
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import default_tools

_BANNER = (
    "agentkernel REPL - type your message and press enter. "
    "Commands: /exit, /clear, /system, /tools, /trace, /cost."
)
_PROMPT = "> "
_EXIT_WORDS = {"exit", "quit", ":q"}


def build_runtime(
    config: Config,
    *,
    mcp_servers: list[MCPServerConfig] | None = None,
    verbose: bool = False,
    budget: BudgetGuard | None = None,
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
    agent = Agent(provider, registry, context, approver, telemetry, config, budget=budget)
    return agent, telemetry, mcp_clients


def _handle_slash(
    line: str,
    agent: Agent,
    profile: SimpleNamespace,
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
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Read-eval-print chat over one Agent (context persists across messages)."""
    output_fn(_BANNER)
    profile: SimpleNamespace = SimpleNamespace(system_prompt=None)
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
            if not _handle_slash(line, agent, profile, output_fn):
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
    output_fn: Callable[[str], None] = print,
) -> int:
    """Execute a single non-interactive turn and print the final answer."""
    try:
        answer = agent.run(prompt)
    except ProviderError as exc:
        output_fn(f"[provider error] {exc}")
        return 1
    output_fn(answer)
    return 0


def _read_prompt_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"could not read prompt file: {exc}")


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
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("repl", help="interactive REPL")
    run_parser = subparsers.add_parser("run", help="single non-interactive run")
    run_parser.add_argument("prompt", nargs="?", help="text prompt")
    run_parser.add_argument("--file", help="path to a file containing the prompt")

    args = parser.parse_args(argv)
    command = getattr(args, "command", None) or "repl"

    if command == "run" and not args.prompt and not args.file:
        run_parser.error("the following arguments are required: prompt or --file")

    config = Config.load(args.config)
    mcp_servers = load_mcp_servers(args.config)
    budget = BudgetGuard(
        max_cost_usd=config.max_cost_usd,
        max_input_tokens=config.max_input_tokens_per_run,
        model=config.model,
    )
    try:
        agent, telemetry, mcp_clients = build_runtime(
            config,
            mcp_servers=mcp_servers,
            verbose=args.verbose_trace,
            budget=budget,
        )
    except (ProviderError, MCPError) as exc:
        print(f"[startup error] {exc}")
        return 1

    if not args.no_progress:
        telemetry = ProgressTelemetry(telemetry, output_fn=print)
        agent.telemetry = telemetry

    try:
        print(f"[session trace: {telemetry.path}]")
        if mcp_clients:
            print(f"[connected MCP servers: {', '.join(s.name for s in mcp_servers)}]")

        if command == "run":
            prompt = args.prompt
            if args.file:
                prompt = _read_prompt_file(args.file)
            return run_once(agent, prompt or "")
        return repl(agent)
    finally:
        telemetry.close()
        for client in mcp_clients:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
