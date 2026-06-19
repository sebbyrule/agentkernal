"""Interactive REPL entry point (design §16, M4).

Wires a live provider, the builtin tools inside a ``LocalSandbox``, a
``CliApprover``, and JSONL telemetry into an ``Agent``, then runs a simple
read-eval-print chat. The wiring (``build_runtime``) and the loop (``repl``) are
separated so the loop can be tested offline with a scripted provider.
"""

from __future__ import annotations

import argparse
from typing import Callable

from dataclasses import replace

from agentkernel.agent import Agent
from agentkernel.approval import CliApprover, LocalSandbox
from agentkernel.config import Config
from agentkernel.context import ContextManager, ModelSummarizer
from agentkernel.providers import ProviderError, make_provider
from agentkernel.telemetry import JsonlTelemetry
from agentkernel.tools import ToolRegistry
from agentkernel.tools.builtin import default_tools

_BANNER = (
    "agentkernel REPL - type your message and press enter. "
    "Commands: 'exit' / 'quit' to leave."
)
_PROMPT = "> "
_EXIT_WORDS = {"exit", "quit", ":q"}


def build_runtime(config: Config, *, verbose: bool = False) -> tuple[Agent, JsonlTelemetry]:
    """Construct an Agent and its telemetry from config. Keys come from env."""
    provider = make_provider(config)
    registry = ToolRegistry()
    for spec in default_tools(
        LocalSandbox(),
        config.working_dir,
        max_result_tokens=config.max_tool_result_tokens,
    ):
        registry.register(spec)
    budget = provider.context_window - config.output_reserve
    # Use a cheap model for compaction summaries when one is configured;
    # otherwise the structural fallback is used.
    summarizer = None
    if config.summarizer_model:
        summarizer = ModelSummarizer(
            make_provider(replace(config, model=config.summarizer_model))
        )
    context = ContextManager(
        budget=budget,
        keep_recent_turns=config.keep_recent_turns,
        summarizer=summarizer,
    )
    approver = CliApprover(config.approval_policy, allowlist=config.approval_allowlist)
    telemetry = JsonlTelemetry(config.log_dir, config.model, verbose=verbose)
    agent = Agent(provider, registry, context, approver, telemetry, config)
    return agent, telemetry


def repl(
    agent: Agent,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Read-eval-print chat over one Agent (context persists across messages)."""
    output_fn(_BANNER)
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
        try:
            answer = agent.run(line)
        except ProviderError as exc:
            output_fn(f"[provider error] {exc}")
            continue
        output_fn(answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentkernel", description="Agent kernel REPL")
    parser.add_argument(
        "--config", default="agentkernel.toml", help="path to the TOML config file"
    )
    parser.add_argument(
        "--verbose-trace",
        action="store_true",
        help="log raw tool arguments to the trace (local debugging only)",
    )
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    try:
        agent, telemetry = build_runtime(config, verbose=args.verbose_trace)
    except ProviderError as exc:
        print(f"[startup error] {exc}")
        return 1
    try:
        print(f"[session trace: {telemetry.path}]")
        return repl(agent)
    finally:
        telemetry.close()


if __name__ == "__main__":
    raise SystemExit(main())
