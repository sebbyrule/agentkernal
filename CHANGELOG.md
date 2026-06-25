# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-25

Additive release: new capabilities, no breaking changes. Every feature is opt-in
or defaults to the previous behavior.

### Added
- **Multimodality**: image input in canonical messages (`Message.images` /
  `ImageContent`), translated by each adapter and gated by
  `Provider.supports_images`. Attach via `agentkernel run --image PATH_OR_URL`
  (repeatable) or the REPL `/image` command. Text-only providers drop images
  rather than erroring.
- **More providers**: named OpenAI-compatible adapters for **OpenRouter**,
  **DeepSeek**, and **Gemini** (`provider = "openrouter" | "deepseek" |
  "gemini"`), each with a default endpoint, key env var, and capability.
- **Memory recall tuning**: opt-in recency/importance-weighted recall
  (`memory_recency_weight` / `memory_importance_weight` / `memory_half_life_days`)
  and per-project **namespaces** via `memory_scope` (recall returns global +
  active scope). `remember(global=true)` pins a universal fact under a scope.
- **Shareable skill bundles**: `agentkernel skill pack|install|list` package a
  skill (`SKILL.md` + resources) as a portable `.skill.zip`.
- **Shell completions**: `agentkernel completion bash|zsh|fish`.

### Changed
- Auxiliary models (compaction, risk classifier, curation, eval judge) now
  resolve through a single named-role router (`agentkernel/roles.py`); the
  per-role config fields are unchanged.
- Memory consolidation is scope-aware: it consolidates each namespace
  independently and preserves each note's scope.

## [0.1.0] — first public release

First packaged release: `agentkernel` is installable from PyPI and runnable from
any directory.

### Packaging
- Installable as a CLI: `uv tool install agentkernel-cli` / `pipx install
  agentkernel-cli` / `pip install agentkernel-cli` (the PyPI distribution name;
  the command and import name remain `agentkernel`), plus `python -m agentkernel`.
- Full project metadata (Apache-2.0 license, authors, classifiers, URLs); the
  scaffolding `templates/` ship inside the wheel so `agentkernel new` works after
  a global install.
- `agentkernel init` scaffolds a starter project config (`--global` for the
  user-global one).
- GitHub Actions: CI (lint + tests on 3.11–3.13) and a tag-triggered PyPI release
  via Trusted Publishing.

### Highlights of the kernel
- Provider-agnostic agent loop with Anthropic / OpenAI / local adapters, a
  stable cacheable prefix, context compaction, and per-turn telemetry.
- Gated tool execution with `LocalSandbox` and a container-isolating
  `DockerSandbox`.
- MCP client, skills (Anthropic-style progressive disclosure), profiles, a
  knowledge graph, sub-agent `spawn`, an eval harness, and loop engineering.
- Long-term memory: model-controlled `remember`/`recall`, semantic recall, and
  `agentkernel memory extract`/`consolidate` for self-curation.
- Runs anywhere: layered config discovery (global + project), a "global brain,
  project sessions" state policy, and the `-C/--cwd` flag.
