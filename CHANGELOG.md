# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

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
