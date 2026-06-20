# Templates

Copy-paste starting points for the things you author repeatedly. Each file is an
annotated skeleton — read the comments, fill in the blanks, drop it in the right
place.

| Template | Create with | Lives in | Loaded by |
|---|---|---|---|
| [`SKILL.md`](SKILL.md) | `agentkernel new skill <name>` | `skills/<name>/SKILL.md` | discovered from `skills_dir` |
| [`profile.toml`](profile.toml) | `agentkernel new profile <name>` | `profiles/<name>.toml` | `--profile <name>` |
| [`loop.toml`](loop.toml) | `agentkernel new loop <name>` | `loops/<name>.toml` | `loop --file …` |
| [`eval-suite.toml`](eval-suite.toml) | `agentkernel new eval <name>` | `evals/<name>.toml` | `eval --suite …` |
| [`mcp-servers.toml`](mcp-servers.toml) | (copy by hand) | your `agentkernel.toml` | MCP client at startup |
| [`tool_module.py`](tool_module.py) | (copy by hand) | wherever you load tools | your runtime / plugin loader |

## Scaffolding command

`agentkernel new <kind> <name>` copies the matching template into place with the
name filled in (the `{{name}}` placeholder is substituted):

```bash
agentkernel new skill   commit-helper      # -> skills/commit-helper/SKILL.md
agentkernel new profile reviewer-strict    # -> profiles/reviewer-strict.toml
agentkernel new loop    until-docs-build    # -> loops/until-docs-build.toml
agentkernel new eval    smoke               # -> evals/smoke.toml
```

It refuses to overwrite an existing file unless you pass `--force`, and rejects
names containing path separators. `new` finds this `templates/` directory by
walking up from the current directory, so run it from inside the project.

## See also

The bundled starter content built from these templates: [`skills/`](../skills),
[`profiles/`](../profiles), and [`loops/`](../loops).
