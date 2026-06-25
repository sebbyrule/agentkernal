"""Shell completion script generation (design §18.7)."""

from __future__ import annotations

from agentkernel.cli import main, run_completion

COMMANDS = ["run", "repl", "eval", "skill"]


def test_bash_completion_lists_commands_and_registers():
    out: list[str] = []
    assert run_completion("bash", COMMANDS, output_fn=out.append) == 0
    script = "\n".join(out)
    assert "complete -F _agentkernel_completion agentkernel" in script
    assert "run repl eval skill" in script


def test_zsh_completion_has_compdef_header():
    out: list[str] = []
    assert run_completion("zsh", COMMANDS, output_fn=out.append) == 0
    assert out[0].startswith("#compdef agentkernel")


def test_fish_completion_uses_subcommand_predicate():
    out: list[str] = []
    assert run_completion("fish", COMMANDS, output_fn=out.append) == 0
    assert "__fish_use_subcommand" in "\n".join(out)


def test_unsupported_shell_errors():
    out: list[str] = []
    assert run_completion("powershell", COMMANDS, output_fn=out.append) == 1


def test_completion_via_main_includes_real_subcommands(capsys):
    # End-to-end through argparse: the script reflects actual registered commands.
    assert main(["completion", "bash"]) == 0
    captured = capsys.readouterr().out
    for cmd in ("run", "repl", "skill", "completion"):
        assert cmd in captured
