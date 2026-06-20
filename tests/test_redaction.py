"""Secret-redaction tests (design §18.1): the pure redactor and its wiring into
the agent loop's single tool-result processing point."""

from __future__ import annotations

from agentkernel.config import Config
from agentkernel.redaction import redact_secrets
from agentkernel.types import ToolCall, ToolResult
from tests.conftest import build_agent
from tests.fakes import FakeProvider, text_response, tool_call_response

# --- the pure redactor --------------------------------------------------------

def test_redacts_known_token_prefixes():
    for secret in (
        "sk-ant-api03-abcdef1234567890ABCDEFxyz",
        "sk-proj-abcdefghij1234567890ABCD",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIA1234567890ABCDEF",
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
        "xoxb-1234567890-abcdefghij",
        "hf_abcdefghijklmnopqrstuvwxyz0123456789",
    ):
        out, n = redact_secrets(f"the key is {secret} ok")
        assert n >= 1
        assert secret not in out
        assert "[REDACTED]" in out


def test_redacts_assignment_keeps_label():
    out, n = redact_secrets("API_KEY=supersecretvalue1234567890")
    assert n == 1
    assert "supersecretvalue1234567890" not in out
    assert out.startswith("API_KEY=")  # label + operator preserved
    assert "[REDACTED]" in out


def test_redacts_authorization_header():
    out, n = redact_secrets("Authorization: Bearer abcdefghijklmnop1234567890")
    assert n == 1
    assert "abcdefghijklmnop1234567890" not in out
    assert out.lower().startswith("authorization: bearer ")


def test_redacts_pem_private_key_block():
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA1234567890\nabcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out, n = redact_secrets(f"here it is:\n{pem}\ndone")
    assert n == 1
    assert "MIIEowIBAAKCAQEA" not in out
    assert "here it is:" in out and "done" in out


def test_clean_text_is_untouched():
    text = "The token bucket holds 5 items; the password prompt is on line 3."
    out, n = redact_secrets(text)
    assert n == 0
    assert out == text  # prose with the words token/password but no secret values


def test_empty_input():
    assert redact_secrets("") == ("", 0)


# --- wiring into the loop -----------------------------------------------------

def _leaky_tool():
    from agentkernel.tools.base import ToolSpec

    def handler(_args):
        return ToolResult("", "fetched: sk-ant-api03-leakedSECRET1234567890abcd")

    return ToolSpec(
        name="fetch",
        description="fetch",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=handler,
    )


def _run_with_leaky_tool(config):
    provider = FakeProvider([
        tool_call_response(ToolCall("c1", "fetch", {})),
        text_response("done"),
    ])
    agent = build_agent(provider, config=config)
    agent.registry.register(_leaky_tool())
    agent.run("go")
    # The tool-role message sent back on the 2nd call carries the result content.
    second_request = provider.calls[1]
    tool_msgs = [m for m in second_request if m.role == "tool"]
    return tool_msgs[0].tool_results[0].content


def test_loop_redacts_tool_output_by_default():
    content = _run_with_leaky_tool(Config())
    assert "leakedSECRET" not in content
    assert "[REDACTED]" in content


def test_loop_redaction_can_be_disabled():
    content = _run_with_leaky_tool(Config(redact_tool_output=False))
    assert "sk-ant-api03-leakedSECRET1234567890abcd" in content
