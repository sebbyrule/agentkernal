"""Secret redaction for tool output (design §18.1).

Tool results — `bash` stdout, `read_file` contents, web/MCP output — can carry
API keys and tokens. This module scrubs well-known secret formats from that text
*before* it enters the context window and the trace, so a leaked credential in a
command's output doesn't get memorialized in the conversation or logged.

It is deliberately conservative: it matches high-signal token shapes (provider
key prefixes, PEM private-key blocks, `Authorization` headers, and labelled
`secret = …` assignments) rather than guessing at entropy, to keep false
positives low. Structured `ToolResult.data` is never touched — only text content.
"""

from __future__ import annotations

import re

_PLACEHOLDER = "[REDACTED]"

# PEM private-key blocks (multi-line) — replace the whole block.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

# Standalone, high-signal token formats; the whole match is replaced. Anthropic's
# `sk-ant-` must come before the generic `sk-` so it wins.
_TOKEN_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[posru]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}"),
)

# Labelled assignments — keep the label + operator, redact the value. The value
# must be a contiguous 12+ char run (no spaces), which keeps prose like
# "token: the next word" from matching.
_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)"
    r"(\s*[=:]\s*)"
    r"['\"]?([A-Za-z0-9._\-/+=]{12,})['\"]?"
)

# `Authorization: Bearer <token>` — keep the header prefix, redact the token.
_AUTH_HEADER = re.compile(
    r"(?i)(authorization\s*:\s*(?:bearer\s+)?)([A-Za-z0-9._\-]{16,})"
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Return ``(scrubbed_text, count)`` with known secret formats replaced.

    ``count`` is how many secrets were redacted (0 means the text was clean).
    """
    if not text:
        return text, 0

    count = 0

    def _apply(pattern: re.Pattern[str], repl, s: str) -> str:
        nonlocal count
        s, n = pattern.subn(repl, s)
        count += n
        return s

    out = _apply(_PRIVATE_KEY, _PLACEHOLDER, text)
    for pattern in _TOKEN_PATTERNS:
        out = _apply(pattern, _PLACEHOLDER, out)
    out = _apply(_ASSIGNMENT, lambda m: f"{m.group(1)}{m.group(2)}{_PLACEHOLDER}", out)
    out = _apply(_AUTH_HEADER, lambda m: f"{m.group(1)}{_PLACEHOLDER}", out)
    return out, count
