---
name: security-review
description: Review code for security vulnerabilities. Use when asked to do a security review, check for vulnerabilities, audit for injection or auth flaws, or assess the safety of a change.
---

# Security review

Focus on exploitable flaws with real impact, not theoretical lint. For each
finding, state the **vector** (how it's reached), the **impact**, and a **fix**.

## What to look for
- **Injection** — user/external input reaching a shell, SQL, eval, file path, or
  template without escaping or parameterization. Trace tainted data to its sink.
- **Path traversal** — `..` or absolute paths escaping an intended directory.
- **Secrets** — hardcoded keys/tokens/passwords; secrets logged, echoed, or
  written to disk/traces. Credentials should come only from the environment.
- **AuthN/AuthZ** — missing or bypassable checks; trusting client-supplied identity
  or roles; insecure defaults.
- **Deserialization / parsing** — untrusted input to `pickle`, YAML `load`, etc.
- **SSRF / unvalidated requests** — user-controlled URLs fetched server-side.
- **Crypto** — home-rolled crypto, weak/`md5`/`sha1` for passwords, predictable
  randomness for security tokens.

## Method
1. Identify trust boundaries — where does external input enter?
2. Follow each tainted input to a dangerous sink.
3. For each reachable sink, judge exploitability before flagging.

## Output format
Order findings by severity (**Critical / High / Medium / Low**). Each: vector,
impact, `file:line`, and a concrete fix. Don't pad the list with non-issues — if
it's clean, say so.
