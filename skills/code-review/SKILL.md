---
name: code-review
description: Review a code change for correctness bugs and clarity issues. Use whenever asked to review, critique, check, or give feedback on a diff or file.
---

# Code review

Work in this order and report findings concisely — correctness before style.

## 1. Correctness (must-fix)
- Off-by-one errors, reversed conditions, and unhandled `None`/empty/zero cases.
- Every error path returns or raises — no silently swallowed exceptions.
- Resource cleanup (files, sockets, locks, subprocesses) on *all* paths, including errors.
- Concurrency: shared state mutated without guarding; await/async misuse.
- Inputs validated before use; no trusting unchecked external data.

## 2. Clarity (should-fix)
- Names say what they hold or do; flag misleading or vague ones.
- Prefer early returns over deep nesting.
- Dead code, duplicated logic, and comments that no longer match the code.

## 3. Tests
- Does the change have coverage for the new behavior and at least one failure path?
- Flag assertions that would pass even if the code were broken.

## Output format
Group findings as **Bugs** (must fix) and **Nits** (optional). Each gets a
`file:line` reference and a one-line suggested fix. If the change is clean, say
so plainly rather than inventing problems.
