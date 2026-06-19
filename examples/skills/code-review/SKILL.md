---
name: code-review
description: Review a code diff for correctness bugs and clear style issues. Use when asked to review, critique, or check a change.
---

# Code review

When reviewing a diff, work in this order and report findings concisely.

## Correctness first
- Look for off-by-one errors, unhandled `None`/empty cases, and reversed conditions.
- Check that every error path returns or raises; no silently swallowed exceptions.
- Verify resource cleanup (files, sockets, locks) on all paths.

## Then clarity
- Names should say what they hold or do; flag misleading ones.
- Prefer early returns over deep nesting.

## Output format
Group findings as **Bugs** (must fix) and **Nits** (optional), each with a
`file:line` reference and a one-line fix. If the diff is clean, say so plainly.
