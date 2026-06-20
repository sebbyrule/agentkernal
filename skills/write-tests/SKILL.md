---
name: write-tests
description: Write focused, meaningful tests for code. Use when asked to add tests, improve coverage, or write a test for a function, module, or bug fix.
---

# Write tests

A good test fails when the code is wrong and passes when it's right — nothing more.

## Before writing
- Read the code under test and match the project's existing test style (framework,
  fixtures, naming). `find_files` for `test_*` and read one or two as a model.
- Identify the behavior to lock in: the happy path, the boundaries, and the failure modes.

## What to cover
- **Happy path** — the normal, expected use.
- **Edges** — empty/zero/one, max, `None`, off-by-one boundaries.
- **Failures** — invalid input raises/returns the right error; error paths are exercised.
- **Regressions** — for a bug fix, add a test that fails on the old code and passes on the new.

## Quality bar
- Each test asserts one behavior; the name says what it checks.
- Avoid assertions that would pass even if the code were broken (e.g. asserting a
  value is "not None" when you can assert the exact value).
- No network, no real clock, no shared mutable state between tests — keep them deterministic.
- Prefer real objects over mocks unless the dependency is slow, external, or nondeterministic.

## Finish
Run the new tests and confirm they pass; confirm they fail if you temporarily
break the code (a test that can't fail isn't testing anything).
