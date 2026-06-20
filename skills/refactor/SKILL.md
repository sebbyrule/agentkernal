---
name: refactor
description: Restructure code without changing its behavior. Use when asked to refactor, clean up, simplify, deduplicate, or improve the structure of existing code.
---

# Refactor

Refactoring changes structure, never behavior. If behavior changes, it's not a
refactor — call it that and treat it as a feature/fix.

## Ground rules
- **Tests are your safety net.** Confirm the suite is green *before* you start.
  If there are no tests for the code you're touching, write a characterization
  test first (see the write-tests skill).
- Change one thing at a time; keep each step small enough to verify.
- Match the surrounding code's idioms, naming, and comment density — don't impose
  a new style mid-file.

## Good moves
- Extract a well-named function from a long one; collapse duplicated logic into one place.
- Replace deep nesting with early returns.
- Rename misleading identifiers to say what they hold or do.
- Delete dead code and stale comments.

## Avoid
- Mixing a refactor with a behavior change in the same step — they hide each other.
- "Improving" code you don't understand; understand it first.
- Broad reformatting that buries the real change in noise.

## Finish
Re-run the full test suite (and linter, if present) and confirm it's still green.
State plainly what you restructured and that behavior is unchanged.
