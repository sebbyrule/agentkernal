---
name: commit-and-pr
description: Write clear git commits and pull-request descriptions. Use when asked to commit changes, write a commit message, open or describe a PR, or summarize a change for review.
---

# Commit and PR

Good history explains *why*, not just *what* — the diff already shows the what.

## Commit messages (Conventional Commits)
Format the subject as `type: concise summary` (≤ 72 chars, imperative mood):

- `feat:` a new capability   `fix:` a bug fix   `refactor:` structure, no behavior change
- `docs:` documentation   `test:` tests only   `chore:` tooling/deps/maintenance

Body (optional, wrapped at ~72 cols): explain the motivation and any non-obvious
trade-offs or consequences. Reference issues. One logical change per commit — if
the summary needs an "and", consider splitting.

## Before committing
- Review the actual diff (`git diff --staged`), not your memory of it.
- Stage only what belongs in this commit; don't sweep in unrelated changes.
- Confirm tests/lint pass. Never commit secrets or generated artifacts.

## Pull-request descriptions
- **What & why** — one short paragraph: the problem and the approach.
- **Changes** — a tight bullet list of the meaningful changes (not every file).
- **Testing** — how you verified it (commands run, results).
- Call out anything risky, deferred, or needing reviewer attention.

Keep it scannable: a reviewer should grasp the change in under a minute.
