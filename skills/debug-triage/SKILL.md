---
name: debug-triage
description: Systematically diagnose a bug or failing test. Use when asked to debug, investigate a failure, find the cause of an error, or figure out why something is broken.
---

# Debug triage

Resist the urge to guess-and-patch. Find the cause first, then fix it once.

## 1. Reproduce
- Get the exact failing command and its full output (stack trace, error message).
- Note what changed recently (`search_text` for the symbol, check the failing test).
- If you can't reproduce it, say so — don't fix a bug you can't observe.

## 2. Localize
- Read the stack trace bottom-up to the first frame in our code.
- Read the failing test and the code under test together; state the expected vs actual behavior in one sentence.
- Narrow with `search_text` / `find_files`; form one hypothesis about the cause.

## 3. Confirm the cause
- Verify the hypothesis before editing — add a print/log or read the relevant state.
- A fix you can't explain is a guess. Be able to say *why* this line is wrong.

## 4. Fix minimally
- Make the smallest change that addresses the root cause, not the symptom.
- Don't change a test to make it pass unless the test itself is wrong (and say so).
- Re-run the exact failing command to confirm green, then check nothing else broke.

## Output format
Report: **Cause** (one sentence), **Fix** (what changed and why), **Verification**
(the command you ran and its result).
