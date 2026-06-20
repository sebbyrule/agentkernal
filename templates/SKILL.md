---
name: {{name}}
# One line that does double duty: WHAT this skill does AND WHEN to use it.
# Be specific about trigger phrases — this line is all the model sees until it
# decides to load the skill, so it's the whole basis for that decision.
description: <what this does>. Use when <specific situations / phrases that should trigger it>.
---

# {{name}}

<!--
Progressive disclosure: only the name + description above sit in the prompt until
the model loads this skill with the use_skill tool. Keep the body focused and
actionable — instructions the model follows, not background prose. A few hundred
lines max; if it's growing, split detail into bundled files next to this one and
point at them.
-->

## When to use
- <concrete situation 1>
- <concrete situation 2>

## Steps
1. <first thing to do>
2. <next thing>

## Output format
<How the result should be structured, if it matters.>
