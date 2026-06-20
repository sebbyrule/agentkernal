"""A tiny calculator module used by the example tasks.

NOTE: `subtract` has a deliberate bug — it adds instead of subtracting. One of
the example prompts asks the agent to find and fix it with edit_file.
"""


def add(a, b):
    return a + b


def subtract(a, b):
    return a + b  # BUG: should be a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b
