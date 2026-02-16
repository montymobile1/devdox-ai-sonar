"""
This module intentionally violates Sonar rule python:S7503.

S7503 raises an issue when a function is declared async but does not use any
asynchronous features (no `await`, no `async for`, no `async with`).
"""

# Intentionally "bad": async function with zero async features inside.
async def build_greeting(name: str) -> str:
    # No await/async-for/async-with here -> should trigger S7503.
    return f"Hello, {name}!"
