"""Fixture for python:S7503 (async function without await).

Each async function below performs no awaitable work. SonarCloud is
expected to report one S7503 finding per function (3 total).

Routing: AsyncToSyncHandler (rule_handler.py:1245) -> DEFAULT prompts.
"""


async def fetch_constant() -> int:
    return 42


async def make_label(prefix: str, value: int) -> str:
    return f"{prefix}-{value:04d}"


async def is_even(n: int) -> bool:
    return n % 2 == 0
