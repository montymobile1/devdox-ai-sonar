"""Adapter for OpenHands' curated provider/model catalogue (API #1).

``openhands.sdk.llm.VERIFIED_MODELS`` is a hand-maintained ``dict`` mapping a
provider family (``"openai"``, ``"anthropic"``, ``"gemini"``, ``"openhands"``,
``"mistral"``, ``"deepseek"``, ``"moonshot"``, ``"minimax"``) to the list of
model IDs the OpenHands team has vouched for. In the UI these are the
"recommended" (⭐) entries.

The adapter exists so callers depend on our module surface instead of the SDK
constant directly — tests replace ``list_providers`` via
``monkeypatch.setattr`` without having to patch OpenHands internals.
"""

from openhands.sdk.llm import VERIFIED_MODELS


def list_providers() -> dict[str, list[str]]:
    """Return the curated provider-to-models map.

    Returns:
        A mapping from provider family name (e.g. ``"openai"``) to the list of
        verified model IDs for that family. The returned object is the SDK's
        live constant; callers must not mutate it.
    """
    return VERIFIED_MODELS
