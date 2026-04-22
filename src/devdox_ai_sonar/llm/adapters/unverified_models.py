"""Adapter for OpenHands' full litellm catalogue minus Bedrock (API #2).

``openhands.sdk.llm.UNVERIFIED_MODELS_EXCLUDING_BEDROCK`` is a snapshot
computed at module import time from ``litellm.model_list`` and
``litellm.model_cost``; any entry already present in ``VERIFIED_MODELS`` is
de-duplicated out. The result is every provider/model combination litellm can
route to that is *not* already on OpenHands' curated list. In the UI these are
the non-⭐ "advanced" entries.

Using the cached snapshot (rather than the callable
``get_unverified_models(...)``) is intentional: we never need the AWS Bedrock
catalogue and the snapshot is free to read, whereas the callable rebuilds the
dict on every invocation.
"""

from openhands.sdk.llm import UNVERIFIED_MODELS_EXCLUDING_BEDROCK


def list_providers() -> dict[str, list[str]]:
    """Return the non-verified portion of the litellm catalogue.

    Returns:
        A mapping from provider family name (e.g. ``"together_ai"``,
        ``"openrouter"``, ``"groq"``) to the list of model IDs litellm can
        route to. Models already present in ``VERIFIED_MODELS`` for a given
        provider are excluded from this view.
    """
    return UNVERIFIED_MODELS_EXCLUDING_BEDROCK
