"""Adapter for OpenHands' ``infer_litellm_provider`` helper (API #4).

Given a ``provider/model-id`` string (and optionally an ``api_base``), this
returns the litellm provider name that will handle the request, or ``None``
when the string cannot be routed. Used to validate user-entered model strings
on the "Custom" path and to sanity-check picks from the menus.

Caveat: litellm does **not** know about the ``"openhands"`` family. That
family is routed through the OpenHands Cloud gateway rather than a litellm
adapter, so ``infer_provider("openhands/<something>", ...)`` returns ``None``.
Callers that need to accept ``openhands/*`` strings must also look up
``VERIFIED_MODELS["openhands"]`` themselves.
"""

from openhands.sdk.llm.utils.litellm_provider import infer_litellm_provider


def infer_provider(model: str, api_base: str | None) -> str | None:
    """Resolve a model string to a litellm provider name.

    Args:
        model: A model identifier. Either fully-qualified
            (``"openai/gpt-4o"``, ``"gemini/gemini-2.5-flash"``) or a bare
            model ID (``"gpt-4o"``) that litellm can auto-attribute.
        api_base: Optional custom endpoint URL. Only meaningful for providers
            like ``"azure"`` whose routing depends on the base URL.

    Returns:
        The litellm provider name (e.g. ``"openai"``, ``"together_ai"``,
        ``"vllm"``) when the string is routable, or ``None`` if the prefix is
        unknown, the bare model cannot be attributed, or the string is
        malformed.
    """
    return infer_litellm_provider(model=model, api_base=api_base)
