"""Thin adapters around the OpenHands SDK and litellm.

This module is the single permitted gateway between devdox-ai-sonar and the
external ``openhands.sdk`` / ``litellm`` libraries. Every other module in the
``llm`` package depends on the functions exported here so tests can substitute
them via ``monkeypatch.setattr`` without touching the real SDKs.

Four adapters live here:

- :func:`list_verified_models` — the OpenHands curated provider/model map
  (``openhands.sdk.llm.VERIFIED_MODELS``).
- :func:`list_unverified_models` — the non-curated litellm catalogue minus
  Bedrock (``openhands.sdk.llm.UNVERIFIED_MODELS_EXCLUDING_BEDROCK``).
- :func:`infer_provider` — litellm's provider-routing helper, used to validate
  user-entered model strings.
- :func:`probe` — a live credential check with a tiny ``litellm.completion``
  call, translating litellm's exception taxonomy into a small closed
  vocabulary so callers never need to import litellm exception types.
"""

from dataclasses import dataclass
from typing import Literal

import litellm
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from openhands.sdk.llm import (
    UNVERIFIED_MODELS_EXCLUDING_BEDROCK,
    VERIFIED_MODELS,
)
from openhands.sdk.llm.utils.litellm_provider import infer_litellm_provider


__all__ = [
    "KeyProbeOutcome",
    "FailureKind",
    "list_verified_models",
    "list_unverified_models",
    "infer_provider",
    "probe",
]


# ---------------------------------------------------------------------------
# Catalogue adapters
# ---------------------------------------------------------------------------


def list_verified_models() -> dict[str, list[str]]:
    """Return the OpenHands-curated provider-to-models map.

    Returns:
        A mapping from provider family name (e.g. ``"openai"``) to the list of
        verified model IDs for that family. The returned object is the SDK's
        live constant; callers must not mutate it.
    """
    return VERIFIED_MODELS


def list_unverified_models() -> dict[str, list[str]]:
    """Return the non-verified portion of the litellm catalogue.

    Uses the SDK's cached ``UNVERIFIED_MODELS_EXCLUDING_BEDROCK`` snapshot
    rather than the callable ``get_unverified_models(...)``: we never need
    the AWS Bedrock catalogue and the snapshot is free to read.

    Returns:
        A mapping from provider family name (``"together_ai"``, ``"groq"``,
        ``"openrouter"``, etc.) to the list of model IDs litellm can route
        to. Models already present in ``VERIFIED_MODELS`` for a given
        provider are excluded from this view.
    """
    return UNVERIFIED_MODELS_EXCLUDING_BEDROCK


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------


def infer_provider(model: str, api_base: str | None) -> str | None:
    """Resolve a model string to a litellm provider name.

    Caveat: litellm does **not** know about the ``"openhands"`` family.
    That family is routed through the OpenHands Cloud gateway rather than a
    litellm adapter, so ``infer_provider("openhands/<something>", ...)``
    returns ``None``. Callers that accept ``openhands/*`` strings must also
    look up :func:`list_verified_models`'s ``"openhands"`` entry themselves.

    Args:
        model: A model identifier. Either fully-qualified
            (``"openai/gpt-4o"``, ``"gemini/gemini-2.5-flash"``) or a bare
            model ID (``"gpt-4o"``) that litellm can auto-attribute.
        api_base: Optional custom endpoint URL. Only meaningful for
            providers like ``"azure"`` whose routing depends on the base URL.

    Returns:
        The litellm provider name (e.g. ``"openai"``, ``"together_ai"``,
        ``"vllm"``) when the string is routable, or ``None`` if the prefix
        is unknown, the bare model cannot be attributed, or the string is
        malformed.
    """
    return infer_litellm_provider(model=model, api_base=api_base)


# ---------------------------------------------------------------------------
# Credential probe
# ---------------------------------------------------------------------------


FailureKind = Literal[
    "auth", "connection", "bad_request", "rate_limit", "unknown"
]


@dataclass(frozen=True)
class KeyProbeOutcome:
    """Outcome of a live credential probe.

    Attributes:
        ok: ``True`` if the endpoint accepted the probe request.
        failure_kind: Absent when ``ok`` is ``True``; otherwise a short label
            classifying the failure so callers can render the right message.
        detail: Human-readable explanation of the failure (e.g. the str of
            the underlying exception). Empty when ``ok`` is ``True``.
    """

    ok: bool
    failure_kind: FailureKind | None = None
    detail: str = ""


def probe(
    model: str,
    api_key: str | None,
    api_base: str | None,
) -> KeyProbeOutcome:
    """Send a minimal completion request to verify credentials.

    Args:
        model: Fully-qualified model string understood by litellm
            (e.g. ``"openai/gpt-4o"``).
        api_key: The credential to test. ``None`` / empty string is forwarded
            as-is; endpoints that require no auth will accept this.
        api_base: Optional custom endpoint URL (used for self-hosted or
            proxy deployments).

    Returns:
        A :class:`KeyProbeOutcome` describing either success or the category
        of failure. Never raises.
    """
    messages = [{"role": "user", "content": "ping"}]
    try:
        litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key or None,
            api_base=api_base or None,
            max_tokens=10,
        )
    except AuthenticationError as exc:
        return KeyProbeOutcome(ok=False, failure_kind="auth", detail=str(exc))
    except APIConnectionError as exc:
        return KeyProbeOutcome(
            ok=False, failure_kind="connection", detail=str(exc)
        )
    except (Timeout, ServiceUnavailableError) as exc:
        return KeyProbeOutcome(
            ok=False, failure_kind="connection", detail=str(exc)
        )
    except BadRequestError as exc:
        return KeyProbeOutcome(
            ok=False, failure_kind="bad_request", detail=str(exc)
        )
    except RateLimitError as exc:
        return KeyProbeOutcome(
            ok=False, failure_kind="rate_limit", detail=str(exc)
        )
    except Exception as exc:
        return KeyProbeOutcome(ok=False, failure_kind="unknown", detail=str(exc))

    return KeyProbeOutcome(ok=True)
