"""Adapter for litellm's live credential check.

Wraps ``litellm.completion`` in a 10-token probe call: if the endpoint
accepts the request, the key is valid; if it rejects with an auth error, it
is not; if the endpoint is unreachable, that is a separate kind of failure.

The adapter categorises failures into a small closed vocabulary
(:class:`KeyProbeOutcome`) so the validation layer can produce actionable
error messages without having to import litellm exception types itself —
that boundary is the whole point of having adapters.
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


FailureKind = Literal["auth", "connection", "bad_request", "rate_limit", "unknown"]


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
            (e.g. ``"openai/gpt-4o"``, ``"together_ai/..."``).
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
        return KeyProbeOutcome(ok=False, failure_kind="connection", detail=str(exc))
    except (Timeout, ServiceUnavailableError) as exc:
        return KeyProbeOutcome(ok=False, failure_kind="connection", detail=str(exc))
    except BadRequestError as exc:
        return KeyProbeOutcome(ok=False, failure_kind="bad_request", detail=str(exc))
    except RateLimitError as exc:
        return KeyProbeOutcome(ok=False, failure_kind="rate_limit", detail=str(exc))
    except Exception as exc:
        return KeyProbeOutcome(ok=False, failure_kind="unknown", detail=str(exc))

    return KeyProbeOutcome(ok=True)
