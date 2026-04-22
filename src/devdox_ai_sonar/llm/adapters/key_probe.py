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
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)


FailureKind = Literal[
    "auth",
    "connection",
    "bad_request",
    "not_found",
    "rate_limit",
    "unknown",
]


@dataclass(frozen=True)
class KeyProbeOutcome:
    """Outcome of a live credential probe.

    Attributes:
        ok: ``True`` if the endpoint accepted the probe request.
        failure_kind: Absent when ``ok`` is ``True``; otherwise a short
            label classifying the failure so callers can render the right
            message.
        detail: Human-readable explanation of the failure (typically the
            string form of the underlying exception). Empty when ``ok``
            is ``True``.
        provider: litellm's identifier for the upstream provider
            (``"openai"``, ``"gemini"``, ``"together_ai"`` …) when the
            exception carries one. Populated on failure; ``None`` on
            success or when the exception didn't expose it. Lets callers
            see at a glance whether the error came from the provider's
            own API as opposed to our code or the OpenHands gateway.
        status_code: HTTP status code from the upstream response when
            available (401, 429, 500 …). Useful for distinguishing
            "credentials rejected" from "endpoint down".
        exception_class: Fully-qualified class name of the underlying
            exception (e.g. ``"litellm.exceptions.RateLimitError"``).
            Makes it unambiguous in logs / UI that the failure was
            raised by litellm rather than our wrapper.
    """

    ok: bool
    failure_kind: FailureKind | None = None
    detail: str = ""
    provider: str | None = None
    status_code: int | None = None
    exception_class: str | None = None


def _qualified_class_name(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _failure_from_exception(kind: FailureKind, exc: BaseException) -> KeyProbeOutcome:
    """Build a :class:`KeyProbeOutcome` populated with whatever provenance
    the litellm exception exposes.

    Not every exception type carries ``llm_provider`` / ``status_code``
    (only the ones litellm raises internally do). ``getattr`` with
    ``None`` default keeps this safe for generic exceptions that fall
    into the ``"unknown"`` branch.
    """
    return KeyProbeOutcome(
        ok=False,
        failure_kind=kind,
        detail=str(exc),
        provider=getattr(exc, "llm_provider", None),
        status_code=getattr(exc, "status_code", None),
        exception_class=_qualified_class_name(exc),
    )


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
        A :class:`KeyProbeOutcome` describing either success or the
        category of failure, with provenance fields populated on
        failure. Never raises.
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
        return _failure_from_exception("auth", exc)
    except APIConnectionError as exc:
        return _failure_from_exception("connection", exc)
    except (Timeout, ServiceUnavailableError) as exc:
        return _failure_from_exception("connection", exc)
    except NotFoundError as exc:
        # Thrown when the model ID isn't known at the upstream endpoint
        # (e.g. a deprecated Gemini model still in litellm's catalogue).
        # Distinct from BadRequestError so the UI can steer the user
        # back to the model picker rather than re-prompting for a key.
        return _failure_from_exception("not_found", exc)
    except BadRequestError as exc:
        return _failure_from_exception("bad_request", exc)
    except RateLimitError as exc:
        return _failure_from_exception("rate_limit", exc)
    except Exception as exc:
        return _failure_from_exception("unknown", exc)

    return KeyProbeOutcome(ok=True)
