"""Validation for user-supplied model strings and API keys.

Two entry points, both returning :data:`Result` so callers can branch on
failure without dealing with exceptions:

- :func:`validate_model_string` -- checks the model string can be routed.
- :func:`probe_api_key` -- live-probes the endpoint to verify credentials.

Strict vs lenient validation
----------------------------
:func:`validate_model_string` accepts a ``strict`` keyword argument
(default ``True``). When strict, any model whose prefix the adapters
cannot resolve is rejected. When lenient, such a model is still accepted
if the caller supplies a ``base_url`` -- taken as "the user is deliberately
pointing at a custom endpoint and accepts responsibility". The live probe
is the real safety net in either mode.

``openhands/*`` special case
----------------------------
The OpenHands Cloud gateway is not a litellm provider, so
:func:`openhands.sdk.llm.utils.litellm_provider.infer_litellm_provider`
returns ``None`` for models in that namespace. This module compensates by
falling back to a direct lookup in the verified catalog when the prefix
is ``openhands/``.
"""

from __future__ import annotations

from dataclasses import dataclass

from devdox_ai_sonar.llm.adapters import (
    key_probe,
    provider_inference,
    verified_models,
)
from devdox_ai_sonar.llm.errors import (
    Err,
    InvalidModelFormatError,
    KeyProbeError,
    Ok,
    Result,
    UnknownProviderError,
)


__all__ = ["ValidatedModel", "validate_model_string", "probe_api_key"]


_OPENHANDS_FAMILY = "openhands"
_MODEL_SEPARATOR = "/"


@dataclass(frozen=True)
class ValidatedModel:
    """The successful outcome of :func:`validate_model_string`.

    Attributes:
        model: The full model string as supplied by the caller (not
            rewritten).
        provider: The resolved provider name, or ``None`` when the model
            was accepted leniently against a caller-supplied ``base_url``
            with an unknown prefix.
        base_url: Echo of the caller-supplied ``base_url`` (including
            ``None``), carried through so downstream code doesn't have to
            re-thread it.
    """

    model: str
    provider: str | None
    base_url: str | None


def validate_model_string(
    model: str,
    base_url: str | None,
    *,
    strict: bool = True,
) -> Result[ValidatedModel, InvalidModelFormatError | UnknownProviderError]:
    """Check that ``model`` is a routable model string.

    Args:
        model: Candidate model identifier. The OpenHands/litellm canonical
            form is ``provider/model-id``; bare identifiers are only
            accepted for the small set of providers litellm can
            auto-infer (notably raw OpenAI model IDs).
        base_url: Optional custom endpoint URL. In strict mode the
            validator considers this advisory. In lenient mode, a supplied
            ``base_url`` lets unknown-prefix models pass -- the caller
            accepts responsibility for routing correctness.
        strict: When ``True`` (default), reject any model the adapters
            cannot resolve. When ``False``, accept such models if
            ``base_url`` is provided.

    Returns:
        ``Ok(ValidatedModel)`` on success, ``Err(InvalidModelFormatError)``
        for structurally broken input, or ``Err(UnknownProviderError)``
        when the prefix doesn't map to any known provider.
    """
    format_err = _check_model_format(model)
    if format_err is not None:
        return Err(format_err)

    if _is_openhands_verified(model):
        return Ok(ValidatedModel(model=model, provider=_OPENHANDS_FAMILY, base_url=base_url))

    provider = provider_inference.infer_provider(model=model, api_base=base_url)
    if provider is not None:
        return Ok(ValidatedModel(model=model, provider=provider, base_url=base_url))

    # Provider is unresolved. Lenient mode accepts only with a base_url.
    if not strict and base_url:
        return Ok(ValidatedModel(model=model, provider=None, base_url=base_url))

    return Err(UnknownProviderError(model=model))


def probe_api_key(
    model: str,
    api_key: str | None,
    base_url: str | None,
    *,
    validate_key: bool = True,
) -> Result[None, KeyProbeError]:
    """Live-probe the endpoint to verify that credentials work.

    Args:
        model: The validated model string (typically the one returned by
            :func:`validate_model_string`).
        api_key: Credential to test. May be empty / ``None`` for
            endpoints that accept unauthenticated requests.
        base_url: Optional custom endpoint URL.
        validate_key: When ``False``, return ``Ok(None)`` immediately
            without contacting the network. Useful for offline
            configuration flows.

    Returns:
        ``Ok(None)`` on success (or when ``validate_key`` is disabled),
        otherwise ``Err(KeyProbeError)`` with a categorised failure kind.
    """
    if not validate_key:
        return Ok(None)

    outcome = key_probe.probe(model=model, api_key=api_key, api_base=base_url)
    if outcome.ok:
        return Ok(None)

    # outcome.failure_kind is guaranteed non-None when ok is False, but the
    # type checker can't see that; coerce to the expected literal set.
    return Err(
        KeyProbeError(
            kind=outcome.failure_kind or "unknown",
            detail=outcome.detail,
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_model_format(model: str) -> InvalidModelFormatError | None:
    """Catch structural problems before we query any adapter."""
    if not model or not model.strip():
        return InvalidModelFormatError(model=model, reason="empty")
    return None


def _is_openhands_verified(model: str) -> bool:
    """Does this model belong to the OpenHands Cloud verified family?

    litellm doesn't know how to route ``openhands/*`` strings (that's the
    OpenHands gateway's job), so we can't rely on
    :func:`provider_inference.infer_provider` for this family -- we look
    in the verified catalog directly instead.
    """
    prefix = _OPENHANDS_FAMILY + _MODEL_SEPARATOR
    if not model.startswith(prefix):
        return False
    model_id = model[len(prefix):]
    if not model_id:
        return False
    verified = verified_models.list_providers().get(_OPENHANDS_FAMILY, [])
    return model_id in verified
