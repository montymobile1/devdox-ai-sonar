"""Error types for the ``llm`` package.

Two kinds of error surface, used intentionally for different purposes:

**Exceptions** (``raise`` + ``try/except``) are for unrecoverable conditions
where the caller is not expected to branch on the outcome -- the program
either stops or logs and aborts. These derive from
:class:`devdox_ai_sonar.utils.exceptions.DevDoxSonarError` so they fit the
existing repo-wide exception hierarchy.

**Branchable error dataclasses** (carried inside :data:`Result`) are for
validation and probe outcomes where the caller is *expected* to inspect the
failure and react differently (retry with a new value, show a specific
message, fall back). They are frozen dataclasses -- not exceptions -- so
callers cannot accidentally raise them and bypass the ``Result`` discipline.

Rule of thumb: if the UI flow has a "retry" branch for a case, it's a
branchable error. If the only sensible response is "log and abort", it's an
exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from devdox_ai_sonar.utils.exceptions import DevDoxSonarError
from devdox_ai_sonar.utils.result import Err, Ok, Result


__all__ = [
    # Exceptions
    "OpenHandsSdkUnavailableError",
    "ConfigStorageError",
    "ProfileNotFoundError",
    # Branchable error dataclasses
    "InvalidModelFormatError",
    "UnknownProviderError",
    "KeyProbeError",
    # Result re-exports (so llm modules don't also import utils.result)
    "Ok",
    "Err",
    "Result",
]


# ---------------------------------------------------------------------------
# Exceptions -- for unrecoverable / programmer-error conditions.
# ---------------------------------------------------------------------------


class OpenHandsSdkUnavailableError(DevDoxSonarError):
    """Raised when the OpenHands SDK cannot be imported or initialised.

    Indicates an installation or environment problem, not user input. No
    retry path -- the application cannot proceed without the SDK.
    """


class ConfigStorageError(DevDoxSonarError):
    """Raised when ``config.toml`` cannot be read, written, or interpreted.

    Also raised when the file carries a schema that is no longer
    supported (e.g. an older ``[[llm.providers]]`` array), in which
    case the caller is told to delete the file and reconfigure.
    """


class ProfileNotFoundError(DevDoxSonarError):
    """Raised when a profile lookup fails.

    The caller asked for a profile by name (or expected a default) and none
    exists. Typically surfaces from
    :func:`devdox_ai_sonar.llm.profile_store.update_profile` /
    :func:`delete_profile` when the target is missing.
    """


# ---------------------------------------------------------------------------
# Branchable error dataclasses -- for Result[..., E] failure arms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvalidModelFormatError:
    """The model string is structurally malformed.

    Distinct from :class:`UnknownProviderError` (which applies when the
    structure is fine but the prefix isn't routable).

    Attributes:
        model: The offending input as supplied by the caller.
        reason: Short, user-facing explanation (e.g. ``"empty"``,
            ``"missing provider prefix (expected provider/model-name)"``).
    """

    model: str
    reason: str


@dataclass(frozen=True)
class UnknownProviderError:
    """The model prefix is not a provider that can be routed.

    Applies when :func:`openhands.sdk.llm.utils.litellm_provider.infer_litellm_provider`
    returns ``None`` *and* the special ``openhands/*`` family lookup does
    not match either.

    Attributes:
        model: The full input model string.
    """

    model: str


# The failure taxonomy mirrors
# :class:`devdox_ai_sonar.llm.adapters.key_probe.KeyProbeOutcome.failure_kind`
# plus the "format" case (caught before we ever probe the endpoint).
KeyProbeFailureKind = Literal[
    "auth",
    "connection",
    "bad_request",
    "not_found",
    "rate_limit",
    "unknown",
]


@dataclass(frozen=True)
class KeyProbeError:
    """The live key-probe request could not be completed successfully.

    Attributes:
        kind: Short label classifying the failure so the UI can render
            an actionable message (e.g. "Key rejected" vs "Endpoint
            unreachable").
        detail: Human-readable detail, typically the underlying
            exception's ``str()`` representation.
        provider: litellm's identifier for the upstream provider when
            the exception exposed one (``"gemini"``, ``"openai"``,
            ``"together_ai"`` …). Lets callers tell whether the error
            came from the provider's own API or from somewhere else.
        status_code: Upstream HTTP status code (401, 429, 500 …) when
            the underlying exception carried it.
        exception_class: Fully-qualified class name of the underlying
            exception (e.g. ``"litellm.exceptions.RateLimitError"``).
            Makes it unambiguous which library raised the failure.
    """

    kind: KeyProbeFailureKind
    detail: str
    provider: str | None = None
    status_code: int | None = None
    exception_class: str | None = None
