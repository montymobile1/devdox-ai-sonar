"""Build an :class:`LLMProfile` from environment-sourced :class:`Settings`.

This module implements layer 2 of the LLM precedence chain described in
:mod:`devdox_ai_sonar.config`. The CLI layer calls :func:`profile_from_env`
*after* checking CLI flags and *before* falling through to the
``default_profile`` saved in ``~/devdox/config.toml``.

The synthesised profile uses the reserved name ``"__env__"`` so it can be
distinguished from user-named profiles at a glance -- in particular, the
profile store never persists the ``"__env__"`` name; it's purely in-memory.
"""

from __future__ import annotations

from typing import Protocol

from devdox_ai_sonar.llm.profile import LLMProfile


__all__ = ["ENV_PROFILE_NAME", "profile_from_env"]


ENV_PROFILE_NAME = "__env__"


class _SettingsLike(Protocol):
    """Structural type covering the fields this module reads from
    :class:`devdox_ai_sonar.config.Settings`.

    Declared as a Protocol (rather than importing Settings) so tests can
    pass a lightweight object without constructing the full Settings
    class.
    """

    LLM_MODEL: str
    LLM_API_KEY: str
    LLM_BASE_URL: str


def profile_from_env(settings: _SettingsLike) -> LLMProfile | None:
    """Return an :class:`LLMProfile` if ``LLM_MODEL`` is set, else ``None``.

    A blank ``LLM_MODEL`` means "no env-level LLM configuration"; the
    caller should continue to the next precedence layer instead of
    treating this as a failure.

    Args:
        settings: Any object exposing the ``LLM_MODEL``, ``LLM_API_KEY``,
            and ``LLM_BASE_URL`` string attributes -- typically a live
            :class:`devdox_ai_sonar.config.Settings` instance.

    Returns:
        A profile with ``name = "__env__"``, or ``None`` when
        ``LLM_MODEL`` is blank/unset.
    """
    model = (settings.LLM_MODEL or "").strip()
    if not model:
        return None

    base_url = (settings.LLM_BASE_URL or "").strip() or None

    return LLMProfile(
        name=ENV_PROFILE_NAME,
        model=model,
        api_key=settings.LLM_API_KEY or "",
        base_url=base_url,
    )
