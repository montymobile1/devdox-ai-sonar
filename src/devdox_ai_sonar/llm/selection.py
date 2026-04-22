"""Pure functions that build provider/model menus for the UI.

The functions here compose the :mod:`adapters.verified_models` and
:mod:`adapters.unverified_models` adapters into the lists shown to the user
during configuration. They carry no side effects and never touch the
OpenHands SDK directly -- tests substitute the two adapters via
``monkeypatch.setattr`` to get deterministic, offline menu output.

Menu shape (provider + model):
  * The provider picker is the **union** of verified and unverified
    provider names, alphabetised. Verified entries are flagged for the UI
    to render with a ``⭐`` marker.
  * The model picker for a given provider is **verified first, then
    advanced**. :data:`UNVERIFIED_MODELS_EXCLUDING_BEDROCK` already
    de-duplicates against :data:`VERIFIED_MODELS`, so no further merging
    is needed.

Custom vs real provider:
  The string :data:`CUSTOM_PROVIDER_SENTINEL` represents the "Custom"
  escape-hatch option. Callers should *not* try to look up models for it
  via :func:`build_model_menu`; they prompt the user for a model string
  directly instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from devdox_ai_sonar.llm.adapters import unverified_models, verified_models


__all__ = [
    "CUSTOM_PROVIDER_SENTINEL",
    "ProviderInfo",
    "ModelInfo",
    "build_provider_menu",
    "build_model_menu",
]


# Internal sentinel used by the UI layer to represent the "Custom" option.
# Chosen to be obviously synthetic (double underscores) so it cannot
# collide with any real litellm provider name.
CUSTOM_PROVIDER_SENTINEL: str = "__custom__"


@dataclass(frozen=True)
class ProviderInfo:
    """One entry in the provider picker.

    Attributes:
        name: The provider family name (e.g. ``"openai"``, ``"together_ai"``,
            ``"openhands"``). Empty strings are not produced by the menu
            builders -- if you see one, something is wrong upstream.
        verified: ``True`` when this provider appears in
            ``VERIFIED_MODELS``. The UI should mark verified entries with
            a star so users can tell at a glance which are vouched for.
    """

    name: str
    verified: bool


@dataclass(frozen=True)
class ModelInfo:
    """One entry in the model picker.

    Attributes:
        model_id: The bare model identifier (e.g. ``"gpt-4o"``, *not*
            ``"openai/gpt-4o"``). The UI concatenates this with the
            provider name when constructing the full model string.
        verified: ``True`` when this exact model appears in the verified
            catalogue for its provider.
    """

    model_id: str
    verified: bool


def build_provider_menu(
    *,
    excluded_providers: Iterable[str] = (),
) -> list[ProviderInfo]:
    """Return the provider picker entries.

    Union of verified and unverified provider names, alphabetised, with
    the ``verified`` flag set on entries present in the curated list.
    Does **not** include the "Custom" option -- the UI layer appends
    that itself after this list is rendered.

    Args:
        excluded_providers: Provider names the user has hidden via
            ``[llm].excluded_providers`` in config.toml. Matching is
            exact string equality against the union of both
            catalogues' keys; unknown exclusion entries are silently
            ignored (so a typo in the config doesn't crash the UI).
    """
    verified = verified_models.list_providers()
    unverified = unverified_models.list_providers()
    blocked = frozenset(excluded_providers)

    all_names = sorted(
        (set(verified.keys()) | set(unverified.keys())) - blocked
    )
    return [ProviderInfo(name=name, verified=name in verified) for name in all_names]


def build_model_menu(
    provider: str,
    *,
    excluded_models: Iterable[str] = (),
) -> list[ModelInfo]:
    """Return the model picker entries for a given provider.

    Verified entries come first (for ⭐-prefixed display), followed by
    the rest of the catalogue for that provider.
    ``UNVERIFIED_MODELS_EXCLUDING_BEDROCK`` is already disjoint from
    ``VERIFIED_MODELS`` for any given provider, so no de-duplication
    is performed here.

    Args:
        provider: A real provider family name. Passing
            :data:`CUSTOM_PROVIDER_SENTINEL` is a caller bug -- use
            the custom-input prompt instead.
        excluded_models: Models the user has hidden via
            ``[llm].excluded_models`` in config.toml, in full
            ``provider/model-id`` form (so a ``"gemini/gpt-4o"``
            typo doesn't accidentally hide OpenAI's gpt-4o).
            Matching is exact.

    Returns:
        A list of :class:`ModelInfo`, empty if the provider is
        unknown to both catalogues or if every model has been
        excluded.
    """
    verified = verified_models.list_providers().get(provider, [])
    unverified = unverified_models.list_providers().get(provider, [])
    blocked = frozenset(excluded_models)

    def visible(model_id: str) -> bool:
        return f"{provider}/{model_id}" not in blocked

    entries = [
        ModelInfo(model_id=m, verified=True) for m in verified if visible(m)
    ]
    entries += [
        ModelInfo(model_id=m, verified=False) for m in unverified if visible(m)
    ]
    return entries
