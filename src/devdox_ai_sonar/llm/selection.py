"""Pure functions that build provider/model menus for the UI.

The functions here compose the catalogue calls in :mod:`adapters` into the
lists shown to the user during configuration. They carry no side effects
and never touch the OpenHands SDK directly — tests substitute the adapter
functions via ``monkeypatch.setattr`` to get deterministic, offline menu
output.

Menu shape:
  * The provider picker is the alphabetised **union** of verified and
    unverified provider names. Each entry carries a ``verified`` flag so
    the UI can mark the curated ones with a star.
  * The model picker for a given provider concatenates its verified
    entries (marked) followed by its unverified entries (unmarked).
"""

from dataclasses import dataclass
from typing import Iterable

from devdox_ai_sonar.llm import adapters


__all__ = [
    "ProviderInfo",
    "ModelInfo",
    "build_provider_menu",
    "build_model_menu",
]


@dataclass(frozen=True)
class ProviderInfo:
    """One entry in the provider picker.

    Attributes:
        name: The provider family name (e.g. ``"openai"``, ``"together_ai"``,
            ``"openhands"``). Empty strings are not produced by the menu
            builders — if you see one, something is wrong upstream.
        verified: ``True`` when this provider appears in the
            OpenHands-curated ``VERIFIED_MODELS`` map. The UI should
            mark verified entries with a star so users can tell at a
            glance which are vouched for.
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
    included_providers: Iterable[str] = (),
) -> list[ProviderInfo]:
    """Return the provider picker entries.

    Union of verified and unverified provider names, alphabetised, with
    the ``verified`` flag set on entries present in the curated list.

    Filter semantics (inclusion narrows, exclusion always wins):

    * If ``included_providers`` is empty, no inclusion constraint.
    * If ``included_providers`` is non-empty, only those providers
      survive.
    * After any inclusion narrowing, any provider in
      ``excluded_providers`` is dropped.

    Unknown entries in either filter are silently ignored (so a typo
    in the developer-owned list can't crash the UI).
    """
    verified = adapters.list_verified_models()
    unverified = adapters.list_unverified_models()
    universe = set(verified.keys()) | set(unverified.keys())

    allowed = frozenset(included_providers)
    blocked = frozenset(excluded_providers)
    if allowed:
        universe &= allowed
    universe -= blocked

    return [
        ProviderInfo(name=name, verified=name in verified)
        for name in sorted(universe)
    ]


def build_model_menu(
    provider: str,
    *,
    excluded_models: Iterable[str] = (),
    included_models: Iterable[str] = (),
) -> list[ModelInfo]:
    """Return the model picker entries for a given provider.

    Verified entries come first (for ⭐-prefixed display), alphabetised
    among themselves, followed by the unverified catalogue, also
    alphabetised. Duplicates are collapsed.

    Filter semantics (same as :func:`build_provider_menu`): inclusion
    narrows, exclusion always wins. All filter strings are full
    ``provider/model-id`` form, so a ``gemini/gpt-4o`` typo cannot
    accidentally hide OpenAI's gpt-4o.

    Args:
        provider: A provider family name.
        excluded_models: Full ``provider/model-id`` strings to drop.
        included_models: Full ``provider/model-id`` strings to keep.
            Empty means "no inclusion constraint".

    Returns:
        A list of :class:`ModelInfo`, empty if the provider is unknown
        to both catalogues or if the filters leave nothing visible.
    """
    verified = adapters.list_verified_models().get(provider, [])
    unverified = adapters.list_unverified_models().get(provider, [])
    allowed = frozenset(included_models)
    blocked = frozenset(excluded_models)

    def visible(model_id: str) -> bool:
        full = f"{provider}/{model_id}"
        if allowed and full not in allowed:
            return False
        return full not in blocked

    entries: list[ModelInfo] = []
    seen: set[str] = set()
    for model_id in sorted(verified):
        if model_id in seen or not visible(model_id):
            continue
        seen.add(model_id)
        entries.append(ModelInfo(model_id=model_id, verified=True))
    for model_id in sorted(unverified):
        if model_id in seen or not visible(model_id):
            continue
        seen.add(model_id)
        entries.append(ModelInfo(model_id=model_id, verified=False))
    return entries
