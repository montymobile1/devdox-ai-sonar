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


def build_provider_menu() -> list[ProviderInfo]:
    """Return the provider picker entries.

    Union of verified and unverified provider names, alphabetised, with
    the ``verified`` flag set on entries present in the curated list.
    """
    verified = adapters.list_verified_models()
    unverified = adapters.list_unverified_models()

    all_names = sorted(set(verified.keys()) | set(unverified.keys()))
    return [
        ProviderInfo(name=name, verified=name in verified)
        for name in all_names
    ]


def build_model_menu(provider: str) -> list[ModelInfo]:
    """Return the model picker entries for a given provider.

    Verified entries come first (for ⭐-prefixed display), alphabetised
    among themselves, followed by the unverified catalogue, also
    alphabetised. Duplicates are collapsed: the upstream
    ``UNVERIFIED_MODELS_EXCLUDING_BEDROCK`` catalogue contains repeated
    ids within its own lists for many providers, and the
    verified-vs-unverified disjointness is a convention rather than a
    guarantee -- we rely on neither.

    Args:
        provider: A provider family name.

    Returns:
        A list of :class:`ModelInfo`, empty if the provider is unknown
        to both catalogues.
    """
    verified = adapters.list_verified_models().get(provider, [])
    unverified = adapters.list_unverified_models().get(provider, [])

    entries: list[ModelInfo] = []
    seen: set[str] = set()
    for model_id in sorted(verified):
        if model_id in seen:
            continue
        seen.add(model_id)
        entries.append(ModelInfo(model_id=model_id, verified=True))
    for model_id in sorted(unverified):
        if model_id in seen:
            continue
        seen.add(model_id)
        entries.append(ModelInfo(model_id=model_id, verified=False))
    return entries
