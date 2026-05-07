"""Pure functions that build provider/model menus for the UI.

The functions here compose the catalogue calls in :mod:`adapters` into the
lists shown to the user during configuration. They carry no side effects
and never touch the OpenHands SDK directly — tests substitute the adapter
functions via ``monkeypatch.setattr`` to get deterministic, offline menu
output.

Menu shape:
  * The provider picker is the alphabetised **union** of verified and
    unverified provider names.
  * The model picker for a given provider is the concatenation of its
    verified entries followed by its unverified entries, as supplied by
    the upstream catalogues.
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
    """

    name: str


@dataclass(frozen=True)
class ModelInfo:
    """One entry in the model picker.

    Attributes:
        model_id: The bare model identifier (e.g. ``"gpt-4o"``, *not*
            ``"openai/gpt-4o"``). The UI concatenates this with the
            provider name when constructing the full model string.
    """

    model_id: str


def build_provider_menu() -> list[ProviderInfo]:
    """Return the provider picker entries.

    Union of verified and unverified provider names, alphabetised.
    """
    verified = adapters.list_verified_models()
    unverified = adapters.list_unverified_models()

    all_names = sorted(set(verified.keys()) | set(unverified.keys()))
    return [ProviderInfo(name=name) for name in all_names]


def build_model_menu(provider: str) -> list[ModelInfo]:
    """Return the model picker entries for a given provider.

    Verified entries come first, followed by the rest of the catalogue for
    that provider. The upstream unverified catalogue is already disjoint
    from the verified one for any given provider, so no de-duplication is
    performed here.

    Args:
        provider: A provider family name.

    Returns:
        A list of :class:`ModelInfo`, empty if the provider is unknown to
        both catalogues.
    """
    verified = adapters.list_verified_models().get(provider, [])
    unverified = adapters.list_unverified_models().get(provider, [])

    entries = [ModelInfo(model_id=m) for m in verified]
    entries += [ModelInfo(model_id=m) for m in unverified]
    return entries
