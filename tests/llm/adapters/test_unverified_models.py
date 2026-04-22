"""Tests for the unverified_models adapter (API #2)."""

from devdox_ai_sonar.llm.adapters import unverified_models


def test_list_providers_returns_dict():
    result = unverified_models.list_providers()
    assert isinstance(result, dict)


def test_list_providers_includes_well_known_names():
    """The litellm catalogue is large and fluid, but a handful of providers
    are too well-established to disappear — they also happen to be the ones
    users are most likely to care about. If any of these drop out of the
    catalogue after an SDK upgrade, something has changed upstream and the
    expectation below needs re-evaluating, not silent updating."""
    result = unverified_models.list_providers()
    expected_some_of = {
        "openai",
        "gemini",
        "together_ai",
        "openrouter",
        "groq",
        "ollama",
    }
    assert expected_some_of.issubset(result.keys())


def test_each_family_maps_to_list_of_strings():
    result = unverified_models.list_providers()
    for family, models in result.items():
        assert isinstance(models, list), f"{family} is not a list"
        for model in models:
            assert isinstance(model, str), f"{family} contains non-string: {model!r}"


def test_catalog_is_substantial():
    """The full litellm-minus-Bedrock catalogue covers ~60+ providers. A
    sudden collapse to a tiny number would signal an SDK regression worth
    investigating before tests are updated."""
    result = unverified_models.list_providers()
    assert len(result) >= 20
