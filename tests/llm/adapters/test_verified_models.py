"""Tests for the verified_models adapter (API #1)."""

from devdox_ai_sonar.llm.adapters import verified_models


def test_list_providers_returns_dict():
    result = verified_models.list_providers()
    assert isinstance(result, dict)


def test_list_providers_has_expected_families():
    """OpenHands curates a small set of provider families.

    The list is hand-maintained by OpenHands and may grow, but these core
    eight should always be present. If this test fails after an SDK upgrade,
    review the new family list before updating the expected set.
    """
    result = verified_models.list_providers()
    expected = {
        "openhands",
        "anthropic",
        "openai",
        "mistral",
        "gemini",
        "deepseek",
        "moonshot",
        "minimax",
    }
    assert expected.issubset(result.keys())


def test_each_family_maps_to_list_of_strings():
    result = verified_models.list_providers()
    for family, models in result.items():
        assert isinstance(models, list), f"{family} is not a list"
        for model in models:
            assert isinstance(model, str), f"{family} contains non-string: {model!r}"


def test_openhands_family_is_non_empty():
    """The ``openhands`` family is the OpenHands Cloud gateway and must
    always surface at least one model (otherwise the UI has nothing to show
    under that picker entry)."""
    result = verified_models.list_providers()
    assert len(result["openhands"]) > 0
