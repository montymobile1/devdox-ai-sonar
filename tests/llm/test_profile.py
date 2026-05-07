"""Tests for llm/profile.py."""

import pytest

from devdox_ai_sonar.llm.profile import LLMProfile


def test_construct_with_required_fields_only():
    profile = LLMProfile(name="test", model="openai/gpt-4o")
    assert profile.name == "test"
    assert profile.model == "openai/gpt-4o"
    assert profile.api_key == ""
    assert profile.base_url is None


def test_to_toml_dict_omits_base_url_when_none():
    """A profile backed by the provider's default endpoint should not
    carry an empty base_url field into config.toml -- the file stays
    readable."""
    profile = LLMProfile(name="p", model="openai/gpt-4o", api_key="sk-x")
    payload = profile.to_toml_dict()
    assert payload == {"name": "p", "model": "openai/gpt-4o", "api_key": "sk-x"}
    assert "base_url" not in payload


def test_to_toml_dict_includes_base_url_when_set():
    profile = LLMProfile(
        name="vllm",
        model="openai/my-model",
        api_key="",
        base_url="https://vllm.example.com",
    )
    payload = profile.to_toml_dict()
    assert payload["base_url"] == "https://vllm.example.com"


def test_from_toml_dict_round_trips():
    original = LLMProfile(
        name="p",
        model="gemini/gemini-2.5-flash",
        api_key="k",
        base_url="https://x.example.com",
    )
    restored = LLMProfile.from_toml_dict(original.to_toml_dict())
    assert restored == original


def test_from_toml_dict_tolerates_missing_optional_fields():
    """A minimal entry (just name + model) should load, with api_key and
    base_url taking their defaults."""
    profile = LLMProfile.from_toml_dict({"name": "min", "model": "openai/gpt-4o"})
    assert profile.api_key == ""
    assert profile.base_url is None


def test_from_toml_dict_requires_name_and_model():
    with pytest.raises(KeyError):
        LLMProfile.from_toml_dict({"model": "openai/gpt-4o"})
    with pytest.raises(KeyError):
        LLMProfile.from_toml_dict({"name": "x"})


@pytest.mark.parametrize(
    ("model", "expected_family"),
    [
        ("openai/gpt-4o", "openai"),
        ("gemini/gemini-2.5-flash", "gemini"),
        ("together_ai/meta-llama/Llama-3-70b-chat", "together_ai"),
        ("openrouter/anthropic/claude-sonnet-4", "openrouter"),
        ("openhands/gpt-5.4", "openhands"),
        # No separator -- whole string is returned (bare model case).
        ("gpt-4o", "gpt-4o"),
    ],
)
def test_family_extracts_prefix_correctly(model, expected_family):
    profile = LLMProfile(name="x", model=model)
    assert profile.family() == expected_family
