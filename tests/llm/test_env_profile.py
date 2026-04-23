"""Tests for llm/env_profile.py."""

from dataclasses import dataclass

import pytest

from devdox_ai_sonar.llm.env_profile import ENV_PROFILE_NAME, profile_from_env
from devdox_ai_sonar.llm.profile import LLMProfile


@dataclass
class FakeSettings:
    """Minimal structural stand-in for devdox_ai_sonar.config.Settings."""

    LLM_MODEL: str = ""
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""


def test_returns_none_when_model_blank():
    assert profile_from_env(FakeSettings()) is None


@pytest.mark.parametrize("blank_model", ["", "   ", "\n", "\t"])
def test_treats_whitespace_only_model_as_blank(blank_model):
    assert profile_from_env(FakeSettings(LLM_MODEL=blank_model)) is None


def test_builds_profile_when_model_set():
    settings = FakeSettings(
        LLM_MODEL="gemini/gemini-2.5-flash",
        LLM_API_KEY="g-key",
        LLM_BASE_URL="",
    )
    profile = profile_from_env(settings)
    assert profile == LLMProfile(
        name=ENV_PROFILE_NAME,
        model="gemini/gemini-2.5-flash",
        api_key="g-key",
        base_url=None,
    )


def test_blank_base_url_becomes_none():
    """An empty LLM_BASE_URL env var must not populate ``base_url=""``
    on the profile, because downstream code treats None vs empty
    distinctly (None means "use provider default endpoint")."""
    settings = FakeSettings(
        LLM_MODEL="openai/gpt-4o",
        LLM_API_KEY="sk-x",
        LLM_BASE_URL="",
    )
    profile = profile_from_env(settings)
    assert profile is not None
    assert profile.base_url is None


def test_whitespace_base_url_becomes_none():
    settings = FakeSettings(
        LLM_MODEL="openai/gpt-4o",
        LLM_API_KEY="sk-x",
        LLM_BASE_URL="   ",
    )
    profile = profile_from_env(settings)
    assert profile is not None
    assert profile.base_url is None


def test_custom_endpoint_keeps_base_url():
    settings = FakeSettings(
        LLM_MODEL="openai/my-finetune",
        LLM_API_KEY="local-key",
        LLM_BASE_URL="https://vllm.example.com",
    )
    profile = profile_from_env(settings)
    assert profile is not None
    assert profile.base_url == "https://vllm.example.com"


def test_empty_api_key_preserved_as_empty_string():
    """A missing key is a valid configuration (endpoints accepting
    unauthenticated requests); the profile should carry the empty string
    rather than any kind of fallback."""
    settings = FakeSettings(LLM_MODEL="ollama/llama3", LLM_API_KEY="")
    profile = profile_from_env(settings)
    assert profile is not None
    assert profile.api_key == ""


def test_profile_name_is_reserved_sentinel():
    settings = FakeSettings(LLM_MODEL="openai/gpt-4o")
    profile = profile_from_env(settings)
    assert profile is not None
    assert profile.name == ENV_PROFILE_NAME
