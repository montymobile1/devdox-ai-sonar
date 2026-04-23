"""Tests for llm/exclusions.py.

The module exports frozenset data + one function
(``check_profile_against_filters``) that produces a warning string or
None. Every test patches the data via ``monkeypatch.setattr`` so the
shipped denylist/allowlist doesn't leak into the expectations.
"""

from devdox_ai_sonar.llm import exclusions
from devdox_ai_sonar.llm.exclusions import check_profile_against_filters
from devdox_ai_sonar.llm.profile import LLMProfile


# ---------------------------------------------------------------------------
# No filters -> no warning
# ---------------------------------------------------------------------------


def test_no_filters_no_warning(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_MODELS", frozenset())
    profile = LLMProfile(name="p", model="openai/gpt-4o", api_key="k")
    assert check_profile_against_filters(profile) is None


# ---------------------------------------------------------------------------
# Exclusion wins outright
# ---------------------------------------------------------------------------


def test_model_on_exclusion_list_warns_about_model(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "EXCLUDED_MODELS", frozenset({"gemini/gemini-1.5-flash"})
    )
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_MODELS", frozenset())
    profile = LLMProfile(name="p", model="gemini/gemini-1.5-flash", api_key="k")
    warning = check_profile_against_filters(profile)
    assert warning is not None
    assert "gemini/gemini-1.5-flash" in warning
    assert "exclusion list" in warning


def test_provider_on_exclusion_list_warns_about_provider(monkeypatch):
    monkeypatch.setattr(
        exclusions, "EXCLUDED_PROVIDERS", frozenset({"aws_polly"})
    )
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_MODELS", frozenset())
    profile = LLMProfile(name="p", model="aws_polly/something", api_key="k")
    warning = check_profile_against_filters(profile)
    assert warning is not None
    assert "aws_polly" in warning


def test_model_exclusion_takes_precedence_over_inclusion(monkeypatch):
    """Exclusion always wins even when the model is on an inclusion list."""
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "EXCLUDED_MODELS", frozenset({"openai/gpt-3.5"})
    )
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_MODELS", frozenset({"openai/gpt-3.5"})
    )
    profile = LLMProfile(name="p", model="openai/gpt-3.5", api_key="k")
    warning = check_profile_against_filters(profile)
    assert warning is not None
    assert "exclusion list" in warning


# ---------------------------------------------------------------------------
# Inclusion misses
# ---------------------------------------------------------------------------


def test_empty_inclusion_list_imposes_no_constraint(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_MODELS", frozenset())
    profile = LLMProfile(name="p", model="unknown/some-model", api_key="k")
    assert check_profile_against_filters(profile) is None


def test_model_not_on_non_empty_inclusion_list_warns(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_MODELS", frozenset({"openai/gpt-4o"})
    )
    profile = LLMProfile(name="p", model="openai/gpt-3.5", api_key="k")
    warning = check_profile_against_filters(profile)
    assert warning is not None
    assert "curated model list" in warning


def test_model_on_inclusion_list_passes(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_MODELS", frozenset({"openai/gpt-4o"})
    )
    profile = LLMProfile(name="p", model="openai/gpt-4o", api_key="k")
    assert check_profile_against_filters(profile) is None


def test_provider_not_on_non_empty_inclusion_list_warns(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_PROVIDERS", frozenset({"openai"})
    )
    monkeypatch.setattr(exclusions, "INCLUDED_MODELS", frozenset())
    profile = LLMProfile(name="p", model="anthropic/claude-4", api_key="k")
    warning = check_profile_against_filters(profile)
    assert warning is not None
    assert "curated provider list" in warning


def test_model_inclusion_overrides_provider_inclusion_miss(monkeypatch):
    """A model-specific allowlist entry (more specific) clears a broader
    provider inclusion miss -- carving out an exception rather than
    fencing the dev in."""
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_PROVIDERS", frozenset({"openai"})
    )
    monkeypatch.setattr(
        exclusions, "INCLUDED_MODELS", frozenset({"anthropic/claude-4"})
    )
    profile = LLMProfile(name="p", model="anthropic/claude-4", api_key="k")
    assert check_profile_against_filters(profile) is None
