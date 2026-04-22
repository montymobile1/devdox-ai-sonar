"""Tests for llm/exclusions.py.

Exercises the developer-owned exclusion lists and the runtime warning
helper that informs users when their saved profile references an
entry on the list. Tests patch the module-level constants via
``monkeypatch.setattr`` so they stay hermetic even as the shipped
lists evolve.
"""

from devdox_ai_sonar.llm import exclusions
from devdox_ai_sonar.llm.profile import LLMProfile


def test_constants_are_frozensets_of_strings():
    """Exclusion lists must be ``frozenset[str]`` so they're immutable
    at runtime and support O(1) membership tests inside the hot
    picker paths."""
    assert isinstance(exclusions.EXCLUDED_PROVIDERS, frozenset)
    assert isinstance(exclusions.EXCLUDED_MODELS, frozenset)
    for value in exclusions.EXCLUDED_PROVIDERS | exclusions.EXCLUDED_MODELS:
        assert isinstance(value, str)


def test_shipped_model_exclusions_follow_provider_slash_id_convention():
    """Every shipped ``EXCLUDED_MODELS`` entry must be a full
    ``provider/model-id`` string (the same convention the selection
    module uses), not a bare model name. A bare entry would never
    match anything and silently do nothing."""
    for entry in exclusions.EXCLUDED_MODELS:
        assert "/" in entry, (
            f"{entry!r} must be in 'provider/model-id' form; "
            "bare model names cannot match the menu data."
        )


def test_check_returns_none_for_clean_profile(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())

    profile = LLMProfile(
        name="clean", model="openai/gpt-4o", api_key="sk"
    )
    assert exclusions.check_profile_against_exclusions(profile) is None


def test_check_warns_on_excluded_model(monkeypatch):
    monkeypatch.setattr(
        exclusions,
        "EXCLUDED_MODELS",
        frozenset({"gemini/gemini-1.5-flash"}),
    )
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())

    profile = LLMProfile(
        name="stale",
        model="gemini/gemini-1.5-flash",
        api_key="g",
    )
    warning = exclusions.check_profile_against_exclusions(profile)
    assert warning is not None
    assert "gemini/gemini-1.5-flash" in warning
    assert "stale" in warning  # profile name surfaces in the message
    assert "update_provider" in warning  # nudges the user to the fix


def test_check_warns_on_excluded_provider(monkeypatch):
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(
        exclusions,
        "EXCLUDED_PROVIDERS",
        frozenset({"aws_polly"}),
    )

    profile = LLMProfile(
        name="polly",
        model="aws_polly/some-voice",
        api_key="x",
    )
    warning = exclusions.check_profile_against_exclusions(profile)
    assert warning is not None
    assert "aws_polly" in warning
    assert "polly" in warning


def test_model_exclusion_takes_precedence_over_provider_match(monkeypatch):
    """If a profile matches BOTH a full-model exclusion and a
    provider-family exclusion, the more-specific model-level message
    wins. A user excluding exactly one model of a provider shouldn't
    also get a broader 'whole provider is excluded' warning."""
    monkeypatch.setattr(
        exclusions,
        "EXCLUDED_MODELS",
        frozenset({"gemini/gemini-1.5-flash"}),
    )
    monkeypatch.setattr(
        exclusions, "EXCLUDED_PROVIDERS", frozenset({"gemini"})
    )

    profile = LLMProfile(
        name="p", model="gemini/gemini-1.5-flash", api_key="g"
    )
    warning = exclusions.check_profile_against_exclusions(profile)
    assert warning is not None
    # Specific model wording ("uses 'gemini/gemini-1.5-flash'") appears;
    # the coarser provider wording ("uses provider 'gemini'") does not.
    assert "'gemini/gemini-1.5-flash'" in warning
    assert "uses provider 'gemini'" not in warning


def test_check_ignores_clean_custom_style_profile(monkeypatch):
    """A profile pointing at a self-hosted vLLM with a fully-qualified
    but never-excluded model must not trip either list."""
    monkeypatch.setattr(
        exclusions,
        "EXCLUDED_MODELS",
        frozenset({"gemini/gemini-1.5-flash"}),
    )
    monkeypatch.setattr(
        exclusions,
        "EXCLUDED_PROVIDERS",
        frozenset({"aws_polly"}),
    )

    profile = LLMProfile(
        name="local",
        model="vllm/my-local-model",
        api_key="",
        base_url="https://vllm.company.internal/v1",
    )
    assert exclusions.check_profile_against_exclusions(profile) is None
