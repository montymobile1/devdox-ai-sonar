"""Tests for llm/exclusions.py.

Exercises the developer-owned exclusion and inclusion lists and the
runtime warning helper that informs users when their saved profile
references a filtered entry. Tests patch the module-level constants
via ``monkeypatch.setattr`` so they stay hermetic even as the shipped
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
    assert exclusions.check_profile_against_filters(profile) is None


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
    warning = exclusions.check_profile_against_filters(profile)
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
    warning = exclusions.check_profile_against_filters(profile)
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
    warning = exclusions.check_profile_against_filters(profile)
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
    assert exclusions.check_profile_against_filters(profile) is None


# ---------------------------------------------------------------------------
# Inclusion lists (INCLUDED_PROVIDERS / INCLUDED_MODELS)
# ---------------------------------------------------------------------------


def test_inclusion_constants_default_to_empty():
    """Ship day: inclusion lists are empty so behavior matches the
    legacy denylist-only regime. Anyone opening the module should see
    two obvious 'off' switches."""
    assert exclusions.INCLUDED_PROVIDERS == frozenset()
    assert exclusions.INCLUDED_MODELS == frozenset()


def test_inclusion_constants_are_frozensets_of_strings():
    """Same runtime guarantees as the exclusion lists -- immutable and
    O(1) membership checks on the hot picker path."""
    assert isinstance(exclusions.INCLUDED_PROVIDERS, frozenset)
    assert isinstance(exclusions.INCLUDED_MODELS, frozenset)
    for value in exclusions.INCLUDED_PROVIDERS | exclusions.INCLUDED_MODELS:
        assert isinstance(value, str)


def test_check_warns_on_model_not_in_included_models(monkeypatch):
    """Non-empty ``INCLUDED_MODELS`` acts as an allowlist; a profile
    whose model is unlisted gets a curated-list warning."""
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions,
        "INCLUDED_MODELS",
        frozenset({"openai/gpt-4o"}),
    )

    profile = LLMProfile(
        name="stray", model="openai/gpt-3.5", api_key="sk"
    )
    warning = exclusions.check_profile_against_filters(profile)
    assert warning is not None
    assert "openai/gpt-3.5" in warning
    assert "curated" in warning  # distinguishes from exclusion warning


def test_check_warns_on_provider_not_in_included_providers(monkeypatch):
    """Non-empty ``INCLUDED_PROVIDERS`` narrows by family; a profile
    targeting an unlisted family gets a curated-provider warning."""
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_PROVIDERS", frozenset({"openai", "anthropic"})
    )
    monkeypatch.setattr(exclusions, "INCLUDED_MODELS", frozenset())

    profile = LLMProfile(
        name="off-list", model="groq/llama-3", api_key="k"
    )
    warning = exclusions.check_profile_against_filters(profile)
    assert warning is not None
    assert "groq" in warning
    assert "curated" in warning


def test_check_empty_inclusion_lists_are_no_op(monkeypatch):
    """Empty allowlists must act as 'no constraint' -- any profile
    passes when both inclusion lists are empty and no exclusion fires."""
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(exclusions, "INCLUDED_MODELS", frozenset())

    profile = LLMProfile(
        name="anything", model="totally/made-up", api_key=""
    )
    assert exclusions.check_profile_against_filters(profile) is None


def test_check_model_inclusion_match_silences_provider_inclusion_miss(monkeypatch):
    """A profile explicitly on ``INCLUDED_MODELS`` must not also trip
    ``INCLUDED_PROVIDERS`` just because its family happens to be
    outside the provider allowlist. The more specific match wins."""
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_PROVIDERS", frozenset({"openai"})
    )
    monkeypatch.setattr(
        exclusions, "INCLUDED_MODELS", frozenset({"anthropic/claude-4"})
    )

    profile = LLMProfile(
        name="exception", model="anthropic/claude-4", api_key="k"
    )
    assert exclusions.check_profile_against_filters(profile) is None


def test_exclusion_wins_over_inclusion_match(monkeypatch):
    """A model listed on BOTH ``INCLUDED_MODELS`` and ``EXCLUDED_MODELS``
    must still trip the exclusion warning. 'No' always wins."""
    monkeypatch.setattr(
        exclusions,
        "EXCLUDED_MODELS",
        frozenset({"openai/gpt-4o"}),
    )
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "INCLUDED_PROVIDERS", frozenset()
    )
    monkeypatch.setattr(
        exclusions, "INCLUDED_MODELS", frozenset({"openai/gpt-4o"})
    )

    profile = LLMProfile(
        name="conflict", model="openai/gpt-4o", api_key="k"
    )
    warning = exclusions.check_profile_against_filters(profile)
    assert warning is not None
    # Exclusion wording wins; curated wording must not appear.
    assert "exclusion list" in warning
    assert "curated" not in warning
