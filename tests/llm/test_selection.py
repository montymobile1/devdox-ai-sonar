"""Tests for llm/selection.py.

Every test replaces the two catalog adapter functions via
``monkeypatch.setattr`` so the menu builders run against controlled fixture
data instead of the live OpenHands SDK.
"""

from devdox_ai_sonar.llm import adapters, selection
from devdox_ai_sonar.llm.selection import (
    ModelInfo,
    ProviderInfo,
    build_model_menu,
    build_provider_menu,
)


def _fake_catalogs(monkeypatch, *, verified: dict, unverified: dict):
    monkeypatch.setattr(adapters, "list_verified_models", lambda: verified)
    monkeypatch.setattr(adapters, "list_unverified_models", lambda: unverified)


def test_provider_menu_merges_and_alphabetises(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"]},
        unverified={"groq": ["llama-3"], "together_ai": ["mixtral"]},
    )
    result = build_provider_menu()
    names = [p.name for p in result]
    assert names == ["groq", "openai", "together_ai"]


def test_provider_menu_dedupes_providers_in_both_catalogs(monkeypatch):
    """A provider listed in both catalogues should appear only once."""
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"]},
        unverified={"openai": ["gpt-3.5"], "groq": ["llama-3"]},
    )
    names = [p.name for p in build_provider_menu()]
    assert names.count("openai") == 1


def test_provider_menu_empty_catalogs_returns_empty(monkeypatch):
    _fake_catalogs(monkeypatch, verified={}, unverified={})
    assert build_provider_menu() == []


def test_model_menu_verified_first(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o", "gpt-5"]},
        unverified={"openai": ["gpt-3.5", "sora-2"]},
    )
    result = build_model_menu("openai")
    assert result == [
        ModelInfo(model_id="gpt-4o", verified=True),
        ModelInfo(model_id="gpt-5", verified=True),
        ModelInfo(model_id="gpt-3.5", verified=False),
        ModelInfo(model_id="sora-2", verified=False),
    ]


def test_model_menu_verified_only_when_no_advanced_entries(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"gemini": ["gemini-3.1-pro-preview"]},
        unverified={},
    )
    result = build_model_menu("gemini")
    assert result == [ModelInfo(model_id="gemini-3.1-pro-preview", verified=True)]


def test_model_menu_advanced_only_when_not_in_verified(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={},
        unverified={"together_ai": ["mixtral", "llama"]},
    )
    result = build_model_menu("together_ai")
    assert all(not m.verified for m in result)
    # Alpha-sorted within the unverified group.
    assert [m.model_id for m in result] == ["llama", "mixtral"]


def test_model_menu_dedupes_repeated_entries(monkeypatch):
    """The upstream unverified catalogue ships duplicates inside its
    per-provider lists (e.g. gemini has ~54 repeated ids). A model
    listed in both verified and unverified collapses to the verified
    entry. Repeated ids within a single list collapse to one entry."""
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o", "gpt-5"]},
        unverified={
            "openai": [
                "gpt-4o-mini",
                "gpt-3.5",
                "gpt-4o-mini",   # intra-list duplicate
                "gpt-4o",        # duplicate of a verified entry
                "gpt-3.5",       # intra-list duplicate
            ]
        },
    )
    result = build_model_menu("openai")
    # Verified first (alpha), then unverified (alpha), each unique.
    assert result == [
        ModelInfo(model_id="gpt-4o", verified=True),
        ModelInfo(model_id="gpt-5", verified=True),
        ModelInfo(model_id="gpt-3.5", verified=False),
        ModelInfo(model_id="gpt-4o-mini", verified=False),
    ]


def test_model_menu_unknown_provider_returns_empty(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"]},
        unverified={"groq": ["llama-3"]},
    )
    assert build_model_menu("never-heard-of-it") == []


# ---------------------------------------------------------------------------
# Filter lists (exclusion + inclusion)
# ---------------------------------------------------------------------------


def test_provider_menu_excluded_providers_dropped(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"]},
        unverified={"stability": ["sd-xl"], "groq": ["llama-3"]},
    )
    names = [p.name for p in build_provider_menu(
        excluded_providers={"stability"}
    )]
    assert "stability" not in names
    assert "openai" in names
    assert "groq" in names


def test_provider_menu_inclusion_narrows_to_listed(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"], "anthropic": ["claude-4"]},
        unverified={"groq": ["llama-3"]},
    )
    names = [p.name for p in build_provider_menu(
        included_providers={"openai"}
    )]
    assert names == ["openai"]


def test_provider_menu_exclusion_wins_over_inclusion(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"], "anthropic": ["claude-4"]},
        unverified={},
    )
    names = [p.name for p in build_provider_menu(
        included_providers={"openai", "anthropic"},
        excluded_providers={"openai"},
    )]
    assert names == ["anthropic"]


def test_model_menu_excluded_models_dropped(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o", "gpt-5"]},
        unverified={"openai": ["gpt-3.5"]},
    )
    ids = [m.model_id for m in build_model_menu(
        "openai", excluded_models={"openai/gpt-5"}
    )]
    assert "gpt-5" not in ids
    assert "gpt-4o" in ids
    assert "gpt-3.5" in ids


def test_model_menu_inclusion_narrows_to_listed(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o", "gpt-5"]},
        unverified={"openai": ["gpt-3.5"]},
    )
    ids = [m.model_id for m in build_model_menu(
        "openai", included_models={"openai/gpt-4o"}
    )]
    assert ids == ["gpt-4o"]


def test_model_menu_exclusion_wins_over_inclusion(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o", "gpt-5"]},
        unverified={},
    )
    ids = [m.model_id for m in build_model_menu(
        "openai",
        included_models={"openai/gpt-4o", "openai/gpt-5"},
        excluded_models={"openai/gpt-5"},
    )]
    assert ids == ["gpt-4o"]


def test_model_menu_filter_is_scoped_to_matching_provider(monkeypatch):
    """A filter string is the full ``provider/model-id`` form; it cannot
    accidentally hide a same-named model under a different provider."""
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"], "gemini": ["gpt-4o"]},
        unverified={},
    )
    openai_ids = [m.model_id for m in build_model_menu(
        "openai", excluded_models={"gemini/gpt-4o"}
    )]
    assert openai_ids == ["gpt-4o"]   # unaffected


def test_selection_does_not_touch_sdk_directly():
    """Selection imports only from the adapters module. If someone adds a
    direct openhands / litellm import, this breaks."""
    import inspect

    source = inspect.getsource(selection)
    assert "from openhands" not in source
    assert "import openhands" not in source
    assert "import litellm" not in source
    assert "from litellm" not in source
