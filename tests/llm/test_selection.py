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
        ModelInfo(model_id="gpt-4o"),
        ModelInfo(model_id="gpt-5"),
        ModelInfo(model_id="gpt-3.5"),
        ModelInfo(model_id="sora-2"),
    ]


def test_model_menu_verified_only_when_no_advanced_entries(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"gemini": ["gemini-3.1-pro-preview"]},
        unverified={},
    )
    result = build_model_menu("gemini")
    assert result == [ModelInfo(model_id="gemini-3.1-pro-preview")]


def test_model_menu_advanced_only_when_not_in_verified(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={},
        unverified={"together_ai": ["mixtral", "llama"]},
    )
    result = build_model_menu("together_ai")
    assert [m.model_id for m in result] == ["mixtral", "llama"]


def test_model_menu_unknown_provider_returns_empty(monkeypatch):
    _fake_catalogs(
        monkeypatch,
        verified={"openai": ["gpt-4o"]},
        unverified={"groq": ["llama-3"]},
    )
    assert build_model_menu("never-heard-of-it") == []


def test_selection_does_not_touch_sdk_directly():
    """Selection imports only from the adapters module. If someone adds a
    direct openhands / litellm import, this breaks."""
    import inspect

    source = inspect.getsource(selection)
    assert "from openhands" not in source
    assert "import openhands" not in source
    assert "import litellm" not in source
    assert "from litellm" not in source
