"""Tests for the LLM adapters module.

The real probe call hits a remote endpoint, so every probe test below
replaces ``litellm.completion`` with a stub via ``monkeypatch.setattr``.
The adapter's job is to translate what ``litellm.completion`` does (return
normally vs raise a specific exception type) into a small, predictable
:class:`KeyProbeOutcome` value.
"""

import pytest
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from devdox_ai_sonar.llm import adapters
from devdox_ai_sonar.llm.adapters import KeyProbeOutcome


# ---------------------------------------------------------------------------
# list_verified_models
# ---------------------------------------------------------------------------


def test_verified_returns_dict():
    assert isinstance(adapters.list_verified_models(), dict)


def test_verified_has_expected_families():
    """OpenHands curates a small set of provider families.

    The list is hand-maintained by OpenHands and may grow, but these core
    eight should always be present. If this test fails after an SDK upgrade,
    review the new family list before updating the expected set.
    """
    result = adapters.list_verified_models()
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


def test_verified_each_family_maps_to_list_of_strings():
    for family, models in adapters.list_verified_models().items():
        assert isinstance(models, list), f"{family} is not a list"
        for model in models:
            assert isinstance(model, str), f"{family} contains non-string: {model!r}"


def test_verified_openhands_family_is_non_empty():
    """The ``openhands`` family is the OpenHands Cloud gateway and must
    always surface at least one model (otherwise the UI has nothing to show
    under that picker entry)."""
    assert len(adapters.list_verified_models()["openhands"]) > 0


# ---------------------------------------------------------------------------
# list_unverified_models
# ---------------------------------------------------------------------------


def test_unverified_returns_dict():
    assert isinstance(adapters.list_unverified_models(), dict)


def test_unverified_includes_well_known_names():
    """The litellm catalogue is large and fluid, but a handful of providers
    are too well-established to disappear — they also happen to be the ones
    users are most likely to care about. If any of these drop out of the
    catalogue after an SDK upgrade, something has changed upstream and the
    expectation below needs re-evaluating, not silent updating."""
    result = adapters.list_unverified_models()
    expected_some_of = {
        "openai",
        "gemini",
        "together_ai",
        "openrouter",
        "groq",
        "ollama",
    }
    assert expected_some_of.issubset(result.keys())


def test_unverified_each_family_maps_to_list_of_strings():
    for family, models in adapters.list_unverified_models().items():
        assert isinstance(models, list), f"{family} is not a list"
        for model in models:
            assert isinstance(model, str), f"{family} contains non-string: {model!r}"


def test_unverified_catalog_is_substantial():
    """The full litellm-minus-Bedrock catalogue covers ~60+ providers. A
    sudden collapse to a tiny number would signal an SDK regression worth
    investigating before tests are updated."""
    assert len(adapters.list_unverified_models()) >= 20


# ---------------------------------------------------------------------------
# infer_provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "api_base", "expected"),
    [
        ("openai/gpt-4o", None, "openai"),
        ("gemini/gemini-2.5-flash", None, "gemini"),
        ("anthropic/claude-sonnet-4-6", None, "anthropic"),
        ("together_ai/meta-llama/Llama-3-70b-chat", None, "together_ai"),
        ("openrouter/anthropic/claude-sonnet-4", None, "openrouter"),
        ("ollama/llama3", None, "ollama"),
        ("gpt-4o", None, "openai"),
        ("vllm/qwen-72b", "https://my-vllm.example.com", "vllm"),
        ("azure/gpt-4", "https://my-azure.openai.azure.com", "azure"),
    ],
)
def test_infer_provider_known_prefixes_resolve(model, api_base, expected):
    assert adapters.infer_provider(model=model, api_base=api_base) == expected


@pytest.mark.parametrize(
    "model",
    [
        "bogus/whatever",
        "my-company/my-model",
        "claude-3-5-sonnet-20241022",
        "not-a-real-model",
    ],
)
def test_infer_provider_unknown_or_unresolvable_returns_none(model):
    assert adapters.infer_provider(model=model, api_base=None) is None


def test_infer_provider_openhands_prefix_is_not_routable_by_litellm():
    """The ``openhands/*`` namespace is handled by the OpenHands Cloud
    gateway rather than a litellm adapter. Validation callers must treat
    this ``None`` specially by falling back to the verified-models map.
    """
    assert adapters.infer_provider(model="openhands/gpt-5.4", api_base=None) is None


# ---------------------------------------------------------------------------
# probe (credential check)
# ---------------------------------------------------------------------------


def _install_completion(monkeypatch, behaviour):
    """Patch ``litellm.completion`` in the adapter's module namespace."""
    monkeypatch.setattr(adapters.litellm, "completion", behaviour)


def test_probe_success_returns_ok(monkeypatch):
    _install_completion(monkeypatch, lambda **_: None)
    result = adapters.probe(model="openai/gpt-4o", api_key="sk-valid", api_base=None)
    assert result == KeyProbeOutcome(ok=True)


def test_probe_authentication_error(monkeypatch):
    def raise_auth(**_):
        raise AuthenticationError("bad key", llm_provider="openai", model="gpt-4o")
    _install_completion(monkeypatch, raise_auth)
    result = adapters.probe(model="openai/gpt-4o", api_key="sk-bad", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "auth"
    assert "bad key" in result.detail


def test_probe_connection_error(monkeypatch):
    def raise_conn(**_):
        raise APIConnectionError("unreachable", llm_provider="openai", model="gpt-4o")
    _install_completion(monkeypatch, raise_conn)
    result = adapters.probe(
        model="openai/gpt-4o",
        api_key="sk-x",
        api_base="https://unreachable.example.com",
    )
    assert result.ok is False
    assert result.failure_kind == "connection"


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: Timeout("too slow", model="gpt-4o", llm_provider="openai"),
        lambda: ServiceUnavailableError("down", llm_provider="openai", model="gpt-4o"),
    ],
)
def test_probe_timeout_and_service_unavailable(monkeypatch, exc_factory):
    def raise_exc(**_):
        raise exc_factory()
    _install_completion(monkeypatch, raise_exc)
    result = adapters.probe(model="openai/gpt-4o", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "connection"


def test_probe_bad_request_error(monkeypatch):
    def raise_bad(**_):
        raise BadRequestError(
            "unknown model", model="openai/made-up", llm_provider="openai"
        )
    _install_completion(monkeypatch, raise_bad)
    result = adapters.probe(model="openai/made-up", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "bad_request"


def test_probe_rate_limit_error(monkeypatch):
    def raise_rl(**_):
        raise RateLimitError("slow down", llm_provider="openai", model="gpt-4o")
    _install_completion(monkeypatch, raise_rl)
    result = adapters.probe(model="openai/gpt-4o", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "rate_limit"


def test_probe_generic_exception_is_unknown(monkeypatch):
    def raise_generic(**_):
        raise RuntimeError("something else")
    _install_completion(monkeypatch, raise_generic)
    result = adapters.probe(model="openai/gpt-4o", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "unknown"
    assert "something else" in result.detail


def test_probe_empty_api_key_is_forwarded_as_none(monkeypatch):
    """Endpoints that require no auth should be probeable by passing ``""``
    or ``None`` through. The adapter must not reject the call locally — it
    just asks the endpoint and reports what comes back."""
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)

    _install_completion(monkeypatch, capture)
    adapters.probe(
        model="ollama/llama3", api_key="", api_base="http://localhost:11434"
    )
    assert captured["api_key"] is None
    assert captured["api_base"] == "http://localhost:11434"
