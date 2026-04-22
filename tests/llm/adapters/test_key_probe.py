"""Tests for the key_probe adapter.

The real probe hits a remote endpoint, so every test here replaces
``litellm.completion`` with a stub via ``monkeypatch.setattr``. The adapter's
job is to translate what ``litellm.completion`` does (return normally vs
raise a specific exception type) into a small, predictable
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

from devdox_ai_sonar.llm.adapters import key_probe
from devdox_ai_sonar.llm.adapters.key_probe import KeyProbeOutcome


def _install_completion(monkeypatch, behaviour):
    """Patch ``litellm.completion`` in the adapter's module namespace."""
    monkeypatch.setattr(key_probe.litellm, "completion", behaviour)


def test_successful_completion_returns_ok(monkeypatch):
    _install_completion(monkeypatch, lambda **_: None)
    result = key_probe.probe(model="openai/gpt-4o", api_key="sk-valid", api_base=None)
    assert result == KeyProbeOutcome(ok=True)


def test_authentication_error_categorised_as_auth(monkeypatch):
    def raise_auth(**_):
        raise AuthenticationError("bad key", llm_provider="openai", model="gpt-4o")
    _install_completion(monkeypatch, raise_auth)
    result = key_probe.probe(model="openai/gpt-4o", api_key="sk-bad", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "auth"
    assert "bad key" in result.detail


def test_failure_carries_provenance_fields(monkeypatch):
    """Every non-success outcome must include provider / status_code /
    exception_class so callers can report where the failure came from."""
    def raise_auth(**_):
        exc = AuthenticationError("bad key", llm_provider="gemini", model="gem")
        # litellm's real exceptions set status_code; the test stand-ins
        # don't always, so set it explicitly here to pin the contract.
        exc.status_code = 401
        raise exc
    _install_completion(monkeypatch, raise_auth)
    result = key_probe.probe(
        model="gemini/gemini-2.5-flash", api_key="sk-bad", api_base=None
    )
    assert result.provider == "gemini"
    assert result.status_code == 401
    assert result.exception_class == (
        "litellm.exceptions.AuthenticationError"
    )


def test_generic_exception_still_populates_exception_class(monkeypatch):
    """Plain ``Exception`` caught by the catch-all branch doesn't carry
    litellm's ``llm_provider`` / ``status_code`` attrs, but we must
    still record the exception class so the developer knows this came
    from somewhere unexpected."""
    def raise_generic(**_):
        raise RuntimeError("something else")
    _install_completion(monkeypatch, raise_generic)
    result = key_probe.probe(model="openai/gpt-4o", api_key="sk", api_base=None)
    assert result.failure_kind == "unknown"
    assert result.provider is None
    assert result.status_code is None
    assert result.exception_class == "builtins.RuntimeError"


def test_connection_error_categorised_as_connection(monkeypatch):
    def raise_conn(**_):
        raise APIConnectionError(
            "unreachable", llm_provider="openai", model="gpt-4o"
        )
    _install_completion(monkeypatch, raise_conn)
    result = key_probe.probe(
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
        lambda: ServiceUnavailableError(
            "down", llm_provider="openai", model="gpt-4o"
        ),
    ],
)
def test_timeout_and_service_unavailable_are_connection_failures(
    monkeypatch, exc_factory
):
    def raise_exc(**_):
        raise exc_factory()
    _install_completion(monkeypatch, raise_exc)
    result = key_probe.probe(model="openai/gpt-4o", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "connection"


def test_bad_request_error_categorised_as_bad_request(monkeypatch):
    def raise_bad(**_):
        raise BadRequestError(
            "unknown model", model="openai/made-up", llm_provider="openai"
        )
    _install_completion(monkeypatch, raise_bad)
    result = key_probe.probe(model="openai/made-up", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "bad_request"


def test_rate_limit_error_categorised_as_rate_limit(monkeypatch):
    def raise_rl(**_):
        raise RateLimitError("slow down", llm_provider="openai", model="gpt-4o")
    _install_completion(monkeypatch, raise_rl)
    result = key_probe.probe(model="openai/gpt-4o", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "rate_limit"


def test_generic_exception_categorised_as_unknown(monkeypatch):
    def raise_generic(**_):
        raise RuntimeError("something else")
    _install_completion(monkeypatch, raise_generic)
    result = key_probe.probe(model="openai/gpt-4o", api_key="sk-x", api_base=None)
    assert result.ok is False
    assert result.failure_kind == "unknown"
    assert "something else" in result.detail


def test_empty_api_key_is_forwarded_as_none(monkeypatch):
    """Endpoints that require no auth should be probeable by passing ``""``
    or ``None`` through. The adapter must not reject the call locally — it
    just asks the endpoint and reports what comes back."""
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)

    _install_completion(monkeypatch, capture)
    key_probe.probe(model="ollama/llama3", api_key="", api_base="http://localhost:11434")
    assert captured["api_key"] is None
    assert captured["api_base"] == "http://localhost:11434"
