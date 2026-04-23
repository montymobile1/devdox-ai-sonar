"""Tests for llm/validation.py."""

import pytest

from devdox_ai_sonar.llm import adapters, validation
from devdox_ai_sonar.llm.adapters import KeyProbeOutcome
from devdox_ai_sonar.llm.errors import (
    InvalidModelFormatError,
    KeyProbeError,
    UnknownProviderError,
)
from devdox_ai_sonar.llm.validation import (
    ValidatedModel,
    probe_api_key,
    validate_model_string,
)


# ---------------------------------------------------------------------------
# validate_model_string -- format checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_input", ["", "   ", "\n"])
def test_empty_or_whitespace_model_is_invalid_format(bad_input, monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: "openai")
    result = validate_model_string(bad_input, base_url=None)
    assert result.is_err()
    assert isinstance(result.error, InvalidModelFormatError)
    assert result.error.reason == "empty"


# ---------------------------------------------------------------------------
# validate_model_string -- strict mode (default)
# ---------------------------------------------------------------------------


def test_known_prefix_accepted_in_strict_mode(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: "openai")
    result = validate_model_string("openai/gpt-4o", base_url=None)
    assert result.is_ok()
    assert result.unwrap() == ValidatedModel(
        model="openai/gpt-4o", provider="openai", base_url=None
    )


def test_unknown_prefix_rejected_in_strict_mode(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {})
    result = validate_model_string("bogus/whatever", base_url=None)
    assert result.is_err()
    assert isinstance(result.error, UnknownProviderError)
    assert result.error.model == "bogus/whatever"


def test_unknown_prefix_rejected_even_with_base_url_in_strict_mode(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {})
    result = validate_model_string(
        "bogus/whatever", base_url="https://endpoint.example.com"
    )
    assert result.is_err()
    assert isinstance(result.error, UnknownProviderError)


# ---------------------------------------------------------------------------
# validate_model_string -- lenient mode
# ---------------------------------------------------------------------------


def test_unknown_prefix_with_base_url_accepted_in_lenient_mode(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {})
    result = validate_model_string(
        "mycorp/custom-model",
        base_url="https://endpoint.example.com",
        strict=False,
    )
    assert result.is_ok()
    validated = result.unwrap()
    assert validated.provider is None
    assert validated.base_url == "https://endpoint.example.com"


def test_unknown_prefix_without_base_url_rejected_in_lenient_mode(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {})
    result = validate_model_string("bogus/whatever", base_url=None, strict=False)
    assert result.is_err()
    assert isinstance(result.error, UnknownProviderError)


def test_known_prefix_still_accepted_in_lenient_mode(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: "openai")
    result = validate_model_string("openai/gpt-4o", base_url=None, strict=False)
    assert result.is_ok()


# ---------------------------------------------------------------------------
# validate_model_string -- openhands/* special case
# ---------------------------------------------------------------------------


def test_openhands_model_in_verified_catalog_accepted(monkeypatch):
    """The openhands/ prefix is unknown to litellm; validation falls back
    to the verified catalog."""
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {"openhands": ["gpt-5.4"]})
    result = validate_model_string("openhands/gpt-5.4", base_url=None)
    assert result.is_ok()
    assert result.unwrap().provider == "openhands"


def test_openhands_model_not_in_verified_catalog_rejected(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {"openhands": ["only-this-one"]})
    result = validate_model_string("openhands/some-other-model", base_url=None)
    assert result.is_err()
    assert isinstance(result.error, UnknownProviderError)


def test_openhands_prefix_without_model_id_rejected(monkeypatch):
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {"openhands": ["gpt-5.4"]})
    result = validate_model_string("openhands/", base_url=None)
    assert result.is_err()


def test_openhands_prefix_takes_precedence_over_strict_rejection(monkeypatch):
    """Even in strict mode without a base_url, openhands/* should pass
    if it's in the verified catalog. This is the whole reason the special
    case exists."""
    _install_inference(monkeypatch, lambda model, api_base: None)
    _install_verified(monkeypatch, {"openhands": ["gpt-5.4"]})
    result = validate_model_string("openhands/gpt-5.4", base_url=None, strict=True)
    assert result.is_ok()


# ---------------------------------------------------------------------------
# probe_api_key
# ---------------------------------------------------------------------------


def test_probe_returns_ok_when_validate_key_disabled(monkeypatch):
    """validate_key=False short-circuits without calling the adapter."""
    called = {"count": 0}

    def should_not_be_called(**_):
        called["count"] += 1
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(adapters, "probe", should_not_be_called)
    result = probe_api_key(
        model="openai/gpt-4o",
        api_key="sk-x",
        base_url=None,
        validate_key=False,
    )
    assert result.is_ok()
    assert called["count"] == 0


def test_probe_returns_ok_on_successful_outcome(monkeypatch):
    monkeypatch.setattr(
        adapters, "probe", lambda **_: KeyProbeOutcome(ok=True)
    )
    result = probe_api_key(model="openai/gpt-4o", api_key="sk-x", base_url=None)
    assert result.is_ok()


def test_probe_returns_err_on_auth_failure(monkeypatch):
    monkeypatch.setattr(
        adapters,
        "probe",
        lambda **_: KeyProbeOutcome(ok=False, failure_kind="auth", detail="nope"),
    )
    result = probe_api_key(model="openai/gpt-4o", api_key="sk-bad", base_url=None)
    assert result.is_err()
    assert isinstance(result.error, KeyProbeError)
    assert result.error.kind == "auth"
    assert result.error.detail == "nope"


@pytest.mark.parametrize(
    "outcome_kind",
    ["connection", "bad_request", "rate_limit", "unknown"],
)
def test_probe_propagates_every_failure_kind(monkeypatch, outcome_kind):
    monkeypatch.setattr(
        adapters,
        "probe",
        lambda **_: KeyProbeOutcome(ok=False, failure_kind=outcome_kind, detail="x"),
    )
    result = probe_api_key(model="openai/gpt-4o", api_key="sk-x", base_url=None)
    assert result.is_err()
    assert result.error.kind == outcome_kind


# ---------------------------------------------------------------------------
# Isolation smoke test
# ---------------------------------------------------------------------------


def test_validation_does_not_touch_sdk_directly():
    """Validation depends on adapters only; no direct openhands/litellm
    imports are allowed."""
    import inspect

    source = inspect.getsource(validation)
    assert "from openhands" not in source
    assert "import openhands" not in source
    assert "import litellm" not in source
    assert "from litellm" not in source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_inference(monkeypatch, fn):
    monkeypatch.setattr(adapters, "infer_provider", fn)


def _install_verified(monkeypatch, data):
    monkeypatch.setattr(adapters, "list_verified_models", lambda: data)
