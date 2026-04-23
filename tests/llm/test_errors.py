"""Tests for llm/errors.py."""

import pytest

from devdox_ai_sonar.llm.errors import (
    ConfigStorageError,
    Err,
    InvalidModelFormatError,
    KeyProbeError,
    Ok,
    OpenHandsSdkUnavailableError,
    ProfileNotFoundError,
    Result,
    UnknownProviderError,
)
from devdox_ai_sonar.utils.exceptions import DevDoxSonarError


@pytest.mark.parametrize(
    "exc_cls",
    [
        OpenHandsSdkUnavailableError,
        ConfigStorageError,
        ProfileNotFoundError,
    ],
)
def test_exceptions_are_raiseable_and_carry_message(exc_cls):
    with pytest.raises(exc_cls) as excinfo:
        raise exc_cls("oops")
    assert "oops" in str(excinfo.value)


def test_exceptions_inherit_from_devdox_sonar_error():
    """Project-wide error handlers catch DevDoxSonarError; every new
    exception in this package must fit that tree or it will be silently
    uncaught."""
    for exc_cls in (
        OpenHandsSdkUnavailableError,
        ConfigStorageError,
        ProfileNotFoundError,
    ):
        assert issubclass(exc_cls, DevDoxSonarError)


def test_invalid_model_format_error_fields():
    err = InvalidModelFormatError(model="", reason="empty")
    assert err.model == ""
    assert err.reason == "empty"


def test_unknown_provider_error_fields():
    err = UnknownProviderError(model="bogus/whatever")
    assert err.model == "bogus/whatever"


def test_branchable_errors_are_frozen():
    """Frozen dataclasses prevent callers from mutating an error after it's
    been constructed -- makes downstream reasoning about failures simpler."""
    err = UnknownProviderError(model="x")
    with pytest.raises(Exception):
        err.model = "y"  # type: ignore[misc]


def test_key_probe_error_kind_vocabulary():
    """Each documented KeyProbeError.kind must construct cleanly."""
    for kind in ("auth", "connection", "bad_request", "rate_limit", "unknown"):
        err = KeyProbeError(kind=kind, detail="example")  # type: ignore[arg-type]
        assert err.kind == kind
        assert err.detail == "example"


def test_result_ok_pattern():
    """Ok/Err come from utils.result; re-exporting means llm modules don't
    have to cross the package boundary."""
    r: Result[int, InvalidModelFormatError] = Ok(42)
    assert r.is_ok()
    assert r.unwrap() == 42


def test_result_err_pattern():
    err = InvalidModelFormatError(model="", reason="empty")
    r: Result[int, InvalidModelFormatError] = Err(err)
    assert r.is_err()
    assert r.error == err
