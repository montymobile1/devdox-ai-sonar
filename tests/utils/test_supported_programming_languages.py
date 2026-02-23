import pytest

from devdox_ai_sonar.utils.supported_programming_languages import (
    PYTHON,
    LanguageConfig,
    is_file_processable,
    lang_from_file_ext,
    lang_from_rule_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def python_config():
    return PYTHON


@pytest.fixture
def java_config():
    """Fake Java config for multi-language tests."""
    return LanguageConfig(
        name="java",
        sonar_language_key="java",
        sonar_repositories=frozenset({"java", "javasecurity"}),
        file_extensions=frozenset({".java"}),
    )


# ---------------------------------------------------------------------------
# LanguageConfig tests
# ---------------------------------------------------------------------------


class TestLanguageConfig:
    def test_python_has_sonar_language_key(self, python_config):
        assert python_config.sonar_language_key == "py"

    def test_python_includes_pythonsecurity_repo(self, python_config):
        assert "pythonsecurity" in python_config.sonar_repositories

    def test_python_includes_python_repo(self, python_config):
        assert "python" in python_config.sonar_repositories

    def test_python_file_extensions(self, python_config):
        assert ".py" in python_config.file_extensions
        assert ".pyw" in python_config.file_extensions

    def test_frozen_dataclass(self, python_config):
        with pytest.raises(AttributeError):
            python_config.name = "something_else"


# ---------------------------------------------------------------------------
# lang_from_rule_key / lang_from_file_ext tests
# ---------------------------------------------------------------------------


class TestLangLookupFunctions:
    def test_resolves_python_rule_key(self):
        lang = lang_from_rule_key("python:S1234")
        assert lang is not None
        assert lang.name == "python"

    def test_resolves_pythonsecurity_rule_key(self):
        lang = lang_from_rule_key("pythonsecurity:S5445")
        assert lang is not None
        assert lang.name == "python"

    def test_resolves_bare_repo_name(self):
        lang = lang_from_rule_key("pythonsecurity")
        assert lang is not None
        assert lang.name == "python"

    def test_returns_none_for_unknown_rule_key(self):
        assert lang_from_rule_key("java:S1234") is None

    def test_returns_none_for_empty_rule_key(self):
        assert lang_from_rule_key("") is None

    def test_resolves_file_extension_py(self):
        lang = lang_from_file_ext("src/utils/helpers.py")
        assert lang is not None
        assert lang.name == "python"

    def test_resolves_file_extension_pyw(self):
        lang = lang_from_file_ext("script.pyw")
        assert lang is not None
        assert lang.name == "python"

    def test_returns_none_for_unknown_extension(self):
        assert lang_from_file_ext("Foo.java") is None


# ---------------------------------------------------------------------------
# is_file_processable tests
# ---------------------------------------------------------------------------


class TestIsFileProcessable:
    def test_no_args_allows_all(self):
        assert is_file_processable("anything.rs") is True
        assert is_file_processable("test_foo.java") is True

    def test_single_language_suffixes(self, python_config):
        suffixes = python_config.file_extensions
        assert is_file_processable("foo.py", allowed_suffixes=suffixes) is True
        assert is_file_processable("bar.pyw", allowed_suffixes=suffixes) is True
        assert is_file_processable("baz.js", allowed_suffixes=suffixes) is False

    def test_multiple_language_suffixes(self, python_config, java_config):
        suffixes = python_config.file_extensions | java_config.file_extensions
        assert is_file_processable("foo.py", allowed_suffixes=suffixes) is True
        assert is_file_processable("Bar.java", allowed_suffixes=suffixes) is True
        assert is_file_processable("baz.js", allowed_suffixes=suffixes) is False

    def test_explicit_suffixes_override(self, python_config):
        """Explicit allowed_suffixes can be narrower than the language."""
        assert is_file_processable("foo.py", allowed_suffixes={".py"}) is True
        assert is_file_processable("bar.pyw", allowed_suffixes={".py"}) is False

    def test_excluded_prefixes(self, python_config):
        assert (
            is_file_processable(
                "test_foo.py",
                allowed_suffixes=python_config.file_extensions,
                excluded_prefixes={"test_"},
            )
            is False
        )
        assert (
            is_file_processable(
                "foo.py",
                allowed_suffixes=python_config.file_extensions,
                excluded_prefixes={"test_"},
            )
            is True
        )

    def test_none_suffixes_with_explicit_set(self):
        """No language, but explicit suffixes given."""
        assert is_file_processable("readme.txt", allowed_suffixes={".txt"}) is True
        assert is_file_processable("readme.md", allowed_suffixes={".txt"}) is False

    def test_empty_suffixes_allows_all(self):
        """Empty set behaves as allow-all."""
        assert is_file_processable("anything.xyz") is True
