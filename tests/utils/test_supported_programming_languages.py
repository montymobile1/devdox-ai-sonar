import pytest

from devdox_ai_sonar.utils.supported_programming_languages import (
    FileFilter,
    LanguageConfig,
    LanguageRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return LanguageRegistry()


@pytest.fixture
def python_config(registry):
    return registry.get(LanguageRegistry.PYTHON)


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
# LanguageRegistry tests
# ---------------------------------------------------------------------------

class TestLanguageRegistry:

    def test_resolves_python_rule_key(self, registry):
        lang = registry.from_sonar_rule_key("python:S1234")
        assert lang is not None
        assert lang.name == "python"

    def test_resolves_pythonsecurity_rule_key(self, registry):
        lang = registry.from_sonar_rule_key("pythonsecurity:S5445")
        assert lang is not None
        assert lang.name == "python"

    def test_resolves_bare_repo_name(self, registry):
        lang = registry.from_sonar_rule_key("pythonsecurity")
        assert lang is not None
        assert lang.name == "python"

    def test_returns_none_for_unknown_rule_key(self, registry):
        assert registry.from_sonar_rule_key("java:S1234") is None

    def test_returns_none_for_empty_rule_key(self, registry):
        assert registry.from_sonar_rule_key("") is None

    def test_resolves_file_extension_py(self, registry):
        lang = registry.from_file_extension("src/utils/helpers.py")
        assert lang is not None
        assert lang.name == "python"

    def test_resolves_file_extension_pyw(self, registry):
        lang = registry.from_file_extension("script.pyw")
        assert lang is not None
        assert lang.name == "python"

    def test_returns_none_for_unknown_extension(self, registry):
        assert registry.from_file_extension("Foo.java") is None

    def test_get_by_name(self, registry):
        lang = registry.get("python")
        assert lang is not None
        assert lang.sonar_language_key == "py"

    def test_get_returns_none_for_unknown(self, registry):
        assert registry.get("ruby") is None

    def test_duplicate_repo_raises(self):
        with pytest.raises(ValueError, match="Duplicate sonar repository"):
            LanguageRegistry(
                languages={
                    "a": LanguageConfig(
                        name="a",
                        sonar_language_key="a",
                        sonar_repositories=frozenset({"shared"}),
                        file_extensions=frozenset({".a"}),
                    ),
                    "b": LanguageConfig(
                        name="b",
                        sonar_language_key="b",
                        sonar_repositories=frozenset({"shared"}),
                        file_extensions=frozenset({".b"}),
                    ),
                }
            )

    def test_key_name_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            LanguageRegistry(
                languages={
                    "wrong_key": LanguageConfig(
                        name="correct_name",
                        sonar_language_key="x",
                        sonar_repositories=frozenset({"x"}),
                        file_extensions=frozenset({".x"}),
                    ),
                }
            )


# ---------------------------------------------------------------------------
# FileFilter.for_languages tests
# ---------------------------------------------------------------------------

class TestFileFilterForLanguages:

    def test_no_args_allows_all(self):
        f = FileFilter.for_languages()
        assert f.is_processable("anything.rs") is True
        assert f.is_processable("test_foo.java") is True

    def test_single_language_seeds_suffixes(self, python_config):
        f = FileFilter.for_languages([python_config])
        assert f.is_processable("foo.py") is True
        assert f.is_processable("bar.pyw") is True
        assert f.is_processable("baz.js") is False

    def test_multiple_languages_merge_extensions(self, python_config, java_config):
        f = FileFilter.for_languages([python_config, java_config])
        assert f.is_processable("foo.py") is True
        assert f.is_processable("Bar.java") is True
        assert f.is_processable("baz.js") is False

    def test_allowed_suffixes_overrides_language(self, python_config):
        """Explicit allowed_suffixes takes precedence over language extensions."""
        f = FileFilter.for_languages(
            [python_config],
            allowed_suffixes={".py"},
        )
        assert f.is_processable("foo.py") is True
        assert f.is_processable("bar.pyw") is False

    def test_excluded_prefixes(self, python_config):
        f = FileFilter.for_languages(
            [python_config],
            excluded_prefixes={"test_"},
        )
        assert f.is_processable("test_foo.py") is False
        assert f.is_processable("foo.py") is True

    def test_none_languages_with_explicit_suffixes(self):
        """No language provided but explicit suffixes given."""
        f = FileFilter.for_languages(
            None,
            allowed_suffixes={".txt"},
        )
        assert f.is_processable("readme.txt") is True
        assert f.is_processable("readme.md") is False

    def test_empty_list_is_allow_all(self):
        """Empty list (not None) behaves as allow-all."""
        f = FileFilter.for_languages([])
        assert f.is_processable("anything.xyz") is True
