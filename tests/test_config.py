"""Tests for devdox_ai_sonar/config.py."""

import os
import tempfile

import pytest
from pathlib import Path
from devdox_ai_sonar.config import Settings, settings


class TestSettingsDefaults:
    """Default values on a freshly-constructed Settings."""

    def test_version_default(self):
        s = Settings()
        assert s.VERSION == "0.0.7"
        assert isinstance(s.VERSION, str)

    def test_config_dir_default(self):
        s = Settings()
        expected = Path.home() / "devdox"
        assert s.CONFIG_DIR == expected
        assert isinstance(s.CONFIG_DIR, Path)

    def test_llm_env_defaults_blank(self):
        """The three LLM env fields default to empty strings so that a
        blank LLM_MODEL unambiguously means 'fall back to config.toml'."""
        s = Settings()
        assert s.LLM_MODEL == ""
        assert s.LLM_API_KEY == ""
        assert s.LLM_BASE_URL == ""

    def test_sonar_fields_default_empty(self):
        s = Settings()
        assert s.SONAR_TOKEN == ""
        assert s.SONAR_ORGANIZATION == ""
        assert s.SONAR_PROJECT_KEY == ""

    def test_max_fixes_defaults(self):
        s = Settings()
        assert s.MAX_FIXES_LIMIT == 20
        assert s.DEFAULT_MAX_FIXES == 5

    def test_project_path_default(self):
        s = Settings()
        assert s.PROJECT_PATH == Path(".")

    def test_exc_info_default_false(self):
        s = Settings()
        assert s.EXC_INFO is False


class TestSettingsComputedProperties:
    def test_config_file_path_property(self):
        s = Settings()
        assert s.config_file_path == s.CONFIG_DIR / "config.toml"

    def test_auth_file_path_property(self):
        s = Settings()
        assert s.auth_file_path == s.CONFIG_DIR / "auth.json"

    def test_computed_paths_follow_config_dir(self):
        custom_dir = Path(tempfile.gettempdir()) / "custom_config"
        s = Settings(CONFIG_DIR=custom_dir)
        assert s.config_file_path == custom_dir / "config.toml"
        assert s.auth_file_path == custom_dir / "auth.json"


class TestSettingsCustomization:
    def test_custom_config_dir(self):
        custom_path = Path(tempfile.gettempdir()) / "test_config"
        s = Settings(CONFIG_DIR=custom_path)
        assert s.CONFIG_DIR == custom_path

    def test_custom_max_fixes_limit(self):
        s = Settings(MAX_FIXES_LIMIT=50)
        assert s.MAX_FIXES_LIMIT == 50

    def test_custom_project_path_as_string(self):
        s = Settings(PROJECT_PATH=os.path.join(tempfile.gettempdir(), "my_project"))
        assert isinstance(s.PROJECT_PATH, Path)

    @pytest.mark.parametrize("limit,default", [(10, 3), (20, 5), (100, 10)])
    def test_various_fix_limits(self, limit, default):
        s = Settings(MAX_FIXES_LIMIT=limit, DEFAULT_MAX_FIXES=default)
        assert s.MAX_FIXES_LIMIT == limit
        assert s.DEFAULT_MAX_FIXES == default


class TestSettingsEnvironmentVariables:
    """Loading values from process environment."""

    def test_load_sonar_token_from_env(self, monkeypatch):
        monkeypatch.setenv("SONAR_TOKEN", "test_token_12345")
        assert Settings().SONAR_TOKEN == "test_token_12345"

    def test_load_sonar_organization_from_env(self, monkeypatch):
        monkeypatch.setenv("SONAR_ORGANIZATION", "test_org")
        assert Settings().SONAR_ORGANIZATION == "test_org"

    def test_load_sonar_project_from_env(self, monkeypatch):
        monkeypatch.setenv("SONAR_PROJECT_KEY", "test_project")
        assert Settings().SONAR_PROJECT_KEY == "test_project"

    def test_load_llm_model_from_env(self, monkeypatch):
        """Provider is encoded in the model prefix; no separate LLM_PROVIDER."""
        monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-flash")
        assert Settings().LLM_MODEL == "gemini/gemini-2.5-flash"

    def test_load_llm_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
        assert Settings().LLM_API_KEY == "sk-test-key"

    def test_load_llm_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://vllm.example.com")
        assert Settings().LLM_BASE_URL == "https://vllm.example.com"

    def test_load_numeric_values_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_FIXES_LIMIT", "30")
        monkeypatch.setenv("DEFAULT_MAX_FIXES", "10")
        s = Settings()
        assert s.MAX_FIXES_LIMIT == 30
        assert s.DEFAULT_MAX_FIXES == 10

    def test_load_boolean_from_env(self, monkeypatch):
        monkeypatch.setenv("EXC_INFO", "true")
        assert Settings().EXC_INFO is True


class TestRemovedFieldsAreSilentlyIgnored:
    """Older provider-specific env vars must not leak into Settings attributes.

    ``extra = "ignore"`` in Pydantic config means unknown vars from a
    user's old ``.env`` file do not crash the app -- they're just unused.
    These tests make that contract explicit.
    """

    @pytest.mark.parametrize(
        "old_field",
        ["OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER"],
    )
    def test_old_env_vars_do_not_produce_attributes(self, monkeypatch, old_field):
        monkeypatch.setenv(old_field, "leftover-value")
        s = Settings()
        assert not hasattr(s, old_field)


class TestSettingsValidation:
    def test_invalid_max_fixes_limit_type(self):
        with pytest.raises(Exception):
            Settings(MAX_FIXES_LIMIT="not_a_number")

    def test_invalid_boolean_type(self):
        with pytest.raises(Exception):
            Settings(EXC_INFO="not_a_boolean")

    def test_extra_fields_ignored(self):
        s = Settings(UNKNOWN_FIELD="value")
        assert not hasattr(s, "UNKNOWN_FIELD")


class TestSettingsConfigClass:
    def test_env_file_setting(self):
        assert Settings.Config.env_file == ".env"

    def test_case_sensitive_setting(self):
        assert Settings.Config.case_sensitive is True

    def test_extra_fields_setting(self):
        assert Settings.Config.extra == "ignore"


class TestGlobalSettingsInstance:
    def test_settings_instance_exists(self):
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_instance_has_defaults(self):
        assert settings.VERSION == "0.0.7"
        assert settings.MAX_FIXES_LIMIT == 20

    def test_settings_is_importable(self):
        from devdox_ai_sonar.config import settings as s
        assert hasattr(s, "config_file_path")
        assert hasattr(s, "auth_file_path")


class TestSettingsEdgeCases:
    def test_empty_string_llm_fields(self):
        s = Settings(LLM_MODEL="", LLM_API_KEY="", LLM_BASE_URL="")
        assert s.LLM_MODEL == ""
        assert s.LLM_API_KEY == ""
        assert s.LLM_BASE_URL == ""

    def test_zero_max_fixes(self):
        s = Settings(MAX_FIXES_LIMIT=0, DEFAULT_MAX_FIXES=0)
        assert s.MAX_FIXES_LIMIT == 0

    def test_path_with_spaces(self):
        path_with_spaces = Path(tempfile.gettempdir()) / "my project" / "config"
        s = Settings(CONFIG_DIR=path_with_spaces)
        assert s.CONFIG_DIR == path_with_spaces
