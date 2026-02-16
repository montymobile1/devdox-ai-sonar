"""
Comprehensive tests for devdox_ai_sonar/config.py
Target Coverage: >95%
"""

import pytest
from pathlib import Path
from devdox_ai_sonar.config import Settings, settings


class TestSettingsDefaults:
    """Test default configuration values"""

    def test_version_default(self):
        """VERSION should be '0.0.1-beta'"""
        s = Settings()
        assert s.VERSION == "0.0.1-beta"
        assert isinstance(s.VERSION, str)

    def test_config_dir_default(self):
        """CONFIG_DIR should default to ~/devdox"""
        s = Settings()
        expected = Path.home() / "devdox"
        assert s.CONFIG_DIR == expected
        assert isinstance(s.CONFIG_DIR, Path)

    def test_api_keys_default_empty(self):
        """API keys should default to empty strings"""
        s = Settings()
        assert s.API_KEY == ""
        assert s.OPENAI_API_KEY == ""
        assert s.GEMINI_API_KEY == ""

    def test_llm_provider_defaults(self):
        """LLM provider configuration defaults"""
        s = Settings()
        assert s.LLM_PROVIDER == "LLM_PROVIDER"
        assert s.LLM_MODEL == "gemini-2.5-flash"

    def test_sonar_fields_default_empty(self):
        """SonarCloud fields should default to empty"""
        s = Settings()
        assert s.SONAR_TOKEN == ""
        assert s.SONAR_ORGANIZATION == ""
        assert s.SONAR_PROJECT_KEY == ""

    def test_max_fixes_defaults(self):
        """Max fixes limits should have correct defaults"""
        s = Settings()
        assert s.MAX_FIXES_LIMIT == 20
        assert s.DEFAULT_MAX_FIXES == 5
        assert isinstance(s.MAX_FIXES_LIMIT, int)
        assert isinstance(s.DEFAULT_MAX_FIXES, int)

    def test_project_path_default(self):
        """PROJECT_PATH should have default"""
        s = Settings()
        assert s.PROJECT_PATH == Path("/your/project/path")
        assert isinstance(s.PROJECT_PATH, Path)

    def test_exc_info_default_false(self):
        """EXC_INFO should default to False"""
        s = Settings()
        assert s.EXC_INFO is False
        assert isinstance(s.EXC_INFO, bool)


class TestSettingsComputedProperties:
    """Test computed property paths"""

    def test_config_file_path_property(self):
        """config_file_path should return CONFIG_DIR/config.toml"""
        s = Settings()
        expected = s.CONFIG_DIR / "config.toml"
        assert s.config_file_path == expected
        assert isinstance(s.config_file_path, Path)

    def test_auth_file_path_property(self):
        """auth_file_path should return CONFIG_DIR/auth.json"""
        s = Settings()
        expected = s.CONFIG_DIR / "auth.json"
        assert s.auth_file_path == expected
        assert isinstance(s.auth_file_path, Path)

    def test_computed_paths_follow_config_dir(self):
        """Computed paths should change with CONFIG_DIR"""
        custom_dir = Path("/tmp/custom_config")
        s = Settings(CONFIG_DIR=custom_dir)

        assert s.config_file_path == custom_dir / "config.toml"
        assert s.auth_file_path == custom_dir / "auth.json"


class TestSettingsCustomization:
    """Test custom configuration values"""

    def test_custom_config_dir(self):
        """Should accept custom CONFIG_DIR"""
        custom_path = Path("/tmp/test_config")
        s = Settings(CONFIG_DIR=custom_path)
        assert s.CONFIG_DIR == custom_path

    def test_custom_max_fixes_limit(self):
        """Should accept custom MAX_FIXES_LIMIT"""
        s = Settings(MAX_FIXES_LIMIT=50)
        assert s.MAX_FIXES_LIMIT == 50

    def test_custom_project_path_as_string(self):
        """Should convert string to Path for PROJECT_PATH"""
        s = Settings(PROJECT_PATH="/tmp/my_project")
        assert isinstance(s.PROJECT_PATH, Path)
        assert s.PROJECT_PATH == Path("/tmp/my_project")

    def test_custom_project_path_as_path(self):
        """Should accept Path object for PROJECT_PATH"""
        path_obj = Path("/tmp/my_project")
        s = Settings(PROJECT_PATH=path_obj)
        assert s.PROJECT_PATH == path_obj

    @pytest.mark.parametrize("limit,default", [
        (10, 3),
        (20, 5),
        (100, 10),
    ])
    def test_various_fix_limits(self, limit, default):
        """Should accept various fix limit combinations"""
        s = Settings(MAX_FIXES_LIMIT=limit, DEFAULT_MAX_FIXES=default)
        assert s.MAX_FIXES_LIMIT == limit
        assert s.DEFAULT_MAX_FIXES == default


class TestSettingsEnvironmentVariables:
    """Test loading from environment variables"""

    def test_load_sonar_token_from_env(self, monkeypatch):
        """Should load SONAR_TOKEN from environment"""
        monkeypatch.setenv("SONAR_TOKEN", "test_token_12345")
        s = Settings()
        assert s.SONAR_TOKEN == "test_token_12345"

    def test_load_sonar_organization_from_env(self, monkeypatch):
        """Should load SONAR_ORGANIZATION from environment"""
        monkeypatch.setenv("SONAR_ORGANIZATION", "test_org")
        s = Settings()
        assert s.SONAR_ORGANIZATION == "test_org"

    def test_load_sonar_project_from_env(self, monkeypatch):
        """Should load SONAR_PROJECT_KEY from environment"""
        monkeypatch.setenv("SONAR_PROJECT_KEY", "test_project")
        s = Settings()
        assert s.SONAR_PROJECT_KEY == "test_project"

    def test_load_api_keys_from_env(self, monkeypatch):
        """Should load API keys from environment"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

        s = Settings()
        assert s.OPENAI_API_KEY == "sk-test-openai"
        assert s.GEMINI_API_KEY == "test-gemini-key"

    def test_load_numeric_values_from_env(self, monkeypatch):
        """Should parse numeric values from environment"""
        monkeypatch.setenv("MAX_FIXES_LIMIT", "30")
        monkeypatch.setenv("DEFAULT_MAX_FIXES", "10")

        s = Settings()
        assert s.MAX_FIXES_LIMIT == 30
        assert s.DEFAULT_MAX_FIXES == 10

    def test_load_boolean_from_env(self, monkeypatch):
        """Should parse boolean values from environment"""
        monkeypatch.setenv("EXC_INFO", "true")
        s = Settings()
        assert s.EXC_INFO is True

    def test_env_overrides_defaults(self, monkeypatch):
        """Environment variables should override defaults"""
        monkeypatch.setenv("LLM_MODEL", "gpt-4-turbo")
        s = Settings()
        assert s.LLM_MODEL == "gpt-4-turbo"
        assert s.LLM_MODEL != "gemini-2.5-flash"


class TestSettingsValidation:
    """Test Pydantic validation"""

    def test_invalid_max_fixes_limit_type(self):
        """Should reject non-integer MAX_FIXES_LIMIT"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            Settings(MAX_FIXES_LIMIT="not_a_number")

    def test_invalid_boolean_type(self):
        """Should reject non-boolean EXC_INFO"""
        with pytest.raises(Exception):
            Settings(EXC_INFO="not_a_boolean")

    def test_extra_fields_ignored(self):
        """Should ignore extra fields (Config.extra = 'ignore')"""
        s = Settings(UNKNOWN_FIELD="value")
        assert not hasattr(s, 'UNKNOWN_FIELD')




class TestSettingsConfigClass:
    """Test Config class attributes"""

    def test_env_file_setting(self):
        """Config should specify .env file"""
        assert Settings.Config.env_file == ".env"

    def test_case_sensitive_setting(self):
        """Config should be case-sensitive"""
        assert Settings.Config.case_sensitive is True

    def test_extra_fields_setting(self):
        """Config should ignore extra fields"""
        assert Settings.Config.extra == "ignore"


class TestGlobalSettingsInstance:
    """Test the global settings singleton"""

    def test_settings_instance_exists(self):
        """Global settings instance should exist"""
        from devdox_ai_sonar.config import settings
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_instance_has_defaults(self):
        """Global settings should have default values"""
        from devdox_ai_sonar.config import settings
        assert settings.VERSION == "0.0.1-beta"
        assert settings.MAX_FIXES_LIMIT == 20

    def test_settings_is_importable(self):
        """Should be able to import settings"""
        from devdox_ai_sonar.config import settings
        assert hasattr(settings, 'config_file_path')
        assert hasattr(settings, 'auth_file_path')


class TestSettingsEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_string_values(self):
        """Should accept empty strings for string fields"""
        s = Settings(
            API_KEY="",
            SONAR_TOKEN="",
            LLM_PROVIDER=""
        )
        assert s.API_KEY == ""
        assert s.SONAR_TOKEN == ""
        assert s.LLM_PROVIDER == ""

    def test_zero_max_fixes(self):
        """Should accept zero for max fixes"""
        s = Settings(MAX_FIXES_LIMIT=0, DEFAULT_MAX_FIXES=0)
        assert s.MAX_FIXES_LIMIT == 0
        assert s.DEFAULT_MAX_FIXES == 0

    def test_very_large_max_fixes(self):
        """Should accept large values for max fixes"""
        s = Settings(MAX_FIXES_LIMIT=10000)
        assert s.MAX_FIXES_LIMIT == 10000

    def test_path_with_spaces(self):
        """Should handle paths with spaces"""
        path_with_spaces = Path("/tmp/my project/config")
        s = Settings(CONFIG_DIR=path_with_spaces)
        assert s.CONFIG_DIR == path_with_spaces
        assert " " in str(s.CONFIG_DIR)

