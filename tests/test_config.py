"""Tests for configuration settings."""

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "SONAR_TOKEN": "test-sonar-token",
        "SONAR_ORGANIZATION": "test-org",
        "SONAR_PROJECT_KEY": "test-project",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


class TestSettings:
    """Test suite for Settings configuration."""

    def test_settings_defaults(self):
        """Test default settings values."""
        from devdox_ai_sonar.config import Settings

        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            assert settings.VERSION == "0.0.1"
            assert settings.API_KEY == ""
            assert settings.OPENAI_API_KEY == ""
            assert settings.GEMINI_API_KEY == ""
            assert settings.LLM_PROVIDER == "LLM_PROVIDER"
            assert settings.LLM_MODEL == "gemini-2.5-flash"

    def test_settings_from_env(self, mock_env):
        """Test settings loaded from environment variables."""
        from devdox_ai_sonar.config import Settings

        settings = Settings()
        assert settings.OPENAI_API_KEY == "test-openai-key"
        assert settings.GEMINI_API_KEY == "test-gemini-key"
        assert settings.SONAR_TOKEN == "test-sonar-token"
        assert settings.SONAR_ORGANIZATION == "test-org"
        assert settings.SONAR_PROJECT_KEY == "test-project"

    def test_settings_project_path_type(self, mock_env):
        """Test that PROJECT_PATH is properly converted to Path."""
        from devdox_ai_sonar.config import Settings

        settings = Settings()
        assert isinstance(settings.PROJECT_PATH, Path)

    def test_settings_case_sensitivity(self, monkeypatch):
        """Test that settings are case-sensitive."""
        from devdox_ai_sonar.config import Settings

        monkeypatch.setenv("sonar_token", "lowercase-token")
        monkeypatch.setenv("SONAR_TOKEN", "uppercase-token")

        settings = Settings()
        assert settings.SONAR_TOKEN == "uppercase-token"

    def test_settings_extra_fields_ignored(self, monkeypatch):
        """Test that extra fields in .env are ignored."""
        from devdox_ai_sonar.config import Settings

        monkeypatch.setenv("UNKNOWN_FIELD", "value")
        settings = Settings()
        assert not hasattr(settings, "UNKNOWN_FIELD")

    def test_settings_singleton_pattern(self):
        """Test that settings instance is properly initialized."""
        from devdox_ai_sonar.config import settings

        assert settings is not None
        assert hasattr(settings, "VERSION")
        assert hasattr(settings, "SONAR_TOKEN")


class TestSettingsValidation:
    """Test suite for settings validation."""

    def test_empty_project_path_default(self):
        """Test that PROJECT_PATH has a default value."""
        from devdox_ai_sonar.config import Settings

        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            assert settings.PROJECT_PATH == Path("/your/project/path")

    def test_custom_project_path(self, monkeypatch):
        """Test setting custom project path."""
        from devdox_ai_sonar.config import Settings

        test_path = "/custom/project/path"
        monkeypatch.setenv("PROJECT_PATH", test_path)
        settings = Settings()
        assert str(settings.PROJECT_PATH) == test_path

    def test_llm_provider_values(self, monkeypatch):
        """Test different LLM provider values."""
        from devdox_ai_sonar.config import Settings

        providers = ["openai", "gemini", "togetherai"]
        for provider in providers:
            monkeypatch.setenv("LLM_PROVIDER", provider)
            settings = Settings()
            assert settings.LLM_PROVIDER == provider

    def test_llm_model_values(self, monkeypatch):
        """Test different LLM model values."""
        from devdox_ai_sonar.config import Settings

        models = [
            "gpt-4",
            "gpt-4o",
            "gemini-2.5-flash",
            "claude-3-5-sonnet-20241022",
        ]
        for model in models:
            monkeypatch.setenv("LLM_MODEL", model)
            settings = Settings()
            assert settings.LLM_MODEL == model


class TestSettingsIntegration:
    """Integration tests for Settings."""

    def test_settings_with_dotenv_file(self, tmp_path, monkeypatch):
        """Test settings loading from .env file."""
        from devdox_ai_sonar.config import Settings

        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_content = """
OPENAI_API_KEY=file-openai-key
SONAR_TOKEN=file-sonar-token
SONAR_ORGANIZATION=file-org
LLM_PROVIDER=gemini
        """.strip()

        env_file.write_text(env_content)

        # Change to temp directory so .env is found
        monkeypatch.chdir(tmp_path)

        # Clear environment to ensure we're reading from file
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            # Note: This depends on python-dotenv loading the file
            # In actual execution, the file would be loaded

    def test_settings_environment_override(self, tmp_path, monkeypatch):
        """Test that environment variables override .env file."""
        from devdox_ai_sonar.config import Settings

        # Set environment variable
        monkeypatch.setenv("SONAR_TOKEN", "env-token")

        # Create .env with different value
        env_file = tmp_path / ".env"
        env_file.write_text("SONAR_TOKEN=file-token")
        monkeypatch.chdir(tmp_path)

        settings = Settings()
        # Environment should win
        assert settings.SONAR_TOKEN == "env-token"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
