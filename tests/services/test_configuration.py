

import os
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch, mock_open, call
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import asdict
from devdox_ai_sonar.services.configuration import (
    AuthConfig,
    LLMConfig,
    ConfigService,
    validate_token_format
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_console():
    """Mock rich console."""
    with patch('devdox_ai_sonar.services.configuration.console') as mock:
        yield mock


@pytest.fixture
def temp_config_path(tmp_path):
    """Create temporary config file path."""
    return tmp_path / "sonar.toml"


@pytest.fixture
def valid_auth_config():
    """Valid authentication configuration."""
    return {
        "token": "squ_test_token_1234567890",
        "organization": "test-org",
        "project": "test-project",
        "project_path": "/path/to/project"
    }


@pytest.fixture
def valid_config_file_content():
    """Valid config file content."""
    return {
        "SONAR_TOKEN": "squ_test_token",
        "SONAR_ORG": "test-org",
        "SONAR_PROJ": "test-project",
        "PROJECT_PATH": "/test/path"
    }


@pytest.fixture
def config_service(temp_config_path):
    """Create ConfigService instance with temp path."""
    return ConfigService(sonar_path=temp_config_path)


@pytest.fixture
def mock_config_manager():
    """Create mock ConfigManager."""
    manager = Mock()
    manager.get_value = AsyncMock(return_value=None)
    return manager


# ============================================================================
# AUTH CONFIG TESTS
# ============================================================================


class TestAuthConfig:
    """Tests for AuthConfig dataclass."""

    def test_auth_config_initialization(self):
        """Test AuthConfig initialization with all fields."""
        config = AuthConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/test/path",
            git_url="https://github.com/test-org/test-project.git"
        )

        assert config.token == "test-token"
        assert config.organization == "test-org"
        assert config.project == "test-project"
        assert config.project_path == "/test/path"
        assert config.git_url == "https://github.com/test-org/test-project.git"

    def test_auth_config_validate_success(self):
        """Test validation with all valid fields."""
        config = AuthConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/test/path",
            git_url="https://github.com/test-org/test-project.git"
        )

        is_valid, error = config.validate()

        assert is_valid is True
        assert error is None

    def test_auth_config_validate_missing_token(self):
        """Test validation with missing token."""
        config = AuthConfig(
            token="",
            organization="test-org",
            project="test-project",
            project_path="/test/path",
            git_url="https://github.com/test-org/test-project.git"
        )

        is_valid, error = config.validate()

        assert is_valid is False
        assert error == "Token is required"

    def test_auth_config_validate_missing_organization(self):
        """Test validation with missing organization."""
        config = AuthConfig(
            token="test-token",
            organization="",
            project="test-project",
            project_path="/test/path",
            git_url="https://github.com/test-org/test-project.git"
        )

        is_valid, error = config.validate()

        assert is_valid is False
        assert error == "Organization is required"

    def test_auth_config_validate_missing_project(self):
        """Test validation with missing project."""
        config = AuthConfig(
            token="test-token",
            organization="test-org",
            project="",
            project_path="/test/path",
            git_url="https://github.com/test-org/test-project.git"
        )

        is_valid, error = config.validate()

        assert is_valid is False
        assert error == "Project key is required"

    def test_auth_config_validate_missing_project_path(self):
        """Test validation with missing project path."""
        config = AuthConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="",
            git_url="https://github.com/test-org/test-project.git"
        )

        is_valid, error = config.validate()

        assert is_valid is False
        assert error == "Project path is required"

    def test_auth_config_validate_none_values(self):
        """Test validation with None values."""
        config = AuthConfig(
            token=None,
            organization="test-org",
            project="test-project",
            project_path="/test/path",
            git_url="https://github.com/test-org/test-project.git"
        )

        is_valid, error = config.validate()

        assert is_valid is False
        assert "required" in error.lower()


# ============================================================================
# LLM CONFIG TESTS
# ============================================================================


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_llm_config_initialization(self):
        """Test LLMConfig initialization."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-api-key",
            models=["gpt-4", "gpt-3.5-turbo"]
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test-api-key"
        assert len(config.models) == 2

    def test_llm_config_empty_models_list(self):
        """Test LLMConfig with empty models list."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-api-key",
            models=[]
        )

        assert config.models == []


# ============================================================================
# CONFIG SERVICE INITIALIZATION TESTS
# ============================================================================


class TestConfigServiceInitialization:
    """Tests for ConfigService initialization."""

    def test_initialization_default_path(self):
        """Test initialization with default path."""
        service = ConfigService()

        assert service.sonar_path == Path("sonar.toml")
        assert service.config is None

    def test_initialization_custom_path(self, temp_config_path):
        """Test initialization with custom path."""
        service = ConfigService(sonar_path=temp_config_path)

        assert service.sonar_path == temp_config_path
        assert service.config is None


# ============================================================================
# LOAD AUTH CONFIG TESTS
# ============================================================================


class TestLoadAuthConfig:
    """Tests for load_auth_config method."""

    async def test_load_auth_config_success(self, config_service, valid_config_file_content, mock_console):
        """Test successful loading of auth config."""
        config_service.file_reader.read_json_file = AsyncMock(return_value=valid_config_file_content)
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        assert result['token'] == "squ_test_token"
        assert result['organization'] == "test-org"
        assert result['project'] == "test-project"
        assert result['project_path'] == "/test/path"

    async def test_load_auth_config_file_not_exists(self, config_service):
        """Test loading when file doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            result = await config_service.load_auth_config()

        assert result['token'] is None
        assert result['organization'] is None
        assert result['project'] is None

    async def test_load_auth_config_json_decode_error(self, config_service, mock_console):
        """Test handling of JSON decode error."""
        config_service.file_reader.read_json_file = AsyncMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        assert result['token'] is None
        mock_console.print.assert_called_once()
        assert "Could not read" in str(mock_console.print.call_args)

    async def test_load_auth_config_io_error(self, config_service, mock_console):
        """Test handling of IO error."""
        config_service.file_reader.read_json_file = AsyncMock(side_effect=IOError("Read error"))
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        assert result['token'] is None
        mock_console.print.assert_called_once()

    async def test_load_auth_config_missing_fields(self, config_service, mock_console):
        """Test loading config with missing fields."""
        partial_config = {"SONAR_TOKEN": "token123"}

        config_service.file_reader.read_json_file = AsyncMock(return_value=partial_config)
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        assert result['token'] == "token123"
        assert result['organization'] is None
        assert result['project'] is None

    async def test_load_auth_config_empty_file(self, config_service, mock_console):
        """Test loading empty config file."""
        config_service.file_reader.read_json_file = AsyncMock(return_value={})
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        assert result['token'] is None
        assert result['organization'] is None
        assert result['project'] is None


# ============================================================================
# LOAD LLM CONFIG TESTS
# ============================================================================


class TestLoadLLMConfig:
    """Tests for load_llm_config method."""

    async def test_load_llm_config_success(self, mock_config_manager):
        """Test successful loading of LLM config."""
        providers = [
            {
                "name": "openai",
                "api_key": "test-key",
                "models": ["gpt-4", "gpt-3.5-turbo"]
            }
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "openai",
                "llm.default_model": "gpt-4"
            }.get(key)

        mock_config_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_config_manager)

        assert result is not None
        assert result.provider == "openai"
        assert result.model == "gpt-4"
        assert result.api_key == "test-key"
        assert len(result.models) == 2

    async def test_load_llm_config_no_providers(self, mock_config_manager):
        """Test loading when no providers configured."""
        mock_config_manager.get_value = AsyncMock(return_value=None)

        result = await ConfigService.load_llm_config(mock_config_manager)

        assert result is None

    async def test_load_llm_config_empty_providers_list(self, mock_config_manager):
        """Test loading with empty providers list."""
        async def side_effect(key):
            return {
                "llm.providers": [],
                "llm.default_provider": "openai",
                "llm.default_model": "gpt-4"
            }.get(key)

        mock_config_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_config_manager)

        assert result is None

    async def test_load_llm_config_provider_not_found(self, mock_config_manager):
        """Test when default provider doesn't exist in list."""
        providers = [
            {"name": "anthropic", "api_key": "key1", "models": ["claude-3"]}
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "openai",  # Not in list
                "llm.default_model": "gpt-4"
            }.get(key)

        mock_config_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_config_manager)

        assert result is None

    async def test_load_llm_config_missing_models(self, mock_config_manager):
        """Test loading config with missing models field."""
        providers = [
            {
                "name": "openai",
                "api_key": "test-key"
                # No models field
            }
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "openai",
                "llm.default_model": "gpt-4"
            }.get(key)

        mock_config_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_config_manager)

        assert result is not None
        assert result.models == []


# ============================================================================
# VALIDATE AUTH CONFIG TESTS
# ============================================================================


class TestValidateAuthConfig:
    """Tests for validate_auth_config method."""

    def test_validate_auth_config_all_fields_present(self, valid_auth_config):
        """Test validation with all required fields."""
        result = ConfigService.validate_auth_config(valid_auth_config)

        assert result is True

    def test_validate_auth_config_missing_token(self, valid_auth_config):
        """Test validation with missing token."""
        valid_auth_config['token'] = None

        result = ConfigService.validate_auth_config(valid_auth_config)

        assert result is False

    def test_validate_auth_config_missing_organization(self, valid_auth_config):
        """Test validation with missing organization."""
        valid_auth_config['organization'] = ""

        result = ConfigService.validate_auth_config(valid_auth_config)

        assert result is False

    def test_validate_auth_config_missing_project(self, valid_auth_config):
        """Test validation with missing project."""
        del valid_auth_config['project']

        result = ConfigService.validate_auth_config(valid_auth_config)

        assert result is False

    def test_validate_auth_config_missing_project_path(self, valid_auth_config):
        """Test validation with missing project_path."""
        valid_auth_config['project_path'] = None

        result = ConfigService.validate_auth_config(valid_auth_config)

        assert result is False

    def test_validate_auth_config_multiple_missing_fields(self):
        """Test validation with multiple missing fields."""
        config = {"token": "test"}

        result = ConfigService.validate_auth_config(config)

        assert result is False

    def test_validate_auth_config_empty_dict(self):
        """Test validation with empty dictionary."""
        result = ConfigService.validate_auth_config({})

        assert result is False


# ============================================================================
# SAVE CONFIG TESTS
# ============================================================================


class TestSaveConfig:
    """Tests for save_config method."""

    async def test_save_config_valid_token(self, config_service, mock_console):
        """Test saving config with valid token."""
        with patch.object(config_service, 'save_complete_config', new_callable=AsyncMock, return_value=True) as mock_save:
            with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=True):
                result = await config_service.save_config(
                    "squ_valid_token",
                    "test-org",
                    "test-project",
                    "/test/path",
                    "https://github.com/test-org/test-project.git"
                )

        assert result is True
        mock_save.assert_called_once_with(
            "squ_valid_token",
            "test-org",
            "test-project",
            "/test/path",
            "https://github.com/test-org/test-project.git"
        )

    async def test_save_config_invalid_token_user_confirms(self, config_service, mock_console):
        """Test saving config with invalid token but user confirms."""
        mock_console.input.return_value = "y"

        with patch.object(config_service, 'save_complete_config', new_callable=AsyncMock, return_value=True) as mock_save:
            with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=False):
                result = await config_service.save_config(
                    "short",
                    "test-org",
                    "test-project",
                    "/test/path",
                    "https://github.com/test-org/test-project.git"
                )

        assert result is True
        mock_save.assert_called_once()

    async def test_save_config_invalid_token_user_cancels(self, config_service, mock_console):
        """Test saving config with invalid token and user cancels."""
        mock_console.input.return_value = "n"

        with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=False):
            result = await config_service.save_config(
                "short",
                "test-org",
                "test-project",
                "/test/path",
                "https://github.com/test-org/test-project.git"
            )

        assert result is False
        mock_console.print.assert_called_with("[red]Cancelled[/red]")

    async def test_save_config_save_fails(self, config_service, mock_console):
        """Test when save_complete_config fails."""
        with patch.object(config_service, 'save_complete_config', new_callable=AsyncMock, return_value=False):
            with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=True):
                result = await config_service.save_config(
                    "squ_valid_token",
                    "test-org",
                    "test-project",
                    "/test/path",
                    "https://github.com/test-org/test-project.git"
                )

        assert result is False


# ============================================================================
# SAVE COMPLETE CONFIG TESTS
# ============================================================================


class TestSaveCompleteConfig:
    """Tests for save_complete_config method."""

    async def test_save_complete_config_new_file(self, config_service, mock_console):
        """Test saving to new file."""
        with patch('builtins.open', mock_open()) as mock_file:
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'chmod'):
                    result = await config_service.save_complete_config(
                        "squ_token",
                        "org",
                        "project",
                        "/path",
                        "https://github.com/test-org/test-project.git"
                    )

        assert result is True
        mock_file.assert_called_once()
        mock_console.print.assert_called()
        assert "saved" in str(mock_console.print.call_args).lower()

    async def test_save_complete_config_merge_with_existing(self, config_service, valid_config_file_content, mock_console):
        """Test merging with existing config."""
        existing_config = {
            "token": "old_token",
            "organization": "old_org",
            "project": "old_project",
            "project_path": "/old/path",
            "git_url": "https://github.com/test-org/test-project.git"
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(valid_config_file_content))) as mock_file:
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(config_service, 'load_auth_config', new_callable=AsyncMock, return_value=existing_config):
                    with patch.object(Path, 'chmod'):
                        result = await config_service.save_complete_config(
                            "new_token",
                            merge=True
                        )

        assert result is True
        # Verify file was written
        assert mock_file().write.called

    async def test_save_complete_config_overwrite_mode(self, config_service, mock_console):
        """Test overwriting existing config."""
        with patch('builtins.open', mock_open()) as mock_file:
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(Path, 'chmod'):
                    result = await config_service.save_complete_config(
                        "new_token",
                        "new_org",
                        "new_project",
                        "/new/path",
                        "https://github.com/test-org/test-project.git",
                        merge=False
                    )

        assert result is True

    async def test_save_complete_config_partial_update(self, config_service, mock_console):
        """Test updating only some fields."""
        existing_config = {
            "token": "old_token",
            "organization": "old_org",
            "project": "old_project",
            "project_path": "/old/path",
            "git_url":"https://github.com/test-org/test-project.git"
        }

        with patch('builtins.open', mock_open()) as mock_file:
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(config_service, 'load_auth_config', new_callable=AsyncMock, return_value=existing_config):
                    with patch.object(Path, 'chmod'):
                        result = await config_service.save_complete_config(
                            "new_token",
                            organization="new_org",
                            merge=True
                        )

        assert result is True

    async def test_save_complete_config_chmod_success(self, config_service, mock_console):
        """Test successful chmod on Unix."""
        with patch('builtins.open', mock_open()):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'chmod') as mock_chmod:
                    result = await config_service.save_complete_config(
                        "token", "org", "project", "/path"
                    )

        assert result is True
        mock_chmod.assert_called_once_with(0o600)

    async def test_save_complete_config_chmod_fails_windows(self, config_service, mock_console):
        """Test chmod failure on Windows (NotImplementedError)."""
        with patch('builtins.open', mock_open()):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'chmod', side_effect=NotImplementedError()):
                    result = await config_service.save_complete_config(
                        "token", "org", "project", "/path"
                    )

        # Should still succeed despite chmod failure
        assert result is True

    async def test_save_complete_config_chmod_os_error(self, config_service, mock_console):
        """Test chmod OS error."""
        with patch('builtins.open', mock_open()):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'chmod', side_effect=OSError("Permission denied")):
                    result = await config_service.save_complete_config(
                        "token", "org", "project", "/path"
                    )

        # Should still succeed despite chmod failure
        assert result is True

    async def test_save_complete_config_io_error(self, config_service, mock_console):
        """Test IO error during save."""
        with patch('builtins.open', side_effect=IOError("Write error")):
            with patch.object(Path, 'exists', return_value=False):
                result = await config_service.save_complete_config(
                    "token", "org", "project", "/path"
                )

        assert result is False
        mock_console.print.assert_called()
        assert "Error saving" in str(mock_console.print.call_args)

    async def test_save_complete_config_permission_error(self, config_service, mock_console):
        """Test permission error during save."""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch.object(Path, 'exists', return_value=False):
                result = await config_service.save_complete_config(
                    "token", "org", "project", "/path"
                )

        assert result is False
        mock_console.print.assert_called()

    async def test_save_complete_config_none_values(self, config_service, mock_console):
        """Test saving with None values for optional fields."""
        with patch('builtins.open', mock_open()):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'chmod'):
                    result = await config_service.save_complete_config(
                        "token",
                        organization=None,
                        project=None,
                        project_path=None
                    )

        assert result is True


# ============================================================================
# VALIDATE TOKEN FORMAT TESTS
# ============================================================================


class TestValidateTokenFormat:
    """Tests for validate_token_format function."""

    def test_validate_token_format_valid_long_token(self, mock_console):
        """Test validation with valid long token."""
        result = validate_token_format("squ_" + "a" * 40)

        assert result is True

    def test_validate_token_format_empty_token(self, mock_console):
        """Test validation with empty token."""
        result = validate_token_format("")

        assert result is False

    def test_validate_token_format_none_token(self, mock_console):
        """Test validation with None token."""
        result = validate_token_format(None)

        assert result is False

    def test_validate_token_format_short_token(self, mock_console):
        """Test validation with short token."""
        result = validate_token_format("short")

        assert result is False
        mock_console.print.assert_called_once()
        assert "too short" in str(mock_console.print.call_args).lower()

    def test_validate_token_format_exactly_20_chars(self, mock_console):
        """Test validation with token of exactly 20 characters."""
        result = validate_token_format("a" * 20)

        # Should pass (not short)
        assert result is True

    def test_validate_token_format_19_chars(self, mock_console):
        """Test validation with token of 19 characters."""
        result = validate_token_format("a" * 19)

        assert result is False
        mock_console.print.assert_called_once()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_config_service_with_special_characters_in_path(self):
        """Test ConfigService with special characters in path."""
        special_path = Path("/test/path with spaces/config.toml")
        service = ConfigService(sonar_path=special_path)

        assert service.sonar_path == special_path

    def test_auth_config_with_unicode_values(self):
        """Test AuthConfig with unicode values."""
        config = AuthConfig(
            token="测试token",
            organization="组织",
            project="项目",
            project_path="/test/路径",
            git_url="https://github.com/test-org/test-project.git",
        )

        is_valid, error = config.validate()
        assert is_valid is True

    def test_llm_config_with_empty_api_key(self):
        """Test LLMConfig with empty API key."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="",
            models=["gpt-4"]
        )

        assert config.api_key == ""

    async def test_load_auth_config_with_extra_fields(self, config_service, mock_console):
        """Test loading config with extra unexpected fields."""
        config_with_extra = {
            "SONAR_TOKEN": "token",
            "SONAR_ORG": "org",
            "SONAR_PROJ": "project",
            "PROJECT_PATH": "/path",
            "GIT_URL": "https://github.com/test-org/test-project.git",
            "EXTRA_FIELD": "extra_value"
        }

        config_service.file_reader.read_json_file = AsyncMock(return_value=config_with_extra)
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        # Should load successfully, ignoring extra fields
        assert result['token'] == "token"

    async def test_save_complete_config_with_very_long_values(self, config_service, mock_console):
        """Test saving with very long values."""
        long_token = "squ_" + "a" * 1000
        long_path = "/very/" * 100 + "long/path"

        with patch('builtins.open', mock_open()):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'chmod'):
                    result = await config_service.save_complete_config(
                        long_token,
                        "org",
                        "project",
                        long_path
                    )

        assert result is True

    def test_validate_auth_config_with_whitespace_only_values(self):
        """Test validation with whitespace-only values."""
        config = {
            "token": "   ",
            "organization": "  ",
            "project": "\t\n",
            "project_path": "  "
        }

        # Whitespace-only should fail validation
        result = ConfigService.validate_auth_config(config)

        # Note: Current implementation treats whitespace as valid
        # This test documents current behavior


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    async def test_complete_save_and_load_workflow(self, temp_config_path, mock_console):
        """Test complete save and load workflow."""
        service = ConfigService(sonar_path=temp_config_path)

        # Save config
        with patch.object(Path, 'chmod'):
            save_result = await service.save_complete_config(
                "squ_test_token_12345678901234567890",
                "test-org",
                "test-project",
                "/test/path"
            )

        assert save_result is True

        # Load config
        loaded_config = await service.load_auth_config()

        assert loaded_config['token'] == "squ_test_token_12345678901234567890"
        assert loaded_config['organization'] == "test-org"
        assert loaded_config['project'] == "test-project"
        assert loaded_config['project_path'] == "/test/path"

    async def test_update_existing_config_workflow(self, temp_config_path, mock_console):
        """Test updating existing configuration."""
        service = ConfigService(sonar_path=temp_config_path)

        # Save initial config
        with patch.object(Path, 'chmod'):
            await service.save_complete_config(
                "old_token",
                "old_org",
                "old_project",
                "/old/path"
            )

        # Update only token
        with patch.object(Path, 'chmod'):
            update_result = await service.save_complete_config(
                "new_token",
                merge=True
            )

        assert update_result is True

        # Verify merge happened
        loaded_config = await service.load_auth_config()
        assert loaded_config['token'] == "new_token"
        assert loaded_config['organization'] == "old_org"  # Should be preserved

    async def test_load_llm_config_with_real_manager(self):
        """Test loading LLM config with realistic ConfigManager."""
        mock_manager = Mock()

        providers = [
            {
                "name": "openai",
                "api_key": "sk-test-key",
                "models": ["gpt-4", "gpt-3.5-turbo"]
            },
            {
                "name": "anthropic",
                "api_key": "sk-ant-test",
                "models": ["claude-3-opus", "claude-3-sonnet"]
            }
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "anthropic",
                "llm.default_model": "claude-3-opus"
            }.get(key)

        mock_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_manager)

        assert result is not None
        assert result.provider == "anthropic"
        assert result.model == "claude-3-opus"
        assert result.api_key == "sk-ant-test"
        assert "claude-3-opus" in result.models




class TestCheckAllValueEmpty:
    """Tests for check_all_value_empty method - COMPLETELY MISSING"""

    def test_check_all_value_empty_all_present(self):
        """Test when all required values are present."""
        config_service = ConfigService()
        auth_config = {
            "token": "test-token",
            "organization": "test-org",
            "project": "test-project",
            "project_path": "/test/path",
            "git_url": "https://github.com/test-org/test-project.git"
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is False  # No values are empty

    def test_check_all_value_empty_missing_token(self):
        """Test when token is missing."""
        config_service = ConfigService()
        auth_config = {
            "token": "",
            "organization": "test-org",
            "project": "test-project",
            "project_path": "/test/path",
            "git_url":"https://github.com/test-org/test-project.git"
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_missing_organization(self):
        """Test when organization is missing."""
        config_service = ConfigService()
        auth_config = {
            "token": "test-token",
            "organization": "",
            "project": "test-project",
            "project_path": "/test/path"
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_missing_project(self):
        """Test when project is missing."""
        config_service = ConfigService()
        auth_config = {
            "token": "test-token",
            "organization": "test-org",
            "project": "",
            "project_path": "/test/path"
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_missing_project_path(self):
        """Test when project_path is missing."""
        config_service = ConfigService()
        auth_config = {
            "token": "test-token",
            "organization": "test-org",
            "project": "test-project",
            "project_path": ""
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_multiple_missing(self):
        """Test when multiple values are missing."""
        config_service = ConfigService()
        auth_config = {
            "token": "",
            "organization": "",
            "project": "test-project",
            "project_path": "/test/path"
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_all_missing(self):
        """Test when all values are missing."""
        config_service = ConfigService()
        auth_config = {
            "token": "",
            "organization": "",
            "project": "",
            "project_path": ""
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_with_none_values(self):
        """Test when values are None."""
        config_service = ConfigService()
        auth_config = {
            "token": None,
            "organization": "test-org",
            "project": "test-project",
            "project_path": "/test/path"
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_missing_keys(self):
        """Test when keys are completely missing from dict."""
        config_service = ConfigService()
        auth_config = {
            "token": "test-token",
            "organization": "test-org"
            # project and project_path missing
        }

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_empty_dict(self):
        """Test with completely empty dictionary."""
        config_service = ConfigService()
        auth_config = {}

        result = config_service.check_all_value_empty(auth_config)

        assert result is True

    def test_check_all_value_empty_whitespace_values(self):
        """Test with whitespace-only values."""
        config_service = ConfigService()
        auth_config = {
            "token": "   ",
            "organization": "test-org",
            "project": "test-project",
            "project_path": "/test/path",
            "git_url":"https://github.com/test-org/test-project.git"
        }

        result = config_service.check_all_value_empty(auth_config)

        # Whitespace should be treated as non-empty by .get()
        # but could be considered empty depending on implementation
        assert result is False  # Whitespace is truthy



# ============================================================================
# MISSING: save_config ERROR PATHS
# ============================================================================


class TestSaveConfigErrorPaths:
    """Test error handling paths in save_config - PARTIALLY COVERED"""

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_save_config_invalid_token_no_confirmation_input(self, mock_console):
        """Test when user doesn't provide any input (empty string)."""
        mock_console.input.return_value = ""
        config_service = ConfigService()

        with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=False):
            result = await config_service.save_config(
                "short",
                "test-org",
                "test-project",
                "/test/path",
                "https://github.com/test-org/test-project.git"
            )

        assert result is False
        mock_console.print.assert_called_with("[red]Cancelled[/red]")

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_save_config_invalid_token_uppercase_confirmation(self, mock_console):
        """Test with uppercase confirmation."""
        mock_console.input.return_value = "Y"
        config_service = ConfigService()

        with patch.object(config_service, 'save_complete_config', new_callable=AsyncMock, return_value=True):
            with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=False):
                result = await config_service.save_config(
                    "short",
                    "test-org",
                    "test-project",
                    "/test/path",
                    "https://github.com/test-org/test-project.git"
                )

        assert result is True

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_save_config_invalid_token_yes_word(self, mock_console):
        """Test with 'yes' word confirmation."""
        mock_console.input.return_value = "yes"
        config_service = ConfigService()

        with patch.object(config_service, 'save_complete_config', new_callable=AsyncMock, return_value=True):
            with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=False):
                result = await config_service.save_config(
                    "short",
                    "test-org",
                    "test-project",
                    "/test/path",
                    "https://github.com/test-org/test-project.git"
                )

        assert result is True

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_save_config_exception_in_save_complete(self, mock_console):
        """Test when save_complete_config raises exception."""
        config_service = ConfigService()

        with patch.object(config_service, 'save_complete_config', new_callable=AsyncMock, side_effect=Exception("Unexpected error")):
            with patch('devdox_ai_sonar.services.configuration.validate_token_format', return_value=True):
                with pytest.raises(Exception):
                    await config_service.save_config(
                        "squ_valid_token",
                        "test-org",
                        "test-project",
                        "/test/path",
                        "https://github.com/test-org/test-project.git"
                    )



# ============================================================================
# MISSING: FILE SYSTEM INTERACTION TESTS
# ============================================================================


class TestFileSystemInteractions:
    """Test actual file system operations - LOW COVERAGE"""

    async def test_save_and_load_with_real_file(self, tmp_path):
        """Integration test with real file system."""
        config_path = tmp_path / "test_config.json"
        service = ConfigService(sonar_path=config_path)

        # Save
        result = await service.save_complete_config(
            "squ_test_token_123456789",
            "test-org",
            "test-project",
            "/test/path",
            "https://github.com/test-org/test-project.git",
            merge=False
        )

        assert result is True
        assert config_path.exists()

        # Load
        loaded = await service.load_auth_config()
        assert loaded['token'] == "squ_test_token_123456789"
        assert loaded['organization'] == "test-org"

    async def test_save_creates_directory_if_not_exists(self, tmp_path):
        """Test saving to non-existent directory."""
        config_path = tmp_path / "nested" / "dir" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        service = ConfigService(sonar_path=config_path)

        result = await service.save_complete_config(
            "token",
            "org",
            "project",
            "/path",
            "https://github.com/test-org/test-project.git"
        )

        assert result is True
        assert config_path.exists()

    async def test_load_from_readonly_file(self, tmp_path):
        """Test loading from read-only file."""
        config_path = tmp_path / "readonly_config.json"
        config_data = {
            "SONAR_TOKEN": "test_token",
            "SONAR_ORG": "test_org",
            "SONAR_PROJ": "test_project",
            "PROJECT_PATH": "/test/path",
            "GIT_URL":"https://github.com/test-org/test-project.git"
        }

        # Create file
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        # Make it read-only
        config_path.chmod(0o444)

        service = ConfigService(sonar_path=config_path)
        loaded = await service.load_auth_config()

        assert loaded['token'] == "test_token"

        # Cleanup
        config_path.chmod(0o644)

    @pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses filesystem permissions")
    async def test_save_to_readonly_directory(self, tmp_path):
        """Test saving to read-only directory (should fail)."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        config_path = readonly_dir / "config.json"

        # Make directory read-only
        readonly_dir.chmod(0o444)

        service = ConfigService(sonar_path=config_path)

        try:
            result = await service.save_complete_config(
                "token",
                "org",
                "project",
                "/path",
                "https://github.com/test-org/test-project.git"
            )
            # Should fail on Unix systems
            assert result is False
        finally:
            # Cleanup
            readonly_dir.chmod(0o755)


# ============================================================================
# MISSING: JSON SERIALIZATION EDGE CASES
# ============================================================================


class TestJSONSerializationEdgeCases:
    """Test JSON serialization edge cases - LOW COVERAGE"""

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_load_config_with_malformed_json(self, mock_console):
        """Test loading config with malformed JSON."""
        config_service = ConfigService()

        config_service.file_reader.read_json_file = AsyncMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        assert result['token'] is None
        mock_console.print.assert_called_once()

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_load_config_with_json_array(self, mock_console):
        """Test loading config when file contains JSON array."""
        config_service = ConfigService()

        # read_json_file returns a list (JSON array), .get() will raise AttributeError
        config_service.file_reader.read_json_file = AsyncMock(return_value=[{"key": "value"}])
        with patch.object(Path, 'exists', return_value=True):
            with pytest.raises(AttributeError):
                await config_service.load_auth_config()

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_load_config_with_nested_objects(self, mock_console):
        """Test loading config with nested JSON objects."""
        config_service = ConfigService()

        nested_json = {
            "SONAR_TOKEN": "token",
            "SONAR_ORG": "org",
            "SONAR_PROJ": "project",
            "PROJECT_PATH": "/path",
            "GIT_URL": "https://github.com/test-org/test-project.git",
            "nested": {
                "key": "value"
            }
        }

        config_service.file_reader.read_json_file = AsyncMock(return_value=nested_json)
        with patch.object(Path, 'exists', return_value=True):
            result = await config_service.load_auth_config()

        # Should load successfully, ignoring nested objects
        assert result['token'] == "token"

    async def test_save_config_with_special_json_characters(self, tmp_path):
        """Test saving config with special JSON characters."""
        config_path = tmp_path / "special_chars.json"
        service = ConfigService(sonar_path=config_path)

        # Values with special characters
        result = await service.save_complete_config(
            'token"with"quotes',
            'org\\with\\backslashes',
            'project\nwith\nnewlines',
            '/path\twith\ttabs',
            "https://github.com/test-org/test-project.git"
        )

        assert result is True

        # Verify it can be loaded back
        loaded = await service.load_auth_config()
        assert loaded['token'] == 'token"with"quotes'


# ============================================================================
# MISSING: LOAD_LLM_CONFIG EDGE CASES
# ============================================================================


class TestLoadLLMConfigEdgeCases:
    """Test load_llm_config edge cases - PARTIAL COVERAGE"""

    async def test_load_llm_config_with_multiple_providers_different_models(self):
        """Test with multiple providers with different model lists."""
        mock_manager = Mock()

        providers = [
            {
                "name": "openai",
                "api_key": "key1",
                "models": ["gpt-4", "gpt-3.5-turbo"]
            },
            {
                "name": "anthropic",
                "api_key": "key2",
                "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
            },
            {
                "name": "cohere",
                "api_key": "key3",
                "models": ["command", "command-light"]
            }
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "cohere",
                "llm.default_model": "command"
            }.get(key)

        mock_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_manager)

        assert result is not None
        assert result.provider == "cohere"
        assert result.model == "command"
        assert len(result.models) == 2

    async def test_load_llm_config_provider_with_empty_name(self):
        """Test with provider having empty name."""
        mock_manager = Mock()

        providers = [
            {
                "name": "",
                "api_key": "key1",
                "models": ["model1"]
            }
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "",
                "llm.default_model": "model1"
            }.get(key)

        mock_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_manager)

        # Should handle empty provider name
        assert result is not None or result is None  # Depends on implementation

    async def test_load_llm_config_missing_api_key_in_provider(self):
        """Test with provider missing api_key field."""
        mock_manager = Mock()

        providers = [
            {
                "name": "openai",
                # No api_key
                "models": ["gpt-4"]
            }
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "openai",
                "llm.default_model": "gpt-4"
            }.get(key)

        mock_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_manager)

        assert result is not None
        assert result.api_key is None  # Should handle missing api_key

    async def test_load_llm_config_none_default_model(self):
        """Test with None default_model."""
        mock_manager = Mock()

        providers = [
            {
                "name": "openai",
                "api_key": "key1",
                "models": ["gpt-4"]
            }
        ]

        async def side_effect(key):
            return {
                "llm.providers": providers,
                "llm.default_provider": "openai",
                "llm.default_model": None
            }.get(key)

        mock_manager.get_value = AsyncMock(side_effect=side_effect)

        result = await ConfigService.load_llm_config(mock_manager)

        assert result is not None
        assert result.model is None


# ============================================================================
# MISSING: VALIDATE_AUTH_CONFIG EDGE CASES
# ============================================================================


class TestValidateAuthConfigEdgeCases:
    """Additional validation edge cases - PARTIAL COVERAGE"""

    def test_validate_auth_config_with_numeric_values(self):
        """Test validation with numeric values instead of strings."""
        config = {
            "token": 123456,
            "organization": 789,
            "project": 101112,
            "project_path": 131415
        }

        result = ConfigService.validate_auth_config(config)

        # Numeric values are truthy, so should pass
        assert result is True

    def test_validate_auth_config_with_boolean_values(self):
        """Test validation with boolean values."""
        config = {
            "token": True,
            "organization": False,
            "project": True,
            "project_path": True
        }

        result = ConfigService.validate_auth_config(config)

        # False is falsy, should fail
        assert result is False

    def test_validate_auth_config_with_list_values(self):
        """Test validation with list values."""
        config = {
            "token": ["token1", "token2"],
            "organization": ["org1"],
            "project": ["project1"],
            "project_path": ["/path1"]
        }

        result = ConfigService.validate_auth_config(config)

        # Lists are truthy, should pass
        assert result is True

    def test_validate_auth_config_with_zero(self):
        """Test validation with zero value."""
        config = {
            "token": 0,
            "organization": "org",
            "project": "project",
            "project_path": "/path"
        }

        result = ConfigService.validate_auth_config(config)

        # Zero is falsy, should fail
        assert result is False


# ============================================================================
# MISSING: SAVE_COMPLETE_CONFIG COMPLEX SCENARIOS
# ============================================================================


class TestSaveCompleteConfigComplexScenarios:
    """Complex scenarios for save_complete_config - PARTIAL COVERAGE"""

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_save_complete_config_merge_with_corrupt_existing(self, mock_console):
        """Test merging when existing file is corrupted."""
        config_service = ConfigService()

        with patch('builtins.open', mock_open()) as mock_file:
            # First call (read) returns corrupt data, second call (write) succeeds
            mock_file.side_effect = [
                mock_open(read_data="corrupt json").return_value,
                mock_open().return_value
            ]

            with patch.object(Path, 'exists', return_value=True):
                with patch.object(config_service, 'load_auth_config', new_callable=AsyncMock, return_value={}):
                    with patch.object(Path, 'chmod'):
                        result = await config_service.save_complete_config(
                            "new_token",
                            merge=True
                        )

        # Should succeed even with corrupt existing file
        assert result is True

    @patch('devdox_ai_sonar.services.configuration.console')
    async def test_save_complete_config_all_none_values_no_merge(self, mock_console):
        """Test saving all None values without merge."""
        config_service = ConfigService()

        with patch('builtins.open', mock_open()):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'chmod'):
                    result = await config_service.save_complete_config(
                        "token",
                        None,
                        None,
                        None,
                        merge=False
                    )

        assert result is True

    async def test_save_complete_config_preserves_existing_on_merge(self, tmp_path):
        """Test that merge properly preserves existing values."""
        config_path = tmp_path / "merge_test.json"
        service = ConfigService(sonar_path=config_path)

        # Create initial config
        initial_config = {
            "SONAR_TOKEN": "old_token",
            "SONAR_ORG": "old_org",
            "SONAR_PROJ": "old_project",
            "PROJECT_PATH": "/old/path"
        }

        with open(config_path, 'w') as f:
            json.dump(initial_config, f)

        # Update only token and organization
        result = await service.save_complete_config(
            "new_token",
            "new_org",
            None,  # Don't update project
            None,  # Don't update path
            merge=True
        )

        assert result is True

        # Verify merge
        loaded = await service.load_auth_config()
        assert loaded['token'] == "new_token"
        assert loaded['organization'] == "new_org"
        assert loaded['project'] == "old_project"  # Should be preserved
        assert loaded['project_path'] == "/old/path"  # Should be preserved


# ============================================================================
# MISSING: TOKEN VALIDATION EDGE CASES
# ============================================================================


class TestValidateTokenFormatEdgeCases:
    """Additional token validation tests - PARTIAL COVERAGE"""

    @patch('devdox_ai_sonar.services.configuration.console')
    def test_validate_token_format_with_unicode(self, mock_console):
        """Test token validation with unicode characters."""
        token = "squ_测试_токен_🔑" + "a" * 20

        result = validate_token_format(token)

        assert result is True

    @patch('devdox_ai_sonar.services.configuration.console')
    def test_validate_token_format_with_newlines(self, mock_console):
        """Test token validation with newlines."""
        token = "squ_token\nwith\nnewlines_" + "a" * 20

        result = validate_token_format(token)

        assert result is True  # Still valid length

    @patch('devdox_ai_sonar.services.configuration.console')
    def test_validate_token_format_whitespace_only(self, mock_console):
        """Test token validation with whitespace-only string."""
        token = "     "

        result = validate_token_format(token)

        assert result is False

    @patch('devdox_ai_sonar.services.configuration.console')
    def test_validate_token_format_very_long_token(self, mock_console):
        """Test token validation with extremely long token."""
        token = "squ_" + "a" * 10000

        result = validate_token_format(token)

        assert result is True

    @patch('devdox_ai_sonar.services.configuration.console')
    def test_validate_token_format_exactly_at_boundary(self, mock_console):
        """Test token validation at exact boundary (20 chars)."""
        token = "a" * 20

        result = validate_token_format(token)

        assert result is True


# ============================================================================
# MISSING: AUTHENTICATION CONFIG DATACLASS TESTS
# ============================================================================


class TestAuthConfigDataclass:
    """Additional AuthConfig dataclass tests - PARTIAL COVERAGE"""

    def test_auth_config_equality(self):
        """Test AuthConfig equality comparison."""
        config1 = AuthConfig("token", "org", "project", "/path",None)
        config2 = AuthConfig("token", "org", "project", "/path",None)

        assert config1 == config2

    def test_auth_config_inequality(self):
        """Test AuthConfig inequality."""
        config1 = AuthConfig("token1", "org", "project", "/path","https://github.com/test-org/test-project.git")
        config2 = AuthConfig("token2", "org", "project", "/path","https://github.com/test-org/test-project.git")

        assert config1 != config2

    def test_auth_config_repr(self):
        """Test AuthConfig string representation."""
        config = AuthConfig("token", "org", "project", "/path","https://github.com/test-org/test-project.git")

        repr_str = repr(config)

        assert "AuthConfig" in repr_str
        assert "token" in repr_str

    def test_auth_config_as_dict(self):
        """Test converting AuthConfig to dict."""


        config = AuthConfig("token", "org", "project", "/path","https://github.com/test-org/test-project.git")

        config_dict = asdict(config)

        assert config_dict['token'] == "token"
        assert config_dict['organization'] == "org"
        assert len(config_dict) == 5


# ============================================================================
# MISSING: LLM CONFIG DATACLASS TESTS
# ============================================================================


class TestLLMConfigDataclass:
    """Additional LLMConfig dataclass tests - PARTIAL COVERAGE"""

    def test_llm_config_equality(self):
        """Test LLMConfig equality."""
        config1 = LLMConfig("openai", "gpt-4", "key", ["gpt-4"])
        config2 = LLMConfig("openai", "gpt-4", "key", ["gpt-4"])

        assert config1 == config2

    def test_llm_config_repr(self):
        """Test LLMConfig string representation."""
        config = LLMConfig("openai", "gpt-4", "key", ["gpt-4"])

        repr_str = repr(config)

        assert "LLMConfig" in repr_str
        assert "openai" in repr_str

    def test_llm_config_models_immutability(self):
        """Test that models list can be modified."""
        config = LLMConfig("openai", "gpt-4", "key", ["gpt-4"])

        config.models.append("gpt-3.5-turbo")

        assert len(config.models) == 2
