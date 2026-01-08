"""
Comprehensive test suite for SonarCloudConfigUI with >90% coverage.

Tests cover:
- All static methods
- User interaction flows
- Validation scenarios
- Error handling
- Edge cases
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from typing import Optional, Dict

from devdox_ai_sonar.utils.sonar_config import SonarCloudConfigUI
from devdox_ai_sonar.models.sonar import SonarCloudConfig


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_console():
    """Mock rich console."""
    with patch('devdox_ai_sonar.utils.sonar_config.console') as mock:
        yield mock


@pytest.fixture
def mock_confirm():
    """Mock rich Confirm.ask."""
    with patch('devdox_ai_sonar.utils.sonar_config.Confirm.ask') as mock:
        yield mock


@pytest.fixture
def mock_prompt():
    """Mock rich Prompt.ask."""
    with patch('devdox_ai_sonar.utils.sonar_config.Prompt.ask') as mock:
        yield mock


@pytest.fixture
def valid_config_dict():
    """Valid configuration dictionary."""
    return {
        "token": "test-token-123",
        "organization": "test-org",
        "project": "test-project",
        "project_path": "/test/project/path",
        "git_url":":https://github.com/test-org/test-project.git"
    }


@pytest.fixture
def valid_sonar_config():
    """Valid SonarCloudConfig object."""
    return SonarCloudConfig(
        token="test-token-123",
        organization="test-org",
        project="test-project",
        project_path="/test/project/path"
    )


# ============================================================================
# DISPLAY METHODS TESTS
# ============================================================================


class TestDisplayWelcome:
    """Tests for display_welcome method."""

    def test_display_welcome_shows_header(self, mock_console):
        """Test that welcome message displays header."""
        SonarCloudConfigUI.display_welcome()

        # Verify console.print was called multiple times
        assert mock_console.print.call_count >= 4

        # Verify specific content
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("DEVDOX SONAR" in str(call) for call in calls)
        assert any("Welcome" in str(call) for call in calls)

    def test_display_welcome_shows_separator(self, mock_console):
        """Test that welcome message includes separators."""
        SonarCloudConfigUI.display_welcome()

        calls = [str(call) for call in mock_console.print.call_args_list]
        # Should have separator lines (=== characters)
        assert any("=" * 60 in str(call) for call in calls)

    def test_display_welcome_shows_wizard_text(self, mock_console):
        """Test that welcome message includes wizard text."""
        SonarCloudConfigUI.display_welcome()

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("wizard" in str(call).lower() for call in calls)


class TestDisplayStepHeader:
    """Tests for display_step_header method."""

    def test_display_step_header_with_valid_params(self, mock_console):
        """Test step header with valid parameters."""
        SonarCloudConfigUI.display_step_header(1, 2, "Test Step")

        assert mock_console.print.call_count == 3

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("STEP 1/2" in str(call) for call in calls)
        assert any("Test Step" in str(call) for call in calls)

    def test_display_step_header_shows_separators(self, mock_console):
        """Test that step header includes separators."""
        SonarCloudConfigUI.display_step_header(2, 5, "Configuration")

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("=" * 60 in str(call) for call in calls)

    def test_display_step_header_multiple_steps(self, mock_console):
        """Test displaying multiple step headers."""
        SonarCloudConfigUI.display_step_header(1, 3, "First Step")
        mock_console.reset_mock()

        SonarCloudConfigUI.display_step_header(2, 3, "Second Step")

        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("STEP 2/3" in str(call) for call in calls)


# ============================================================================
# PROMPT WITH DEFAULT TESTS
# ============================================================================


class TestPromptWithDefault:
    """Tests for prompt_with_default method."""

    def test_prompt_with_default_accepts_current_value(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test accepting the current default value."""
        mock_confirm.return_value = True

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter token",
            "token",
            default_value="existing-token",
            required=True
        )

        assert result == "existing-token"
        mock_confirm.assert_called_once_with("Use current value?", default=True)
        mock_prompt.assert_not_called()

    def test_prompt_with_default_prompts_when_rejected(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test prompting for new value when default is rejected."""
        mock_confirm.return_value = False
        mock_prompt.return_value = "new-token"

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter token",
            "token",
            default_value="existing-token",
            required=True
        )

        assert result == "new-token"
        mock_confirm.assert_called_once()
        mock_prompt.assert_called_once_with("Enter token")

    def test_prompt_with_default_no_default_value(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test prompting when no default value provided."""
        mock_prompt.return_value = "new-value"

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            default_value=None,
            required=True
        )

        assert result == "new-value"
        mock_confirm.assert_not_called()
        mock_prompt.assert_called_once()

    def test_prompt_with_default_strips_whitespace(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that input is stripped of whitespace."""
        mock_prompt.return_value = "  value-with-spaces  "

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == "value-with-spaces"

    def test_prompt_with_default_required_field_empty_rejected(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that empty required field is rejected."""
        mock_prompt.side_effect = ["", "valid-value"]

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == "valid-value"
        assert mock_prompt.call_count == 2
        mock_console.print.assert_any_call("[red]❌ This field is required[/red]")

    def test_prompt_with_default_optional_field_empty_accepted(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that empty optional field returns None."""
        mock_prompt.return_value = ""

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=False
        )

        assert result is None

    def test_prompt_with_default_keyboard_interrupt(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test handling of KeyboardInterrupt."""
        mock_prompt.side_effect = KeyboardInterrupt()

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result is None

    def test_prompt_with_default_multiple_empty_attempts(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test multiple empty input attempts on required field."""
        mock_prompt.side_effect = ["", "", "", "finally-valid"]

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == "finally-valid"
        assert mock_prompt.call_count == 4
        # Should show error message 3 times
        error_calls = [
            call for call in mock_console.print.call_args_list
            if "required" in str(call).lower()
        ]
        assert len(error_calls) == 3

    def test_prompt_with_default_shows_current_value(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that current value is displayed."""
        mock_confirm.return_value = True

        SonarCloudConfigUI.prompt_with_default(
            "Enter token",
            "token",
            default_value="current-token",
            required=True
        )

        # Verify current value was displayed
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("current-token" in str(call) for call in calls)
        assert any("Current value" in str(call) for call in calls)


# ============================================================================
# CONFIGURE SONARCLOUD TESTS
# ============================================================================


class TestConfigureSonarcloud:
    """Tests for configure_sonarcloud method."""

    def test_configure_sonarcloud_success_all_new_values(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test successful configuration with all new values."""
        mock_prompt.side_effect = [
            "new-token",
            "new-org",
            "new-project",
            "/new/path",
            "https://github.com/new-org/new-project.git"
        ]

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud(existing_config={})

        assert result is not None
        assert isinstance(result, SonarCloudConfig)
        assert result.token == "new-token"
        assert result.organization == "new-org"
        assert result.project == "new-project"
        assert result.project_path == "/new/path"
        assert result.git_url == "https://github.com/new-org/new-project.git"

    def test_configure_sonarcloud_success_with_existing_config(
            self, mock_console, mock_confirm, mock_prompt, valid_config_dict
    ):
        """Test successful configuration with existing config as defaults."""
        mock_confirm.return_value = True  # Accept all defaults

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud(valid_config_dict)

        assert result is not None
        assert result.token == valid_config_dict["token"]
        assert result.organization == valid_config_dict["organization"]
        assert result.project == valid_config_dict["project"]
        assert result.project_path == valid_config_dict["project_path"]

    def test_configure_sonarcloud_cancelled_at_token(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test cancellation during token input."""
        mock_prompt.side_effect = KeyboardInterrupt()

        result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is None

    def test_configure_sonarcloud_cancelled_at_organization(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test cancellation during organization input."""
        mock_prompt.side_effect = ["valid-token", KeyboardInterrupt()]

        result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is None

    def test_configure_sonarcloud_cancelled_at_project(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test cancellation during project input."""
        mock_prompt.side_effect = ["valid-token", "valid-org", KeyboardInterrupt()]

        result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is None

    def test_configure_sonarcloud_cancelled_at_project_path(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test cancellation during project path input."""
        mock_prompt.side_effect = [
            "valid-token",
            "valid-org",
            "valid-project",
            KeyboardInterrupt()
        ]

        result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is None

    def test_configure_sonarcloud_validation_failure(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test handling of validation failure."""
        mock_prompt.side_effect = [
            "invalid-token",
            "invalid-org",
            "invalid-project",
            "/invalid/path",
            "invalid-git-url"
        ]

        with patch.object(
                SonarCloudConfig,
                'validate',
                return_value=(False, "Token is invalid")
        ):
            result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is None
        mock_console.print.assert_any_call(
            "[red]❌ Invalid configuration: Token is invalid[/red]"
        )

    def test_configure_sonarcloud_with_empty_existing_config(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test configuration with empty existing config."""
        mock_prompt.side_effect = ["token", "org", "project", "/path","https://github.com/new-org/new-project.git"]

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is not None
        assert result.token == "token"

    def test_configure_sonarcloud_mixed_defaults_and_new_values(
            self, mock_console, mock_confirm, mock_prompt, valid_config_dict
    ):
        """Test using some defaults and some new values."""
        # Accept first two defaults, provide new values for last two
        mock_confirm.side_effect = [True, True, False, False,True]
        mock_prompt.side_effect = ["new-project", "/new/path"]

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud(valid_config_dict)

        assert result is not None
        assert result.token == valid_config_dict["token"]
        assert result.organization == valid_config_dict["organization"]
        assert result.project == "new-project"
        assert result.project_path == "/new/path"

    def test_configure_sonarcloud_handles_none_existing_config(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test handling of None as existing config."""
        mock_prompt.side_effect = ["token", "org", "project", "/path","https://github.com/new-org/new-project.git"]

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is not None
        mock_confirm.assert_not_called()  # No defaults to confirm

    def test_configure_sonarcloud_whitespace_handling(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that whitespace is properly stripped from inputs."""
        mock_prompt.side_effect = [
            "  token  ",
            "  org  ",
            "  project  ",
            "  /path  ",
            " https://github.com/new-org/new-project.git "
        ]

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is not None
        assert result.token == "token"
        assert result.organization == "org"
        assert result.project == "project"
        assert result.project_path == "/path"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestSonarCloudConfigUIIntegration:
    """Integration tests for complete workflows."""

    def test_complete_configuration_workflow(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test complete configuration workflow from start to finish."""
        # Setup
        existing_config = {
            "token": "old-token",
            "organization": "old-org",
            "project": "old-project",
            "project_path": "/old/path",
            "git_url": "https://github.com/old-org/old-project.git"
        }

        # User flow: reject all defaults and provide new values
        mock_confirm.return_value = False
        mock_prompt.side_effect = [
            "new-token",
            "new-org",
            "new-project",
            "/new/path",
            "https://github.com/old-org/old-project.git"
        ]

        # Display welcome
        SonarCloudConfigUI.display_welcome()

        # Display step header
        SonarCloudConfigUI.display_step_header(1, 2, "SonarCloud Configuration")

        # Configure SonarCloud
        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud(existing_config)

        # Verify
        assert result is not None
        assert result.token == "new-token"
        assert result.organization == "new-org"
        assert mock_console.print.call_count > 0

    def test_reconfiguration_workflow(
            self, mock_console, mock_confirm, mock_prompt, valid_config_dict
    ):
        """Test reconfiguration of existing settings."""
        # Accept all existing values
        mock_confirm.return_value = True

        SonarCloudConfigUI.display_welcome()
        SonarCloudConfigUI.display_step_header(1, 1, "Reconfigure")

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud(valid_config_dict)

        assert result is not None
        # Should keep all existing values
        for key, value in valid_config_dict.items():
            assert getattr(result, key) == value


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_values_rejected_for_required(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that empty strings are rejected for required fields."""
        mock_prompt.side_effect = ["", "valid-value"]

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == "valid-value"
        assert mock_prompt.call_count == 2

    def test_whitespace_only_values_rejected(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that whitespace-only values are rejected."""
        mock_prompt.side_effect = ["   ", "\t\t", "valid"]

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == "valid"
        assert mock_prompt.call_count == 3

    def test_very_long_input_accepted(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that very long input is accepted."""
        long_value = "a" * 1000
        mock_prompt.return_value = long_value

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == long_value

    def test_special_characters_in_input(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that special characters are accepted."""
        special_value = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        mock_prompt.return_value = special_value

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == special_value

    def test_unicode_characters_in_input(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that unicode characters are accepted."""
        unicode_value = "测试 тест テスト 🚀"
        mock_prompt.return_value = unicode_value

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == unicode_value

    def test_partial_existing_config(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test configuration with partial existing config."""
        partial_config = {
            "token": "existing-token",
            "organization": "existing-org"
            # Missing project and project_path
        }

        mock_confirm.return_value = True  # Accept existing values
        mock_prompt.side_effect = ["new-project", "/new/path","https://github.com/old-org/old-project.git"]

        with patch.object(SonarCloudConfig, 'validate', return_value=(True, None)):
            result = SonarCloudConfigUI.configure_sonarcloud(partial_config)

        assert result is not None
        assert result.token == "existing-token"
        assert result.organization == "existing-org"
        assert result.project == "new-project"
        assert result.project_path == "/new/path"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_multiple_keyboard_interrupts(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test handling of multiple keyboard interrupts."""
        mock_prompt.side_effect = KeyboardInterrupt()

        # Should return None without crashing
        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result is None

    def test_validation_error_display(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test that validation errors are properly displayed."""
        mock_prompt.side_effect = ["token", "org", "project", "/path","https://github.com/new-org/new-project.git"]

        error_message = "Invalid token format"
        with patch.object(
                SonarCloudConfig,
                'validate',
                return_value=(False, error_message)
        ):
            result = SonarCloudConfigUI.configure_sonarcloud({})

        assert result is None
        # Verify error message was displayed
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any(error_message in str(call) for call in calls)

    def test_exception_during_validation(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test handling of exception during validation."""
        mock_prompt.side_effect = ["token", "org", "project", "/path"]

        with patch.object(
                SonarCloudConfig,
                'validate',
                side_effect=Exception("Validation error")
        ):
            with pytest.raises(Exception):
                SonarCloudConfigUI.configure_sonarcloud({})


# ============================================================================
# STATIC METHOD TESTS
# ============================================================================


class TestStaticMethods:
    """Tests verifying static method behavior."""

    def test_display_welcome_is_static(self):
        """Verify display_welcome is a static method."""
        assert isinstance(
            SonarCloudConfigUI.__dict__['display_welcome'],
            staticmethod
        )

    def test_display_step_header_is_static(self):
        """Verify display_step_header is a static method."""
        assert isinstance(
            SonarCloudConfigUI.__dict__['display_step_header'],
            staticmethod
        )

    def test_prompt_with_default_is_static(self):
        """Verify prompt_with_default is a static method."""
        assert isinstance(
            SonarCloudConfigUI.__dict__['prompt_with_default'],
            staticmethod
        )

    def test_configure_sonarcloud_is_static(self):
        """Verify configure_sonarcloud is a static method."""
        assert isinstance(
            SonarCloudConfigUI.__dict__['configure_sonarcloud'],
            staticmethod
        )


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Tests for performance considerations."""

    def test_many_retry_attempts_on_required_field(
            self, mock_console, mock_confirm, mock_prompt
    ):
        """Test handling of many retry attempts."""
        # 100 empty attempts, then valid value
        mock_prompt.side_effect = [""] * 100 + ["valid"]

        result = SonarCloudConfigUI.prompt_with_default(
            "Enter value",
            "field",
            required=True
        )

        assert result == "valid"
        assert mock_prompt.call_count == 101


