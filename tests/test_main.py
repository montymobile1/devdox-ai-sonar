"""
Comprehensive test suite for CLI module with >90% coverage.

Tests cover:
- All helper functions
- Main CLI commands
- Error handling paths
- Edge cases and boundary conditions
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from click.testing import CliRunner
import click

# Import the module under test (adjust import path as needed)
from devdox_ai_sonar.main import (
    main,
    init_config,
    update_provider,
    _initialize_managers,
    _configure_sonarcloud,
    _configure_providers_loop,
    _handle_provider_configuration,
    _should_stop_configuring,
    _check_reconfiguration_consent,
    _select_existing_provider,
    _display_operation_header,
    _display_completion_message,
    _handle_cli_error,
    _handle_add_new_provider,
    _handle_update_existing_provider,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_managers():
    """Create mock manager instances."""
    manager = Mock()
    ui = Mock()
    validator = Mock()
    provider_manager = Mock()
    sonar_ui = Mock()
    config_service = Mock()

    return {
        "manager": manager,
        "ui": ui,
        "validator": validator,
        "provider_manager": provider_manager,
        "sonar_ui": sonar_ui,
        "config_service": config_service,
    }


@pytest.fixture
def mock_sonar_config():
    """Create mock SonarCloud configuration."""
    config = Mock()
    config.token = "test-token"
    config.organization = "test-org"
    config.project = "test-project"
    config.project_path = "/test/path"
    return config


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


# ============================================================================
# HELPER FUNCTIONS TESTS
# ============================================================================


class TestInitializeManagers:
    """Tests for _initialize_managers function."""

    @patch('devdox_ai_sonar.main.ConfigManager')
    @patch('devdox_ai_sonar.main.ProviderConfigUI')
    @patch('devdox_ai_sonar.main.ProviderValidator')
    @patch('devdox_ai_sonar.main.ProviderConfigManager')
    @patch('devdox_ai_sonar.main.SonarCloudConfigUI')
    @patch('devdox_ai_sonar.main.ConfigService')
    @patch('devdox_ai_sonar.main.settings')
    def test_initialize_managers_returns_all_components(
            self, mock_settings, mock_config_service, mock_sonar_ui,
            mock_provider_manager, mock_validator, mock_ui, mock_manager
    ):
        """Test that all manager components are initialized and returned."""
        mock_settings.config_file_path = "/test/config.yaml"
        mock_settings.auth_file_path = "/test/auth.yaml"

        result = _initialize_managers()

        assert len(result) == 6
        mock_manager.assert_called_once_with(config_path="/test/config.yaml")
        mock_ui.assert_called_once()
        mock_validator.assert_called_once()
        mock_sonar_ui.assert_called_once()
        mock_config_service.assert_called_once()


class TestConfigureSonarcloud:
    """Tests for _configure_sonarcloud function."""

    def test_configure_sonarcloud_success(self, mock_managers, mock_sonar_config):
        """Test successful SonarCloud configuration."""
        sonar_ui = mock_managers["sonar_ui"]
        config_service = mock_managers["config_service"]

        config_service.load_auth_config.return_value = {"existing": "config"}
        sonar_ui.configure_sonarcloud.return_value = mock_sonar_config
        config_service.save_config.return_value = True

        result = _configure_sonarcloud(sonar_ui, config_service)

        assert result is True
        sonar_ui.display_welcome.assert_called_once()
        sonar_ui.display_step_header.assert_called_once_with(1, 2, "SonarCloud Configuration")
        config_service.load_auth_config.assert_called_once()
        sonar_ui.configure_sonarcloud.assert_called_once()
        config_service.save_config.assert_called_once_with(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/test/path",
        )

    def test_configure_sonarcloud_cancelled(self, mock_managers):
        """Test SonarCloud configuration when user cancels."""
        sonar_ui = mock_managers["sonar_ui"]
        config_service = mock_managers["config_service"]

        sonar_ui.configure_sonarcloud.return_value = None

        result = _configure_sonarcloud(sonar_ui, config_service)

        assert result is False
        config_service.save_config.assert_not_called()

    def test_configure_sonarcloud_save_failed(self, mock_managers, mock_sonar_config):
        """Test SonarCloud configuration when save fails."""
        sonar_ui = mock_managers["sonar_ui"]
        config_service = mock_managers["config_service"]

        sonar_ui.configure_sonarcloud.return_value = mock_sonar_config
        config_service.save_config.return_value = False

        result = _configure_sonarcloud(sonar_ui, config_service)

        assert result is False


class TestHandleProviderConfiguration:
    """Tests for _handle_provider_configuration function."""

    def test_provider_configuration_success(self, mock_managers):
        """Test successful provider configuration."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        available_providers = ["openai", "anthropic"]

        provider_manager.configure_new_provider.return_value = {
            "config": {"api_key": "test-key"},
            "set_as_default": True,
        }

        result = _handle_provider_configuration(
            provider_manager, manager, "openai", available_providers
        )

        assert result is True
        assert "openai" not in available_providers
        manager.add_provider.assert_called_once_with(
            {"api_key": "test-key"}, set_as_default=True
        )

    def test_provider_configuration_failed(self, mock_managers):
        """Test failed provider configuration."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        available_providers = ["openai", "anthropic"]

        provider_manager.configure_new_provider.return_value = None

        result = _handle_provider_configuration(
            provider_manager, manager, "openai", available_providers
        )

        assert result is False
        assert "openai" in available_providers  # Should not be removed
        manager.add_provider.assert_not_called()


class TestConfigureProvidersLoop:
    """Tests for _configure_providers_loop function."""

    @patch('devdox_ai_sonar.main._should_stop_configuring')
    @patch('devdox_ai_sonar.main._handle_provider_configuration')
    def test_configure_single_provider(
            self, mock_handle_config, mock_should_stop, mock_managers
    ):
        """Test configuring a single provider."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        ui = mock_managers["ui"]
        available_providers = ["openai"]

        ui.select_provider_from_list.return_value = "openai"
        mock_handle_config.return_value = True
        mock_should_stop.return_value = True

        _configure_providers_loop(provider_manager, manager, ui, available_providers)

        ui.select_provider_from_list.assert_called_once()
        mock_handle_config.assert_called_once_with(
            provider_manager, manager, "openai", available_providers
        )

    @patch('devdox_ai_sonar.main._should_stop_configuring')
    @patch('devdox_ai_sonar.main._handle_provider_configuration')
    def test_configure_multiple_providers(
            self, mock_handle_config, mock_should_stop, mock_managers
    ):
        """Test configuring multiple providers."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        ui = mock_managers["ui"]
        available_providers = ["openai", "anthropic", "together"]

        ui.select_provider_from_list.side_effect = ["openai", "anthropic"]
        mock_handle_config.return_value = True
        mock_should_stop.side_effect = [False, True]

        _configure_providers_loop(provider_manager, manager, ui, available_providers)

        assert ui.select_provider_from_list.call_count == 2
        assert mock_handle_config.call_count == 2

    def test_configure_providers_cancelled(self, mock_managers):
        """Test when user cancels provider selection."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        ui = mock_managers["ui"]
        available_providers = ["openai"]

        ui.select_provider_from_list.return_value = None

        _configure_providers_loop(provider_manager, manager, ui, available_providers)

        ui.select_provider_from_list.assert_called_once()
        # Loop should exit when None is returned


class TestShouldStopConfiguring:
    """Tests for _should_stop_configuring function."""

    def test_should_stop_when_no_providers_left(self):
        """Test stopping when all providers are configured."""
        result = _should_stop_configuring([])
        assert result is True

    @patch('devdox_ai_sonar.main.Confirm.ask')
    def test_should_stop_when_user_declines(self, mock_confirm):
        """Test stopping when user doesn't want to add more."""
        mock_confirm.return_value = False
        result = _should_stop_configuring(["openai"])
        assert result is True

    @patch('devdox_ai_sonar.main.Confirm.ask')
    def test_should_continue_when_user_confirms(self, mock_confirm):
        """Test continuing when user wants to add more."""
        mock_confirm.return_value = True
        result = _should_stop_configuring(["openai"])
        assert result is False


class TestCheckReconfigurationConsent:
    """Tests for _check_reconfiguration_consent function."""

    def test_consent_when_no_existing_providers(self):
        """Test consent granted when no providers exist."""
        result = _check_reconfiguration_consent([])
        assert result is True

        result = _check_reconfiguration_consent(None)
        assert result is True

    @patch('devdox_ai_sonar.main.Confirm.ask')
    def test_consent_when_user_confirms_reconfiguration(self, mock_confirm):
        """Test consent when user wants to reconfigure."""
        mock_confirm.return_value = True
        result = _check_reconfiguration_consent(["existing_provider"])
        assert result is True

    @patch('devdox_ai_sonar.main.Confirm.ask')
    def test_no_consent_when_user_declines(self, mock_confirm):
        """Test no consent when user declines reconfiguration."""
        mock_confirm.return_value = False
        result = _check_reconfiguration_consent(["existing_provider"])
        assert result is False


class TestSelectExistingProvider:
    """Tests for _select_existing_provider function."""

    @patch('devdox_ai_sonar.main.inquirer.prompt')
    def test_select_provider_success(self, mock_prompt):
        """Test successful provider selection."""
        mock_prompt.return_value = {"provider": "openai"}

        result = _select_existing_provider(["openai", "anthropic"])

        assert result == "openai"
        mock_prompt.assert_called_once()

    @patch('devdox_ai_sonar.main.inquirer.prompt')
    def test_select_provider_cancelled(self, mock_prompt):
        """Test when selection is cancelled."""
        mock_prompt.return_value = None

        result = _select_existing_provider(["openai"])

        assert result == ""

    @patch('devdox_ai_sonar.main.inquirer.prompt')
    def test_select_provider_no_answer(self, mock_prompt):
        """Test when no provider is selected."""
        mock_prompt.return_value = {"provider": None}

        result = _select_existing_provider(["openai"])

        assert result == ""


class TestDisplayFunctions:
    """Tests for display helper functions."""

    @patch('devdox_ai_sonar.main.console')
    def test_display_operation_header(self, mock_console):
        """Test operation header display."""
        _display_operation_header("TEST OPERATION")

        assert mock_console.print.call_count == 3
        calls = mock_console.print.call_args_list
        assert "TEST OPERATION" in str(calls[1])

    @patch('devdox_ai_sonar.main.console')
    @patch('devdox_ai_sonar.main.settings')
    def test_display_completion_message(self, mock_settings, mock_console):
        """Test completion message display."""
        mock_settings.config_file_path = "/test/config.yaml"

        _display_completion_message()

        assert mock_console.print.call_count == 4


class TestHandleCliError:
    """Tests for _handle_cli_error function."""

 

    @patch('devdox_ai_sonar.main.console')
    def test_handle_generic_exception(self, mock_console):
        """Test handling generic exceptions."""
        with pytest.raises(click.ClickException) as exc_info:
            _handle_cli_error(ValueError("Test error"))

        assert "Test error" in str(exc_info.value)
        mock_console.print.assert_called_once()


class TestHandleAddNewProvider:
    """Tests for _handle_add_new_provider function."""

    @patch('devdox_ai_sonar.main._display_operation_header')
    @patch('devdox_ai_sonar.main._configure_providers_loop')
    def test_add_new_provider_success(
            self, mock_configure_loop, mock_display_header, mock_managers
    ):
        """Test successfully adding new provider."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        ui = mock_managers["ui"]

        provider_manager.get_available_providers.return_value = ["openai"]

        _handle_add_new_provider(provider_manager, manager, ui)

        mock_display_header.assert_called_once()
        mock_configure_loop.assert_called_once()
        manager.save_config.assert_called_once_with(create_backup=False)

    def test_add_new_provider_no_available(self, mock_managers):
        """Test when no providers are available to add."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        ui = mock_managers["ui"]

        provider_manager.get_available_providers.return_value = []

        with pytest.raises(click.Abort):
            _handle_add_new_provider(provider_manager, manager, ui)


class TestHandleUpdateExistingProvider:
    """Tests for _handle_update_existing_provider function."""

    @patch('devdox_ai_sonar.main._display_operation_header')
    @patch('devdox_ai_sonar.main._select_existing_provider')
    def test_update_existing_provider_success(
            self, mock_select, mock_display_header, mock_managers
    ):
        """Test successfully updating existing provider."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]

        mock_select.return_value = "openai"
        provider_manager.update_existing_provider.return_value = True

        _handle_update_existing_provider(
            provider_manager, manager, ["openai", "anthropic"]
        )

        mock_display_header.assert_called_once()
        provider_manager.update_existing_provider.assert_called_once_with("openai")
        manager.save_config.assert_called_once_with(create_backup=False)

    @patch('devdox_ai_sonar.main._display_operation_header')
    @patch('devdox_ai_sonar.main._select_existing_provider')
    def test_update_existing_provider_cancelled(
            self, mock_select, mock_display_header, mock_managers
    ):
        """Test when provider selection is cancelled."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]

        mock_select.return_value = ""

        with pytest.raises(click.Abort):
            _handle_update_existing_provider(
                provider_manager, manager, ["openai"]
            )

    @patch('devdox_ai_sonar.main._display_operation_header')
    @patch('devdox_ai_sonar.main._select_existing_provider')
    def test_update_existing_provider_failed(
            self, mock_select, mock_display_header, mock_managers
    ):
        """Test when provider update fails."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]

        mock_select.return_value = "openai"
        provider_manager.update_existing_provider.return_value = False

        _handle_update_existing_provider(
            provider_manager, manager, ["openai"]
        )

        manager.save_config.assert_not_called()


# ============================================================================
# CLI COMMAND TESTS
# ============================================================================


class TestMainCommand:
    """Tests for main CLI command."""

    def test_main_command_version(self, cli_runner):
        """Test version option."""
        result = cli_runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Version should be displayed

    def test_main_command_help(self, cli_runner):
        """Test help option."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SonarCloud Analyzer" in result.output

    def test_main_command_verbose(self, cli_runner):
        """Test verbose flag."""
        result = cli_runner.invoke(main, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestInitConfigCommand:
    """Tests for init_config CLI command."""

    @patch('devdox_ai_sonar.main._initialize_managers')
    @patch('devdox_ai_sonar.main._check_reconfiguration_consent')
    @patch('devdox_ai_sonar.main._configure_sonarcloud')
    @patch('devdox_ai_sonar.main._configure_providers_loop')
    def test_init_config_success_new_config(
            self,
            mock_providers_loop,
            mock_sonar_config,
            mock_check_consent,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test successful init_config with new configuration."""
        manager = mock_managers["manager"]
        manager.get_value.return_value = None

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )
        mock_check_consent.return_value = True
        mock_sonar_config.return_value = True
        mock_managers["provider_manager"].get_available_providers.return_value = ["openai"]

        result = cli_runner.invoke(init_config)

        assert result.exit_code == 0
        manager.create_default_config.assert_called_once()
        manager.load_config.assert_called_once()
        manager.save_config.assert_called_once()

    @patch('devdox_ai_sonar.main._initialize_managers')
    @patch('devdox_ai_sonar.main._check_reconfiguration_consent')
    def test_init_config_user_declines_reconfiguration(
            self,
            mock_check_consent,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test init_config when user declines reconfiguration."""
        manager = mock_managers["manager"]
        manager.get_value.return_value = ["existing_provider"]

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )
        mock_check_consent.return_value = False

        result = cli_runner.invoke(init_config)

        assert result.exit_code == 0
        manager.save_config.assert_not_called()

    @patch('devdox_ai_sonar.main._initialize_managers')
    @patch('devdox_ai_sonar.main._check_reconfiguration_consent')
    @patch('devdox_ai_sonar.main._configure_sonarcloud')
    def test_init_config_sonarcloud_cancelled(
            self,
            mock_sonar_config,
            mock_check_consent,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test init_config when SonarCloud configuration is cancelled."""
        manager = mock_managers["manager"]
        manager.get_value.return_value = None

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )
        mock_check_consent.return_value = True
        mock_sonar_config.return_value = False

        result = cli_runner.invoke(init_config)

        assert result.exit_code == 1  # Should abort

    @patch('devdox_ai_sonar.main._initialize_managers')
    @patch('devdox_ai_sonar.main._check_reconfiguration_consent')
    @patch('devdox_ai_sonar.main._configure_sonarcloud')
    def test_init_config_no_available_providers(
            self,
            mock_sonar_config,
            mock_check_consent,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test init_config when no providers are available."""
        manager = mock_managers["manager"]
        manager.get_value.return_value = None

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )
        mock_check_consent.return_value = True
        mock_sonar_config.return_value = True
        mock_managers["provider_manager"].get_available_providers.return_value = []

        result = cli_runner.invoke(init_config)

        assert result.exit_code == 1

    @patch('devdox_ai_sonar.main._initialize_managers')
    def test_init_config_exception_handling(
            self,
            mock_init_managers,
            cli_runner,
    ):
        """Test init_config exception handling."""
        mock_init_managers.side_effect = Exception("Test error")

        result = cli_runner.invoke(init_config)

        assert result.exit_code != 0
        assert "Test error" in result.output


class TestUpdateProviderCommand:
    """Tests for update_provider CLI command."""

    @patch('devdox_ai_sonar.main._initialize_managers')
    @patch('devdox_ai_sonar.main._handle_add_new_provider')
    @patch('devdox_ai_sonar.main._display_completion_message')
    @patch('devdox_ai_sonar.main.Confirm.ask')
    def test_update_provider_add_new(
            self,
            mock_confirm,
            mock_display_completion,
            mock_handle_add,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test update_provider when adding new provider."""
        manager = mock_managers["manager"]
        mock_managers["provider_manager"].get_existing_providers.return_value = ["openai"]

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )
        mock_confirm.return_value = True

        result = cli_runner.invoke(update_provider)

        assert result.exit_code == 0
        mock_handle_add.assert_called_once()
        mock_display_completion.assert_called_once()

    @patch('devdox_ai_sonar.main._initialize_managers')
    @patch('devdox_ai_sonar.main._handle_update_existing_provider')
    @patch('devdox_ai_sonar.main._display_completion_message')
    @patch('devdox_ai_sonar.main.Confirm.ask')
    def test_update_provider_update_existing(
            self,
            mock_confirm,
            mock_display_completion,
            mock_handle_update,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test update_provider when updating existing provider."""
        manager = mock_managers["manager"]
        existing_providers = ["openai", "anthropic"]
        mock_managers["provider_manager"].get_existing_providers.return_value = existing_providers

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )
        mock_confirm.return_value = False

        result = cli_runner.invoke(update_provider)

        assert result.exit_code == 0
        mock_handle_update.assert_called_once_with(
            mock_managers["provider_manager"], manager, existing_providers
        )
        mock_display_completion.assert_called_once()

    @patch('devdox_ai_sonar.main._initialize_managers')
    def test_update_provider_no_existing_providers(
            self,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test update_provider when no providers exist."""
        manager = mock_managers["manager"]
        mock_managers["provider_manager"].get_existing_providers.return_value = []

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )

        result = cli_runner.invoke(update_provider)

        assert result.exit_code == 1
        assert "No providers configured" in result.output

    @patch('devdox_ai_sonar.main._initialize_managers')
    def test_update_provider_exception_handling(
            self,
            mock_init_managers,
            cli_runner,
    ):
        """Test update_provider exception handling."""
        mock_init_managers.side_effect = Exception("Test error")

        result = cli_runner.invoke(update_provider)

        assert result.exit_code != 0
        assert "Test error" in result.output


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch('devdox_ai_sonar.main._initialize_managers')
    @patch('devdox_ai_sonar.main._check_reconfiguration_consent')
    @patch('devdox_ai_sonar.main._configure_sonarcloud')
    @patch('devdox_ai_sonar.main.Confirm.ask')
    @patch('devdox_ai_sonar.main.console')
    def test_full_init_config_workflow(
            self,
            mock_console,
            mock_confirm,
            mock_sonar_config,
            mock_check_consent,
            mock_init_managers,
            cli_runner,
            mock_managers,
    ):
        """Test complete init_config workflow."""
        # Setup
        manager = mock_managers["manager"]
        manager.get_value.return_value = None
        mock_managers["provider_manager"].get_available_providers.return_value = ["openai", "anthropic"]
        mock_managers["provider_manager"].configure_new_provider.return_value = {
            "config": {"api_key": "test"},
            "set_as_default": True,
        }
        mock_managers["ui"].select_provider_from_list.side_effect = ["openai", None]

        mock_init_managers.return_value = (
            manager,
            mock_managers["ui"],
            mock_managers["validator"],
            mock_managers["provider_manager"],
            mock_managers["sonar_ui"],
            mock_managers["config_service"],
        )
        mock_check_consent.return_value = True
        mock_sonar_config.return_value = True
        mock_confirm.return_value = False  # Don't add another provider

        # Execute
        result = cli_runner.invoke(init_config)

        # Verify
        assert result.exit_code == 0
        manager.create_default_config.assert_called_once()
        manager.save_config.assert_called_once()


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_provider_list(self):
        """Test behavior with empty provider list."""
        result = _should_stop_configuring([])
        assert result is True

    def test_none_provider_list(self):
        """Test behavior with None provider list."""
        result = _check_reconfiguration_consent(None)
        assert result is True

    @patch('devdox_ai_sonar.main.inquirer.prompt')
    def test_select_provider_with_empty_list(self, mock_prompt):
        """Test selecting from empty provider list."""
        mock_prompt.return_value = {"provider": None}
        result = _select_existing_provider([])
        assert result == ""

    @patch('devdox_ai_sonar.main.console')
    def test_handle_error_with_none_message(self, mock_console):
        """Test error handling with exception without message."""

        class CustomException(Exception):
            def __str__(self):
                return ""

        with pytest.raises(click.ClickException):
            _handle_cli_error(CustomException())


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================


class TestPerformance:
    """Tests for performance considerations."""

    @patch('devdox_ai_sonar.main._should_stop_configuring')
    @patch('devdox_ai_sonar.main._handle_provider_configuration')
    def test_configure_many_providers(
            self, mock_handle_config, mock_should_stop, mock_managers
    ):
        """Test configuring a large number of providers."""
        provider_manager = mock_managers["provider_manager"]
        manager = mock_managers["manager"]
        ui = mock_managers["ui"]

        # Create a large list of providers
        available_providers = [f"provider_{i}" for i in range(100)]
        original_count = len(available_providers)

        ui.select_provider_from_list.side_effect = available_providers[:10] + [None]
        mock_handle_config.return_value = True
        mock_should_stop.return_value = False

        _configure_providers_loop(provider_manager, manager, ui, available_providers)

        # Should process providers until None is returned
        assert ui.select_provider_from_list.call_count <= original_count


