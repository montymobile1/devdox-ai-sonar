import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any, Optional

from devdox_ai_sonar.utils.provider_config import (
    ProviderUpdateContext,
    ProviderConfigUI,
    ProviderConfigManager,
)
from devdox_ai_sonar.models.llm import ProviderType
from devdox_ai_sonar.utils.exceptions import ValidationError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_console():
    """Mock rich console."""
    with patch('devdox_ai_sonar.utils.provider_config.console') as mock:
        yield mock


@pytest.fixture
def mock_confirm():
    """Mock rich Confirm.ask."""
    with patch('devdox_ai_sonar.utils.provider_config.Confirm.ask') as mock:
        yield mock


@pytest.fixture
def mock_prompt():
    """Mock rich Prompt.ask."""
    with patch('devdox_ai_sonar.utils.provider_config.Prompt.ask') as mock:
        yield mock


@pytest.fixture
def mock_terminal_menu():
    """Mock TerminalMenu."""
    with patch('devdox_ai_sonar.utils.provider_config.TerminalMenu') as mock:
        yield mock


@pytest.fixture
def mock_config_manager():
    """Create mock config manager."""
    manager = Mock()
    manager.get_value = Mock(return_value=None)
    manager.set_value = Mock()
    manager.save_config = Mock()
    manager.update_provider = Mock()
    return manager


@pytest.fixture
def mock_validator():
    """Create mock provider validator."""
    validator = Mock()
    validation_result = Mock()
    validation_result.success = True
    validation_result.models = ["model-1", "model-2", "model-3"]
    validation_result.error_message = None
    validator.validate = Mock(return_value=validation_result)
    return validator


@pytest.fixture
def sample_provider_config():
    """Sample provider configuration."""
    return {
        "name": "openai",
        "api_key": "test-api-key",
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "default_model": "gpt-4"
    }


@pytest.fixture
def provider_config_manager(mock_config_manager, mock_validator):
    """Create ProviderConfigManager instance."""
    ui = ProviderConfigUI()
    return ProviderConfigManager(mock_config_manager, ui, mock_validator)


# ============================================================================
# PROVIDER UPDATE CONTEXT TESTS
# ============================================================================


class TestProviderUpdateContext:
    """Tests for ProviderUpdateContext class."""

    def test_initialization_with_valid_provider(self, sample_provider_config):
        """Test context initialization with valid provider."""
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        assert ctx.provider == sample_provider_config
        assert ctx.provider_name == "openai"
        assert ctx.updates == {}
        assert ctx.current_model == "gpt-4"
        assert ctx.available_models == []

    def test_initialization_without_default_model(self):
        """Test context initialization when provider has no default model."""
        provider = {
            "name": "anthropic",
            "api_key": "test-key",
            "models": ["claude-3"]
        }

        ctx = ProviderUpdateContext(provider, "anthropic")

        assert ctx.current_model == ""
        assert ctx.provider_name == "anthropic"

    def test_updates_dictionary_is_empty_initially(self, sample_provider_config):
        """Test that updates dictionary starts empty."""
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        assert isinstance(ctx.updates, dict)
        assert len(ctx.updates) == 0

    def test_context_attributes_mutable(self, sample_provider_config):
        """Test that context attributes can be modified."""
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        ctx.updates["api_key"] = "new-key"
        ctx.current_model = "gpt-3.5-turbo"
        ctx.available_models = ["model-1", "model-2"]

        assert ctx.updates["api_key"] == "new-key"
        assert ctx.current_model == "gpt-3.5-turbo"
        assert len(ctx.available_models) == 2


# ============================================================================
# PROVIDER CONFIG UI TESTS
# ============================================================================


class TestPromptForApiKey:
    """Tests for prompt_for_api_key method."""

    def test_prompt_for_api_key_success(self, mock_console, mock_prompt):
        """Test successful API key prompt."""
        mock_prompt.return_value = "test-api-key-123"

        result = ProviderConfigUI.prompt_for_api_key("openai")

        assert result == "test-api-key-123"
        mock_prompt.assert_called_once_with("Enter your OPENAI API key")

    def test_prompt_for_api_key_strips_whitespace(self, mock_console, mock_prompt):
        """Test that API key input is stripped."""
        mock_prompt.return_value = "  test-api-key  "

        result = ProviderConfigUI.prompt_for_api_key("anthropic")

        assert result == "test-api-key"

    def test_prompt_for_api_key_empty_rejected(self, mock_console, mock_prompt):
        """Test that empty API key is rejected."""
        mock_prompt.return_value = ""

        result = ProviderConfigUI.prompt_for_api_key("openai")

        assert result is None
        mock_console.print.assert_called_with("[red]❌ API key cannot be empty[/red]")

    def test_prompt_for_api_key_whitespace_only_rejected(self, mock_console, mock_prompt):
        """Test that whitespace-only API key is rejected."""
        mock_prompt.return_value = "   "

        result = ProviderConfigUI.prompt_for_api_key("openai")

        assert result is None

    def test_prompt_for_api_key_keyboard_interrupt(self, mock_console, mock_prompt):
        """Test handling of keyboard interrupt."""
        mock_prompt.side_effect = KeyboardInterrupt()

        result = ProviderConfigUI.prompt_for_api_key("openai")

        assert result is None

    def test_prompt_for_api_key_multiple_providers(self, mock_console, mock_prompt):
        """Test prompting for different providers."""
        mock_prompt.return_value = "key"

        ProviderConfigUI.prompt_for_api_key("openai")
        assert "OPENAI" in str(mock_prompt.call_args)

        mock_prompt.reset_mock()
        ProviderConfigUI.prompt_for_api_key("anthropic")
        assert "ANTHROPIC" in str(mock_prompt.call_args)


class TestSelectProviderFromList:
    """Tests for select_provider_from_list method."""

    def test_select_provider_success(self, mock_terminal_menu):
        """Test successful provider selection."""
        providers = ["openai", "anthropic", "together"]
        mock_menu_instance = Mock()
        mock_menu_instance.show.return_value = 1
        mock_terminal_menu.return_value = mock_menu_instance

        result = ProviderConfigUI.select_provider_from_list(providers, "Select provider")

        assert result == "anthropic"
        mock_terminal_menu.assert_called_once()
        mock_menu_instance.show.assert_called_once()

    def test_select_provider_first_item(self, mock_terminal_menu):
        """Test selecting first item from list."""
        providers = ["openai", "anthropic"]
        mock_menu_instance = Mock()
        mock_menu_instance.show.return_value = 0
        mock_terminal_menu.return_value = mock_menu_instance

        result = ProviderConfigUI.select_provider_from_list(providers, "Select")

        assert result == "openai"

    def test_select_provider_cancelled(self, mock_terminal_menu):
        """Test when selection is cancelled."""
        providers = ["openai"]
        mock_menu_instance = Mock()
        mock_menu_instance.show.return_value = None
        mock_terminal_menu.return_value = mock_menu_instance

        result = ProviderConfigUI.select_provider_from_list(providers, "Select")

        assert result is None

    def test_select_provider_empty_list(self, mock_terminal_menu):
        """Test with empty provider list."""
        result = ProviderConfigUI.select_provider_from_list([], "Select")

        assert result is None
        mock_terminal_menu.assert_not_called()

    def test_select_provider_menu_configuration(self, mock_terminal_menu):
        """Test that menu is configured correctly."""
        providers = ["openai"]
        mock_menu_instance = Mock()
        mock_menu_instance.show.return_value = 0
        mock_terminal_menu.return_value = mock_menu_instance

        ProviderConfigUI.select_provider_from_list(providers, "Test message")

        call_kwargs = mock_terminal_menu.call_args[1]
        assert call_kwargs['search_key'] == "/"
        assert call_kwargs['search_case_sensitive'] is False
        assert call_kwargs['title'] == "Test message"
        assert call_kwargs['menu_cursor'] == "➤ "


class TestConfirmDefault:
    """Tests for confirm_default method."""

    def test_confirm_default_accepts(self, mock_confirm):
        """Test when user confirms default."""
        mock_confirm.return_value = True

        result = ProviderConfigUI.confirm_default("Test message?")

        assert result is True
        mock_confirm.assert_called_once_with("Test message?", default=False)

    def test_confirm_default_declines(self, mock_confirm):
        """Test when user declines default."""
        mock_confirm.return_value = False

        result = ProviderConfigUI.confirm_default()

        assert result is False

    def test_confirm_default_keyboard_interrupt(self, mock_confirm):
        """Test handling of keyboard interrupt."""
        mock_confirm.side_effect = KeyboardInterrupt()

        result = ProviderConfigUI.confirm_default()

        assert result is False

    def test_confirm_default_uses_default_message(self, mock_confirm):
        """Test default message when none provided."""
        mock_confirm.return_value = True

        ProviderConfigUI.confirm_default()

        assert "default provider" in str(mock_confirm.call_args)


# ============================================================================
# PROVIDER CONFIG MANAGER - BASIC METHODS TESTS
# ============================================================================


class TestGetAvailableProviders:
    """Tests for get_available_providers method."""

    def test_get_available_providers_none_configured(self, provider_config_manager, mock_config_manager):
        """Test when no providers are configured."""
        mock_config_manager.get_value.return_value = []

        with patch.object(ProviderType, 'choices', return_value=["openai", "anthropic", "together"]):
            result = provider_config_manager.get_available_providers()

        assert len(result) == 3
        assert "openai" in result
        assert "anthropic" in result
        assert "together" in result

    def test_get_available_providers_some_configured(self, provider_config_manager, mock_config_manager):
        """Test when some providers are already configured."""
        existing = [
            {"name": "openai", "api_key": "key1"},
            {"name": "anthropic", "api_key": "key2"}
        ]
        mock_config_manager.get_value.return_value = existing

        with patch.object(ProviderType, 'choices', return_value=["openai", "anthropic", "together"]):
            result = provider_config_manager.get_available_providers()

        assert len(result) == 1
        assert result == ["together"]

    def test_get_available_providers_all_configured(self, provider_config_manager, mock_config_manager):
        """Test when all providers are configured."""
        existing = [
            {"name": "openai"},
            {"name": "anthropic"},
            {"name": "together"}
        ]
        mock_config_manager.get_value.return_value = existing

        with patch.object(ProviderType, 'choices', return_value=["openai", "anthropic", "together"]):
            result = provider_config_manager.get_available_providers()

        assert result == []

    def test_get_available_providers_null_config(self, provider_config_manager, mock_config_manager):
        """Test when config returns None."""
        mock_config_manager.get_value.return_value = None

        with patch.object(ProviderType, 'choices', return_value=["openai"]):
            result = provider_config_manager.get_available_providers()

        assert "openai" in result


class TestGetDefaultProvider:
    """Tests for get_default_provider method."""

    def test_get_default_provider_exists(self, provider_config_manager, mock_config_manager):
        """Test when default provider is set."""
        mock_config_manager.get_value.return_value = "openai"

        result = provider_config_manager.get_default_provider()

        assert result == "openai"
        mock_config_manager.get_value.assert_called_once_with("llm.default_provider")

    def test_get_default_provider_none(self, provider_config_manager, mock_config_manager):
        """Test when no default provider is set."""
        mock_config_manager.get_value.return_value = None

        result = provider_config_manager.get_default_provider()

        assert result is None


class TestGetExistingProviders:
    """Tests for get_existing_providers method."""

    def test_get_existing_providers_multiple(self, provider_config_manager, mock_config_manager):
        """Test getting multiple existing providers."""
        existing = [
            {"name": "openai", "api_key": "key1"},
            {"name": "anthropic", "api_key": "key2"},
            {"name": "together", "api_key": "key3"}
        ]
        mock_config_manager.get_value.return_value = existing

        result = provider_config_manager.get_existing_providers()

        assert result == ["openai", "anthropic", "together"]

    def test_get_existing_providers_empty(self, provider_config_manager, mock_config_manager):
        """Test when no providers exist."""
        mock_config_manager.get_value.return_value = []

        result = provider_config_manager.get_existing_providers()

        assert result == []

    def test_get_existing_providers_none(self, provider_config_manager, mock_config_manager):
        """Test when config returns None."""
        mock_config_manager.get_value.return_value = None

        result = provider_config_manager.get_existing_providers()

        assert result == []


# ============================================================================
# CONFIGURE NEW PROVIDER TESTS
# ============================================================================


class TestConfigureNewProvider:
    """Tests for configure_new_provider method."""

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.select_provider_from_list')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.confirm_default')
    def test_configure_new_provider_success(
            self, mock_confirm, mock_select, mock_prompt_key,
            provider_config_manager, mock_console, mock_validator
    ):
        """Test successful new provider configuration."""
        mock_prompt_key.return_value = "test-api-key"
        mock_select.return_value = "model-1"
        mock_confirm.return_value = True

        result = provider_config_manager.configure_new_provider("openai")

        assert result is not None
        assert result["config"]["name"] == "openai"
        assert result["config"]["api_key"] == "test-api-key"
        assert result["config"]["default_model"] == "model-1"
        assert result["set_as_default"] is True
        mock_validator.validate.assert_called_once()

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    def test_configure_new_provider_invalid_name(
            self, mock_prompt_key, provider_config_manager, mock_console
    ):
        """Test with invalid provider name."""
        result = provider_config_manager.configure_new_provider("invalid_provider")

        assert result is None
        mock_console.print.assert_called_with("[red]❌ Unknown provider: invalid_provider[/red]")

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    def test_configure_new_provider_no_api_key(
            self, mock_prompt_key, provider_config_manager
    ):
        """Test when API key prompt is cancelled."""
        mock_prompt_key.return_value = None

        result = provider_config_manager.configure_new_provider("openai")

        assert result is None

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    def test_configure_new_provider_validation_failed(
            self, mock_prompt_key, provider_config_manager, mock_validator, mock_console
    ):
        """Test when API key validation fails."""
        mock_prompt_key.return_value = "invalid-key"
        mock_validator.validate.return_value.success = False
        mock_validator.validate.return_value.error_message = "Invalid API key"

        result = provider_config_manager.configure_new_provider("openai")

        assert result is None
        mock_console.print.assert_any_call("[red]❌ Invalid API key[/red]")

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.select_provider_from_list')
    def test_configure_new_provider_model_selection_cancelled(
            self, mock_select, mock_prompt_key, provider_config_manager, mock_console
    ):
        """Test when model selection is cancelled."""
        mock_prompt_key.return_value = "test-key"
        mock_select.return_value = None

        result = provider_config_manager.configure_new_provider("openai")

        assert result is None
        mock_console.print.assert_called_with("[yellow]⚠ Model selection cancelled[/yellow]")

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.select_provider_from_list')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.confirm_default')
    def test_configure_new_provider_not_default(
            self, mock_confirm, mock_select, mock_prompt_key, provider_config_manager
    ):
        """Test configuring provider without setting as default."""
        mock_prompt_key.return_value = "test-key"
        mock_select.return_value = "model-1"
        mock_confirm.return_value = False

        result = provider_config_manager.configure_new_provider("openai")

        assert result["set_as_default"] is False


# ============================================================================
# UPDATE EXISTING PROVIDER TESTS
# ============================================================================


class TestUpdateExistingProvider:
    """Tests for update_existing_provider method."""

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_update_existing_provider_not_found(
            self, mock_confirm, provider_config_manager, mock_config_manager, mock_console
    ):
        """Test updating non-existent provider."""
        mock_config_manager.get_value.return_value = []

        result = provider_config_manager.update_existing_provider("openai")

        assert result is False
        mock_console.print.assert_called_with("[red]❌ Provider openai not found[/red]")

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_update_existing_provider_keep_all_settings(
            self, mock_confirm, provider_config_manager, mock_config_manager, sample_provider_config
    ):
        """Test updating provider without changing anything."""
        mock_config_manager.get_value.return_value = [sample_provider_config]
        mock_confirm.side_effect = [False, True, False]  # Don't change API key, keep model, not default

        result = provider_config_manager.update_existing_provider("openai")

        assert result is False  # No updates applied

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    def test_update_existing_provider_change_api_key(
            self, mock_prompt_key, mock_confirm, provider_config_manager,
            mock_config_manager, sample_provider_config, mock_validator
    ):
        """Test updating provider API key."""
        mock_config_manager.get_value.return_value = [sample_provider_config]
        mock_confirm.side_effect = [True, True, False]  # Change API key, keep model, not default
        mock_prompt_key.return_value = "new-api-key"

        result = provider_config_manager.update_existing_provider("openai")

        assert result is True
        mock_config_manager.update_provider.assert_called_once()

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.select_provider_from_list')
    def test_update_existing_provider_change_model(
            self, mock_select, mock_confirm, provider_config_manager,
            mock_config_manager, sample_provider_config
    ):
        """Test updating provider default model."""
        mock_config_manager.get_value.return_value = [sample_provider_config]
        mock_confirm.side_effect = [False, False, False]  # Don't change API key, change model, not default
        mock_select.return_value = "gpt-3.5-turbo"

        result = provider_config_manager.update_existing_provider("openai")

        assert result is True
        call_args = mock_config_manager.update_provider.call_args[1]
        assert call_args['updates']['default_model'] == "gpt-3.5-turbo"

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_update_existing_provider_set_as_default(
            self, mock_confirm, provider_config_manager,
            mock_config_manager, sample_provider_config
    ):
        """Test setting provider as default."""
        mock_config_manager.get_value.return_value = [sample_provider_config]
        mock_confirm.side_effect = [False, True, True]  # Don't change API key, keep model, set as default

        result = provider_config_manager.update_existing_provider("openai")

        assert result is True
        call_args = mock_config_manager.update_provider.call_args[1]
        assert call_args['set_as_default'] is True


# ============================================================================
# BRANCH AND PR SELECTION TESTS
# ============================================================================


class TestBranchOrPrPrompt:
    """Tests for branch_or_pr_prompt method."""

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    def test_branch_or_pr_prompt_select_branch(
            self, mock_prompt, mock_confirm, provider_config_manager,
            mock_config_manager, mock_console
    ):
        """Test selecting branch analysis."""
        mock_confirm.side_effect = [False, False]  # Don't analyze PR, don't save as default
        mock_prompt.return_value = "develop"
        mock_config_manager.get_value.return_value = "main"

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_branch_name', return_value="develop"):
            branch, pr = provider_config_manager.branch_or_pr_prompt()

        assert branch == "develop"
        assert pr == 0

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    def test_branch_or_pr_prompt_select_pr(
            self, mock_prompt, mock_confirm, provider_config_manager,
            mock_config_manager, mock_console
    ):
        """Test selecting pull request analysis."""
        mock_confirm.side_effect = [True, False]  # Analyze PR, don't save as default
        mock_prompt.return_value = "123"
        mock_config_manager.get_value.return_value = None

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_pull_request_number',
                   return_value=123):
            branch, pr = provider_config_manager.branch_or_pr_prompt()

        assert branch == ""
        assert pr == 123


class TestHandleBranchSelection:
    """Tests for _handle_branch_selection method."""

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_branch_selection_default_branch(
            self, mock_confirm, mock_prompt, provider_config_manager,
            mock_config_manager, mock_console
    ):
        """Test branch selection using default."""
        mock_config_manager.get_value.return_value = "main"
        mock_prompt.return_value = "main"
        mock_confirm.return_value = False

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_branch_name', return_value="main"):
            branch, pr = provider_config_manager._handle_branch_selection()

        assert branch == "main"
        assert pr == 0

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_branch_selection_custom_branch(
            self, mock_confirm, mock_prompt, provider_config_manager,
            mock_config_manager, mock_console
    ):
        """Test branch selection with custom branch."""
        mock_config_manager.get_value.return_value = "main"
        mock_prompt.return_value = "feature/new-feature"
        mock_confirm.return_value = False

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_branch_name',
                   return_value="feature/new-feature"):
            branch, pr = provider_config_manager._handle_branch_selection()

        assert branch == "feature/new-feature"

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    def test_handle_branch_selection_invalid_branch(
            self, mock_prompt, provider_config_manager, mock_config_manager, mock_console
    ):
        """Test branch selection with invalid branch name."""
        mock_config_manager.get_value.return_value = "main"
        mock_prompt.return_value = ""

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_branch_name',
                   side_effect=ValidationError("Invalid branch","branch")):
            with pytest.raises(Exception):  # Should raise click.Abort
                provider_config_manager._handle_branch_selection()

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_branch_selection_save_as_default(
            self, mock_confirm, mock_prompt, provider_config_manager, mock_config_manager
    ):
        """Test saving new branch as default."""
        mock_config_manager.get_value.return_value = "main"
        mock_prompt.return_value = "develop"
        mock_confirm.return_value = True

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_branch_name', return_value="develop"):
            provider_config_manager._handle_branch_selection()

        mock_config_manager.set_value.assert_called_with("sonar.default_branch", "develop")
        mock_config_manager.save_config.assert_called_once()


class TestHandlePullRequestSelection:
    """Tests for _handle_pull_request_selection method."""

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_pull_request_selection_default(
            self, mock_confirm, mock_prompt, provider_config_manager, mock_config_manager
    ):
        """Test PR selection using default."""
        mock_config_manager.get_value.return_value = 100
        mock_prompt.return_value = "100"
        mock_confirm.return_value = False

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_pull_request_number',
                   return_value=100):
            branch, pr = provider_config_manager._handle_pull_request_selection()

        assert branch == ""
        assert pr == 100

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_pull_request_selection_custom(
            self, mock_confirm, mock_prompt, provider_config_manager, mock_config_manager
    ):
        """Test PR selection with custom number."""
        mock_config_manager.get_value.return_value = None
        mock_prompt.return_value = "42"
        mock_confirm.return_value = False

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_pull_request_number',
                   return_value=42):
            branch, pr = provider_config_manager._handle_pull_request_selection()

        assert pr == 42

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    def test_handle_pull_request_selection_invalid(
            self, mock_prompt, provider_config_manager, mock_config_manager, mock_console
    ):
        """Test PR selection with invalid number."""
        mock_config_manager.get_value.return_value = None
        mock_prompt.return_value = "invalid"

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_pull_request_number',
                   side_effect=ValidationError("Invalid PR", "pull_request_number")):
            with pytest.raises(Exception):
                provider_config_manager._handle_pull_request_selection()

    @patch('devdox_ai_sonar.utils.provider_config.Prompt.ask')
    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_pull_request_selection_save_as_default(
            self, mock_confirm, mock_prompt, provider_config_manager, mock_config_manager
    ):
        """Test saving PR as default."""
        mock_config_manager.get_value.return_value = 100
        mock_prompt.return_value = "200"
        mock_confirm.return_value = True

        with patch('devdox_ai_sonar.utils.provider_config.InputValidator.validate_pull_request_number',
                   return_value=200):
            provider_config_manager._handle_pull_request_selection()

        mock_config_manager.set_value.assert_called_with("sonar.default_pull", 200)


# ============================================================================
# PRIVATE METHOD TESTS
# ============================================================================


class TestHandleApiKeyUpdate:
    """Tests for _handle_api_key_update method."""

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_api_key_update_keep_existing(
            self, mock_confirm, provider_config_manager, sample_provider_config
    ):
        """Test keeping existing API key."""
        mock_confirm.return_value = False
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        result = provider_config_manager._handle_api_key_update(ctx)

        assert result is True
        assert len(ctx.updates) == 0
        assert ctx.available_models == sample_provider_config["models"]

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    def test_handle_api_key_update_change_success(
            self, mock_prompt_key, mock_confirm, provider_config_manager,
            sample_provider_config, mock_validator
    ):
        """Test successfully changing API key."""
        mock_confirm.return_value = True
        mock_prompt_key.return_value = "new-api-key"
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        result = provider_config_manager._handle_api_key_update(ctx)

        assert result is True
        assert ctx.updates["api_key"] == "new-api-key"
        assert len(ctx.available_models) > 0

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    def test_handle_api_key_update_cancelled(
            self, mock_prompt_key, mock_confirm, provider_config_manager, sample_provider_config
    ):
        """Test cancelling API key update."""
        mock_confirm.return_value = True
        mock_prompt_key.return_value = None
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        result = provider_config_manager._handle_api_key_update(ctx)

        assert result is False


class TestHandleModelUpdate:
    """Tests for _handle_model_update method."""

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_handle_model_update_keep_current(
            self, mock_confirm, provider_config_manager, sample_provider_config
    ):
        """Test keeping current model."""
        mock_confirm.return_value = True
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        provider_config_manager._handle_model_update(ctx)

        assert "default_model" not in ctx.updates

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.select_provider_from_list')
    def test_handle_model_update_change_model(
            self, mock_select, mock_confirm, provider_config_manager, sample_provider_config
    ):
        """Test changing to new model."""
        mock_confirm.return_value = False
        mock_select.return_value = "gpt-3.5-turbo"
        ctx = ProviderUpdateContext(sample_provider_config, "openai")
        ctx.available_models = sample_provider_config["models"]

        provider_config_manager._handle_model_update(ctx)

        assert ctx.updates["default_model"] == "gpt-3.5-turbo"
        assert ctx.current_model == "gpt-3.5-turbo"

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.select_provider_from_list')
    def test_handle_model_update_selection_cancelled(
            self, mock_select, mock_confirm, provider_config_manager, sample_provider_config
    ):
        """Test cancelling model selection."""
        mock_confirm.return_value = False
        mock_select.return_value = None
        ctx = ProviderUpdateContext(sample_provider_config, "openai")
        ctx.available_models = ["model-1"]

        provider_config_manager._handle_model_update(ctx)

        assert "default_model" not in ctx.updates


class TestGetAvailableModels:
    """Tests for _get_available_models method."""

    def test_get_available_models_from_context(
            self, provider_config_manager, sample_provider_config
    ):
        """Test getting models from context cache."""
        ctx = ProviderUpdateContext(sample_provider_config, "openai")
        ctx.available_models = ["model-1", "model-2"]

        result = provider_config_manager._get_available_models(ctx)

        assert result == ["model-1", "model-2"]

    def test_get_available_models_fetch_from_api(
            self, provider_config_manager, sample_provider_config, mock_validator
    ):
        """Test fetching models from API."""
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        result = provider_config_manager._get_available_models(ctx)

        assert len(result) > 0
        mock_validator.validate.assert_called_once()

    def test_get_available_models_no_api_key(
            self, provider_config_manager
    ):
        """Test when provider has no API key."""
        provider = {"name": "openai", "models": []}
        ctx = ProviderUpdateContext(provider, "openai")

        result = provider_config_manager._get_available_models(ctx)

        assert result == []


class TestApplyProviderUpdates:
    """Tests for _apply_provider_updates method."""

    def test_apply_provider_updates_with_changes(
            self, provider_config_manager, mock_config_manager, sample_provider_config
    ):
        """Test applying updates when changes exist."""
        ctx = ProviderUpdateContext(sample_provider_config, "openai")
        ctx.updates = {"api_key": "new-key", "default_model": "new-model"}

        result = provider_config_manager._apply_provider_updates(ctx, True)

        assert result is True
        mock_config_manager.update_provider.assert_called_once()

    def test_apply_provider_updates_no_changes(
            self, provider_config_manager, mock_config_manager, sample_provider_config
    ):
        """Test when no updates to apply."""
        ctx = ProviderUpdateContext(sample_provider_config, "openai")

        result = provider_config_manager._apply_provider_updates(ctx, False)

        assert result is False
        mock_config_manager.update_provider.assert_not_called()


class TestSaveAsDefaultIfChanged:
    """Tests for _save_as_default_if_changed method."""

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_save_as_default_value_unchanged(
            self, mock_confirm, provider_config_manager, mock_config_manager
    ):
        """Test when value hasn't changed."""
        provider_config_manager._save_as_default_if_changed(
            "test.key", "value", "value", "test"
        )

        mock_confirm.assert_not_called()
        mock_config_manager.set_value.assert_not_called()

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_save_as_default_user_declines(
            self, mock_confirm, provider_config_manager, mock_config_manager
    ):
        """Test when user declines to save."""
        mock_confirm.return_value = False

        provider_config_manager._save_as_default_if_changed(
            "test.key", "new", "old", "test"
        )

        mock_config_manager.set_value.assert_not_called()

    @patch('devdox_ai_sonar.utils.provider_config.Confirm.ask')
    def test_save_as_default_user_accepts(
            self, mock_confirm, provider_config_manager, mock_config_manager, mock_console
    ):
        """Test when user accepts to save."""
        mock_confirm.return_value = True

        provider_config_manager._save_as_default_if_changed(
            "test.key", "new", "old", "test value"
        )

        mock_config_manager.set_value.assert_called_once_with("test.key", "new")
        mock_config_manager.save_config.assert_called_once()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_provider_name(self, provider_config_manager, mock_console):
        """Test with empty provider name."""
        result = provider_config_manager.configure_new_provider("")
        assert result is None

    def test_special_characters_in_api_key(self, mock_prompt):
        """Test API key with special characters."""
        special_key = "sk-!@#$%^&*()_+-=[]{}|;':\",./<>?"
        mock_prompt.return_value = special_key

        result = ProviderConfigUI.prompt_for_api_key("openai")

        assert result == special_key

    def test_very_long_api_key(self, mock_prompt):
        """Test with very long API key."""
        long_key = "a" * 1000
        mock_prompt.return_value = long_key

        result = ProviderConfigUI.prompt_for_api_key("openai")

        assert result == long_key

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    def test_validation_exception_handling(
            self, mock_prompt_key, provider_config_manager, mock_validator, mock_console
    ):
        """Test handling of validation exceptions."""
        mock_prompt_key.return_value = "test-key"
        mock_validator.validate.side_effect = Exception("Validation error")

        with pytest.raises(Exception):
            provider_config_manager.configure_new_provider("openai")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.prompt_for_api_key')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.select_provider_from_list')
    @patch('devdox_ai_sonar.utils.provider_config.ProviderConfigUI.confirm_default')
    def test_complete_provider_setup_workflow(
            self, mock_confirm, mock_select, mock_prompt_key,
            provider_config_manager, mock_validator
    ):
        """Test complete provider setup from start to finish."""
        # User provides API key
        mock_prompt_key.return_value = "test-api-key"

        # Validation succeeds with models
        mock_validator.validate.return_value.success = True
        mock_validator.validate.return_value.models = ["model-1", "model-2"]

        # User selects model
        mock_select.return_value = "model-1"

        # User sets as default
        mock_confirm.return_value = True

        result = provider_config_manager.configure_new_provider("openai")

        assert result is not None
        assert result["config"]["api_key"] == "test-api-key"
        assert result["config"]["default_model"] == "model-1"
        assert result["set_as_default"] is True


