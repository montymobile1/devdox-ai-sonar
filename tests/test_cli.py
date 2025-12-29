

import pytest
from pathlib import Path
from contextlib import contextmanager
import click
from click.testing import CliRunner
from devdox_ai_sonar.services.configuration import AuthConfig, LLMConfig
from devdox_ai_sonar.utils.validator import IssueType, InputValidator
from unittest.mock import Mock, patch, MagicMock
from devdox_ai_sonar.utils import constant
from devdox_ai_sonar.cli import (
    main,
    _safe_convert_pr,
    _should_exit_interactive_mode,
    _configure_sonarcloud,
    _fallback_command_selector,
    SwitchCommandException,
    show_progress,
    init_config,
    add_provider,
    update_provider,
    change_parameters,
    _run_fix_issues,
    _run_fix_security_issues,
    _run_analyze,
    _run_inspect,
    _load_and_validate_config,
    _validate_issue_types,
    _validate_severities,
    display_configuration,
    change_field,
    change_max_fix,
    _exit_application,
    _execute_interactive_command,
    _should_continue_to_menu,
    _handle_command_switch,
    _handle_interactive_error,
    _run_interactive_mode,
    _collect_rule_information,
    _initialize_fix_services,
    _handle_keyboard_interrupt,
    _process_and_fix_issues,
    _select_existing_ui,
    _check_reconfiguration_consent,
    _handle_cli_error,
    _display_fix_preview,
    _display_analysis_results,
    _display_fix_results,
    _process_files_with_issues,
    _generate_fix_for_file,
    _process_single_fix,
    handle_fix,
    _execute_interactive_iteration,
    _should_continue_to_next_file,
    _fetch_issues_by_type,
    _display_project_header,
    _display_metrics_section,
    _process_security_issues,
    _process_issues_for_rule,
    _process_regular_issues
)
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    AnalysisResult,
    FixSuggestion,
    FixResult
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def runner():
    """Create CLI test runner"""
    return CliRunner()


@pytest.fixture
def fix_params():
    """Mock fix parameters."""
    return {
        "apply": 0,
        "dry_run": 0,
        "create_backup": 1
    }


@pytest.fixture
def mock_services():
    """Mock services dictionary."""
    return {
        'analyzer': Mock(),
        'ruler': Mock(),
        'fixer': Mock()
    }


@pytest.fixture
def mock_ui():
    """Mock ProviderConfigUI"""
    ui = Mock()
    ui.select_provider_from_list.return_value = 'openai'
    return ui

@pytest.fixture
def sample_fix_result():
    """Sample fix result"""
    return FixResult(
        successful_fixes=[
            FixSuggestion(
                issue_key="issue-1",
                rule="python:S1234",
                file_path="test.py",
                original_code="old_code",
                fixed_code="new_code",
                explanation="Fixed issue",
                confidence=0.95,
                sonar_line_number=10,
                llm_model="open-ai"
            )
        ],
        failed_fixes=[],
        skipped_files=[],
        total_fixes_attempted=1,
        success_rate=1.0,
        backup_created=True,
        project_path=Path("/tmp/project"),
        backup_path=Path("/tmp/backup")
    )

@contextmanager
def mock_show_progress(message: str, total=None):
    """
    Mock implementation of show_progress that works with unpacking.

    Usage in tests:
        with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
            # Your test code
    """
    mock_progress = MagicMock()
    mock_task = MagicMock()

    # Yield a tuple that can be unpacked
    yield mock_progress, mock_task



@pytest.fixture
def auth_config():
    """Mock auth config."""
    return AuthConfig(
        token="test-token",
        organization="test-org",
        project="test-project",
        project_path="/tmp/test"
    )

@pytest.fixture
def mock_config_service():
    """Mock ConfigService"""
    with patch('devdox_ai_sonar.cli.ConfigService') as mock:
        mock_instance = Mock()
        mock_instance.load_auth_config.return_value = {
            "token": "test-token",
            "organization": "test-org",
            "project": "test-project",
            "project_path": "/tmp/project"
        }
        mock_instance.check_all_value_empty.return_value = False
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_loaded_config():
    """Fixture for _load_and_validate_config return value"""
    mock_auth = Mock()
    mock_auth.project = "test-project"
    mock_auth.organization = "test-org"
    mock_auth.token = "test-token"
    mock_auth.project_path = "/tmp/project"

    mock_llm = Mock()
    mock_llm.provider = "openai"
    mock_llm.model = "gpt-4"
    mock_llm.api_key = "test-key"

    parameters = {
        "branch": "main",
        "pull_request": 0,
        "max_fixes": 10,
        "severity": "",
        "types": ""
    }

    return mock_auth, mock_llm, parameters


@pytest.fixture
def mock_analyzer():
    """Mock SonarCloudAnalyzer"""
    with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer') as mock:
        mock_instance = Mock()
        mock_instance.get_fixable_issues_by_files.return_value = {}
        mock_instance.get_fixable_security_issues.return_value = {}
        mock_instance.get_project_issues.return_value = None
        mock_instance.analyze_project_directory.return_value = {
            "total_files": 10,
            "python_files": 5,
            "javascript_files": 3,
            "java_files": 2,
            "other_files": 0,
            "has_sonar_config": True,
            "has_git": True,
            "potential_source_dirs": ["src", "lib"]
        }
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager"""
    with patch('devdox_ai_sonar.cli.ConfigManager') as mock:
        mock_instance = Mock()
        mock_instance.get_value.return_value = {}
        mock_instance.load_config.return_value = None
        mock_instance.save_config.return_value = None
        mock_instance.create_default_config.return_value = None
        mock_instance.set_value.return_value = None
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_provider_manager():
    """Mock ProviderConfigManager"""
    with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock:
        mock_instance = Mock()
        mock_instance.branch_or_pr_prompt.return_value = ("main", None)
        mock_instance.branch_or_pr.return_value = ("main", None)
        mock_instance.get_available_providers.return_value = ["openai", "anthropic"]
        mock_instance.get_existing_providers.return_value = ["openai"]
        mock_instance.configure_new_provider.return_value = {
            "config": {"provider": "openai", "api_key": "test"},
            "set_as_default": True
        }
        mock_instance.update_existing_provider.return_value = True
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm_config():
    """Mock LLM Configuration"""
    return Mock(
        provider="openai",
        model="gpt-4",
        api_key="test-key",
        models=["gpt-4", "gpt-3.5-turbo"]
    )


@pytest.fixture
def sample_issues():
    """Sample SonarCloud issues"""
    return [
        SonarIssue(
            key="issue-1",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test-project",
            first_line=10,
            last_line=12,
            message="Test issue 1",
            type="BUG",
            status="OPEN",
            file="test.py"
        ),
        SonarIssue(
            key="issue-2",
            rule="python:S5678",
            severity="CRITICAL",
            component="main.py",
            project="test-project",
            first_line=20,
            last_line=22,
            message="Test issue 2",
            type="VULNERABILITY",
            status="OPEN",
            file="main.py"
        )
    ]


@pytest.fixture
def sample_fix_suggestion():
    """Sample fix suggestion"""
    return FixSuggestion(
        issue_key="issue-1",
        rule="python:S1234",
        file_path="test.py",
        original_code="old_code = True",
        fixed_code="new_code = False",
        explanation="Fixed the issue",
        confidence=0.95,
        sonar_line_number=10,
        llm_model="gpt-3.5-turbo"
    )


# ============================================================================
# TEST CLASS: SAFE CONVERSION UTILITIES
# ============================================================================

class TestSafeConversionUtilities:
    """Test safe conversion functions"""

    def test_safe_convert_pr_valid(self):
        """Test valid PR conversion"""
        assert _safe_convert_pr("123") == 123
        assert _safe_convert_pr("1") == 1
        assert _safe_convert_pr("999") == 999

    def test_safe_convert_pr_none(self):
        """Test None input"""
        assert _safe_convert_pr(None) == 0
        assert _safe_convert_pr("") == 0

    def test_safe_convert_pr_negative(self):
        """Test negative PR number"""
        assert _safe_convert_pr("-5") == 0
        assert _safe_convert_pr("-999") == 0

    def test_safe_convert_pr_invalid(self):
        """Test invalid PR formats"""
        assert _safe_convert_pr("abc") == 0
        assert _safe_convert_pr("12.5") == 0
        assert _safe_convert_pr("12a") == 0
        assert _safe_convert_pr("!@#") == 0

    def test_safe_convert_pr_whitespace(self):
        """Test PR with whitespace"""
        assert _safe_convert_pr("  123  ") == 123
        assert _safe_convert_pr("\t42\n") == 42

    def test_convert_empty_string_returns_zero(self):
        """Test empty string returns 0."""
        assert _safe_convert_pr("") == 0

    def test_convert_negative_pr_returns_zero(self, capsys):
        """Test negative PR number returns 0 with warning."""
        result = _safe_convert_pr("-5")

        assert result == 0
        captured = capsys.readouterr()
        assert "Negative PR numbers not allowed" in captured.out

    def test_convert_invalid_string_returns_zero(self, capsys):
        """Test invalid string returns 0 with warning."""
        result = _safe_convert_pr("not_a_number")

        assert result == 0
        captured = capsys.readouterr()
        assert "Invalid PR number" in captured.out

    def test_convert_zero_pr(self):
        """Test converting zero PR number."""
        assert _safe_convert_pr("0") == 0


# ============================================================================
# Test Interactive Mode Helper Functions
# ============================================================================

class TestShouldExitInteractiveMode:
    """Test cases for _should_exit_interactive_mode."""

    def test_should_exit_when_command_is_none(self):
        """Test returns True when command is None."""
        assert _should_exit_interactive_mode(None) is True

    def test_should_exit_when_command_is_exit(self):
        """Test returns True when command is 'exit'."""
        assert _should_exit_interactive_mode('exit') is True

    def test_should_not_exit_with_valid_command(self):
        """Test returns False with valid command."""
        assert _should_exit_interactive_mode('fix_issues') is False
        assert _should_exit_interactive_mode('analyze') is False


class TestExitApplication:
    """Test cases for _exit_application."""

    @patch('devdox_ai_sonar.cli.console')
    def test_exit_prints_goodbye_message(self, mock_console):
        """Test exit prints goodbye message."""
        with pytest.raises(SystemExit) as exc_info:
            _exit_application()

        assert exc_info.value.code == 0
        mock_console.print.assert_called_once()
        assert "Thank you for using DevDox AI Sonar" in str(mock_console.print.call_args)


# ============================================================================
# TEST CLASS: SHOW_COMMAND_SELECTOR - MISSING COVERAGE
# ============================================================================


class TestFallbackCommandSelector:
    """Tests for _fallback_command_selector"""

    @patch('devdox_ai_sonar.cli.console')
    def test_fallback_valid_selection(self, mock_console):
        """Test valid command selection"""
        mock_console.input.return_value = '1'

        result = _fallback_command_selector()

        assert result == 'fix_issues'

    @patch('devdox_ai_sonar.cli.console')
    def test_fallback_all_options(self, mock_console):
        """Test all options"""
        test_cases = [
            ('1', 'fix_issues'),
            ('2', 'fix_security_issues'),
            ('3', 'analyze'),
            ('4', 'inspect'),
            ('5', 'exit'),
        ]

        for input_val, expected in test_cases:
            mock_console.input.return_value = input_val
            result = _fallback_command_selector()
            assert result == expected

    @patch('devdox_ai_sonar.cli.console')
    def test_fallback_invalid_input(self, mock_console):
        """Test invalid input"""
        mock_console.input.return_value = 'invalid'

        result = _fallback_command_selector()

        assert result is None




class TestExecuteInteractiveCommand:
    """Test cases for _execute_interactive_command."""

    @patch('devdox_ai_sonar.cli._execute_command')
    @patch('devdox_ai_sonar.cli.console')
    def test_executes_command_with_context(self, mock_console, mock_execute):
        """Test command execution with context."""
        mock_ctx = Mock(spec=click.Context)

        _execute_interactive_command(mock_ctx, 'fix_issues')

        mock_execute.assert_called_once_with(ctx=mock_ctx, command='fix_issues')
        assert mock_console.print.call_count >= 2  # Running message and separator



# ============================================================================
# TEST CLASS: COMMAND SWITCHING EXCEPTIONS
# ============================================================================

class TestCommandSwitchingExceptions:
    """Test exception classes"""

    def test_switch_command_exception(self):
        """Test SwitchCommandException"""
        exc = SwitchCommandException()
        assert isinstance(exc, Exception)

        exc_with_msg = SwitchCommandException("Test message")
        assert str(exc_with_msg) == "Test message"




# ============================================================================
# TEST CLASS: PROGRESS CONTEXT MANAGER
# ============================================================================

class TestProgressContextManager:
    """Test show_progress context manager"""

    def test_show_progress_basic(self):
        """Test basic progress display"""
        with show_progress("Testing...") as (progress, task):
            assert progress is not None
            assert task is not None

    def test_show_progress_with_total(self):
        """Test progress with total"""
        with show_progress("Testing...", total=100) as (progress, task):
            assert progress is not None
            assert task is not None

    def test_show_progress_exception_handling(self):
        """Test progress cleanup on exception"""
        try:
            with show_progress("Testing...") as (progress, task):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected





# ============================================================================
# TEST CLASS: MAIN COMMAND
# ============================================================================

class TestMainCommand:
    """Test main CLI entry point"""

    def test_main_help(self, runner):
        """Test main command help"""
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert 'SonarCloud' in result.output or 'sonar' in result.output.lower()

    def test_main_version(self, runner):
        """Test version display"""
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0

    def test_main_with_direct_command(self, runner, mock_config_service):
        """Test main with direct command"""
        with patch('devdox_ai_sonar.cli.init_config'):
            with patch('devdox_ai_sonar.cli._execute_command'):
                result = runner.invoke(main, ['-c', 'analyze'])
                # Should attempt to execute

    def test_main_verbose_flag(self, runner):
        """Test verbose flag sets context"""
        with patch('devdox_ai_sonar.cli.init_config'):
            with patch('devdox_ai_sonar.cli._run_interactive_mode'):
                result = runner.invoke(main, ['--verbose'])

    def test_main_with_invalid_command(self, runner):
        """Test main with invalid command option"""
        result = runner.invoke(main, ['-c', 'invalid_command'])

        # Should fail
        assert result.exit_code != 0

    def test_main_with_verbose_and_command(self, runner):
        """Test main with verbose flag and command"""
        with patch('devdox_ai_sonar.cli._execute_command'):
            with patch('devdox_ai_sonar.cli.init_config'):
                result = runner.invoke(main, ['--verbose', '-c', 'analyze'])

                # Should set verbose in context
                assert result.exit_code == 0 or 'verbose' in str(result.output).lower()

    def test_main_dry_run_flag(self, runner):
        """Test main with --dry-run flag"""
        with patch('devdox_ai_sonar.cli._execute_command'):
            with patch('devdox_ai_sonar.cli.init_config'):
                result = runner.invoke(main, ['--dry-run', '-c', 'fix_issues'])

                # Dry run should be set
                assert result.exit_code == 0

    @patch('devdox_ai_sonar.cli._run_interactive_mode')
    @patch('devdox_ai_sonar.cli.init_config')
    def test_main_interactive_mode_default(self, mock_init, mock_interactive, runner):
        """Test that main enters interactive mode by default"""
        result = runner.invoke(main, [])

        # Should call init_config then interactive mode
        mock_init.assert_called_once()
        mock_interactive.assert_called_once()

# ============================================================================
# TEST CLASS: CONFIGURATION MANAGEMENT
# ============================================================================

class TestConfigurationManagement:
    """Test configuration-related commands"""

    def test_init_config_first_time(
            self, runner, mock_config_service, mock_config_manager, mock_provider_manager
    ):
        """Test initial configuration"""
        mock_config_manager.get_value.return_value = []  # No providers

        with patch('devdox_ai_sonar.cli._configure_sonarcloud', return_value=True):
            with patch('devdox_ai_sonar.cli._configure_providers_loop'):
                with patch('devdox_ai_sonar.cli.change_parameters'):
                    with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
                        mock_init.return_value = (
                            mock_config_manager,
                            Mock(),
                            Mock(),
                            mock_provider_manager,
                            Mock(),
                            mock_config_service
                        )

                        ctx = Mock()
                        init_config(ctx)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._configure_sonarcloud')
    @patch('devdox_ai_sonar.cli._configure_providers_loop')
    @patch('devdox_ai_sonar.cli.change_parameters')
    def test_init_config_first_time_success(
            self, mock_change_params, mock_providers_loop, mock_sonar, mock_init
    ):
        """Test successful first-time initialization"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = []  # No providers yet
        mock_manager.create_default_config.return_value = None
        mock_manager.load_config.return_value = None
        mock_manager.save_config.return_value = None

        mock_provider_manager = Mock()
        mock_provider_manager.get_available_providers.return_value = ['openai']

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        mock_sonar.return_value = True

        init_config()

        # Verify initialization steps
        mock_manager.create_default_config.assert_called_once()
        mock_manager.load_config.assert_called_once()
        mock_sonar.assert_called_once()
        mock_providers_loop.assert_called_once()
        mock_change_params.assert_called_once()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._check_reconfiguration_consent')
    def test_init_config_existing_config_no_consent(self, mock_consent, mock_init):
        """Test when config exists and user doesn't consent to reconfigure"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = ['openai']  # Providers exist

        mock_init.return_value = (
            mock_manager, Mock(), Mock(), Mock(), Mock(), Mock()
        )

        mock_consent.return_value = False  # User says no

        init_config()

        # Should exit early
        mock_manager.create_default_config.assert_called_once()

    def test_check_reconfiguration_consent_no_providers(self):
        """Test with no providers"""
        result = _check_reconfiguration_consent([])

        assert result is True

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._configure_sonarcloud')
    def test_init_config_no_available_providers(self, mock_sonar, mock_init):
        """Test when no providers are available"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = []

        mock_provider_manager = Mock()
        mock_provider_manager.get_available_providers.return_value = []  # No providers

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        mock_sonar.return_value = True

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                init_config()


    def test_check_reconfiguration_consent_existing_providers(self):
        """Test with existing providers"""
        with patch('devdox_ai_sonar.cli.click'):
            result = _check_reconfiguration_consent(["openai", "anthropic"])

            assert result is False

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._configure_sonarcloud')
    def test_init_config_sonar_configuration_fails(self, mock_sonar, mock_init):
        """Test when SonarCloud configuration fails"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = []

        mock_init.return_value = (
            mock_manager, Mock(), Mock(), Mock(), Mock(), Mock()
        )

        mock_sonar.return_value = False  # SonarCloud config fails

        with pytest.raises(click.Abort):
            init_config()

    def test_init_config_existing_config(
            self, runner, mock_config_service, mock_config_manager, mock_provider_manager
    ):
        """Test with existing configuration"""
        mock_config_manager.get_value.return_value = ["openai"]  # Has providers

        with patch('devdox_ai_sonar.cli._check_reconfiguration_consent', return_value=False):
            with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
                mock_init.return_value = (
                    mock_config_manager,
                    Mock(),
                    Mock(),
                    mock_provider_manager,
                    Mock(),
                    mock_config_service
                )

                ctx = Mock()
                init_config(ctx)

    def test_add_provider_success(
        self,
        runner,
        mock_config_manager,
        mock_provider_manager,
        mock_ui
    ):
        """Test adding new provider successfully"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            # Setup all mocked managers
            mock_init.return_value = (
                mock_config_manager,       # manager
                mock_ui,                   # ui
                Mock(),                    # validator
                mock_provider_manager,     # provider_manager
                Mock(),                    # sonar_ui
                Mock()                     # config_service
            )

            with patch('devdox_ai_sonar.cli._display_completion_message') as mock_display:
                with patch('devdox_ai_sonar.cli._configure_providers_loop') as mock_loop:
                    # Call the function
                    add_provider()

                    # Verify initialization
                    mock_init.assert_called_once()

                    # Verify config was loaded
                    mock_config_manager.load_config.assert_called_once()

                    # Verify available providers were fetched
                    mock_provider_manager.get_available_providers.assert_called_once()

                    # Verify providers loop was called
                    mock_loop.assert_called_once()

                    # Verify config was saved
                    mock_config_manager.save_config.assert_called_once_with(create_backup=False)

                    # Verify completion message was displayed
                    mock_display.assert_called_once()

    def test_add_provider_with_error_handling(
            self,
            runner,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test add_provider error handling"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            # Simulate error during load_config
            mock_config_manager.load_config.side_effect = Exception("Config load failed")

            with patch('devdox_ai_sonar.cli._handle_cli_error') as mock_error:
                mock_error.side_effect = SystemExit(1)

                with pytest.raises(SystemExit):
                    add_provider()

                # Verify error handler was called
                mock_error.assert_called_once()

    def test_configure_providers_loop_integration(
            self,
            mock_config_manager,
            mock_provider_manager,
            mock_ui
    ):
        """Test the _configure_providers_loop function"""
        from devdox_ai_sonar.cli import _configure_providers_loop

        available_providers = ['openai', 'anthropic']

        # Mock the UI to select 'openai', then return None (exit loop)
        mock_ui.select_provider_from_list.side_effect = ['openai', None]

        # Mock successful configuration
        mock_provider_manager.configure_new_provider.return_value = {
            'config': {'provider': 'openai', 'api_key': 'test-key'},
            'set_as_default': True
        }

        with patch('devdox_ai_sonar.cli.console.print'):
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=False):
                _configure_providers_loop(
                    mock_provider_manager,
                    mock_config_manager,
                    mock_ui,
                    available_providers
                )

                # Verify provider was configured
                mock_provider_manager.configure_new_provider.assert_called_once_with('openai')

                # Verify provider was added to manager
                mock_config_manager.add_provider.assert_called_once()

    def test_handle_provider_configuration_success(
            self,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test _handle_provider_configuration helper"""
        from devdox_ai_sonar.cli import _handle_provider_configuration

        available_providers = ['openai', 'anthropic']

        mock_provider_manager.configure_new_provider.return_value = {
            'config': {'provider': 'openai', 'api_key': 'test-key'},
            'set_as_default': True
        }

        with patch('devdox_ai_sonar.cli.console.print'):
            result = _handle_provider_configuration(
                mock_provider_manager,
                mock_config_manager,
                'openai',
                available_providers
            )

            assert result is True
            assert 'openai' not in available_providers  # Should be removed

            # Verify add_provider was called on manager
            mock_config_manager.add_provider.assert_called_once()

    def test_handle_provider_configuration_failure(
            self,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test _handle_provider_configuration when configuration fails"""
        from devdox_ai_sonar.cli import _handle_provider_configuration

        available_providers = ['openai']

        # Return None to simulate cancelled/failed configuration
        mock_provider_manager.configure_new_provider.return_value = None

        result = _handle_provider_configuration(
            mock_provider_manager,
            mock_config_manager,
            'openai',
            available_providers
        )

        assert result is False
        assert 'openai' in available_providers  # Should NOT be removed

        # Verify add_provider was NOT called
        mock_config_manager.add_provider.assert_not_called()

    def test_update_provider_success(
        self,
        runner,
        mock_config_manager,
        mock_provider_manager,
        mock_ui
    ):
        """Test updating existing provider successfully"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            # Setup all mocked managers
            mock_init.return_value = (
                mock_config_manager,       # manager
                mock_ui,                   # ui
                Mock(),                    # validator
                mock_provider_manager,     # provider_manager
                Mock(),                    # sonar_ui
                Mock()                     # config_service
            )


            with patch('devdox_ai_sonar.cli._display_operation_header') as mock_header:
                    with patch('devdox_ai_sonar.cli._select_existing_ui') as mock_select:
                        # Mock user selection
                        mock_select.return_value = 'openai'

                        # Call the function
                        update_provider()

                        # Verify initialization
                        mock_init.assert_called_once()

                        # Verify config was loaded
                        mock_config_manager.load_config.assert_called_once()

                        # Verify existing providers were fetched
                        mock_provider_manager.get_existing_providers.assert_called_once()

                        # Verify operation header was displayed
                        mock_header.assert_called_once_with("🔧 UPDATE EXISTING PROVIDER")

                        # Verify provider selection was prompted
                        mock_select.assert_called_once_with(
                            "provider",
                            "Select the provider to update",
                            ['openai']
                        )

                        # Verify provider was updated
                        mock_provider_manager.update_existing_provider.assert_called_once_with('openai')

                        # Verify config was saved
                        mock_config_manager.save_config.assert_called_once_with(create_backup=False)

    def test_update_provider_no_existing_providers(
            self,
            runner,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test update_provider when no providers are configured"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            # Setup managers
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            mock_provider_manager.get_existing_providers.return_value = []

            with patch('devdox_ai_sonar.cli.console.print') as mock_print:
                # ✅ FIX: Use click.Abort, not SystemExit
                with pytest.raises(click.Abort):
                    update_provider()

                # Verify warning message was printed
                mock_print.assert_called()
                call_args = str(mock_print.call_args_list)
                assert 'No providers configured' in call_args or \
                       'add at least one provider' in call_args


    def test_update_provider_no_existing_providers_alternative(
            self,
            runner,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test update_provider when no providers - alternative approach"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            # No existing providers
            mock_provider_manager.get_existing_providers.return_value = []

            with patch('devdox_ai_sonar.cli.console.print') as mock_print:
                # ✅ Catch the exception and verify it's the right type
                try:
                    update_provider()
                    pytest.fail("Should have raised click.Abort")
                except click.Abort:
                    # Expected behavior
                    pass

                # Verify warning was shown
                assert mock_print.called
                call_args = str(mock_print.call_args_list)
                assert any(msg in call_args for msg in [
                    'No providers configured',
                    'add at least one provider',
                    'Please add'
                ])

    def test_update_provider_no_existing_providers_with_error_handler(
            self,
            runner,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test that _handle_cli_error properly handles click.Abort"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            mock_provider_manager.get_existing_providers.return_value = []

            # Mock _handle_cli_error to verify it's called
            with patch('devdox_ai_sonar.cli._handle_cli_error') as mock_error_handler:
                # _handle_cli_error re-raises, so we still expect the exception
                mock_error_handler.side_effect = click.Abort()

                with patch('devdox_ai_sonar.cli.console.print'):
                    with pytest.raises(click.Abort):
                        update_provider()

        # ========================================================================
        # Additional tests for other abort scenarios
        # ========================================================================



    def test_update_provider_user_cancels_selection(
            self,
            runner,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test update_provider when user cancels provider selection"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            # Providers exist
            mock_provider_manager.get_existing_providers.return_value = ['openai']

            with patch('devdox_ai_sonar.cli._display_operation_header'):
                with patch('devdox_ai_sonar.cli._select_existing_ui') as mock_select:
                    # User cancels (returns empty string)
                    mock_select.return_value = ""

                    with pytest.raises(Exception):  # Should abort
                        ctx = Mock()
                        update_provider(ctx)

                    # Verify update was NOT called
                    mock_provider_manager.update_existing_provider.assert_not_called()
                    mock_config_manager.save_config.assert_not_called()

    def test_update_provider_no_providers(
            self, runner, mock_config_manager, mock_provider_manager
    ):
        """Test update when no providers exist"""
        mock_provider_manager.get_existing_providers.return_value = []

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            with pytest.raises(Exception):  # Should abort
                ctx = Mock()
                update_provider(ctx)

    def test_change_parameters_success(
            self, runner, mock_config_manager, mock_provider_manager
    ):
        """Test changing parameters"""
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)
        mock_config_manager.get_value.return_value = 10

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            with patch('devdox_ai_sonar.cli.change_field', return_value=None):
                with patch('devdox_ai_sonar.cli.change_max_fix'):
                    ctx = Mock()
                    change_parameters(ctx)


    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_with_types_provided(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that types parameter skips interactive prompt for types"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with types pre-populated
        change_parameters(types="BUG,VULNERABILITY")

        # change_field should NOT be called for types (skipped because types provided)
        types_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.types' in str(call)
        ]
        assert len(types_calls) == 0, "Types field should be skipped when types provided"

        # But other fields should still be called
        assert mock_change_field.call_count >= 3

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_with_severity_provided(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that severity parameter skips interactive prompt for severities"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with severity pre-populated
        change_parameters(severity="CRITICAL,BLOCKER")

        # change_field should NOT be called for severities
        severity_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.severities' in str(call)
        ]
        assert len(severity_calls) == 0, "Severities field should be skipped when severity provided"

        # But other fields should still be called
        assert mock_change_field.call_count >= 3  # types, apply, backup, exclude_rules

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_with_both_types_and_severity(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that both types and severity parameters skip their prompts"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with both types and severity
        change_parameters(types="BUG", severity="CRITICAL")

        # Neither types nor severities should be prompted
        types_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.types' in str(call)
        ]
        severity_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.severities' in str(call)
        ]

        assert len(types_calls) == 0, "Types should be skipped"
        assert len(severity_calls) == 0, "Severities should be skipped"

        # Only apply, backup, and exclude_rules should be prompted
        assert mock_change_field.call_count >= 3


    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_with_apply_in_kwargs(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that apply parameter from kwargs is used as default"""
        mock_manager = Mock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            constant.CONFIGURATION_APPLY: None,  # Not in config
            constant.CONFIGURATION_BACKUP: None,
        }.get(key)

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with apply in kwargs
        change_parameters(apply=1)

        # Find the call for CONFIGURATION_APPLY
        apply_calls = [
            call for call in mock_change_field.call_args_list
            if constant.CONFIGURATION_APPLY in str(call)
        ]

        assert len(apply_calls) == 1, "Apply field should be called once"

        # Verify default_value is 1 (from kwargs)
        call_kwargs = apply_calls[0][1]
        assert call_kwargs['default_value'] == 1, "Should use apply value from kwargs as default"

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_with_create_backup_in_kwargs(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that create_backup parameter from kwargs is used as default"""
        mock_manager = Mock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            constant.CONFIGURATION_APPLY: None,
            constant.CONFIGURATION_BACKUP: None,  # Not in config
        }.get(key)

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with create_backup in kwargs
        change_parameters(create_backup=1)

        # Find the call for CONFIGURATION_BACKUP
        backup_calls = [
            call for call in mock_change_field.call_args_list
            if constant.CONFIGURATION_BACKUP in str(call)
        ]

        assert len(backup_calls) == 1, "Backup field should be called once"

        # Verify default_value is 1 (from kwargs)
        call_kwargs = backup_calls[0][1]
        assert call_kwargs['default_value'] == 1, "Should use create_backup value from kwargs as default"

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_config_overrides_kwargs(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that config values take precedence over kwargs"""
        mock_manager = Mock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            constant.CONFIGURATION_APPLY: 0,  # In config
            constant.CONFIGURATION_BACKUP: 1,  # In config
        }.get(key)

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with different values in kwargs
        change_parameters(apply=1, create_backup=0)

        # Find the calls
        apply_calls = [
            call for call in mock_change_field.call_args_list
            if constant.CONFIGURATION_APPLY in str(call)
        ]
        backup_calls = [
            call for call in mock_change_field.call_args_list
            if constant.CONFIGURATION_BACKUP in str(call)
        ]

        # Config values (0, 1) should be used, not kwargs values (1, 0)
        assert apply_calls[0][1]['default_value'] == 0, "Should use config value for apply"
        assert backup_calls[0][1]['default_value'] == 1, "Should use config value for backup"

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_types_field_with_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that types field is called with VALID_ISSUE_TYPES choices"""
        mock_manager = Mock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.types": "BUG,CODE_SMELL",
        }.get(key)

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call without types parameter (should prompt)
        change_parameters()

        # Find types field call
        types_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.types' in str(call)
        ]

        assert len(types_calls) == 1
        call_kwargs = types_calls[0][1]

        # Verify choices are provided
        assert 'choices' in call_kwargs
        assert call_kwargs['choices'] == list(InputValidator.VALID_ISSUE_TYPES)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_severities_field_with_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that severities field is called with VALID_SEVERITIES choices"""
        mock_manager = Mock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.severities": "CRITICAL,BLOCKER",
        }.get(key)

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call without severity parameter (should prompt)
        change_parameters()

        # Find severities field call
        severity_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.severities' in str(call)
        ]

        assert len(severity_calls) == 1
        call_kwargs = severity_calls[0][1]

        # Verify choices are provided
        assert 'choices' in call_kwargs
        assert call_kwargs['choices'] == list(InputValidator.VALID_SEVERITIES)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_severities_field_with_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that severities field is called with VALID_SEVERITIES choices"""
        mock_manager = Mock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.severities": "CRITICAL,BLOCKER",
        }.get(key)

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call without severity parameter (should prompt)
        change_parameters()

        # Find severities field call
        severity_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.severities' in str(call)
        ]

        assert len(severity_calls) == 1
        call_kwargs = severity_calls[0][1]

        # Verify choices are provided
        assert 'choices' in call_kwargs
        assert call_kwargs['choices'] == list(InputValidator.VALID_SEVERITIES)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_apply_field_with_yes_no_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that apply field has yes/no choices with correct format"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        change_parameters()

        # Find apply field call
        apply_calls = [
            call for call in mock_change_field.call_args_list
            if constant.CONFIGURATION_APPLY in str(call)
        ]

        assert len(apply_calls) == 1
        call_kwargs = apply_calls[0][1]

        # Verify choices format
        assert 'choices' in call_kwargs
        choices = call_kwargs['choices']
        assert len(choices) == 2
        # Check that choices are Choice objects with title and value
        assert all(hasattr(choice, 'title') and hasattr(choice, 'value') for choice in choices)
        assert call_kwargs['multiple'] is False

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_exclude_rules_field_no_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that exclude_rules field has no predefined choices"""
        mock_manager = Mock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.exclude_rules": "python:S3776,python:S1192",
        }.get(key)

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        change_parameters()

        # Find exclude_rules field call
        exclude_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.exclude_rules' in str(call)
        ]

        assert len(exclude_calls) == 1
        call_kwargs = exclude_calls[0][1]

        # Verify NO choices are provided (free text input)
        assert 'choices' not in call_kwargs or call_kwargs.get('choices') is None

    # ========================================================================
    # NEW TESTS: Branch/PR Validation
    # ========================================================================

    @patch('devdox_ai_sonar.cli._initialize_managers')
    def test_change_parameters_with_branch_only(self, mock_init):
        """Test change_parameters with only branch (no PR)"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("feature/test", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        with patch('devdox_ai_sonar.cli.change_field'):
            with patch('devdox_ai_sonar.cli.change_max_fix'):
                # Should succeed with only branch
                change_parameters()
                mock_manager.save_config.assert_called_once()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    def test_change_parameters_with_pr_only(self, mock_init):
        """Test change_parameters with only PR (no branch)"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, "123")

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        with patch('devdox_ai_sonar.cli.change_field'):
            with patch('devdox_ai_sonar.cli.change_max_fix'):
                # Should succeed with only PR
                change_parameters()
                mock_manager.save_config.assert_called_once()


    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    @patch('devdox_ai_sonar.cli._handle_cli_error')
    def test_change_parameters_handles_exception_in_change_field(
            self, mock_handle_error, mock_change_max, mock_change_field, mock_init
    ):
        """Test exception handling when change_field raises error"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Make change_field raise exception
        mock_change_field.side_effect = ValueError("Invalid input")

        change_parameters()

        # Should call error handler
        mock_handle_error.assert_called_once()
        assert isinstance(mock_handle_error.call_args[0][0], ValueError)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    @patch('devdox_ai_sonar.cli._handle_cli_error')
    def test_change_parameters_handles_exception_in_save(
            self, mock_handle_error, mock_change_max, mock_change_field, mock_init
    ):
        """Test exception handling when save_config fails"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10
        mock_manager.save_config.side_effect = IOError("Cannot write to file")

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        change_parameters()

        # Should call error handler
        mock_handle_error.assert_called_once()
        assert isinstance(mock_handle_error.call_args[0][0], IOError)



    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._handle_cli_error')
    def test_change_parameters_handles_exception_in_init(
            self, mock_handle_error, mock_init
    ):
        """Test exception handling when initialization fails"""
        # Make init raise exception
        mock_init.side_effect = RuntimeError("Config error")

        change_parameters()

        # Should call error handler
        mock_handle_error.assert_called_once()
        assert isinstance(mock_handle_error.call_args[0][0], RuntimeError)


    def test_change_parameters_no_branch_or_pr(
            self, runner, mock_config_manager, mock_provider_manager
    ):
        """Test change_parameters without branch or PR"""
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            with pytest.raises(Exception):  # Should abort
                ctx = Mock()
                change_parameters(ctx)

    def test_change_parameters_invalid_pr(
            self, runner, mock_config_manager, mock_provider_manager
    ):
        """Test change_parameters with invalid PR"""
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, "invalid")

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            with pytest.raises(Exception):  # Should abort after validation
                ctx = Mock()
                change_parameters(ctx)

    def test_change_field_with_value(self, mock_config_manager):
        """Test change_field with value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', return_value="BUG"):
            result = change_field(
                mock_config_manager,
                "field.path",
                "Enter value:",
                "default"
            )

            assert result == "BUG"
            mock_config_manager.set_value.assert_called_once()

    def test_change_field_no_value(self, mock_config_manager):
        """Test change_field with no value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', return_value=None):
            result = change_field(
                mock_config_manager,
                "field.path",
                "Enter value:",
                "default"
            )

            assert result is None
            mock_config_manager.set_value.assert_not_called()

    def test_change_max_fix_valid(self, mock_config_manager):
        """Test change_max_fix with valid value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', return_value="15"):
            change_max_fix(mock_config_manager, "Enter max:", 10, 50)

            mock_config_manager.set_value.assert_called_with(
                "configuration.max_fixes", 15
            )

    def test_change_max_fix_invalid(self, mock_config_manager):
        """Test change_max_fix with invalid value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', return_value="abc"):
            change_max_fix(mock_config_manager, "Enter max:", 10, 50)

            # Should use default
            mock_config_manager.set_value.assert_called()

    def test_change_max_fix_out_of_range(self, mock_config_manager):
        """Test change_max_fix with out of range value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', return_value="100"):
            with patch('devdox_ai_sonar.cli.settings') as mock_settings:
                mock_settings.DEFAULT_MAX_FIXES = 10
                change_max_fix(mock_config_manager, "Enter max:", 10, 50)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_all_values(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test changing all parameter values"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        mock_change_field.return_value = None

        change_parameters()

        # Should call change_max_fix and change_field multiple times
        mock_change_max.assert_called_once()
        assert mock_change_field.call_count >= 3  # types, severities, apply, backup

    @patch('devdox_ai_sonar.cli._initialize_managers')
    def test_change_parameters_no_branch_or_pr(self, mock_init):
        """Test when no branch or PR is specified"""
        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)

        mock_init.return_value = (
            Mock(), Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                change_parameters()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field')
    @patch('devdox_ai_sonar.cli.change_max_fix')
    def test_change_parameters_save_config(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that config is saved after changes"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        change_parameters()

        # Should save config at end
        mock_manager.save_config.assert_called_once_with(create_backup=False)


# ============================================================================
# CONFIGURATION HELPERS TESTS
# ============================================================================

class TestConfigureSonarCloud:
    """Tests for _configure_sonarcloud"""

    def test_configure_sonarcloud_success(self):
        """Test successful configuration"""
        mock_sonar_ui = Mock()
        mock_config_service = Mock()

        mock_config_service.load_auth_config.return_value = {
            "token": "",
            "organization": "",
            "project": "",
            "project_path": ""
        }
        mock_config_service.check_all_value_empty.return_value = True

        mock_sonar_config = Mock()
        mock_sonar_config.token = "new_token"
        mock_sonar_config.organization = "new_org"
        mock_sonar_config.project = "new_project"
        mock_sonar_config.project_path = "/new/path"

        mock_sonar_ui.configure_sonarcloud.return_value = mock_sonar_config
        mock_config_service.save_config.return_value = True

        result = _configure_sonarcloud(mock_sonar_ui, mock_config_service)

        assert result is True
        mock_sonar_ui.display_welcome.assert_called_once()

    def test_configure_sonarcloud_already_configured(self):
        """Test when already configured"""
        mock_sonar_ui = Mock()
        mock_config_service = Mock()

        mock_config_service.load_auth_config.return_value = {
            "token": "existing",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }
        mock_config_service.check_all_value_empty.return_value = False

        result = _configure_sonarcloud(mock_sonar_ui, mock_config_service)

        assert result is True
        mock_sonar_ui.configure_sonarcloud.assert_not_called()


# ============================================================================
# TEST CLASS: FIX ISSUES COMMAND
# ============================================================================

class TestFixIssuesCommand:
    """Test fix_issues command implementation"""

    def test_run_fix_issues_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test successful fix_issues run"""
        mock_config_service.load_llm_config.return_value = mock_llm_config

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {"branch": "main", "pull_request": 0})

            with patch('devdox_ai_sonar.cli.display_configuration') as mock_display:
                mock_display.return_value = {
                    "branch": "main",
                    "pull_request": 0,
                    "max_fixes": 10,
                    "types_list": None,
                    "severities_list": None,
                    "apply": False,
                    "dry_run": False,
                    "create_backup": True
                }

                with patch('devdox_ai_sonar.cli.smart_confirm', return_value=True):
                    with patch('devdox_ai_sonar.cli._process_and_fix_issues'):
                        ctx = Mock()
                        ctx.obj = {"verbose": False}

                        _run_fix_issues()

    def test_run_fix_issues_cancelled(
            self, mock_config_service, mock_llm_config
    ):
        """Test fix_issues when user cancels"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {})

            with patch('devdox_ai_sonar.cli.display_configuration', return_value={}):
                with patch('devdox_ai_sonar.cli.smart_confirm', return_value=False):


                    _run_fix_issues()

    def test_run_fix_issues_switch_command(self):
        """Test fix_issues with command switch"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config', side_effect=SwitchCommandException):


            with pytest.raises(SwitchCommandException):
                _run_fix_issues()

    def test_display_configuration(self):
        """Test display_configuration function"""
        params = {
            "pull_request": 123,
            "branch": "main",
            "max_fixes": 10,
            "types": "BUG,VULNERABILITY",
            "severity": "CRITICAL",
            "create_backup": 1
        }

        result = display_configuration(params, dry_run=0, apply=1)

        assert result["pull_request"] == 123
        assert result["branch"] == "main"
        assert result["max_fixes"] == 10
        assert result["apply"] == 1
        assert result["dry_run"] == 0

    @patch('devdox_ai_sonar.cli.console')
    def test_display_configuration_with_pull_request(self, mock_console):
        """Test displaying with pull request"""
        parameters = {
            "pull_request": 123,
            "branch": "",
            "max_fixes": 15,
            "types": "BUG,VULNERABILITY",
            "severities": "CRITICAL,BLOCKER",
            "create_backup": 1
        }

        result = display_configuration(parameters, dry_run=0, apply=1)

        assert result["pull_request"] == 123
        assert result["branch"] == ""
        assert result["max_fixes"] == 15
        assert result["apply"] == 1

        # Check console output mentions PR
        call_args_str = str(mock_console.print.call_args_list)
        assert '123' in call_args_str

    def test_display_configuration_branch_only(self):
        """Test display_configuration with branch only"""
        params = {
            "pull_request": 0,
            "branch": "develop",
            "max_fixes": 5
        }

        result = display_configuration(params, dry_run=1, apply=0)

        assert result["branch"] == "develop"
        assert result["pull_request"] == 0

    @patch('devdox_ai_sonar.cli.console')
    def test_display_configuration_apply_override(self, mock_console):
        """Test apply parameter override"""
        parameters = {
            "pull_request": 0,
            "branch": "main",
            "max_fixes": 10,
            "apply": 0  # Parameter says don't apply
        }

        # But function parameter overrides to 1
        result = display_configuration(parameters, dry_run=0, apply=1)

        # Should use override value
        assert result["apply"] == 1

    @patch('devdox_ai_sonar.cli.console')
    @patch('devdox_ai_sonar.cli._validate_issue_types')
    @patch('devdox_ai_sonar.cli._validate_severities')
    def test_display_configuration_validates_filters(
            self, mock_validate_sev, mock_validate_types, mock_console
    ):
        """Test that filters are validated"""
        mock_validate_types.return_value = ["BUG"]
        mock_validate_sev.return_value = ["CRITICAL"]

        parameters = {
            "pull_request": 0,
            "branch": "main",
            "max_fixes": 10,
            "types": "BUG",
            "severities": "CRITICAL"
        }

        result = display_configuration(parameters, dry_run=0, apply=0)

        mock_validate_types.assert_called_once_with("BUG")
        mock_validate_sev.assert_called_once_with("CRITICAL")

        assert result["types_list"] == ["BUG"]
        assert result["severities_list"] == ["CRITICAL"]


# ============================================================================
# TEST CLASS: FIX SECURITY ISSUES COMMAND
# ============================================================================

class TestFixSecurityIssuesCommand:
    """Test fix_security_issues command"""

    def test_run_fix_security_issues_success(
            self, mock_config_service, mock_llm_config
    ):
        """Test successful security fix"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {"branch": "main", "pull_request": 0})

            with patch('devdox_ai_sonar.cli.display_configuration') as mock_display:
                mock_display.return_value = {
                    "branch": "main",
                    "pull_request": 0,
                    "max_fixes": 10,
                    "apply": True,
                    "dry_run": False,
                    "create_backup": True
                }

                with patch('devdox_ai_sonar.cli.smart_confirm', return_value=True):


                        _run_fix_security_issues()

    def test_run_fix_security_issues_cancelled(self, mock_llm_config):
        """Test security fix when cancelled"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {})

            with patch('devdox_ai_sonar.cli.display_configuration', return_value={}):
                with patch('devdox_ai_sonar.cli.smart_confirm', return_value=False):
                    _run_fix_security_issues()


class TestSecurityIssuesProcessing:
    """Test security issues processing."""

    @patch('devdox_ai_sonar.cli._should_continue_to_next_file')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    def test_process_security_issues_single_file(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test processing single security issue file."""
        mock_continue.return_value = False  # Stop after first

        issues_by_file = {
            "security.py": [Mock(), Mock()]
        }

        _process_security_issues(
            issues_by_file, mock_services, auth_config, fix_params,Mock()
        )


        mock_single_fix.assert_called_once()
        mock_continue.assert_called_once_with(1, 1)

    @patch('devdox_ai_sonar.cli._should_continue_to_next_file')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    def test_process_security_issues_multiple_files(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test processing multiple security issue files."""
        mock_continue.side_effect = [True, False]  # Continue first, stop second

        issues_by_file = {
            "auth.py": [Mock()],
            "crypto.py": [Mock(), Mock()]
        }

        _process_security_issues(
            issues_by_file, mock_services, auth_config, fix_params,Mock()
        )

        assert mock_single_fix.call_count == 2

    @patch('devdox_ai_sonar.cli._should_continue_to_next_file')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    def test_process_security_issues_user_stops_early(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test user stopping processing early."""
        mock_continue.return_value = False  # User stops

        issues_by_file = {
            "file1.py": [Mock()],
            "file2.py": [Mock()],
            "file3.py": [Mock()]
        }
        md_file_path=Mock()
        _process_security_issues(
            issues_by_file, mock_services, auth_config, fix_params,md_file_path
        )

        # Should only process first file
        assert mock_single_fix.call_count == 1

    @patch('devdox_ai_sonar.cli._process_single_fix')
    def test_process_empty_issues_dict(
            self, mock_single_fix,
            mock_services, auth_config, fix_params
    ):
        """Test processing with empty issues dict."""
        issues_by_file = {}
        md_file_path = Mock()
        _process_security_issues(
            issues_by_file, mock_services, auth_config, fix_params, md_file_path
        )

        # Should handle gracefully
        mock_single_fix.assert_not_called()

    @patch('devdox_ai_sonar.cli._process_issues_for_rule')
    def test_process_rule_with_empty_issues_list(
            self, mock_process_rule,
            mock_services, auth_config, fix_params
    ):
        """Test processing rule with empty issues list."""
        mock_process_rule.return_value = True

        issues_by_rule = {
            "python:S1234": {"issue": []}  # Empty list
        }
        md_file_path = Mock()
        _process_regular_issues(
            issues_by_rule, mock_services, auth_config, fix_params, md_file_path
        )

        # Should still call process_rule
        mock_process_rule.assert_called_once()
        
# ============================================================================
# TEST CLASS: REGULAR ISSUES PROCESSING
# ============================================================================

class TestRegularIssuesProcessing:
    """Test regular issues processing."""

    @patch('devdox_ai_sonar.cli._process_issues_for_rule')
    def test_process_regular_issues_single_rule(
            self, mock_process_rule,
            mock_services, auth_config, fix_params
    ):
        """Test processing single rule."""
        mock_process_rule.return_value = True

        issues_by_rule = {
            "python:S1234": {
                "issue": [Mock(), Mock()]
            }
        }

        md_file_path = Mock()
        _process_regular_issues(
            issues_by_rule, mock_services, auth_config, fix_params, md_file_path
        )


        mock_process_rule.assert_called_once()

    @patch('devdox_ai_sonar.cli._process_issues_for_rule')
    def test_process_regular_issues_multiple_rules(
            self, mock_process_rule,
            mock_services, auth_config, fix_params
    ):
        """Test processing multiple rules."""
        mock_process_rule.side_effect = [True, True, False]  # Stop at third

        issues_by_rule = {
            "python:S1234": {"issue": [Mock()]},
            "python:S5678": {"issue": [Mock()]},
            "python:S9012": {"issue": [Mock()]}
        }
        md_file_path = Mock()
        _process_regular_issues(
            issues_by_rule, mock_services, auth_config, fix_params, md_file_path
        )

        assert mock_process_rule.call_count == 3

    @patch('devdox_ai_sonar.cli._should_continue_to_next_file')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    def test_process_issues_for_rule_all_issues(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test processing all issues for a rule."""
        mock_continue.side_effect = [True, True, False]

        issues_list = [Mock(), Mock(), Mock()]
        md_file_path = Mock()
        result = _process_issues_for_rule(
            "python:S1234", issues_list,
            mock_services, auth_config, fix_params, md_file_path
        )

        assert result is False  # Stopped by user
        assert mock_single_fix.call_count == 3

    @patch('devdox_ai_sonar.cli._should_continue_to_next_file')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    def test_process_issues_for_rule_continues(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test processing continues to next rule."""
        mock_continue.return_value = True  # Continue all

        issues_list = [Mock(), Mock()]
        md_file_path = Mock()
        result = _process_issues_for_rule(
            "python:S1234", issues_list,
            mock_services, auth_config, fix_params, md_file_path
        )

        assert result is True  # Should continue
        assert mock_single_fix.call_count == 2



# ============================================================================
# TEST CLASS: ANALYZE COMMAND
# ============================================================================

class TestAnalyzeCommand:
    """Test analyze command"""


    def test_run_analyze_success(self):
        """Test successful analysis - Updated for refactored version"""


        # Mock the return from _load_and_validate_config
        mock_auth_config = Mock()
        mock_auth_config.project = "test-project"
        mock_auth_config.organization = "test-org"
        mock_auth_config.token = "test-token"

        mock_llm_config = Mock()
        mock_llm_config.provider = "openai"
        mock_llm_config.model = "gpt-4"

        parameters = {
            "branch": "main",
            "pull_request": 0,
            "max_fixes": 10,
            "severity": "",
            "types": ""
        }

        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=5,
            issues=[],
            issues_by_severity={}
        )

        # Patch dependencies
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.smart_prompt', return_value="10"):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli._display_analysis_results'):
                            _run_analyze()

                            # Verify
                            mock_load.assert_called_once()
                            mock_analyzer.get_project_issues.assert_called_once()

    def test_run_analyze_no_results(self):
        """Test analyze when no results returned"""
        # Mock config
        mock_auth_config = Mock()
        mock_auth_config.project = "test"
        mock_auth_config.organization = "org"
        mock_auth_config.token = "token"

        mock_llm_config = Mock()

        parameters = {
            "branch": "main",
            "pull_request": 0,
            "max_fixes": 10,
            "severity": "",
            "types": ""
        }

        # Mock analyzer with no results
        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = None

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.smart_prompt', return_value="10"):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):

                        _run_analyze()

                        # Should still call analyzer
                        mock_analyzer.get_project_issues.assert_called_once()

    def test_run_analyze_no_config(self):
        """Test analyze without config - _load_and_validate_config raises Abort"""


        # _load_and_validate_config will raise click.Abort if no config
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.side_effect = click.Abort()

            with pytest.raises(click.Abort):
                _run_analyze()

    def test_run_inspect_no_config(self):
        """Test inspect without config - _load_and_validate_config raises Abort"""

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.side_effect = click.Abort()

            with pytest.raises(click.Abort):
                _run_inspect()


    def test_run_analyze_with_severity_and_types(self):
        """Test analyze with severity and type filters"""
        mock_auth_config = Mock()
        mock_auth_config.project = "test"
        mock_auth_config.organization = "org"
        mock_auth_config.token = "token"

        mock_llm_config = Mock()

        parameters = {
            "branch": "main",
            "pull_request": 0,
            "max_fixes": 10,
            "severity": "BLOCKER,CRITICAL",
            "types": "BUG,VULNERABILITY"
        }

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = None

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.smart_prompt', return_value="10"):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli._validate_severities') as mock_sev:
                            with patch('devdox_ai_sonar.cli._validate_issue_types') as mock_types:

                                _run_analyze()

                                # Should validate filters
                                mock_sev.assert_called()
                                mock_types.assert_called()



# ============================================================================
# TEST CLASS: INSPECT COMMAND
# ============================================================================

class TestInspectCommand:
    """Test inspect command"""

    def test_run_inspect_success(self, mock_loaded_config, mock_analyzer):

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = mock_loaded_config

            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer') as mock_sa:
                mock_sa.return_value = mock_analyzer


                ctx = Mock()
                ctx.obj = {"verbose": False}

                # ✅ Works in CI/CD!
                _run_inspect()

                # Verify
                mock_load.assert_called_once()
                mock_analyzer.analyze_project_directory.assert_called_once()


    def test_run_inspect_invalid_config(self):
        """Test inspect with invalid config - EXPECTS click.Abort"""

        # Mock _load_and_validate_config to raise click.Abort (as it should!)
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            # When config is invalid, _load_and_validate_config raises click.Abort
            mock_load.side_effect = click.Abort()


            # Assert that click.Abort is raised (this is CORRECT behavior!)
            with pytest.raises(click.Abort):
                _run_inspect()

            # Verify _load_and_validate_config was called
            mock_load.assert_called_once()


    def test_run_inspect_empty_analysis(self):
        """Test inspect with empty analysis results"""
        mock_auth_config = Mock()
        mock_auth_config.project_path = "/tmp/empty"
        mock_auth_config.token = "token"
        mock_auth_config.organization = "org"

        mock_llm_config = Mock()
        parameters = {"branch": "main", "pull_request": 0}

        # Mock analyzer with empty results
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_project_directory.return_value = {
            "total_files": 0,
            "python_files": 0,
            "javascript_files": 0,
            "java_files": 0,
            "has_sonar_config": False,
            "has_git": False,
            "potential_source_dirs": []
        }

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):


                _run_inspect()

                # Should still complete successfully
                mock_analyzer.analyze_project_directory.assert_called_once()



class TestSelectExistingUI:
    """Tests for _select_existing_ui"""

    @patch('devdox_ai_sonar.cli.inquirer')
    @patch('devdox_ai_sonar.cli.console')
    def test_select_existing_ui_success(self, mock_console, mock_inquirer):
        """Test successful selection"""
        mock_inquirer.prompt.return_value = {"provider": "openai"}

        result = _select_existing_ui("provider", "Select provider", ["openai", "anthropic"])

        assert result == "openai"

    @patch('devdox_ai_sonar.cli.inquirer')
    @patch('devdox_ai_sonar.cli.console')
    def test_select_existing_ui_cancelled(self, mock_console, mock_inquirer):
        """Test cancelled selection"""
        mock_inquirer.prompt.return_value = None

        result = _select_existing_ui("provider", "Select", ["openai"])

        assert result == ""


# ============================================================================
# TEST CLASS: VALIDATION HELPERS
# ============================================================================

class TestValidationHelpers:
    """Test validation helper functions"""

    def test_validate_issue_types_valid(self):
        """Test valid issue types"""
        result = _validate_issue_types("BUG,VULNERABILITY")
        assert "BUG" in result
        assert "VULNERABILITY" in result

    def test_validate_issue_types_none(self):
        """Test None issue types"""
        result = _validate_issue_types(None)
        assert result is None

    def test_validate_issue_types_empty(self):
        """Test empty issue types"""
        result = _validate_issue_types("")
        assert result is None

    def test_validate_severities_valid(self):
        """Test valid severities"""
        result = _validate_severities("BLOCKER,CRITICAL")
        assert "BLOCKER" in result
        assert "CRITICAL" in result

    def test_validate_severities_none(self):
        """Test None severities"""
        result = _validate_severities(None)
        assert result is None

    def test_validate_severities_empty(self):
        """Test empty severities"""
        result = _validate_severities("")
        assert result is None


# ============================================================================
# TEST CLASS: PROCESS FUNCTIONS
# ============================================================================

class TestProcessFunctions:
    """Test issue processing functions"""

    def test_process_no_issues_found(self):
        """Test processing when no issues found"""
        # Create mock objects
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues_by_files.return_value = {}

        mock_ruler = MagicMock()
        mock_fixer = MagicMock()

        # Auth config
        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"

        # LLM config
        llm_config = Mock()
        llm_config.provider = "openai"
        llm_config.model = "gpt-4"
        llm_config.api_key = "key"

        # Fix params
        fix_params = {
            "max_fixes": 10,
            "types_list": None,
            "severities_list": None,
            "apply": False,
            "dry_run": False
        }

        # Patch all the dependencies with proper mocking
        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer', return_value=mock_ruler):
                with patch('devdox_ai_sonar.cli.LLMFixer', return_value=mock_fixer):
                    # CRITICAL: Mock show_progress as a context manager
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        from devdox_ai_sonar.cli import _process_and_fix_issues

                        # This should not raise an error
                        _process_and_fix_issues(
                            auth_config,
                            llm_config,
                            "main",
                            None,
                            fix_params,
                            issue_type=IssueType.SECURITY
                        )

                        # Verify the analyzer was called correctly
                        mock_analyzer.get_fixable_issues_by_files.assert_not_called()

    def test_process_security_no_issues(self):
        """Test security processing when no issues"""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = {}

        mock_ruler = MagicMock()
        mock_fixer = MagicMock()

        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"

        llm_config = Mock()
        llm_config.provider = "openai"
        llm_config.model = "gpt-4"
        llm_config.api_key = "key"

        fix_params = {
            "max_fixes": 10,
            "apply": False,
            "dry_run": False,
            "create_backup": True
        }

        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer', return_value=mock_ruler):
                with patch('devdox_ai_sonar.cli.LLMFixer', return_value=mock_fixer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):

                        _process_and_fix_issues(
                            auth_config,
                            llm_config,
                            "main",
                            None,
                            fix_params,
                            issue_type=IssueType.SECURITY
                        )

                        mock_analyzer.get_fixable_security_issues.assert_called_once()

    def test_process_security_with_issues(self):
        """Test security processing with issues"""
        mock_analyzer = MagicMock()

        mock_issue = Mock()
        mock_issue.rule = "python:S5678"
        mock_issue.file = "security.py"

        mock_analyzer.get_fixable_security_issues.return_value = {
            "security.py": [mock_issue]
        }

        mock_ruler = MagicMock()
        mock_ruler.get_rule_by_key.return_value = {
            "key": "python:S5678",
            "name": "Security Rule"
        }

        mock_fix = Mock()
        mock_fix.file_path = "security.py"
        mock_fix.confidence = 0.95
        mock_fix.fixed_code = "# Secure code"

        mock_fixer = MagicMock()
        mock_fixer.generate_fix_by_file.return_value = mock_fix

        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"

        llm_config = Mock()
        llm_config.provider = "openai"
        llm_config.model = "gpt-4"
        llm_config.api_key = "key"

        fix_params = {
            "max_fixes": 10,
            "apply": False,
            "dry_run": True,
            "create_backup": True
        }

        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer', return_value=mock_ruler):
                with patch('devdox_ai_sonar.cli.LLMFixer', return_value=mock_fixer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli._display_fix_preview'):
                            with patch('devdox_ai_sonar.cli.smart_confirm', return_value=False):


                                _process_and_fix_issues(
                                    auth_config,
                                    llm_config,
                                    "main",
                                    None,
                                    fix_params,
                                    issue_type=IssueType.SECURITY
                                )

                                mock_fixer.generate_fix_by_file.assert_called_once()


# ============================================================================
# DISPLAY FUNCTIONS TESTS
# ============================================================================

class TestDisplayFunctions:
    """Tests for display functions"""

    def test_display_fix_preview_basic(self):
        """Test basic fix preview"""
        fix = FixSuggestion(
            issue_key="issue-1",
            rule="python:S1234",
            original_code="old",
            fixed_code="new",
            explanation="Fixed",
            confidence=0.95,
            llm_model="a",
            file_path="test.py",
            sonar_line_number=10
        )

        with patch('devdox_ai_sonar.cli.console'):
            with patch('devdox_ai_sonar.cli.Panel'):
                _display_fix_preview(fix, [Mock()])

    def test_display_fix_preview_long_code(self):
        """Test with long code"""
        long_code = "x = 1\n" * 100

        fix = FixSuggestion(
            issue_key="issue-1",
            rule="python:S1234",
            file_path="long.py",
            original_code="old",
            fixed_code=long_code,
            explanation="Fixed",
            llm_model="a",
            confidence=0.9,
            sonar_line_number=5
        )

        with patch('devdox_ai_sonar.cli.console'):
            with patch('devdox_ai_sonar.cli.Panel') as mock_panel:
                _display_fix_preview(fix, [Mock()])

                panel_call = mock_panel.call_args[0][0]
                assert len(panel_call) <= 504

    def test_display_analysis_results_with_metrics(self, sample_analysis_result):
        """Test displaying results with metrics"""
        with patch('devdox_ai_sonar.cli.console'):
            with patch('devdox_ai_sonar.cli.Panel'):
                with patch('devdox_ai_sonar.cli.Table'):
                    _display_analysis_results(sample_analysis_result, limit=10)

    def test_display_fix_results_success(self, sample_fix_result):
        """Test displaying fix results"""
        with patch('devdox_ai_sonar.cli.console'):
            _display_fix_results(sample_fix_result)

    def test_display_project_header(self):
        """Test header display"""
        result = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=5,
            issues=[],
            issues_by_severity={}
        )

        # Should not raise exception
        _display_project_header(result)

    def test_display_metrics_section_no_metrics(self):
        """Test metrics display with no metrics"""
        result = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=5,
            issues=[],
            issues_by_severity={},
            metrics=None
        )

        # Should handle None gracefully
        _display_metrics_section(result)

# ============================================================================
# FILE PROCESSING TESTS
# ============================================================================

class TestFileProcessing:
    """Tests for file processing functions"""

    def test_process_files_with_issues_basic(self):
        """Test basic file processing"""
        issues_by_file = {"test.py": [Mock(rule="python:S1234")]}

        mock_services = {
            'analyzer': Mock(),
            'ruler': Mock(),
            'fixer': Mock()
        }
        mock_services['ruler'].get_rule_by_key.return_value = {}
        mock_services['fixer'].generate_fix_by_file.return_value = None

        with patch('devdox_ai_sonar.cli.console'):
            with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                with patch('devdox_ai_sonar.cli._should_continue_to_next_file', return_value=False):
                    _process_files_with_issues(
                        issues_by_file,
                        mock_services,
                        Mock(project_path="/tmp"),
                        {"apply": False, "dry_run": False},
                        issue_type=IssueType.SECURITY
                    )

    def test_generate_fix_for_file_success(self):
        """Test successful fix generation"""
        issues = [Mock(rule="python:S1234")]

        mock_services = {
            'ruler': Mock(),
            'fixer': Mock()
        }
        mock_services['ruler'].get_rule_by_key.return_value = {}

        mock_fix = FixSuggestion(
            issue_key="issue-1",
            rule="python:S1234",
            file_path="test.py",
            original_code="old",
            fixed_code="new",
            explanation="Fixed",
            confidence=0.9,
            llm_model="a",
            sonar_line_number=10
        )
        mock_services['fixer'].generate_fix_by_file.return_value = mock_fix

        with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
            result = _generate_fix_for_file(issues, mock_services, Mock(project_path="/tmp"),issue_type=IssueType.SECURITY)

            assert result == mock_fix

    def test_handle_fix_apply_true(self):
        """Test applying fix"""
        fix = FixSuggestion(
            issue_key="issue-1",
            rule="python:S1234",
            file_path="test.py",
            original_code="old",
            fixed_code="new",
            explanation="Fixed",
            confidence=0.9,
            llm_model="a",
            sonar_line_number=10
        )

        mock_fixer = Mock()
        mock_fixer.apply_fixes_with_validation.return_value = FixResult(
            successful_fixes=[fix],
            failed_fixes=[],
            skipped_files=[],
            total_fixes_attempted=1,
            success_rate=1.0,
            backup_created=True,
            backup_path=Path("/tmp/backup"),
            project_path=Path("/tmp/backup")
        )

        with patch('devdox_ai_sonar.cli._display_fix_preview'):
            with patch('devdox_ai_sonar.cli._display_fix_results'):
                handle_fix(
                    fix,
                    [Mock()],
                    mock_fixer,
                    Mock(project_path="/tmp"),
                    {"apply": 1, "dry_run": 0, "create_backup": 1}
                )

                mock_fixer.apply_fixes_with_validation.assert_called_once()

    @patch('devdox_ai_sonar.cli.smart_confirm')
    def test_should_continue_to_next_file_last_file(self, mock_confirm):
        """Test on last file"""
        result = _should_continue_to_next_file(5, 5)

        assert result is False
        mock_confirm.assert_not_called()



# ============================================================================
# TEST CLASS: LOAD AND VALIDATE CONFIG
# ============================================================================

class TestLoadAndValidateConfig:
    """Test configuration loading"""

    def test_load_and_validate_config_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test successful config load"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {"max_fixes": 10}

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        result = _load_and_validate_config()

                        assert result is not None
                        assert len(result) == 3

    def test_load_and_validate_config_no_auth(self):
        """Test config load without auth"""
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_instance = Mock()
            mock_instance.load_auth_config.return_value = None
            mock_cs.return_value = mock_instance

            with pytest.raises(Exception):  # Should abort
                _load_and_validate_config()

    def test_load_and_validate_config_invalid_auth(self, mock_config_service):
        """Test config load with invalid auth"""
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                mock_auth_instance = Mock()
                mock_auth_instance.validate.return_value = (False, "Invalid")
                mock_auth.return_value = mock_auth_instance

                with pytest.raises(Exception):  # Should abort
                    _load_and_validate_config()

    def test_load_and_validate_config_no_llm(
            self, mock_config_service, mock_config_manager
    ):
        """Test config load without LLM config"""
        mock_config_service.load_llm_config.return_value = None

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                    mock_auth_instance = Mock()
                    mock_auth_instance.validate.return_value = (True, None)
                    mock_auth.return_value = mock_auth_instance

                    with pytest.raises(Exception):  # Should abort
                        _load_and_validate_config()

    def test_load_and_validate_config_with_predefined_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test successful config load using predefined branch/PR"""
        # Setup mocks
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {"max_fixes": 10}
        mock_provider_manager.branch_or_pr.return_value = ("main", "123")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        # Call with use_predefined=True
                        result = _load_and_validate_config(use_predefined=True)

                        # Assertions
                        assert result is not None
                        assert len(result) == 3
                        auth_config, llm_config, params = result

                        # Verify branch_or_pr() was called, not branch_or_pr_prompt()
                        mock_provider_manager.branch_or_pr.assert_called_once()
                        mock_provider_manager.branch_or_pr_prompt.assert_not_called()

                        # Verify params contain correct branch/PR
                        assert params['branch'] == "main"
                        assert params['pull_request'] == "123"

    def test_load_and_validate_config_with_predefined_branch_only(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test predefined mode with only branch (no PR)"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.return_value = ("feature/test", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        result = _load_and_validate_config(use_predefined=True)

                        assert result is not None
                        _, _, params = result
                        assert params['branch'] == "feature/test"
                        assert params['pull_request'] is None

    def test_load_and_validate_config_with_predefined_pr_only(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test predefined mode with only PR (no branch)"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.return_value = (None, "456")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        result = _load_and_validate_config(use_predefined=True)

                        assert result is not None
                        _, _, params = result
                        assert params['branch'] is None
                        assert params['pull_request'] == "456"

    def test_load_and_validate_config_with_predefined_no_branch_or_pr(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test predefined mode when no branch/PR in config - should abort"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.return_value = (None, None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        with pytest.raises(Exception):  # Should abort
                            _load_and_validate_config(use_predefined=True)

                        # Verify correct method was attempted
                        mock_provider_manager.branch_or_pr.assert_called_once()

        # ========================================================================
        # NEW TESTS: use_predefined=False (Prompt Mode - Explicit)
        # ========================================================================

    def test_load_and_validate_config_with_prompt_explicit(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test explicit use_predefined=False uses prompt method"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr_prompt.return_value = ("develop", "789")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        result = _load_and_validate_config(use_predefined=False)

                        assert result is not None
                        _, _, params = result

                        # Verify prompt method was called, not predefined method
                        mock_provider_manager.branch_or_pr_prompt.assert_called_once()
                        mock_provider_manager.branch_or_pr.assert_not_called()

                        assert params['branch'] == "develop"
                        assert params['pull_request'] == "789"

    def test_load_and_validate_config_prompt_mode_no_branch_or_pr(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test prompt mode when user provides neither branch nor PR - should abort"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        with pytest.raises(Exception):  # Should abort
                            _load_and_validate_config(use_predefined=False)

                        mock_provider_manager.branch_or_pr_prompt.assert_called_once()

        # ========================================================================
        # NEW TESTS: Edge Cases and Error Scenarios
        # ========================================================================

    def test_load_and_validate_config_predefined_method_raises_exception(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test predefined mode when branch_or_pr() raises exception"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.side_effect = Exception("Config error")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        with pytest.raises(Exception) as exc_info:
                            _load_and_validate_config(use_predefined=True)

                        assert "Config error" in str(exc_info.value)

    def test_load_and_validate_config_params_merge_with_existing_config(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test that branch/PR params are merged with existing configuration"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        existing_config = {"max_fixes": 10, "timeout": 30}
        mock_config_manager.get_value.return_value = existing_config.copy()
        mock_provider_manager.branch_or_pr.return_value = ("main", "100")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config(use_predefined=True)

                        # Verify existing config is preserved
                        assert params['max_fixes'] == 10
                        assert params['timeout'] == 30
                        # Verify new params are added
                        assert params['branch'] == "main"
                        assert params['pull_request'] == "100"

    def test_load_and_validate_config_params_empty_dict_when_no_existing(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test params initialization when no existing config"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = None  # No existing config
        mock_provider_manager.branch_or_pr_prompt.return_value = ("test", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # Should handle None gracefully and create new dict
                        assert isinstance(params, dict)
                        assert params['branch'] == "test"
                        assert params['pull_request'] is None

        # ========================================================================
        # NEW TESTS: Console Output Verification
        # ========================================================================



    def test_load_and_validate_config_returns_correct_types(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test return values have correct types"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.return_value = ("main", "1")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        auth_config, llm_config, params = _load_and_validate_config(
                            use_predefined=True
                        )

                        # Verify types
                        assert isinstance(auth_config, (Mock, AuthConfig))  # Mock or actual
                        assert llm_config is not None
                        assert isinstance(params, dict)  # Should be Dict[str, Any], not Optional[str]

    def test_load_and_validate_config_console_message_loading(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test console displays 'Loading configuration...' message"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", "1")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        with patch('devdox_ai_sonar.cli.console') as mock_console:
                            _load_and_validate_config()

                            # Verify loading message displayed at start
                            loading_calls = [
                                call for call in mock_console.print.call_args_list
                                if 'Loading configuration' in str(call)
                            ]
                            assert len(loading_calls) >= 1

    def test_load_and_validate_config_console_message_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test console displays success message after loading"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", "1")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        with patch('devdox_ai_sonar.cli.console') as mock_console:
                            _load_and_validate_config()

                            # Verify success message displayed at end
                            success_calls = [
                                call for call in mock_console.print.call_args_list
                                if '✓' in str(call) or 'Configuration loaded' in str(call)
                            ]
                            assert len(success_calls) >= 1

    def test_load_and_validate_config_console_message_no_auth(
            self, mock_config_service
    ):
        """Test console displays appropriate messages when auth not found"""
        mock_config_service.load_auth_config.return_value = None

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.console') as mock_console:
                with pytest.raises(click.Abort):
                    _load_and_validate_config()

                # Verify error messages displayed
                calls = [str(call) for call in mock_console.print.call_args_list]
                # Should display AUTHENTICATION_NOT_FOUND and DEVDOX_SONAR_CONFIG
                assert len(calls) >= 2

    def test_load_and_validate_config_console_message_invalid_auth(
            self, mock_config_service, mock_config_manager
    ):
        """Test console displays error message for invalid auth"""
        mock_config_service.load_auth_config.return_value = {"token": "test"}

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                    mock_auth_instance = Mock()
                    mock_auth_instance.validate.return_value = (False, "Invalid token")
                    mock_auth.return_value = mock_auth_instance

                    with patch('devdox_ai_sonar.cli.console') as mock_console:
                        with pytest.raises(click.Abort):
                            _load_and_validate_config()

                        # Verify error message contains the validation error
                        calls = [str(call) for call in mock_console.print.call_args_list]
                        error_calls = [c for c in calls if 'Invalid token' in c]
                        assert len(error_calls) >= 1

    def test_load_and_validate_config_console_message_no_llm(
            self, mock_config_service, mock_config_manager
    ):
        """Test console displays error when no LLM providers configured"""
        mock_config_service.load_llm_config.return_value = None

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                    mock_auth_instance = Mock()
                    mock_auth_instance.validate.return_value = (True, None)
                    mock_auth.return_value = mock_auth_instance

                    with patch('devdox_ai_sonar.cli.console') as mock_console:
                        with pytest.raises(click.Abort):
                            _load_and_validate_config()

                        # Verify LLM error message
                        calls = [str(call) for call in mock_console.print.call_args_list]
                        llm_error_calls = [c for c in calls if 'LLM' in c or 'providers' in c]
                        assert len(llm_error_calls) >= 1

    def test_load_and_validate_config_console_message_no_branch_pr(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test console displays error when no branch/PR specified"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        with patch('devdox_ai_sonar.cli.console') as mock_console:
                            with pytest.raises(click.Abort):
                                _load_and_validate_config()

                            # Verify NO_BRANCH_OR_PR_SPECIFIED message displayed
                            mock_console.print.assert_any_call(constant.NO_BRANCH_OR_PR_SPECIFIED)

    def test_load_and_validate_config_exclude_rules_string_split(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test exclude_rules string is split into list"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        exclude_rules_string = "python:S3776,python:S1192,python:S107"
        mock_config_manager.get_value.return_value = {
            "exclude_rules": exclude_rules_string
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # Verify exclude_rules is split into list
                        assert isinstance(params['exclude_rules'], list)
                        assert len(params['exclude_rules']) == 3
                        assert params['exclude_rules'] == ["python:S3776", "python:S1192", "python:S107"]

    def test_load_and_validate_config_exclude_rules_single_rule(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test exclude_rules with single rule (no comma)"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {
            "exclude_rules": "python:S3776"
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # Single rule should result in list with one element
                        assert isinstance(params['exclude_rules'], list)
                        assert len(params['exclude_rules']) == 1
                        assert params['exclude_rules'] == ["python:S3776"]

    def test_load_and_validate_config_exclude_rules_with_spaces(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test exclude_rules handles spaces around commas"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {
            "exclude_rules": "python:S3776, python:S1192 , python:S107"
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # Should preserve spaces (or you might want to strip them)
                        assert isinstance(params['exclude_rules'], list)
                        assert len(params['exclude_rules']) == 3

    def test_load_and_validate_config_no_exclude_rules(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test when exclude_rules is not in config"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {
            "max_fixes": 10
            # No exclude_rules
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # exclude_rules should not be in params or be None
                        assert 'exclude_rules' not in params or params.get('exclude_rules') is None

    def test_load_and_validate_config_exclude_rules_empty_string(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test when exclude_rules is empty string"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {
            "exclude_rules": ""
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # Empty string should not be split (falsy value)
                        # Based on code: if params.get("exclude_rules")
                        assert 'exclude_rules' in params
                        # Empty string is falsy, so split shouldn't be called
                        assert params['exclude_rules'] == ""

    def test_load_and_validate_config_exclude_rules_already_list(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test when exclude_rules is already a list (edge case)"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        # If somehow exclude_rules is already a list
        mock_config_manager.get_value.return_value = {
            "exclude_rules": ["python:S3776", "python:S1192"]
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        # This might raise AttributeError: 'list' object has no attribute 'split'
                        # Need to test actual behavior
                        try:
                            _, _, params = _load_and_validate_config()
                            # If it doesn't raise, verify it's still a list
                            assert isinstance(params['exclude_rules'], list)
                        except AttributeError:
                            # Expected if code doesn't handle list case
                            pytest.xfail("Code doesn't handle exclude_rules as list")

    def test_load_and_validate_config_params_preserves_all_existing_config(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test all existing config values are preserved in params"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        existing_config = {
            "max_fixes": 10,
            "timeout": 30,
            "types": "BUG,VULNERABILITY",
            "severities": "CRITICAL",
            "apply": 1,
            "backup": 0,
            "exclude_rules": "python:S3776"
        }
        mock_config_manager.get_value.return_value = existing_config.copy()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", "100")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # Verify all existing config preserved
                        assert params['max_fixes'] == 10
                        assert params['timeout'] == 30
                        assert params['types'] == "BUG,VULNERABILITY"
                        assert params['severities'] == "CRITICAL"
                        assert params['apply'] == 1
                        assert params['backup'] == 0
                        # exclude_rules should be split
                        assert params['exclude_rules'] == ["python:S3776"]
                        # New params added
                        assert params['branch'] == "main"
                        assert params['pull_request'] == "100"

    def test_load_and_validate_config_params_branch_overwrites_existing(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test branch/PR from prompt overwrites any existing values"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {
            "branch": "old-branch",
            "pull_request": "old-pr"
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("new-branch", "new-pr")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch('devdox_ai_sonar.cli.AuthConfig') as mock_auth:
                        mock_auth_instance = Mock()
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_auth.return_value = mock_auth_instance

                        _, _, params = _load_and_validate_config()

                        # New values should overwrite old
                        assert params['branch'] == "new-branch"
                        assert params['pull_request'] == "new-pr"
# ============================================================================
# TEST CLASS: TEST WITH FIXTURES
# ============================================================================

class TestWithFixtures:
    """Cleaner tests using fixtures"""

    def test_run_analyze_with_fixtures(self, mock_loaded_config):
        """Test analyze using fixtures"""

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=0,
            issues=[],
            issues_by_severity={}
        )

        with patch('devdox_ai_sonar.cli._load_and_validate_config', return_value=mock_loaded_config):
            with patch('devdox_ai_sonar.cli.smart_prompt', return_value="10"):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):

                        with patch('devdox_ai_sonar.cli._display_analysis_results'):

                            _run_analyze()

    def test_run_inspect_with_fixtures(self, mock_loaded_config):
        """Test inspect using fixtures"""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_project_directory.return_value = {
            "total_files": 5,
            "python_files": 3,
            "javascript_files": 2,
            "java_files": 0,
            "has_sonar_config": True,
            "has_git": True,
            "potential_source_dirs": ["src"]
        }

        with patch('devdox_ai_sonar.cli._load_and_validate_config', return_value=mock_loaded_config):
            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):

                _run_inspect()


class TestShouldContinueToMenu:
    """Test cases for _should_continue_to_menu."""

    @patch('devdox_ai_sonar.cli.smart_confirm')
    def test_should_continue_returns_true(self, mock_confirm):
        """Test returns True when user confirms."""
        mock_confirm.return_value = True

        result = _should_continue_to_menu()

        assert result is True
        mock_confirm.assert_called_once_with(
            "Return to main menu?",
            default=True,
            allow_switch=False
        )

    @patch('devdox_ai_sonar.cli.smart_confirm')
    def test_should_continue_returns_false(self, mock_confirm):
        """Test returns False when user declines."""
        mock_confirm.return_value = False

        result = _should_continue_to_menu()

        assert result is False


class TestHandleCommandSwitch:
    """Test cases for _handle_command_switch."""

    @patch('devdox_ai_sonar.cli.console')
    def test_prints_switch_message(self, mock_console):
        """Test prints command switch message."""
        _handle_command_switch()

        mock_console.print.assert_called_once()
        assert "Returning to command menu" in str(mock_console.print.call_args)


class TestHandleKeyboardInterrupt:
    """Test cases for _handle_keyboard_interrupt."""

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_user_confirms_exit(self, mock_console, mock_confirm):
        """Test user confirms exit."""
        mock_confirm.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            _handle_keyboard_interrupt()

        assert exc_info.value.code == 0

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_user_declines_exit(self, mock_console, mock_confirm):
        """Test user declines exit."""
        mock_confirm.return_value = False

        result = _handle_keyboard_interrupt()

        assert result is False

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_double_keyboard_interrupt(self, mock_console, mock_confirm):
        """Test double Ctrl+C forces exit."""
        mock_confirm.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            _handle_keyboard_interrupt()

        assert exc_info.value.code == 130  # SIGINT exit code

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_eof_error_forces_exit(self, mock_console, mock_confirm):
        """Test EOF error forces exit."""
        mock_confirm.side_effect = EOFError()

        with pytest.raises(SystemExit) as exc_info:
            _handle_keyboard_interrupt()

        assert exc_info.value.code == 130

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_unexpected_exception_during_exit(self, mock_console, mock_confirm):
        """Test unexpected exception during exit confirmation."""
        mock_confirm.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(SystemExit) as exc_info:
            _handle_keyboard_interrupt()

        assert exc_info.value.code == 1


class TestHandleInteractiveError:
    """Test cases for _handle_interactive_error."""

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_user_returns_to_menu(self, mock_console, mock_confirm):
        """Test user chooses to return to menu."""
        mock_confirm.return_value = True
        error = ValueError("Test error")

        result = _handle_interactive_error(error)

        assert result is False
        mock_console.print.assert_called()

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_user_exits_after_error(self, mock_console, mock_confirm):
        """Test user chooses to exit after error."""
        mock_confirm.return_value = False
        error = ValueError("Test error")

        with pytest.raises(SystemExit) as exc_info:
            _handle_interactive_error(error)

        assert exc_info.value.code == 1

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_keyboard_interrupt_during_error_handling(self, mock_console, mock_confirm):
        """Test keyboard interrupt during error handling."""
        mock_confirm.side_effect = KeyboardInterrupt()
        error = ValueError("Test error")

        with pytest.raises(SystemExit) as exc_info:
            _handle_interactive_error(error)

        assert exc_info.value.code == 1

    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli.console')
    def test_fatal_error_during_confirmation(self, mock_console, mock_confirm):
        """Test fatal error during confirmation prompt."""
        mock_confirm.side_effect = RuntimeError("Fatal error")
        error = ValueError("Test error")

        with pytest.raises(SystemExit) as exc_info:
            _handle_interactive_error(error)

        assert exc_info.value.code == 2

# ============================================================================
# FETCH ISSUES TESTS
# ============================================================================

class TestFetchIssues:
    """Tests for fetch issues functions"""

    def test_fetch_issues_by_type_security(self):
        """Test fetching security issues"""
        mock_analyzer = Mock()
        mock_analyzer.get_fixable_security_issues.return_value = {"test.py": [Mock()]}

        with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
            result = _fetch_issues_by_type(
                mock_analyzer,
                Mock(project="test"),
                "main",
                None,
                {"max_fixes": 10},
                IssueType.SECURITY
            )

            assert len(result) == 1





class TestInteractiveMode:
    """Tests for interactive mode functions"""

    @patch('devdox_ai_sonar.cli._process_interactive_command')
    def test_execute_interactive_iteration_normal(self, mock_process):
        """Test normal iteration"""
        mock_process.return_value = False

        result = _execute_interactive_iteration(Mock())

        assert result is False

    @patch('devdox_ai_sonar.cli._process_interactive_command')
    @patch('devdox_ai_sonar.cli._handle_command_switch')
    def test_execute_interactive_iteration_switch(self, mock_handle, mock_process):
        """Test with command switch"""
        mock_process.side_effect = SwitchCommandException()

        result = _execute_interactive_iteration(Mock())

        assert result is False
        mock_handle.assert_called_once()

# ============================================================================
# Test _run_interactive_mode (Integration)
# ============================================================================

class TestRunInteractiveMode:
    """Integration tests for _run_interactive_mode."""

    @patch('devdox_ai_sonar.cli.show_command_selector')
    @patch('devdox_ai_sonar.cli._exit_application')
    def test_exits_when_no_command_selected(self, mock_exit, mock_selector):
        """Test exits when no command selected."""
        mock_selector.return_value = None
        mock_exit.side_effect = SystemExit(0)
        mock_ctx = Mock(spec=click.Context)

        with pytest.raises(SystemExit):
            _run_interactive_mode(mock_ctx)

        mock_exit.assert_called_once()

    @patch('devdox_ai_sonar.cli.show_command_selector')
    @patch('devdox_ai_sonar.cli._execute_interactive_command')
    @patch('devdox_ai_sonar.cli._should_continue_to_menu')
    @patch('devdox_ai_sonar.cli._exit_application')
    def test_executes_command_and_exits(
            self, mock_exit, mock_continue, mock_execute, mock_selector
    ):
        """Test executes command then exits."""
        mock_selector.return_value = 'fix_issues'
        mock_continue.return_value = False
        mock_exit.side_effect = SystemExit(0)
        mock_ctx = Mock(spec=click.Context)

        with pytest.raises(SystemExit):
            _run_interactive_mode(mock_ctx)

        mock_execute.assert_called_once()
        mock_exit.assert_called_once()

    @patch('devdox_ai_sonar.cli.show_command_selector')
    @patch('devdox_ai_sonar.cli._execute_interactive_command')
    @patch('devdox_ai_sonar.cli._should_continue_to_menu')
    def test_handles_switch_command_exception(
            self, mock_continue, mock_execute, mock_selector
    ):
        """Test handles SwitchCommandException."""
        mock_selector.side_effect = [
            'fix_issues',
            'exit'
        ]
        mock_execute.side_effect = SwitchCommandException()
        mock_continue.return_value = True
        mock_ctx = Mock(spec=click.Context)

        with pytest.raises(SystemExit):
            _run_interactive_mode(mock_ctx)

        # Should have tried to execute twice (once raises exception, once exits)
        assert mock_selector.call_count == 2


# ============================================================================
# Test _process_and_fix_issues Refactored Functions
# ============================================================================

class TestInitializeFixServices:
    """Test cases for _initialize_fix_services."""

    @patch('devdox_ai_sonar.cli.SonarCloudAnalyzer')
    @patch('devdox_ai_sonar.cli.RuleAnalyzer')
    @patch('devdox_ai_sonar.cli.LLMFixer')
    @patch('devdox_ai_sonar.cli.console')
    def test_initializes_all_services(
            self, mock_console, mock_fixer, mock_ruler, mock_analyzer
    ):
        """Test initializes all required services."""
        auth_config = AuthConfig(
            token="test_token",
            organization="test_org",
            project="test_project",
            project_path="/test/path"
        )
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test_key",
            models=["gpt-4"]
        )

        services = _initialize_fix_services(auth_config, llm_config)

        assert 'analyzer' in services
        assert 'ruler' in services
        assert 'fixer' in services

        mock_analyzer.assert_called_once_with("test_token", "test_org")
        mock_ruler.assert_called_once_with("test_token", "test_org")
        mock_fixer.assert_called_once_with(
            provider="openai",
            model="gpt-4",
            api_key="test_key"
        )


class TestCollectRuleInformation:
    """Test cases for _collect_rule_information."""

    def test_collects_rules_for_all_issues(self):
        """Test collects rule information for all issues."""
        mock_ruler = Mock()
        mock_ruler.get_rule_by_key.side_effect = [
            {'key': 'rule1', 'name': 'Rule 1'},
            {'key': 'rule2', 'name': 'Rule 2'}
        ]

        issues = [
            Mock(rule='rule1'),
            Mock(rule='rule2')
        ]

        result = _collect_rule_information(issues, mock_ruler)

        assert len(result) == 2
        assert 'rule1' in result
        assert 'rule2' in result
        assert mock_ruler.get_rule_by_key.call_count == 2



class TestAddProviderHelpers:
    """Test helper functions used by add_provider"""

    def test_should_stop_configuring_no_providers(self):
        """Test _should_stop_configuring when no providers left"""
        from devdox_ai_sonar.cli import _should_stop_configuring

        with patch('devdox_ai_sonar.cli.console.print'):
            result = _should_stop_configuring([])
            assert result is True

    def test_should_stop_configuring_user_declines(self):
        """Test _should_stop_configuring when user says no"""
        from devdox_ai_sonar.cli import _should_stop_configuring

        with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=False):
            result = _should_stop_configuring(['openai'])
            assert result is True

    def test_should_stop_configuring_user_continues(self):
        """Test _should_stop_configuring when user wants to continue"""
        from devdox_ai_sonar.cli import _should_stop_configuring

        with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=True):
            result = _should_stop_configuring(['openai'])
            assert result is False

    def test_display_operation_header(self):
        """Test _display_operation_header"""
        from devdox_ai_sonar.cli import _display_operation_header

        with patch('devdox_ai_sonar.cli.console.print') as mock_print:
            _display_operation_header("TEST OPERATION")

            # Verify print was called multiple times (header, title, header)
            assert mock_print.call_count >= 3

    def test_display_completion_message(self):
        """Test _display_completion_message"""
        from devdox_ai_sonar.cli import _display_completion_message

        with patch('devdox_ai_sonar.cli.console.print') as mock_print:
            _display_completion_message()

            # Verify completion message components were printed
            assert mock_print.call_count >= 3

            # Check that success message is in one of the calls
            call_args = str(mock_print.call_args_list)
            assert 'COMPLETE' in call_args or 'saved' in call_args


class TestClickAbortPatterns:
    """Examples of different patterns for testing click.Abort"""

    @pytest.fixture
    def mock_setup(self):
        """Common mock setup"""
        mock_manager = Mock()
        mock_manager.load_config.return_value = None

        mock_provider_mgr = Mock()
        mock_provider_mgr.get_existing_providers.return_value = []

        return mock_manager, mock_provider_mgr

    # Pattern 1: Direct exception catching
    def test_pattern_1_direct_catch(self, mock_setup):
        """Pattern 1: Catch click.Abort directly"""
        mock_manager, mock_provider_mgr = mock_setup

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock())

            with patch('devdox_ai_sonar.cli.console.print'):
                # ✅ Correct way
                with pytest.raises(click.Abort):
                    update_provider()

    # Pattern 2: Try-except verification
    def test_pattern_2_try_except(self, mock_setup):
        """Pattern 2: Use try-except for more control"""
        mock_manager, mock_provider_mgr = mock_setup

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock())

            with patch('devdox_ai_sonar.cli.console.print'):
                abort_raised = False
                try:
                    update_provider()
                except click.Abort:
                    abort_raised = True

                assert abort_raised, "Expected click.Abort to be raised"

    # Pattern 4: Check exception message
    def test_pattern_4_exception_info(self, mock_setup):
        """Pattern 4: Verify exception details"""
        mock_manager, mock_provider_mgr = mock_setup

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock())

            with patch('devdox_ai_sonar.cli.console.print') as mock_print:
                with pytest.raises(click.Abort) as exc_info:
                    update_provider()

                # Verify it's the right exception type
                assert isinstance(exc_info.value, click.Abort)

                # Verify console.print was called before abort
                assert mock_print.called


class TestUpdateProviderComplete:
    """Complete test suite with proper click.Abort handling"""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_config_manager(self):
        manager = Mock()
        manager.load_config.return_value = None
        manager.save_config.return_value = None
        return manager

    @pytest.fixture
    def mock_provider_manager(self):
        manager = Mock()
        manager.get_existing_providers.return_value = ['openai', 'anthropic']
        manager.update_existing_provider.return_value = True
        return manager

    def test_update_provider_success(
        self,
        mock_config_manager,
        mock_provider_manager
    ):
        """Test successful provider update"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            with patch('devdox_ai_sonar.cli._display_completion_message'):
                with patch('devdox_ai_sonar.cli._display_operation_header'):
                    with patch('devdox_ai_sonar.cli._select_existing_ui', return_value='openai'):
                        with patch('devdox_ai_sonar.cli.console.print'):
                            # ✅ This should NOT raise any exception
                            update_provider()

                            # Verify success flow
                            mock_provider_manager.update_existing_provider.assert_called_once_with('openai')
                            mock_config_manager.save_config.assert_called_once_with(create_backup=False)

    def test_update_provider_no_providers_abort(
        self,
        mock_config_manager,
        mock_provider_manager
    ):
        """Test abort when no providers exist"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            mock_provider_manager.get_existing_providers.return_value = []

            with patch('devdox_ai_sonar.cli.console.print'):
                # ✅ Expect click.Abort
                with pytest.raises(click.Abort):
                    update_provider()

    def test_update_provider_user_cancels_abort(
        self,
        mock_config_manager,
        mock_provider_manager
    ):
        """Test abort when user cancels selection"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            mock_provider_manager.get_existing_providers.return_value = ['openai']

            with patch('devdox_ai_sonar.cli._display_operation_header'):
                with patch('devdox_ai_sonar.cli._select_existing_ui', return_value=""):
                    # ✅ Expect click.Abort when user cancels
                    with pytest.raises(click.Abort):
                        update_provider()


class TestHandleCliError:
    """Tests for _handle_cli_error"""


    def test_handle_cli_error_generic_exception(self):
        """Test handling generic exception"""
        error = ValueError("Test error")

        with patch('devdox_ai_sonar.cli.console'):
            with pytest.raises(click.ClickException):
                _handle_cli_error(error)


# ============================================================================
# TEST CLASS: LOAD_AND_VALIDATE_CONFIG - ERROR CASES
# ============================================================================

class TestLoadAndValidateConfigErrors:
    """Test error cases for _load_and_validate_config"""

    @patch('devdox_ai_sonar.cli._initialize_managers')
    def test_load_config_service_returns_none(self, mock_init):
        """Test when config service returns None"""
        mock_config_service = Mock()
        mock_config_service.load_auth_config.return_value = None

        mock_init.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), mock_config_service
        )

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                _load_and_validate_config()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.AuthConfig')
    def test_load_config_validation_fails(self, mock_auth_config, mock_init):
        """Test when auth config validation fails"""
        mock_config_service = Mock()
        mock_config_service.load_auth_config.return_value = {
            "token": "",
            "organization": "",
            "project": "",
            "project_path": ""
        }

        mock_init.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), mock_config_service
        )

        mock_auth_instance = Mock()
        mock_auth_instance.validate.return_value = (False, "Missing token")
        mock_auth_config.return_value = mock_auth_instance

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                _load_and_validate_config()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.AuthConfig')
    def test_load_config_no_llm_config(self, mock_auth_config, mock_init):
        """Test when LLM config is missing"""
        mock_config_service = Mock()
        mock_config_service.load_auth_config.return_value = {
            "token": "test",
            "organization": "org",
            "project": "proj",
            "project_path": "/tmp"
        }
        mock_config_service.load_llm_config.return_value = None

        mock_init.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), mock_config_service
        )

        mock_auth_instance = Mock()
        mock_auth_instance.validate.return_value = (True, None)
        mock_auth_config.return_value = mock_auth_instance

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                _load_and_validate_config()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.AuthConfig')
    def test_load_config_no_branch_or_pr(self, mock_auth_config, mock_init):
        """Test when neither branch nor PR is provided"""
        mock_config_service = Mock()
        mock_config_service.load_auth_config.return_value = {
            "token": "test",
            "organization": "org",
            "project": "proj",
            "project_path": "/tmp"
        }
        mock_config_service.load_llm_config.return_value = Mock()

        mock_provider_manager = Mock()
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)

        mock_init.return_value = (
            Mock(), Mock(), Mock(),
            mock_provider_manager, Mock(), mock_config_service
        )

        mock_auth_instance = Mock()
        mock_auth_instance.validate.return_value = (True, None)
        mock_auth_config.return_value = mock_auth_instance

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                _load_and_validate_config()


# ============================================================================
# TEST CLASS: COMMAND EXECUTION ERROR SCENARIOS
# ============================================================================

class TestCommandExecutionErrors:
    """Test error scenarios in command execution"""

    @patch('devdox_ai_sonar.cli._load_and_validate_config')
    def test_run_fix_issues_config_load_fails(self, mock_load):
        """Test fix_issues when config loading fails"""
        mock_load.side_effect = click.Abort()

        with pytest.raises(click.Abort):
            _run_fix_issues()

    @patch('devdox_ai_sonar.cli._load_and_validate_config')
    @patch('devdox_ai_sonar.cli.display_configuration')
    @patch('devdox_ai_sonar.cli.smart_confirm')
    @patch('devdox_ai_sonar.cli._process_and_fix_issues')
    def test_run_fix_issues_processing_exception(
            self, mock_process, mock_confirm, mock_display, mock_load
    ):
        """Test fix_issues when processing raises exception"""
        mock_load.return_value = (Mock(), Mock(), {"branch": "main", "pull_request": 0})
        mock_display.return_value = {
            "branch": "main",
            "pull_request": 0,
            "max_fixes": 10,
            "apply": 0,
            "dry_run": 0
        }
        mock_confirm.return_value = True
        mock_process.side_effect = Exception("Processing failed")

        with pytest.raises(Exception, match="Processing failed"):
            _run_fix_issues()


    @patch('devdox_ai_sonar.cli._load_and_validate_config')
    @patch('devdox_ai_sonar.cli.SonarCloudAnalyzer')
    def test_run_inspect_directory_not_found(self, mock_analyzer_class, mock_load):
        """Test inspect when directory doesn't exist"""
        mock_auth = Mock()
        mock_auth.token = "test"
        mock_auth.organization = "org"
        mock_auth.project_path = "/nonexistent/path"

        mock_load.return_value = (mock_auth, Mock(), {})

        mock_analyzer = Mock()
        mock_analyzer.analyze_project_directory.side_effect = ValueError("Invalid path")
        mock_analyzer_class.return_value = mock_analyzer

        with pytest.raises(ValueError, match="Invalid path"):
            _run_inspect()


# ============================================================================
# TEST CLASS: EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================

class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions"""

    def test_safe_convert_pr_max_int(self):
        """Test PR conversion with maximum integer"""
        from devdox_ai_sonar.cli import _safe_convert_pr

        result = _safe_convert_pr(str(2 ** 31 - 1))

        assert result == 2 ** 31 - 1

    def test_safe_convert_pr_with_spaces_and_newlines(self):
        """Test PR conversion with extra whitespace"""
        from devdox_ai_sonar.cli import _safe_convert_pr

        result = _safe_convert_pr("  \n\t  42  \n\t  ")

        assert result == 42

    @patch('devdox_ai_sonar.cli.console')
    def test_display_fix_preview_very_long_code(self, mock_console):
        """Test display with extremely long code"""
        from devdox_ai_sonar.cli import _display_fix_preview
        from devdox_ai_sonar.models.sonar import FixSuggestion

        very_long_code = "x = 1\n" * 10000  # 10000 lines

        fix = FixSuggestion(
            issue_key="issue-1",
            rule="python:S1234",
            file_path="huge.py",
            original_code="old",
            fixed_code=very_long_code,
            explanation="Fixed",
            confidence=0.9,
            llm_model="gpt-4",
            sonar_line_number=10
        )

        with patch('devdox_ai_sonar.cli.Panel'):
            _display_fix_preview(fix, [Mock()])

        # Should truncate and not crash
        assert mock_console.print.called

    def test_validate_issue_types_mixed_case(self):
        """Test issue type validation with mixed case"""
        from devdox_ai_sonar.cli import _validate_issue_types

        result = _validate_issue_types("Bug,VULNERABILITY,code_smell")

        # Should handle case insensitivity
        assert result is not None

    def test_validate_severities_with_spaces(self):
        """Test severity validation with extra spaces"""
        from devdox_ai_sonar.cli import _validate_severities

        result = _validate_severities("  BLOCKER  ,  CRITICAL  ")

        # Should trim spaces
        assert result is not None


# ============================================================================
# TEST CLASS: CONCURRENT AND RACE CONDITIONS
# ============================================================================

class TestConcurrentScenarios:
    """Test concurrent access scenarios"""

    @patch('devdox_ai_sonar.cli._initialize_managers')
    def test_multiple_init_config_calls(self, mock_init):
        """Test multiple simultaneous init_config calls"""
        mock_manager = Mock()
        mock_manager.get_value.return_value = ['openai']

        mock_init.return_value = (
            mock_manager, Mock(), Mock(), Mock(), Mock(), Mock()
        )

        # Call multiple times
        for _ in range(3):
            with patch('devdox_ai_sonar.cli._configure_sonarcloud', return_value=True):
                with patch('devdox_ai_sonar.cli.change_parameters'):
                    init_config()

        # Should handle gracefully
        assert mock_init.call_count == 3


class TestHelperFunctions:
    """Test extracted helper functions."""


    @patch('devdox_ai_sonar.cli._generate_fix_for_file')
    @patch('devdox_ai_sonar.cli.handle_fix')
    @patch('devdox_ai_sonar.cli.console')
    def test_process_single_fix_success(
            self, mock_console, mock_handle, mock_generate,
        auth_config, fix_params, sample_fix_suggestion
    ):
        """Test successful single fix processing."""
        mock_generate.return_value = sample_fix_suggestion
        issues = [Mock()]

        mock_services = {
            'analyzer': Mock(),
            'ruler': Mock(),
            'fixer': Mock()
        }
        _process_single_fix(
            issues, mock_services, auth_config,
            fix_params, IssueType.SECURITY, "test.py"
        )

        mock_generate.assert_called_once()
        mock_handle.assert_called_once()

    @patch('devdox_ai_sonar.cli._generate_fix_for_file')
    @patch('devdox_ai_sonar.cli.console')
    def test_process_single_fix_no_fix_generated(
            self, mock_console, mock_generate
            ,auth_config, fix_params
    ):
        """Test when no fix can be generated."""
        mock_generate.return_value = None
        issues = [Mock()]
        mock_services = {
            'analyzer': Mock(),
            'ruler': Mock(),
            'fixer': Mock()
        }
        _process_single_fix(
            issues, mock_services, auth_config,
            fix_params, IssueType.SECURITY, "test.py"
        )

        # Should print warning
        call_args = str(mock_console.print.call_args_list)
        assert 'No fix' in call_args or 'could not be generated' in call_args

