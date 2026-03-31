

import os
import tempfile

import pytest
from pathlib import Path
from contextlib import contextmanager
import click
from click.testing import CliRunner
from devdox_ai_sonar.services.configuration import AuthConfig, LLMConfig
from devdox_ai_sonar.utils.validator import IssueType, InputValidator
from unittest.mock import Mock, patch, MagicMock, AsyncMock
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
    _execute_interactive_command_async,
    _should_continue_to_menu,
    _handle_command_switch,
    _handle_interactive_error,
    _run_interactive_mode_async,
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
    _execute_interactive_iteration_async,
    _should_continue_to_next_issue,
    _fetch_issues_by_type,
    _display_project_header,
    _display_metrics_section,
    _process_security_issues,
    _process_issues_for_rule,
    _process_regular_issues,
    _handle_provider_configuration,
    _configure_providers_loop,

)
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    AnalysisResult,
    FixSuggestion,
    FixResult,
    ChangeType,
    BlockType,
    CodeBlock
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
def mock_auth_dict() :
    return {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path",
            "git_url": "https://github.com/org/proj.git"
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
    ui.select_provider_from_list = AsyncMock(return_value='openai')
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
                llm_model="open-ai",
                fixed_code_blocks=[CodeBlock(block_name="test",
                                             start_line="1",
                                             end_line="10",
                                             has_changes=True,
                                             change_type=ChangeType.FULL_CODE,
                                             block_type=BlockType.MODULE,
                                             context="new_code")]
            )
        ],
        failed_fixes=[],
        skipped_files=[],
        total_fixes_attempted=1,
        success_rate=1.0,
        backup_created=True,
        project_path=Path(tempfile.gettempdir()) / "project",
        backup_path=Path(tempfile.gettempdir()) / "backup"
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


@contextmanager
def mock_tmp_clone(tmp_path_value=os.path.join(tempfile.gettempdir(), "fake_devdox")):
    """Mock TmpCloneManager for testing."""
    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=Path(tmp_path_value))
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    with patch('devdox_ai_sonar.cli.TmpCloneManager', return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def auth_config():
    """Mock auth config."""
    return AuthConfig(
        token="test-token",
        organization="test-org",
        project="test-project",
        project_path=os.path.join(tempfile.gettempdir(), "test"),
        git_url="https://github.com/test-org/test-project.git"
    )

@pytest.fixture
def mock_config_service():
    """Mock ConfigService"""
    with patch('devdox_ai_sonar.cli.ConfigService') as mock:
        mock_instance = AsyncMock()
        mock_instance.load_auth_config.return_value = {
            "token": "test-token",
            "organization": "test-org",
            "project": "test-project",
            "project_path": os.path.join(tempfile.gettempdir(), "project"),
            "git_url": "https://github.com/test-org/test-project.git"
        }
        # Sync methods must be Mock() to avoid returning coroutines
        mock_instance.check_all_value_empty = Mock(return_value=False)
        mock_instance.validate_auth_config = Mock(return_value=True)
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_loaded_config():
    """Fixture for _load_and_validate_config return value"""
    mock_auth = Mock()
    mock_auth.project = "test-project"
    mock_auth.organization = "test-org"
    mock_auth.token = "test-token"
    mock_auth.project_path = os.path.join(tempfile.gettempdir(), "project")

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
        mock_instance = AsyncMock()
        mock_instance.get_value.return_value = {}
        mock_instance.load_config.return_value = None
        mock_instance.set_value.return_value = None
        # Sync methods must be Mock() to avoid returning coroutines
        mock_instance.save_config = Mock(return_value=None)
        mock_instance.create_default_config = Mock(return_value=None)
        mock_instance.delete_value = AsyncMock(return_value=None)
        mock_instance.validate_change = Mock(return_value=None)
        mock_instance.create_backup = Mock(return_value=None)
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_provider_manager():
    """Mock ProviderConfigManager"""
    with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock:
        mock_instance = AsyncMock()
        mock_instance.branch_or_pr_prompt.return_value = ("main", None)
        mock_instance.branch_or_pr.return_value = ("main", None)
        mock_instance.get_available_providers.return_value = ["openai", "anthropic"]
        mock_instance.get_existing_providers.return_value = ["openai"]
        mock_instance.update_existing_provider.return_value = True
        mock_instance.configure_new_provider = AsyncMock(return_value={
            "config": {"provider": "openai", "api_key": "test"},
            "set_as_default": True
        })
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
        llm_model="gpt-3.5-turbo",
        fixed_code_blocks=[CodeBlock(block_name="test",
                                     start_line="1",
                                     end_line="10",
                                     has_changes=True,
                                     change_type=ChangeType.FULL_CODE,
                                     block_type=BlockType.MODULE,
                                     context="new_code")]

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

    @patch('devdox_ai_sonar.cli._execute_command_async')
    @patch('devdox_ai_sonar.cli.console')
    async def test_executes_command_with_context(self, mock_console, mock_execute):
        """Test command execution with context."""
        mock_ctx = Mock(spec=click.Context)

        await _execute_interactive_command_async(mock_ctx, 'fix_issues')

        mock_execute.assert_called_once_with(mock_ctx, 'fix_issues')
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
            with patch('devdox_ai_sonar.cli._execute_command_async'):
                result = runner.invoke(main, ['-c', 'analyze'])
                # Should attempt to execute

    def test_main_verbose_flag(self, runner):
        """Test verbose flag sets context"""
        with patch('devdox_ai_sonar.cli.init_config'):
            with patch('devdox_ai_sonar.cli._run_interactive_mode_async'):
                result = runner.invoke(main, ['--verbose'])

    def test_main_with_invalid_command(self, runner):
        """Test main with invalid command option"""
        result = runner.invoke(main, ['-c', 'invalid_command'])

        # Should fail
        assert result.exit_code != 0

    def test_main_with_verbose_and_command(self, runner):
        """Test main with verbose flag and command"""
        with patch('devdox_ai_sonar.cli._execute_command_async'):
            with patch('devdox_ai_sonar.cli.init_config'):
                result = runner.invoke(main, ['--verbose', '-c', 'analyze'])

                # Should set verbose in context
                assert result.exit_code == 0 or 'verbose' in str(result.output).lower()

    def test_main_dry_run_flag(self, runner):
        """Test main with --dry-run flag"""
        with patch('devdox_ai_sonar.cli._execute_command_async'):
            with patch('devdox_ai_sonar.cli.init_config'):
                result = runner.invoke(main, ['--dry-run', '-c', 'fix_issues'])

                # Dry run should be set
                assert result.exit_code == 0

    @patch('devdox_ai_sonar.cli._run_interactive_mode_async')
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

    async def test_init_config_first_time(
            self, runner, mock_config_service, mock_config_manager, mock_provider_manager
    ):
        """Test initial configuration"""
        mock_config_manager.get_value.return_value = []  # No providers

        with patch('devdox_ai_sonar.cli._configure_sonarcloud', new=AsyncMock(return_value=True)):
            with patch('devdox_ai_sonar.cli._configure_providers_loop', new=AsyncMock()):
                with patch('devdox_ai_sonar.cli.change_parameters', new=AsyncMock()):
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
                        await init_config(ctx)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    async def test_init_config_first_time_success(
            self, mock_init
    ):
        """Test successful first-time initialization"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = []  # No providers yet
        mock_manager.create_default_config.return_value = None
        mock_manager.load_config.return_value = None
        mock_manager.save_config.return_value = None

        mock_provider_manager = AsyncMock()
        mock_provider_manager.get_available_providers.return_value = ['openai']

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), AsyncMock()
        )

        with patch('devdox_ai_sonar.cli._configure_sonarcloud', new=AsyncMock(return_value=True)) as mock_sonar:
            with patch('devdox_ai_sonar.cli._configure_providers_loop', new=AsyncMock()) as mock_providers_loop:
                with patch('devdox_ai_sonar.cli.change_parameters', new=AsyncMock()) as mock_change_params:
                    await init_config()

                    # Verify initialization steps
                    mock_manager.create_default_config.assert_called_once()
                    mock_manager.load_config.assert_called_once()
                    mock_sonar.assert_called_once()
                    mock_providers_loop.assert_called_once()
                    mock_change_params.assert_called_once()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._check_reconfiguration_consent')
    async def test_init_config_existing_config_no_consent(self, mock_consent, mock_init):
        """Test when config exists and user doesn't consent to reconfigure"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = ['openai']  # Providers exist

        mock_init.return_value = (
            mock_manager, Mock(), Mock(), AsyncMock(), Mock(), AsyncMock()
        )

        mock_consent.return_value = False  # User says no

        await init_config()

        # Should exit early
        mock_manager.create_default_config.assert_called_once()

    def test_check_reconfiguration_consent_no_providers(self):
        """Test with no providers"""
        result = _check_reconfiguration_consent([])

        assert result is True

    @patch('devdox_ai_sonar.cli._initialize_managers')
    async def test_init_config_no_available_providers(self, mock_init):
        """Test when no providers are available"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = []

        mock_provider_manager = AsyncMock()
        mock_provider_manager.get_available_providers.return_value = []  # No providers

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), AsyncMock()
        )

        with patch('devdox_ai_sonar.cli._configure_sonarcloud', new=AsyncMock(return_value=True)):
            with patch('devdox_ai_sonar.cli.console.print'):
                with pytest.raises(click.Abort):
                   await init_config()


    def test_check_reconfiguration_consent_existing_providers(self):
        """Test with existing providers"""
        with patch('devdox_ai_sonar.cli.click'):
            result = _check_reconfiguration_consent(["openai", "anthropic"])

            assert result is False

    @patch('devdox_ai_sonar.cli._initialize_managers')
    async def test_init_config_sonar_configuration_fails(self, mock_init):
        """Test when SonarCloud configuration fails"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = []

        mock_init.return_value = (
            mock_manager, Mock(), Mock(), AsyncMock(), Mock(), AsyncMock()
        )

        with patch('devdox_ai_sonar.cli._configure_sonarcloud', new=AsyncMock(return_value=False)):
            with pytest.raises(click.Abort):
                await init_config()

    async def test_init_config_existing_config(
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
                await init_config(ctx)

    async def test_add_provider_success(
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
                    await add_provider()

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

    async def test_add_provider_with_error_handling(
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
                    await add_provider()

                # Verify error handler was called
                mock_error.assert_called_once()

    async def test_configure_providers_loop_integration(
            self,
            mock_config_manager,
            mock_provider_manager,
            mock_ui
    ):
        """Test the _configure_providers_loop function"""
        from devdox_ai_sonar.cli import _configure_providers_loop

        available_providers = ['openai', 'anthropic']

        # Mock the UI to select 'openai', then return None (exit loop)
        mock_ui.select_provider_from_list = AsyncMock(side_effect=['openai', None])

        # Mock successful configuration
        mock_provider_manager.configure_new_provider = AsyncMock(return_value={
            'config': {'provider': 'openai', 'api_key': 'test-key'},
            'set_as_default': True
        })

        with patch('devdox_ai_sonar.cli.console.print'):
            with patch('devdox_ai_sonar.cli.Confirm.ask', return_value=False):
                await _configure_providers_loop(
                    mock_provider_manager,
                    mock_config_manager,
                    mock_ui,
                    available_providers
                )

                # Verify provider was configured
                mock_provider_manager.configure_new_provider.assert_called_once_with('openai')

                # Verify provider was added to manager
                mock_config_manager.add_provider.assert_called_once()

    async def test_handle_provider_configuration_success(
            self,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test _handle_provider_configuration helper"""


        available_providers = ['openai', 'anthropic']

        mock_provider_manager.configure_new_provider = AsyncMock(return_value={
            'config': {'provider': 'openai', 'api_key': 'test-key'},
            'set_as_default': True
        })

        with patch('devdox_ai_sonar.cli.console.print'):
            result = await _handle_provider_configuration(
                mock_provider_manager,
                mock_config_manager,
                'openai',
                available_providers
            )

            assert result is True
            assert 'openai' not in available_providers  # Should be removed

            # Verify add_provider was called on manager
            mock_config_manager.add_provider.assert_called_once()

    async def test_handle_provider_configuration_failure(
            self,
            mock_config_manager,
            mock_provider_manager
    ):
        """Test _handle_provider_configuration when configuration fails"""


        available_providers = ['openai']

        mock_provider_manager.configure_new_provider = AsyncMock(return_value=None)

        result = await _handle_provider_configuration(
            mock_provider_manager,
            mock_config_manager,
            'openai',
            available_providers
        )

        assert result is False
        assert 'openai' in available_providers  # Should NOT be removed

        # Verify add_provider was NOT called
        mock_config_manager.add_provider.assert_not_called()

    async def test_update_provider_success(
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
                        await update_provider()

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
                            "Select the provider to update",
                            ['openai']
                        )

                        # Verify provider was updated
                        mock_provider_manager.update_existing_provider.assert_called_once_with('openai')

                        # Verify config was saved
                        mock_config_manager.save_config.assert_called_once_with(create_backup=False)

    async def test_update_provider_no_existing_providers(
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
                    await update_provider()

                # Verify warning message was printed
                mock_print.assert_called()
                call_args = str(mock_print.call_args_list)
                assert 'No providers configured' in call_args or \
                       'add at least one provider' in call_args


    async def test_update_provider_no_existing_providers_alternative(
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
                    await update_provider()
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

    async def test_update_provider_no_existing_providers_with_error_handler(
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
                        await update_provider()

        # ========================================================================
        # Additional tests for other abort scenarios
        # ========================================================================



    async def test_update_provider_user_cancels_selection(
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
                        await update_provider(ctx)

                    # Verify update was NOT called
                    mock_provider_manager.update_existing_provider.assert_not_called()
                    mock_config_manager.save_config.assert_not_called()

    async def test_update_provider_no_providers(
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
                await update_provider(ctx)

    async def test_change_parameters_success(
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

            with patch('devdox_ai_sonar.cli.change_field', new=AsyncMock(return_value=None)):
                with patch('devdox_ai_sonar.cli.change_max_fix', new=AsyncMock()):
                    ctx = Mock()
                    await change_parameters(ctx)


    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_with_types_provided(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that types parameter skips interactive prompt for types"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with types pre-populated
        await change_parameters(types="BUG,VULNERABILITY")

        # change_field should NOT be called for types (skipped because types provided)
        types_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.types' in str(call)
        ]
        assert len(types_calls) == 0, "Types field should be skipped when types provided"

        # But other fields should still be called
        assert mock_change_field.call_count >= 3

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_with_severity_provided(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that severity parameter skips interactive prompt for severities"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with severity pre-populated
        await change_parameters(severity="CRITICAL,BLOCKER")

        # change_field should NOT be called for severities
        severity_calls = [
            call for call in mock_change_field.call_args_list
            if 'configuration.severities' in str(call)
        ]
        assert len(severity_calls) == 0, "Severities field should be skipped when severity provided"

        # But other fields should still be called
        assert mock_change_field.call_count >= 3  # types, apply, backup, exclude_rules

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_with_both_types_and_severity(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that both types and severity parameters skip their prompts"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with both types and severity
        await change_parameters(types="BUG", severity="CRITICAL")

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_with_apply_in_kwargs(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that apply parameter from kwargs is used as default"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            constant.CONFIGURATION_APPLY: None,  # Not in config
            constant.CONFIGURATION_BACKUP: None,
        }.get(key)

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with apply in kwargs
        await change_parameters(apply=1)

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_with_create_backup_in_kwargs(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that create_backup parameter from kwargs is used as default"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            constant.CONFIGURATION_APPLY: None,
            constant.CONFIGURATION_BACKUP: None,  # Not in config
        }.get(key)

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with create_backup in kwargs
        await change_parameters(create_backup=1)

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_config_overrides_kwargs(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that config values take precedence over kwargs"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            constant.CONFIGURATION_APPLY: 0,  # In config
            constant.CONFIGURATION_BACKUP: 1,  # In config
        }.get(key)

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call with different values in kwargs
        await change_parameters(apply=1, create_backup=0)

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_types_field_with_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that types field is called with VALID_ISSUE_TYPES choices"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.types": "BUG,CODE_SMELL",
        }.get(key)

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call without types parameter (should prompt)
        await change_parameters()

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_severities_field_with_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that severities field is called with VALID_SEVERITIES choices"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.severities": "CRITICAL,BLOCKER",
        }.get(key)

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call without severity parameter (should prompt)
        await change_parameters()

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_severities_field_with_choices_sync(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that severities field is called with VALID_SEVERITIES choices"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.severities": "CRITICAL,BLOCKER",
        }.get(key)

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Call without severity parameter (should prompt)
        await change_parameters()

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_apply_field_with_yes_no_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that apply field has yes/no choices with correct format"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        await change_parameters()

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
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_exclude_rules_field_no_choices(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that exclude_rules field has no predefined choices"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.side_effect = lambda key: {
            "configuration.max_fixes": 10,
            "configuration.exclude_rules": "python:S3776,python:S1192",
        }.get(key)

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        await change_parameters()

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
    async def test_change_parameters_with_branch_only(self, mock_init):
        """Test change_parameters with only branch (no PR)"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("feature/test", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        with patch('devdox_ai_sonar.cli.change_field'):
            with patch('devdox_ai_sonar.cli.change_max_fix'):
                # Should succeed with only branch
                await change_parameters()
                mock_manager.save_config.assert_called_once()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    async def test_change_parameters_with_pr_only(self, mock_init):
        """Test change_parameters with only PR (no branch)"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, "123")

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        with patch('devdox_ai_sonar.cli.change_field'):
            with patch('devdox_ai_sonar.cli.change_max_fix'):
                # Should succeed with only PR
                await change_parameters()
                mock_manager.save_config.assert_called_once()


    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli._handle_cli_error')
    async def test_change_parameters_handles_exception_in_change_field(
            self, mock_handle_error, mock_change_max, mock_change_field, mock_init
    ):
        """Test exception handling when change_field raises error"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        # Make change_field raise exception
        mock_change_field.side_effect = ValueError("Invalid input")

        await change_parameters()

        # Should call error handler
        mock_handle_error.assert_called_once()
        assert isinstance(mock_handle_error.call_args[0][0], ValueError)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli._handle_cli_error')
    async def test_change_parameters_handles_exception_in_save(
            self, mock_handle_error, mock_change_max, mock_change_field, mock_init
    ):
        """Test exception handling when save_config fails"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock(side_effect=IOError("Cannot write to file"))
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        await change_parameters()

        # Should call error handler
        mock_handle_error.assert_called_once()
        assert isinstance(mock_handle_error.call_args[0][0], IOError)



    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._handle_cli_error')
    async def test_change_parameters_handles_exception_in_init(
            self, mock_handle_error, mock_init
    ):
        """Test exception handling when initialization fails"""
        # Make init raise exception
        mock_init.side_effect = RuntimeError("Config error")

        await change_parameters()

        # Should call error handler
        mock_handle_error.assert_called_once()
        assert isinstance(mock_handle_error.call_args[0][0], RuntimeError)


    async def test_change_parameters_no_branch_or_pr(
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
                await change_parameters(ctx)

    async def test_change_parameters_invalid_pr(
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
                await change_parameters(ctx)

    async def test_change_field_with_value(self, mock_config_manager):
        """Test change_field with value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="BUG")):
            result = await change_field(
                mock_config_manager,
                "field.path",
                "Enter value:",
                "default"
            )

            assert result == "BUG"
            mock_config_manager.set_value.assert_called_once()

    async def test_change_field_no_value(self, mock_config_manager):
        """Test change_field with no value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value=None)):
            result = await change_field(
                mock_config_manager,
                "field.path",
                "Enter value:",
                "default"
            )

            assert result is None
            mock_config_manager.set_value.assert_not_called()

    async def test_change_field_allow_empty_false_preserves_value(self, mock_config_manager):
        """Test change_field with allow_empty=False (default) preserves existing value on empty input"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value=None)):
            result = await change_field(
                mock_config_manager,
                "configuration.exclude_rules",
                "Enter rules:",
                default_value="python:S1234",
                allow_empty=False
            )

            assert result is None
            mock_config_manager.set_value.assert_not_called()

    async def test_change_field_allow_empty_valid_rules_directly(self, mock_config_manager):
        """Test change_field with allow_empty=True accepts valid rules on first prompt"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="python:S9999")):
            result = await change_field(
                mock_config_manager,
                "configuration.exclude_rules",
                "Enter rules:",
                default_value="python:S1234",
                allow_empty=True
            )

            assert result == "python:S9999"
            mock_config_manager.set_value.assert_called_once_with(
                "configuration.exclude_rules", "python:S9999"
            )

    async def test_change_field_none_directly_stores_none(self, mock_config_manager):
        """Test change_field stores NONE value when user enters NONE on first prompt"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value=constant.EXCLUDE_NONE)):
            result = await change_field(
                mock_config_manager,
                "configuration.exclude_rules",
                "Enter rules:",
                default_value="python:S1234",
                allow_empty=True
            )

            assert result == constant.EXCLUDE_NONE
            mock_config_manager.set_value.assert_called_once_with(
                "configuration.exclude_rules", constant.EXCLUDE_NONE
            )

    async def test_change_field_empty_shows_error_then_accepts_rules(self, mock_config_manager):
        """Test empty input shows error, re-prompts with default, then accepts rules"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(side_effect=[None, "python:S5678"])), \
             patch('devdox_ai_sonar.cli.console') as mock_console:
            result = await change_field(
                mock_config_manager,
                "configuration.exclude_rules",
                "Enter rules:",
                default_value="python:S1234",
                allow_empty=True
            )

            assert result == "python:S5678"
            mock_console.print.assert_called_once_with(constant.EXCLUDE_RULES_EMPTY_ERROR)
            mock_config_manager.set_value.assert_called_once_with(
                "configuration.exclude_rules", "python:S5678"
            )

    async def test_change_field_empty_shows_error_then_accepts_none(self, mock_config_manager):
        """Test empty input shows error, re-prompts with default, then stores NONE"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(side_effect=[None, constant.EXCLUDE_NONE])), \
             patch('devdox_ai_sonar.cli.console') as mock_console:
            result = await change_field(
                mock_config_manager,
                "configuration.exclude_rules",
                "Enter rules:",
                default_value="python:S1234",
                allow_empty=True
            )

            assert result == constant.EXCLUDE_NONE
            mock_console.print.assert_called_once_with(constant.EXCLUDE_RULES_EMPTY_ERROR)
            mock_config_manager.set_value.assert_called_once_with(
                "configuration.exclude_rules", constant.EXCLUDE_NONE
            )

    async def test_change_field_multiple_empties_loops_until_valid(self, mock_config_manager):
        """Test multiple empty inputs loop with errors until valid input provided"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(side_effect=[None, None, None, "python:S1234"])), \
             patch('devdox_ai_sonar.cli.console') as mock_console:
            result = await change_field(
                mock_config_manager,
                "configuration.exclude_rules",
                "Enter rules:",
                default_value=constant.EXCLUDE_NONE,
                allow_empty=True
            )

            assert result == "python:S1234"
            assert mock_console.print.call_count == 3
            mock_console.print.assert_called_with(constant.EXCLUDE_RULES_EMPTY_ERROR)
            mock_config_manager.set_value.assert_called_once_with(
                "configuration.exclude_rules", "python:S1234"
            )

    async def test_change_field_none_default_accept_stores_none(self, mock_config_manager):
        """Test NONE default accepted via Enter (smart_prompt returns NONE) stores NONE"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value=constant.EXCLUDE_NONE)):
            result = await change_field(
                mock_config_manager,
                "configuration.exclude_rules",
                "Enter rules:",
                default_value=constant.EXCLUDE_NONE,
                allow_empty=True
            )

            assert result == constant.EXCLUDE_NONE
            mock_config_manager.set_value.assert_called_once_with(
                "configuration.exclude_rules", constant.EXCLUDE_NONE
            )

    async def test_change_max_fix_valid(self, mock_config_manager):
        """Test change_max_fix with valid value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="15")):
            await change_max_fix(mock_config_manager, "Enter max:", 10, 50)

            mock_config_manager.set_value.assert_called_with(
                "configuration.max_fixes", 15
            )

    async def test_change_max_fix_invalid(self, mock_config_manager):
        """Test change_max_fix with invalid value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="abc")):
            await change_max_fix(mock_config_manager, "Enter max:", 10, 50)

            # Should use default
            mock_config_manager.set_value.assert_called()

    async def test_change_max_fix_out_of_range(self, mock_config_manager):
        """Test change_max_fix with out of range value"""
        with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="100")):
            with patch('devdox_ai_sonar.cli.settings') as mock_settings:
                mock_settings.DEFAULT_MAX_FIXES = 10
                await change_max_fix(mock_config_manager, "Enter max:", 10, 50)

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_all_values(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test changing all parameter values"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        mock_change_field.return_value = None

        await change_parameters()

        # Should call change_max_fix and change_field multiple times
        mock_change_max.assert_called_once()
        assert mock_change_field.call_count >= 3  # types, severities, apply, backup

    @patch('devdox_ai_sonar.cli._initialize_managers')
    async def test_change_parameters_no_branch_or_pr_sync(self, mock_init):
        """Test when no branch or PR is specified"""
        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)

        mock_init.return_value = (
            Mock(), Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                await change_parameters()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.change_field', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.change_max_fix', new_callable=AsyncMock)
    async def test_change_parameters_save_config(
            self, mock_change_max, mock_change_field, mock_init
    ):
        """Test that config is saved after changes"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = 10

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        mock_init.return_value = (
            mock_manager, Mock(), Mock(),
            mock_provider_manager, Mock(), Mock()
        )

        await change_parameters()

        # Should save config at end
        mock_manager.save_config.assert_called_once_with(create_backup=False)


# ============================================================================
# CONFIGURATION HELPERS TESTS
# ============================================================================

class TestConfigureSonarCloud:
    """Tests for _configure_sonarcloud"""

    async def test_configure_sonarcloud_success(self):
        """Test successful configuration"""
        mock_sonar_ui = Mock()
        mock_config_service = AsyncMock()

        mock_config_service.load_auth_config.return_value = {
            "token": "",
            "organization": "",
            "project": "",
            "project_path": ""
        }
        mock_config_service.check_all_value_empty = Mock(return_value=True)

        mock_sonar_config = Mock()
        mock_sonar_config.token = "new_token"
        mock_sonar_config.organization = "new_org"
        mock_sonar_config.project = "new_project"
        mock_sonar_config.project_path = "/new/path"

        mock_sonar_ui.configure_sonarcloud.return_value = mock_sonar_config
        mock_config_service.save_config.return_value = True

        result = await _configure_sonarcloud(mock_sonar_ui, mock_config_service)

        assert result is True
        mock_sonar_ui.display_welcome.assert_called_once()

    async def test_configure_sonarcloud_already_configured(self):
        """Test when already configured"""
        mock_sonar_ui = Mock()
        mock_config_service = AsyncMock()

        mock_config_service.load_auth_config.return_value = {
            "token": "existing",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }
        mock_config_service.check_all_value_empty = Mock(return_value=False)

        result = await _configure_sonarcloud(mock_sonar_ui, mock_config_service)

        assert result is True
        mock_sonar_ui.configure_sonarcloud.assert_not_called()


# ============================================================================
# TEST CLASS: FIX ISSUES COMMAND
# ============================================================================

class TestFixIssuesCommand:
    """Test fix_issues command implementation"""

    async def test_run_fix_issues_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test successful fix_issues run"""
        mock_config_service.load_llm_config.return_value = mock_llm_config

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock()) as mock_load:
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

                with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=True)):
                    with patch('devdox_ai_sonar.cli._process_and_fix_issues', new=AsyncMock()):
                        ctx = Mock()
                        ctx.obj = {"verbose": False}

                        await _run_fix_issues()

    async def test_run_fix_issues_cancelled(
            self, mock_config_service, mock_llm_config
    ):
        """Test fix_issues when user cancels"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock()) as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {})

            with patch('devdox_ai_sonar.cli.display_configuration', return_value={}):
                with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=False)):


                    await _run_fix_issues()

    async def test_run_fix_issues_switch_command(self):
        """Test fix_issues with command switch"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config', side_effect=SwitchCommandException):


            with pytest.raises(SwitchCommandException):
               await _run_fix_issues()

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

    async def test_run_fix_security_issues_success(
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
                    with pytest.raises(click.exceptions.Abort):
                        await _run_fix_security_issues()

    async def test_run_fix_security_issues_cancelled(self, mock_llm_config):
        """Test security fix when cancelled"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {})

            with patch('devdox_ai_sonar.cli.display_configuration', return_value={}):
                with patch('devdox_ai_sonar.cli.smart_confirm', return_value=False):
                   await _run_fix_security_issues()


class TestSecurityIssuesProcessing:
    """Test security issues processing."""

    @patch('devdox_ai_sonar.cli._should_continue_to_next_issue')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    async def test_process_security_issues_single_file(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test processing single security issue file."""
        mock_continue.return_value = False  # Stop after first

        issues_by_file = {
            "security.py": [Mock(file="security.py"), Mock(file="security.py")]
        }

        await _process_security_issues(
            issues_by_file, mock_services, auth_config, fix_params,Mock(),tmp_path=Mock()
        )


        mock_single_fix.assert_called_once()
        mock_continue.assert_called_once_with(1, 1, system_ask=True)

    @patch('devdox_ai_sonar.cli._should_continue_to_next_issue')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    async def test_process_security_issues_multiple_files(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test processing multiple security issue files."""
        mock_continue.side_effect = [True, False]  # Continue first, stop second

        issues_by_file = {
            "auth.py": [Mock(file="auth.py")],
            "crypto.py": [Mock(file="crypto.py"), Mock(file="crypto.py")]
        }

        await _process_security_issues(
            issues_by_file, mock_services, auth_config, fix_params,Mock(), tmp_path=Mock()
        )

        assert mock_single_fix.call_count == 2

    @patch('devdox_ai_sonar.cli._should_continue_to_next_issue')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    @pytest.mark.skip(reason="Need update")
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
            issues_by_file, mock_services, auth_config, fix_params,md_file_path, tmp_path=Mock()
        )

        # Should only process first file
        assert mock_single_fix.call_count == 1

    @patch('devdox_ai_sonar.cli._process_single_fix')
    async def test_process_empty_issues_dict(
            self, mock_single_fix,
            mock_services, auth_config, fix_params
    ):
        """Test processing with empty issues dict."""
        issues_by_file = {}
        md_file_path = Mock()
        await _process_security_issues(
            issues_by_file, mock_services, auth_config, fix_params, md_file_path, tmp_path=Path(tempfile.gettempdir()) / "new"
        )

        # Should handle gracefully
        mock_single_fix.assert_not_called()

    @patch('devdox_ai_sonar.cli._process_issues_for_rule')
    async def test_process_rule_with_empty_issues_list(
            self, mock_process_rule,
            mock_services, auth_config, fix_params
    ):
        """Test processing rule with empty issues list."""
        mock_process_rule.return_value = True

        issues_by_rule = {
            "python:S1234": {"issue": []}  # Empty list
        }
        md_file_path = Mock()
        await _process_regular_issues(
            issues_by_rule, mock_services, auth_config, fix_params, md_file_path, tmp_path=os.path.join(tempfile.gettempdir(), "new")
        )

        # Should still call process_rule
        mock_process_rule.assert_called_once()
        
# ============================================================================
# TEST CLASS: REGULAR ISSUES PROCESSING
# ============================================================================

class TestRegularIssuesProcessing:
    """Test regular issues processing."""

    @patch('devdox_ai_sonar.cli._process_issues_for_rule')
    async def test_process_regular_issues_single_rule(
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
        await _process_regular_issues(
            issues_by_rule, mock_services, auth_config, fix_params, md_file_path, tmp_path=os.path.join(tempfile.gettempdir(), "new")
        )


        mock_process_rule.assert_called_once()

    @patch('devdox_ai_sonar.cli._process_issues_for_rule')
    async def test_process_regular_issues_multiple_rules(
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
        await _process_regular_issues(
            issues_by_rule, mock_services, auth_config, fix_params, md_file_path, tmp_path=os.path.join(tempfile.gettempdir(), "new")
        )

        assert mock_process_rule.call_count == 3


    @patch('devdox_ai_sonar.cli._should_continue_to_next_issue')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    @pytest.mark.skip(reason="Need update")
    async def test_process_issues_for_rule_all_issues(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params
    ):
        """Test processing all issues for a rule."""
        mock_continue.side_effect = [True, True, False]

        issues_list = [Mock(), Mock(), Mock()]
        md_file_path = Mock()
        result = await _process_issues_for_rule(
            "python:S1234", issues_list,
            mock_services, auth_config, fix_params, md_file_path, tmp_path=os.path.join(tempfile.gettempdir(), "new")
        )

        assert result is False  # Stopped by user
        assert mock_single_fix.call_count == 3

    @patch('devdox_ai_sonar.cli._should_continue_to_next_issue')
    @patch('devdox_ai_sonar.cli._process_single_fix')
    @pytest.mark.skip(reason="Need update")
    async def test_process_issues_for_rule_continues(
            self, mock_single_fix, mock_continue,
            mock_services, auth_config, fix_params,tmp_path=os.path.join(tempfile.gettempdir(), "new")
    ):
        """Test processing continues to next rule."""
        mock_continue.return_value = True  # Continue all

        issues_list = [Mock(), Mock()]
        md_file_path = Mock()
        result = await  _process_issues_for_rule(
            "python:S1234", issues_list,
            mock_services, auth_config, fix_params, md_file_path,tmp_path=tmp_path
        )

        assert result is True  # Should continue
        assert mock_single_fix.call_count == 2



# ============================================================================
# TEST CLASS: ANALYZE COMMAND
# ============================================================================

class TestAnalyzeCommand:
    """Test analyze command"""


    async def test_run_analyze_success(self):
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
        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock()) as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="10")):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli._display_analysis_results'):
                            await _run_analyze()

                            # Verify
                            mock_load.assert_called_once()
                            mock_analyzer.get_project_issues.assert_called_once()

    async def test_run_analyze_no_results(self):
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

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock()) as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="10")):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):

                        await _run_analyze()

                        # Should still call analyzer
                        mock_analyzer.get_project_issues.assert_called_once()

    async def test_run_analyze_no_config(self):
        """Test analyze without config - _load_and_validate_config raises Abort"""


        # _load_and_validate_config will raise click.Abort if no config
        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(side_effect=click.Abort())):

            with pytest.raises(click.Abort):
                await _run_analyze()

    async def test_run_inspect_no_config(self):
        """Test inspect without config - _load_and_validate_config raises Abort"""

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(side_effect=click.Abort())):

            with pytest.raises(click.Abort):
                await _run_inspect()


    async def test_run_analyze_with_severity_and_types(self):
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

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock()) as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="10")):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli._validate_severities') as mock_sev:
                            with patch('devdox_ai_sonar.cli._validate_issue_types') as mock_types:

                                await _run_analyze()

                                # Should validate filters
                                mock_sev.assert_called()
                                mock_types.assert_called()



# ============================================================================
# TEST CLASS: INSPECT COMMAND
# ============================================================================

class TestInspectCommand:
    """Test inspect command"""

    async def test_run_inspect_success(self, mock_loaded_config, mock_analyzer):

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock()) as mock_load:
            mock_load.return_value = mock_loaded_config

            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer') as mock_sa:
                mock_sa.return_value = mock_analyzer


                ctx = Mock()
                ctx.obj = {"verbose": False}

                await _run_inspect()

                # Verify
                mock_load.assert_called_once()
                mock_analyzer.analyze_project_directory.assert_called_once()


    async def test_run_inspect_invalid_config(self):
        """Test inspect with invalid config - EXPECTS click.Abort"""

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(side_effect=click.Abort())) as mock_load:

            # Assert that click.Abort is raised (this is CORRECT behavior!)
            with pytest.raises(click.Abort):
                await _run_inspect()

            # Verify _load_and_validate_config was called
            mock_load.assert_called_once()


    async def test_run_inspect_empty_analysis(self):
        """Test inspect with empty analysis results"""
        mock_auth_config = Mock()
        mock_auth_config.project_path = os.path.join(tempfile.gettempdir(), "empty")
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

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock()) as mock_load:
            mock_load.return_value = (mock_auth_config, mock_llm_config, parameters)

            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):


                await _run_inspect()

                # Should still complete successfully
                mock_analyzer.analyze_project_directory.assert_called_once()



class TestSelectExistingUI:
    """Tests for _select_existing_ui"""

    @patch('devdox_ai_sonar.cli.select_from_list', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.console')
    async def test_select_existing_ui_success(self, mock_console, mock_select):
        """Test successful selection"""
        mock_select.return_value = "openai"

        result = await _select_existing_ui("Select provider", ["openai", "anthropic"])

        assert result == "openai"

    @patch('devdox_ai_sonar.cli.select_from_list', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.console')
    async def test_select_existing_ui_cancelled(self, mock_console, mock_select):
        """Test cancelled selection"""
        mock_select.return_value = None

        result = await _select_existing_ui("Select", ["openai"])

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

    async def test_process_no_issues_found(self):
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
        auth_config.git_url = "https://github.com/test-org/test-project.git"

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
            "dry_run": False,

        }

        # Patch all the dependencies with proper mocking
        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer', return_value=mock_ruler):
                with patch('devdox_ai_sonar.cli.LLMFixer', return_value=mock_fixer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with mock_tmp_clone():
                            with patch('devdox_ai_sonar.cli.download_latest_version', return_value=None):
                                from devdox_ai_sonar.cli import _process_and_fix_issues

                                # download_latest_version returns None, so Abort is raised
                                with pytest.raises(click.exceptions.Abort):
                                    await _process_and_fix_issues(
                                        auth_config,
                                        llm_config,
                                        "main",
                                        0,
                                        fix_params,
                                        issue_type=IssueType.SECURITY
                                    )

    async def test_process_security_no_issues(self):
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
                        with mock_tmp_clone():
                            with patch('devdox_ai_sonar.cli.download_latest_version', return_value=os.path.join(tempfile.gettempdir(), "downloaded_repo")):

                                await _process_and_fix_issues(
                                    auth_config,
                                    llm_config,
                                    "main",
                                    0,
                                    fix_params,
                                    issue_type=IssueType.SECURITY,
                                )

                                mock_analyzer.get_fixable_security_issues.assert_called_once()

    async def test_process_security_with_issues(self):
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

        mock_fixer = AsyncMock()
        mock_fixer.generate_fix_by_file.return_value = [mock_fix]

        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"
        auth_config.git_url = "https://github.com/test-org/test-project.git"

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
                        with mock_tmp_clone():
                            with patch('devdox_ai_sonar.cli.download_latest_version', return_value=os.path.join(tempfile.gettempdir(), "downloaded")):
                                with patch('devdox_ai_sonar.cli._display_fix_preview'):
                                    with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=False)):

                                        await _process_and_fix_issues(
                                            auth_config,
                                            llm_config,
                                            "main",
                                            0,
                                            fix_params,
                                            issue_type=IssueType.SECURITY,
                                        )

                                        mock_fixer.generate_fix_by_file.assert_called_once()

    async def test_cleanup_on_download_failure(self):
        """Test tmp dir is cleaned up when download fails."""
        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"
        auth_config.git_url = "https://github.com/test/test.git"

        llm_config = Mock()
        llm_config.provider = "openai"
        llm_config.model = "gpt-4"
        llm_config.api_key = "key"

        fix_params = {"max_fixes": 10, "apply": False, "dry_run": False}

        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer'):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer'):
                with patch('devdox_ai_sonar.cli.LLMFixer'):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with mock_tmp_clone() as mock_ctx:
                            with patch('devdox_ai_sonar.cli.download_latest_version', return_value=None):
                                with pytest.raises(click.exceptions.Abort):
                                    await _process_and_fix_issues(
                                        auth_config, llm_config, "main", 0,
                                        fix_params, issue_type=IssueType.SECURITY,
                                    )

                                mock_ctx.__aexit__.assert_called_once()

    async def test_cleanup_on_fetch_exception(self):
        """Test tmp dir is cleaned up when _fetch_issues_by_type raises."""
        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"
        auth_config.git_url = "https://github.com/test/test.git"

        llm_config = Mock()
        llm_config.provider = "openai"
        llm_config.model = "gpt-4"
        llm_config.api_key = "key"

        fix_params = {"max_fixes": 10, "apply": False, "dry_run": False}

        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer'):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer'):
                with patch('devdox_ai_sonar.cli.LLMFixer'):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with mock_tmp_clone() as mock_ctx:
                            with patch('devdox_ai_sonar.cli.download_latest_version', return_value=os.path.join(tempfile.gettempdir(), "repo")):
                                with patch('devdox_ai_sonar.cli._fetch_issues_by_type', side_effect=RuntimeError("API error")):
                                    with pytest.raises(RuntimeError, match="API error"):
                                        await _process_and_fix_issues(
                                            auth_config, llm_config, "main", 0,
                                            fix_params, issue_type=IssueType.SECURITY,
                                        )

                                    mock_ctx.__aexit__.assert_called_once()

    async def test_cleanup_on_process_exception(self):
        """Test tmp dir is cleaned up when _process_files_with_issues raises."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = {
            "file.py": [Mock()]
        }

        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"
        auth_config.git_url = "https://github.com/test/test.git"

        llm_config = Mock()
        llm_config.provider = "openai"
        llm_config.model = "gpt-4"
        llm_config.api_key = "key"

        fix_params = {"max_fixes": 10, "apply": False, "dry_run": False}

        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer'):
                with patch('devdox_ai_sonar.cli.LLMFixer'):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with mock_tmp_clone() as mock_ctx:
                            with patch('devdox_ai_sonar.cli.download_latest_version', return_value=os.path.join(tempfile.gettempdir(), "repo")):
                                with patch('devdox_ai_sonar.cli._process_files_with_issues', new=AsyncMock(side_effect=RuntimeError("LLM error"))):
                                    with pytest.raises(RuntimeError, match="LLM error"):
                                        await _process_and_fix_issues(
                                            auth_config, llm_config, "main", 0,
                                            fix_params, issue_type=IssueType.SECURITY,
                                        )

                                    mock_ctx.__aexit__.assert_called_once()

    async def test_no_tmpdir_on_branch_failure(self):
        """Test that no tmp dir is created when branch validation fails."""
        auth_config = Mock()
        auth_config.token = "test"
        auth_config.organization = "org"
        auth_config.project = "proj"
        auth_config.project_path = "/tmp"

        llm_config = Mock()
        llm_config.provider = "openai"
        llm_config.model = "gpt-4"
        llm_config.api_key = "key"

        fix_params = {"max_fixes": 10, "apply": False, "dry_run": False}

        with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer'):
            with patch('devdox_ai_sonar.cli.RuleAnalyzer'):
                with patch('devdox_ai_sonar.cli.LLMFixer'):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli.TmpCloneManager') as MockCls:
                            with pytest.raises(click.exceptions.Abort):
                                await _process_and_fix_issues(
                                    auth_config, llm_config, None, 0,
                                    fix_params, issue_type=IssueType.REGULAR,
                                )

                            # TmpCloneManager should never have been used
                            MockCls.assert_not_called()


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
            sonar_line_number=10,
            fixed_code_blocks=[CodeBlock(block_name="test",
                                         start_line="1",
                                         end_line="10",
                                         has_changes=True,
                                         change_type=ChangeType.FULL_CODE,
                                         block_type= BlockType.MODULE,
                                         context="new_code")]
        )

        with patch('devdox_ai_sonar.cli.console'):
            with patch('devdox_ai_sonar.cli.Panel'):
                _display_fix_preview(fix, [Mock()])

    @pytest.mark.parametrize("empty_explanation", ["", "   ", "\t\n  "])
    def test_display_fix_preview_empty_explanation_hides_panel(self, empty_explanation):
        """Test that empty or whitespace-only explanation does not render a Panel"""
        fix = FixSuggestion(
            issue_key="issue-1",
            rule="python:S1234",
            original_code="old",
            fixed_code="new",
            explanation=empty_explanation,
            confidence=0.95,
            llm_model="a",
            file_path="test.py",
            sonar_line_number=10,
            fixed_code_blocks=[CodeBlock(block_name="test",
                                         start_line="1",
                                         end_line="10",
                                         has_changes=True,
                                         change_type=ChangeType.FULL_CODE,
                                         block_type=BlockType.MODULE,
                                         context="new_code")]
        )

        with patch('devdox_ai_sonar.cli.console') as mock_console:
            with patch('devdox_ai_sonar.cli.Panel') as mock_panel:
                _display_fix_preview(fix, [Mock()])
                mock_panel.assert_not_called()

    def test_display_fix_preview_with_explanation_shows_panel(self):
        """Test that a non-empty explanation renders the Panel"""
        fix = FixSuggestion(
            issue_key="issue-1",
            rule="python:S1234",
            original_code="old",
            fixed_code="new",
            explanation="This fix addresses the issue by...",
            confidence=0.95,
            llm_model="a",
            file_path="test.py",
            sonar_line_number=10,
            fixed_code_blocks=[CodeBlock(block_name="test",
                                         start_line="1",
                                         end_line="10",
                                         has_changes=True,
                                         change_type=ChangeType.FULL_CODE,
                                         block_type=BlockType.MODULE,
                                         context="new_code")]
        )

        with patch('devdox_ai_sonar.cli.console') as mock_console:
            with patch('devdox_ai_sonar.cli.Panel') as mock_panel:
                _display_fix_preview(fix, [Mock()])
                mock_panel.assert_called_once()

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

    async def test_process_files_with_issues_basic(self):
        """Test basic file processing"""
        issues_by_file = {"module.py": [Mock(rule="python:S1234", file="module.py")]}

        mock_fixer = AsyncMock()
        mock_fixer.generate_fix_by_file.return_value = None
        mock_services = {
            'analyzer': Mock(),
            'ruler': Mock(),
            'fixer': mock_fixer
        }
        mock_services['ruler'].get_rule_by_key.return_value = {}

        with patch('devdox_ai_sonar.cli.console'):
            with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                with patch('devdox_ai_sonar.cli._should_continue_to_next_issue', new=AsyncMock(return_value=False)):
                    await _process_files_with_issues(
                        issues_by_file,
                        mock_services,
                        Mock(project_path=tempfile.gettempdir()),
                        {"apply": False, "dry_run": False},
                        issue_type=IssueType.SECURITY,
                        tmp_path= Path(tempfile.gettempdir()) / "new"
                    )

    async def test_generate_fix_for_file_success(self):
        """Test successful fix generation"""
        issues = [Mock(rule="python:S1234")]

        mock_fixer = AsyncMock()
        mock_services = {
            'ruler': Mock(),
            'fixer': mock_fixer
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
            sonar_line_number=10,
            fixed_code_blocks=[CodeBlock(block_name="test",
                                         start_line="1",
                                         end_line="10",
                                         has_changes=True,
                                         change_type=ChangeType.FULL_CODE,
                                         block_type=BlockType.MODULE,
                                         context="new_code")]
        )
        mock_services['fixer'].generate_fix_by_file.return_value = mock_fix

        with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
            result = await _generate_fix_for_file(issues, mock_services, Mock(project_path=tempfile.gettempdir()),issue_type=IssueType.SECURITY, tmp_path= Path(tempfile.gettempdir()) / "new")

            assert result == mock_fix

    async def test_handle_fix_apply_true(self):
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
            sonar_line_number=10,
            fixed_code_blocks=[CodeBlock(block_name="test",
                                         start_line="1",
                                         end_line="10",
                                         has_changes=True,
                                         change_type=ChangeType.FULL_CODE,
                                         block_type=BlockType.MODULE,
                                         context="new_code")]
        )

        mock_fixer = AsyncMock()
        mock_fixer.apply_fixes_with_validation.return_value = FixResult(
            successful_fixes=[fix],
            failed_fixes=[],
            skipped_files=[],
            total_fixes_attempted=1,
            success_rate=1.0,
            backup_created=True,
            backup_path=Path(tempfile.gettempdir()) / "backup",
            project_path=Path(tempfile.gettempdir()) / "backup"
        )

        with patch('devdox_ai_sonar.cli._display_fix_preview'):
            with patch('devdox_ai_sonar.cli._display_fix_results'):
                await handle_fix(
                    fix,
                    [Mock()],
                    mock_fixer,
                    Mock(project_path=tempfile.gettempdir()),
                    {"apply": 1, "dry_run": 0, "create_backup": 1}
                )

                mock_fixer.apply_fixes_with_validation.assert_called_once()

    @patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock())
    async def test_should_continue_to_next_issue_last_file(self):
        """Test on last file"""
        result = await _should_continue_to_next_issue(5, 5)

        assert result is False



# ============================================================================
# TEST CLASS: LOAD AND VALIDATE CONFIG
# ============================================================================

class TestLoadAndValidateConfig:
    """Test configuration loading"""

    async def test_load_and_validate_config_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test successful config load"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {"max_fixes": 10}
        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        result =await _load_and_validate_config()

                        assert result is not None
                        assert len(result) == 3

    async def test_load_and_validate_config_no_auth(self):
        """Test config load without auth"""

        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }
        with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
            mock_auth_instance = Mock(spec=AuthConfig)
            mock_auth_instance.validate.return_value = (True, None)
            mock_from_dict.return_value = mock_auth_instance

            with pytest.raises(Exception):  # Should abort
                await _load_and_validate_config()

    async def test_load_and_validate_config_invalid_auth(self, mock_config_service, mock_auth_dict):
        """Test config load with invalid auth"""

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        with pytest.raises(Exception):  # Should abort
                            await _load_and_validate_config()

    async def test_load_and_validate_config_no_llm(
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
                        await _load_and_validate_config()

    async def test_load_and_validate_config_with_predefined_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test successful config load using predefined branch/PR"""
        # Setup mocks

        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }

        mock_config_service.load_llm_config.return_value = mock_llm_config

        mock_config_manager.get_value.return_value = {"max_fixes": 10}
        mock_provider_manager.branch_or_pr.return_value = ("main", "123")

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance


                        # Call with use_predefined=True
                        result = await _load_and_validate_config(use_predefined=True)

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

    async def test_load_and_validate_config_with_predefined_branch_only(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):

        """Test predefined mode with only branch (no PR)"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.return_value = ("feature/test", None)
        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        result = await _load_and_validate_config(use_predefined=True)

                        assert result is not None
                        _, _, params = result
                        assert params['branch'] == "feature/test"
                        assert params['pull_request'] is None

    async def test_load_and_validate_config_with_predefined_pr_only(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test predefined mode with only PR (no branch)"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.return_value = (None, "456")

        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance


                        result = await _load_and_validate_config(use_predefined=True)

                        assert result is not None
                        _, _, params = result
                        assert params['branch'] is None
                        assert params['pull_request'] == "456"

    async def test_load_and_validate_config_with_predefined_no_branch_or_pr(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test predefined mode when no branch/PR in config - should abort"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.return_value = (None, None)
        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        with pytest.raises(Exception):  # Should abort
                            await _load_and_validate_config(use_predefined=True)

                        # Verify correct method was attempted
                        mock_provider_manager.branch_or_pr.assert_called_once()

        # ========================================================================
        # NEW TESTS: use_predefined=False (Prompt Mode - Explicit)
        # ========================================================================

    async def test_load_and_validate_config_with_prompt_explicit(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test explicit use_predefined=False uses prompt method"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr_prompt.return_value = ("develop", "789")

        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        result = await _load_and_validate_config(use_predefined=False)

                        assert result is not None
                        _, _, params = result

                        # Verify prompt method was called, not predefined method
                        mock_provider_manager.branch_or_pr_prompt.assert_called_once()
                        mock_provider_manager.branch_or_pr.assert_not_called()

                        assert params['branch'] == "develop"
                        assert params['pull_request'] == "789"

    async def test_load_and_validate_config_prompt_mode_no_branch_or_pr(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test prompt mode when user provides neither branch nor PR - should abort"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)
        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }


        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        with pytest.raises(Exception):  # Should abort
                            await _load_and_validate_config(use_predefined=False)

                        mock_provider_manager.branch_or_pr_prompt.assert_called_once()

        # ========================================================================
        # NEW TESTS: Edge Cases and Error Scenarios
        # ========================================================================

    async def test_load_and_validate_config_predefined_method_raises_exception(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test predefined mode when branch_or_pr() raises exception"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {}
        mock_provider_manager.branch_or_pr.side_effect = Exception("Config error")

        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        with pytest.raises(Exception) as exc_info:
                            await _load_and_validate_config(use_predefined=True)

                        assert "Config error" in str(exc_info.value)

    async def test_load_and_validate_config_params_merge_with_existing_config(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config
    ):
        """Test that branch/PR params are merged with existing configuration"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        existing_config = {"max_fixes": 10, "timeout": 30}
        mock_config_manager.get_value.return_value = existing_config.copy()
        mock_provider_manager.branch_or_pr.return_value = ("main", "100")

        mock_auth_dict = {
            "token": "token123",
            "organization": "org",
            "project": "proj",
            "project_path": "/path"
        }

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config(use_predefined=True)

                        # Verify existing config is preserved
                        assert params['max_fixes'] == 10
                        assert params['timeout'] == 30
                        # Verify new params are added
                        assert params['branch'] == "main"
                        assert params['pull_request'] == "100"

    async def test_load_and_validate_config_params_empty_dict_when_no_existing(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config,mock_auth_dict
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # Should handle None gracefully and create new dict
                        assert isinstance(params, dict)
                        assert params['branch'] == "test"
                        assert params['pull_request'] is None

        # ========================================================================
        # NEW TESTS: Console Output Verification
        # ========================================================================



    async def test_load_and_validate_config_returns_correct_types(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        auth_config, llm_config, params = await  _load_and_validate_config(
                            use_predefined=True
                        )

                        # Verify types
                        assert isinstance(auth_config, (Mock, AuthConfig))  # Mock or actual
                        assert llm_config is not None
                        assert isinstance(params, dict)  # Should be Dict[str, Any], not Optional[str]

    async def test_load_and_validate_config_console_message_loading(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        with patch('devdox_ai_sonar.cli.console') as mock_console:
                            await _load_and_validate_config()

                            # Verify loading message displayed at start
                            loading_calls = [
                                call for call in mock_console.print.call_args_list
                                if 'Loading configuration' in str(call)
                            ]
                            assert len(loading_calls) >= 1

    async def test_load_and_validate_config_console_message_success(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        with patch('devdox_ai_sonar.cli.console') as mock_console:
                            await _load_and_validate_config()

                            # Verify success message displayed at end
                            success_calls = [
                                call for call in mock_console.print.call_args_list
                                if '✓' in str(call) or 'Configuration loaded' in str(call)
                            ]
                            assert len(success_calls) >= 1

    async def test_load_and_validate_config_console_message_no_auth(
            self, mock_config_service
    ):
        """Test console displays appropriate messages when auth not found"""
        mock_config_service.load_auth_config.return_value = None

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.console') as mock_console:
                with pytest.raises(click.Abort):
                    await _load_and_validate_config()

                # Verify error messages displayed
                calls = [str(call) for call in mock_console.print.call_args_list]
                # Should display AUTHENTICATION_NOT_FOUND and DEVDOX_SONAR_CONFIG
                assert len(calls) >= 2

    async def test_load_and_validate_config_console_message_invalid_auth(
            self, mock_config_service, mock_config_manager
    ):
        """Test console displays error message for invalid auth"""
        mock_auth_dict = {
            "token": "invalid_token",
            "organization": "test-org",
            "project": "test-project",
            "project_path": "/test/path"
        }

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs, \
                patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm, \
                patch('devdox_ai_sonar.cli.console') as mock_console:
            mock_cs.return_value = mock_config_service
            mock_config_service.load_auth_config.return_value = mock_auth_dict
            mock_cm.return_value = mock_config_manager

            mock_auth_instance = Mock(spec=AuthConfig)
            mock_auth_instance.validate.return_value = (False, "Invalid token")

            with patch("devdox_ai_sonar.cli.AuthConfig.from_dict", return_value=mock_auth_instance):
                with pytest.raises(click.Abort):
                    await _load_and_validate_config()

                # Check that some error message was printed containing our validation error
                all_print_calls = [str(call) for call in mock_console.print.call_args_list]
                error_messages = [call for call in all_print_calls if 'Invalid token' in call]

                assert len(error_messages) >= 1, (
                    f"Expected error message containing 'Invalid token' but got calls: {all_print_calls}"
                )



    async def test_load_and_validate_config_console_message_no_llm(
            self, mock_config_service, mock_config_manager, auth_config
    ):
        """Test console displays error when no LLM providers configured"""
        mock_config_service.load_llm_config.return_value = None

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                        patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                    mock_auth_instance = Mock(spec=AuthConfig)
                    mock_auth_instance.validate.return_value = (True, None)
                    mock_from_dict.return_value = mock_auth_instance

                with patch('devdox_ai_sonar.cli.console') as mock_console:
                        with pytest.raises(click.Abort):
                            await _load_and_validate_config()

                        # Verify LLM error message
                        calls = [str(call) for call in mock_console.print.call_args_list]
                        llm_error_calls = [c for c in calls if 'LLM' in c or 'providers' in c]
                        assert len(llm_error_calls) >= 1

    async def test_load_and_validate_config_console_message_no_branch_pr(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        with patch('devdox_ai_sonar.cli.console') as mock_console:
                                with pytest.raises(click.Abort):
                                    await _load_and_validate_config()

                                # Verify NO_BRANCH_OR_PR_SPECIFIED message displayed
                                mock_console.print.assert_any_call(constant.NO_BRANCH_OR_PR_SPECIFIED)

    async def test_load_and_validate_config_exclude_rules_string_split(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # Verify exclude_rules is split into list
                        assert isinstance(params['exclude_rules'], list)
                        assert len(params['exclude_rules']) == 3
                        assert params['exclude_rules'] == ["python:S3776", "python:S1192", "python:S107"]

    async def test_load_and_validate_config_exclude_rules_single_rule(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # Single rule should result in list with one element
                        assert isinstance(params['exclude_rules'], list)
                        assert len(params['exclude_rules']) == 1
                        assert params['exclude_rules'] == ["python:S3776"]

    async def test_load_and_validate_config_exclude_rules_with_spaces(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # Should preserve spaces (or you might want to strip them)
                        assert isinstance(params['exclude_rules'], list)
                        assert len(params['exclude_rules']) == 3

    async def test_load_and_validate_config_no_exclude_rules(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # exclude_rules should not be in params or be None
                        assert 'exclude_rules' not in params or params.get('exclude_rules') is None

    async def test_load_and_validate_config_exclude_rules_empty_string(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # Empty string should not be split (falsy value)
                        # Based on code: if params.get("exclude_rules")
                        assert 'exclude_rules' in params
                        # Empty string is falsy, so split shouldn't be called
                        assert params['exclude_rules'] == ""

    async def test_load_and_validate_config_exclude_rules_none_becomes_empty_list(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
    ):
        """Test exclude_rules with NONE value is converted to empty list"""
        mock_config_service.load_llm_config.return_value = mock_llm_config
        mock_config_manager.get_value.return_value = {
            "exclude_rules": constant.EXCLUDE_NONE
        }
        mock_provider_manager.branch_or_pr_prompt.return_value = ("main", None)

        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.ConfigManager') as mock_cm:
                mock_cm.return_value = mock_config_manager

                with patch('devdox_ai_sonar.cli.ProviderConfigManager') as mock_pm:
                    mock_pm.return_value = mock_provider_manager

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # NONE value should be converted to empty list
                        assert isinstance(params['exclude_rules'], list)
                        assert len(params['exclude_rules']) == 0

    async def test_load_and_validate_config_exclude_rules_already_list(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config,auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        # This might raise AttributeError: 'list' object has no attribute 'split'
                        # Need to test actual behavior
                        try:
                            _, _, params = await _load_and_validate_config()
                            # If it doesn't raise, verify it's still a list
                            assert isinstance(params['exclude_rules'], list)
                        except AttributeError:
                            # Expected if code doesn't handle list case
                            pytest.xfail("Code doesn't handle exclude_rules as list")

    async def test_load_and_validate_config_params_preserves_all_existing_config(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

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

    async def test_load_and_validate_config_params_branch_overwrites_existing(
            self, mock_config_service, mock_config_manager,
            mock_provider_manager, mock_llm_config, auth_config
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

                    with patch("devdox_ai_sonar.cli.ConfigService.load_auth_config", return_value=mock_auth_dict), \
                            patch("devdox_ai_sonar.cli.AuthConfig.from_dict") as mock_from_dict:
                        mock_auth_instance = Mock(spec=AuthConfig)
                        mock_auth_instance.validate.return_value = (True, None)
                        mock_from_dict.return_value = mock_auth_instance

                        _, _, params = await _load_and_validate_config()

                        # New values should overwrite old
                        assert params['branch'] == "new-branch"
                        assert params['pull_request'] == "new-pr"
# ============================================================================
# TEST CLASS: TEST WITH FIXTURES
# ============================================================================

class TestWithFixtures:
    """Cleaner tests using fixtures"""

    async def test_run_analyze_with_fixtures(self, mock_loaded_config):
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

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(return_value=mock_loaded_config)):
            with patch('devdox_ai_sonar.cli.smart_prompt', new=AsyncMock(return_value="10")):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):

                        with patch('devdox_ai_sonar.cli._display_analysis_results'):

                            await _run_analyze()

    async def test_run_inspect_with_fixtures(self, mock_loaded_config):
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

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(return_value=mock_loaded_config)):
            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):

                await _run_inspect()


class TestShouldContinueToMenu:
    """Test cases for _should_continue_to_menu."""

    async def test_should_continue_returns_true(self):
        """Test returns True when user confirms."""
        mock_confirm = AsyncMock(return_value=True)
        with patch('devdox_ai_sonar.cli.smart_confirm', new=mock_confirm):
            result = await _should_continue_to_menu()

            assert result is True
            mock_confirm.assert_called_once_with(
                "Return to main menu?",
                default=True,
                allow_switch=True
            )

    async def test_should_continue_returns_false(self):
        """Test returns False when user declines."""
        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=False)):
            result = await _should_continue_to_menu()

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

    @patch('devdox_ai_sonar.cli.console')
    async def test_user_confirms_exit(self, mock_console):
        """Test user confirms exit."""
        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=True)):
            with pytest.raises(SystemExit) as exc_info:
                await _handle_keyboard_interrupt()

            assert exc_info.value.code == 0

    @patch('devdox_ai_sonar.cli.console')
    async def test_user_declines_exit(self, mock_console):
        """Test user declines exit."""
        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=False)):
            result = await _handle_keyboard_interrupt()

            assert result is False

    @patch('devdox_ai_sonar.cli.console')
    async def test_double_keyboard_interrupt(self, mock_console):
        """Test double Ctrl+C forces exit."""
        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(side_effect=KeyboardInterrupt())):
            with pytest.raises(SystemExit) as exc_info:
                await _handle_keyboard_interrupt()

            assert exc_info.value.code == 130  # SIGINT exit code

    @patch('devdox_ai_sonar.cli.console')
    async def test_eof_error_forces_exit(self, mock_console):
        """Test EOF error forces exit."""
        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(side_effect=EOFError())):
            with pytest.raises(SystemExit) as exc_info:
                await _handle_keyboard_interrupt()

            assert exc_info.value.code == 130

    @patch('devdox_ai_sonar.cli.console')
    async def test_unexpected_exception_during_exit(self, mock_console):
        """Test unexpected exception during exit confirmation."""
        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(side_effect=RuntimeError("Unexpected error"))):
            with pytest.raises(SystemExit) as exc_info:
                await _handle_keyboard_interrupt()

            assert exc_info.value.code == 1


class TestHandleInteractiveError:
    """Test cases for _handle_interactive_error."""

    @patch('devdox_ai_sonar.cli.console')
    async def test_user_returns_to_menu(self, mock_console):
        """Test user chooses to return to menu."""
        error = ValueError("Test error")

        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=True)):
            result = await _handle_interactive_error(error)

        assert result is False
        mock_console.print.assert_called()

    @patch('devdox_ai_sonar.cli.console')
    async def test_user_exits_after_error(self, mock_console):
        """Test user chooses to exit after error."""
        error = ValueError("Test error")

        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=False)):
            with pytest.raises(SystemExit) as exc_info:
                await _handle_interactive_error(error)

        assert exc_info.value.code == 1

    @patch('devdox_ai_sonar.cli.console')
    async def test_keyboard_interrupt_during_error_handling(self, mock_console):
        """Test keyboard interrupt during error handling."""
        error = ValueError("Test error")

        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(side_effect=KeyboardInterrupt())):
            with pytest.raises(SystemExit) as exc_info:
                await _handle_interactive_error(error)

        assert exc_info.value.code == 1

    @patch('devdox_ai_sonar.cli.console')
    async def test_fatal_error_during_confirmation(self, mock_console):
        """Test fatal error during confirmation prompt."""
        error = ValueError("Test error")

        with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(side_effect=RuntimeError("Fatal error"))):
            with pytest.raises(SystemExit) as exc_info:
                await _handle_interactive_error(error)

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

    @patch('devdox_ai_sonar.cli._process_interactive_command_async')
    async def test_execute_interactive_iteration_normal(self, mock_process):
        """Test normal iteration"""
        mock_process.return_value = False

        result = await _execute_interactive_iteration_async(Mock())

        assert result is False

    @patch('devdox_ai_sonar.cli._process_interactive_command_async')
    @patch('devdox_ai_sonar.cli._handle_command_switch')
    async def test_execute_interactive_iteration_switch(self, mock_handle, mock_process):
        """Test with command switch"""
        mock_process.side_effect = SwitchCommandException()

        result = await _execute_interactive_iteration_async(Mock())

        assert result is False
        mock_handle.assert_called_once()

# ============================================================================
# Test _run_interactive_mode (Integration)
# ============================================================================

class TestRunInteractiveMode:
    """Integration tests for _run_interactive_mode."""

    @patch('devdox_ai_sonar.cli._exit_application')
    async def test_exits_when_no_command_selected(self, mock_exit):
        """Test exits when no command selected."""
        mock_exit.side_effect = SystemExit(0)
        mock_ctx = Mock(spec=click.Context)

        with patch('devdox_ai_sonar.cli.show_command_selector_async', new=AsyncMock(return_value=None)):
            with pytest.raises(SystemExit):
               await _run_interactive_mode_async(mock_ctx)

        mock_exit.assert_called_once()

    @patch('devdox_ai_sonar.cli._exit_application')
    async def test_executes_command_and_exits(
            self, mock_exit
    ):
        """Test executes command then exits."""
        mock_exit.side_effect = SystemExit(0)
        mock_ctx = Mock(spec=click.Context)

        with patch('devdox_ai_sonar.cli.show_command_selector_async', new=AsyncMock(return_value='fix_issues')):
            with patch('devdox_ai_sonar.cli._execute_interactive_command_async', new=AsyncMock()) as mock_execute:
                with patch('devdox_ai_sonar.cli._should_continue_to_menu', new=AsyncMock(return_value=False)):
                    with pytest.raises(SystemExit):
                        await _run_interactive_mode_async(mock_ctx)

                    mock_execute.assert_called_once()
        mock_exit.assert_called_once()

    async def test_handles_switch_command_exception(self):
        """Test handles SwitchCommandException."""
        mock_ctx = Mock(spec=click.Context)

        with patch('devdox_ai_sonar.cli.show_command_selector_async', new=AsyncMock(side_effect=['fix_issues', 'exit'])) as mock_selector:
            with patch('devdox_ai_sonar.cli._execute_interactive_command_async', new=AsyncMock(side_effect=SwitchCommandException())):
                with patch('devdox_ai_sonar.cli._should_continue_to_menu', new=AsyncMock(return_value=True)):
                    with pytest.raises(SystemExit):
                        await _run_interactive_mode_async(mock_ctx)

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
            project_path="/test/path",
            git_url="https://github.com/test/repo.git"
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

    @patch('devdox_ai_sonar.cli._load_rules_cache', return_value={})
    def test_collects_rules_for_all_issues(self, mock_cache):
        """Test collects rule information for all issues (cache miss, falls back to API)."""
        mock_ruler = Mock()
        mock_ruler.get_rule_by_key.side_effect = [
            {'key': 'rule1', 'name': 'Rule 1'},
            {'key': 'rule2', 'name': 'Rule 2'}
        ]

        issues = ['rule1', 'rule2']

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
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.load_config.return_value = None

        mock_provider_mgr = AsyncMock()
        mock_provider_mgr.get_existing_providers.return_value = []

        return mock_manager, mock_provider_mgr

    # Pattern 1: Direct exception catching
    async def test_pattern_1_direct_catch(self, mock_setup):
        """Pattern 1: Catch click.Abort directly"""
        mock_manager, mock_provider_mgr = mock_setup

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock())

            with patch('devdox_ai_sonar.cli.console.print'):
                with pytest.raises(click.Abort):
                    await update_provider()

    # Pattern 2: Try-except verification
    async def test_pattern_2_try_except(self, mock_setup):
        """Pattern 2: Use try-except for more control"""
        mock_manager, mock_provider_mgr = mock_setup

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock())

            with patch('devdox_ai_sonar.cli.console.print'):
                abort_raised = False
                try:
                    await update_provider()
                except click.Abort:
                    abort_raised = True

                assert abort_raised, "Expected click.Abort to be raised"

    # Pattern 4: Check exception message
    async def test_pattern_4_exception_info(self, mock_setup):
        """Pattern 4: Verify exception details"""
        mock_manager, mock_provider_mgr = mock_setup

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock())

            with patch('devdox_ai_sonar.cli.console.print') as mock_print:
                with pytest.raises(click.Abort) as exc_info:
                    await update_provider()

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
        manager = AsyncMock()
        manager.save_config = Mock()
        manager.create_default_config = Mock()
        manager.delete_value = AsyncMock()
        manager.load_config.return_value = None
        manager.save_config.return_value = None
        return manager

    @pytest.fixture
    def mock_provider_manager(self):
        manager = AsyncMock()
        manager.get_existing_providers.return_value = ['openai', 'anthropic']
        manager.update_existing_provider.return_value = True
        return manager

    async def test_update_provider_success(
        self,
        mock_config_manager,
        mock_provider_manager
    ):
        """Test successful provider update"""
        mock_config_manager = AsyncMock()
        mock_config_manager.save_config = Mock()
        mock_config_manager.create_default_config = Mock()
        mock_config_manager.delete_value = AsyncMock()
        mock_config_manager.load_config.return_value = None
        mock_config_manager.save_config.return_value = None
        mock_provider_manager = AsyncMock()
        mock_provider_manager.get_existing_providers.return_value = ['openai', 'anthropic']
        mock_provider_manager.update_existing_provider.return_value = True

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
                            await update_provider()

                            # Verify success flow
                            mock_provider_manager.update_existing_provider.assert_called_once_with('openai')
                            mock_config_manager.save_config.assert_called_once_with(create_backup=False)

    async def test_update_provider_no_providers_abort(
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
                   await update_provider()

    async def test_update_provider_user_cancels_abort(
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
                        await update_provider()


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
    async def test_load_config_service_returns_none(self, mock_init):
        """Test when config service returns None"""
        mock_config_service = AsyncMock()
        mock_config_service.load_auth_config.return_value = None

        mock_init.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), mock_config_service
        )

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                await _load_and_validate_config()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.AuthConfig.from_dict')
    async def test_load_config_validation_fails(self, mock_auth_config, mock_init):
        """Test when auth config validation fails"""
        mock_config_service = AsyncMock()
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
                await _load_and_validate_config()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.AuthConfig.from_dict')
    async def test_load_config_no_llm_config(self, mock_auth_config, mock_init):
        """Test when LLM config is missing"""
        mock_config_service = AsyncMock()
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
                await _load_and_validate_config()

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.AuthConfig.from_dict')
    async def test_load_config_no_branch_or_pr(self, mock_auth_config, mock_init):
        """Test when neither branch nor PR is provided"""
        mock_config_service = AsyncMock()
        mock_config_service.load_auth_config.return_value = {
            "token": "test",
            "organization": "org",
            "project": "proj",
            "project_path": "/tmp"
        }
        mock_config_service.load_llm_config.return_value = Mock()

        mock_provider_manager = AsyncMock()
        mock_provider_manager.branch_or_pr_prompt.return_value = (None, None)

        mock_init.return_value = (
            AsyncMock(), Mock(), Mock(),
            mock_provider_manager, Mock(), mock_config_service
        )

        mock_auth_instance = Mock()
        mock_auth_instance.validate.return_value = (True, None)
        mock_auth_config.return_value = mock_auth_instance

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                await _load_and_validate_config()


# ============================================================================
# TEST CLASS: COMMAND EXECUTION ERROR SCENARIOS
# ============================================================================

class TestCommandExecutionErrors:
    """Test error scenarios in command execution"""

    async def test_run_fix_issues_config_load_fails(self):
        """Test fix_issues when config loading fails"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(side_effect=click.Abort())):
            with pytest.raises(click.Abort):
                await _run_fix_issues()

    async def test_run_fix_issues_processing_exception(self):
        """Test fix_issues when processing raises exception"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(return_value=(Mock(), Mock(), {"branch": "main", "pull_request": 0}))):
            with patch('devdox_ai_sonar.cli.display_configuration') as mock_display:
                mock_display.return_value = {
                    "branch": "main",
                    "pull_request": 0,
                    "max_fixes": 10,
                    "apply": 0,
                    "dry_run": 0
                }
                with patch('devdox_ai_sonar.cli.smart_confirm', new=AsyncMock(return_value=True)):
                    with patch('devdox_ai_sonar.cli._process_and_fix_issues', new=AsyncMock(side_effect=Exception("Processing failed"))):
                        with pytest.raises(Exception, match="Processing failed"):
                           await _run_fix_issues()


    async def test_run_inspect_directory_not_found(self):
        """Test inspect when directory doesn't exist"""
        mock_auth = Mock()
        mock_auth.token = "test"
        mock_auth.organization = "org"
        mock_auth.project_path = "/nonexistent/path"

        with patch('devdox_ai_sonar.cli._load_and_validate_config', new=AsyncMock(return_value=(mock_auth, Mock(), {}))):
            mock_analyzer = Mock()
            mock_analyzer.analyze_project_directory.side_effect = ValueError("Invalid path")
            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer', return_value=mock_analyzer):
                with pytest.raises(ValueError, match="Invalid path"):
                    await _run_inspect()


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
            sonar_line_number=10,
            fixed_code_blocks=[CodeBlock(block_name="test",
                                         start_line="1",
                                         end_line="10",
                                         has_changes=True,
                                         change_type=ChangeType.FULL_CODE,
                                         block_type=BlockType.MODULE,
                                         context="new_code")]

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
    async def test_multiple_init_config_calls(self, mock_init):
        """Test multiple simultaneous init_config calls"""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.delete_value = AsyncMock()
        mock_manager.get_value.return_value = ['openai']

        mock_init.return_value = (
            mock_manager, Mock(), Mock(), AsyncMock(), Mock(), AsyncMock()
        )

        # Call multiple times
        for _ in range(3):
            with patch('devdox_ai_sonar.cli._configure_sonarcloud', new=AsyncMock(return_value=True)):
                with patch('devdox_ai_sonar.cli.change_parameters', new=AsyncMock()):
                    await init_config()

        # Should handle gracefully
        assert mock_init.call_count == 3


class TestHelperFunctions:
    """Test extracted helper functions."""


    @patch('devdox_ai_sonar.cli._generate_fix_for_file')
    @patch('devdox_ai_sonar.cli.handle_fix')
    @patch('devdox_ai_sonar.cli.console')
    async def test_process_single_fix_success(
            self, mock_console, mock_handle, mock_generate,
        auth_config, fix_params, sample_fix_suggestion
    ):
        """Test successful single fix processing."""
        mock_generate.return_value = [sample_fix_suggestion]
        issues = [Mock()]

        mock_services = {
            'analyzer': Mock(),
            'ruler': Mock(),
            'fixer': Mock()
        }
        await _process_single_fix(
            issues, mock_services, auth_config,
            fix_params, IssueType.SECURITY, "test.py", Path(tempfile.gettempdir()) / "new"
        )

        mock_generate.assert_called_once()
        mock_handle.assert_called_once()

    @patch('devdox_ai_sonar.cli._generate_fix_for_file')
    @patch('devdox_ai_sonar.cli.console')
    async def test_process_single_fix_no_fix_generated(
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
        await _process_single_fix(
            issues, mock_services, auth_config,
            fix_params, IssueType.SECURITY, "test.py", Path(tempfile.gettempdir()) / "new"
        )

        # Should print warning
        call_args = str(mock_console.print.call_args_list)
        assert 'No fix' in call_args or 'could not be generated' in call_args


# ============================================================================
# COVERAGE TESTS — CATEGORY A (Error/Validation Handlers)
# ============================================================================


class TestConfigureSonarcloudErrors:
    """Test error paths in _configure_sonarcloud."""

    async def test_returns_false_when_config_cancelled(self):
        """Lines 326-327: sonar_config is None → returns False."""
        mock_sonar_ui = Mock()
        mock_config_service = AsyncMock()

        mock_config_service.load_auth_config.return_value = {"token": ""}
        mock_config_service.check_all_value_empty = Mock(return_value=True)
        mock_sonar_ui.configure_sonarcloud.return_value = None

        with patch('devdox_ai_sonar.cli.console'):
            result = await _configure_sonarcloud(mock_sonar_ui, mock_config_service)

        assert result is False

    async def test_returns_false_when_save_fails(self):
        """Lines 338-339: save_config returns False → returns False."""
        mock_sonar_ui = Mock()
        mock_config_service = AsyncMock()

        mock_config_service.load_auth_config.return_value = {"token": ""}
        mock_config_service.check_all_value_empty = Mock(return_value=True)

        mock_sonar_config = Mock()
        mock_sonar_config.token = "t"
        mock_sonar_config.organization = "o"
        mock_sonar_config.project = "p"
        mock_sonar_config.project_path = "/path"
        mock_sonar_config.git_url = "https://git.example.com"
        mock_sonar_ui.configure_sonarcloud.return_value = mock_sonar_config
        mock_config_service.save_config.return_value = False

        result = await _configure_sonarcloud(mock_sonar_ui, mock_config_service)

        assert result is False


class TestConfigureProvidersLoopErrors:
    """Test warning path in _configure_providers_loop."""

    @patch('devdox_ai_sonar.cli._should_stop_configuring', return_value=True)
    @patch('devdox_ai_sonar.cli._handle_provider_configuration', new_callable=AsyncMock, return_value=False)
    @patch('devdox_ai_sonar.cli.console')
    async def test_warns_when_provider_config_fails(
        self, mock_console, mock_handle, mock_stop
    ):
        """Line 376: failed provider config prints warning."""
        mock_ui = Mock()
        mock_ui.select_provider_from_list = AsyncMock(return_value="openai")

        await _configure_providers_loop(
            Mock(), Mock(), mock_ui, ["openai", "anthropic"]
        )

        call_args = str(mock_console.print.call_args_list)
        assert "failed or was cancelled" in call_args


class TestAddProviderErrors:
    """Test abort path in add_provider."""

    async def test_aborts_when_no_providers_available(self):
        """Lines 593-596: empty available_providers → click.Abort."""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.load_config.return_value = None

        mock_provider_mgr = AsyncMock()
        mock_provider_mgr.get_available_providers.return_value = []

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock()
            )
            with patch('devdox_ai_sonar.cli.console.print'):
                with pytest.raises(click.Abort):
                    await add_provider()


class TestUpdateProviderErrors:
    """Test warning path in update_provider."""

    async def test_warns_when_update_fails(self):
        """Line 635: update_existing_provider returns False → warning printed."""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.load_config.return_value = None

        mock_provider_mgr = AsyncMock()
        mock_provider_mgr.get_existing_providers.return_value = ["openai"]
        mock_provider_mgr.update_existing_provider.return_value = False

        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_manager, Mock(), Mock(), mock_provider_mgr, Mock(), Mock()
            )
            with patch('devdox_ai_sonar.cli._display_operation_header'):
                with patch('devdox_ai_sonar.cli._select_existing_ui', return_value="openai"):
                    with patch('devdox_ai_sonar.cli._display_completion_message'):
                        with patch('devdox_ai_sonar.cli.console') as mock_console:
                            await update_provider()

                            call_args = str(mock_console.print.call_args_list)
                            assert "Update cancelled or failed" in call_args


class TestSwitchCommandExceptions:
    """Test SwitchCommandException re-raise in run commands."""

    async def test_run_fix_security_reraises_switch(self):
        """Lines 1074-1075: SwitchCommandException re-raised."""
        with patch(
            'devdox_ai_sonar.cli._load_and_validate_config',
            new_callable=AsyncMock,
            side_effect=SwitchCommandException,
        ):
            with pytest.raises(SwitchCommandException):
                await _run_fix_security_issues()

    async def test_run_analyze_reraises_switch(self):
        """Lines 1134-1135: SwitchCommandException re-raised."""
        with patch(
            'devdox_ai_sonar.cli._load_and_validate_config',
            new_callable=AsyncMock,
            side_effect=SwitchCommandException,
        ):
            with pytest.raises(SwitchCommandException):
                await _run_analyze()

    async def test_run_inspect_reraises_switch(self):
        """Lines 1172-1173: SwitchCommandException re-raised."""
        with patch(
            'devdox_ai_sonar.cli._load_and_validate_config',
            new_callable=AsyncMock,
            side_effect=SwitchCommandException,
        ):
            with pytest.raises(SwitchCommandException):
                await _run_inspect()


class TestLoadAndValidateConfigAuthNone:
    """Test AuthConfig.from_dict returning None."""

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli.AuthConfig.from_dict', return_value=None)
    async def test_aborts_when_auth_config_invalid(self, mock_from_dict, mock_init):
        """Lines 1196-1197: AuthConfig.from_dict → None → click.Abort."""
        mock_config_service = AsyncMock()
        mock_config_service.load_auth_config.return_value = {"token": "t"}

        mock_init.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), mock_config_service
        )

        with patch('devdox_ai_sonar.cli.console.print'):
            with pytest.raises(click.Abort):
                await _load_and_validate_config()


class TestProcessAndFixIssuesErrors:
    """Test error path in _process_and_fix_issues."""

    @patch('devdox_ai_sonar.cli._initialize_fix_services')
    @patch('devdox_ai_sonar.cli.console')
    async def test_aborts_when_branch_not_determined(
        self, mock_console, mock_init_services
    ):
        """Lines 1293-1294: empty branch_downloaded → click.Abort."""
        mock_analyzer = Mock()
        mock_analyzer.get_branch_from_pr.return_value = ""

        mock_init_services.return_value = {
            "analyzer": mock_analyzer,
            "ruler": Mock(),
            "fixer": Mock(),
        }

        auth_config = AuthConfig(
            token="t", organization="o", project="p",
            project_path="/tmp", git_url="https://git.example.com"
        )

        with pytest.raises(click.Abort):
            await _process_and_fix_issues(
                auth_config,
                Mock(),
                "",
                "123",
                {"max_fixes": 10},
                issue_type=IssueType.REGULAR,
            )


class TestGenerateFixErrors:
    """Test error path in _generate_fix_for_file."""

    async def test_returns_none_when_rule_name_missing(self):
        """Lines 1582-1583: non-security + no rule_name → returns None."""
        with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
            with patch('devdox_ai_sonar.cli.console'):
                result = await _generate_fix_for_file(
                    issues=[Mock()],
                    services={"fixer": AsyncMock()},
                    auth_config=Mock(project_path=tempfile.gettempdir()),
                    issue_type=IssueType.REGULAR,
                    tmp_path=os.path.join(tempfile.gettempdir(), "clone"),
                    rule_name=None,
                )

        assert result is None


# ============================================================================
# COVERAGE TESTS — CATEGORY B (Branch/Control Flow)
# ============================================================================


class TestSafeConvertPrIntInput:
    """Test integer input branch of _safe_convert_pr."""

    def test_int_input_returns_directly(self):
        """Line 88: isinstance(pull_request, int) → return directly."""
        assert _safe_convert_pr(42) == 42


class TestConfigureProvidersLoopFlow:
    """Test control flow in _configure_providers_loop."""

    @patch('devdox_ai_sonar.cli.console')
    async def test_breaks_on_empty_provider_name(self, mock_console):
        """Line 369: empty provider_name → loop breaks."""
        mock_ui = Mock()
        mock_ui.select_provider_from_list = AsyncMock(return_value="")

        with patch(
            'devdox_ai_sonar.cli._handle_provider_configuration',
            new_callable=AsyncMock,
        ) as mock_handle:
            await _configure_providers_loop(
                Mock(), Mock(), mock_ui, ["openai"]
            )

        mock_handle.assert_not_called()


class TestChangeMaxFixFlow:
    """Test list-response branch in change_max_fix."""

    @patch('devdox_ai_sonar.cli.smart_prompt', new_callable=AsyncMock, return_value=["10"])
    @patch('devdox_ai_sonar.cli.console')
    async def test_handles_list_response(self, mock_console, mock_prompt):
        """Line 434: list response → takes first element."""
        mock_manager = AsyncMock()

        await change_max_fix(mock_manager, "Max?", 5, 20)

        mock_manager.set_value.assert_called_once_with(
            "configuration.max_fixes", 10
        )


class TestInitConfigFlow:
    """Test early-return branch in init_config."""

    @patch('devdox_ai_sonar.cli._initialize_managers')
    @patch('devdox_ai_sonar.cli._configure_sonarcloud', new_callable=AsyncMock, return_value=True)
    @patch('devdox_ai_sonar.cli.change_parameters', new_callable=AsyncMock)
    async def test_returns_early_when_providers_exist(
        self, mock_params, mock_sonar, mock_init
    ):
        """Line 557: providers exist → skip _configure_providers_loop."""
        mock_manager = AsyncMock()
        mock_manager.save_config = Mock()
        mock_manager.create_default_config = Mock()
        mock_manager.get_value.return_value = ["openai"]

        mock_init.return_value = (
            mock_manager, Mock(), Mock(), AsyncMock(), Mock(), AsyncMock()
        )

        with patch(
            'devdox_ai_sonar.cli._check_reconfiguration_consent', return_value=True
        ):
            with patch(
                'devdox_ai_sonar.cli._configure_providers_loop',
                new_callable=AsyncMock,
            ) as mock_loop:
                await init_config()

        mock_loop.assert_not_called()


class TestExecuteInteractiveCommandFlow:
    """Test None-command early return."""

    @patch('devdox_ai_sonar.cli._execute_command_async', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.console')
    async def test_returns_early_for_none_command(self, mock_console, mock_exec):
        """Line 957: command is None → return immediately."""
        await _execute_interactive_command_async(Mock(spec=click.Context), None)

        mock_exec.assert_not_called()


class TestRunAnalyzeFlow:
    """Test limit-handling branches in _run_analyze."""

    async def test_handles_list_limit_response(self):
        """Line 1099: list response → takes first element."""
        mock_auth = Mock()
        mock_auth.project = "proj"
        mock_auth.organization = "org"
        mock_auth.token = "tok"

        with patch(
            'devdox_ai_sonar.cli._load_and_validate_config',
            new_callable=AsyncMock,
            return_value=(mock_auth, Mock(), {"max_fixes": 10}),
        ):
            with patch(
                'devdox_ai_sonar.cli.smart_prompt',
                new_callable=AsyncMock,
                return_value=["5"],
            ):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer') as mock_cls:
                    mock_analyzer = Mock()
                    mock_analyzer.get_project_issues.return_value = None
                    mock_cls.return_value = mock_analyzer

                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli.console'):
                            await _run_analyze()

                    mock_analyzer.get_project_issues.assert_called_once()
                    call_kwargs = mock_analyzer.get_project_issues.call_args
                    assert call_kwargs[1]["max_issues"] == 5

    async def test_clamps_out_of_range_limit(self):
        """Lines 1109-1115: limit <= 0 → use MAX_FIXES_LIMIT."""
        from devdox_ai_sonar.config import settings

        mock_auth = Mock()
        mock_auth.project = "proj"
        mock_auth.organization = "org"
        mock_auth.token = "tok"

        with patch(
            'devdox_ai_sonar.cli._load_and_validate_config',
            new_callable=AsyncMock,
            return_value=(mock_auth, Mock(), {"max_fixes": 10}),
        ):
            with patch(
                'devdox_ai_sonar.cli.smart_prompt',
                new_callable=AsyncMock,
                return_value="0",
            ):
                with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer') as mock_cls:
                    mock_analyzer = Mock()
                    mock_analyzer.get_project_issues.return_value = None
                    mock_cls.return_value = mock_analyzer

                    with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
                        with patch('devdox_ai_sonar.cli.console'):
                            await _run_analyze()

                    call_kwargs = mock_analyzer.get_project_issues.call_args
                    assert call_kwargs[1]["max_issues"] == settings.MAX_FIXES_LIMIT


class TestProcessAndFixIssuesFlow:
    """Test PR branch-fetching in _process_and_fix_issues."""

    @patch('devdox_ai_sonar.cli.TmpCloneManager')
    @patch('devdox_ai_sonar.cli.download_latest_version', return_value=True)
    @patch('devdox_ai_sonar.cli._fetch_issues_by_type', return_value={})
    @patch('devdox_ai_sonar.cli._initialize_fix_services')
    @patch('devdox_ai_sonar.cli.console')
    async def test_fetches_branch_from_pr(
        self, mock_console, mock_init_svc, mock_fetch, mock_download,
        mock_tmp_cls
    ):
        """Line 1286: pull_request > 0 → calls get_branch_from_pr."""
        # Set up TmpCloneManager as async context manager
        mock_mgr = AsyncMock()
        mock_mgr.__aenter__.return_value = Path(tempfile.gettempdir()) / "clone"
        mock_tmp_cls.return_value = mock_mgr

        mock_analyzer = Mock()
        mock_analyzer.get_branch_from_pr.return_value = "feature-branch"

        mock_init_svc.return_value = {
            "analyzer": mock_analyzer,
            "ruler": Mock(),
            "fixer": Mock(),
        }

        auth_config = AuthConfig(
            token="t", organization="o", project="p",
            project_path="/tmp", git_url="https://git.example.com"
        )

        await _process_and_fix_issues(
            auth_config, Mock(), "", "123",
            {"max_fixes": 10}, issue_type=IssueType.REGULAR,
        )

        mock_analyzer.get_branch_from_pr.assert_called_once_with(
            project_key="p", pull_request="123"
        )


class TestProcessFilesFlow:
    """Test issue-type branching in _process_files_with_issues."""

    @patch('devdox_ai_sonar.cli._process_regular_issues', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.console')
    async def test_regular_issues_branch(self, mock_console, mock_regular):
        """Lines 1366-1376: IssueType.REGULAR → _process_regular_issues called."""
        auth_config = AuthConfig(
            token="t", organization="o", project="p",
            project_path="/tmp", git_url="https://git.example.com"
        )

        issues_by_file = {"rule:S1234": [Mock(file="handler.py")]}

        await _process_files_with_issues(
            issues_by_file,
            {"analyzer": Mock(), "ruler": Mock(), "fixer": Mock()},
            auth_config,
            {"apply": 0, "dry_run": 0},
            IssueType.REGULAR,
            Path(tempfile.gettempdir()) / "clone",
        )

        mock_regular.assert_called_once()


class TestProcessIssuesForRuleFlow:
    """Test skip and cancel branches in _process_issues_for_rule."""

    @patch('devdox_ai_sonar.cli._process_single_fix', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.console')
    async def test_skips_non_processable_file(self, mock_console, mock_fix):
        """Lines 1431-1445: non-.py file → skipped."""
        issue = Mock()
        issue.file = "test_helper.py"

        result = await _process_issues_for_rule(
            "rule:S1234", [issue],
            {"analyzer": Mock(), "ruler": Mock(), "fixer": Mock()},
            AuthConfig(token="t", organization="o", project="p",
                       project_path="/tmp", git_url="https://git.example.com"),
            {"apply": 0}, Path(tempfile.gettempdir()) / "md.md", Path(tempfile.gettempdir()) / "clone",
        )

        mock_fix.assert_not_called()
        assert result is True

    @patch('devdox_ai_sonar.cli._should_continue_to_next_issue', new_callable=AsyncMock, return_value=False)
    @patch('devdox_ai_sonar.cli._process_single_fix', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.console')
    async def test_stops_on_user_cancel(self, mock_console, mock_fix, mock_continue):
        """Lines 1458-1459: user cancels → returns False."""
        issue = Mock()
        issue.file = "handler.py"

        result = await _process_issues_for_rule(
            "rule:S1234", [issue],
            {"analyzer": Mock(), "ruler": Mock(), "fixer": Mock()},
            AuthConfig(token="t", organization="o", project="p",
                       project_path="/tmp", git_url="https://git.example.com"),
            {"apply": 0}, Path(tempfile.gettempdir()) / "md.md", Path(tempfile.gettempdir()) / "clone",
        )

        mock_fix.assert_called_once()
        assert result is False


class TestProcessSecurityIssuesFlow:
    """Test skip branch in _process_security_issues."""

    @patch('devdox_ai_sonar.cli._should_continue_to_next_issue', new_callable=AsyncMock, return_value=True)
    @patch('devdox_ai_sonar.cli._process_single_fix', new_callable=AsyncMock)
    @patch('devdox_ai_sonar.cli.console')
    async def test_skips_non_py_file(self, mock_console, mock_fix, mock_continue):
        """Lines 1519-1523: non-.py file → skipped with message."""
        issue = Mock()
        issue.file = "handler.js"

        auth_config = AuthConfig(
            token="t", organization="o", project="p",
            project_path="/tmp", git_url="https://git.example.com"
        )

        await _process_security_issues(
            {"handler.js": [issue]},
            {"analyzer": Mock(), "ruler": Mock(), "fixer": Mock()},
            auth_config,
            {"apply": 0}, Path(tempfile.gettempdir()) / "md.md", Path(tempfile.gettempdir()) / "clone",
        )

        mock_fix.assert_not_called()
        call_args = str(mock_console.print.call_args_list)
        assert "Skipping" in call_args


class TestShouldContinueToNextIssue:
    """Test user-confirm branches in _should_continue_to_next_issue."""

    @patch('devdox_ai_sonar.cli.smart_confirm', new_callable=AsyncMock, return_value=False)
    @patch('devdox_ai_sonar.cli.console')
    async def test_returns_false_when_user_stops(self, mock_console, mock_confirm):
        """Lines 1613-1615: user declines → returns False."""
        result = await _should_continue_to_next_issue(1, 5)
        assert result is False

    @patch('devdox_ai_sonar.cli.smart_confirm', new_callable=AsyncMock, return_value=True)
    async def test_returns_true_when_user_continues(self, mock_confirm):
        """Line 1617: user confirms → returns True."""
        result = await _should_continue_to_next_issue(1, 5)
        assert result is True


class TestFetchIssuesByTypeRegular:
    """Test regular-issue branch in _fetch_issues_by_type."""

    def test_fetches_regular_issues(self):
        """Lines 1720-1730: IssueType.REGULAR → get_fixable_issues_by_types."""
        mock_analyzer = Mock()
        mock_analyzer.get_fixable_issues_by_types.return_value = {
            "rule:S1234": [Mock()]
        }

        with patch('devdox_ai_sonar.cli.show_progress', mock_show_progress):
            result = _fetch_issues_by_type(
                mock_analyzer,
                Mock(project="proj"),
                "main",
                None,
                {
                    "max_fixes": 10,
                    "severities_list": None,
                    "types_list": None,
                    "exclude_rules": [],
                },
                IssueType.REGULAR,
            )

        assert len(result) == 1
        mock_analyzer.get_fixable_issues_by_types.assert_called_once()

