

import pytest
from contextlib import contextmanager
import click
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
from devdox_ai_sonar.cli import (
    main,
    _safe_convert_pr,
    smart_prompt,
    SwitchCommandException,
    ReturnToMenuException,
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
)
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    AnalysisResult,
    FixSuggestion,
    FixResult,
    ProjectMetrics,
    Severity
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def runner():
    """Create CLI test runner"""
    return CliRunner()


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
        sonar_line_number=10
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

    def test_return_to_menu_exception(self):
        """Test ReturnToMenuException"""
        exc = ReturnToMenuException()
        assert isinstance(exc, Exception)


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
# TEST CLASS: SMART PROMPTS
# ============================================================================

class TestSmartPrompts:
    """Test smart prompt functions"""

    def test_smart_prompt_text_with_questionary(self):
        """Test text prompt with questionary"""
        with patch('devdox_ai_sonar.cli.questionary') as mock_q:
            mock_q.text.return_value.ask.return_value = "test_input"

            result = smart_prompt("Enter value:")

            assert result == "test_input"
            mock_q.text.assert_called_once()

    def test_smart_prompt_select_with_questionary(self):
        """Test select prompt with questionary"""
        with patch('devdox_ai_sonar.cli.questionary') as mock_q:
            mock_q.select.return_value.ask.return_value = "option1"

            result = smart_prompt(
                "Select option:",
                choices=["option1", "option2"]
            )

            assert result == "option1"

    def test_smart_prompt_checkbox_with_questionary(self):
        """Test checkbox prompt with questionary"""
        with patch('devdox_ai_sonar.cli.questionary') as mock_q:
            mock_q.checkbox.return_value.ask.return_value = ["opt1", "opt2"]

            result = smart_prompt(
                "Select multiple:",
                choices=["opt1", "opt2", "opt3"],
                multiple=True
            )

            assert result == ["opt1", "opt2"]

    def test_smart_prompt_switch_command(self):
        """Test switch command trigger"""
        with patch('devdox_ai_sonar.cli.questionary') as mock_q:
            mock_q.text.return_value.ask.return_value = "/"

            with pytest.raises(SwitchCommandException):
                smart_prompt("Enter value:", allow_switch=True)




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
            self, runner, mock_config_manager, mock_provider_manager
    ):
        """Test adding new provider"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            with patch('devdox_ai_sonar.cli._handle_add_new_provider'):
                with patch('devdox_ai_sonar.cli._display_completion_message'):
                    ctx = Mock()
                    add_provider(ctx)

    def test_update_provider_success(
            self, runner, mock_config_manager, mock_provider_manager
    ):
        """Test updating existing provider"""
        with patch('devdox_ai_sonar.cli._initialize_managers') as mock_init:
            mock_init.return_value = (
                mock_config_manager,
                Mock(),
                Mock(),
                mock_provider_manager,
                Mock(),
                Mock()
            )

            with patch('devdox_ai_sonar.cli._handle_update_existing_provider'):
                with patch('devdox_ai_sonar.cli._display_completion_message'):
                    ctx = Mock()
                    update_provider(ctx)

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

                        _run_fix_issues(ctx)

    def test_run_fix_issues_cancelled(
            self, mock_config_service, mock_llm_config
    ):
        """Test fix_issues when user cancels"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {})

            with patch('devdox_ai_sonar.cli.display_configuration', return_value={}):
                with patch('devdox_ai_sonar.cli.smart_confirm', return_value=False):
                    ctx = Mock()
                    ctx.obj = {"verbose": False}

                    _run_fix_issues(ctx)

    def test_run_fix_issues_switch_command(self):
        """Test fix_issues with command switch"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config', side_effect=SwitchCommandException):
            ctx = Mock()
            ctx.obj = {"verbose": False}

            with pytest.raises(SwitchCommandException):
                _run_fix_issues(ctx)

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
                    with patch('devdox_ai_sonar.cli._process_security_issues'):
                        ctx = Mock()
                        ctx.obj = {"verbose": False}

                        _run_fix_security_issues(ctx)

    def test_run_fix_security_issues_cancelled(self, mock_llm_config):
        """Test security fix when cancelled"""
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.return_value = (Mock(), mock_llm_config, {})

            with patch('devdox_ai_sonar.cli.display_configuration', return_value={}):
                with patch('devdox_ai_sonar.cli.smart_confirm', return_value=False):
                    ctx = Mock()
                    ctx.obj = {"verbose": False}

                    _run_fix_security_issues(ctx)


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
                            

                            ctx = Mock()
                            ctx.obj = {"verbose": False}

                            _run_analyze(ctx)

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
                        

                        ctx = Mock()
                        ctx.obj = {"verbose": False}

                        _run_analyze(ctx)

                        # Should still call analyzer
                        mock_analyzer.get_project_issues.assert_called_once()

    def test_run_analyze_no_config(self):
        """Test analyze without config - _load_and_validate_config raises Abort"""


        # _load_and_validate_config will raise click.Abort if no config
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.side_effect = click.Abort()

            

            ctx = Mock()
            ctx.obj = {"verbose": False}

            with pytest.raises(click.Abort):
                _run_analyze(ctx)

    def test_run_inspect_no_config(self):
        """Test inspect without config - _load_and_validate_config raises Abort"""

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.side_effect = click.Abort()

            ctx = Mock()
            ctx.obj = {"verbose": False}

            with pytest.raises(click.Abort):
                _run_inspect(ctx)


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

                                ctx = Mock()
                                ctx.obj = {"verbose": False}

                                _run_analyze(ctx)

                                # Should validate filters
                                mock_sev.assert_called()
                                mock_types.assert_called()



# ============================================================================
# TEST CLASS: INSPECT COMMAND
# ============================================================================

class TestInspectCommand:
    """Test inspect command"""

    def test_run_inspect_success(self, mock_config_service, mock_analyzer):
        """Test successful inspection"""
        with patch('devdox_ai_sonar.cli.ConfigService') as mock_cs:
            mock_cs.return_value = mock_config_service

            with patch('devdox_ai_sonar.cli.SonarCloudAnalyzer') as mock_sa:
                mock_sa.return_value = mock_analyzer

                ctx = Mock()
                ctx.obj = {"verbose": False}

                _run_inspect(ctx)

    def test_run_inspect_no_config(self):
        """Test inspect without config - _load_and_validate_config raises Abort"""
        import click

        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            mock_load.side_effect = click.Abort()


            ctx = Mock()
            ctx.obj = {"verbose": False}

            with pytest.raises(click.Abort):
                _run_inspect(ctx)

    def test_run_inspect_invalid_config(self):
        """Test inspect with invalid config - EXPECTS click.Abort"""

        # Mock _load_and_validate_config to raise click.Abort (as it should!)
        with patch('devdox_ai_sonar.cli._load_and_validate_config') as mock_load:
            # When config is invalid, _load_and_validate_config raises click.Abort
            mock_load.side_effect = click.Abort()


            ctx = Mock()
            ctx.obj = {"verbose": False}

            # Assert that click.Abort is raised (this is CORRECT behavior!)
            with pytest.raises(click.Abort):
                _run_inspect(ctx)

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

                ctx = Mock()
                ctx.obj = {"verbose": False}

                _run_inspect(ctx)

                # Should still complete successfully
                mock_analyzer.analyze_project_directory.assert_called_once()


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
                            fix_params
                        )

                        # Verify the analyzer was called correctly
                        mock_analyzer.get_fixable_issues_by_files.assert_called_once()

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
                        from devdox_ai_sonar.cli import _process_security_issues

                        _process_security_issues(
                            auth_config,
                            llm_config,
                            "main",
                            None,
                            fix_params
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
                                from devdox_ai_sonar.cli import _process_security_issues

                                _process_security_issues(
                                    auth_config,
                                    llm_config,
                                    "main",
                                    None,
                                    fix_params
                                )

                                mock_fixer.generate_fix_by_file.assert_called_once()


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
                            from devdox_ai_sonar.cli import _run_analyze

                            ctx = Mock()
                            ctx.obj = {"verbose": False}

                            _run_analyze(ctx)

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

                ctx = Mock()
                ctx.obj = {"verbose": False}

                _run_inspect(ctx)
