"""Comprehensive tests for CLI commands based on actual implementation."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import click
from click.testing import CliRunner
from devdox_ai_sonar.cli import main, analyze, fix, inspect, _parse_filters, _get_severity_color, select_fixes_interactively
from devdox_ai_sonar.models import (
    SonarIssue,
    FixSuggestion,
    AnalysisResult,
    ProjectMetrics,
    FixResult,
    Severity,
    IssueType,
)


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_analyzer():
    """Mock SonarCloudAnalyzer."""
    analyzer = MagicMock()
    return analyzer


@pytest.fixture
def mock_llm_fixer():
    """Mock LLMFixer."""
    fixer = MagicMock()
    fixer.provider = "openai"
    fixer.model = "gpt-4"
    fixer.api_key = "test-key"
    return fixer


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    issue = SonarIssue(
        key="test:src/test.py:S1481",
        rule="python:S1481",
        severity=Severity.MAJOR,
        component="test:src/test.py",
        project="test-project",
        first_line=10,
        last_line=10,
        message="Test issue",
        type=IssueType.CODE_SMELL,
        file="src/test.py",
    )
    
    return AnalysisResult(
        project_key="test-project",
        organization="test-org",
        branch="main",
        total_issues=1,
        issues=[issue],
        metrics=ProjectMetrics(
            project_key="test-project",
            lines_of_code=1000,
            coverage=85.5,
            bugs=5,
            vulnerabilities=2,
            code_smells=15,
        ),
    )


@pytest.fixture
def sample_fix_suggestion():
    """Sample fix suggestion."""
    return FixSuggestion(
        issue_key="test:src/test.py:S1481",
        original_code="unused_var = 42",
        fixed_code="# removed unused variable",
        explanation="Removed unused variable",
        confidence=0.95,
        llm_model="gpt-4",
        file_path="src/test.py",
        line_number=10,
        sonar_line_number=10,
        last_line_number=10,
    )


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_main_help(self, runner):
        """Test main command help."""
        from devdox_ai_sonar.cli import main

        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SonarCloud Analyzer" in result.output or "Usage" in result.output

    def test_main_version(self, runner):
        """Test version option."""
        from devdox_ai_sonar.cli import main

        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_main_verbose_flag(self, runner):
        """Test verbose flag is accepted."""
        from devdox_ai_sonar.cli import main

        result = runner.invoke(main, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestAnalyzeCommand:
    """Test analyze command."""

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_basic(self, mock_analyzer_class, runner, sample_analysis_result):
        """Test basic analyze command."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
            ],
        )

        assert result.exit_code == 0
        mock_analyzer.get_project_issues.assert_called_once()

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_branch(self, mock_analyzer_class, runner, sample_analysis_result):
        """Test analyze with branch parameter."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--branch", "develop",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_analyzer.get_project_issues.call_args
        assert call_args[1]["branch"] == "develop"

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_pull_request(self, mock_analyzer_class, runner, sample_analysis_result):
        """Test analyze with pull request parameter."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--pull-request", "123",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_severity_filter(self, mock_analyzer_class, runner, sample_analysis_result):
        """Test analyze with severity filter."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--severity", "BLOCKER",
                "--severity", "CRITICAL",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_type_filter(self, mock_analyzer_class, runner, sample_analysis_result):
        """Test analyze with type filter."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--type", "BUG",
                "--type", "VULNERABILITY",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_limit(self, mock_analyzer_class, runner, sample_analysis_result):
        """Test analyze with issue limit."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--limit", "10",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_output_file(self, mock_analyzer_class, runner, sample_analysis_result, tmp_path):
        """Test analyze with output file."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        output_file = tmp_path / "results.json"

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--output", str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        
        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
            assert data["project_key"] == "test-project"

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_no_issues_found(self, mock_analyzer_class, runner):
        """Test analyze when no issues are returned."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.return_value = None
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
            ],
        )

        assert result.exit_code == 1
        assert "Failed to fetch issues" in result.output

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_exception_handling(self, mock_analyzer_class, runner):
        """Test analyze error handling."""
        from devdox_ai_sonar.cli import analyze

        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.side_effect = Exception("API Error")
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
            ],
        )

        assert result.exit_code == 1
        assert "Error" in result.output


class TestFixCommand:
    """Test fix command."""

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_basic(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path, sample_fix_suggestion):
        """Test basic fix command."""
        from devdox_ai_sonar.cli import fix

        # Setup mocks
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.api_key = "test-key"
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--provider", "openai",
                "--api-key", "test-api-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_with_provider_options(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix command with different providers."""
        from devdox_ai_sonar.cli import fix

        providers = ["openai", "gemini", "togetherai"]
        
        for provider in providers:
            mock_analyzer = MagicMock()
            mock_analyzer.get_fixable_issues.return_value = []
            mock_analyzer_class.return_value = mock_analyzer

            mock_fixer = MagicMock()
            mock_fixer.provider = provider
            mock_fixer.model = "test-model"
            mock_fixer.api_key = "test-key"
            mock_fixer_class.return_value = mock_fixer

            result = runner.invoke(
                fix,
                [
                    "--token", "test-token",
                    "--organization", "test-org",
                    "--project", "test-project",
                    "--project-path", str(tmp_path),
                    "--provider", provider,
                    "--api-key", "test-key",
                ],
            )

            assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_with_severity_filter(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix with severity filter."""
        from devdox_ai_sonar.cli import fix

        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.api_key = "test-key"
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--severity", "BLOCKER,CRITICAL",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_with_types_filter(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix with types filter."""
        from devdox_ai_sonar.cli import fix

        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.api_key = "test-key"
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--types", "BUG,CODE_SMELL",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_invalid_severity(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix with invalid severity."""
        from devdox_ai_sonar.cli import fix

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--severity", "INVALID",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid severities" in result.output

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_invalid_type(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix with invalid type."""
        from devdox_ai_sonar.cli import fix

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--types", "INVALID_TYPE",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid issue types" in result.output

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_dry_run(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix in dry-run mode."""
        from devdox_ai_sonar.cli import fix

        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.api_key = "test-key"
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--dry-run",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_no_backup(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix without backup."""
        from devdox_ai_sonar.cli import fix

        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.api_key = "test-key"
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--no-backup",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    def test_main_invocation(self,runner):
        """Test that the main CLI can be invoked with no commands."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SonarCloud Analyzer" in result.output

    def test_parse_filters_empty(self):
        """Test that empty filters return None."""
        severity_list, types_list = _parse_filters(None, None, {"BUG"}, {"BLOCKER"})
        assert severity_list is None
        assert types_list is None

    def test_get_severity_color_default(self):
        """Test unknown severity returns white."""
        from devdox_ai_sonar.models import Severity
        class FakeSeverity:
            pass

        assert _get_severity_color(FakeSeverity()) == "white"

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_invalid_token(self, mock_analyzer_class, runner):
        """Test analyze exits on exception."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_project_issues.side_effect = Exception("Invalid token")
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(analyze, ["--token", "bad", "--organization", "org", "--project", "proj"])
        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_inspect_nonexistent_path(self, runner):
        """Test inspect exits on invalid path."""
        result = runner.invoke(inspect, ["/does/not/exist"])
        assert result.exit_code != 0

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_fixes_invalid_input(self,mock_prompt, sample_fix_suggestion):
        """Test interactive fix selection with invalid input."""
        mock_prompt.return_value = "invalid"
        selected = select_fixes_interactively([sample_fix_suggestion])
        assert selected == []

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_dry_run_behavior(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix dry-run executes without applying changes."""
        from devdox_ai_sonar.cli import fix
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(fix, [
            "--token", "t", "--organization", "o", "--project", "p",
            "--project-path", str(tmp_path),
            "--dry-run",
            "--provider", "openai",
            "--api-key", "key"
        ])
        assert result.exit_code == 0

    def test_parse_filters_invalid_severity_type(self):
        """Test _parse_filters raises BadParameter on invalid inputs."""

        with pytest.raises(click.BadParameter):
            _parse_filters("INVALID", None, {"BUG"}, {"BLOCKER"})
        with pytest.raises(click.BadParameter):
            _parse_filters(None, "INVALID", {"BUG"}, {"BLOCKER"})


class TestInspectCommand:
    """Test inspect command."""

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_inspect_basic(self, mock_analyzer_class, runner, tmp_path):
        """Test basic inspect command."""
        from devdox_ai_sonar.cli import inspect

        # Create test project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").write_text("print('test')")
        (tmp_path / ".git").mkdir()

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_project_directory.return_value = {
            "path": str(tmp_path),
            "total_files": 1,
            "python_files": 1,
            "javascript_files": 0,
            "java_files": 0,
            "other_files": 0,
            "directories": ["src"],
            "has_sonar_config": False,
            "has_git": True,
            "potential_source_dirs": ["src"],
        }
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(inspect, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Project Analysis" in result.output

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_inspect_nonexistent_path(self, mock_analyzer_class, runner):
        """Test inspect with nonexistent path."""
        from devdox_ai_sonar.cli import inspect

        result = runner.invoke(inspect, ["/nonexistent/path"])

        assert result.exit_code != 0

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_inspect_exception_handling(self, mock_analyzer_class, runner, tmp_path):
        """Test inspect error handling."""
        from devdox_ai_sonar.cli import inspect

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_project_directory.side_effect = Exception("Analysis error")
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(inspect, [str(tmp_path)])

        assert result.exit_code == 1
        assert "Error" in result.output


class TestHelperFunctions:
    """Test CLI helper functions."""

    def test_parse_filters_valid_severity(self):
        """Test parsing valid severity filters."""
        from devdox_ai_sonar.cli import _parse_filters

        valid_types = {"BUG", "CODE_SMELL"}
        valid_severities = {"BLOCKER", "CRITICAL", "MAJOR"}

        severity_list, types_list = _parse_filters(
            "BLOCKER,CRITICAL",
            None,
            valid_types,
            valid_severities
        )

        assert severity_list == ["BLOCKER", "CRITICAL"]
        assert types_list is None

    def test_parse_filters_valid_types(self):
        """Test parsing valid type filters."""
        from devdox_ai_sonar.cli import _parse_filters

        valid_types = {"BUG", "CODE_SMELL"}
        valid_severities = {"BLOCKER", "CRITICAL"}

        severity_list, types_list = _parse_filters(
            None,
            "BUG,CODE_SMELL",
            valid_types,
            valid_severities
        )

        assert severity_list is None
        assert types_list == ["BUG", "CODE_SMELL"]

    def test_parse_filters_invalid_severity(self):
        """Test parsing invalid severity."""
        from devdox_ai_sonar.cli import _parse_filters
        from click import BadParameter

        valid_types = {"BUG"}
        valid_severities = {"BLOCKER"}

        with pytest.raises(BadParameter, match="Invalid severities"):
            _parse_filters("INVALID", None, valid_types, valid_severities)

    def test_parse_filters_invalid_type(self):
        """Test parsing invalid type."""
        from devdox_ai_sonar.cli import _parse_filters
        from click import BadParameter

        valid_types = {"BUG"}
        valid_severities = {"BLOCKER"}

        with pytest.raises(BadParameter, match="Invalid issue types"):
            _parse_filters(None, "INVALID", valid_types, valid_severities)

    def test_get_severity_color(self):
        """Test severity color mapping."""
        from devdox_ai_sonar.cli import _get_severity_color

        assert _get_severity_color(Severity.BLOCKER) == "red"
        assert _get_severity_color(Severity.CRITICAL) == "red"
        assert _get_severity_color(Severity.MAJOR) == "yellow"
        assert _get_severity_color(Severity.MINOR) == "blue"
        assert _get_severity_color(Severity.INFO) == "green"


class TestDisplayFunctions:
    """Test display helper functions."""

    def test_display_analysis_results(self, sample_analysis_result):
        """Test displaying analysis results."""
        from devdox_ai_sonar.cli import _display_analysis_results

        # Should not raise any exceptions
        _display_analysis_results(sample_analysis_result, limit=None)

    def test_display_analysis_results_with_limit(self, sample_analysis_result):
        """Test displaying analysis results with limit."""
        from devdox_ai_sonar.cli import _display_analysis_results

        _display_analysis_results(sample_analysis_result, limit=5)

    def test_display_fix_results(self):
        """Test displaying fix results."""
        from devdox_ai_sonar.cli import _display_fix_results

        result = FixResult(
            project_path=Path("/test"),
            total_fixes_attempted=10,
            successful_fixes=[],
            failed_fixes=[{"error": "Test error"}],
            backup_created=True,
            backup_path=Path("/test/backup"),
        )

        _display_fix_results(result)

    def test_save_results(self, sample_analysis_result, tmp_path):
        """Test saving analysis results."""
        from devdox_ai_sonar.cli import _save_results

        output_file = tmp_path / "test_results.json"
        _save_results(sample_analysis_result, str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert data["project_key"] == "test-project"


class TestSelectFixesInteractively:
    """Test interactive fix selection."""

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_all_fixes(self, mock_prompt, sample_fix_suggestion):
        """Test selecting all fixes."""
        from devdox_ai_sonar.cli import select_fixes_interactively

        mock_prompt.return_value = "all"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 1
        assert selected[0] == sample_fix_suggestion

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_no_fixes(self, mock_prompt, sample_fix_suggestion):
        """Test selecting no fixes."""
        from devdox_ai_sonar.cli import select_fixes_interactively

        mock_prompt.return_value = "none"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 0

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_specific_fixes(self, mock_prompt, sample_fix_suggestion):
        """Test selecting specific fixes by number."""
        from devdox_ai_sonar.cli import select_fixes_interactively

        mock_prompt.return_value = "1"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 1

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_invalid_input(self, mock_prompt, sample_fix_suggestion):
        """Test handling invalid input."""
        from devdox_ai_sonar.cli import select_fixes_interactively

        mock_prompt.return_value = "invalid"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
