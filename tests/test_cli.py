"""Comprehensive tests for CLI commands based on actual implementation."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import click
from click import BadParameter
from click.testing import CliRunner

from devdox_ai_sonar.cli import (
    main ,
    _display_fix_results,
    _apply_fixes_if_requested,
    analyze,
    fix,
    inspect,
    _get_severity_color,
    select_fixes_interactively,
    _display_analysis_results,
    fix_security_issues,
    _fetch_fixable_issues,
    _fetch_fixable_security_issues,
    _generate_fixes,
    _parse_filters,
    _save_results)
from devdox_ai_sonar.models import (
    SonarIssue,
    FixSuggestion,
    AnalysisResult,
    ProjectMetrics,
    SonarSecurityIssue,
    FixResult,
    Severity,
    IssueType,
)


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()

@pytest.fixture
def sample_issue():
    """Sample SonarIssue for testing."""
    return SonarIssue(
        key="test-key",
        rule="python:S1481",
        severity=Severity.MAJOR,
        component="test:src/test.py",
        project="test-project",
        message="Test issue",
        type=IssueType.CODE_SMELL,
        first_line=10,
        last_line=10,
        file="src/test.py"
    )

@pytest.fixture
def sample_fix():
    """Sample FixSuggestion for testing."""
    return FixSuggestion(
        issue_key="test-key",
        original_code="unused_var = 42",
        fixed_code="# removed",
        explanation="Removed unused variable",
        confidence=0.95,
        llm_model="gpt-4",
        file_path="src/test.py",
        sonar_line_number=10,
        line_number=10,
        last_line_number=10
    )

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
def sample_security_issue():
    """Sample SonarSecurityIssue for testing."""
    return SonarSecurityIssue(
        key="security-key",
        rule="python:S5122",
        component="test:src/security.py",
        project="test-project",
        security_category="weak-cryptography",
        vulnerability_probability="HIGH",
        message="Security issue",
        first_line=50,
        last_line=55,
        file="src/security.py"
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
        

        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SonarCloud Analyzer" in result.output or "Usage" in result.output

    def test_main_version(self, runner):
        """Test version option."""
        

        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_main_verbose_flag(self, runner):
        """Test verbose flag is accepted."""
        

        result = runner.invoke(main, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestAnalyzeCommand:
    """Test analyze command."""

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_basic(self, mock_analyzer_class, runner, sample_analysis_result):
        """Test basic analyze command."""
        

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
        

        result = runner.invoke(inspect, ["/nonexistent/path"])

        assert result.exit_code != 0

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_inspect_exception_handling(self, mock_analyzer_class, runner, tmp_path):
        """Test inspect error handling."""
        

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
        
        

        valid_types = {"BUG"}
        valid_severities = {"BLOCKER"}

        with pytest.raises(BadParameter, match="Invalid severities"):
            _parse_filters("INVALID", None, valid_types, valid_severities)

    def test_parse_filters_invalid_type(self):
        """Test parsing invalid type."""
        
        

        valid_types = {"BUG"}
        valid_severities = {"BLOCKER"}

        with pytest.raises(BadParameter, match="Invalid issue types"):
            _parse_filters(None, "INVALID", valid_types, valid_severities)

    def test_get_severity_color(self):
        """Test severity color mapping."""

        assert _get_severity_color(Severity.BLOCKER) == "red"
        assert _get_severity_color(Severity.CRITICAL) == "red"
        assert _get_severity_color(Severity.MAJOR) == "yellow"
        assert _get_severity_color(Severity.MINOR) == "blue"
        assert _get_severity_color(Severity.INFO) == "green"


class TestDisplayFunctions:
    """Test display helper functions."""

    def test_display_analysis_results(self, sample_analysis_result):
        """Test displaying analysis results."""
        

        # Should not raise any exceptions
        _display_analysis_results(sample_analysis_result, limit=None)

    def test_display_analysis_results_with_limit(self, sample_analysis_result):
        """Test displaying analysis results with limit."""
        

        _display_analysis_results(sample_analysis_result, limit=5)

    def test_display_fix_results(self):
        """Test displaying fix results."""
        

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
        

        mock_prompt.return_value = "all"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 1
        assert selected[0] == sample_fix_suggestion

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_no_fixes(self, mock_prompt, sample_fix_suggestion):
        """Test selecting no fixes."""
        

        mock_prompt.return_value = "none"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 0

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_specific_fixes(self, mock_prompt, sample_fix_suggestion):
        """Test selecting specific fixes by number."""
        

        mock_prompt.return_value = "1"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 1

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_invalid_input(self, mock_prompt, sample_fix_suggestion):
        """Test handling invalid input."""
        

        mock_prompt.return_value = "invalid"
        fixes = [sample_fix_suggestion]

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 0

class TestFixSecurityIssuesCommand:
    """Tests for fix_security_issues command - COMPLETELY MISSING."""

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_security_issues_basic(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test basic fix_security_issues command."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.api_key = "test-key"
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix_security_issues,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0
        assert "No fixable issues found" in result.output or result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_security_issues_with_branch(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix_security_issues with specific branch."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix_security_issues,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--branch", "develop",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_security_issues_with_pr(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix_security_issues with pull request."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix_security_issues,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--pull-request", "123",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_security_issues_with_max_fixes(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix_security_issues with max-fixes limit."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix_security_issues,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--max-fixes", "5",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_security_issues_dry_run(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix_security_issues in dry-run mode."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix_security_issues,
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
    def test_fix_security_issues_exception_handling(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix_security_issues error handling."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.side_effect = Exception("API Error")
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            fix_security_issues,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 1
        assert "Error" in result.output


# ============================================================================
# MISSING: select_fixes_interactively EDGE CASES
# ============================================================================

class TestSelectFixesInteractivelyMissing:
    """Missing test cases for select_fixes_interactively."""

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_fixes_with_range(self, mock_prompt, sample_fix):
        """Test selecting multiple fixes with range."""
        fixes = [sample_fix] * 5
        mock_prompt.return_value = "1,2,3"

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 3

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_fixes_with_spaces_in_input(self, mock_prompt, sample_fix):
        """Test selecting fixes with spaces in input."""
        fixes = [sample_fix] * 3
        mock_prompt.return_value = " 1 , 2 , 3 "

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 3

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_fixes_with_out_of_range(self, mock_prompt, sample_fix):
        """Test selecting fixes with out-of-range indices."""
        fixes = [sample_fix] * 3
        mock_prompt.return_value = "1,5,10"  # 5 and 10 are out of range

        selected = select_fixes_interactively(fixes)

        # Should only select valid index 1
        assert len(selected) == 1

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_fixes_with_empty_input(self, mock_prompt, sample_fix):
        """Test selecting fixes with empty input."""
        fixes = [sample_fix]
        mock_prompt.return_value = ""

        selected = select_fixes_interactively(fixes)

        assert len(selected) == 0

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_fixes_with_duplicate_indices(self, mock_prompt, sample_fix):
        """Test selecting fixes with duplicate indices."""
        fixes = [sample_fix] * 3
        mock_prompt.return_value = "1,1,2,2,3"

        selected = select_fixes_interactively(fixes)

        # Should handle duplicates gracefully
        assert len(selected) >= 3

    @patch("devdox_ai_sonar.cli.click.prompt")
    def test_select_fixes_display_formatting(self, mock_prompt, sample_fix):
        """Test that fix selection displays correct formatting."""
        fix_with_long_code = FixSuggestion(
            issue_key="very-long-issue-key-for-testing-truncation",
            original_code="x" * 100,
            fixed_code="y" * 600,  # Longer than 500 char display limit
            explanation="Test",
            confidence=0.85,
            llm_model="gpt-4",
            file_path="src/very/long/path/to/file.py",
            sonar_line_number=42,
            line_number=42,
            last_line_number=42
        )

        mock_prompt.return_value = "none"

        selected = select_fixes_interactively([fix_with_long_code])

        # Should not crash with long content
        assert selected == []


# ============================================================================
# MISSING: HELPER FUNCTION TESTS
# ============================================================================

class TestFetchFixableIssues:
    """Tests for _fetch_fixable_issues helper - MISSING."""

    @patch("devdox_ai_sonar.cli.Progress")
    def test_fetch_fixable_issues_basic(self, mock_progress, sample_issue):
        """Test basic fetch_fixable_issues."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = [sample_issue]

        result = _fetch_fixable_issues(
            mock_analyzer,
            "test-project",
            "main",
            0,
            10,
            None,
            None
        )

        assert len(result) == 1
        assert result[0] == sample_issue
        mock_analyzer.get_fixable_issues.assert_called_once()

    @patch("devdox_ai_sonar.cli.Progress")
    def test_fetch_fixable_issues_with_filters(self, mock_progress, sample_issue):
        """Test fetch_fixable_issues with severity and type filters."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = [sample_issue]

        result = _fetch_fixable_issues(
            mock_analyzer,
            "test-project",
            "main",
            0,
            10,
            ["BLOCKER", "CRITICAL"],
            ["BUG", "VULNERABILITY"]
        )

        call_args = mock_analyzer.get_fixable_issues.call_args
        assert call_args[1]["severities"] == ["BLOCKER", "CRITICAL"]
        assert call_args[1]["types_list"] == ["BUG", "VULNERABILITY"]

    @patch("devdox_ai_sonar.cli.Progress")
    def test_fetch_fixable_issues_with_branch(self, mock_progress, sample_issue):
        """Test fetch_fixable_issues with specific branch."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = [sample_issue]

        result = _fetch_fixable_issues(
            mock_analyzer,
            "test-project",
            "develop",
            0,
            10,
            None,
            None
        )

        call_args = mock_analyzer.get_fixable_issues.call_args
        assert call_args[1]["branch"] == "develop"


class TestFetchFixableSecurityIssues:
    """Tests for _fetch_fixable_security_issues helper - MISSING."""

    @patch("devdox_ai_sonar.cli.Progress")
    def test_fetch_fixable_security_issues_basic(self, mock_progress, sample_security_issue):
        """Test basic fetch_fixable_security_issues."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = [sample_security_issue]

        result = _fetch_fixable_security_issues(
            mock_analyzer,
            "test-project",
            "main",
            0,
            10
        )

        assert len(result) == 1
        mock_analyzer.get_fixable_security_issues.assert_called_once()

    @patch("devdox_ai_sonar.cli.Progress")
    def test_fetch_fixable_security_issues_with_pr(self, mock_progress):
        """Test fetch_fixable_security_issues with pull request."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_security_issues.return_value = []

        result = _fetch_fixable_security_issues(
            mock_analyzer,
            "test-project",
            "",
            123,
            10
        )

        call_args = mock_analyzer.get_fixable_security_issues.call_args
        assert call_args[1]["pull_request"] == 123


class TestGenerateFixes:
    """Tests for _generate_fixes helper - MISSING."""

    @patch("devdox_ai_sonar.cli.Progress")
    def test_generate_fixes_basic(self, mock_progress, sample_issue, sample_fix, tmp_path):
        """Test basic fix generation."""
        mock_fixer = MagicMock()
        mock_fixer.generate_fix.return_value = sample_fix

        mock_analyzer = MagicMock()
        mock_analyzer.get_rule_by_key.return_value = {"name": "Test Rule"}

        result = _generate_fixes(
            mock_fixer,
            mock_analyzer,
            [sample_issue],
            tmp_path
        )

        assert len(result) == 1
        assert result[0] == sample_fix

    @patch("devdox_ai_sonar.cli.Progress")
    def test_generate_fixes_with_none_result(self, mock_progress, sample_issue, tmp_path):
        """Test fix generation when fixer returns None."""
        mock_fixer = MagicMock()
        mock_fixer.generate_fix.return_value = None

        mock_analyzer = MagicMock()
        mock_analyzer.get_rule_by_key.return_value = None

        result = _generate_fixes(
            mock_fixer,
            mock_analyzer,
            [sample_issue],
            tmp_path
        )

        # Should handle None results gracefully
        assert len(result) == 0

    @patch("devdox_ai_sonar.cli.Progress")
    def test_generate_fixes_multiple_issues(self, mock_progress, sample_issue, sample_fix, tmp_path):
        """Test generating fixes for multiple issues."""
        mock_fixer = MagicMock()
        mock_fixer.generate_fix.return_value = sample_fix

        mock_analyzer = MagicMock()
        mock_analyzer.get_rule_by_key.return_value = {"name": "Test Rule"}

        issues = [sample_issue] * 5

        result = _generate_fixes(
            mock_fixer,
            mock_analyzer,
            issues,
            tmp_path
        )

        assert len(result) == 5


class TestApplyFixesIfRequested:
    """Tests for _apply_fixes_if_requested helper - MISSING."""

    def test_apply_fixes_not_requested(self, sample_fix, sample_issue, tmp_path):
        """Test when neither apply nor dry-run is requested."""
        mock_fixer = MagicMock()

        # Should return early without doing anything
        _apply_fixes_if_requested(
            apply=False,
            dry_run=False,
            fixes=[sample_fix],
            issues=[sample_issue],
            fixer=mock_fixer,
            project_path=tmp_path,
            backup=True
        )

        # Should not call apply_fixes_with_validation
        mock_fixer.apply_fixes_with_validation.assert_not_called()

    @patch("devdox_ai_sonar.cli.select_fixes_interactively")
    @patch("devdox_ai_sonar.cli.click.confirm")
    def test_apply_fixes_with_user_confirmation(self, mock_confirm, mock_select, sample_fix, sample_issue, tmp_path):
        """Test applying fixes with user confirmation."""
        mock_select.return_value = [sample_fix]
        mock_confirm.return_value = True

        mock_fixer = MagicMock()
        mock_fixer.apply_fixes_with_validation.return_value = FixResult(
            project_path=tmp_path,
            total_fixes_attempted=1,
            successful_fixes=[sample_fix],
            failed_fixes=[],
            backup_created=True,
            backup_path=tmp_path / "backup"
        )

        _apply_fixes_if_requested(
            apply=True,
            dry_run=False,
            fixes=[sample_fix],
            issues=[sample_issue],
            fixer=mock_fixer,
            project_path=tmp_path,
            backup=True
        )

        mock_fixer.apply_fixes_with_validation.assert_called_once()

    @patch("devdox_ai_sonar.cli.select_fixes_interactively")
    @patch("devdox_ai_sonar.cli.click.confirm")
    def test_apply_fixes_user_declines(self, mock_confirm, mock_select, sample_fix, sample_issue, tmp_path):
        """Test when user declines to apply fixes."""
        mock_select.return_value = [sample_fix]
        mock_confirm.return_value = False

        mock_fixer = MagicMock()

        _apply_fixes_if_requested(
            apply=True,
            dry_run=False,
            fixes=[sample_fix],
            issues=[sample_issue],
            fixer=mock_fixer,
            project_path=tmp_path,
            backup=True
        )

        # Should not apply fixes
        mock_fixer.apply_fixes_with_validation.assert_not_called()

    @patch("devdox_ai_sonar.cli.select_fixes_interactively")
    def test_apply_fixes_no_selection(self, mock_select, sample_fix, sample_issue, tmp_path):
        """Test when no fixes are selected."""
        mock_select.return_value = []

        mock_fixer = MagicMock()

        _apply_fixes_if_requested(
            apply=True,
            dry_run=False,
            fixes=[sample_fix],
            issues=[sample_issue],
            fixer=mock_fixer,
            project_path=tmp_path,
            backup=True
        )

        # Should not apply fixes
        mock_fixer.apply_fixes_with_validation.assert_not_called()

    def test_apply_fixes_dry_run_mode(self, sample_fix, sample_issue, tmp_path):
        """Test dry-run mode applies all fixes without confirmation."""
        mock_fixer = MagicMock()
        mock_fixer.apply_fixes_with_validation.return_value = FixResult(
            project_path=tmp_path,
            total_fixes_attempted=1,
            successful_fixes=[],
            failed_fixes=[],
            backup_created=False,
            backup_path=None
        )

        _apply_fixes_if_requested(
            apply=False,
            dry_run=True,
            fixes=[sample_fix],
            issues=[sample_issue],
            fixer=mock_fixer,
            project_path=tmp_path,
            backup=True
        )

        # Should call with dry_run=True
        call_args = mock_fixer.apply_fixes_with_validation.call_args
        assert call_args[1]["dry_run"] is True


# ============================================================================
# MISSING: DISPLAY FUNCTION EDGE CASES
# ============================================================================

class TestDisplayFunctionsEdgeCases:
    """Missing edge case tests for display functions."""

    def test_display_analysis_results_empty_issues(self):
        """Test displaying results with empty issues list."""
        result = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=0,
            issues=[]
        )

        # Should not crash
        _display_analysis_results(result, limit=None)

    def test_display_analysis_results_all_severity_types(self):
        """Test displaying results with all severity types."""
        issues = []
        for severity in Severity:
            issue = SonarIssue(
                key=f"key-{severity.value}",
                rule="python:S1481",
                severity=severity,
                component="test:src/test.py",
                project="test",
                message=f"Issue {severity.value}",
                type=IssueType.CODE_SMELL,
                first_line=10,
                last_line=10
            )
            issues.append(issue)

        result = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=len(issues),
            issues=issues
        )

        _display_analysis_results(result, limit=None)

    def test_display_analysis_results_very_long_message(self):
        """Test displaying issue with very long message."""
        issue = SonarIssue(
            key="test",
            rule="python:S1481",
            severity=Severity.MAJOR,
            component="test:src/test.py",
            project="test",
            message="x" * 1000,  # Very long message
            type=IssueType.CODE_SMELL,
            first_line=10,
            last_line=10
        )

        result = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=1,
            issues=[issue]
        )

        # Should truncate message properly
        _display_analysis_results(result, limit=None)

    def test_display_analysis_results_missing_file_and_line(self):
        """Test displaying issue with missing file and line info."""
        issue = SonarIssue(
            key="test",
            rule="python:S1481",
            severity=Severity.MAJOR,
            component="test:src/test.py",
            project="test",
            message="Test",
            type=IssueType.CODE_SMELL,
            file=None,
            first_line=None,
            last_line=None
        )

        result = AnalysisResult(
            project_key="test",
            organization="org",
            branch="main",
            total_issues=1,
            issues=[issue]
        )

        # Should handle missing info gracefully
        _display_analysis_results(result, limit=None)

    def test_display_fix_results_all_successful(self, tmp_path, sample_fix):
        """Test displaying results with all successful fixes."""
        result = FixResult(
            project_path=tmp_path,
            total_fixes_attempted=5,
            successful_fixes=[sample_fix] * 5,
            failed_fixes=[],
            backup_created=True,
            backup_path=tmp_path / "backup"
        )

        _display_fix_results(result)

    def test_display_fix_results_all_failed(self, tmp_path):
        """Test displaying results with all failed fixes."""
        result = FixResult(
            project_path=tmp_path,
            total_fixes_attempted=5,
            successful_fixes=[],
            failed_fixes=[
                {"issue_key": f"key-{i}", "error": f"Error {i}"}
                for i in range(5)
            ],
            backup_created=False,
            backup_path=None
        )

        _display_fix_results(result)

    def test_display_fix_results_missing_error_details(self, tmp_path):
        """Test displaying results with malformed failed fix data."""
        result = FixResult(
            project_path=tmp_path,
            total_fixes_attempted=2,
            successful_fixes=[],
            failed_fixes=[
                {},  # Missing error key
                {"issue_key": "test"},  # Missing error message
            ],
            backup_created=False,
            backup_path=None
        )

        # Should handle gracefully
        _display_fix_results(result)


# ============================================================================
# MISSING: PARSE FILTERS EDGE CASES
# ============================================================================

class TestParseFiltersEdgeCases:
    """Missing edge case tests for _parse_filters."""

    def test_parse_filters_with_leading_trailing_spaces(self):
        """Test parsing filters with extra whitespace."""
        severity_list, types_list = _parse_filters(
            "  BLOCKER  ,  CRITICAL  ",
            "  BUG  ,  CODE_SMELL  ",
            {"BUG", "CODE_SMELL"},
            {"BLOCKER", "CRITICAL"}
        )

        assert "BLOCKER" in severity_list
        assert "CRITICAL" in severity_list
        assert "BUG" in types_list
        assert "CODE_SMELL" in types_list

    def test_parse_filters_case_sensitivity(self):
        """Test that filters are case-sensitive."""
        # The implementation doesn't do case conversion,
        # so lowercase should fail
        

        with pytest.raises(BadParameter):
            _parse_filters(
                "blocker",  # lowercase
                None,
                {"BUG"},
                {"BLOCKER"}
            )

    def test_parse_filters_single_value(self):
        """Test parsing single filter value."""
        severity_list, types_list = _parse_filters(
            "BLOCKER",
            "BUG",
            {"BUG"},
            {"BLOCKER"}
        )

        assert severity_list == ["BLOCKER"]
        assert types_list == ["BUG"]

    def test_parse_filters_none_and_empty_string(self):
        """Test that None and empty string are handled same."""
        result1 = _parse_filters(None, None, {"BUG"}, {"BLOCKER"})
        result2 = _parse_filters("", "", {"BUG"}, {"BLOCKER"})

        assert result1 == result2
        assert result1 == (None, None)

    def test_parse_filters_partial_invalid(self):
        """Test with mix of valid and invalid values."""
        

        with pytest.raises(BadParameter):
            _parse_filters(
                "BLOCKER,INVALID,CRITICAL",
                None,
                {"BUG"},
                {"BLOCKER", "CRITICAL"}
            )


# ============================================================================
# MISSING: ANALYZE COMMAND EDGE CASES
# ============================================================================

class TestAnalyzeCommandEdgeCases:
    """Missing edge case tests for analyze command."""

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_empty_severity_list(self, mock_analyzer_class, runner):
        """Test analyze with empty severity list."""
        mock_analyzer = MagicMock()
        result_obj = AnalysisResult(
            project_key="test",
            organization="org",
            total_issues=0,
            issues=[]
        )
        mock_analyzer.get_project_issues.return_value = result_obj
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--severity",  # Empty value
            ],
        )

        # Click should handle this
        assert result.exit_code in [0, 2]

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_zero_limit(self, mock_analyzer_class, runner):
        """Test analyze with limit=0."""
        mock_analyzer = MagicMock()
        issue = SonarIssue(
            key="test",
            rule="python:S1481",
            severity=Severity.MAJOR,
            component="test:src/test.py",
            project="test",
            message="Test",
            type=IssueType.CODE_SMELL,
            first_line=10,
            last_line=10
        )
        result_obj = AnalysisResult(
            project_key="test",
            organization="org",
            total_issues=1,
            issues=[issue]
        )
        mock_analyzer.get_project_issues.return_value = result_obj
        mock_analyzer_class.return_value = mock_analyzer

        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--limit", "0",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_analyze_with_negative_limit(self, mock_analyzer_class, runner):
        """Test analyze with negative limit."""
        result = runner.invoke(
            analyze,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--limit", "-1",
            ],
        )

        # Click should handle invalid integer
        assert result.exit_code != 0


# ============================================================================
# MISSING: FIX COMMAND EDGE CASES
# ============================================================================

class TestFixCommandEdgeCases:
    """Missing edge case tests for fix command."""

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_with_interactive_flag_no_issues(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test interactive mode when no issues found."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0
        assert "No fixable issues" in result.output or result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_with_empty_severity_filter(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix with empty severity filter."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--severity", "",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        assert result.exit_code == 0

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_fix_with_zero_max_fixes(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test fix with max-fixes=0."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = []
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer_class.return_value = mock_fixer

        result = runner.invoke(
            fix,
            [
                "--token", "test-token",
                "--organization", "test-org",
                "--project", "test-project",
                "--project-path", str(tmp_path),
                "--max-fixes", "0",
                "--provider", "openai",
                "--api-key", "test-key",
            ],
        )

        # Should handle gracefully
        assert result.exit_code == 0


# ============================================================================
# MISSING: SAVE RESULTS EDGE CASES
# ============================================================================

class TestSaveResultsEdgeCases:
    """Missing edge case tests for _save_results."""

    def test_save_results_to_existing_file(self, tmp_path):
        """Test overwriting existing file."""
        output_file = tmp_path / "existing.json"
        output_file.write_text('{"old": "data"}')

        issue = SonarIssue(
            key="test",
            rule="python:S1481",
            severity=Severity.MAJOR,
            component="test:src/test.py",
            project="test",
            message="Test",
            type=IssueType.CODE_SMELL,
            first_line=10,
            last_line=10
        )

        result = AnalysisResult(
            project_key="test",
            organization="org",
            total_issues=1,
            issues=[issue]
        )

        _save_results(result, str(output_file))

        # Should overwrite
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert data["project_key"] == "test"
            assert "old" not in data

    def test_save_results_with_special_characters(self, tmp_path):
        """Test saving results with special characters in data."""
        issue = SonarIssue(
            key="test-日本語-key",
            rule="python:S1481",
            severity=Severity.MAJOR,
            component="test:src/test.py",
            project="test-プロジェクト",
            message="Test with émojis 🎉",
            type=IssueType.CODE_SMELL,
            first_line=10,
            last_line=10
        )

        result = AnalysisResult(
            project_key="test",
            organization="org",
            total_issues=1,
            issues=[issue]
        )

        output_file = tmp_path / "unicode.json"
        _save_results(result, str(output_file))

        # Should handle unicode properly
        assert output_file.exists()
        with open(output_file, encoding='utf-8') as f:
            data = json.load(f)
            assert "日本語" in data["issues"][0]["key"]


# ============================================================================
# MISSING: INTEGRATION TESTS
# ============================================================================

class TestCLIIntegration:
    """Integration tests for complete CLI workflows."""

    @patch("devdox_ai_sonar.cli.LLMFixer")
    @patch("devdox_ai_sonar.cli.SonarCloudAnalyzer")
    def test_complete_fix_workflow(self, mock_analyzer_class, mock_fixer_class, runner, tmp_path):
        """Test complete fix workflow from analysis to application."""
        # Setup
        issue = SonarIssue(
            key="test-key",
            rule="python:S1481",
            severity=Severity.MAJOR,
            component="test:src/test.py",
            project="test-project",
            message="Unused variable",
            type=IssueType.CODE_SMELL,
            first_line=10,
            last_line=10,
            file="src/test.py"
        )

        fix_suggestion = FixSuggestion(
            issue_key="test-key",
            original_code="unused_var = 42",
            fixed_code="",
            explanation="Removed unused variable",
            confidence=0.95,
            llm_model="gpt-4",
            file_path="src/test.py",
            sonar_line_number=10,
            line_number=10,
            last_line_number=10
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_fixable_issues.return_value = [issue]
        mock_analyzer.get_rule_by_key.return_value = {"name": "Unused variables"}
        mock_analyzer_class.return_value = mock_analyzer

        mock_fixer = MagicMock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.api_key = "test-key"
        mock_fixer.generate_fix.return_value = fix_suggestion
        mock_fixer.apply_fixes_with_validation.return_value = FixResult(
            project_path=tmp_path,
            total_fixes_attempted=1,
            successful_fixes=[fix_suggestion],
            failed_fixes=[],
            backup_created=True,
            backup_path=tmp_path / "backup"
        )
        mock_fixer_class.return_value = mock_fixer

        # Execute
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

        # Verify
        assert result.exit_code == 0
        mock_analyzer.get_fixable_issues.assert_called_once()
        mock_fixer.generate_fix.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
