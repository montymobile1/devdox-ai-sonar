import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from devdox_ai_sonar.models.sonar import (
        SonarType,
        Severity,
        IssueType,
        Impact,
        SonarCloudConfig,
        SonarIssue,
        SonarSecurityIssue,
        ProjectMetrics,
        SecurityAnalysisResult,
        AnalysisResult,
        ProcessedRules,
        FixSuggestion,
        FixResult,
    )


# ============================================================================
# Test Enums
# ============================================================================

class TestEnums:
    """Test all enum types."""

    def test_sonar_type_values(self):
        """Test SonarType enum values."""
        assert SonarType.BRANCH.value == "branch"
        assert SonarType.PR.value == "pr"
        assert len(SonarType) == 2

    def test_severity_values(self):
        """Test Severity enum values."""
        assert Severity.BLOCKER.value == "BLOCKER"
        assert Severity.CRITICAL.value == "CRITICAL"
        assert Severity.MAJOR.value == "MAJOR"
        assert Severity.MINOR.value == "MINOR"
        assert Severity.INFO.value == "INFO"
        assert len(Severity) == 5

    def test_issue_type_values(self):
        """Test IssueType enum values."""
        assert IssueType.BUG.value == "BUG"
        assert IssueType.VULNERABILITY.value == "VULNERABILITY"
        assert IssueType.CODE_SMELL.value == "CODE_SMELL"
        assert IssueType.SECURITY_HOTSPOT.value == "SECURITY_HOTSPOT"
        assert len(IssueType) == 4

    def test_impact_values(self):
        """Test Impact enum values."""
        assert Impact.HIGH.value == "HIGH"
        assert Impact.MEDIUM.value == "MEDIUM"
        assert Impact.LOW.value == "LOW"
        assert len(Impact) == 3


# ============================================================================
# Test SonarCloudConfig
# ============================================================================

class TestSonarCloudConfig:
    """Test SonarCloudConfig model."""

    def test_valid_config_creation(self):
        """Test creating a valid SonarCloudConfig."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        assert config.token == "test-token"
        assert config.organization == "test-org"
        assert config.project == "test-project"
        assert config.project_path == "/path/to/project"
        assert config.git_url == "https://github.com/user/repo.git"

    def test_validate_success(self):
        """Test successful validation."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is True
        assert error is None

    def test_validate_empty_token(self):
        """Test validation with empty token."""
        config = SonarCloudConfig(
            token="",
            organization="test-org",
            project="test-project",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Token cannot be empty"

    def test_validate_whitespace_token(self):
        """Test validation with whitespace token."""
        config = SonarCloudConfig(
            token="   ",
            organization="test-org",
            project="test-project",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Token cannot be empty"

    def test_validate_empty_organization(self):
        """Test validation with empty organization."""
        config = SonarCloudConfig(
            token="test-token",
            organization="",
            project="test-project",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Organization key cannot be empty"

    def test_validate_whitespace_organization(self):
        """Test validation with whitespace organization."""
        config = SonarCloudConfig(
            token="test-token",
            organization="   ",
            project="test-project",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Organization key cannot be empty"

    def test_validate_empty_project(self):
        """Test validation with empty project."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Project key cannot be empty"

    def test_validate_whitespace_project(self):
        """Test validation with whitespace project."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="   ",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Project key cannot be empty"

    def test_validate_empty_project_path(self):
        """Test validation with empty project path."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="",
            git_url="https://github.com/user/repo.git"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Project path cannot be empty"

    def test_validate_empty_git_url(self):
        """Test validation with empty git URL."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/path/to/project",
            git_url=""
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Git URL cannot be empty"

    def test_validate_whitespace_git_url(self):
        """Test validation with whitespace git URL."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/path/to/project",
            git_url="   "
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Git URL cannot be empty"

    def test_validate_git_url_without_dot_git(self):
        """Test validation with git URL not ending in .git."""
        config = SonarCloudConfig(
            token="test-token",
            organization="test-org",
            project="test-project",
            project_path="/path/to/project",
            git_url="https://github.com/user/repo"
        )
        is_valid, error = config.validate()
        assert is_valid is False
        assert error == "Git URL must end with .git"


# ============================================================================
# Test SonarIssue
# ============================================================================

class TestSonarIssue:
    """Test SonarIssue model."""

    def test_sonar_issue_creation(self):
        """Test creating a SonarIssue."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            first_line=10,
            last_line=15,
            message="Fix this issue",
            type=IssueType.BUG,
            impact=Impact.HIGH,
            file="src/main.py",
            branch="main"
        )
        assert issue.key == "issue-123"
        assert issue.rule == "python:S1234"
        assert issue.severity == Severity.MAJOR
        assert issue.type == IssueType.BUG
        assert issue.first_line == 10
        assert issue.last_line == 15

    def test_sonar_issue_with_defaults(self):
        """Test SonarIssue with default values."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.BUG
        )
        assert issue.status == "OPEN"
        assert issue.tags == []
        assert issue.first_line is None
        assert issue.last_line is None

    def test_file_path_property(self):
        """Test file_path property returns Path object."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.BUG,
            file="src/main.py"
        )
        assert isinstance(issue.file_path, Path)
        assert str(issue.file_path) == "src/main.py"

    def test_file_path_property_none(self):
        """Test file_path property when file is None."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.BUG
        )
        assert issue.file_path is None

    def test_is_fixable_true_for_bug(self):
        """Test is_fixable returns True for bug with line numbers."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.BUG,
            first_line=10,
            last_line=15
        )
        assert issue.is_fixable is True

    def test_is_fixable_true_for_code_smell(self):
        """Test is_fixable returns True for code smell with line numbers."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.CODE_SMELL,
            first_line=10,
            last_line=15
        )
        assert issue.is_fixable is True

    def test_is_fixable_false_for_vulnerability(self):
        """Test is_fixable returns False for vulnerability."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.VULNERABILITY,
            first_line=10,
            last_line=15
        )
        assert issue.is_fixable is False

    def test_is_fixable_false_without_line_numbers(self):
        """Test is_fixable returns False without line numbers."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.BUG
        )
        assert issue.is_fixable is False

    def test_sonar_issue_with_optional_fields(self):
        """Test SonarIssue with all optional fields."""
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message="Fix this issue",
            type=IssueType.BUG,
            first_line=10,
            last_line=15,
            file="src/main.py",
            branch="develop",
            status="RESOLVED",
            creation_date="2024-01-01T00:00:00Z",
            update_date="2024-01-02T00:00:00Z",
            tags=["security", "performance"],
            effort="1h",
            debt="30min",
            impact=Impact.HIGH
        )
        assert issue.status == "RESOLVED"
        assert issue.creation_date == "2024-01-01T00:00:00Z"
        assert issue.update_date == "2024-01-02T00:00:00Z"
        assert "security" in issue.tags
        assert issue.effort == "1h"
        assert issue.debt == "30min"


# ============================================================================
# Test SonarSecurityIssue
# ============================================================================

class TestSonarSecurityIssue:
    """Test SonarSecurityIssue model."""

    def test_security_issue_creation(self):
        """Test creating a SonarSecurityIssue."""
        issue = SonarSecurityIssue(
            key="sec-123",
            component="project:src/main.py",
            rule="python:S5146",
            project="test-project",
            security_category="sql-injection",
            vulnerability_probability="HIGH",
            first_line=20,
            last_line=25,
            message="SQL injection vulnerability",
            file="src/main.py"
        )
        assert issue.key == "sec-123"
        assert issue.security_category == "sql-injection"
        assert issue.vulnerability_probability == "HIGH"
        assert issue.first_line == 20
        assert issue.last_line == 25

    def test_security_issue_with_defaults(self):
        """Test SonarSecurityIssue with default values."""
        issue = SonarSecurityIssue(
            key="sec-123",
            component="project:src/main.py",
            rule="python:S5146",
            project="test-project",
            security_category="sql-injection",
            vulnerability_probability="HIGH",
            message="SQL injection vulnerability"
        )
        assert issue.status == "OPEN"
        assert issue.first_line is None
        assert issue.last_line is None
        assert issue.file is None

    def test_security_issue_file_path_property(self):
        """Test file_path property for security issue."""
        issue = SonarSecurityIssue(
            key="sec-123",
            component="project:src/main.py",
            rule="python:S5146",
            project="test-project",
            security_category="sql-injection",
            vulnerability_probability="HIGH",
            message="SQL injection vulnerability",
            file="src/database.py"
        )
        assert isinstance(issue.file_path, Path)
        assert str(issue.file_path) == "src/database.py"

    def test_security_issue_file_path_none(self):
        """Test file_path property when file is None."""
        issue = SonarSecurityIssue(
            key="sec-123",
            component="project:src/main.py",
            rule="python:S5146",
            project="test-project",
            security_category="sql-injection",
            vulnerability_probability="HIGH",
            message="SQL injection vulnerability"
        )
        assert issue.file_path is None

    def test_security_issue_with_dates(self):
        """Test security issue with creation and update dates."""
        issue = SonarSecurityIssue(
            key="sec-123",
            component="project:src/main.py",
            rule="python:S5146",
            project="test-project",
            security_category="sql-injection",
            vulnerability_probability="HIGH",
            message="SQL injection vulnerability",
            creation_date="2024-01-01T00:00:00Z",
            update_date="2024-01-02T00:00:00Z"
        )
        assert issue.creation_date == "2024-01-01T00:00:00Z"
        assert issue.update_date == "2024-01-02T00:00:00Z"


# ============================================================================
# Test ProjectMetrics
# ============================================================================

class TestProjectMetrics:
    """Test ProjectMetrics model."""

    def test_project_metrics_creation(self):
        """Test creating ProjectMetrics."""
        metrics = ProjectMetrics(
            project_key="test-project",
            lines_of_code=10000,
            coverage=85.5,
            duplicated_lines_density=2.3,
            maintainability_rating="A",
            reliability_rating="A",
            security_rating="A",
            bugs=5,
            vulnerabilities=2,
            code_smells=15,
            technical_debt="2d 3h"
        )
        assert metrics.project_key == "test-project"
        assert metrics.lines_of_code == 10000
        assert metrics.coverage == 85.5
        assert metrics.bugs == 5

    def test_project_metrics_with_none_values(self):
        """Test ProjectMetrics with None values."""
        metrics = ProjectMetrics(project_key="test-project")
        assert metrics.project_key == "test-project"
        assert metrics.lines_of_code is None
        assert metrics.coverage is None
        assert metrics.bugs is None

    def test_project_metrics_partial(self):
        """Test ProjectMetrics with partial data."""
        metrics = ProjectMetrics(
            project_key="test-project",
            lines_of_code=5000,
            bugs=10
        )
        assert metrics.project_key == "test-project"
        assert metrics.lines_of_code == 5000
        assert metrics.bugs == 10
        assert metrics.coverage is None
        assert metrics.vulnerabilities is None


# ============================================================================
# Test SecurityAnalysisResult
# ============================================================================

class TestSecurityAnalysisResult:
    """Test SecurityAnalysisResult model."""

    def test_security_analysis_result_creation(self):
        """Test creating SecurityAnalysisResult."""
        issues = [
            SonarSecurityIssue(
                key="sec-1",
                component="project:src/main.py",
                rule="python:S5146",
                project="test-project",
                security_category="sql-injection",
                vulnerability_probability="HIGH",
                message="SQL injection",
                file="src/main.py",
                first_line=10,
                last_line=15
            ),
            SonarSecurityIssue(
                key="sec-2",
                component="project:src/auth.py",
                rule="python:S5144",
                project="test-project",
                security_category="auth",
                vulnerability_probability="MEDIUM",
                message="Weak authentication",
                file="src/auth.py",
                first_line=20,
                last_line=25
            )
        ]

        result = SecurityAnalysisResult(
            project_key="test-project",
            organization="test-org",
            branch="main",
            total_issues=2,
            issues=issues
        )

        assert result.project_key == "test-project"
        assert result.organization == "test-org"
        assert result.total_issues == 2
        assert len(result.issues) == 2

    def test_security_analysis_result_fixable_issues_grouping(self):
        """Test that fixable_issues_by_file is populated correctly."""
        issues = [
            SonarSecurityIssue(
                key="sec-1",
                component="project:src/main.py",
                rule="python:S5146",
                project="test-project",
                security_category="sql-injection",
                vulnerability_probability="HIGH",
                message="SQL injection",
                file="src/main.py",
                first_line=10,
                last_line=15
            ),
            SonarSecurityIssue(
                key="sec-2",
                component="project:src/main.py",
                rule="python:S5147",
                project="test-project",
                security_category="xss",
                vulnerability_probability="MEDIUM",
                message="XSS vulnerability",
                file="src/main.py",
                first_line=30,
                last_line=35
            )
        ]

        result = SecurityAnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=2,
            issues=issues
        )

        assert len(result.fixable_issues_by_file) == 1
        assert "src/main.py" in result.fixable_issues_by_file
        assert len(result.fixable_issues_by_file["src/main.py"]) == 2

    def test_security_analysis_result_with_timestamp(self):
        """Test SecurityAnalysisResult with analysis timestamp."""
        result = SecurityAnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=0,
            issues=[],
            analysis_timestamp="2024-01-08T12:00:00Z"
        )
        assert result.analysis_timestamp == "2024-01-08T12:00:00Z"

    def test_security_analysis_result_default_branch(self):
        """Test SecurityAnalysisResult default branch value."""
        result = SecurityAnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=0,
            issues=[]
        )
        assert result.branch == "main"


# ============================================================================
# Test AnalysisResult
# ============================================================================

class TestAnalysisResult:
    """Test AnalysisResult model."""

    def test_analysis_result_creation(self):
        """Test creating AnalysisResult."""
        issues = [
            SonarIssue(
                key="issue-1",
                rule="python:S1234",
                severity=Severity.MAJOR,
                component="project:src/main.py",
                project="test-project",
                message="Fix this",
                type=IssueType.BUG,
                first_line=10,
                last_line=15,
                file="src/main.py"
            )
        ]

        metrics = ProjectMetrics(
            project_key="test-project",
            lines_of_code=5000,
            bugs=10
        )

        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            branch="develop",
            total_issues=1,
            issues=issues,
            metrics=metrics
        )

        assert result.project_key == "test-project"
        assert result.branch == "develop"
        assert result.total_issues == 1
        assert result.metrics is not None

    def test_analysis_result_fixable_issues_auto_population(self):
        """Test that fixable_issues is automatically populated."""
        issues = [
            SonarIssue(
                key="issue-1",
                rule="python:S1234",
                severity=Severity.MAJOR,
                component="project:src/main.py",
                project="test-project",
                message="Fix this bug",
                type=IssueType.BUG,
                first_line=10,
                last_line=15,
                file="src/main.py"
            ),
            SonarIssue(
                key="issue-2",
                rule="python:S5678",
                severity=Severity.CRITICAL,
                component="project:src/main.py",
                project="test-project",
                message="Vulnerability",
                type=IssueType.VULNERABILITY,
                first_line=20,
                last_line=25,
                file="src/main.py"
            )
        ]

        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=2,
            issues=issues
        )

        # Only the bug should be fixable, not the vulnerability
        assert len(result.fixable_issues) == 1
        assert result.fixable_issues[0].type == IssueType.BUG

    def test_analysis_result_fixable_issues_by_file(self):
        """Test fixable_issues_by_file grouping."""
        issues = [
            SonarIssue(
                key="issue-1",
                rule="python:S1234",
                severity=Severity.MAJOR,
                component="project:src/main.py",
                project="test-project",
                message="Fix this",
                type=IssueType.BUG,
                first_line=10,
                last_line=15,
                file="src/main.py"
            ),
            SonarIssue(
                key="issue-2",
                rule="python:S5678",
                severity=Severity.MINOR,
                component="project:src/utils.py",
                project="test-project",
                message="Code smell",
                type=IssueType.CODE_SMELL,
                first_line=5,
                last_line=10,
                file="src/utils.py"
            )
        ]

        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=2,
            issues=issues
        )

        assert len(result.fixable_issues_by_file) == 2
        assert "src/main.py" in result.fixable_issues_by_file
        assert "src/utils.py" in result.fixable_issues_by_file
        assert len(result.fixable_issues_by_file["src/main.py"]) == 1
        assert len(result.fixable_issues_by_file["src/utils.py"]) == 1

    def test_issues_by_severity_property(self):
        """Test issues_by_severity property."""
        issues = [
            SonarIssue(
                key="issue-1",
                rule="python:S1234",
                severity=Severity.CRITICAL,
                component="project:src/main.py",
                project="test-project",
                message="Critical bug",
                type=IssueType.BUG,
                file="src/main.py"
            ),
            SonarIssue(
                key="issue-2",
                rule="python:S5678",
                severity=Severity.CRITICAL,
                component="project:src/utils.py",
                project="test-project",
                message="Another critical",
                type=IssueType.BUG,
                file="src/utils.py"
            ),
            SonarIssue(
                key="issue-3",
                rule="python:S9999",
                severity=Severity.MINOR,
                component="project:src/helper.py",
                project="test-project",
                message="Minor issue",
                type=IssueType.CODE_SMELL,
                file="src/helper.py"
            )
        ]

        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=3,
            issues=issues
        )

        by_severity = result.issues_by_severity
        assert len(by_severity[Severity.CRITICAL]) == 2
        assert len(by_severity[Severity.MINOR]) == 1
        assert len(by_severity[Severity.BLOCKER]) == 0

    def test_issues_by_type_property(self):
        """Test issues_by_type property."""
        issues = [
            SonarIssue(
                key="issue-1",
                rule="python:S1234",
                severity=Severity.MAJOR,
                component="project:src/main.py",
                project="test-project",
                message="Bug 1",
                type=IssueType.BUG,
                file="src/main.py"
            ),
            SonarIssue(
                key="issue-2",
                rule="python:S5678",
                severity=Severity.MAJOR,
                component="project:src/utils.py",
                project="test-project",
                message="Bug 2",
                type=IssueType.BUG,
                file="src/utils.py"
            ),
            SonarIssue(
                key="issue-3",
                rule="python:S9999",
                severity=Severity.CRITICAL,
                component="project:src/auth.py",
                project="test-project",
                message="Vulnerability",
                type=IssueType.VULNERABILITY,
                file="src/auth.py"
            )
        ]

        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=3,
            issues=issues
        )

        by_type = result.issues_by_type
        assert len(by_type[IssueType.BUG]) == 2
        assert len(by_type[IssueType.VULNERABILITY]) == 1
        assert len(by_type[IssueType.CODE_SMELL]) == 0

    def test_analysis_result_with_no_metrics(self):
        """Test AnalysisResult without metrics."""
        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=0,
            issues=[]
        )
        assert result.metrics is None

    def test_analysis_result_default_branch(self):
        """Test AnalysisResult default branch."""
        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=0,
            issues=[]
        )
        assert result.branch == "main"


# ============================================================================
# Test FixSuggestion
# ============================================================================

class TestFixSuggestion:
    """Test FixSuggestion model."""

    def test_fix_suggestion_creation(self):
        """Test creating a FixSuggestion."""
        suggestion = FixSuggestion(
            issue_key="issue-123",
            original_code="x = 1\ny = 2",
            fixed_code="x = 1\ny = 2  # noqa",
            explanation="Added noqa comment",
            confidence=0.95,
            llm_model="claude-sonnet-4"
        )
        assert suggestion.issue_key == "issue-123"
        assert suggestion.confidence == 0.95
        assert suggestion.llm_model == "claude-sonnet-4"

    def test_fix_suggestion_with_all_fields(self):
        """Test FixSuggestion with all optional fields."""
        suggestion = FixSuggestion(
            issue_key="issue-123",
            original_code="def bad_function():\n    pass",
            fixed_code="def good_function():\n    \"\"\"Docstring\"\"\"\n    pass",
            helper_code="# Helper comment",
            placement_helper="# Place at line 10",
            explanation="Added docstring and improved naming",
            confidence=0.85,
            llm_model="claude-sonnet-4",
            rule_description="Functions should have docstrings",
            file_path="src/main.py",
            sonar_line_number=10,
            line_number=10,
            last_line_number=12
        )
        assert suggestion.helper_code == "# Helper comment"
        assert suggestion.placement_helper == "# Place at line 10"
        assert suggestion.rule_description == "Functions should have docstrings"
        assert suggestion.file_path == "src/main.py"
        assert suggestion.line_number == 10
        assert suggestion.last_line_number == 12

    def test_is_high_confidence_true(self):
        """Test is_high_confidence returns True for confidence >= 0.8."""
        suggestion = FixSuggestion(
            issue_key="issue-123",
            original_code="x = 1",
            fixed_code="x = 1  # noqa",
            explanation="Added noqa",
            confidence=0.85,
            llm_model="claude-sonnet-4"
        )
        assert suggestion.is_high_confidence is True

    def test_is_high_confidence_false(self):
        """Test is_high_confidence returns False for confidence < 0.8."""
        suggestion = FixSuggestion(
            issue_key="issue-123",
            original_code="x = 1",
            fixed_code="x = 1  # noqa",
            explanation="Added noqa",
            confidence=0.75,
            llm_model="claude-sonnet-4"
        )
        assert suggestion.is_high_confidence is False

    def test_is_high_confidence_boundary(self):
        """Test is_high_confidence at boundary (0.8)."""
        suggestion = FixSuggestion(
            issue_key="issue-123",
            original_code="x = 1",
            fixed_code="x = 1  # noqa",
            explanation="Added noqa",
            confidence=0.8,
            llm_model="claude-sonnet-4"
        )
        assert suggestion.is_high_confidence is True

    def test_fix_suggestion_confidence_validation(self):
        """Test confidence validation (0.0 to 1.0)."""
        # Valid confidence values
        for conf in [0.0, 0.5, 1.0]:
            suggestion = FixSuggestion(
                issue_key="issue-123",
                original_code="x = 1",
                fixed_code="x = 2",
                explanation="Changed value",
                confidence=conf,
                llm_model="claude-sonnet-4"
            )
            assert suggestion.confidence == conf

    def test_fix_suggestion_with_empty_helper_fields(self):
        """Test FixSuggestion with empty helper fields."""
        suggestion = FixSuggestion(
            issue_key="issue-123",
            original_code="x = 1",
            fixed_code="x = 2",
            explanation="Changed value",
            confidence=0.9,
            llm_model="claude-sonnet-4"
        )
        assert suggestion.helper_code == ""
        assert suggestion.placement_helper == ""


# ============================================================================
# Test FixResult
# ============================================================================

class TestFixResult:
    """Test FixResult model."""

    def test_fix_result_creation(self):
        """Test creating a FixResult."""
        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=5
        )
        assert result.project_path == Path("/path/to/project")
        assert result.total_fixes_attempted == 5
        assert len(result.successful_fixes) == 0
        assert len(result.failed_fixes) == 0

    def test_fix_result_with_successful_fixes(self):
        """Test FixResult with successful fixes."""
        suggestion = FixSuggestion(
            issue_key="issue-123",
            original_code="x = 1",
            fixed_code="x = 2",
            explanation="Changed value",
            confidence=0.9,
            llm_model="claude-sonnet-4"
        )

        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=2,
            successful_fixes=[suggestion]
        )
        assert len(result.successful_fixes) == 1
        assert result.successful_fixes[0].issue_key == "issue-123"

    def test_fix_result_with_failed_fixes(self):
        """Test FixResult with failed fixes."""
        failed_fix = {
            "issue_key": "issue-456",
            "error": "Failed to apply patch",
            "file": "src/main.py"
        }

        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=2,
            failed_fixes=[failed_fix]
        )
        assert len(result.failed_fixes) == 1
        assert result.failed_fixes[0]["issue_key"] == "issue-456"

    def test_fix_result_with_skipped_issues(self):
        """Test FixResult with skipped issues."""
        issue = SonarIssue(
            key="issue-789",
            rule="python:S1234",
            severity=Severity.MINOR,
            component="project:src/main.py",
            project="test-project",
            message="Minor issue",
            type=IssueType.CODE_SMELL
        )

        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=3,
            skipped_issues=[issue]
        )
        assert len(result.skipped_issues) == 1
        assert result.skipped_issues[0].key == "issue-789"

    def test_fix_result_with_backup(self):
        """Test FixResult with backup information."""
        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=5,
            backup_created=True,
            backup_path=Path("/path/to/backup")
        )
        assert result.backup_created is True
        assert result.backup_path == Path("/path/to/backup")

    def test_success_rate_all_successful(self):
        """Test success_rate with all fixes successful."""
        suggestions = [
            FixSuggestion(
                issue_key=f"issue-{i}",
                original_code="x = 1",
                fixed_code="x = 2",
                explanation="Changed value",
                confidence=0.9,
                llm_model="claude-sonnet-4"
            )
            for i in range(5)
        ]

        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=5,
            successful_fixes=suggestions
        )
        assert result.success_rate == 1.0

    def test_success_rate_partial_success(self):
        """Test success_rate with partial success."""
        suggestions = [
            FixSuggestion(
                issue_key=f"issue-{i}",
                original_code="x = 1",
                fixed_code="x = 2",
                explanation="Changed value",
                confidence=0.9,
                llm_model="claude-sonnet-4"
            )
            for i in range(3)
        ]

        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=10,
            successful_fixes=suggestions
        )
        assert result.success_rate == 0.3

    def test_success_rate_no_attempts(self):
        """Test success_rate with no attempts."""
        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=0
        )
        assert result.success_rate == 0.0

    def test_success_rate_all_failed(self):
        """Test success_rate with all fixes failed."""
        failed_fixes = [
            {"issue_key": f"issue-{i}", "error": "Failed"}
            for i in range(5)
        ]

        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=5,
            failed_fixes=failed_fixes
        )
        assert result.success_rate == 0.0

    def test_fix_result_complete_scenario(self):
        """Test FixResult with a complete scenario."""
        successful = [
            FixSuggestion(
                issue_key="issue-1",
                original_code="x = 1",
                fixed_code="x = 2",
                explanation="Fixed",
                confidence=0.9,
                llm_model="claude-sonnet-4"
            )
        ]

        failed = [
            {"issue_key": "issue-2", "error": "Syntax error"}
        ]

        skipped = [
            SonarIssue(
                key="issue-3",
                rule="python:S1234",
                severity=Severity.INFO,
                component="project:src/main.py",
                project="test-project",
                message="Info issue",
                type=IssueType.CODE_SMELL
            )
        ]

        result = FixResult(
            project_path=Path("/path/to/project"),
            total_fixes_attempted=3,
            successful_fixes=successful,
            failed_fixes=failed,
            skipped_issues=skipped,
            backup_created=True,
            backup_path=Path("/path/to/backup")
        )

        assert len(result.successful_fixes) == 1
        assert len(result.failed_fixes) == 1
        assert len(result.skipped_issues) == 1
        assert result.success_rate == pytest.approx(1 / 3, rel=1e-2)
        assert result.backup_created is True


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_sonar_issue_with_very_long_message(self):
        """Test SonarIssue with very long message."""
        long_message = "A" * 10000
        issue = SonarIssue(
            key="issue-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="project:src/main.py",
            project="test-project",
            message=long_message,
            type=IssueType.BUG
        )
        assert len(issue.message) == 10000

    def test_analysis_result_with_many_issues(self):
        """Test AnalysisResult with large number of issues."""
        issues = [
            SonarIssue(
                key=f"issue-{i}",
                rule="python:S1234",
                severity=Severity.MAJOR,
                component="project:src/main.py",
                project="test-project",
                message=f"Issue {i}",
                type=IssueType.BUG,
                file=f"src/file{i % 10}.py",
                first_line=i,
                last_line=i + 5
            )
            for i in range(1000)
        ]

        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=1000,
            issues=issues
        )

        assert len(result.issues) == 1000
        assert len(result.fixable_issues) == 1000
        # Should be grouped into 10 files (i % 10)
        assert len(result.fixable_issues_by_file) == 10

    def test_fix_result_with_path_object(self):
        """Test FixResult accepts Path objects correctly."""
        result = FixResult(
            project_path=Path("/absolute/path/to/project"),
            total_fixes_attempted=1,
            backup_path=Path("relative/backup")
        )
        assert isinstance(result.project_path, Path)
        assert isinstance(result.backup_path, Path)

    def test_sonar_cloud_config_with_special_characters(self):
        """Test SonarCloudConfig with special characters in fields."""
        config = SonarCloudConfig(
            token="token-with-special-chars!@#$%",
            organization="org-name_123",
            project="project.name-123",
            project_path="/path/with spaces/and-special_chars",
            git_url="https://github.com/user/repo-name_123.git"
        )
        is_valid, error = config.validate()
        assert is_valid is True

    def test_multiple_issues_same_file_different_lines(self):
        """Test handling multiple issues in same file at different lines."""
        issues = [
            SonarIssue(
                key=f"issue-{i}",
                rule="python:S1234",
                severity=Severity.MAJOR,
                component="project:src/main.py",
                project="test-project",
                message=f"Issue at line {i * 10}",
                type=IssueType.BUG,
                file="src/main.py",
                first_line=i * 10,
                last_line=i * 10 + 5
            )
            for i in range(1, 6)
        ]

        result = AnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=5,
            issues=issues
        )

        assert "src/main.py" in result.fixable_issues_by_file
        assert len(result.fixable_issues_by_file["src/main.py"]) == 5

    def test_security_analysis_with_no_fixable_issues(self):
        """Test SecurityAnalysisResult with issues but no file paths."""
        issues = [
            SonarSecurityIssue(
                key="sec-1",
                component="project:src/main.py",
                rule="python:S5146",
                project="test-project",
                security_category="sql-injection",
                vulnerability_probability="HIGH",
                message="SQL injection"
                # No file path provided
            )
        ]

        result = SecurityAnalysisResult(
            project_key="test-project",
            organization="test-org",
            total_issues=1,
            issues=issues
        )

        # Should not be in fixable_issues_by_file since no file path
        assert len(result.fixable_issues_by_file) == 0

