"""Comprehensive tests for SonarCloud analyzer."""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from devdox_ai_sonar.models import (
    SonarIssue,
    AnalysisResult,
    ProjectMetrics,
    Severity,
    IssueType,
    Impact,
)


@pytest.fixture
def mock_session():
    """Create a mock requests session."""
    session = Mock(spec=requests.Session)
    session.headers = {}
    return session


@pytest.fixture
def analyzer(mock_session):
    """Create a SonarCloudAnalyzer with mocked session."""
    with patch("devdox_ai_sonar.sonar_analyzer.requests.Session", return_value=mock_session):
        from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer

        analyzer = SonarCloudAnalyzer(
            token="test-token", organization="test-org",
        )
        analyzer.session = mock_session
        return analyzer


@pytest.fixture
def sample_issue_data():
    """Sample issue data from SonarCloud API."""
    return {
        "key": "AXqT8...example",
        "rule": "python:S1481",
        "severity": "MAJOR",
        "component": "project:src/main.py",
        "project": "my-project",
        "line": 42,
        "message": 'Remove the unused local variable "unused_var".',
        "type": "CODE_SMELL",
        "status": "OPEN",
        "creationDate": "2024-01-01T10:00:00+0000",
        "updateDate": "2024-01-02T10:00:00+0000",
        "tags": ["unused"],
        "impacts": [{"softwareQuality": "MAINTAINABILITY", "severity": "MEDIUM"}],
        "flows": [
            {
                "locations": [
                    {
                        "textRange": {
                            "startLine": 42,
                            "endLine": 45,
                            "startOffset": 0,
                            "endOffset": 20,
                        }
                    }
                ]
            }
        ],
    }


class TestSonarCloudAnalyzerInitialization:
    """Test SonarCloudAnalyzer initialization."""

    def test_initialization_with_defaults(self):
        """Test analyzer initialization with default values."""
        with patch("devdox_ai_sonar.sonar_analyzer.requests.Session"):
            from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer

            analyzer = SonarCloudAnalyzer(
                token="test-token", organization="test-org"
            )

            assert analyzer.token == "test-token"
            assert analyzer.organization == "test-org"
            assert analyzer.base_url == "https://sonarcloud.io"
            assert analyzer.timeout == 30

    def test_initialization_with_custom_params(self):
        """Test analyzer initialization with custom parameters."""
        with patch("devdox_ai_sonar.sonar_analyzer.requests.Session"):
            from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer

            analyzer = SonarCloudAnalyzer(
                token="test-token",
                organization="test-org",
                base_url="https://custom.sonarcloud.io",
                timeout=60,
                max_retries=5,
            )

            assert analyzer.base_url == "https://custom.sonarcloud.io"
            assert analyzer.timeout == 60

    def test_session_headers_configured(self, analyzer):
        """Test that session headers are properly configured."""
        assert "Authorization" in analyzer.session.headers
        assert analyzer.session.headers["Authorization"] == "Bearer test-token"
        assert analyzer.session.headers["Accept"] == "application/json"


class TestGetProjectIssues:
    """Test fetching project issues."""

    def test_get_project_issues_success(self, analyzer, sample_issue_data):
        """Test successful issue retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": [sample_issue_data],
            "paging": {"total": 1, "pageIndex": 1, "pageSize": 500},
        }

        analyzer.session.get.return_value = mock_response

        result = analyzer.get_project_issues(
            project_key="test-project", branch="main"
        )

        assert result is not None
        assert result.project_key == "test-project"
        assert result.total_issues == 1
        assert len(result.issues) == 1
        assert result.issues[0].rule == "python:S1481"


    def test_get_project_issues_with_filters(self, analyzer):
        """Test issue retrieval with filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": [],
            "paging": {"total": 0, "pageIndex": 1, "pageSize": 500},
        }

        analyzer.session.get.return_value = mock_response

        result = analyzer.get_project_issues(
            project_key="test-project",
            branch="develop",
            statuses=["OPEN", "CONFIRMED"],
            severities=["BLOCKER", "CRITICAL"],
            types=["BUG"],
        )

        # Verify filters were applied
        call_args = analyzer.session.get.call_args_list
        params = call_args[0][1]["params"]
        assert "OPEN" in params["issueStatuses"]
        assert "BLOCKER" in params.get("severities", "")

    def test_get_project_issues_http_error(self, analyzer):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        # Create HTTPError with response attached
        http_error = requests.HTTPError("404 Client Error")
        http_error.response = mock_response

        mock_response.raise_for_status.side_effect = http_error
        analyzer.session.get.return_value = mock_response

        result = analyzer.get_project_issues(
            project_key="nonexistent-project",
            branch="main",

        )

        assert result is None



    def test_get_project_issues_timeout(self, analyzer):
        """Test handling of request timeout."""
        analyzer.session.get.side_effect = requests.Timeout()


        result = analyzer.get_project_issues(
            project_key="test-project", branch="main"
        )

        assert result is None


class TestGetProjectMetrics:
    """Test fetching project metrics."""

    def test_get_project_metrics_success(self, analyzer):
        """Test successful metrics retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc", "value": "1000"},
                    {"metric": "coverage", "value": "85.5"},
                    {"metric": "bugs", "value": "5"},
                    {"metric": "vulnerabilities", "value": "2"},
                    {"metric": "code_smells", "value": "15"},
                    {"metric": "sqale_rating", "value": "A"},
                    {"metric": "reliability_rating", "value": "B"},
                    {"metric": "security_rating", "value": "A"},
                ]
            }
        }

        analyzer.session.get.return_value = mock_response

        metrics = analyzer.get_project_metrics("test-project")

        assert metrics is not None
        assert metrics.lines_of_code == 1000
        assert metrics.coverage == 85.5
        assert metrics.bugs == 5
        assert metrics.vulnerabilities == 2
        assert metrics.code_smells == 15

    def test_get_project_metrics_empty_response(self, analyzer):
        """Test handling of empty metrics response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"component": {"measures": []}}

        analyzer.session.get.return_value = mock_response

        metrics = analyzer.get_project_metrics("test-project")

        assert metrics is not None
        assert metrics.lines_of_code is None
        assert metrics.coverage is None

    def test_get_project_metrics_http_error(self, analyzer):
        """Test handling of HTTP error in metrics retrieval."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)

        analyzer.session.get.return_value = mock_response

        metrics = analyzer.get_project_metrics("test-project")

        assert metrics is None


class TestParseIssues:
    """Test issue parsing."""

    def test_parse_issues_success(self, analyzer, sample_issue_data):
        """Test successful issue parsing."""
        issues = analyzer._parse_issues([sample_issue_data])

        assert len(issues) == 1
        issue = issues[0]
        assert issue.key == sample_issue_data["key"]
        assert issue.rule == sample_issue_data["rule"]
        assert issue.severity == Severity.MAJOR
        assert issue.type == IssueType.CODE_SMELL
        assert issue.first_line == 42
        assert issue.last_line == 45

    def test_parse_issues_with_invalid_data(self, analyzer):
        """Test parsing with invalid issue data."""
        invalid_data = [
            {
                "key": "test-key",
                # Missing required fields
            }
        ]

        issues = analyzer._parse_issues(invalid_data)

        # Should handle gracefully, possibly returning empty list or partial data
        assert isinstance(issues, list)

    def test_parse_issues_severity_mapping(self, analyzer):
        """Test severity enum mapping."""
        test_data = [
            {"severity": "BLOCKER", "type": "BUG", "key": "1", "rule": "rule1", "component": "c", "project": "p", "message": "m"},
            {"severity": "CRITICAL", "type": "BUG", "key": "2", "rule": "rule2", "component": "c", "project": "p", "message": "m"},
            {"severity": "MAJOR", "type": "BUG", "key": "3", "rule": "rule3", "component": "c", "project": "p", "message": "m"},
            {"severity": "MINOR", "type": "BUG", "key": "4", "rule": "rule4", "component": "c", "project": "p", "message": "m"},
            {"severity": "INFO", "type": "BUG", "key": "5", "rule": "rule5", "component": "c", "project": "p", "message": "m"},
        ]

        issues = analyzer._parse_issues(test_data)

        assert len(issues) == 5
        assert issues[0].severity == Severity.BLOCKER
        assert issues[1].severity == Severity.CRITICAL
        assert issues[2].severity == Severity.MAJOR
        assert issues[3].severity == Severity.MINOR
        assert issues[4].severity == Severity.INFO


class TestExtractFilePath:
    """Test file path extraction."""

    def test_extract_file_path_with_colon(self, analyzer):
        """Test extracting file path from component with colon."""
        component = "project-key:src/main/java/MyClass.java"
        file_path = analyzer._extract_file_path(component)

        assert file_path == "src/main/java/MyClass.java"

    def test_extract_file_path_without_colon(self, analyzer):
        """Test extracting file path from component without colon."""
        component = "src/main.py"
        file_path = analyzer._extract_file_path(component)

        assert file_path == "src/main.py"

    def test_extract_file_path_empty(self, analyzer):
        """Test extracting file path from empty component."""
        file_path = analyzer._extract_file_path("")

        assert file_path is None


class TestGetFixableIssues:
    """Test filtering fixable issues."""

    def test_get_fixable_issues(self, analyzer, sample_issue_data):
        """Test retrieving only fixable issues."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": [sample_issue_data],
            "paging": {"total": 1, "pageIndex": 1, "pageSize": 500},
        }

        analyzer.session.get.return_value = mock_response

        fixable = analyzer.get_fixable_issues(
            project_key="test-project", branch="main"
        )

        assert isinstance(fixable, list)
        # Should only include issues that are fixable
        for issue in fixable:
            assert issue.is_fixable

    def test_get_fixable_issues_with_max_limit(self, analyzer, sample_issue_data):
        """Test limiting number of fixable issues."""
        # Create multiple issues
        issues = [dict(sample_issue_data, key=f"issue-{i}") for i in range(10)]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": issues,
            "paging": {"total": 10, "pageIndex": 1, "pageSize": 500},
        }

        analyzer.session.get.return_value = mock_response

        fixable = analyzer.get_fixable_issues(
            project_key="test-project", branch="main", max_issues=5
        )

        assert len(fixable) <= 5


class TestRuleFetching:
    """Test SonarCloud rule fetching."""

    def test_fetch_all_rules(self, analyzer):
        """Test fetching all rules."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rules": [
                {
                    "key": "python:S1481",
                    "name": "Unused local variables",
                    "lang": "python",
                    "type": "CODE_SMELL",
                    "severity": "MAJOR",
                    "htmlDesc": "Unused variables should be removed",
                }
            ],
            "total": 1,
        }

        analyzer.session.get.return_value = mock_response

        rules = analyzer.fetch_all_rules()

        assert "rules" in rules
        assert "metadata" in rules
        assert rules["metadata"]["total_rules"] > 0

    def test_get_rule_by_key(self, analyzer):
        """Test fetching specific rule by key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rule": {
                "key": "python:S1481",
                "name": "Unused local variables",
                "lang": "python",
                "htmlDesc": "Description",
            }
        }

        analyzer.session.get.return_value = mock_response

        rule = analyzer.get_rule_by_key("python:S1481")

        assert rule is not None
        assert rule["name"] == "Unused local variables"


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager_enter_exit(self):
        """Test using analyzer as context manager."""
        with patch("devdox_ai_sonar.sonar_analyzer.requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer

            with SonarCloudAnalyzer(
                token="test-token", organization="test-org"
            ) as analyzer:
                assert analyzer is not None

            # Session should be closed after exiting context
            mock_session.close.assert_called_once()


class TestProjectAnalysis:
    """Test project directory analysis."""

    def test_analyze_project_directory(self, analyzer, tmp_path):
        """Test analyzing a project directory structure."""
        # Create test project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "src" / "utils.py").write_text("def helper(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        (tmp_path / ".git").mkdir()

        analysis = analyzer.analyze_project_directory(tmp_path)

        assert analysis["total_files"] >= 3
        assert analysis["python_files"] >= 3
        assert analysis["has_git"] is True

    def test_analyze_project_invalid_path(self, analyzer):
        """Test analyzing non-existent project directory."""
        with pytest.raises(ValueError):
            analyzer.analyze_project_directory("/nonexistent/path")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
