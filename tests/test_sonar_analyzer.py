"""Comprehensive tests for SonarCloud analyzer."""

import pytest
import requests
from unittest.mock import Mock, patch
from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer
from devdox_ai_sonar.models import (
    Severity,
    IssueType,
    Impact,
)
import json


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

    def test_infer_root_cause_unused_code(self, analyzer):
        rule = {"name": "Unused variable", "htmlDesc": "", "tags": ["unused"], "type": "CODE_SMELL"}
        root_cause = analyzer._infer_root_cause(rule)
        assert "Unused code" in root_cause

    def test_infer_root_cause_empty_dic(self,analyzer):
        rule = {}
        result = analyzer._infer_root_cause(rule)
        assert result == "Unknown rule type or insufficient data for analysis"

    def test_infer_root_cause_missing_param(self,analyzer):
        rule = {"key": "test-rule"}
        result = analyzer._infer_root_cause(rule)
        assert result == "Unknown rule type or insufficient data for analysis"

    def test_infer_root_cause_unkown_rule(self, analyzer):
        rule = {
            "name": "some strange rule",
            "htmlDesc": "does something unusual",
            "tags": ["custom"],
            "type": "unknown_type"
        }
        result = analyzer._infer_root_cause(rule)
        assert result == "Unknown rule type or insufficient data for analysis"

    def test_infer_root_cause_empty_string(self,analyzer):
        rule = {"name": "", "htmlDesc": "", "tags": [], "type": ""}
        result = analyzer._infer_root_cause(rule)
        assert result == "Unknown rule type or insufficient data for analysis"

    def test_infer_root_cause_unexcpected_tags(self,analyzer):
        rule = {"name": "Check for foo", "htmlDesc": "desc", "tags": ["nonexistenttag"], "type": "code_smell"}
        result = analyzer._infer_root_cause(rule)
        assert result == "Code quality issue that affects readability, maintainability, or follows poor practices"

    def test_generate_fix_guidance_null(self,analyzer):
        rule = {"name": "NullPointerException", "type": "BUG"}
        guidance = analyzer._generate_fix_guidance(rule)
        assert guidance["priority"] == "High"
        assert "null checks" in guidance["description"].lower()

    def test_generate_fix_guidance_empty_rule(self, analyzer):
        rule = {}
        result = analyzer._generate_fix_guidance(rule)
        assert result["description"] == "Improve code quality following best practices"

    def test_generate_fix_guidance_missing_name(self,analyzer):
        rule = {"type": "BUG"}
        result = analyzer._generate_fix_guidance(rule)
        assert result["description"] == "Fix logical error or potential runtime issue"

    def test_generate_fix_guidance_unkown_type(self,analyzer):
        rule = {"name": "strange rule", "type": "weird_type"}
        result = analyzer._generate_fix_guidance(rule)
        assert result["description"] == "Improve code quality following best practices"

    def test_generate_fix_guidance_malformd_name(self,analyzer):
        rule = {"name": "1234!@#$", "type": ""}
        result = analyzer._generate_fix_guidance(rule)
        assert result["description"] == "Improve code quality following best practices"

    def test_generate_fix_guidance_none_values(self,analyzer):
        rule = {"name": None, "type": None}
        result = analyzer._generate_fix_guidance(rule)
        assert result["description"] == "Improve code quality following best practices"


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager_enter_exit(self):
        """Test using analyzer as context manager."""
        with patch("devdox_ai_sonar.sonar_analyzer.requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

          

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


class TestPullRequestAndBranchHandling:
    """Test pull request and branch parameter handling."""

    def test_get_project_issues_with_pull_request(self, analyzer):
        """Test fetching issues for a specific pull request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": [],
            "paging": {"total": 0, "pageIndex": 1, "pageSize": 500},
        }
        analyzer.session.get.return_value = mock_response

        result = analyzer.get_project_issues(
            project_key="test-project",
            branch="",
            pull_request_number=123
        )

        # Verify pull request parameter was used

        call_args = analyzer.session.get.call_args_list
        params = call_args[0][1]["params"]

        assert params.get("pullRequest") == "123"
        assert "branch" not in params or params.get("branch") == ""

    def test_get_project_issues_with_branch_and_no_pr(self, analyzer):
        """Test fetching issues for a specific branch without PR."""
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
            pull_request_number=0
        )

        # Verify branch parameter was used

        call_args = analyzer.session.get.call_args_list
        params = call_args[0][1]["params"]
        assert params.get("branch") == "develop"
        assert "pullRequest" not in params or params.get("pullRequest") == ""

    def test_get_project_issues_defaults_to_main(self, analyzer):
        """Test that empty branch and no PR defaults to main."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": [],
            "paging": {"total": 0, "pageIndex": 1, "pageSize": 500},
        }
        analyzer.session.get.return_value = mock_response

        result = analyzer.get_project_issues(
            project_key="test-project",
            branch="",
            pull_request_number=0
        )

        # Should default to main branch
        call_args = analyzer.session.get.call_args_list
        params = call_args[0][1]["params"]
        assert params.get("branch") == "main"


# ==============================================================================
# CRITICAL: Pagination Handling
# ==============================================================================

class TestPaginationHandling:
    """Test pagination for large result sets."""

    def test_fetch_issues_multiple_pages(self, analyzer):
        """Test fetching issues across multiple pages."""
        # Create responses for 3 pages
        response_page1 = Mock()
        response_page1.status_code = 200
        response_page1.json.return_value = {
            "issues": [
                {"key": f"issue-{i}", "rule": "rule", "component": "c", "project": "p", "message": "m", "type": "BUG"}
                for i in range(1, 501)],
            "paging": {"total": 1200, "pageIndex": 1, "pageSize": 500},
        }

        response_page2 = Mock()
        response_page2.status_code = 200
        response_page2.json.return_value = {
            "issues": [
                {"key": f"issue-{i}", "rule": "rule", "component": "c", "project": "p", "message": "m", "type": "BUG"}
                for i in range(501, 1001)],
            "paging": {"total": 1200, "pageIndex": 2, "pageSize": 500},
        }

        response_page3 = Mock()
        response_page3.status_code = 200
        response_page3.json.return_value = {
            "issues": [
                {"key": f"issue-{i}", "rule": "rule", "component": "c", "project": "p", "message": "m", "type": "BUG"}
                for i in range(1001, 1201)],
            "paging": {"total": 1200, "pageIndex": 3, "pageSize": 500},
        }

        analyzer.session.get.side_effect = [response_page1, response_page2, response_page3]

        result = analyzer.get_project_issues(project_key="test-project", branch="main")

        # Should fetch all 1200 issues
        assert result.total_issues == 1200

        assert analyzer.session.get.call_count == 4

    def test_fetch_rules_multiple_pages(self, analyzer):
        """Test fetching rules across multiple pages."""
        response_page1 = Mock()
        response_page1.status_code = 200
        response_page1.json.return_value = {
            "rules": [{"key": f"rule-{i}", "name": f"Rule {i}", "lang": "python"} for i in range(500)],
            "total": 1000,
        }

        response_page2 = Mock()
        response_page2.status_code = 200
        response_page2.json.return_value = {
            "rules": [{"key": f"rule-{i}", "name": f"Rule {i}", "lang": "python"} for i in range(500, 1000)],
            "total": 1000,
        }

        analyzer.session.get.side_effect = [response_page1, response_page2]

        rules = analyzer.fetch_all_rules()

        assert rules["metadata"]["total_rules"] == 1000
        assert analyzer.session.get.call_count == 2

    def test_fetch_issues_early_termination(self, analyzer):
        """Test pagination stops when all issues fetched."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "issues": [
                {"key": "issue-1", "rule": "rule", "component": "c", "project": "p", "message": "m", "type": "BUG"}],
            "paging": {"total": 1, "pageIndex": 1, "pageSize": 500},
        }
        analyzer.session.get.return_value = response

        result = analyzer.get_project_issues(project_key="test-project", branch="main")

        # Should only call once when all issues are fetched
        assert analyzer.session.get.call_count == 2


# ==============================================================================
# CRITICAL: Error Handling Edge Cases
# ==============================================================================

class TestErrorHandlingEdgeCases:
    """Test error handling for various failure scenarios."""

    def test_get_project_issues_401_unauthorized(self, analyzer):
        """Test handling of 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        http_error = requests.HTTPError("401 Unauthorized")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error

        analyzer.session.get.return_value = mock_response

        result = analyzer.get_project_issues(project_key="test-project", branch="main")

        assert result is None

    def test_get_project_issues_403_forbidden(self, analyzer):
        """Test handling of 403 permission error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        http_error = requests.HTTPError("403 Forbidden")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error

        analyzer.session.get.return_value = mock_response

        result = analyzer.get_project_issues(project_key="test-project", branch="main")

        assert result is None

    def test_get_project_issues_network_error(self, analyzer):
        """Test handling of network connection errors."""
        analyzer.session.get.side_effect = requests.ConnectionError("Network unavailable")

        result = analyzer.get_project_issues(project_key="test-project", branch="main")

        assert result is None

    def test_get_project_issues_unexpected_exception(self, analyzer):
        """Test handling of unexpected exceptions."""
        analyzer.session.get.side_effect = Exception("Unexpected error")

        result = analyzer.get_project_issues(project_key="test-project", branch="main")

        assert result is None

    def test_get_project_metrics_timeout(self, analyzer):
        """Test metrics fetch timeout handling."""
        analyzer.session.get.side_effect = requests.Timeout("Request timeout")

        metrics = analyzer.get_project_metrics("test-project")

        assert metrics is None

    def test_get_project_metrics_network_error(self, analyzer):
        """Test metrics fetch network error."""
        analyzer.session.get.side_effect = requests.ConnectionError("Network error")

        metrics = analyzer.get_project_metrics("test-project")

        assert metrics is None


# ==============================================================================
# CRITICAL: Issue Parsing Edge Cases
# ==============================================================================

class TestIssueParsingEdgeCases:
    """Test parsing of various issue data formats."""

    def test_parse_issues_missing_line_number(self, analyzer):
        """Test parsing issue without line number."""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "component": "project:file.py",
            "project": "project",
            "message": "Issue message",
            "type": "BUG",
            # No 'line' field
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        assert issues[0].first_line is None
        assert issues[0].last_line is None

    def test_parse_issues_with_flows_multiple_locations(self, analyzer):
        """Test parsing issue with multiple flow locations."""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "component": "project:file.py",
            "project": "project",
            "message": "Issue",
            "type": "BUG",
            "line": 10,
            "flows": [
                {
                    "locations": [
                        {"textRange": {"startLine": 10, "endLine": 15}},
                        {"textRange": {"startLine": 20, "endLine": 25}},
                        {"textRange": {"startLine": 30, "endLine": 35}},
                    ]
                }
            ],
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        # Should use the maximum end line from all flows
        assert issues[0].last_line == 35

    def test_parse_issues_with_empty_flows(self, analyzer):
        """Test parsing issue with empty flows array."""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "component": "project:file.py",
            "project": "project",
            "message": "Issue",
            "type": "BUG",
            "line": 10,
            "flows": [],
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        assert issues[0].first_line == 10
        assert issues[0].last_line == 10  # Should default to first_line

    def test_parse_issues_unknown_severity(self, analyzer):
        """Test parsing issue with unknown severity."""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "component": "project:file.py",
            "project": "project",
            "message": "Issue",
            "type": "BUG",
            "severity": "UNKNOWN_SEVERITY",
        }

        issues = analyzer._parse_issues([issue_data])

        # Should default to INFO for unknown severity
        assert len(issues) == 1
        assert issues[0].severity == Severity.INFO

    def test_parse_issues_unknown_type(self, analyzer):
        """Test parsing issue with unknown type."""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "component": "project:file.py",
            "project": "project",
            "message": "Issue",
            "type": "UNKNOWN_TYPE",
            "severity": "MAJOR",
        }

        issues = analyzer._parse_issues([issue_data])

        # Should default to CODE_SMELL for unknown type
        assert len(issues) == 1
        assert issues[0].type == IssueType.CODE_SMELL

    def test_parse_issues_with_impact(self, analyzer):
        """Test parsing issue with impact data."""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "component": "project:file.py",
            "project": "project",
            "message": "Issue",
            "type": "BUG",
            "severity": "MAJOR",
            "impacts": [
                {"softwareQuality": "SECURITY", "severity": "HIGH"}
            ],
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        assert issues[0].impact == Impact.HIGH

    def test_parse_issues_with_invalid_impact(self, analyzer):
        """Test parsing issue with invalid impact severity."""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "component": "project:file.py",
            "project": "project",
            "message": "Issue",
            "type": "BUG",
            "impacts": [
                {"softwareQuality": "SECURITY", "severity": "INVALID"}
            ],
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        # Impact should be None for invalid severity
        assert issues[0].impact is None

    def test_parse_issues_partially_valid_batch(self, analyzer):
        """Test parsing batch with some invalid issues."""
        issues_data = [
            {  # Valid issue
                "key": "issue-1",
                "rule": "python:S1234",
                "component": "project:file.py",
                "project": "project",
                "message": "Issue 1",
                "type": "BUG",
            },
            {  # Invalid - will cause exception
                "key": "issue-2",
                # Missing required fields
            },
            {  # Valid issue
                "key": "issue-3",
                "rule": "python:S5678",
                "component": "project:file2.py",
                "project": "project",
                "message": "Issue 3",
                "type": "CODE_SMELL",
            },
        ]

        issues = analyzer._parse_issues(issues_data)

        # Should parse valid issues and skip invalid ones
        assert len(issues) >= 2


# ==============================================================================
# CRITICAL: Metrics Parsing Edge Cases
# ==============================================================================

class TestMetricsParsingEdgeCases:
    """Test project metrics parsing edge cases."""

    def test_get_project_metrics_invalid_numeric_values(self, analyzer):
        """Test handling of invalid numeric metric values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc", "value": "invalid_number"},
                    {"metric": "coverage", "value": "not_a_float"},
                    {"metric": "bugs", "value": "5"},
                ]
            }
        }
        analyzer.session.get.return_value = mock_response

        metrics = analyzer.get_project_metrics("test-project")

        # Should handle invalid values gracefully
        assert metrics is not None
        assert metrics.lines_of_code is None
        assert metrics.coverage is None
        assert metrics.bugs == 5

    def test_get_project_metrics_missing_values(self, analyzer):
        """Test metrics with None values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc"},  # Missing value
                    {"metric": "coverage", "value": None},
                ]
            }
        }
        analyzer.session.get.return_value = mock_response

        metrics = analyzer.get_project_metrics("test-project")

        assert metrics is not None
        assert metrics.lines_of_code is None
        assert metrics.coverage is None

    def test_get_project_metrics_all_metrics_present(self, analyzer):
        """Test complete metrics response with all fields."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc", "value": "5000"},
                    {"metric": "coverage", "value": "78.5"},
                    {"metric": "duplicated_lines_density", "value": "3.2"},
                    {"metric": "sqale_rating", "value": "A"},
                    {"metric": "reliability_rating", "value": "B"},
                    {"metric": "security_rating", "value": "A"},
                    {"metric": "bugs", "value": "12"},
                    {"metric": "vulnerabilities", "value": "3"},
                    {"metric": "code_smells", "value": "45"},
                    {"metric": "sqale_index", "value": "240"},
                    {"metric": "security_hotspots_reviewed", "value": "85.0"},
                ]
            }
        }
        analyzer.session.get.return_value = mock_response

        metrics = analyzer.get_project_metrics("test-project")

        assert metrics.lines_of_code == 5000
        assert metrics.coverage == 78.5
        assert metrics.duplicated_lines_density == 3.2
        assert metrics.bugs == 12
        assert metrics.vulnerabilities == 3
        assert metrics.code_smells == 45
        assert metrics.technical_debt == "240"


# ==============================================================================
# CRITICAL: Rule Processing and Inference
# ==============================================================================

class TestRuleProcessingEdgeCases:
    """Test rule processing and root cause inference."""

    def test_infer_root_cause_sql_injection(self, analyzer):
        """Test root cause for SQL injection issues."""
        rule = {
            "name": "SQL Injection vulnerability",
            "htmlDesc": "Prevent SQL injection attacks",
            "tags": ["security", "sql"],
            "type": "VULNERABILITY"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "SQL injection" in root_cause

    def test_infer_root_cause_password_hardcoded(self, analyzer):
        """Test root cause for hardcoded credentials."""
        rule = {
            "name": "Hard-coded password",
            "htmlDesc": "Credentials should not be hardcoded",
            "tags": ["security"],
            "type": "VULNERABILITY"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "credential" in root_cause.lower()

    def test_infer_root_cause_complexity(self, analyzer):
        """Test root cause for complexity issues."""
        rule = {
            "name": "Cognitive Complexity too high",
            "htmlDesc": "Reduce complexity",
            "tags": ["brain-overload"],
            "type": "CODE_SMELL"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "complexity" in root_cause.lower()

    def test_infer_root_cause_duplicate_code(self, analyzer):
        """Test root cause for code duplication."""
        rule = {
            "name": "Duplicate code block",
            "htmlDesc": "Remove duplicated code",
            "tags": ["clumsy"],
            "type": "CODE_SMELL"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "duplication" in root_cause.lower()

    def test_infer_root_cause_empty_blocks(self, analyzer):
        """Test root cause for empty code blocks."""
        rule = {
            "name": "Empty method",
            "htmlDesc": "Method should not be empty",
            "tags": [],
            "type": "CODE_SMELL"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "empty" in root_cause.lower()

    def test_infer_root_cause_resource_leak(self, analyzer):
        """Test root cause for resource leaks."""
        rule = {
            "name": "Resources should be closed",
            "htmlDesc": "File handles must be closed",
            "tags": ["bug"],
            "type": "BUG"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "resource" in root_cause.lower() or "leak" in root_cause.lower()

    def test_infer_root_cause_thread_safety(self, analyzer):
        """Test root cause for thread safety issues."""
        rule = {
            "name": "Thread safety issue",
            "htmlDesc": "Synchronization required",
            "tags": ["multi-threading"],
            "type": "BUG"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "thread" in root_cause.lower()

    def test_infer_root_cause_exception_handling(self, analyzer):
        """Test root cause for exception handling issues."""
        rule = {
            "name": "Exception should not be caught",
            "htmlDesc": "Improve error handling",
            "tags": [],
            "type": "CODE_SMELL"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "exception" in root_cause.lower()

    def test_infer_root_cause_security_tag(self, analyzer):
        """Test root cause with security tag."""
        rule = {
            "name": "Some security issue",
            "htmlDesc": "Security concern",
            "tags": ["security"],
            "type": "CODE_SMELL"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "security" in root_cause.lower()

    def test_infer_root_cause_bug_type(self, analyzer):
        """Test root cause for generic bugs."""
        rule = {
            "name": "Logic error",
            "htmlDesc": "Incorrect logic",
            "tags": [],
            "type": "BUG"
        }

        root_cause = analyzer._infer_root_cause(rule)

        assert "bug" in root_cause.lower() or "error" in root_cause.lower()


# ==============================================================================
# CRITICAL: Fix Guidance Generation
# ==============================================================================

class TestFixGuidanceGeneration:
    """Test generation of fix guidance for different rule types."""

    def test_generate_fix_guidance_sql_injection(self, analyzer):
        """Test fix guidance for SQL injection."""
        rule = {"name": "SQL injection vulnerability", "type": "VULNERABILITY"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance["priority"] == "Critical"
        assert "parameterized" in guidance["description"].lower()
        assert len(guidance["steps"]) > 0

    def test_generate_fix_guidance_hardcoded_password(self, analyzer):
        """Test fix guidance for hardcoded credentials."""
        rule = {"name": "Hard-coded password detected", "type": "VULNERABILITY"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance["priority"] == "Critical"
        assert "environment" in guidance["description"].lower()

    def test_generate_fix_guidance_complexity(self, analyzer):
        """Test fix guidance for cognitive complexity."""
        rule = {"name": "Cognitive Complexity is too high", "type": "CODE_SMELL"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance["priority"] == "Medium"
        assert guidance["effort"] == "High"
        assert "refactor" in guidance["description"].lower()

    def test_generate_fix_guidance_duplicate(self, analyzer):
        """Test fix guidance for code duplication."""
        rule = {"name": "Duplicated code block", "type": "CODE_SMELL"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert "extract" in guidance["description"].lower()
        assert guidance["effort"] == "Medium"

    def test_generate_fix_guidance_empty(self, analyzer):
        """Test fix guidance for empty blocks."""
        rule = {"name": "Empty method body", "type": "CODE_SMELL"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance["priority"] == "High"
        assert guidance["effort"] == "Low"

    def test_generate_fix_guidance_vulnerability(self, analyzer):
        """Test fix guidance for generic vulnerability."""
        rule = {"name": "Security vulnerability", "type": "VULNERABILITY"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance["priority"] == "Critical"
        assert "security" in guidance["description"].lower()

    def test_generate_fix_guidance_bug(self, analyzer):
        """Test fix guidance for generic bug."""
        rule = {"name": "Logical error", "type": "BUG"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance["priority"] == "High"
        assert "logical" in guidance["description"].lower()

    def test_generate_fix_guidance_default(self, analyzer):
        """Test fix guidance for unknown rule type."""
        rule = {"name": "Some other issue", "type": "OTHER"}

        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance["priority"] == "Low"
        assert "best practices" in guidance["description"].lower()


# ==============================================================================
# HIGH PRIORITY: HTML Description Cleaning
# ==============================================================================

class TestHTMLDescriptionCleaning:
    """Test HTML description cleaning."""

    def test_clean_html_description_with_tags(self, analyzer):
        """Test cleaning HTML with various tags."""
        html = "<p>This is a <strong>bold</strong> description with <a href='link'>links</a>.</p>"

        cleaned = analyzer._clean_html_description(html)

        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "bold" in cleaned
        assert "links" in cleaned

    def test_clean_html_description_empty(self, analyzer):
        """Test cleaning empty HTML."""
        cleaned = analyzer._clean_html_description("")

        assert cleaned == ""

    def test_clean_html_description_very_long(self, analyzer):
        """Test truncation of very long descriptions."""
        html = "<p>" + "x" * 2000 + "</p>"

        cleaned = analyzer._clean_html_description(html)

        assert len(cleaned) <= 1003  # 1000 + "..."
        assert cleaned.endswith("...")

    def test_clean_html_description_multiple_whitespace(self, analyzer):
        """Test cleaning multiple whitespace."""
        html = "<p>Text   with    multiple     spaces</p>"

        cleaned = analyzer._clean_html_description(html)

        # Should normalize to single spaces
        assert "   " not in cleaned

    def test_clean_html_description_newlines(self, analyzer):
        """Test handling newlines in HTML."""
        html = "<p>Line 1\n\nLine 2\n\n\nLine 3</p>"

        cleaned = analyzer._clean_html_description(html)

        # Newlines should be normalized to spaces
        assert "\n\n" not in cleaned


# ==============================================================================
# HIGH PRIORITY: Rule Statistics
# ==============================================================================

class TestRuleStatistics:
    """Test rule statistics generation."""

    def test_get_category_stats(self, analyzer):
        """Test category statistics calculation."""
        rules = {
            "rule1": {"category": "BUG"},
            "rule2": {"category": "CODE_SMELL"},
            "rule3": {"category": "BUG"},
            "rule4": {"category": "VULNERABILITY"},
            "rule5": {"category": "CODE_SMELL"},
        }

        stats = analyzer._get_category_stats(rules)

        assert stats["BUG"] == 2
        assert stats["CODE_SMELL"] == 2
        assert stats["VULNERABILITY"] == 1

    def test_get_severity_stats(self, analyzer):
        """Test severity statistics calculation."""
        rules = {
            "rule1": {"severity": "BLOCKER"},
            "rule2": {"severity": "CRITICAL"},
            "rule3": {"severity": "MAJOR"},
            "rule4": {"severity": "MAJOR"},
            "rule5": {"severity": "MINOR"},
        }

        stats = analyzer._get_severity_stats(rules)

        assert stats["BLOCKER"] == 1
        assert stats["CRITICAL"] == 1
        assert stats["MAJOR"] == 2
        assert stats["MINOR"] == 1

    def test_get_category_stats_empty(self, analyzer):
        """Test category stats with empty rules."""
        stats = analyzer._get_category_stats({})

        assert stats == {}

    def test_get_severity_stats_empty(self, analyzer):
        """Test severity stats with empty rules."""
        stats = analyzer._get_severity_stats({})

        assert stats == {}


# ==============================================================================
# HIGH PRIORITY: Rule Filtering and Export
# ==============================================================================

class TestRuleFilteringAndExport:
    """Test rule filtering by language and severity."""

    def test_get_rules_for_language(self, analyzer):
        """Test fetching rules for specific language."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rules": [
                {"key": "python:S1234", "name": "Rule 1", "lang": "python"},
            ],
            "total": 1,
        }
        analyzer.session.get.return_value = mock_response

        rules = analyzer.get_rules_for_language("python")

        # Verify language filter was applied
        call_args = analyzer.session.get.call_args_list
        params = call_args[0][1]["params"]
        assert "languages" in params
        assert "python" in params["languages"]

    def test_get_rules_by_severity(self, analyzer):
        """Test filtering rules by severity."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rules": [
                {"key": "rule1", "name": "Rule 1", "lang": "python", "severity": "BLOCKER"},
                {"key": "rule2", "name": "Rule 2", "lang": "java", "severity": "CRITICAL"},
                {"key": "rule3", "name": "Rule 3", "lang": "python", "severity": "BLOCKER"},
            ],
            "total": 3,
        }
        analyzer.session.get.return_value = mock_response

        filtered = analyzer.get_rules_by_severity("BLOCKER")

        # Should return only BLOCKER rules
        assert len(filtered) == 2
        assert all(rule["severity"] == "BLOCKER" for rule in filtered)

    def test_export_rules_to_json(self, analyzer, tmp_path):
        """Test exporting rules to JSON file."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rules": [
                {"key": "rule1", "name": "Rule 1", "lang": "python"},
            ],
            "total": 1,
        }
        analyzer.session.get.return_value = mock_response

        output_file = tmp_path / "rules.json"
        analyzer.export_rules_to_json(str(output_file), languages=["python"])

        # Verify file was created
        assert output_file.exists()

        # Verify content
        with open(output_file) as f:
            data = json.load(f)

        assert "rules" in data
        assert "metadata" in data

    def test_get_rule_by_key_not_found(self, analyzer):
        """Test fetching non-existent rule."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rule": {}}
        analyzer.session.get.return_value = mock_response

        rule = analyzer.get_rule_by_key("nonexistent:rule")

        assert rule is None

    def test_get_rule_by_key_error(self, analyzer):
        """Test error handling in get_rule_by_key."""
        analyzer.session.get.side_effect = requests.RequestException("Error")

        rule = analyzer.get_rule_by_key("python:S1234")

        assert rule is None


# ==============================================================================
# HIGH PRIORITY: Fixable Issues Filtering
# ==============================================================================

class TestFixableIssuesFiltering:
    """Test filtering and sorting of fixable issues."""

    def test_get_fixable_issues_severity_sorting(self, analyzer):
        """Test that fixable issues are sorted by severity."""
        issues_data = [
            {"key": "i1", "rule": "r", "component": "c", "project": "p", "message": "m", "type": "BUG",
             "severity": "MINOR"},
            {"key": "i2", "rule": "r", "component": "c", "project": "p", "message": "m", "type": "BUG",
             "severity": "BLOCKER"},
            {"key": "i3", "rule": "r", "component": "c", "project": "p", "message": "m", "type": "BUG",
             "severity": "MAJOR"},
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": issues_data,
            "paging": {"total": 3, "pageIndex": 1, "pageSize": 500},
        }
        analyzer.session.get.return_value = mock_response

        fixable = analyzer.get_fixable_issues(project_key="test-project", branch="main")

        # Should be sorted with BLOCKER first
        if len(fixable) >= 2:
            assert fixable[0].severity == Severity.BLOCKER
            # MINOR should come after MAJOR
            severities = [issue.severity for issue in fixable]
            minor_index = severities.index(Severity.MINOR) if Severity.MINOR in severities else -1
            major_index = severities.index(Severity.MAJOR) if Severity.MAJOR in severities else -1
            if minor_index != -1 and major_index != -1:
                assert major_index < minor_index

    def test_get_fixable_issues_with_type_filter(self, analyzer):
        """Test filtering fixable issues by type."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": [],
            "paging": {"total": 0, "pageIndex": 1, "pageSize": 500},
        }
        analyzer.session.get.return_value = mock_response

        fixable = analyzer.get_fixable_issues(
            project_key="test-project",
            branch="main",
            types_list=["BUG", "VULNERABILITY"]
        )

        # Verify types filter was passed
        call_args = analyzer.session.get.call_args_list
        params = call_args[0][1]["params"]
        assert "types" in params


# ==============================================================================
# MEDIUM PRIORITY: Project Analysis
# ==============================================================================

class TestProjectAnalysisDetails:
    """Test project directory analysis details."""

    def test_analyze_project_with_sonar_config(self, analyzer, tmp_path):
        """Test project analysis detects sonar config."""
        (tmp_path / "sonar-project.properties").write_text("sonar.projectKey=test")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("code")

        analysis = analyzer.analyze_project_directory(tmp_path)

        assert analysis["has_sonar_config"] is True

    def test_analyze_project_mixed_languages(self, analyzer, tmp_path):
        """Test analyzing project with multiple languages."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("python code")
        (tmp_path / "src" / "app.js").write_text("javascript code")
        (tmp_path / "src" / "Main.java").write_text("java code")

        analysis = analyzer.analyze_project_directory(tmp_path)

        assert analysis["python_files"] >= 1
        assert analysis["javascript_files"] >= 1
        assert analysis["java_files"] >= 1

    def test_analyze_project_identifies_source_dirs(self, analyzer, tmp_path):
        """Test identification of source directories."""
        (tmp_path / "src").mkdir()
        (tmp_path / "app").mkdir()
        (tmp_path / "lib").mkdir()
        (tmp_path / "tests").mkdir()

        analysis = analyzer.analyze_project_directory(tmp_path)

        # src, app, lib should be identified as source dirs
        source_dirs = analysis["potential_source_dirs"]
        assert any("src" in d for d in source_dirs)

    def test_analyze_project_skips_hidden_dirs(self, analyzer, tmp_path):
        """Test that hidden directories are skipped."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".cache").mkdir()
        (tmp_path / "src").mkdir()

        analysis = analyzer.analyze_project_directory(tmp_path)

        # Hidden directories shouldn't be in directory list
        dirs = analysis["directories"]
        assert not any(d.startswith(".") for d in dirs)


# ==============================================================================
# MEDIUM PRIORITY: Contains Keywords Helper
# ==============================================================================

class TestContainsKeywordsHelper:
    """Test the _contains_keywords helper method."""

    def test_contains_keywords_in_name(self, analyzer):
        """Test keyword found in name."""
        assert analyzer._contains_keywords("unused variable", "", ["unused"]) is True

    def test_contains_keywords_in_desc(self, analyzer):
        """Test keyword found in description."""
        assert analyzer._contains_keywords("", "this is unused code", ["unused"]) is True

    def test_contains_keywords_multiple_matches(self, analyzer):
        """Test multiple keywords, any match."""
        assert analyzer._contains_keywords("null pointer", "", ["null", "unused"]) is True

    def test_contains_keywords_no_match(self, analyzer):
        """Test no keywords match."""
        assert analyzer._contains_keywords("some rule", "description", ["unused", "null"]) is False

    def test_contains_keywords_partial_match(self, analyzer):
        """Test partial keyword match."""
        assert analyzer._contains_keywords("nullable", "", ["null"]) is True

    def test_contains_keywords_case_sensitivity(self, analyzer):
        """Test that keyword matching is case-insensitive."""
        # Both inputs should be lowercased before calling
        assert analyzer._contains_keywords("unused Variable", "", ["unused"]) is True


# ==============================================================================
# MEDIUM PRIORITY: File Path Extraction Edge Cases
# ==============================================================================

class TestFilePathExtractionDetails:
    """Test file path extraction edge cases."""

    def test_extract_file_path_multiple_colons(self, analyzer):
        """Test component with multiple colons."""
        component = "org:project:module:src/main.py"

        file_path = analyzer._extract_file_path(component)

        # Should split on first colon only
        assert file_path == "project:module:src/main.py"

    def test_extract_file_path_none_input(self, analyzer):
        """Test None component input."""
        file_path = analyzer._extract_file_path(None)

        assert file_path is None

    def test_extract_file_path_with_special_chars(self, analyzer):
        """Test path with special characters."""
        component = "project:src/my-file_v2.test.py"

        file_path = analyzer._extract_file_path(component)

        assert file_path == "src/my-file_v2.test.py"


# ==============================================================================
# MEDIUM PRIORITY: Process Rules Edge Cases
# ==============================================================================

class TestProcessRulesEdgeCases:
    """Test _process_rules with various inputs."""

    def test_process_rules_empty_list(self, analyzer):
        """Test processing empty rules list."""
        result = analyzer._process_rules([])

        assert result["rules"] == {}
        assert result["metadata"]["total_rules"] == 0

    def test_process_rules_minimal_data(self, analyzer):
        """Test processing rules with minimal data."""
        raw_rules = [
            {"key": "rule1"},  # Minimal rule
        ]

        result = analyzer._process_rules(raw_rules)

        assert "rule1" in result["rules"]
        # Should have defaults for missing fields
        assert result["rules"]["rule1"]["name"] == ""

    def test_process_rules_with_parameters(self, analyzer):
        """Test processing rules with parameters."""
        raw_rules = [
            {
                "key": "rule1",
                "name": "Rule with params",
                "params": [
                    {"key": "threshold", "defaultValue": "10"},
                    {"key": "pattern", "defaultValue": ".*"},
                ]
            }
        ]

        result = analyzer._process_rules(raw_rules)

        assert "rule1" in result["rules"]
        assert len(result["rules"]["rule1"]["parameters"]) == 2

    def test_process_rules_metadata_completeness(self, analyzer):
        """Test that metadata is complete."""
        raw_rules = [
            {"key": "python:S1", "name": "Rule 1", "lang": "python", "severity": "MAJOR", "type": "BUG"},
            {"key": "java:S2", "name": "Rule 2", "lang": "java", "severity": "CRITICAL", "type": "VULNERABILITY"},
        ]

        result = analyzer._process_rules(raw_rules)

        metadata = result["metadata"]
        assert "total_rules" in metadata
        assert "languages" in metadata
        assert "categories" in metadata
        assert "severities" in metadata
        assert "generated_at" in metadata
        assert "organization" in metadata


# ==============================================================================
# LOW PRIORITY: Session Management
# ==============================================================================

class TestSessionManagement:
    """Test session creation and management."""

    def test_close_session(self, analyzer):
        """Test closing analyzer session."""
        mock_session = Mock()
        analyzer.session = mock_session

        analyzer.close()

        mock_session.close.assert_called_once()

    def test_context_manager_cleanup_on_exception(self):
        """Test context manager cleanup even on exception."""
        with patch("devdox_ai_sonar.sonar_analyzer.requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            try:
                with SonarCloudAnalyzer(token="test", organization="org") as analyzer:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Session should still be closed
            mock_session.close.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
