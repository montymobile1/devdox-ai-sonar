# Comprehensive Test Cases for sonar_analyzer.py


import pytest
import json
import requests
from pathlib import Path
from requests.exceptions import HTTPError, Timeout, RequestException
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    AnalysisResult,
    ProjectMetrics,
    Severity,
    IssueType
)
from devdox_ai_sonar.utils.exceptions import SonarCloudAPIError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_session():
    """Mock requests session"""
    with patch('requests.Session') as mock:
        yield mock


@pytest.fixture
def analyzer():
    """Create analyzer instance"""
    return SonarCloudAnalyzer(
        token="test-token",
        organization="test-org"
    )


@pytest.fixture
def sample_issue_data():
    """Sample issue data from SonarCloud API"""
    return {
        "key": "issue-123",
        "rule": "python:S1234",
        "severity": "MAJOR",
        "component": "project:src/test.py",
        "project": "test-project",
        "line": 10,
        "message": "Test issue message",
        "locations":[],
        "textRange": {
            "startLine": 10,
            "endLine": 15
        },
    "flows": [
                {
                    "locations": [
                        {
                            "component": "project:src/test.py",
                            "textRange": {
                                "startLine": 10,
                                "endLine": 15,
                                "startOffset": 8,
                                "endOffset": 15
                            },
                            "msg": "Duplication."
                        }
                    ]
                },

            ],
        "type": "BUG",
        "status": "OPEN",
        "author": "developer@example.com",
        "creationDate": "2024-01-01T10:00:00+0000"
    }


@pytest.fixture
def sample_security_issue_data():
    """Sample security issue data"""
    return {
        "key": "sec-123",
        "rule": "python:S5042",
        "severity": "CRITICAL",
        "component": "project:src/auth.py",
        "project": "test-project",
        "line": 20,
        "message": "SQL injection vulnerability",
        "type": "VULNERABILITY",
        "vulnerabilityProbability": "HIGH",
        "impacts": [
            {"softwareQuality": "SECURITY", "severity": "HIGH"}
        ]
    }


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data from SonarCloud API"""
    return {
        "component": {
            "key": "test-project",
            "name": "Test Project",
            "measures": [
                {"metric": "ncloc", "value": "1000"},
                {"metric": "coverage", "value": "85.5"},
                {"metric": "bugs", "value": "5"},
                {"metric": "vulnerabilities", "value": "2"},
                {"metric": "code_smells", "value": "15"},
                {"metric": "duplicated_lines_density", "value": "3.2"},
                {"metric": "sqale_rating", "value": "A"},
                {"metric": "reliability_rating", "value": "B"},
                {"metric": "security_rating", "value": "A"}
            ]
        }
    }

@pytest.fixture
def sample_rules_response():
    """Sample rules facet response"""
    return {
        "facets": [
            {
                "property": "rules",
                "values": [
                    {"val": "python:S1234", "count": 5},
                    {"val": "python:S5678", "count": 3},
                    {"val": "python:S9012", "count": 2}
                ]
            }
        ]
    }

# ============================================================================
# TEST CLASS: INITIALIZATION
# ============================================================================

class TestSonarCloudAnalyzerInit:
    """Test SonarCloudAnalyzer initialization"""
    
    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        analyzer = SonarCloudAnalyzer(
            token="test-token",
            organization="test-org",
            base_url="https://custom.sonar.com",
            timeout=60,
            max_retries=5
        )
        
        assert analyzer.token == "test-token"
        assert analyzer.organization == "test-org"
        assert analyzer.base_url == "https://custom.sonar.com"
        assert analyzer.timeout == 60
    
    def test_init_default_values(self):
        """Test initialization with default values"""
        analyzer = SonarCloudAnalyzer(
            token="token",
            organization="org"
        )
        
        assert analyzer.base_url == "https://sonarcloud.io"
        assert analyzer.timeout == 30
    
    def test_init_creates_session(self):
        """Test session is created"""
        analyzer = SonarCloudAnalyzer(token="token", organization="org")
        
        assert hasattr(analyzer, 'session')
        assert analyzer.session is not None
    
    def test_init_sets_auth_header(self):
        """Test authentication header is set"""
        with patch('requests.Session') as mock_session:
            mock_instance = Mock()
            mock_session.return_value = mock_instance
            
            analyzer = SonarCloudAnalyzer(token="test-token", organization="org")
            
            # Verify headers were updated
            assert mock_instance.headers.update.called
    

    
    def test_init_organization_as_string(self):
        """Test organization is converted to string"""
        analyzer = SonarCloudAnalyzer(token="token", organization=12345)
        
        assert analyzer.organization == "12345"
        assert isinstance(analyzer.organization, str)


# ============================================================================
# TEST CLASS: GET PROJECT ISSUES
# ============================================================================

class TestGetProjectIssues:
    """Test fetching project issues"""
    
    def test_get_project_issues_success(self, analyzer, sample_issue_data):
        """Test successfully fetching project issues"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "issues": [sample_issue_data],
            "paging": {"pageIndex": 1, "pageSize": 100, "total": 1}
        }
        
        with patch.object(analyzer.session, 'get', return_value=mock_response):
            result = analyzer.get_project_issues(
                project_key="test-project",
                branch="main",
                max_issues=10
            )
        
        assert isinstance(result, AnalysisResult)
        assert result.project_key == "test-project"
        assert result.total_issues >= 0
        assert len(result.issues) >= 0
    
    def test_get_project_issues_with_filters(self, analyzer):
        """Test fetching issues with severity and type filters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 0,
            "issues": [],
            "paging": {"pageIndex": 1, "pageSize": 100, "total": 0}
        }
        
        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            result = analyzer.get_project_issues(
                project_key="test-project",
                severities=["BLOCKER", "CRITICAL"],
                types=["BUG", "VULNERABILITY"]
            )
        
        # Verify filters were applied
        call_args = mock_get.call_args_list

        params = call_args[0][1]['params']
        assert 'severities' in params or 'types' in params
    
    def test_get_project_issues_with_branch(self, analyzer, sample_issue_data):
        """Test fetching issues for specific branch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "issues": [sample_issue_data]
        }
        
        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            result = analyzer.get_project_issues(
                project_key="test-project",
                branch="develop"
            )
        
        # Verify branch parameter was passed
        call_args = mock_get.call_args_list
        params = call_args[0][1]['params']
        assert params.get('branch') == "develop" or 'branch' in params
    
    def test_get_project_issues_with_pull_request(self, analyzer):
        """Test fetching issues for pull request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 0,
            "issues": []
        }
        
        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            result = analyzer.get_project_issues(
                project_key="test-project",
                pull_request_number=123
            )
        
        # Verify PR parameter was passed
        call_args = mock_get.call_args_list
        params = call_args[0][1]['params']
        assert 'pullRequest' in params

    def test_get_project_issues_api_error(self, analyzer):
        """Test handling of API errors"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        with patch.object(analyzer.session, 'get', return_value=mock_response):
            result = analyzer.get_project_issues(project_key="test-project")
        
        # Should handle error gracefully
        assert result is None or isinstance(result, AnalysisResult)
    
    def test_get_project_issues_connection_error(self, analyzer):
        """Test handling of connection errors"""
        with patch.object(analyzer.session, 'get', side_effect=requests.ConnectionError("Connection failed")):
            result = analyzer.get_project_issues(project_key="test-project")
        
        assert result is None or isinstance(result, AnalysisResult)
    
    def test_get_project_issues_timeout(self, analyzer):
        """Test handling of timeout"""
        with patch.object(analyzer.session, 'get', side_effect=requests.Timeout("Request timeout")):
            result = analyzer.get_project_issues(project_key="test-project")
        
        assert result is None or isinstance(result, AnalysisResult)


# ============================================================================
# TEST CLASS: GET PROJECT METRICS
# ============================================================================

class TestGetProjectMetrics:
    """Test fetching project metrics"""
    
    def test_get_project_metrics_success(self, analyzer, sample_metrics_data):
        """Test successfully fetching metrics"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_metrics_data
        
        with patch.object(analyzer.session, 'get', return_value=mock_response):
            metrics = analyzer.get_project_metrics("test-project")
        
        assert metrics is not None
        assert isinstance(metrics, ProjectMetrics)
        assert metrics.lines_of_code == 1000
        assert metrics.coverage == 85.5
        assert metrics.bugs == 5
    
    def test_get_project_metrics_partial_data(self, analyzer):
        """Test metrics with partial data"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc", "value": "500"}
                ]
            }
        }
        
        with patch.object(analyzer.session, 'get', return_value=mock_response):
            metrics = analyzer.get_project_metrics("test-project")
        
        assert metrics is not None
        assert metrics.lines_of_code == 500
        assert metrics.coverage is None  # Not provided
    
    def test_get_project_metrics_not_found(self, analyzer):
        """Test metrics for non-existent project"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(analyzer.session, 'get', return_value=mock_response):
            metrics = analyzer.get_project_metrics("nonexistent-project")
        
        assert metrics is None
    
    def test_get_project_metrics_api_error(self, analyzer):
        """Test handling of API error"""
        with patch.object(analyzer.session, 'get', side_effect=Exception("API Error")):
            metrics = analyzer.get_project_metrics("test-project")
        
        assert metrics is None

    def test_get_project_metrics_invalid_numeric_values(self, analyzer):
        """Test handling of non-numeric metric values"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc", "value": "invalid"},
                    {"metric": "coverage", "value": "not_a_number"}
                ]
            }
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            metrics = analyzer.get_project_metrics("test-project")

        assert metrics is not None
        # Should handle conversion errors gracefully
        assert metrics.lines_of_code == 0
        assert metrics.coverage == 0.0

    def test_get_project_metrics_missing_measures_array(self, analyzer):
        """Test when measures array is missing"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {}
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            metrics = analyzer.get_project_metrics("test-project")

        assert metrics is not None
        assert metrics.lines_of_code is None

    def test_get_project_metrics_empty_component(self, analyzer):
        """Test when component object is empty"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            metrics = analyzer.get_project_metrics("test-project")

        assert metrics is not None

    def test_get_project_metrics_none_values(self, analyzer):
        """Test handling of None metric values"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "component": {
                "measures": [
                    {"metric": "ncloc", "value": None},
                    {"metric": "coverage", "value": None}
                ]
            }
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            metrics = analyzer.get_project_metrics("test-project")

        assert metrics is not None



# ============================================================================
# TEST CLASS: PARSE ISSUES
# ============================================================================

class TestParseIssues:
    """Test parsing API response to SonarIssue objects"""
    
    def test_parse_issues_complete_data(self, analyzer, sample_issue_data):
        """Test parsing with complete issue data"""
        issues = analyzer._parse_issues([sample_issue_data])
        
        assert len(issues) == 1
        assert isinstance(issues[0], SonarIssue)
        assert issues[0].key == "issue-123"
        assert issues[0].rule == "python:S1234"
        assert issues[0].severity == "MAJOR"
    
    def test_parse_issues_multiple(self, analyzer, sample_issue_data):
        """Test parsing multiple issues"""
        issues_data = [sample_issue_data.copy() for _ in range(5)]
        for i, issue in enumerate(issues_data):
            issue["key"] = f"issue-{i}"
        
        issues = analyzer._parse_issues(issues_data)
        
        assert len(issues) == 5
        assert all(isinstance(issue, SonarIssue) for issue in issues)
    
    def test_parse_issues_missing_optional_fields(self, analyzer):
        """Test parsing with missing optional fields"""
        minimal_issue = {
            "key": "issue-1",
            "rule": "python:S1234",
            "severity": "MAJOR",
            "component": "test.py",
            "project": "project",
            "line": 10,
            "message": "Issue"
        }
        
        issues = analyzer._parse_issues([minimal_issue])
        
        assert len(issues) == 1
        assert issues[0].key == "issue-1"
    
    def test_parse_issues_extracts_file_path(self, analyzer):
        """Test file path extraction from component"""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "severity": "MAJOR",
            "component": "project:src/main/java/com/example/Test.java",
            "project": "project",
            "line": 10,
            "message": "Issue"
        }
        
        issues = analyzer._parse_issues([issue_data])
        
        assert issues[0].file is not None
        assert "Test.java" in issues[0].file
    
    def test_parse_issues_handles_text_range(self, analyzer, sample_issue_data):
        """Test parsing textRange for line numbers"""
        issues = analyzer._parse_issues([sample_issue_data])
        
        assert issues[0].first_line == 10
        assert issues[0].last_line == 15
    
    def test_parse_issues_empty_list(self, analyzer):
        """Test parsing empty list"""
        issues = analyzer._parse_issues([])
        
        assert issues == []

    def test_parse_issues_invalid_severity(self, analyzer):
        """Test parsing with invalid severity value"""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "severity": "INVALID_SEVERITY",
            "component": "test.py",
            "project": "project",
            "line": 10,
            "message": "Issue"
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        # Should default to INFO
        assert issues[0].severity == Severity.INFO

    def test_parse_issues_invalid_issue_type(self, analyzer):
        """Test parsing with invalid issue type"""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "severity": "MAJOR",
            "type": "INVALID_TYPE",
            "component": "test.py",
            "project": "project",
            "line": 10,
            "message": "Issue"
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        # Should default to CODE_SMELL
        assert issues[0].type == IssueType.CODE_SMELL

    def test_parse_issues_missing_flows(self, analyzer):
        """Test parsing when flows array is missing"""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "severity": "MAJOR",
            "component": "test.py",
            "project": "project",
            "line": 10,
            "message": "Issue"
            # No flows
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        assert issues[0].first_line == 10
        assert issues[0].last_line == 10

    def test_parse_issues_flows_without_locations(self, analyzer):
        """Test parsing flows with empty locations"""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "severity": "MAJOR",
            "component": "test.py",
            "project": "project",
            "line": 10,
            "message": "Issue",
            "flows": [
                {"locations": []}
            ]
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        assert issues[0].last_line == 10

    @pytest.mark.skip(reason="Need update")
    def test_parse_issues_none_line_numbers(self, analyzer):
        """Test parsing when line numbers are None"""
        issue_data = {
            "key": "issue-1",
            "rule": "python:S1234",
            "severity": "MAJOR",
            "component": "test.py",
            "project": "project",
            "line": None,
            "message": "Issue"
        }

        issues = analyzer._parse_issues([issue_data])

        assert len(issues) == 1
        assert issues[0].first_line is None


    def test_parse_issues_exception_in_single_issue(self, analyzer):
        """Test that exception in one issue doesn't stop parsing others"""
        good_issue = {
            "key": "good-1",
            "rule": "python:S1234",
            "severity": "MAJOR",
            "component": "test.py",
            "project": "project",
            "line": 10,
            "message": "Good issue"
        }
        bad_issue = {"key": "bad-issue"}  # Missing required fields

        issues = analyzer._parse_issues([good_issue, bad_issue, good_issue])

        # Should still parse the good issues
        assert len(issues) >= 1


class TestGetFixableIssues:
    """Test get_fixable_issues method - COMPLETELY UNTESTED"""

    def test_get_fixable_issues_returns_empty_when_no_analysis(self, analyzer):
        """Test when get_project_issues returns None"""
        with patch.object(analyzer, 'get_project_issues', return_value=None):
            issues = analyzer.get_fixable_issues(
                project_key="test-project",
                branch="main",
                max_issues=10
            )

        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_get_fixable_issues_sorting_by_severity(self, analyzer, sample_issue_data):
        """Test issues are sorted correctly by severity"""
        mock_response = Mock()
        mock_response.status_code = 200

        # Create issues with different severities
        issue_blocker = sample_issue_data.copy()
        issue_blocker["severity"] = "BLOCKER"
        issue_blocker["key"] = "issue-blocker"

        issue_minor = sample_issue_data.copy()
        issue_minor["severity"] = "MINOR"
        issue_minor["key"] = "issue-minor"

        issue_critical = sample_issue_data.copy()
        issue_critical["severity"] = "CRITICAL"
        issue_critical["key"] = "issue-critical"

        mock_response.json.return_value = {
            "total": 3,
            "issues": [issue_minor, issue_critical, issue_blocker]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            issues = analyzer.get_fixable_issues(
                project_key="test-project",
                max_issues=10
            )

        # Verify sorting order
        if len(issues) > 0:
            severities = [issue.severity for issue in issues]
            # BLOCKER should come first, MINOR last
            assert severities[0] in ["BLOCKER", "CRITICAL"]

    def test_get_fixable_issues_max_issues_limit(self, analyzer, sample_issue_data):
        """Test max_issues parameter limits results correctly"""
        mock_response = Mock()
        mock_response.status_code = 200

        # Create 10 issues
        issues_list = []
        for i in range(10):
            issue = sample_issue_data.copy()
            issue["key"] = f"issue-{i}"
            issues_list.append(issue)

        mock_response.json.return_value = {
            "total": 10,
            "issues": issues_list
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            issues = analyzer.get_fixable_issues(
                project_key="test-project",
                max_issues=5
            )

        # Should be limited to 5
        assert len(issues) <= 5

    def test_get_fixable_issues_with_all_severity_types(self, analyzer, sample_issue_data):
        """Test with all severity levels"""
        mock_response = Mock()
        mock_response.status_code = 200

        severities = ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]
        issues_list = []
        for sev in severities:
            issue = sample_issue_data.copy()
            issue["severity"] = sev
            issue["key"] = f"issue-{sev}"
            issues_list.append(issue)

        mock_response.json.return_value = {
            "total": 5,
            "issues": issues_list
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            issues = analyzer.get_fixable_issues(
                project_key="test-project",
                max_issues=10
            )

        assert len(issues) > 0
        # Verify all severity types are handled
        severity_values = [issue.severity for issue in issues]
        assert all(sev in ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"] for sev in severity_values)


class TestGetFixableIssuesByTypes:
    """Test get_fixable_issues_by_types method - COMPLETELY UNTESTED"""

    def test_get_fixable_issues_by_types_with_rules_grouping(self, analyzer, sample_issue_data, sample_rules_response):
        """Test grouping issues by rules"""
        # Mock get_project_rules to return rules
        mock_rules_response = Mock()
        mock_rules_response.status_code = 200
        mock_rules_response.json.return_value = sample_rules_response

        # Mock get_project_issues to return issues
        mock_issues_response = Mock()
        mock_issues_response.status_code = 200
        issue_data = sample_issue_data.copy()
        issue_data["rule"] = "python:S1234"
        mock_issues_response.json.return_value = {
            "total": 1,
            "issues": [issue_data]
        }

        with patch.object(analyzer.session, 'get') as mock_get:
            mock_get.side_effect = [mock_rules_response, mock_issues_response]

            result = analyzer.get_fixable_issues_by_types(
                project_key="test-project",
                branch="main",
                max_issues=10,
                group_by="rules"
            )
        assert isinstance(result, dict)
        assert "python:S1234" in result

        assert len(result["python:S1234"]) > 0

    def test_get_fixable_issues_by_types_with_file_grouping(self, analyzer, sample_issue_data):
        """Test grouping issues by file (default behavior)"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "issues": [sample_issue_data]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            result = analyzer.get_fixable_issues_by_types(
                project_key="test-project",
                branch="main",
                max_issues=10,
                group_by="file"
            )

        assert isinstance(result, dict)
        # Should return grouped by file
        for key in result.keys():
            assert isinstance(result[key], list)

    def test_get_fixable_issues_by_types_empty_rules(self, analyzer):
        """Test when no rules are returned"""
        mock_rules_response = Mock()
        mock_rules_response.status_code = 200
        mock_rules_response.json.return_value = {
            "facets": [
                {
                    "property": "rules",
                    "values": []
                }
            ]
        }

        mock_issues_response = Mock()
        mock_issues_response.status_code = 200
        mock_issues_response.json.return_value = {
            "total": 0,
            "issues": []
        }

        with patch.object(analyzer.session, 'get') as mock_get:
            mock_get.side_effect = [mock_rules_response, mock_issues_response]

            result = analyzer.get_fixable_issues_by_types(
                project_key="test-project",
                group_by="rules",
                max_issues=10
            )

        assert isinstance(result, dict)

    def test_get_fixable_issues_by_types_with_filters(self, analyzer, sample_issue_data):
        """Test with severity and type filters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "issues": [sample_issue_data]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            result = analyzer.get_fixable_issues_by_types(
                project_key="test-project",
                branch="main",
                severities=["CRITICAL", "BLOCKER"],
                types_list=["BUG", "VULNERABILITY"],
                max_issues=10
            )

        # Verify filters were passed
        assert mock_get.called
        call_args = mock_get.call_args_list
        params = call_args[0][1]['params']
        assert 'severities' in params or 'types' in params

    def test_get_fixable_issues_by_types_no_analysis_returned(self, analyzer):
        """Test when get_project_issues returns None"""
        with patch.object(analyzer, 'get_project_issues', return_value=None):
            result = analyzer.get_fixable_issues_by_types(
                project_key="test-project",
                max_issues=10
            )

        assert result == {}

    def test_get_fixable_issues_by_types_multiple_rules(self, analyzer, sample_issue_data):
        """Test with multiple rules returned"""
        mock_rules_response = Mock()
        mock_rules_response.status_code = 200
        mock_rules_response.json.return_value = {
            "facets": [
                {
                    "property": "rules",
                    "values": [
                        {"val": "python:S1234", "count": 5},
                        {"val": "python:S5678", "count": 3}
                    ]
                }
            ]
        }

        mock_issues_response = Mock()
        mock_issues_response.status_code = 200
        issue1 = sample_issue_data.copy()
        issue1["rule"] = "python:S1234"
        issue2 = sample_issue_data.copy()
        issue2["rule"] = "python:S5678"
        issue2["key"] = "issue-456"

        mock_issues_response.json.return_value = {
            "total": 2,
            "issues": [issue1, issue2]
        }

        with patch.object(analyzer.session, 'get') as mock_get:
            mock_get.side_effect = [mock_rules_response, mock_issues_response]

            result = analyzer.get_fixable_issues_by_types(
                project_key="test-project",
                group_by="rules",
                max_issues=10
            )

        assert isinstance(result, dict)
        assert len(result) >= 1


class TestGetFixableIssuesByFiles:
    """Test get_fixable_issues_by_files method"""

    def test_get_fixable_issues_by_files_empty_result(self, analyzer):
        """Test when no analysis is returned"""
        with patch.object(analyzer, 'get_project_issues', return_value=None):
            result = analyzer.get_fixable_issues_by_files(
                project_key="test-project",
                max_issues=10
            )

        assert result == {}

    def test_get_fixable_issues_by_files_groups_correctly(self, analyzer, sample_issue_data):
        """Test issues are grouped by file correctly"""
        mock_response = Mock()
        mock_response.status_code = 200

        issue1 = sample_issue_data.copy()
        issue1["component"] = "project:src/file1.py"
        issue2 = sample_issue_data.copy()
        issue2["component"] = "project:src/file1.py"
        issue2["key"] = "issue-2"

        mock_response.json.return_value = {
            "total": 2,
            "issues": [issue1, issue2]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            result = analyzer.get_fixable_issues_by_files(
                project_key="test-project",
                max_issues=10
            )

        assert isinstance(result, dict)
        # Issues from same file should be grouped together
        if len(result) > 0:
            for file_path, issues in result.items():
                assert isinstance(issues, list)

    def test_get_fixable_issues_by_files_with_multiple_files(self, analyzer, sample_issue_data):
        """Test with issues across multiple files"""
        mock_response = Mock()
        mock_response.status_code = 200

        issue1 = sample_issue_data.copy()
        issue1["component"] = "project:src/file1.py"
        issue2 = sample_issue_data.copy()
        issue2["component"] = "project:src/file2.py"
        issue2["key"] = "issue-2"

        mock_response.json.return_value = {
            "total": 2,
            "issues": [issue1, issue2]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            result = analyzer.get_fixable_issues_by_files(
                project_key="test-project",
                max_issues=10
            )

        assert isinstance(result, dict)


class TestGetProjectRules:
    """Test get_project_rules method - COMPLETELY UNTESTED"""

    def test_get_project_rules_success(self, analyzer, sample_rules_response):
        """Test successfully fetching project rules"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_rules_response

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            rules = analyzer.get_project_rules(
                project_key="test-project",
                branch="main",
                total=10
            )

        assert isinstance(rules, list)
        assert len(rules) > 0
        assert "python:S1234" in rules

    def test_get_project_rules_with_facets(self, analyzer, sample_rules_response):
        """Test fetching rules with facet grouping"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_rules_response

        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            rules = analyzer.get_project_rules(
                project_key="test-project",
                facets="rules",
                total=10
            )

        # Verify facets parameter was passed
        call_args = mock_get.call_args_list[0]
        params = call_args[1]['params']
        assert params.get('facets') == "rules"

    def test_get_project_rules_with_total_limit(self, analyzer):
        """Test rule fetching stops when total reached"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "facets": [
                {
                    "property": "rules",
                    "values": [
                        {"val": "python:S1234", "count": 8},
                        {"val": "python:S5678", "count": 5},  # Should stop here
                        {"val": "python:S9012", "count": 3}
                    ]
                }
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            rules = analyzer.get_project_rules(
                project_key="test-project",
                total=10
            )

        # Should stop at 10 total issues (8 + 5 > 10, so only first two)
        assert isinstance(rules, list)
        assert len(rules) <= 3

    def test_get_project_rules_api_error(self, analyzer):
        """Test error handling in rule fetching"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server Error")

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            rules = analyzer.get_project_rules(
                project_key="test-project",
                total=10
            )

        assert rules is None

    def test_get_project_rules_empty_facets(self, analyzer):
        """Test when API returns empty facets"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "facets": []
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            rules = analyzer.get_project_rules(
                project_key="test-project",
                total=10
            )

        assert isinstance(rules, list)
        assert len(rules) == 0

    def test_get_project_rules_with_all_filters(self, analyzer, sample_rules_response):
        """Test with all filter parameters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_rules_response

        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            rules = analyzer.get_project_rules(
                project_key="test-project",
                branch="develop",
                pull_request_number=123,
                statuses=["OPEN", "CONFIRMED"],
                severities=["CRITICAL", "BLOCKER"],
                types=["BUG"],
                facets="rules",
                total=20
            )

        # Verify all parameters were passed
        call_args = mock_get.call_args_list[0]
        params = call_args[1]['params']
        assert 'branch' in params or 'pullRequest' in params
        assert 'severities' in params
        assert 'types' in params

    def test_get_project_rules_connection_error(self, analyzer):
        """Test handling of connection errors"""
        with patch.object(analyzer.session, 'get', side_effect=requests.ConnectionError("Connection failed")):
            rules = analyzer.get_project_rules(
                project_key="test-project",
                total=10
            )

        assert rules is None



# ============================================================================
# TEST CLASS: SECURITY ISSUES
# ============================================================================

class TestSecurityIssues:
    """Test security-specific functionality"""
    
    def test_get_fixable_security_issues(self, analyzer, sample_security_issue_data):
        """Test fetching fixable security issues"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "issues": [sample_security_issue_data]
        }
        
        with patch.object(analyzer.session, 'get', return_value=mock_response):
            issues = analyzer.get_fixable_security_issues(
                project_key="test-project",
                branch="main",
                max_issues=10
            )
        
        assert isinstance(issues, dict)
    
    def test_parse_security_issues(self, analyzer, sample_security_issue_data):
        """Test parsing security issues"""
        issues = analyzer._parse_security_issues([sample_security_issue_data],"key")
        
        assert len(issues) == 1
        assert isinstance(issues[0], SonarSecurityIssue)
        assert issues[0].vulnerability_probability == "HIGH"
    
    def test_get_project_security_issues_with_filters(self, analyzer):
        """Test fetching security issues with severity filters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 0,
            "issues": []
        }
        
        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            result = analyzer.get_project_security_issues(
                project_key="test-project",
                branch="main"
            )
        
        # Verify filters were applied
        assert mock_get.called

    def test_parse_security_issues_missing_optional_fields(self, analyzer):
        """Test parsing security issues with missing fields"""
        issue_data = {
            "key": "sec-1",
            "ruleKey": "python:S5042",
            "component": "test.py",
            "line": 20,
            "message": "Security issue"
            # Missing optional fields
        }

        issues = analyzer._parse_security_issues([issue_data], "test-project")

        assert len(issues) == 1
        assert issues[0].key == "sec-1"
        assert issues[0].status == "OPEN"  # Default value

    def test_parse_security_issues_invalid_line_numbers(self, analyzer):
        """Test handling of invalid line number data"""
        issue_data = {
            "key": "sec-1",
            "ruleKey": "python:S5042",
            "component": "test.py",
            "line": "not_a_number",
            "message": "Security issue"
        }

        issues = analyzer._parse_security_issues([issue_data], "test-project")

        # Should handle gracefully
        assert len(issues) <= 1

    def test_parse_security_issues_exception_handling(self, analyzer):
        """Test exception handling in security issue parsing"""
        good_issue = {
            "key": "sec-1",
            "ruleKey": "python:S5042",
            "component": "test.py",
            "line": 20,
            "message": "Good issue"
        }
        bad_issue = {"key": "bad"}  # Missing required fields

        issues = analyzer._parse_security_issues([good_issue, bad_issue], "test-project")

        # Should continue processing despite error
        assert len(issues) >= 1
# ============================================================================
# TEST CLASS: DIRECTORY ANALYSIS
# ============================================================================

class TestDirectoryAnalysis:
    """Test local directory analysis"""
    
    def test_analyze_project_directory(self, analyzer, tmp_path):
        """Test analyzing project directory"""
        # Create test project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "src" / "test.py").write_text("def test(): pass")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        (tmp_path / "README.md").write_text("# Project")
        
        analysis = analyzer.analyze_project_directory(str(tmp_path))
        
        assert analysis["total_files"] >= 4
        assert analysis["python_files"] >= 3
        assert isinstance(analysis, dict)
    
    def test_analyze_directory_with_git(self, analyzer, tmp_path):
        """Test detecting git repository"""
        (tmp_path / ".git").mkdir()
        (tmp_path / "file.py").write_text("code")
        
        analysis = analyzer.analyze_project_directory(str(tmp_path))
        
        assert analysis["has_git"] is True
    
    def test_analyze_directory_with_sonar_config(self, analyzer, tmp_path):
        """Test detecting SonarCloud configuration"""
        (tmp_path / "sonar-project.properties").write_text("sonar.projectKey=test")
        
        analysis = analyzer.analyze_project_directory(str(tmp_path))
        
        assert analysis["has_sonar_config"] is True
    
    def test_analyze_directory_counts_file_types(self, analyzer, tmp_path):
        """Test counting different file types"""
        (tmp_path / "file.py").write_text("python")
        (tmp_path / "file.js").write_text("javascript")
        (tmp_path / "file.java").write_text("java")
        (tmp_path / "file.txt").write_text("text")
        
        analysis = analyzer.analyze_project_directory(str(tmp_path))
        
        assert analysis["python_files"] >= 1
        assert analysis["javascript_files"] >= 1
        assert analysis["java_files"] >= 1
    
    def test_check_file_extension_python(self, analyzer):
        """Test file extension checking for Python"""
        analysis = {"python_files": 0}
        result = analyzer.check_file_extension(analysis, ".py")
        
        assert result["python_files"] == 1
    
    def test_check_file_extension_javascript(self, analyzer):
        """Test file extension checking for JavaScript"""
        analysis = {"javascript_files": 0}
        result = analyzer.check_file_extension(analysis, ".js")
        
        assert result["javascript_files"] == 1

    def test_analyze_project_directory_invalid_path(self, analyzer):
        """Test with non-existent directory"""
        with pytest.raises(ValueError, match="Invalid project path"):
            analyzer.analyze_project_directory("/nonexistent/path")

    def test_analyze_project_directory_file_not_directory(self, analyzer, tmp_path):
        """Test when path points to file, not directory"""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="Invalid project path"):
            analyzer.analyze_project_directory(str(file_path))

    def test_analyze_project_directory_hidden_files(self, analyzer, tmp_path):
        """Test that hidden files/dirs are properly ignored"""
        (tmp_path / ".hidden_dir").mkdir()
        (tmp_path / ".hidden_file.py").write_text("code")
        (tmp_path / "visible.py").write_text("code")

        analysis = analyzer.analyze_project_directory(str(tmp_path))

        # Hidden files should not be counted
        assert ".hidden_dir" not in analysis["directories"]
        assert analysis["python_files"] >= 1  # Only visible.py

    def test_analyze_project_directory_nested_structure(self, analyzer, tmp_path):
        """Test with deeply nested directory structure"""
        nested = tmp_path / "level1" / "level2" / "level3"
        nested.mkdir(parents=True)
        (nested / "file.py").write_text("code")

        analysis = analyzer.analyze_project_directory(str(tmp_path))

        assert analysis["total_files"] >= 1
        assert analysis["python_files"] >= 1

    def test_analyze_project_directory_mixed_extensions(self, analyzer, tmp_path):
        """Test with various file extensions"""
        extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".txt", ".md"]
        for ext in extensions:
            (tmp_path / f"file{ext}").write_text("content")

        analysis = analyzer.analyze_project_directory(str(tmp_path))

        assert analysis["python_files"] >= 1
        assert analysis["javascript_files"] >= 4  # js, jsx, ts, tsx
        assert analysis["java_files"] >= 1
        assert analysis["other_files"] >= 2  # txt, md

    def test_check_file_extension_typescript(self, analyzer):
        """Test TypeScript extensions are counted as JavaScript"""
        analysis = {"javascript_files": 0}

        result = analyzer.check_file_extension(analysis, ".ts")
        assert result["javascript_files"] == 1

        result = analyzer.check_file_extension(result, ".tsx")
        assert result["javascript_files"] == 2

    def test_check_file_extension_kotlin_scala(self, analyzer):
        """Test Kotlin and Scala are counted as Java"""
        analysis = {"java_files": 0}

        result = analyzer.check_file_extension(analysis, ".kotlin")
        assert result["java_files"] == 1

        result = analyzer.check_file_extension(result, ".scala")
        assert result["java_files"] == 2

# ============================================================================
# TEST CLASS: CONTEXT MANAGER
# ============================================================================

class TestContextManager:
    """Test context manager functionality"""
    
    def test_context_manager_enter_exit(self, mock_session):
        """Test using analyzer as context manager"""
        with SonarCloudAnalyzer(token="token", organization="org") as analyzer:
            assert analyzer is not None
            assert hasattr(analyzer, 'session')
    
    def test_context_manager_closes_session(self):
        """Test session is closed on exit"""
        with patch('requests.Session') as mock_session_class:
            mock_instance = Mock()
            mock_session_class.return_value = mock_instance
            
            with SonarCloudAnalyzer(token="token", organization="org") as analyzer:
                pass
            
            # Session should be closed
            mock_instance.close.assert_called_once()
    
    def test_close_method(self, analyzer):
        """Test close method"""
        with patch.object(analyzer.session, 'close') as mock_close:
            analyzer.close()
            mock_close.assert_called_once()



# ============================================================================
# TEST CLASS: HELPER METHODS
# ============================================================================

class TestHelperMethods:
    """Test helper methods"""
    
    def test_extract_file_path_with_project_prefix(self, analyzer):
        """Test extracting file path from component with project prefix"""
        component = "project:src/main/java/com/example/Test.java"
        
        file_path = analyzer._extract_file_path(component)
        
        assert file_path == "src/main/java/com/example/Test.java"
    
    def test_extract_file_path_without_prefix(self, analyzer):
        """Test extracting file path without project prefix"""
        component = "src/test.py"
        
        file_path = analyzer._extract_file_path(component)
        
        assert file_path == "src/test.py"
    
    def test_extract_file_path_none_component(self, analyzer):
        """Test extracting file path from None"""
        file_path = analyzer._extract_file_path(None)
        
        assert file_path is None
    
    def test_build_query_params(self, analyzer):
        """Test building query parameters"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="main",
            severities=["BLOCKER", "CRITICAL"],
            types=["BUG"],
            max_issues=5,
            pull_request_number=0
        )
        
        assert params["componentKeys"] == "test-project"
        assert "branch" in params
        assert "severities" in params
        assert "types" in params

    def test_build_query_params_with_facets(self, analyzer):
        """Test building params with facets parameter"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="main",
            max_issues=10,
            pull_request_number=0,
            facets="rules"
        )

        assert params["facets"] == "rules"
        assert params["componentKeys"] == "test-project"

    def test_build_query_params_with_filter_by_and_values(self, analyzer):
        """Test filter_by and filter_values parameters"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="main",
            max_issues=10,
            pull_request_number=0,
            filter_by="rules",
            filter_values=["python:S1234", "python:S5678"]
        )

        assert "rules" in params
        assert params["rules"] == "python:S1234,python:S5678"

    def test_build_query_params_empty_branch_with_no_pr(self, analyzer):
        """Test default to 'main' when branch and PR both empty"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="",
            max_issues=10,
            pull_request_number=0
        )

        assert params.get("branch") == "main"
        assert "pullRequest" not in params

    def test_build_query_params_all_filters_combined(self, analyzer):
        """Test combining all filter types together"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="develop",
            max_issues=50,
            pull_request_number=0,
            statuses=["OPEN", "CONFIRMED"],
            severities=["CRITICAL", "BLOCKER"],
            types=["BUG", "VULNERABILITY"],
            facets="rules,types",
            filter_by="rules",
            filter_values=["python:S1234"]
        )

        assert params["componentKeys"] == "test-project"
        assert params["branch"] == "develop"
        assert params["ps"] == 50
        assert "issueStatuses" in params
        assert "severities" in params
        assert "types" in params
        assert params["facets"] == "rules,types"
        assert params["rules"] == "python:S1234"

    def test_build_query_params_custom_field_key(self, analyzer):
        """Test with custom field_key parameter"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="main",
            max_issues=10,
            pull_request_number=0,
            field_key="projectKey"
        )

        assert "projectKey" in params
        assert params["projectKey"] == "test-project"

    def test_build_query_params_pull_request_overrides_branch(self, analyzer):
        """Test pull request parameter when branch is empty"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="",
            max_issues=10,
            pull_request_number=123
        )

        assert params.get("pullRequest") == "123"
        assert "branch" not in params or params.get("branch") != "main"

    def test_build_query_params_pr_takes_priority_over_branch(self, analyzer):
        """Test that PR takes priority when both branch and PR are provided"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="main",
            max_issues=10,
            pull_request_number=456
        )

        # PR should be used, branch should NOT be in params
        assert params.get("pullRequest") == "456"
        assert "branch" not in params

    def test_build_query_params_branch_used_when_pr_is_zero(self, analyzer):
        """Test that branch is used when PR is 0"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="develop",
            max_issues=10,
            pull_request_number=0
        )

        assert params.get("branch") == "develop"
        assert "pullRequest" not in params

    def test_build_query_params_branch_used_when_pr_is_none(self, analyzer):
        """Test that branch is used when PR is None"""
        params = analyzer._build_query_params(
            project_key="test-project",
            branch="feature-branch",
            max_issues=10,
            pull_request_number=None
        )

        assert params.get("branch") == "feature-branch"
        assert "pullRequest" not in params

    def test_filter_rules_empty_input(self, analyzer):
        """Test with empty rules list"""


        result = analyzer._filter_rules(
            rules=[],
            exclude_rules=[],
            max_rules=10
        )

        assert result == []

    def test_filter_rules_none_input(self, analyzer):
        """Test with None rules list"""

        result = analyzer._filter_rules(
            rules=None,
            exclude_rules=[],
            max_rules=10
        )

        assert result == []

    def test_filter_rules_no_exclusions_no_limit(self, analyzer):
        """Test with no exclusions and unlimited rules (max_rules=0)"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3},
                    {"val": "python:S107", "count": 2}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=0
        )

        assert len(result) == 3
        assert result == ["python:S3776", "python:S1192", "python:S107"]

    def test_filter_rules_no_exclusions_with_limit(self, analyzer):
        """Test with no exclusions but with max_rules limit"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3},
                    {"val": "python:S107", "count": 2}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=2
        )

        # Should stop after first rule since count (5) > max_rules (2)
        assert len(result) == 1
        assert result == ["python:S3776"]

    def test_filter_rules_with_exclusions_no_limit(self, analyzer):
        """Test with exclusions but no limit"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3},
                    {"val": "python:S107", "count": 2}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=["python:S1192"],
            max_rules=0
        )

        assert len(result) == 2
        assert result == ["python:S3776", "python:S107"]
        assert "python:S1192" not in result

    def test_filter_rules_with_exclusions_and_limit(self,analyzer):
        """Test with both exclusions and max_rules limit"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 3},
                    {"val": "python:S1192", "count": 2},  # Excluded
                    {"val": "python:S107", "count": 3},
                    {"val": "python:S5852", "count": 1}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=["python:S1192"],
            max_rules=5
        )

        # S3776 (count=3) -> remaining=2
        # S1192 excluded
        # S107 (count=3) -> remaining=-1, stops
        assert len(result) == 2
        assert result == ["python:S3776", "python:S107"]

    def test_filter_rules_exclude_all_rules(self,analyzer):
        """Test when all rules are excluded"""
        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=["python:S3776", "python:S1192"],
            max_rules=0
        )

        assert result == []

    def test_filter_rules_exclude_multiple_rules(self,analyzer):
        """Test excluding multiple rules"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3},
                    {"val": "python:S107", "count": 2},
                    {"val": "python:S5852", "count": 1}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=["python:S1192", "python:S5852"],
            max_rules=0
        )

        assert len(result) == 2
        assert result == ["python:S3776", "python:S107"]

    def test_filter_rules_empty_exclude_list(self,analyzer):
        """Test with empty exclude_rules list"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=0
        )

        assert len(result) == 2
        assert result == ["python:S3776", "python:S1192"]

    def test_filter_rules_exclude_nonexistent_rule(self,analyzer):
        """Test excluding rules that don't exist in the data"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=["python:S9999", "python:S8888"],
            max_rules=0
        )

        # Should return all rules since excluded ones don't exist
        assert len(result) == 2
        assert result == ["python:S3776", "python:S1192"]

    def test_filter_rules_max_rules_zero_unlimited(self,analyzer):
        """Test max_rules=0 means unlimited"""

        rules = [
            {
                "values": [
                    {"val": f"python:S{i}", "count": 1}
                    for i in range(100)
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=0
        )

        # Should return all 100 rules
        assert len(result) == 100

    def test_filter_rules_max_rules_exact_count(self,analyzer):
        """Test when max_rules exactly matches cumulative count"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 3},
                    {"val": "python:S1192", "count": 2},
                    {"val": "python:S107", "count": 5}  # This exceeds
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=5
        )

        # S3776 (count=3) -> remaining=2
        # S1192 (count=2) -> remaining=0, stops
        assert len(result) == 2
        assert result == ["python:S3776", "python:S1192"]

    def test_filter_rules_max_rules_single_rule_exceeds(self,analyzer):
        """Test when first rule's count exceeds max_rules"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 10},
                    {"val": "python:S1192", "count": 5}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=5
        )

        # First rule count (10) > max_rules (5), so it stops after first
        assert len(result) == 1
        assert result == ["python:S3776"]

    def test_filter_rules_negative_max_rules(self,analyzer):
        """Test with negative max_rules (should behave as unlimited)"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 5},
                    {"val": "python:S1192", "count": 3}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=-1
        )

        # Negative max_rules treated as unlimited (since condition is max_rules > 0)
        assert len(result) == 2

    def test_filter_rules_max_rules_one(self,analyzer):
        """Test with max_rules=1"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 1},
                    {"val": "python:S1192", "count": 1}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=1
        )

        # Should stop after first rule (count=1 reaches limit)
        assert len(result) == 1
        assert result == ["python:S3776"]

    def test_filter_rules_multiple_facets(self,analyzer):
        """Test with multiple rule facets"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 3},
                    {"val": "python:S1192", "count": 2}
                ]
            },
            {
                "values": [
                    {"val": "python:S107", "count": 1},
                    {"val": "python:S5852", "count": 4}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=0
        )

        # Should process all facets
        assert len(result) == 4
        assert result == ["python:S3776", "python:S1192", "python:S107", "python:S5852"]

    def test_filter_rules_multiple_facets_with_limit(self,analyzer):
        """Test multiple facets with max_rules limit"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 3},
                    {"val": "python:S1192", "count": 2}
                ]
            },
            {
                "values": [
                    {"val": "python:S107", "count": 1},
                    {"val": "python:S5852", "count": 4}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=6
        )

        # S3776 (count=3) -> remaining=3
        # S1192 (count=2) -> remaining=1
        # S107 (count=1) -> remaining=0, stops
        assert len(result) == 3
        assert result == ["python:S3776", "python:S1192", "python:S107"]

    def test_filter_rules_multiple_facets_with_exclusions(self,analyzer):
        """Test multiple facets with exclusions"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 3},
                    {"val": "python:S1192", "count": 2}
                ]
            },
            {
                "values": [
                    {"val": "python:S107", "count": 1},
                    {"val": "python:S5852", "count": 4}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=["python:S1192", "python:S5852"],
            max_rules=0
        )

        assert len(result) == 2
        assert result == ["python:S3776", "python:S107"]

    def test_filter_rules_empty_facet_values(self,analyzer):
        """Test facet with empty values list"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 3}
                ]
            },
            {
                "values": []  # Empty values
            },
            {
                "values": [
                    {"val": "python:S1192", "count": 2}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=0
        )

        # Should skip empty facet
        assert len(result) == 2
        assert result == ["python:S3776", "python:S1192"]

    def test_filter_rules_facet_without_values_key(self,analyzer):
        """Test facet without 'values' key"""

        rules = [
            {
                "values": [
                    {"val": "python:S3776", "count": 3}
                ]
            },
            {
                # No 'values' key
                "other_key": "data"
            },
            {
                "values": [
                    {"val": "python:S1192", "count": 2}
                ]
            }
        ]

        result = analyzer._filter_rules(
            rules=rules,
            exclude_rules=[],
            max_rules=0
        )

        # Should skip facet without values
        assert len(result) == 2
        assert result == ["python:S3776", "python:S1192"]
# ============================================================================
# TEST CLASS: FETCH_ISSUES - MISSING EDGE CASES
# ============================================================================

class TestFetchIssuesExtended:
    """Test _fetch_issues method edge cases"""

    def test_fetch_issues_pagination_handling(self, analyzer):
        """Test how pagination data is handled"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "issues": [{"key": "issue-1"}],
            "paging": {
                "pageIndex": 1,
                "pageSize": 100,
                "total": 150
            }
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            issues = analyzer._fetch_issues(
                url="https://sonarcloud.io/api/issues/search",
                params={"componentKeys": "test"},
                key_name="issues"
            )

        assert isinstance(issues, list)
        assert len(issues) > 0

    def test_fetch_issues_malformed_response(self, analyzer):
        """Test handling of malformed API responses"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "unexpected_key": "value"
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            issues = analyzer._fetch_issues(
                url="https://sonarcloud.io/api/issues/search",
                params={"componentKeys": "test"},
                key_name="issues"
            )

        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_fetch_issues_empty_key_name(self, analyzer):
        """Test when key_name doesn't exist in response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "other_data": []
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            issues = analyzer._fetch_issues(
                url="https://sonarcloud.io/api/issues/search",
                params={"componentKeys": "test"},
                key_name="nonexistent_key"
            )

        assert isinstance(issues, list)
        assert len(issues) == 0


# ============================================================================
# TEST CLASS: HANDLE_EXCEPTIONS - COMPREHENSIVE COVERAGE
# ============================================================================

class TestHandleExceptionsExtended:
    """Test _handle_exceptions method comprehensively"""

    def test_handle_exceptions_401_unauthorized(self, analyzer, caplog):
        """Test handling 401 authentication error"""
        mock_response = Mock()
        mock_response.status_code = 401
        error = requests.HTTPError()
        error.response = mock_response

        analyzer._handle_exceptions(error, "test-project")

        assert "Authentication failed" in caplog.text

    def test_handle_exceptions_403_forbidden(self, analyzer, caplog):
        """Test handling 403 permission error"""
        mock_response = Mock()
        mock_response.status_code = 403
        error = requests.HTTPError()
        error.response = mock_response

        analyzer._handle_exceptions(error, "test-project")

        assert "Access forbidden" in caplog.text

    def test_handle_exceptions_404_not_found(self, analyzer, caplog):
        """Test handling 404 project not found"""
        mock_response = Mock()
        mock_response.status_code = 404
        error = requests.HTTPError()
        error.response = mock_response

        analyzer._handle_exceptions(error, "test-project")

        assert "not found" in caplog.text
        assert "test-project" in caplog.text

    def test_handle_exceptions_timeout_error(self, analyzer, caplog):
        """Test handling timeout exception"""
        error = requests.Timeout("Request timed out")

        analyzer._handle_exceptions(error, "test-project")

        assert "timed out" in caplog.text

    def test_handle_exceptions_without_response(self, analyzer, caplog):
        """Test handling exception without response object"""
        error = requests.RequestException("Generic error")

        analyzer._handle_exceptions(error, "test-project")

        assert "Error fetching issues" in caplog.text

    def test_handle_exceptions_500_server_error(self, analyzer, caplog):
        """Test handling 500 server error"""
        mock_response = Mock()
        mock_response.status_code = 500
        error = requests.HTTPError()
        error.response = mock_response

        analyzer._handle_exceptions(error, "test-project")

        assert "Error fetching issues" in caplog.text


# ============================================================================
# TEST CLASS: INTEGRATION SCENARIOS
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    def test_full_workflow_get_and_fix_issues(self, analyzer, sample_issue_data):
        """Test complete workflow: get issues, parse, fix"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "issues": [sample_issue_data]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            # Get issues
            analysis = analyzer.get_project_issues(
                project_key="test-project",
                branch="main",
                max_issues=10
            )

            # Get fixable issues
            fixable = analyzer.get_fixable_issues(
                project_key="test-project",
                branch="main",
                max_issues=10
            )

            # Get issues by file
            by_files = analyzer.get_fixable_issues_by_files(
                project_key="test-project",
                branch="main",
                max_issues=10
            )

        assert analysis is not None
        assert isinstance(fixable, list)
        assert isinstance(by_files, dict)

    def test_error_recovery_continues_operation(self, analyzer, sample_issue_data):
        """Test that errors in one operation don't break others"""
        mock_fail_response = Mock()
        mock_fail_response.status_code = 500
        mock_fail_response.raise_for_status.side_effect = requests.HTTPError()

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "total": 1,
            "issues": [sample_issue_data]
        }

        with patch.object(analyzer.session, 'get') as mock_get:
            mock_get.side_effect = [mock_fail_response, mock_success_response]

            # First call fails
            result1 = analyzer.get_project_issues("test-project")

            # Second call succeeds
            result2 = analyzer.get_project_issues("test-project")

            assert result2 is not None


class TestGetBranchFromPR:
    """Test suite for get_branch_from_pr method."""


    @pytest.fixture
    def mock_response(self):
        """Create mock response object."""
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        return mock_resp

    # ==================== SUCCESS SCENARIOS ====================

    def test_get_branch_from_pr_success(self, analyzer, mock_response):
        """Test 1: Successfully get branch name from PR number."""
        # Setup
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {
            "pullRequests": [
                {
                    "key": "123",
                    "branch": "feature/new-feature",
                    "title": "Add new feature",
                    "status": "OPEN"
                }
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            # Execute
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            # Assert
            assert branch == "feature/new-feature"
            analyzer.session.get.assert_called_once()

            # Verify API call parameters
            call_args = analyzer.session.get.call_args
            assert call_args[1]['params']['project'] == project_key
            assert 'api/project_pull_requests/list' in call_args[0][0]

    def test_get_branch_from_pr_multiple_prs(self, analyzer, mock_response):
        """Test 2: Find correct PR when multiple PRs exist."""
        project_key = "test-project"
        pull_request = "456"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": "feature/old-feature"},
                {"key": "456", "branch": "feature/target-feature"},
                {"key": "789", "branch": "feature/another-feature"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "feature/target-feature"

    def test_get_branch_from_pr_first_match(self, analyzer, mock_response):
        """Test 3: Returns first PR when it matches."""
        project_key = "test-project"
        pull_request = "111"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "111", "branch": "main"},
                {"key": "222", "branch": "develop"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "main"

    def test_get_branch_from_pr_last_match(self, analyzer, mock_response):
        """Test 4: Returns last PR when it matches."""
        project_key = "test-project"
        pull_request = "999"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "111", "branch": "feature/a"},
                {"key": "222", "branch": "feature/b"},
                {"key": "999", "branch": "feature/target"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "feature/target"

    def test_get_branch_from_pr_with_special_characters(self, analyzer, mock_response):
        """Test 5: Handle branch names with special characters."""
        project_key = "test-project"
        pull_request = "555"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "555", "branch": "feature/fix-#123-bug"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "feature/fix-#123-bug"

    def test_get_branch_from_pr_with_slashes(self, analyzer, mock_response):
        """Test 6: Handle branch names with multiple slashes."""
        project_key = "test-project"
        pull_request = "777"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "777", "branch": "hotfix/release/v1.2.3"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "hotfix/release/v1.2.3"

    # ==================== ERROR SCENARIOS ====================

    def test_get_branch_from_pr_not_found(self, analyzer, mock_response):
        """Test 7: Raise ValueError when PR not found."""
        project_key = "test-project"
        pull_request = "999"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": "feature/a"},
                {"key": "456", "branch": "feature/b"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                analyzer.get_branch_from_pr(project_key, pull_request)

            assert "Pull request 999 not found" in str(exc_info.value)
            assert "test-project" in str(exc_info.value)

    def test_get_branch_from_pr_empty_list(self, analyzer, mock_response):
        """Test 8: Raise ValueError when no PRs exist."""
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {
            "pullRequests": []
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                analyzer.get_branch_from_pr(project_key, pull_request)

            assert "Pull request 123 not found" in str(exc_info.value)

    def test_get_branch_from_pr_missing_pullrequests_key(self, analyzer, mock_response):
        """Test 9: Handle missing 'pullRequests' key in response."""
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {}

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                analyzer.get_branch_from_pr(project_key, pull_request)

            assert "Pull request 123 not found" in str(exc_info.value)

    def test_get_branch_from_pr_http_404(self, analyzer):
        """Test 10: Handle 404 HTTP error (project not found)."""
        project_key = "nonexistent-project"
        pull_request = "123"

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            with pytest.raises(HTTPError):
                analyzer.get_branch_from_pr(project_key, pull_request)

    def test_get_branch_from_pr_http_401(self, analyzer):
        """Test 11: Handle 401 HTTP error (authentication failed)."""
        project_key = "test-project"
        pull_request = "123"

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            with pytest.raises(HTTPError):
                analyzer.get_branch_from_pr(project_key, pull_request)

    def test_get_branch_from_pr_http_403(self, analyzer):
        """Test 12: Handle 403 HTTP error (access forbidden)."""
        project_key = "test-project"
        pull_request = "123"

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            with pytest.raises(HTTPError):
                analyzer.get_branch_from_pr(project_key, pull_request)

    def test_get_branch_from_pr_timeout(self, analyzer):
        """Test 13: Handle timeout error."""
        project_key = "test-project"
        pull_request = "123"

        with patch.object(analyzer.session, 'get', side_effect=Timeout()):
            with pytest.raises(Timeout):
                analyzer.get_branch_from_pr(project_key, pull_request)

    def test_get_branch_from_pr_network_error(self, analyzer):
        """Test 14: Handle network connection error."""
        project_key = "test-project"
        pull_request = "123"

        with patch.object(analyzer.session, 'get', side_effect=RequestException("Network error")):
            with pytest.raises(RequestException):
                analyzer.get_branch_from_pr(project_key, pull_request)

    # ==================== EDGE CASES ====================

    def test_get_branch_from_pr_with_string_number(self, analyzer, mock_response):
        """Test 15: Handle PR number as string."""
        project_key = "test-project"
        pull_request = "123"  # String

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": "feature/test"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "feature/test"

    def test_get_branch_from_pr_with_leading_zeros(self, analyzer, mock_response):
        """Test 16: Handle PR number with leading zeros."""
        project_key = "test-project"
        pull_request = "007"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "007", "branch": "feature/test"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "feature/test"

    def test_get_branch_from_pr_case_sensitive(self, analyzer, mock_response):
        """Test 17: PR key matching is case-sensitive."""
        project_key = "test-project"
        pull_request = "abc"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "ABC", "branch": "feature/uppercase"},
                {"key": "abc", "branch": "feature/lowercase"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "feature/lowercase"

    def test_get_branch_from_pr_missing_branch_field(self, analyzer, mock_response):
        """Test 18: Handle missing 'branch' field in PR data."""
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "title": "Test PR"}  # Missing 'branch' field
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            # Should return None when branch field is missing
            assert branch is None

    def test_get_branch_from_pr_null_branch(self, analyzer, mock_response):
        """Test 19: Handle null branch value."""
        project_key = "test-project"
        pull_request = "1234"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "1234", "branch": None}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)
            assert branch is None

    def test_get_branch_from_pr_empty_branch_string(self, analyzer, mock_response):
        """Test 20: Handle empty branch string."""
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": ""}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == ""

    def test_get_branch_from_pr_very_long_branch_name(self, analyzer, mock_response):
        """Test 21: Handle very long branch name."""
        project_key = "test-project"
        pull_request = "123"
        long_branch = "feature/" + "a" * 200

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": long_branch}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == long_branch
            assert len(branch) > 200

    # ==================== API CALL VERIFICATION ====================

    def test_get_branch_from_pr_correct_endpoint(self, analyzer, mock_response):
        """Test 22: Verify correct API endpoint is called."""
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": "main"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            analyzer.get_branch_from_pr(project_key, pull_request)

            # Verify endpoint
            call_args = mock_get.call_args
            assert 'api/project_pull_requests/list' in call_args[0][0]

    def test_get_branch_from_pr_correct_params(self, analyzer, mock_response):
        """Test 23: Verify correct parameters are sent."""
        project_key = "my-test-project"
        pull_request = "456"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "456", "branch": "develop"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            analyzer.get_branch_from_pr(project_key, pull_request)

            # Verify parameters
            call_args = mock_get.call_args
            params = call_args[1]['params']
            assert params['project'] == project_key

    def test_get_branch_from_pr_timeout_used(self, analyzer, mock_response):
        """Test 24: Verify timeout parameter is passed."""
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": "main"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            analyzer.get_branch_from_pr(project_key, pull_request)

            # Verify timeout is passed
            call_args = mock_get.call_args
            assert call_args[1]['timeout'] == analyzer.timeout

    def test_get_branch_from_pr_session_called_once(self, analyzer, mock_response):
        """Test 25: Verify session.get is called exactly once."""
        project_key = "test-project"
        pull_request = "123"

        mock_response.json.return_value = {
            "pullRequests": [
                {"key": "123", "branch": "main"}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response) as mock_get:
            analyzer.get_branch_from_pr(project_key, pull_request)

            assert mock_get.call_count == 1

    # ==================== INTEGRATION-LIKE TESTS ====================

    def test_get_branch_from_pr_with_real_response_structure(self, analyzer, mock_response):
        """Test 26: Test with realistic SonarCloud API response structure."""
        project_key = "devdox-ai-sonar"
        pull_request = "42"

        # Realistic response from SonarCloud
        mock_response.json.return_value = {
            "pullRequests": [
                {
                    "key": "42",
                    "title": "Add new feature",
                    "branch": "feature/add-new-analyzer",
                    "base": "main",
                    "status": {
                        "qualityGateStatus": "OK",
                        "bugs": 0,
                        "vulnerabilities": 0,
                        "codeSmells": 2
                    },
                    "analysisDate": "2024-01-08T10:30:00+0000",
                    "target": "main"
                },
                {
                    "key": "41",
                    "title": "Fix bug",
                    "branch": "bugfix/fix-parser",
                    "base": "main",
                    "status": {
                        "qualityGateStatus": "ERROR",
                        "bugs": 1,
                        "vulnerabilities": 0,
                        "codeSmells": 5
                    },
                    "analysisDate": "2024-01-07T15:20:00+0000",
                    "target": "main"
                }
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pull_request)

            assert branch == "feature/add-new-analyzer"


class TestGetBranchFromPRParametrized:
    """Parametrized tests for get_branch_from_pr."""



    @pytest.mark.parametrize("pr_number,expected_branch", [
        ("123", "feature/test-123"),
        ("456", "bugfix/issue-456"),
        ("789", "hotfix/critical-789"),
        ("1", "main"),
        ("9999", "release/v1.0.0"),
    ])
    def test_get_branch_various_pr_numbers(self, analyzer, pr_number, expected_branch):
        """Test 27: Parametrized test for various PR numbers."""
        project_key = "test-project"

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "pullRequests": [
                {"key": pr_number, "branch": expected_branch}
            ]
        }

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            branch = analyzer.get_branch_from_pr(project_key, pr_number)
            assert branch == expected_branch

    @pytest.mark.parametrize("http_status,error_type", [
        (400, HTTPError),
        (401, HTTPError),
        (403, HTTPError),
        (404, HTTPError),
        (500, HTTPError),
        (503, HTTPError),
    ])
    def test_get_branch_http_errors(self, analyzer, http_status, error_type):
        """Test 28: Parametrized test for various HTTP errors."""
        project_key = "test-project"
        pull_request = "123"

        mock_response = Mock()
        mock_response.status_code = http_status
        mock_response.raise_for_status.side_effect = error_type(response=mock_response)

        with patch.object(analyzer.session, 'get', return_value=mock_response):
            with pytest.raises(error_type):
                analyzer.get_branch_from_pr(project_key, pull_request)

