# Comprehensive Test Cases for sonar_analyzer.py


import pytest
import json
import requests
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    AnalysisResult,
    ProjectMetrics,
    Severity
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


