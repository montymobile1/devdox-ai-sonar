"""Shared pytest fixtures and configuration for devdox-ai-sonar tests."""

import pytest
from pathlib import Path
from unittest.mock import Mock
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    FixSuggestion,
    AnalysisResult,
    ProjectMetrics,
    Severity,
    IssueType,
    Impact,
)


# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_project_dir(tmp_path):
    """Create a sample project directory structure."""
    # Create src directory
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create test Python file
    test_file = src_dir / "example.py"
    test_file.write_text(
        '''"""Example module for testing."""


def unused_function():
    """This function is never called."""
    unused_var = 42
    return unused_var


def complex_function(x, y, z):
    """A function with high cognitive complexity."""
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            if z > 0:
                return x + z
            else:
                return x
    else:
        if y > 0:
            if z > 0:
                return y + z
            else:
                return y
        else:
            if z > 0:
                return z
            else:
                return 0


def sql_injection_vulnerable(user_input):
    """Function with SQL injection vulnerability."""
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return query
'''
    )

    # Create tests directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    test_test_file = tests_dir / "test_example.py"
    test_test_file.write_text(
        '''"""Tests for example module."""


def test_unused_function():
    """Test unused function."""
    pass
'''
    )

    # Create .git directory (to simulate git repo)
    (tmp_path / ".git").mkdir()

    # Create README
    (tmp_path / "README.md").write_text("# Test Project")

    return tmp_path


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def sample_sonar_issue():
    """Create a sample SonarCloud issue."""
    return SonarIssue(
        key="test-project:src/example.py:S1481",
        rule="python:S1481",
        severity=Severity.MAJOR,
        component="test-project:src/example.py",
        project="test-project",
        first_line=7,
        last_line=7,
        message='Remove the unused local variable "unused_var".',
        type=IssueType.CODE_SMELL,
        impact=Impact.MEDIUM,
        file="src/example.py",
        branch="main",
        status="OPEN",
        creation_date="2024-01-01T10:00:00+0000",
        update_date="2024-01-02T10:00:00+0000",
        tags=["unused", "dead-code"],
    )


@pytest.fixture
def sample_complexity_issue():
    """Create a cognitive complexity issue."""
    return SonarIssue(
        key="test-project:src/example.py:S3776",
        rule="python:S3776",
        severity=Severity.CRITICAL,
        component="test-project:src/example.py",
        project="test-project",
        first_line=12,
        last_line=35,
        message="Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed.",
        type=IssueType.CODE_SMELL,
        impact=Impact.HIGH,
        file="src/example.py",
        branch="main",
        status="OPEN",
    )


@pytest.fixture
def sample_security_issue():
    """Create a security vulnerability issue."""
    return SonarIssue(
        key="test-project:src/example.py:S3649",
        rule="python:S3649",
        severity=Severity.BLOCKER,
        component="test-project:src/example.py",
        project="test-project",
        first_line=38,
        last_line=40,
        message="Make sure that executing SQL queries is safe here.",
        type=IssueType.VULNERABILITY,
        impact=Impact.HIGH,
        file="src/example.py",
        branch="main",
        status="OPEN",
    )


@pytest.fixture
def sample_fix_suggestion():
    """Create a sample fix suggestion."""
    return FixSuggestion(
        issue_key="test-project:src/example.py:S1481",
        original_code="    unused_var = 42\n    return unused_var",
        fixed_code="    return 42",
        explanation="Removed unused variable assignment and directly returned the value",
        confidence=0.95,
        llm_model="gpt-4",
        rule_description="Unused local variables should be removed",
        file_path="src/example.py",
        line_number=7,
        sonar_line_number=7,
        last_line_number=8,
    )


@pytest.fixture
def sample_analysis_result(sample_sonar_issue):
    """Create a sample analysis result."""
    return AnalysisResult(
        project_key="test-project",
        organization="test-org",
        branch="main",
        total_issues=1,
        issues=[sample_sonar_issue],
        metrics=ProjectMetrics(
            project_key="test-project",
            lines_of_code=1000,
            coverage=85.5,
            duplicated_lines_density=3.2,
            maintainability_rating="A",
            reliability_rating="B",
            security_rating="A",
            bugs=5,
            vulnerabilities=2,
            code_smells=15,
            technical_debt="2h 30min",
        ),
    )


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_sonar_api_response():
    """Create a mock SonarCloud API response."""
    return {
        "issues": [
            {
                "key": "AXqT8_example",
                "rule": "python:S1481",
                "severity": "MAJOR",
                "component": "test-project:src/example.py",
                "project": "test-project",
                "line": 7,
                "message": 'Remove the unused local variable "unused_var".',
                "type": "CODE_SMELL",
                "status": "OPEN",
                "creationDate": "2024-01-01T10:00:00+0000",
                "updateDate": "2024-01-02T10:00:00+0000",
                "tags": ["unused"],
                "impacts": [
                    {"softwareQuality": "MAINTAINABILITY", "severity": "MEDIUM"}
                ],
                "flows": [
                    {
                        "locations": [
                            {
                                "textRange": {
                                    "startLine": 7,
                                    "endLine": 8,
                                    "startOffset": 4,
                                    "endOffset": 24,
                                }
                            }
                        ]
                    }
                ],
            }
        ],
        "paging": {"total": 1, "pageIndex": 1, "pageSize": 500},
    }


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM API response for fix generation."""
    return {
        "FIXED_SELECTION": "    return 42",
        "NEW_HELPER_CODE": "",
        "PLACEMENT": "SIBLING",
        "EXPLANATION": "Removed unused variable assignment",
        "CONFIDENCE": 0.95,
    }


@pytest.fixture
def mock_rule_info():
    """Create mock rule information from SonarCloud."""
    return {
        "key": "python:S1481",
        "name": "Unused local variables should be removed",
        "language": "python",
        "category": "CODE_SMELL",
        "severity": "MAJOR",
        "description": "If a local variable is declared but not used, it is dead code and should be removed.",
        "root_cause": "Unused code creates clutter and indicates incomplete implementation",
        "how_to_fix": {
            "description": "Remove unused code or implement its intended functionality",
            "steps": [
                "Identify all unused elements",
                "Verify they are truly not needed",
                "Remove unused elements",
                "Run tests to ensure no functionality is broken",
            ],
            "priority": "Medium",
            "effort": "Low",
        },
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        "SONAR_TOKEN": "test-sonar-token",
        "SONAR_ORGANIZATION": "test-org",
        "SONAR_PROJECT_KEY": "test-project",
        "OPENAI_API_KEY": "test-openai-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def mock_settings(mock_env_vars):
    """Create mock settings object."""
    from devdox_ai_sonar.config import Settings

    return Settings()


# ============================================================================
# Network Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_requests_session():
    """Create a mock requests.Session."""
    session = Mock()
    session.headers = {}
    session.get = Mock()
    session.post = Mock()
    session.mount = Mock()
    return session


@pytest.fixture
def mock_successful_request(mock_sonar_api_response):
    """Create a mock successful HTTP request."""
    response = Mock()
    response.status_code = 200
    response.ok = True
    response.json.return_value = mock_sonar_api_response
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def mock_failed_request():
    """Create a mock failed HTTP request."""
    import requests

    response = Mock()
    response.status_code = 404
    response.ok = False
    response.text = "Not Found"
    response.raise_for_status.side_effect = requests.HTTPError()
    return response


# ============================================================================
# Helper Functions
# ============================================================================


def create_test_file(directory: Path, filename: str, content: str) -> Path:
    """
    Create a test file with given content.

    Args:
        directory: Directory to create file in
        filename: Name of the file
        content: Content to write to file

    Returns:
        Path to the created file
    """
    file_path = directory / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API key"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )


# Make helper available as fixture
@pytest.fixture
def create_file():
    """Fixture version of create_test_file helper."""
    return create_test_file
