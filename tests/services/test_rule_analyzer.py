

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open, call
from typing import List, Dict, Any
import json
import requests
from requests.exceptions import RequestException, Timeout, HTTPError

from devdox_ai_sonar.services.rule_analyzer import RuleAnalyzer


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_rule():
    """Sample rule from SonarCloud API."""
    return {
        "key": "java:S1066",
        "name": "Collapsible 'if' statements should be merged",
        "lang": "java",
        "type": "CODE_SMELL",
        "severity": "MAJOR",
        "status": "READY",
        "htmlDesc": "<p>Merging collapsible <code>if</code> statements increases readability.</p>",
        "tags": ["clumsy"],
        "sysTags": ["confusing"],
        "createdAt": "2013-01-21T15:45:00+0100",
        "params": []
    }


@pytest.fixture
def sample_rules_response():
    """Sample paginated response from rules API."""
    return {
        "total": 1000,
        "p": 1,
        "ps": 500,
        "rules": [
            {
                "key": "java:S1066",
                "name": "Test rule 1",
                "lang": "java",
                "type": "CODE_SMELL",
                "severity": "MAJOR"
            },
            {
                "key": "python:S1481",
                "name": "Unused local variables should be removed",
                "lang": "python",
                "type": "CODE_SMELL",
                "severity": "MINOR"
            }
        ]
    }


@pytest.fixture
def analyzer():
    """Create RuleAnalyzer instance with mocked session."""
    with patch('devdox_ai_sonar.services.rule_analyzer.requests.Session'):
        return RuleAnalyzer(
            token="test-token",
            organization="test-org",
            base_url="https://sonarcloud.io",
            timeout=30,
            max_retries=3
        )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestRuleAnalyzerInitialization:
    """Tests for RuleAnalyzer initialization."""

    @patch('devdox_ai_sonar.services.rule_analyzer.requests.Session')
    @patch('devdox_ai_sonar.services.rule_analyzer.HTTPAdapter')
    @patch('devdox_ai_sonar.services.rule_analyzer.Retry')
    def test_initialization_default_values(self, mock_retry, mock_adapter, mock_session):
        """Test initialization with default values."""
        analyzer = RuleAnalyzer(token="test-token", organization="test-org")

        assert analyzer.token == "test-token"
        assert analyzer.organization == "test-org"
        assert analyzer.base_url == "https://sonarcloud.io"
        assert analyzer.timeout == 30

    @patch('devdox_ai_sonar.services.rule_analyzer.requests.Session')
    def test_initialization_custom_values(self, mock_session):
        """Test initialization with custom values."""
        analyzer = RuleAnalyzer(
            token="custom-token",
            organization="custom-org",
            base_url="https://custom.sonarcloud.io",
            timeout=60,
            max_retries=5
        )

        assert analyzer.token == "custom-token"
        assert analyzer.organization == "custom-org"
        assert analyzer.base_url == "https://custom.sonarcloud.io"
        assert analyzer.timeout == 60

    @patch('devdox_ai_sonar.services.rule_analyzer.requests.Session')
    @patch('devdox_ai_sonar.services.rule_analyzer.Retry')
    def test_retry_strategy_configuration(self, mock_retry, mock_session):
        """Test retry strategy is configured correctly."""
        RuleAnalyzer(token="test-token", organization="test-org", max_retries=5)

        mock_retry.assert_called_once_with(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )

    @patch('devdox_ai_sonar.services.rule_analyzer.requests.Session')
    @patch('devdox_ai_sonar.services.rule_analyzer.HTTPAdapter')
    def test_http_adapter_configuration(self, mock_adapter, mock_session):
        """Test HTTP adapter is configured with connection pooling."""
        session_instance = Mock()
        mock_session.return_value = session_instance

        RuleAnalyzer(token="test-token", organization="test-org")

        # Verify adapter was created with correct settings
        mock_adapter.assert_called_once()
        call_kwargs = mock_adapter.call_args[1]
        assert call_kwargs['pool_connections'] == 10
        assert call_kwargs['pool_maxsize'] == 20

    @patch('devdox_ai_sonar.services.rule_analyzer.requests.Session')
    def test_authentication_header_set(self, mock_session):
        """Test authentication header is set correctly."""
        session_instance = Mock()
        mock_session.return_value = session_instance

        RuleAnalyzer(token="my-secret-token", organization="test-org")

        session_instance.headers.update.assert_called_once_with({
            "Authorization": "Bearer my-secret-token",
            "Accept": "application/json"
        })

    @patch('devdox_ai_sonar.services.rule_analyzer.requests.Session')
    def test_organization_converted_to_string(self, mock_session):
        """Test organization is converted to string."""
        analyzer = RuleAnalyzer(token="test", organization=12345)
        assert analyzer.organization == "12345"
        assert isinstance(analyzer.organization, str)

