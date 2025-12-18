import pytest
from unittest.mock import Mock, patch, mock_open
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
def sample_null_rule():
    """Sample rule for null pointer detection."""
    return {
        "key": "java:S2259",
        "name": "Null pointers should not be dereferenced",
        "lang": "java",
        "type": "BUG",
        "severity": "CRITICAL",
        "status": "READY",
        "htmlDesc": "<p>A null pointer dereference will cause a NullPointerException</p>",
        "tags": ["bug"],
        "sysTags": [],
        "createdAt": "2015-01-21T15:45:00+0100",
        "params": []
    }


@pytest.fixture
def sample_sql_injection_rule():
    """Sample rule for SQL injection."""
    return {
        "key": "python:S3649",
        "name": "Database queries should not be vulnerable to SQL injection",
        "lang": "python",
        "type": "VULNERABILITY",
        "severity": "CRITICAL",
        "status": "READY",
        "htmlDesc": "<p>SQL injection attacks can compromise database security</p>",
        "tags": ["sql", "injection", "security"],
        "sysTags": ["security"],
        "createdAt": "2016-01-21T15:45:00+0100",
        "params": []
    }


@pytest.fixture
def sample_password_rule():
    """Sample rule for hardcoded passwords."""
    return {
        "key": "java:S2068",
        "name": "Credentials should not be hard-coded",
        "lang": "java",
        "type": "VULNERABILITY",
        "severity": "BLOCKER",
        "status": "READY",
        "htmlDesc": "<p>Hard-coded credentials are security risks</p>",
        "tags": ["security", "password"],
        "sysTags": [],
        "createdAt": "2014-01-21T15:45:00+0100",
        "params": []
    }


@pytest.fixture
def sample_complexity_rule():
    """Sample rule for cognitive complexity."""
    return {
        "key": "python:S3776",
        "name": "Cognitive Complexity of functions should not be too high",
        "lang": "python",
        "type": "CODE_SMELL",
        "severity": "CRITICAL",
        "status": "READY",
        "htmlDesc": "<p>Functions with high cognitive complexity are hard to maintain</p>",
        "tags": ["brain-overload"],
        "sysTags": [],
        "createdAt": "2017-01-21T15:45:00+0100",
        "params": []
    }


@pytest.fixture
def sample_duplicate_rule():
    """Sample rule for code duplication."""
    return {
        "key": "common:DuplicatedBlocks",
        "name": "Source files should not have any duplicated blocks",
        "lang": "multi",
        "type": "CODE_SMELL",
        "severity": "MAJOR",
        "status": "READY",
        "htmlDesc": "<p>Duplicated code is hard to maintain</p>",
        "tags": ["duplicate"],
        "sysTags": [],
        "createdAt": "2012-01-21T15:45:00+0100",
        "params": []
    }


@pytest.fixture
def sample_empty_rule():
    """Sample rule for empty blocks."""
    return {
        "key": "java:S108",
        "name": "Nested blocks of code should not be left empty",
        "lang": "java",
        "type": "CODE_SMELL",
        "severity": "MAJOR",
        "status": "READY",
        "htmlDesc": "<p>Empty blocks should be removed or filled</p>",
        "tags": ["suspicious"],
        "sysTags": [],
        "createdAt": "2013-01-21T15:45:00+0100",
        "params": []
    }


@pytest.fixture
def sample_unused_rule():
    """Sample rule for unused code."""
    return {
        "key": "python:S1481",
        "name": "Unused local variables should be removed",
        "lang": "python",
        "type": "CODE_SMELL",
        "severity": "MINOR",
        "status": "READY",
        "htmlDesc": "<p>Unused variables clutter the code</p>",
        "tags": ["unused"],
        "sysTags": [],
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
                "severity": "MAJOR",
                "htmlDesc": "<p>Test description</p>",
                "tags": [],
                "sysTags": [],
                "createdAt": "2013-01-21T15:45:00+0100",
                "params": []
            },
            {
                "key": "python:S1481",
                "name": "Unused local variables should be removed",
                "lang": "python",
                "type": "CODE_SMELL",
                "severity": "MINOR",
                "htmlDesc": "<p>Unused variables</p>",
                "tags": ["unused"],
                "sysTags": [],
                "createdAt": "2013-01-21T15:45:00+0100",
                "params": []
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


# ============================================================================
# FETCH ALL RULES TESTS
# ============================================================================


class TestFetchAllRules:
    """Tests for fetch_all_rules method."""

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_fetch_all_rules_single_page(self, mock_sleep, analyzer, sample_rules_response):
        """Test fetching rules when all fit in one page."""
        # Setup mock to return all rules in first page
        sample_rules_response['total'] = 2
        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=sample_rules_response),
            raise_for_status=Mock()
        ))

        result = analyzer.fetch_all_rules()

        # Verify API call
        analyzer.session.get.assert_called_once()
        call_args = analyzer.session.get.call_args
        assert call_args[1]['params']['ps'] == 500
        assert call_args[1]['params']['p'] == 1
        assert call_args[1]['params']['organization'] == 'test-org'
        assert call_args[1]['timeout'] == 30

        # Verify results
        assert 'rules' in result
        assert 'metadata' in result
        assert len(result['rules']) == 2
        mock_sleep.assert_not_called()

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_fetch_all_rules_multiple_pages(self, mock_sleep, analyzer):
        """Test fetching rules across multiple pages."""
        page1_response = {
            "total": 1000,
            "p": 1,
            "ps": 500,
            "rules": [
                {"key": f"rule:{i}", "name": f"Rule {i}", "lang": "java", "type": "CODE_SMELL", "severity": "MAJOR",
                 "htmlDesc": "", "tags": [], "sysTags": [], "createdAt": "", "params": []} for i in range(500)]
        }
        page2_response = {
            "total": 1000,
            "p": 2,
            "ps": 500,
            "rules": [{"key": f"rule:{i}", "name": f"Rule {i}", "lang": "python", "type": "BUG", "severity": "MINOR",
                       "htmlDesc": "", "tags": [], "sysTags": [], "createdAt": "", "params": []} for i in
                      range(500, 1000)]
        }

        analyzer.session.get = Mock(side_effect=[
            Mock(json=Mock(return_value=page1_response), raise_for_status=Mock()),
            Mock(json=Mock(return_value=page2_response), raise_for_status=Mock())
        ])

        result = analyzer.fetch_all_rules()

        # Verify two API calls
        assert analyzer.session.get.call_count == 2

        # Verify page parameters
        first_call = analyzer.session.get.call_args_list[0]
        second_call = analyzer.session.get.call_args_list[1]
        assert first_call[1]['params']['p'] == 1
        assert second_call[1]['params']['p'] == 2

        # Verify results
        assert len(result['rules']) == 1000
        mock_sleep.assert_called_with(0.1)

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_fetch_all_rules_with_language_filter(self, mock_sleep, analyzer, sample_rules_response):
        """Test fetching rules with language filter."""
        sample_rules_response['total'] = 2
        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=sample_rules_response),
            raise_for_status=Mock()
        ))

        result = analyzer.fetch_all_rules(languages=['python', 'java'])

        # Verify language filter in params
        call_args = analyzer.session.get.call_args
        assert call_args[1]['params']['languages'] == 'python,java'

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_fetch_all_rules_empty_response(self, mock_sleep, analyzer):
        """Test handling empty rules response."""
        empty_response = {"total": 0, "p": 1, "ps": 500, "rules": []}
        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=empty_response),
            raise_for_status=Mock()
        ))

        result = analyzer.fetch_all_rules()

        assert len(result['rules']) == 0
        assert result['metadata']['total_rules'] == 0

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_fetch_all_rules_request_exception(self, mock_sleep, analyzer):
        """Test handling request exceptions."""
        analyzer.session.get = Mock(side_effect=RequestException("Network error"))

        result = analyzer.fetch_all_rules()

        # Should return empty processed rules on error
        assert len(result['rules']) == 0
        assert result['metadata']['total_rules'] == 0

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_fetch_all_rules_http_error(self, mock_sleep, analyzer):
        """Test handling HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        analyzer.session.get = Mock(return_value=mock_response)

        result = analyzer.fetch_all_rules()

        assert len(result['rules']) == 0


# ============================================================================
# ROOT CAUSE INFERENCE TESTS
# ============================================================================


class TestRootCauseInference:
    """Tests for _infer_root_cause method."""

    def test_infer_root_cause_unused(self, analyzer, sample_unused_rule):
        """Test inferring root cause for unused code."""
        cause = analyzer._infer_root_cause(sample_unused_rule)
        assert "unused code" in cause.lower()
        assert "clutter" in cause.lower()

    def test_infer_root_cause_null_pointer(self, analyzer, sample_null_rule):
        """Test inferring root cause for null pointer issues."""
        cause = analyzer._infer_root_cause(sample_null_rule)
        assert "null pointer" in cause.lower()
        assert "nullpointerexception" in cause.lower()

    def test_infer_root_cause_sql_injection(self, analyzer, sample_sql_injection_rule):
        """Test inferring root cause for SQL injection."""
        cause = analyzer._infer_root_cause(sample_sql_injection_rule)
        assert "sql injection" in cause.lower()
        assert "security" in cause.lower() or "compromise" in cause.lower()

    def test_infer_root_cause_password(self, analyzer, sample_password_rule):
        """Test inferring root cause for hardcoded credentials."""
        cause = analyzer._infer_root_cause(sample_password_rule)
        assert "credential" in cause.lower() or "password" in cause.lower()
        assert "security" in cause.lower()

    def test_infer_root_cause_complexity(self, analyzer, sample_complexity_rule):
        """Test inferring root cause for high complexity."""
        cause = analyzer._infer_root_cause(sample_complexity_rule)
        assert "complex" in cause.lower()
        assert "maintain" in cause.lower() or "understand" in cause.lower()

    def test_infer_root_cause_duplicate(self, analyzer, sample_duplicate_rule):
        """Test inferring root cause for code duplication."""
        cause = analyzer._infer_root_cause(sample_duplicate_rule)
        assert "duplicate" in cause.lower() or "duplication" in cause.lower()
        assert "maintain" in cause.lower() or "maintenance" in cause.lower()

    def test_infer_root_cause_empty_block(self, analyzer, sample_empty_rule):
        """Test inferring root cause for empty blocks."""
        cause = analyzer._infer_root_cause(sample_empty_rule)
        assert "empty" in cause.lower()
        assert "incomplete" in cause.lower() or "missing" in cause.lower()

    def test_infer_root_cause_resource_leak(self, analyzer):
        """Test inferring root cause for resource leaks."""
        rule = {
            "key": "java:S2095",
            "name": "Resources should be closed",
            "htmlDesc": "Unclosed resources cause memory leaks",
            "tags": [],
            "type": "BUG"
        }
        cause = analyzer._infer_root_cause(rule)
        assert "resource" in cause.lower()
        assert "leak" in cause.lower()

    def test_infer_root_cause_thread_safety(self, analyzer):
        """Test inferring root cause for thread safety issues."""
        rule = {
            "key": "java:S2886",
          "name": "Non-thread-safe fields should not be static",
          "language": "java",
          "category": "BUG",
          "severity": "MAJOR",
          "status": "READY",
          "description": "Why is this an issue? When an object is marked as static , it means that it belongs to the class rather than any class instance. This means there is only one copy of the static object in memory, regardless of how many class instances are created. Static objects are shared among all instances of the class and can be accessed using the class name rather than an instance of the class. A data type is considered thread-safe if it can be used correctly by multiple threads, regardless of how those threads are executed, without requiring additional coordination from the calling code. In other words, a thread-safe data type can be accessed and modified by multiple threads simultaneously without causing any issues or requiring extra work from the programmer to ensure correct behavior. Non-thread-safe objects are objects that are not designed to be used in a multi-threaded environment and can lead to race conditions and data inconsistencies when accessed by multiple threads simultaneously. Using ...",
          "tags": [],
          "system_tags": [
            "multi-threading"
          ],
          "created_at": "2015-05-20T03:01:40+0000",
          "parameters": [],
          "root_cause": "Code duplication increases maintenance burden and creates consistency risks when changes are needed",
          "how_to_fix": {
            "description": "Fix logical error or potential runtime issue",
            "steps": [
              "Understand the root cause of the bug",
              "Implement the correct logic",
              "Add comprehensive tests to prevent regression",
              "Verify the fix doesn't introduce new issues"
            ],
            "priority": "High",
            "effort": "Medium"

        }
        }
        cause = analyzer._infer_root_cause(rule)

        assert "thread" in cause.lower() or "concurrenc" in cause.lower()

    def test_infer_root_cause_exception_handling(self, analyzer):
        """Test inferring root cause for exception handling issues."""
        rule = {
            "key": "python:S1181",
            "name": "Exceptions should not be caught too broadly",
            "htmlDesc": "Catching broad exceptions hides errors",
            "tags": [],
            "type": "CODE_SMELL"
        }
        cause = analyzer._infer_root_cause(rule)
        assert "exception" in cause.lower()

    def test_infer_root_cause_security_tag(self, analyzer):
        """Test inferring root cause for rules with security tag."""
        rule = {
            "key": "test:S001",
            "name": "Generic security rule",
            "htmlDesc": "Security issue",
            "tags": ["security"],
            "type": "CODE_SMELL"
        }
        cause = analyzer._infer_root_cause(rule)
        assert "security" in cause.lower()

    def test_infer_root_cause_vulnerability_type(self, analyzer):
        """Test inferring root cause for vulnerability type rules."""
        rule = {
            "key": "test:S002",
            "name": "Generic vulnerability",
            "htmlDesc": "Vulnerability",
            "tags": [],
            "type": "VULNERABILITY"
        }
        cause = analyzer._infer_root_cause(rule)
        assert "security" in cause.lower()

    def test_infer_root_cause_bug_type(self, analyzer):
        """Test inferring root cause for bug type rules."""
        rule = {
            "key": "test:S003",
            "name": "Generic bug",
            "htmlDesc": "This is a bug",
            "tags": [],
            "type": "BUG"
        }
        cause = analyzer._infer_root_cause(rule)
        assert "bug" in cause.lower() or "coding error" in cause.lower()

    def test_infer_root_cause_code_smell(self, analyzer):
        """Test inferring root cause for code smell type rules."""
        rule = {
            "key": "test:S004",
            "name": "Generic code smell",
            "htmlDesc": "Code quality issue",
            "tags": [],
            "type": "CODE_SMELL"
        }
        cause = analyzer._infer_root_cause(rule)
        assert "code quality" in cause.lower() or "maintainability" in cause.lower()

    def test_infer_root_cause_unknown(self, analyzer):
        """Test inferring root cause for unknown rule type."""
        rule = {
            "key": "test:S005",
            "name": "Unknown rule",
            "htmlDesc": "Some description",
            "tags": [],
            "type": "UNKNOWN"
        }
        cause = analyzer._infer_root_cause(rule)
        assert "unknown" in cause.lower()


# ============================================================================
# CONTAINS KEYWORDS TESTS
# ============================================================================


class TestContainsKeywords:
    """Tests for _contains_keywords helper method."""

    def test_contains_keywords_in_name(self, analyzer):
        """Test keyword detection in name."""
        assert analyzer._contains_keywords("unused variable", "", ["unused"]) is True

    def test_contains_keywords_in_description(self, analyzer):
        """Test keyword detection in description."""
        assert analyzer._contains_keywords("", "this variable is unused", ["unused"]) is True

    def test_contains_keywords_multiple(self, analyzer):
        """Test multiple keyword detection."""
        assert analyzer._contains_keywords("sql query", "", ["sql", "injection"]) is True

    def test_contains_keywords_none_match(self, analyzer):
        """Test when no keywords match."""
        assert analyzer._contains_keywords("some text", "more text", ["unused", "null"]) is False

    def test_contains_keywords_empty_strings(self, analyzer):
        """Test with empty strings."""
        assert analyzer._contains_keywords("", "", ["keyword"]) is False

    def test_contains_keywords_empty_list(self, analyzer):
        """Test with empty keyword list."""
        assert analyzer._contains_keywords("text", "more text", []) is False


# ============================================================================
# PROCESS RULES TESTS
# ============================================================================


class TestProcessRules:
    """Tests for _process_rules method."""

    def test_process_rules_basic(self, analyzer, sample_rule):
        """Test basic rule processing."""
        result = analyzer._process_rules([sample_rule])

        assert 'rules' in result
        assert 'metadata' in result
        assert 'java:S1066' in result['rules']

        processed_rule = result['rules']['java:S1066']
        assert processed_rule['name'] == sample_rule['name']
        assert processed_rule['language'] == 'java'
        assert processed_rule['category'] == 'CODE_SMELL'
        assert processed_rule['severity'] == 'MAJOR'
        assert 'root_cause' in processed_rule
        assert 'how_to_fix' in processed_rule

    def test_process_rules_multiple_languages(self, analyzer):
        """Test processing rules from multiple languages."""
        rules = [
            {
                "key": "java:S001",
                "name": "Java rule",
                "lang": "java",
                "type": "BUG",
                "severity": "MAJOR",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            },
            {
                "key": "python:S002",
                "name": "Python rule",
                "lang": "python",
                "type": "CODE_SMELL",
                "severity": "MINOR",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            },
            {
                "key": "js:S003",
                "name": "JavaScript rule",
                "lang": "js",
                "type": "VULNERABILITY",
                "severity": "CRITICAL",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            }
        ]

        result = analyzer._process_rules(rules)

        assert len(result['rules']) == 3
        assert result['metadata']['languages']['java'] == 1
        assert result['metadata']['languages']['python'] == 1
        assert result['metadata']['languages']['js'] == 1

    def test_process_rules_metadata(self, analyzer):
        """Test metadata generation."""
        rules = [
            {
                "key": "java:S001",
                "name": "Rule 1",
                "lang": "java",
                "type": "BUG",
                "severity": "CRITICAL",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            },
            {
                "key": "java:S002",
                "name": "Rule 2",
                "lang": "java",
                "type": "CODE_SMELL",
                "severity": "MAJOR",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            }
        ]

        result = analyzer._process_rules(rules)

        metadata = result['metadata']
        assert metadata['total_rules'] == 2
        assert metadata['organization'] == 'test-org'
        assert 'generated_at' in metadata
        assert 'categories' in metadata
        assert 'severities' in metadata

    def test_process_rules_empty_list(self, analyzer):
        """Test processing empty rules list."""
        result = analyzer._process_rules([])

        assert len(result['rules']) == 0
        assert result['metadata']['total_rules'] == 0

    def test_process_rules_missing_fields(self, analyzer):
        """Test processing rules with missing fields."""
        rule = {
            "key": "test:S001"
            # Missing most fields
        }

        result = analyzer._process_rules([rule])

        processed = result['rules']['test:S001']
        assert processed['name'] == ''
        assert processed['language'] == 'Generic'
        assert processed['category'] == 'Unknown'
        assert processed['severity'] == 'INFO'


# ============================================================================
# CLEAN HTML DESCRIPTION TESTS
# ============================================================================


class TestCleanHtmlDescription:
    """Tests for _clean_html_description method."""

    def test_clean_html_description_basic(self, analyzer):
        """Test cleaning basic HTML."""
        html = "<p>This is a <strong>test</strong> description</p>"
        result = analyzer._clean_html_description(html)
        assert result == "This is a test description"
        assert "<" not in result
        assert ">" not in result

    def test_clean_html_description_complex(self, analyzer):
        """Test cleaning complex HTML."""
        html = """
        <p>This is a test</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        <code>code snippet</code>
        """
        result = analyzer._clean_html_description(html)
        assert "This is a test" in result
        assert "Item 1" in result
        assert "Item 2" in result
        assert "<" not in result

    def test_clean_html_description_whitespace(self, analyzer):
        """Test whitespace normalization."""
        html = "<p>Too    much    whitespace\n\n\nhere</p>"
        result = analyzer._clean_html_description(html)
        assert result == "Too much whitespace here"

    def test_clean_html_description_empty(self, analyzer):
        """Test empty HTML."""
        result = analyzer._clean_html_description("")
        assert result == ""

    def test_clean_html_description_truncation(self, analyzer):
        """Test long description truncation."""
        long_html = "<p>" + "x" * 1500 + "</p>"
        result = analyzer._clean_html_description(long_html)
        assert len(result) <= 1003  # 1000 chars + "..."
        assert result.endswith("...")

    def test_clean_html_description_no_truncation(self, analyzer):
        """Test description under limit."""
        html = "<p>" + "x" * 500 + "</p>"
        result = analyzer._clean_html_description(html)
        assert len(result) == 500
        assert not result.endswith("...")


# ============================================================================
# GENERATE FIX GUIDANCE TESTS
# ============================================================================


class TestGenerateFixGuidance:
    """Tests for _generate_fix_guidance method."""

    def test_fix_guidance_unused(self, analyzer, sample_unused_rule):
        """Test fix guidance for unused code."""
        guidance = analyzer._generate_fix_guidance(sample_unused_rule)

        assert guidance['priority'] == 'Medium'
        assert guidance['effort'] == 'Low'
        assert 'remove' in guidance['description'].lower()
        assert len(guidance['steps']) > 0

    def test_fix_guidance_null(self, analyzer, sample_null_rule):
        """Test fix guidance for null pointer issues."""
        guidance = analyzer._generate_fix_guidance(sample_null_rule)

        assert guidance['priority'] == 'High'
        assert guidance['effort'] == 'Medium'
        assert 'null check' in guidance['description'].lower()

    def test_fix_guidance_sql_injection(self, analyzer, sample_sql_injection_rule):
        """Test fix guidance for SQL injection."""
        guidance = analyzer._generate_fix_guidance(sample_sql_injection_rule)

        assert guidance['priority'] == 'Critical'
        assert guidance['effort'] == 'Medium'
        assert 'parameterized' in guidance['description'].lower()

    def test_fix_guidance_password(self, analyzer, sample_password_rule):
        """Test fix guidance for hardcoded credentials."""
        guidance = analyzer._generate_fix_guidance(sample_password_rule)

        assert guidance['priority'] == 'Critical'
        assert guidance['effort'] == 'Medium'
        assert 'environment' in guidance['description'].lower()

    def test_fix_guidance_complexity(self, analyzer, sample_complexity_rule):
        """Test fix guidance for complexity issues."""
        guidance = analyzer._generate_fix_guidance(sample_complexity_rule)

        assert guidance['priority'] == 'Medium'
        assert guidance['effort'] == 'High'
        assert 'refactor' in guidance['description'].lower()

    def test_fix_guidance_duplicate(self, analyzer, sample_duplicate_rule):
        """Test fix guidance for code duplication."""
        guidance = analyzer._generate_fix_guidance(sample_duplicate_rule)

        assert guidance['priority'] == 'Medium'
        assert guidance['effort'] == 'Medium'
        assert 'extract' in guidance['description'].lower()

    def test_fix_guidance_empty(self, analyzer, sample_empty_rule):
        """Test fix guidance for empty blocks."""
        guidance = analyzer._generate_fix_guidance(sample_empty_rule)

        assert guidance['priority'] == 'High'
        assert guidance['effort'] == 'Low'
        assert 'implement' in guidance['description'].lower() or 'remove' in guidance['description'].lower()

    def test_fix_guidance_vulnerability(self, analyzer):
        """Test fix guidance for vulnerabilities."""
        rule = {
            "key": "test:S001",
            "name": "Security issue",
            "type": "VULNERABILITY"
        }
        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance['priority'] == 'Critical'
        assert guidance['effort'] == 'High'
        assert 'security' in guidance['description'].lower()

    def test_fix_guidance_bug(self, analyzer):
        """Test fix guidance for bugs."""
        rule = {
            "key": "test:S002",
            "name": "Logic error",
            "type": "BUG"
        }
        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance['priority'] == 'High'
        assert guidance['effort'] == 'Medium'
        assert 'bug' in guidance['description'].lower() or 'fix' in guidance['description'].lower()

    def test_fix_guidance_default(self, analyzer):
        """Test default fix guidance."""
        rule = {
            "key": "test:S003",
            "name": "Generic rule",
            "type": "CODE_SMELL"
        }
        guidance = analyzer._generate_fix_guidance(rule)

        assert guidance['priority'] == 'Low'
        assert guidance['effort'] == 'Low'
        assert 'best practices' in guidance['description'].lower()


# ============================================================================
# STATISTICS TESTS
# ============================================================================


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_category_stats(self, analyzer):
        """Test category statistics calculation."""
        rules = {
            "rule1": {"category": "BUG"},
            "rule2": {"category": "CODE_SMELL"},
            "rule3": {"category": "CODE_SMELL"},
            "rule4": {"category": "VULNERABILITY"}
        }

        stats = analyzer._get_category_stats(rules)

        assert stats['BUG'] == 1
        assert stats['CODE_SMELL'] == 2
        assert stats['VULNERABILITY'] == 1

    def test_get_category_stats_empty(self, analyzer):
        """Test category statistics with empty rules."""
        stats = analyzer._get_category_stats({})
        assert stats == {}

    def test_get_severity_stats(self, analyzer):
        """Test severity statistics calculation."""
        rules = {
            "rule1": {"severity": "CRITICAL"},
            "rule2": {"severity": "MAJOR"},
            "rule3": {"severity": "MAJOR"},
            "rule4": {"severity": "MINOR"}
        }

        stats = analyzer._get_severity_stats(rules)

        assert stats['CRITICAL'] == 1
        assert stats['MAJOR'] == 2
        assert stats['MINOR'] == 1

    def test_get_severity_stats_empty(self, analyzer):
        """Test severity statistics with empty rules."""
        stats = analyzer._get_severity_stats({})
        assert stats == {}


# ============================================================================
# GET RULE BY KEY TESTS
# ============================================================================


class TestGetRuleByKey:
    """Tests for get_rule_by_key method."""

    def test_get_rule_by_key_success(self, analyzer, sample_rule):
        """Test successful rule retrieval by key."""
        api_response = {"rule": sample_rule}
        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=api_response),
            raise_for_status=Mock()
        ))

        result = analyzer.get_rule_by_key("java:S1066")

        assert result is not None
        assert result['name'] == sample_rule['name']
        assert result['language'] == 'java'

        # Verify API call
        call_args = analyzer.session.get.call_args
        assert call_args[1]['params']['key'] == 'java:S1066'
        assert call_args[1]['params']['organization'] == 'test-org'

    def test_get_rule_by_key_not_found(self, analyzer):
        """Test rule not found scenario."""
        api_response = {"rule": {}}
        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=api_response),
            raise_for_status=Mock()
        ))

        result = analyzer.get_rule_by_key("nonexistent:S999")

        assert result is None

    def test_get_rule_by_key_request_exception(self, analyzer):
        """Test handling request exception."""
        analyzer.session.get = Mock(side_effect=RequestException("Network error"))

        result = analyzer.get_rule_by_key("java:S1066")

        assert result is None

    def test_get_rule_by_key_timeout(self, analyzer):
        """Test handling timeout."""
        analyzer.session.get = Mock(side_effect=Timeout("Request timeout"))

        result = analyzer.get_rule_by_key("java:S1066")

        assert result is None


# ============================================================================
# GET RULES FOR LANGUAGE TESTS
# ============================================================================


class TestGetRulesForLanguage:
    """Tests for get_rules_for_language method."""

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_get_rules_for_language(self, mock_sleep, analyzer):
        """Test fetching rules for specific language."""
        response = {
            "total": 1,
            "p": 1,
            "ps": 500,
            "rules": [{
                "key": "python:S001",
                "name": "Python rule",
                "lang": "python",
                "type": "BUG",
                "severity": "MAJOR",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            }]
        }

        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=response),
            raise_for_status=Mock()
        ))

        result = analyzer.get_rules_for_language("python")

        # Verify language filter
        call_args = analyzer.session.get.call_args
        assert call_args[1]['params']['languages'] == 'python'

        assert len(result['rules']) == 1


# ============================================================================
# GET RULES BY SEVERITY TESTS
# ============================================================================


class TestGetRulesBySeverity:
    """Tests for get_rules_by_severity method."""

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_get_rules_by_severity(self, mock_sleep, analyzer):
        """Test filtering rules by severity."""
        response = {
            "total": 3,
            "p": 1,
            "ps": 500,
            "rules": [
                {
                    "key": "java:S001",
                    "name": "Critical rule",
                    "lang": "java",
                    "type": "BUG",
                    "severity": "CRITICAL",
                    "htmlDesc": "",
                    "tags": [],
                    "sysTags": [],
                    "createdAt": "",
                    "params": []
                },
                {
                    "key": "java:S002",
                    "name": "Major rule",
                    "lang": "java",
                    "type": "CODE_SMELL",
                    "severity": "MAJOR",
                    "htmlDesc": "",
                    "tags": [],
                    "sysTags": [],
                    "createdAt": "",
                    "params": []
                },
                {
                    "key": "java:S003",
                    "name": "Another critical",
                    "lang": "java",
                    "type": "VULNERABILITY",
                    "severity": "CRITICAL",
                    "htmlDesc": "",
                    "tags": [],
                    "sysTags": [],
                    "createdAt": "",
                    "params": []
                }
            ]
        }

        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=response),
            raise_for_status=Mock()
        ))

        result = analyzer.get_rules_by_severity("CRITICAL")

        assert len(result) == 2
        assert all(rule['severity'] == 'CRITICAL' for rule in result)
        assert result[0]['key'] == 'java:S001'
        assert result[1]['key'] == 'java:S003'

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_get_rules_by_severity_no_matches(self, mock_sleep, analyzer):
        """Test severity filter with no matches."""
        response = {
            "total": 1,
            "p": 1,
            "ps": 500,
            "rules": [{
                "key": "java:S001",
                "name": "Major rule",
                "lang": "java",
                "type": "CODE_SMELL",
                "severity": "MAJOR",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            }]
        }

        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=response),
            raise_for_status=Mock()
        ))

        result = analyzer.get_rules_by_severity("BLOCKER")

        assert len(result) == 0


# ============================================================================
# EXPORT RULES TO JSON TESTS
# ============================================================================


class TestExportRulesToJson:
    """Tests for export_rules_to_json method."""

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    @patch('builtins.open', new_callable=mock_open)
    def test_export_rules_to_json_success(self, mock_file, mock_sleep, analyzer):
        """Test successful JSON export."""
        response = {
            "total": 1,
            "p": 1,
            "ps": 500,
            "rules": [{
                "key": "java:S001",
                "name": "Test rule",
                "lang": "java",
                "type": "BUG",
                "severity": "MAJOR",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            }]
        }

        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=response),
            raise_for_status=Mock()
        ))

        analyzer.export_rules_to_json("output.json")

        # Verify file was opened for writing
        mock_file.assert_called_once_with("output.json", "w", encoding="utf-8")

        # Verify json.dump was called (indirectly through write calls)
        handle = mock_file()
        assert handle.write.called

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    @patch('builtins.open', new_callable=mock_open)
    def test_export_rules_to_json_with_languages(self, mock_file, mock_sleep, analyzer):
        """Test JSON export with language filter."""
        response = {
            "total": 1,
            "p": 1,
            "ps": 500,
            "rules": [{
                "key": "python:S001",
                "name": "Python rule",
                "lang": "python",
                "type": "BUG",
                "severity": "MAJOR",
                "htmlDesc": "",
                "tags": [],
                "sysTags": [],
                "createdAt": "",
                "params": []
            }]
        }

        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=response),
            raise_for_status=Mock()
        ))

        analyzer.export_rules_to_json("output.json", languages=['python'])

        # Verify language filter was used
        call_args = analyzer.session.get.call_args
        assert call_args[1]['params']['languages'] == 'python'


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple methods."""

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_full_workflow(self, mock_sleep, analyzer):
        """Test complete workflow from fetch to process."""
        response = {
            "total": 2,
            "p": 1,
            "ps": 500,
            "rules": [
                {
                    "key": "java:S1066",
                    "name": "Collapsible if statements",
                    "lang": "java",
                    "type": "CODE_SMELL",
                    "severity": "MAJOR",
                    "status": "READY",
                    "htmlDesc": "<p>Test description</p>",
                    "tags": ["clumsy"],
                    "sysTags": [],
                    "createdAt": "2013-01-21T15:45:00+0100",
                    "params": []
                },
                {
                    "key": "python:S2259",
                    "name": "Null pointers should not be dereferenced",
                    "lang": "python",
                    "type": "BUG",
                    "severity": "CRITICAL",
                    "status": "READY",
                    "htmlDesc": "<p>Null pointer issue</p>",
                    "tags": [],
                    "sysTags": [],
                    "createdAt": "2015-01-21T15:45:00+0100",
                    "params": []
                }
            ]
        }

        analyzer.session.get = Mock(return_value=Mock(
            json=Mock(return_value=response),
            raise_for_status=Mock()
        ))

        # Fetch all rules
        result = analyzer.fetch_all_rules()

        # Verify structure
        assert 'rules' in result
        assert 'metadata' in result
        assert len(result['rules']) == 2

        # Verify metadata
        metadata = result['metadata']
        assert metadata['total_rules'] == 2
        assert metadata['languages']['java'] == 1
        assert metadata['languages']['python'] == 1
        assert metadata['categories']['CODE_SMELL'] == 1
        assert metadata['categories']['BUG'] == 1
        assert metadata['severities']['MAJOR'] == 1
        assert metadata['severities']['CRITICAL'] == 1

        # Verify processed rules
        java_rule = result['rules']['java:S1066']
        assert java_rule['root_cause']
        assert java_rule['how_to_fix']
        assert 'description' in java_rule['how_to_fix']
        assert 'steps' in java_rule['how_to_fix']
        assert 'priority' in java_rule['how_to_fix']
        assert 'effort' in java_rule['how_to_fix']

        python_rule = result['rules']['python:S2259']
        assert 'null' in python_rule['root_cause'].lower()
        assert python_rule['how_to_fix']['priority'] == 'High'


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_rule_with_all_empty_strings(self, analyzer):
        """Test processing rule with all empty strings."""
        rule = {
            "key": "test:S001",
            "name": "",
            "lang": "",
            "type": "",
            "severity": "",
            "htmlDesc": "",
            "tags": [],
            "sysTags": [],
            "createdAt": "",
            "params": []
        }

        result = analyzer._process_rules([rule])
        processed = result['rules']['test:S001']

        assert processed['name'] == ''
        assert processed['language'] == ''
        assert processed['category'] == ''
        assert processed['severity'] == ''

    def test_rule_with_special_characters(self, analyzer):
        """Test rule with special characters in fields."""
        rule = {
            "key": "test:S<>001",
            "name": "Rule with <special> & \"characters\"",
            "lang": "java",
            "type": "CODE_SMELL",
            "severity": "MAJOR",
            "htmlDesc": "<p>Description with &amp; special &lt;chars&gt;</p>",
            "tags": ["tag-with-dashes", "tag_with_underscores"],
            "sysTags": [],
            "createdAt": "",
            "params": []
        }

        result = analyzer._process_rules([rule])
        processed = result['rules']['test:S<>001']
        assert processed['name'] == 'Rule with <special> & "characters"'
        assert "<" not in processed['description']  # HTML entities cleaned

    def test_very_long_description(self, analyzer):
        """Test rule with very long description."""
        long_text = "x" * 2000
        rule = {
            "key": "test:S001",
            "name": "Test",
            "lang": "java",
            "type": "BUG",
            "severity": "MAJOR",
            "htmlDesc": f"<p>{long_text}</p>",
            "tags": [],
            "sysTags": [],
            "createdAt": "",
            "params": []
        }

        result = analyzer._process_rules([rule])
        processed = result['rules']['test:S001']

        assert len(processed['description']) <= 1003  # Truncated
        assert processed['description'].endswith("...")

    def test_multiple_matching_keywords(self, analyzer):
        """Test rule matching multiple keyword patterns."""
        rule = {
            "key": "test:S001",
            "name": "Unused null pointer in SQL query",
            "lang": "java",
            "type": "BUG",
            "severity": "CRITICAL",
            "htmlDesc": "Complex issue with unused, null, and sql",
            "tags": [],
            "sysTags": [],
            "createdAt": "",
            "params": []
        }

        # Should match first pattern (unused)
        cause = analyzer._infer_root_cause(rule)
        assert "unused" in cause.lower()

    @patch('devdox_ai_sonar.services.rule_analyzer.time.sleep')
    def test_api_rate_limiting(self, mock_sleep, analyzer):
        """Test that rate limiting is applied between requests."""
        response = {
            "total": 1500,
            "p": 1,
            "ps": 500,
            "rules": [{"key": f"rule:{i}", "name": f"Rule {i}", "lang": "java", "type": "BUG", "severity": "MAJOR",
                       "htmlDesc": "", "tags": [], "sysTags": [], "createdAt": "", "params": []} for i in range(500)]
        }

        # Mock three pages
        analyzer.session.get = Mock(side_effect=[
            Mock(json=Mock(return_value={**response, "p": 1}), raise_for_status=Mock()),
            Mock(json=Mock(return_value={**response, "p": 2}), raise_for_status=Mock()),
            Mock(json=Mock(return_value={**response, "p": 3}), raise_for_status=Mock())
        ])

        analyzer.fetch_all_rules()

        # Verify sleep was called between pages
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(0.1)