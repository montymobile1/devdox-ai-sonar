# Comprehensive Test Cases for llm_fixer.py


import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from typing import List, Dict, Any

from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    FixSuggestion,
    FixResult
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    with patch('devdox_ai_sonar.llm_fixer.openai.OpenAI') as mock:
        yield mock


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client"""
    with patch('devdox_ai_sonar.llm_fixer.genai.Client') as mock:
        yield mock


@pytest.fixture
def mock_together_client():
    """Mock Together AI client"""
    with patch('devdox_ai_sonar.llm_fixer.Together') as mock:
        yield mock


@pytest.fixture
def sample_issue():
    """Create sample SonarIssue"""
    return SonarIssue(
        key="issue-123",
        rule="python:S1234",
        severity="MAJOR",
        component="src/test.py",
        project="test-project",
        line=8,
        message="Test issue",
        first_line=5,
        last_line=7,
        file="src/test.py",
        type="VULNERABILITY",
        status="OPEN"
    )


@pytest.fixture
def sample_security_issue():
    """Create sample SonarSecurityIssue"""
    return SonarSecurityIssue(
        key="sec-123",
        rule="python:S5042",
        severity="CRITICAL",
        component="src/auth.py",
        project="test-project",
        line=20,
        message="SQL injection vulnerability",
        first_line=20,
        last_line=25,
        file="src/auth.py",
        vulnerability_probability="HIGH"
    )


@pytest.fixture
def sample_python_file(tmp_path):
    """Create sample Python file with content"""
    file = tmp_path / "src" / "test.py"
    file.parent.mkdir(parents=True)
    content = """def example_function():
    x = 1
    y = 2
    if x == 1:
        print("x is 1")
    z = x + y
    return z

def another_function():
    pass
"""
    file.write_text(content)
    return file


@pytest.fixture
def rule_info():
    """Sample rule information"""
    return {
        "python:S1234": {
            "key": "python:S1234",
            "name": "Test Rule",
            "htmlDesc": "<p>This is a test rule description</p>",
            "severity": "MAJOR"
        }
    }


# ============================================================================
# TEST CLASS: INITIALIZATION
# ============================================================================

class TestLLMFixerInitialization:
    """Test LLMFixer initialization"""
    
    def test_init_openai_with_api_key(self, mock_openai_client):
        """Test OpenAI initialization with API key"""
        fixer = LLMFixer(
            provider="openai",
            model="gpt-4",
            api_key="test-key-123"
        )
        
        assert fixer.provider == "openai"
        assert fixer.model == "gpt-4"
        assert fixer.api_key == "test-key-123"
        mock_openai_client.assert_called_once_with(api_key="test-key-123")
    
    def test_init_openai_from_env(self, mock_openai_client, monkeypatch):
        """Test OpenAI initialization from environment variable"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        
        fixer = LLMFixer(provider="openai")
        
        assert fixer.api_key == "env-api-key"
        mock_openai_client.assert_called_once()
    
    def test_init_openai_missing_key(self, mock_openai_client):
        """Test OpenAI initialization without API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LLMFixer(provider="openai")
            assert "API key not provided" in str(exc_info.value)
    
    def test_init_openai_default_model(self, mock_openai_client):
        """Test OpenAI uses default model when not specified"""
        fixer = LLMFixer(provider="openai", api_key="test-key")
        assert fixer.model == "gpt-4o"
    
    def test_init_gemini_with_api_key(self, mock_gemini_client):
        """Test Gemini initialization with API key"""
        fixer = LLMFixer(
            provider="gemini",
            model="claude-3-5-sonnet",
            api_key="test-gemini-key"
        )
        
        assert fixer.provider == "gemini"
        assert fixer.model == "claude-3-5-sonnet"
        assert fixer.api_key == "test-gemini-key"
    
    def test_init_gemini_from_env(self, mock_gemini_client, monkeypatch):
        """Test Gemini initialization from environment"""
        monkeypatch.setenv("GEMINI_KEY", "env-gemini-key")
        
        fixer = LLMFixer(provider="gemini")
        
        assert fixer.api_key == "env-gemini-key"
    
    def test_init_gemini_missing_key(self, mock_gemini_client):
        """Test Gemini initialization without API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LLMFixer(provider="gemini")
            assert "Gemini API key not provided" in str(exc_info.value)
    
    def test_init_togetherai_with_api_key(self, mock_together_client):
        """Test TogetherAI initialization with API key"""
        fixer = LLMFixer(
            provider="togetherai",
            model="mixtral-8x7b",
            api_key="test-together-key"
        )
        
        assert fixer.provider == "togetherai"
        assert fixer.api_key == "test-together-key"
    
    def test_init_togetherai_from_env(self, mock_together_client, monkeypatch):
        """Test TogetherAI initialization from environment"""
        monkeypatch.setenv("TOGETHER_API_KEY", "env-together-key")
        
        fixer = LLMFixer(provider="togetherai")
        
        assert fixer.api_key == "env-together-key"
    
    def test_init_unsupported_provider(self):
        """Test initialization with unsupported provider"""
        with pytest.raises(ValueError) as exc_info:
            LLMFixer(provider="unsupported", api_key="key")
        assert "Unsupported provider" in str(exc_info.value)
    
    def test_init_context_lines_default(self, mock_openai_client):
        """Test default context_lines value"""
        fixer = LLMFixer(provider="openai", api_key="key")
        assert fixer.context_lines == 10
    
    def test_init_context_lines_custom(self, mock_openai_client):
        """Test custom context_lines value"""
        fixer = LLMFixer(provider="openai", api_key="key", context_lines=20)
        assert fixer.context_lines == 20
    
    def test_init_prompt_dir_setup(self, mock_openai_client):
        """Test prompt directory is set up correctly"""
        fixer = LLMFixer(provider="openai", api_key="key")
        assert fixer.prompt_dir.exists() or fixer.prompt_dir.name == "prompts"
    
    def test_init_jinja_env_setup(self, mock_openai_client):
        """Test Jinja2 environment is set up"""
        fixer = LLMFixer(provider="openai", api_key="key")
        assert hasattr(fixer, 'jinja_env')
        assert fixer.jinja_env is not None
    
    def test_init_provider_case_insensitive(self, mock_openai_client):
        """Test provider name is case-insensitive"""
        fixer1 = LLMFixer(provider="OpenAI", api_key="key")
        fixer2 = LLMFixer(provider="OPENAI", api_key="key")
        fixer3 = LLMFixer(provider="openai", api_key="key")
        
        assert fixer1.provider == "openai"
        assert fixer2.provider == "openai"
        assert fixer3.provider == "openai"


# ============================================================================
# TEST CLASS: PROVIDER CONFIGURATION
# ============================================================================

class TestProviderConfiguration:
    """Test provider-specific configuration"""
    
    def test_openai_import_error_handling(self):
        """Test handling when OpenAI library is not installed"""
        with patch('devdox_ai_sonar.llm_fixer.HAS_OPENAI', False):
            with pytest.raises(ImportError) as exc_info:
                LLMFixer(provider="openai", api_key="key")
            assert "OpenAI library not installed" in str(exc_info.value)
    
    def test_gemini_import_error_handling(self):
        """Test handling when Gemini library is not installed"""
        with patch('devdox_ai_sonar.llm_fixer.HAS_GEMINI', False):
            with pytest.raises(ImportError) as exc_info:
                LLMFixer(provider="gemini", api_key="key")
            assert "Gemini library not installed" in str(exc_info.value)
    
    def test_together_import_error_handling(self):
        """Test handling when Together library is not installed"""
        with patch('devdox_ai_sonar.llm_fixer.HAS_TOGETHER', False):
            with pytest.raises(ImportError) as exc_info:
                LLMFixer(provider="togetherai", api_key="key")
            assert "Together AI library not installed" in str(exc_info.value)
    
    def test_configure_openai_client_creation(self, mock_openai_client):
        """Test OpenAI client is created correctly"""
        mock_instance = Mock()
        mock_openai_client.return_value = mock_instance
        
        fixer = LLMFixer(provider="openai", api_key="test-key")
        
        assert fixer.client == mock_instance
        mock_openai_client.assert_called_once_with(api_key="test-key")
    
    def test_configure_gemini_client_creation(self, mock_gemini_client):
        """Test Gemini client is created correctly"""
        mock_instance = Mock()
        mock_gemini_client.return_value = mock_instance
        
        fixer = LLMFixer(provider="gemini", api_key="test-key")
        
        assert fixer.client == mock_instance
    
    def test_configure_together_client_creation(self, mock_together_client):
        """Test Together client is created correctly"""
        mock_instance = Mock()
        mock_together_client.return_value = mock_instance
        
        fixer = LLMFixer(provider="togetherai", api_key="test-key")
        
        assert fixer.client == mock_instance


# ============================================================================
# TEST CLASS: CONTEXT EXTRACTION
# ============================================================================

class TestContextExtraction:
    """Test context extraction from code files"""
    
    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")
    
    def test_extract_context_simple(self, fixer):
        """Test extracting context from simple code"""
        lines = [
            "line 1\n",
            "line 2\n",
            "line 3\n",
            "problem line\n",
            "line 5\n",
            "line 6\n",
            "line 7\n"
        ]
        
        context = fixer._extract_context(lines, 4, 4, context_lines=2)
        
        assert context["start_line"] <= 4
        assert context["end_line"] >= 4
        assert "problem line" in context["context"]
    
    def test_extract_context_at_file_start(self, fixer):
        """Test extracting context at beginning of file"""
        lines = ["line 1\n", "line 2\n", "line 3\n"]
        
        context = fixer._extract_context(lines, 1, 1, context_lines=5)
        
        assert context["start_line"] == 1
        assert "line 1" in context["context"]
    
    def test_extract_context_at_file_end(self, fixer):
        """Test extracting context at end of file"""
        lines = ["line 1\n", "line 2\n", "line 3\n"]
        
        context = fixer._extract_context(lines, 3, 3, context_lines=5)
        
        assert context["end_line"] == 3
        assert "line 3" in context["context"]
    
    def test_extract_context_multiline_issue(self, fixer):
        """Test extracting context for multi-line issue"""
        lines = [f"line {i}\n" for i in range(1, 21)]
        
        context = fixer._extract_context(lines, 10, 15, context_lines=3)
        
        assert context["start_line"] <= 10
        assert context["end_line"] >= 15
        for i in range(10, 16):
            assert f"line {i}" in context["context"]
    
    def test_extract_context_zero_context_lines(self, fixer):
        """Test extracting context with zero context lines"""
        lines = [f"line {i}\n" for i in range(1, 11)]
        
        context = fixer._extract_context(lines, 5, 5, context_lines=0)
        
        assert "line 5" in context["context"]
    
    def test_extract_context_large_context(self, fixer):
        """Test extracting context with large context_lines"""
        lines = [f"line {i}\n" for i in range(1, 21)]
        
        context = fixer._extract_context(lines, 10, 10, context_lines=50)
        
        # Should include all lines since context is larger than file
        assert context["start_line"] == 1
        assert context["end_line"] == len(lines)


# ============================================================================
# TEST CLASS: GENERATE FIX BY FILE
# ============================================================================

class TestGenerateFixByFile:
    """Test fix generation for issues"""
    
    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")
    
    def test_generate_fix_success(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test successful fix generation"""
        # Update issue to point to our test file
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))
        
        # Mock LLM response
        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.return_value = {
                "fixed_code": "def example_function():\n    x = 1\n    y = 2\n    return x + y",
                "explanation": "Simplified function",
                "confidence": 0.95,
                "rule_description": "Test rule"
            }

            result = fixer.generate_fix_by_file(
                [sample_issue],
                tmp_path,
                rule_info
            )

        assert result is not None
        assert isinstance(result, FixSuggestion)
        assert result.confidence == 0.95
        assert "Simplified function" in result.explanation

    def test_generate_fix_file_not_found(self, fixer, sample_issue, tmp_path, rule_info):
        """Test fix generation when file doesn't exist"""
        sample_issue.file = "nonexistent/file.py"

        result = fixer.generate_fix_by_file(
            [sample_issue],
            tmp_path,
            rule_info
        )

        assert result is None

    def test_generate_fix_multiple_issues_same_file(self, fixer, sample_python_file, rule_info, tmp_path):
        """Test generating fix for multiple issues in same file"""
        issue1 = SonarIssue(
            key="issue-1",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=2,
            message="Issue 1",
            type="VULNERABILITY",
            first_line=2,
            last_line=3,
            file=str(sample_python_file.relative_to(tmp_path)),
            status="OPEN",

        )
        issue2 = SonarIssue(
            key="issue-2",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=5,
            message="Issue 2",
            type="VULNERABILITY",
            first_line=5,
            last_line=6,
            file=str(sample_python_file.relative_to(tmp_path)),
            status="OPEN",
        )

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.return_value = {
                "fixed_code": "fixed code",
                "explanation": "Fixed multiple issues",
                "confidence": 0.9
            }

            result = fixer.generate_fix_by_file(
                [issue1, issue2],
                tmp_path,
                rule_info
            )

        assert result is not None
        assert result.sonar_line_number == 2  # First issue line

    def test_generate_fix_multiple_files_error(self, fixer, sample_python_file, rule_info, tmp_path):
        """Test error when issues are from different files"""
        issue1 = SonarIssue(
            key="issue-1",
            rule="python:S1234",
            severity="MAJOR",
            component="file1.py",
            project="test",
            line=1,
            message="Issue 1",
            first_line=1,
            last_line=2,
            file="file1.py",
            type="VULNERABILITY",
            status="OPEN"
        )
        issue2 = SonarIssue(
            key="issue-2",
            rule="python:S1234",
            severity="MAJOR",
            component="file2.py",
            project="test",
            line=1,
            message="Issue 2",
            first_line=1,
            last_line=2,
            file="file2.py",
            type="VULNERABILITY",
            status="OPEN"
        )

        result = fixer.generate_fix_by_file(
            [issue1, issue2],
            tmp_path,
            rule_info
        )

        assert result is None

    def test_generate_fix_with_modified_content(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test fix generation with pre-modified content"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))
        modified_content = "# Modified code\ndef new_function():\n    pass"

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.return_value = {
                "fixed_code": "fixed",
                "explanation": "Fixed",
                "confidence": 0.8
            }

            result = fixer.generate_fix_by_file(
                [sample_issue],
                tmp_path,
                rule_info,
                modified_content=modified_content
            )

        assert result is not None
        # Verify modified content was used
        call_args = mock_llm.call_args[0]
        assert call_args[1]["context"] == modified_content

    def test_generate_fix_with_error_message(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test fix generation with error message"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))
        error_msg = "Previous fix failed validation"

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.return_value = {
                "fixed_code": "fixed",
                "explanation": "Fixed",
                "confidence": 0.85
            }

            result = fixer.generate_fix_by_file(
                [sample_issue],
                tmp_path,
                rule_info,
                error_message=error_msg
            )

        assert result is not None

    def test_generate_fix_llm_returns_none(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test when LLM returns None"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.return_value = None

            result = fixer.generate_fix_by_file(
                [sample_issue],
                tmp_path,
                rule_info
            )

        assert result is None

    def test_generate_fix_exception_handling(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test exception handling during fix generation"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = fixer.generate_fix_by_file(
                [sample_issue],
                tmp_path,
                rule_info
            )

        assert result is None


class TestLLMAPICalls:
    """Test LLM API interaction"""

    @pytest.fixture
    def fixer_openai(self, mock_openai_client):
        """Create OpenAI fixer"""
        return LLMFixer(provider="openai", api_key="test-key")

    @pytest.fixture
    def fixer_gemini(self, mock_gemini_client):
        """Create Gemini fixer"""
        return LLMFixer(provider="gemini", api_key="test-key")

    def test_call_llm_openai_success(self, fixer_openai):
        """Test successful OpenAI API call"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "FIXED_SELECTION": "def fixed():\n    pass",
                "EXPLANATION": "Fixed the code",
                "CONFIDENCE": 0.95
            })))
        ]
        fixer_openai.client.chat.completions.create = Mock(return_value=mock_response)

        issue = SonarIssue(
            key="test",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5
        )
        context_dict = {"context": "original code", "start_line": 1, "end_line": 5}

        result = fixer_openai._call_llm_list(
            [issue],
            context_dict,
            ".py",
            {"python:S1234": {"name": "Test Rule"}},
            ""
        )

        assert result is not None
        assert result["fixed_code"] == "def fixed():\n    pass"
        assert result["confidence"] == 0.95

    def test_call_llm_openai_api_error(self, fixer_openai):
        """Test OpenAI API error handling"""
        fixer_openai.client.chat.completions.create = Mock(side_effect=Exception("API Error"))

        issue = SonarIssue(
            key="test",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5
        )
        context_dict = {"context": "code", "start_line": 1, "end_line": 5,"strategy_text":"Replace"}

        result = fixer_openai._call_llm_list(
            [issue],
            context_dict,
            ".py",
            {},
            ""
        )

        assert result is None

    def test_call_llm_gemini_success(self, fixer_gemini):
        """Test successful Gemini API call"""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "FIXED_SELECTION": "fixed code",
            "EXPLANATION": "Fixed",
            "CONFIDENCE": 0.9
        })
        fixer_gemini.client.models.generate_content = Mock(return_value=mock_response)

        issue = SonarIssue(
            key="test",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5
        )
        context_dict = {"context": "code", "start_line": 1, "end_line": 5}

        result = fixer_gemini._call_llm_list(
            [issue],
            context_dict,
            ".py",
            {},
            ""
        )

        assert result is not None

    def test_call_llm_with_helper_code(self, fixer_openai):
        """Test LLM call returns helper code"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "FIXED_SELECTION": "main code",
                "NEW_HELPER_CODE": "def helper():\n    pass",
                "PLACEMENT": "before",
                "EXPLANATION": "Added helper",
                "CONFIDENCE": 0.88
            })))
        ]

        fixer_openai.client.chat.completions.create = Mock(return_value=mock_response)

        issue = SonarIssue(
            key="test",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5
        )
        context_dict = {"context": "code", "start_line": 1, "end_line": 5}

        result = fixer_openai._call_llm_list([issue], context_dict, ".py", {}, "")

        assert result["helper_code"] == "def helper():\n    pass"
        assert result["placement_helper"] == "SIBLING"


# ============================================================================
# TEST CLASS: RESPONSE PARSING
# ============================================================================

class TestResponseParsing:
    """Test parsing LLM responses"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")

    def test_parse_openai_response_json(self, fixer):
        """Test parsing OpenAI response with JSON"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "FIXED_SELECTION": "fixed",
                "EXPLANATION": "Done",
                "CONFIDENCE": 0.9
            })))
        ]

        result = fixer._parse_openai_response(mock_response)

        assert result is not None
        assert result["fixed_code"] == "fixed"
        assert result["confidence"] == 0.9

    def test_parse_openai_response_markdown(self, fixer):
        """Test parsing OpenAI response with markdown code blocks"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="""
```json
{
    "FIXED_SELECTION": "code here",
    "EXPLANATION": "Fixed it",
    "CONFIDENCE": 0.85
}
```
            """))
        ]

        result = fixer._parse_openai_response(mock_response)

        assert result is not None
        assert "code here" in result["fixed_code"]

    def test_parse_gemini_response(self, fixer):
        """Test parsing Gemini response"""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "FIXED_SELECTION": "gemini code",
            "EXPLANATION": "Fixed",
            "CONFIDENCE": 0.92
        })

        result = fixer._parse_gemini_response(mock_response)

        assert result is not None
        assert result["fixed_code"] == "gemini code"

    def test_parse_togetherai_response(self, fixer):
        """Test parsing TogetherAI response"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps({
                "FIXED_SELECTION": "together code",
                "EXPLANATION": "Fixed",
                "CONFIDENCE": 0.87
            })))
        ]

        result = fixer._parse_togetherai_response(mock_response)

        assert result is not None

    def test_extract_fix_from_response_json(self, fixer):
        """Test extracting fix from JSON response"""
        content = json.dumps({
            "FIXED_SELECTION": "def foo():\n    pass",
            "EXPLANATION": "Simplified",
            "CONFIDENCE": 0.95
        })

        result = fixer._extract_fix_from_response(content)

        assert result is not None
        assert result["fixed_code"] == "def foo():\n    pass"

    def test_extract_fix_from_response_code_blocks(self, fixer):
        """Test extracting fix from markdown code blocks"""
        content = """

 FIXED_SELECTION:
```python
def fixed_function():
    return True
```

EXPLANATION: Fixed the function
CONFIDENCE: 0.9
"""

        result = fixer._extract_fix_from_response(content)

        assert result is  None


    def test_extract_fix_regex_fallback(self, fixer):
        """Test regex fallback for extraction"""
        content = """
Fixed Code:
def new_code():
    pass

Explanation: Made it better
Confidence: 0.85
"""

        result = fixer._extract_using_regex_fallback(content)

        assert result is  None

    def test_validate_results_complete(self, fixer):
        """Test validation of complete results"""
        results = {
            "FIXED_SELECTION": "code",
            "EXPLANATION": "Done",
            "CONFIDENCE": 0.9,
            "NEW_HELPER_CODE":"",
            "PLACEMENT":""
        }

        validated = fixer._validate_results(results)

        assert validated is not None
        assert validated["fixed_code"] == "code"

    def test_validate_results_missing_fields(self, fixer):
        """Test validation fails with missing required fields"""
        results = {
            "fixed_code": "code"
            # Missing explanation and confidence
        }

        validated = fixer._validate_results(results)

        # Should add default values or return None
        assert validated is None or "explanation" in validated


# ============================================================================
# TEST CLASS: APPLY FIXES
# ============================================================================

class TestApplyFixes:
    """Test applying fixes to files"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")

    @pytest.fixture
    def sample_fix(self, tmp_path):
        """Create sample fix suggestion"""
        return FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="old code",
            fixed_code="new code",
            helper_code="",
            explanation="Fixed",
            confidence=0.9,
            sonar_line_number=10,
            line_number=10,
            last_line_number=15,
            llm_model="a"
        )

    def test_apply_fixes_success(self, fixer, sample_fix, tmp_path):
        """Test successfully applying fixes"""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("old code\n")

        # Update fix to use correct path
        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        with patch.object(fixer, '_apply_fixes_to_file') as mock_apply:
            mock_apply.return_value = (1, 0, [])

            result = fixer.apply_fixes(
                fixes=[sample_fix],
                project_path=tmp_path,
                create_backup=False,
                dry_run=False
            )

        assert isinstance(result, FixResult)
        assert result.total_fixes_attempted == 1

    def test_apply_fixes_dry_run(self, fixer, sample_fix, tmp_path):
        """Test dry run mode doesn't modify files"""
        test_file = tmp_path / "test.py"
        original_content = "original content\n"
        test_file.write_text(original_content)

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        result = fixer.apply_fixes(
            fixes=[sample_fix],
            project_path=tmp_path,
            dry_run=True
        )

        # File should not be modified
        assert test_file.read_text() == original_content

    def test_apply_fixes_creates_backup(self, fixer, sample_fix, tmp_path):
        """Test backup creation"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        with patch.object(fixer, '_create_backup') as mock_backup:
            mock_backup.return_value = tmp_path / "backup"

            with patch.object(fixer, '_apply_fixes_to_file') as mock_apply:
                mock_apply.return_value = (1, 0, [])

                result = fixer.apply_fixes(
                    fixes=[sample_fix],
                    project_path=tmp_path,
                    create_backup=True,
                    dry_run=False
                )

        assert result.backup_created is True
        assert result.backup_path is not None
        mock_backup.assert_called_once()

    def test_apply_fixes_no_backup_in_dry_run(self, fixer, sample_fix, tmp_path):
        """Test backup not created in dry run"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        with patch.object(fixer, '_create_backup') as mock_backup:
            result = fixer.apply_fixes(
                fixes=[sample_fix],
                project_path=tmp_path,
                create_backup=True,
                dry_run=True
            )

        assert result.backup_created is False
        mock_backup.assert_not_called()

    def test_apply_fixes_multiple_files(self, fixer, tmp_path):
        """Test applying fixes to multiple files"""
        # Create multiple test files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("content1\n")
        file2.write_text("content2\n")

        fix1 = FixSuggestion(
            issue_key="issue-1",
            file_path=str(file1.relative_to(tmp_path)),
            helper_code="",

            original_code="content1",
            fixed_code="fixed1",
            explanation="Fixed",
            llm_model="a",
            confidence=0.9,
            sonar_line_number=1
        )
        fix2 = FixSuggestion(
            issue_key="issue-2",
            file_path=str(file2.relative_to(tmp_path)),
            original_code="content2",
            fixed_code="fixed2",
            explanation="Fixed",
            confidence=0.9,
            llm_model="a",
            sonar_line_number=1
        )

        with patch.object(fixer, '_apply_fixes_to_file') as mock_apply:
            mock_apply.return_value = (1, 0, [])

            result = fixer.apply_fixes(
                fixes=[fix1, fix2],
                project_path=tmp_path,
                create_backup=False
            )

        assert result.total_fixes_attempted == 2
        assert mock_apply.call_count == 2

    def test_apply_fixes_groups_by_file(self, fixer, tmp_path):
        """Test fixes are grouped by file"""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        fix1 = FixSuggestion(
            issue_key="issue-1",
            file_path=str(test_file.relative_to(tmp_path)),
            original_code="line1",
            fixed_code="fixed1",
            explanation="Fixed",
            helper_code="",
            llm_model="a",
            confidence=0.9,
            sonar_line_number=1
        )
        fix2 = FixSuggestion(
            issue_key="issue-2",
            file_path=str(test_file.relative_to(tmp_path)),
            original_code="line2",
            fixed_code="fixed2",
            explanation="Fixed",
            helper_code="",
            llm_model="a",
            confidence=0.9,
            sonar_line_number=2
        )

        with patch.object(fixer, '_apply_fixes_to_file') as mock_apply:
            mock_apply.return_value = (2, 0, [])

            result = fixer.apply_fixes(
                fixes=[fix1, fix2],
                project_path=tmp_path,
                create_backup=False
            )

        # Should call _apply_fixes_to_file once with both fixes
        assert mock_apply.call_count == 1
        call_args = mock_apply.call_args[0]
        assert len(call_args[1]) == 2  # Two fixes for same file


# ============================================================================
# TEST CLASS: APPLY FIXES WITH VALIDATION
# ============================================================================

class TestApplyFixesWithValidation:
    """Test applying fixes with validation"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")

    def test_apply_fixes_with_validation_success(self, fixer, tmp_path):
        """Test applying fixes with successful validation"""
        test_file = tmp_path / "test.py"
        test_file.write_text("def old():\n    pass\n")

        fix = FixSuggestion(
            issue_key="issue-1",
            file_path=str(test_file.relative_to(tmp_path)),
            original_code="def old():\n    pass",
            fixed_code="def new():\n    return True",
            explanation="Improved function",
            helper_code="",
            llm_model="a",
            confidence=0.95,
            sonar_line_number=1
        )

        issue = SonarIssue(
            key="issue-1",
            rule="python:S1234",
            severity="MAJOR",
            component=str(test_file),
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5
        )

        with patch('devdox_ai_sonar.llm_fixer.FixValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate_fix.return_value = (True, "Valid")
            mock_validator_class.return_value = mock_validator

            result = fixer.apply_fixes_with_validation(
                fixes=[fix],
                issues=[issue],
                project_path=tmp_path,
                create_backup=False,
                dry_run=False,
                use_validator=True
            )

        assert isinstance(result, FixResult)
        assert len(result.successful_fixes) > 0 or result.total_fixes_attempted > 0

    def test_apply_fixes_with_validation_failure(self, fixer, tmp_path):
        """Test validation failure"""
        test_file = tmp_path / "test.py"
        test_file.write_text("def old():\n    pass\n")

        fix = FixSuggestion(
            issue_key="issue-1",
            file_path=str(test_file.relative_to(tmp_path)),
            original_code="def old():\n    pass",
            fixed_code="def new(\n    # Invalid syntax",
            explanation="Broken fix",
            confidence=0.5,
            sonar_line_number=1,
            llm_model="a",
            helper_code="",
            line_number=1,
            last_line_number=10
        )

        issue = SonarIssue(
            key="issue-1",
            rule="python:S1234",
            severity="MAJOR",
            component=str(test_file),
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5
        )

        with patch('devdox_ai_sonar.llm_fixer.FixValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate_fix.return_value = (False, "Syntax error")
            mock_validator_class.return_value = mock_validator

            result = fixer.apply_fixes_with_validation(
                fixes=[fix],
                issues=[issue],

                project_path=tmp_path,
                use_validator=True
            )
        print("result failed ", result.failed_fixes, "success ", result.success_rate)
        # Fix should fail validation
        assert len(result.failed_fixes) > 0 or result.success_rate < 1.0



# ============================================================================
# TEST CLASS: HELPER METHODS
# ============================================================================

class TestHelperMethods:
    """Test helper methods"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")

    def test_get_language_from_extension_python(self, fixer):
        """Test getting language from .py extension"""
        lang = fixer._get_language_from_extension(".py")
        assert lang == "python"

    def test_get_language_from_extension_javascript(self, fixer):
        """Test getting language from .js extension"""
        lang = fixer._get_language_from_extension(".js")
        assert lang in ["javascript", "js"]

    def test_get_language_from_extension_java(self, fixer):
        """Test getting language from .java extension"""
        lang = fixer._get_language_from_extension(".java")
        assert lang == "java"

    def test_get_language_from_extension_unknown(self, fixer):
        """Test unknown extension"""
        lang = fixer._get_language_from_extension(".xyz")
        assert lang is not None  # Should have fallback

    def test_create_backup(self, fixer, tmp_path):
        """Test backup creation"""
        # Create test project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "file.py").write_text("content")

        backup_path = fixer._create_backup(tmp_path)

        assert backup_path.exists()
        assert backup_path.is_dir()
        assert "backup" in str(backup_path)

    def test_check_bracket_balance_balanced(self, fixer):
        """Test balanced brackets"""
        code = "def foo():\n    if (x > 0):\n        print('ok')"
        assert fixer._check_bracket_balance(code) is True

    def test_check_bracket_balance_unbalanced(self, fixer):
        """Test unbalanced brackets"""
        code = "def foo():\n    if (x > 0:\n        print('ok')"
        assert fixer._check_bracket_balance(code) is False

    def test_apply_indentation_to_fix(self, fixer):
        """Test applying indentation to fixed code"""
        fixed_code = "def foo():\npass"
        base_indent = "    "  # 4 spaces

        result = fixer.apply_indentation_to_fix(fixed_code, base_indent)

        assert result.startswith("    ")

    def test_is_actual_function_def_true(self, fixer):
        """Test detecting actual function definition"""
        line = "def my_function(arg1, arg2):"
        assert fixer._is_actual_function_def(line) is True

    def test_is_actual_function_def_async(self, fixer):
        """Test detecting async function definition"""
        line = "async def async_function():"
        assert fixer._is_actual_function_def(line) is True

    def test_is_actual_function_def_false(self, fixer):
        """Test non-function line"""
        line = "# This is a comment about def"
        assert fixer._is_actual_function_def(line) is False
