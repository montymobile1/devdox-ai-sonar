# Comprehensive Test Cases for llm_fixer.py


import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from typing import List, Dict, Any, Union
import tempfile
import shutil
import re

from devdox_ai_sonar.llm_fixer import (LLMFixer, ContextExtractor, _build_fix_suggestion,
                                       _generate_fix_key,
                                       _extract_problem_lines,
                                       )
from devdox_ai_sonar.services.extractor import (_validate_and_extract_issue_info,
                                                get_content_range,
                                                _find_exact_match,
                                                _find_fuzzy_match,
                                                _find_all_single_line_matches,
                                                _create_result
                                                )
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    FixSuggestion,
    SonarSecurityIssue,
    Severity,
    IssueType,
    Impact,
    FixResult,
    ChangeType,
    BlockType,
    CodeBlock

)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_code_block():
    return CodeBlock(block_name="test",
                     start_line="1",
                     end_line="10",
                     has_changes=True,
                     change_type=ChangeType.FULL_CODE,
                     block_type=BlockType.MODULE,
                     context="new_code"
                     )

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
        component="src/auth.py",
        project="test-project",
        line=20,
        message="SQL injection vulnerability",
        first_line=20,
        last_line=25,
        file="src/auth.py",
        vulnerability_probability="HIGH",
        security_category="SQL_INJECTION",
        status="OPEN"
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
def sample_code():
    """Sample Python code for testing."""
    return """import os
import sys

def calculate_sum(a, b):
    result = a + b
    return result

def calculate_product(a, b):
    result = a * b
    return result

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result
"""

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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
        
        context = fixer._extract_context(lines, 4, 4,[3], context_lines=2)
        
        assert context["start_line"] <= 4
        assert context["end_line"] >= 4
        assert "problem line" in context["context"]
    
    def test_extract_context_at_file_start(self, fixer):
        """Test extracting context at beginning of file"""
        lines = ["line 1\n", "line 2\n", "line 3\n"]
        
        context = fixer._extract_context(lines, 1, 1,[1], context_lines=5)
        
        assert context["start_line"] == 1
        assert "line 1" in context["context"]
    
    def test_extract_context_at_file_end(self, fixer):
        """Test extracting context at end of file"""
        lines = ["line 1\n", "line 2\n", "line 3\n"]
        
        context = fixer._extract_context(lines, 3, 3,[3], context_lines=5)
        
        assert context["end_line"] == 3
        assert "line 3" in context["context"]
    
    def test_extract_context_multiline_issue(self, fixer):
        """Test extracting context for multi-line issue"""
        lines = [f"line {i}\n" for i in range(1, 21)]
        
        context = fixer._extract_context(lines, 10, 15,[13,15], context_lines=3)
        
        assert context["start_line"] <= 10
        assert context["end_line"] >= 15
        for i in range(10, 16):
            assert f"line {i}" in context["context"]
    
    def test_extract_context_zero_context_lines(self, fixer):
        """Test extracting context with zero context lines"""
        lines = [f"line {i}\n" for i in range(1, 11)]
        
        context = fixer._extract_context(lines, 5, 5, [5],context_lines=0)
        
        assert "line 5" in context["context"]
    
    def test_extract_context_large_context(self, fixer):
        """Test extracting context with large context_lines"""
        lines = [f"line {i}\n" for i in range(1, 21)]
        
        context = fixer._extract_context(lines, 10, 10, [10],context_lines=50)
        
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

    @pytest.mark.skip(reason="Need update")
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
                tmp_path,
                rule_info,
                file_md=str(sample_python_file.relative_to(tmp_path)),
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
            tmp_path,
            rule_info
        )

        assert result is None

    @pytest.mark.skip(reason="Need update")
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
                tmp_path,
                rule_info,
                file_md=str(sample_python_file.relative_to(tmp_path)),
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
            tmp_path,
            rule_info
        )

        assert result is None

    @pytest.mark.skip(reason="Need update")
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
                tmp_path,
                rule_info,
                modified_content=modified_content,
                file_md=str(sample_python_file.relative_to(tmp_path)),
            )

        assert result is not None
        # Verify modified content was used
        call_args = mock_llm.call_args[0]
        assert call_args[1]["context"] == modified_content

    @pytest.mark.skip(reason="Need update")
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
                tmp_path,
                rule_info,
                error_message=error_msg,
                file_md=str(sample_python_file.relative_to(tmp_path)),
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
        return LLMFixer(provider="gemini", api_key="test-key", model="gemini-pro")

    @pytest.mark.skip(reason="Need update")
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

    @pytest.mark.skip(reason="Need update")
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
        context_dict = {"context_dict":{"context": "code", "start_line": 1, "end_line": 5,"strategy_text":"Replace"},
                        "file_lines": [],
                        "problem_line_content": [],
                        }

        result = fixer_openai._call_llm_list(
            [issue],
            context_dict['context_dict'],
            ".py",
            {},
            ""
        )

        assert result is None

    @pytest.mark.skip(reason="Need update")
    def test_call_llm_gemini_success(self, fixer_gemini):
        """Test successful Gemini API call"""

        # Create mock response with proper text attribute
        mock_response = Mock()
        mock_response.text = json.dumps({
            "FIXED_SELECTION": "fixed code",
            "EXPLANATION": "Fixed",
            "CONFIDENCE": 0.9
        })

        # Mock the client.models.generate_content method
        # This matches the new API structure: self.client.models.generate_content(model=..., contents=...)
        mock_models = Mock()
        mock_models.generate_content = Mock(return_value=mock_response)
        fixer_gemini.client.models = mock_models

        # Create test issue
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

        # Execute
        result = fixer_gemini._call_llm_list(
            [issue],
            context_dict,
            ".py",
            {"python:S1234": {"name": "Test Rule"}},
            ""
        )

        # Assertions
        assert result is not None
        assert result["fixed_code"] == "fixed code"
        assert result["explanation"] == "Fixed"
        assert result["confidence"] == 0.9

        # Verify the API was called correctly
        mock_models.generate_content.assert_called_once()
        call_args = mock_models.generate_content.call_args

        # Verify the call had the correct parameters
        assert call_args[1]["model"] == "gemini-pro"  # or fixer_gemini.model
        assert "contents" in call_args[1]

    @pytest.mark.skip(reason="Need update")
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
# TEST CLASS: APPLY FIXES
# ============================================================================

class TestApplyFixes:
    """Test applying fixes to files"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")

    @pytest.fixture
    def sample_fix(self, tmp_path, sample_code_block):
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
            llm_model="a",
            fixed_code_blocks=[sample_code_block]
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

    def test_apply_fixes_multiple_files(self, fixer, tmp_path,sample_code_block):
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
            sonar_line_number=1,
            fixed_code_blocks=[sample_code_block]
        )
        fix2 = FixSuggestion(
            issue_key="issue-2",
            file_path=str(file2.relative_to(tmp_path)),
            original_code="content2",
            fixed_code="fixed2",
            explanation="Fixed",
            confidence=0.9,
            llm_model="a",
            sonar_line_number=1,
                fixed_code_blocks=[sample_code_block],
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

    def test_apply_fixes_groups_by_file(self, fixer, tmp_path,sample_code_block):
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
            sonar_line_number=1,
            fixed_code_blocks=[sample_code_block]
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
            sonar_line_number=2,
            fixed_code_blocks=[sample_code_block]
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

    @pytest.mark.skip(reason="Need update")
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




# ============================================================================
# TEST CLASS: HELPER METHODS
# ============================================================================

class TestHelperMethods:
    """Test helper methods"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")

    @pytest.fixture
    def context_extractor(self):
        return  ContextExtractor(lines=[])

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

    def test_is_actual_function_def_true(self, context_extractor):
        """Test detecting actual function definition"""
        line = "def my_function(arg1, arg2):"
        assert context_extractor._is_actual_function_def(line) is True

    def test_is_actual_function_def_async(self, context_extractor):
        """Test detecting async function definition"""
        line = "async def async_function():"
        assert context_extractor._is_actual_function_def(line) is True

    def test_is_actual_function_def_false(self, context_extractor):
        """Test non-function line"""
        line = "# This is a comment about def"
        assert context_extractor._is_actual_function_def(line) is False

@pytest.mark.skip(reason="Need update")
class TestWriteExplaination:
    # The actual method being tested (copy from your class)
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def fixer(self, mock_openai_client):
        """Create fixer instance"""
        return LLMFixer(provider="openai", api_key="test-key")

    @pytest.fixture
    def mock_jinja_env(self):
        """Mock Jinja2 environment with template"""
        mock_env = Mock()
        mock_template = Mock()
        mock_template.render.return_value = """---

    ## 🔍 Issue: `{rule}`

    **Severity:** {severity}  
    **File:** `{file_path}`  
    **Line:** {line}

    ### ❗ Problem
    {message}

    ### 🧠 Explanation
    {explanation}

    ### 🛠 Suggested Fix
    {suggestion}"""
        mock_env.get_template.return_value = mock_template
        return mock_env

    @pytest.fixture
    def test_instance(self, mock_jinja_env):
        """Create test instance with mocked jinja environment"""
        instance = Mock()
        instance.jinja_env_templates = mock_jinja_env
        instance.write_explaination = self.write_explaination.__get__(instance, type(instance))
        return instance

    # ============================================================================
    # TEST CASES - HAPPY PATH
    # ============================================================================
    @pytest.mark.skip(reason="Need update")
    def test_single_sonar_issue_basic(self, fixer, temp_dir, sample_sonar_issue):
        """Test writing a single SonarIssue with basic data"""
        file_md = temp_dir / "test_output.md"



        fix_response = {
            "explanation": "Print statements are not suitable for production logging",
            "fixed_code": "Use logging.info() instead"
        }

        original_code = "print('Hello World')"

        fixer.write_explaination(file_md, fix_response, [sample_sonar_issue], original_code)

        # Assertions
        assert file_md.exists(), "Output file should be created"
        content = file_md.read_text()
        assert "python:S1481" in content
        assert "MAJOR" in content
        assert "src/example.py" in content

    @pytest.mark.skip(reason="Need update")
    def test_multiple_sonar_issues(self, fixer, temp_dir):
        """Test writing multiple SonarIssues"""
        file_md = temp_dir / "multiple_issues.md"

        issues = [
            SonarIssue(
                key="test-project:src/example.py:python:S1234",
                rule="python:S1234",
                severity=Severity.MAJOR,
                component="src/file1.py",
                project="test",
                first_line=10,
                last_line=10,
                line=10,
                message="Issue 1",
                type=IssueType.CODE_SMELL,
                impact=Impact.MEDIUM,
                file_path="file1.py",
                branch="main",
                status="OPEN",
                creation_date="2024-01-01T10:00:00+0000",
                update_date="2024-01-02T10:00:00+0000",
                tags=["unused", "dead-code"],
            ),
            SonarIssue(
                key="test-project:src/example.py:python:S5678",
                rule="python:S5678",
                severity=Severity.CRITICAL,
                component="src/file2.py",
                project="test",
                first_line=20,
                last_line=20,
                line=20,
                message="Issue 1",
                type=IssueType.CODE_SMELL,
                impact=Impact.MEDIUM,
                file_path="file2.py",
                branch="main",
                status="OPEN",
                creation_date="2024-01-01T10:00:00+0000",
                update_date="2024-01-02T10:00:00+0000",
                tags=["unused", "dead-code"],
            )

        ]

        fix_response = {
            "explanation": "Common explanation for all issues",
            "fixed_code": "Common fix"
        }

        fixer.write_explaination(file_md, fix_response, issues, "")

        content = file_md.read_text()
        assert "python:S1234" in content
        assert "python:S5678" in content
        assert content.count("## 🔍 Issue:") == 2

    def test_sonar_security_issue(self, fixer, temp_dir,sample_security_issue):
        """Test writing SonarSecurityIssue"""
        file_md = temp_dir / "security_issue.md"



        fix_response = {
            "explanation": "Use parameterized queries to prevent SQL injection",
            "fixed_code": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
        }

        fixer.write_explaination(file_md, fix_response, [sample_security_issue], "")

        assert file_md.exists()
        content = file_md.read_text()
        assert "SQL injection vulnerability" in content


    def test_append_to_existing_file(self, fixer, temp_dir, sample_sonar_issue):
        """Test appending to an existing file"""
        file_md = temp_dir / "existing.md"

        # Create file with initial content
        file_md.write_text("# Existing Content\n\n")


        fix_response = {
            "explanation": "New explanation",
            "fixed_code": "New fix"
        }

        fixer.write_explaination(file_md, fix_response, [sample_sonar_issue], "")

        content = file_md.read_text()
        assert "# Existing Content" in content
        assert "python:S1481" in content

    # ============================================================================
    # TEST CASES - EDGE CASES
    # ============================================================================

    def test_empty_issues_list(self, fixer, temp_dir):
        """Test with empty issues list"""
        file_md = temp_dir / "empty.md"

        fix_response = {
            "explanation": "No issues",
            "fixed_code": "No fix needed"
        }

        fixer.write_explaination(file_md, fix_response, [], "")

        # File should be created but contain no issue entries
        assert file_md.exists()
        content = file_md.read_text()
        assert content == "" or content.isspace()

    @pytest.mark.skip(reason="this test is currently broken")
    def test_missing_issue_attributes(self, fixer, temp_dir):
        """Test handling issues with missing attributes"""
        file_md = temp_dir / "missing_attrs.md"

        # Create issue with missing attributes
        issue = Mock()
        del issue.rule
        del issue.severity
        del issue.message
        del issue.file_path
        del issue.line

        fix_response = {
            "explanation": "Test explanation",
            "fixed_code": "Test fix"
        }

        fixer.write_explaination(file_md, fix_response, [issue], "")

        content = file_md.read_text()
        # Should use default values from getattr
        assert "Unknown rule" in content
        assert "UNKNOWN" in content
        assert "No message provided" in content

    def test_missing_fix_response_keys(self, fixer, temp_dir, sample_sonar_issue):
        """Test with missing keys in fix_response"""
        file_md = temp_dir / "missing_keys.md"


        # Empty fix_response
        fix_response = {}

        fixer.write_explaination(file_md, fix_response, [sample_sonar_issue], "")

        content = file_md.read_text()
        assert "No explanation provided" in content
        assert "No suggestion provided" in content

    def test_nested_directory_creation(self, fixer, temp_dir, sample_sonar_issue):
        """Test creating nested directories"""
        file_md = temp_dir / "deep" / "nested" / "path" / "output.md"



        fix_response = {
            "explanation": "Test",
            "fixed_code": "Test"
        }

        fixer.write_explaination(file_md, fix_response, [sample_sonar_issue], "")

        assert file_md.exists()
        assert file_md.parent.exists()


    def test_unicode_content(self, fixer, temp_dir):
        """Test handling Unicode characters"""
        file_md = temp_dir / "unicode.md"

        issue = SonarIssue(
            key="test-project:src/файл.py.py:S1234",
            rule="python:S1234",
            severity=Severity.MAJOR,
            message="Unicode: 你好世界 🚀",
            component="test-project:src/файл.py",
            project="test-project",
            file_path="src/файл.py",
            type=IssueType.CODE_SMELL,
            line=1
        )

        fix_response = {
            "explanation": "Explication en français avec émojis 🎉",
            "fixed_code": "código_español = 'ñ'"
        }

        fixer.write_explaination(file_md, fix_response, [issue], "")

        content = file_md.read_text(encoding="utf-8")
        assert "你好世界" in content
        assert "🚀" in content
        assert "français" in content

    def test_very_long_content(self, fixer, temp_dir):
        """Test with very long content"""
        file_md = temp_dir / "long_content.md"

        issue = SonarIssue(
            key="test-project:src/test.py.py:S1234",
            rule="python:S1234",
            severity=Severity.MAJOR,
            message="A" * 10000,  # Very long message
            file_path="test.py",
            component="test-project:src/test.py",
            project="test-project",

            type=IssueType.CODE_SMELL,
            line=1
        )

        fix_response = {
            "explanation": "B" * 10000,
            "fixed_code": "C" * 10000
        }

        fixer.write_explaination(file_md, fix_response, [issue], "")

        assert file_md.exists()
        content = file_md.read_text()
        assert len(content) > 30000


    # ============================================================================
    # TEST CASES - ERROR HANDLING
    # ============================================================================

    def test_read_only_directory(self, fixer, temp_dir):
        """Test handling read-only directory (permission error)"""
        file_md = temp_dir / "readonly" / "output.md"
        file_md.parent.mkdir(parents=True, exist_ok=True)

        # Make directory read-only
        import os
        os.chmod(file_md.parent, 0o444)

        issue = SonarIssue(
            key="python:S1234",
            project="test-project",
            rule="python:S1234",
            severity="MAJOR",
            message="Test",
            type=IssueType.CODE_SMELL,
            impact=Impact.MEDIUM,
            file="test.py",
            component="test-project:src/test.py",
            line=1
        )

        fix_response = {
            "explanation": "Test",
            "fixed_code": "Test"
        }

        try:
            with pytest.raises(PermissionError):
                fixer.write_explaination(file_md, fix_response, [issue], "")
        finally:
            # Restore permissions for cleanup
            os.chmod(file_md.parent, 0o755)

    def test_invalid_file_path(self, fixer):
        """Test with invalid file path"""
        # Use invalid path (null byte in filename on Unix)
        if isinstance(Path("/tmp/test\x00.md"), Path):
            file_md = Path("/tmp/test\x00.md")
        else:
            file_md = Path("/dev/null/impossible.md")

        issue = SonarIssue(
            key="a",
            project="a",
            component="a/test.py",
            rule="python:S1234",
            severity="MAJOR",
            message="Test",
            file="test.py",
            type=IssueType.CODE_SMELL,
            impact=Impact.MEDIUM,

            line=1
        )

        fix_response = {
            "explanation": "Test",
            "fixed_code": "Test"
        }

        with pytest.raises((ValueError, OSError, FileNotFoundError)):
            fixer.write_explaination(file_md, fix_response, [issue], "")

    # ============================================================================
    # TEST CASES - CONCURRENT ACCESS
    # ============================================================================

    def test_concurrent_writes(self, fixer, temp_dir):
        """Test multiple writes to the same file (simulating concurrent access)"""
        import threading

        file_md = temp_dir / "concurrent.md"

        def write_issue(issue_num):
            issue = SonarIssue(
                key=f"test-project:src/test.py:S{issue_num}",
                rule=f"python:S{issue_num}",
                severity=Severity.MAJOR,
                component="test-project:src/test.py",
                project="test-project",
                message=f"Issue {issue_num}",
                type=IssueType.CODE_SMELL,
                impact=Impact.MEDIUM,
                file_path="test.py",
                line=issue_num
            )

            fix_response = {
                "explanation": f"Explanation {issue_num}",
                "fixed_code": f"Fix {issue_num}"
            }

            fixer.write_explaination(file_md, fix_response, [issue], "")

        threads = [threading.Thread(target=write_issue, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        content = file_md.read_text()
        # All 5 issues should be written (though order may vary)
        for i in range(5):
            assert f"python:S{i}" in content

    # ============================================================================
    # TEST CASES - PERFORMANCE
    # ============================================================================

    def test_large_number_of_issues(self, fixer, temp_dir):
        """Test writing a large number of issues"""
        import time

        file_md = temp_dir / "many_issues.md"

        # Create 100 issues
        issues = [
            SonarIssue(
                key=f"S{i}",
                rule=f"python:S{i}",
                severity="MAJOR",
                component=f"test-project:src/file{i}.py",
                project="test-project",
                message=f"Issue {i}",
                file_path=f"file{i}.py",
                type=IssueType.CODE_SMELL,
                impact=Impact.MEDIUM,
                status="OPEN",
                line=i
            )
            for i in range(100)
        ]

        fix_response = {
            "explanation": "Common explanation",
            "fixed_code": "Common fix"
        }

        start = time.time()
        fixer.write_explaination(file_md, fix_response, issues, "")
        duration = time.time() - start

        assert file_md.exists()
        assert duration < 5.0  # Should complete in under 5 seconds
        content = file_md.read_text()
        assert content.count("## 🔍 Issue:") == 100

    # ============================================================================
    # TEST CASES - INTEGRATION
    # ============================================================================


    def test_real_world_scenario(self, fixer, temp_dir, sample_sonar_issue):
        """Test realistic scenario with actual code and fixes"""
        file_md = temp_dir / "real_world.md"



        original_code = """
    def connect():
        host = config.get('database')
        port = config.get('database')
        name = config.get('database')
    """

        fix_response = {
            "explanation": "String literals duplicated multiple times should be extracted as constants to improve maintainability and reduce the risk of typos.",
            "fixed_code": """
    DATABASE_KEY = 'database'

    def connect():
        host = config.get(DATABASE_KEY)
        port = config.get(DATABASE_KEY)
        name = config.get(DATABASE_KEY)
    """
        }

        fixer.write_explaination(file_md, fix_response, [sample_sonar_issue], original_code)

        assert file_md.exists()
        content = file_md.read_text()
        assert "python:S1481" in content
        assert "Remove the unused" in content
        assert "DATABASE_KEY" in content


class TestContextExtractorComprehensive:
    """Comprehensive tests for ContextExtractor class"""

    def test_is_function_definition_python_variations(self):
        """Test Python function definition detection variations"""
        extractor = ContextExtractor([])

        # Standard function
        assert extractor._is_function_definition("def foo():")
        assert extractor._is_function_definition("    def bar(x, y):")

        # Async function
        assert extractor._is_function_definition("async def async_foo():")
        assert extractor._is_function_definition("    async def async_bar():")

        # With decorators (should still detect)
        assert extractor._is_function_definition("def decorated():")

        # Not functions
        assert not extractor._is_function_definition("# def not_a_function")
        assert not extractor._is_function_definition("defined = 'string'")
        assert not extractor._is_function_definition("x = def_value")

    def test_is_function_definition_javascript_variations(self):
        """Test JavaScript function definition detection"""
        extractor = ContextExtractor([])

        # Standard function
        assert extractor._is_function_definition("function myFunc() {")
        assert extractor._is_function_definition("async function asyncFunc() {")

        # Method in object
        assert extractor._is_function_definition("myMethod: function() {")
        assert extractor._is_function_definition("myMethod: async function() {")

        # Arrow functions
        assert extractor._is_function_definition("const foo = () => {")
        assert extractor._is_function_definition("const bar = async (x) => {")

        # Class methods
        assert extractor._is_function_definition("  myMethod() {")
        assert extractor._is_function_definition("  async myMethod() {")

    def test_is_function_definition_java_csharp(self):
        """Test Java/C# method definition detection"""
        extractor = ContextExtractor([])

        # Java methods
        assert extractor._is_function_definition("public void doSomething() {")
        assert extractor._is_function_definition("private int calculate(int x) {")
        assert extractor._is_function_definition("protected static String getValue() {")

        # C# methods
        assert extractor._is_function_definition("public void DoSomething() {")
        assert extractor._is_function_definition("internal async Task<int> GetValueAsync() {")

    def test_find_containing_function_simple(self):
        """Test finding containing function - simple case"""
        lines = [
            "class MyClass:\n",
            "    def method1(self):\n",
            "        x = 1\n",
            "        y = 2\n",
            "        return x + y\n",
            "\n",
            "    def method2(self):\n",
            "        pass\n",
        ]
        extractor = ContextExtractor(lines)

        # Line 3 (y = 2) should find method1 at line 1
        result = extractor._find_containing_function(3)
        assert result == 1

    def test_find_containing_function_nested(self):
        """Test finding containing function - nested functions"""
        lines = [
            "def outer():\n",
            "    x = 1\n",
            "    def inner():\n",
            "        y = 2\n",
            "        return y\n",
            "    return inner()\n",
        ]
        extractor = ContextExtractor(lines)

        # Line 3 (y = 2) should find inner function at line 2
        result = extractor._find_containing_function(3)
        assert result == 2

    def test_find_containing_function_not_found(self):
        """Test finding containing function when line is not in a function"""
        lines = [
            "import sys\n",
            "import os\n",
            "\n",
            "x = 1\n",
            "y = 2\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._find_containing_function(3)
        assert result is None


    def test_find_python_function_end_with_nested_blocks(self):
        """Test finding Python function end with nested blocks"""
        lines = [
            "def complex():\n",
            "    if True:\n",
            "        x = 1\n",
            "        if x > 0:\n",
            "            print(x)\n",
            "    return x\n",
            "\n",
            "def another():\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._find_python_function_end(lines, 0)
        assert result == 6

    def test_find_python_function_end_with_multiline_statements(self):
        """Test finding Python function end with multiline statements"""
        lines = [
            "def multiline():\n",
            "    result = (\n",
            "        1 + 2 +\n",
            "        3 + 4\n",
            "    )\n",
            "    return result\n",
            "\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._find_python_function_end(lines, 0)
        assert result is not None

    def test_find_brace_function_end_simple(self):
        """Test finding brace function end - simple JavaScript"""
        lines = [
            "function test() {\n",
            "    var x = 1;\n",
            "    return x;\n",
            "}\n",
            "\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._find_brace_function_end(lines, 0)
        assert result == 3

    def test_find_brace_function_end_nested_braces(self):
        """Test finding brace function end with nested braces"""
        lines = [
            "function test() {\n",
            "    if (true) {\n",
            "        console.log('test');\n",
            "    }\n",
            "    return;\n",
            "}\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._find_brace_function_end(lines, 0)
        assert result == 5

    def test_find_brace_function_end_with_strings(self):
        """Test finding brace function end ignoring braces in strings"""
        lines = [
            "function test() {\n",
            "    var str = 'text with } brace';\n",
            "    var obj = {key: 'value'};\n",
            "    return str;\n",
            "}\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._find_brace_function_end(lines, 0)
        assert result == 4

    def test_remove_strings_and_comments(self):
        """Test removing strings and comments"""
        extractor = ContextExtractor([])

        # Single line comments
        result = extractor._remove_strings_and_comments("var x = 1; // comment")
        assert "//" not in result or "comment" not in result

        # String literals
        result = extractor._remove_strings_and_comments('var str = "text {with} braces";')
        assert "text" not in result or "{with}" not in result

        # Multiple quotes
        result = extractor._remove_strings_and_comments("var s = 'single' + \"double\";")
        assert len(result) < len("var s = 'single' + \"double\";")

    def test_is_line_inside_function_true(self):
        """Test checking if line is inside function - true case"""
        lines = [
            "def foo():\n",
            "    x = 1\n",
            "    y = 2\n",
            "    return x + y\n",
        ]
        extractor = ContextExtractor(lines)

        assert extractor._is_line_inside_function(lines, 2, 0)

    def test_is_line_inside_function_false(self):
        """Test checking if line is inside function - false case"""
        lines = [
            "def foo():\n",
            "    return 1\n",
            "\n",
            "x = 2\n",
        ]
        extractor = ContextExtractor(lines)

        assert not extractor._is_line_inside_function(lines, 3, 0)

    def test_check_indentation_containment(self):
        """Test indentation-based containment check"""
        lines = [
            "def foo():\n",
            "    x = 1\n",
            "    y = 2\n",
            "z = 3\n",
        ]
        extractor = ContextExtractor(lines)

        # Line 2 is inside function (greater indent)
        assert extractor._check_indentation_containment(lines, 2, 0)

        # Line 3 is outside function (same indent as function)
        assert not extractor._check_indentation_containment(lines, 3, 0)

    def test_is_decorator_python(self):
        """Test Python decorator detection"""
        extractor = ContextExtractor([])

        assert extractor._is_decorator("@property")
        assert extractor._is_decorator("@staticmethod")
        assert extractor._is_decorator("@app.route('/path')")
        assert extractor._is_decorator("@decorator.with.dots")

        assert not extractor._is_decorator("# @not_a_decorator")
        assert not extractor._is_decorator("email@example.com")

    def test_find_function_start_with_decorators(self):
        """Test finding function start including decorators"""
        lines = [
            "class MyClass:\n",
            "    @property\n",
            "    @staticmethod\n",
            "    def my_method():\n",
            "        pass\n",
        ]
        extractor = ContextExtractor(lines)

        # Starting from function def line (index 3)
        result = extractor._find_function_start_with_decorators(lines, 3)
        assert result == 1  # Should include both decorators

    def test_extract_function_name_python(self):
        """Test extracting function name from Python definition"""
        extractor = ContextExtractor([])

        assert extractor._extract_function_name("def my_function():") == "my_function"
        assert extractor._extract_function_name("async def async_func(x, y):") == "async_func"
        assert extractor._extract_function_name("    def indented():") == "indented"

    def test_extract_function_name_javascript(self):
        """Test extracting function name from JavaScript definition"""
        extractor = ContextExtractor([])

        assert extractor._extract_function_name("function myFunc() {") == "myFunc"
        assert extractor._extract_function_name("const foo = () => {") == "foo"
        assert extractor._extract_function_name("myMethod: function() {") == "myMethod"

    def test_extract_complete_function_simple(self):
        """Test extracting complete function - simple case"""
        lines = [
            "def simple():\n",
            "    return 1\n",
            "\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._extract_complete_function(0, 0,0)

        assert result is not None
        assert result["is_complete_function"]
        assert result["function_name"] == "simple"
        assert "def simple():" in result["context"]

    def test_extract_complete_function_with_decorators(self):
        """Test extracting complete function with decorators"""
        lines = [
            "@decorator1\n",
            "@decorator2\n",
            "def decorated():\n",
            "    return 1\n",
            "\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._extract_complete_function(2, 2,3)

        assert result is not None
        assert result["has_decorators"]
        assert result["decorator_count"] == 2
        assert "@decorator1" in result["context"]
        assert "@decorator2" in result["context"]


    def test_try_extract_containing_function_success(self):
        """Test extracting containing function - success"""
        lines = [
            "def outer():\n",
            "    x = 1\n",
            "    y = 2\n",
            "    return x + y\n",
        ]
        extractor = ContextExtractor(lines)

        result = extractor._try_extract_containing_function(2,5)

        assert result is not None
        assert result["is_complete_function"]

    def test_try_expand_multiline_to_function_success(self):
        """Test expanding multiline issue to complete function"""
        lines = [
            "def func():\n",
            "    x = 1\n",
            "    y = 2\n",
            "    z = 3\n",
            "    return x + y + z\n",
        ]
        extractor = ContextExtractor(lines)

        # Issue spans lines 1-2, but function extends to line 4
        result = extractor._try_expand_multiline_to_function(1, 2)

        assert result is not None
        assert result["is_complete_function"]

    def test_extract_normal_context(self):
        """Test extracting normal context (not function-based)"""
        lines = [f"line {i}\n" for i in range(1, 11)]
        extractor = ContextExtractor(lines)

        result = extractor._extract_normal_context(5, 5, context_lines=2)

        assert result is not None
        assert not result["is_complete_function"]
        assert result["start_line"] == 4  # 5 - 2 + 1 (1-indexed)
        assert result["end_line"] == 8  # 5 + 2 + 1

    @pytest.mark.skip(reason="Need update")
    def test_get_empty_context(self):
        """Test getting empty context for out-of-range line"""
        extractor = ContextExtractor([])

        result = extractor._get_empty_context(100)
        print("result ", result)
        assert result["context"] == ""
        assert result["problem_lines"] ==[]
        assert result["line_number"] == 100


# ============================================================================
# TEST CLASS: MODULE-LEVEL FUNCTIONS
# ============================================================================

class TestModuleLevelFunctions:
    """Test module-level helper functions"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_generate_fix_key_single_line(self):
        """Test generating fix key for single line"""
        key = _generate_fix_key([10])
        assert key == "fix_L10"

    def test_generate_fix_key_multiple_lines(self):
        """Test generating fix key for multiple lines"""
        key = _generate_fix_key([10, 15, 20])
        assert key == "fix_L10-L20"

    def test_generate_fix_key_empty(self):
        """Test generating fix key for empty list"""
        key = _generate_fix_key([])
        assert key == "fix_unknown"

    def test_extract_problem_lines_valid(self):
        """Test extracting problem lines with valid indices"""
        file_lines = [f"line {i}\n" for i in range(1, 11)]
        problem_lines = _extract_problem_lines(file_lines, [1, 5, 10])

        # Key-based access with line numbers
        assert 1 in problem_lines
        assert 5 in problem_lines
        assert 10 in problem_lines

        assert problem_lines[1] == "line 1\n"
        assert problem_lines[5] == "line 5\n"
        assert problem_lines[10] == "line 10\n"

    def test_extract_problem_lines_out_of_bounds(self):
        """Test extracting problem lines with out-of-bounds indices"""
        file_lines = ["line 1\n", "line 2\n"]
        problem_lines = _extract_problem_lines(file_lines, [1, 100])

        assert isinstance(problem_lines, dict)
        assert len(problem_lines) == 1  # Only line 1 is valid

        # Line 1 should be present with correct content
        assert 1 in problem_lines
        assert problem_lines[1] == "line 1\n"

        # Line 100 should NOT be in the dictionary (out of bounds)
        assert 100 not in problem_lines


    @pytest.mark.skip(reason="Need update")
    def test_validate_and_extract_issue_info_single_issue(self, tmp_path):
        """Test validating and extracting info from single issue"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")

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
            last_line=5,
            file="test.py"
        )

        file_path, line_range = _validate_and_extract_issue_info([issue], tmp_path)

        assert file_path == test_file
        assert line_range["first_line"] == 1
        assert line_range["last_line"] == 5

    @pytest.mark.skip(reason="Need update")
    def test_validate_and_extract_issue_info_multiple_issues(self, tmp_path):
        """Test validating multiple issues from same file"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")

        issue1 = SonarIssue(
            key="test1",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=5,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=5,
            last_line=10,
            file="test.py"
        )
        issue2 = SonarIssue(
            key="test2",
            rule="python:S5678",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=15,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=15,
            last_line=20,
            file="test.py"
        )

        file_path, line_range = _validate_and_extract_issue_info([issue1, issue2], tmp_path)

        assert line_range["first_line"] == 5
        assert line_range["last_line"] == 20

    @pytest.mark.skip(reason="Need update")
    def test_validate_and_extract_issue_info_different_files_error(self, tmp_path):
        """Test error when issues from different files"""
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("content")
        file2.write_text("content")

        issue1 = SonarIssue(
            key="test1",
            rule="python:S1234",
            severity="MAJOR",
            component="file1.py",
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5,
            file="file1.py"
        )
        issue2 = SonarIssue(
            key="test2",
            rule="python:S5678",
            severity="MAJOR",
            component="file2.py",
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5,
            file="file2.py"
        )

        with pytest.raises(ValueError):
            _validate_and_extract_issue_info([issue1, issue2], tmp_path)

    def test_validate_and_extract_issue_info_file_not_found(self, tmp_path):
        """Test error when file doesn't exist"""
        issue = SonarIssue(
            key="test",
            rule="python:S1234",
            severity="MAJOR",
            component="nonexistent.py",
            project="test",
            line=1,
            message="Test",
            type="VULNERABILITY",
            status="OPEN",
            first_line=1,
            last_line=5,
            file="nonexistent.py"
        )

        with pytest.raises(FileNotFoundError):
            _validate_and_extract_issue_info([issue], tmp_path)

    def test_prepare_context_success(self, tmp_path,fixer):
        """Test preparing context successfully"""
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")

        line_range = {
            "first_line": 1,
            "last_line": 2,
            "problem_lines": [1, 2]
        }

        result = fixer._prepare_context(test_file, line_range, "", context_lines=1, language="python")

        assert result is not None
        assert "context_dict" in result
        assert "file_lines" in result

    def test_prepare_context_with_modified_content(self, tmp_path,fixer):
        """Test preparing context with modified content"""
        test_file = tmp_path / "test.py"
        test_file.write_text("original\n")

        line_range = {
            "first_line": 1,
            "last_line": 1,
            "problem_lines": [1]
        }

        result = fixer._prepare_context(
            test_file,
            line_range,
            modified_content="modified content",
            context_lines=1,
            language="python"
        )

        assert result["context_dict"]["context"] == "modified content"

    def test_prepare_context_unicode_error(self, tmp_path, fixer):
        """Test handling unicode decode error"""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b'\x80\x81\x82')  # Invalid UTF-8

        line_range = {
            "first_line": 1,
            "last_line": 1,
            "problem_lines": [1]
        }

        result =fixer. _prepare_context(test_file, line_range, "", context_lines=1, language="python")
        assert result is None

    @pytest.mark.skip(reason="Need update")
    def test_build_fix_suggestion_complete(self, tmp_path):
        """Test building fix suggestion with all fields"""
        fix_response = {
            "fixed_code": "def fixed():\n    pass",
            "helper_code": "def helper():\n    return 1",
            "placement_helper": "BEFORE",
            "explanation": "Fixed the code",
            "confidence": 0.95,
            "rule_description": "Test rule"
        }

        context_info = {
            "context_dict": {
                "context": "original code",
                "start_line": 10,
                "end_line": 15
            }
        }

        file_path = tmp_path / "test.py"
        line_range = {
            "first_line": 10,
            "last_line": 15,
            "problem_lines": [10, 11]
        }

        result = _build_fix_suggestion(
            fix_response,
            context_info,
            file_path,
            tmp_path,
            line_range,
            "gpt-4"
        )

        assert isinstance(result, FixSuggestion)
        assert result.fixed_code == "def fixed():\n    pass"
        assert result.helper_code == "def helper():\n    return 1"
        assert result.confidence == 0.95
        assert result.llm_model == "gpt-4"


# ============================================================================
# TEST CLASS: LLM FIXER - ADDITIONAL METHODS
# ============================================================================

class TestLLMFixerAdditionalMethods:
    """Test additional LLMFixer methods for coverage"""


    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_extract_complexity_info_standard_format(self, fixer):
        """Test extracting complexity info from standard message"""
        message = "Cognitive Complexity from 25 to the 15 allowed"
        result = fixer._extract_complexity_info(message)

        assert result["current"] == "25"
        assert result["target"] == "15"

    def test_extract_complexity_info_alternative_format(self, fixer):
        """Test extracting complexity info from alternative message format"""
        message = "complexity is 30, maximum is 15"
        result = fixer._extract_complexity_info(message)

        assert result["current"] == "30"
        assert result["target"] == "15"

    def test_extract_complexity_info_no_match(self, fixer):
        """Test extracting complexity info when no pattern matches"""
        message = "Some other message"
        result = fixer._extract_complexity_info(message)

        assert result["current"] == "Unknown"
        assert result["target"] == "15"

    def test_is_init_method_python(self, fixer):
        """Test detecting Python __init__ method"""
        context = "def __init__(self):\n    self.x = 1"
        assert fixer._is_init_method(context)

    def test_is_init_method_java_constructor(self, fixer):
        """Test detecting Java constructor"""
        context = "public MyClass() {\n    this.x = 1;\n}"
        assert fixer._is_init_method(context)

    def test_is_init_method_javascript_constructor(self, fixer):
        """Test detecting JavaScript constructor"""
        context = "constructor() {\n    this.x = 1;\n}"
        assert fixer._is_init_method(context)

    def test_is_init_method_not_constructor(self, fixer):
        """Test non-constructor method"""
        context = "def regular_method():\n    return 1"
        assert not fixer._is_init_method(context)

    @pytest.mark.skip(reason="Need update")
    def test_create_fix_prompt_list_multiple_issues(self, fixer):
        """Test creating prompt for multiple issues"""
        issues = [
            SonarIssue(
                key="test1",
                rule="python:S1481",
                severity="MINOR",
                component="test.py",
                project="test",
                line=5,
                message="Unused variable",
                type="CODE_SMELL",
                status="OPEN",
                first_line=5,
                last_line=5
            ),
            SonarIssue(
                key="test2",
                rule="python:S1192",
                severity="MINOR",
                component="test.py",
                project="test",
                line=7,
                message="Duplicate literal",
                type="CODE_SMELL",
                status="OPEN",
                first_line=7,
                last_line=7
            )
        ]

        context = {
            "context": "def func():\n    unused = 1\n    x = 'test'\n    y = 'test'",
            "start_line": 1,
            "end_line": 4
        }

        rule_info_list = {
            "python:S1481": {"name": "Unused variable rule"},
            "python:S1192": {"name": "Duplicate literal rule"}
        }

        prompt = fixer._create_fix_prompt_list(issues, context, rule_info_list, "python")

        assert "S1481" in prompt or "Unused" in prompt
        assert "S1192" in prompt or "Duplicate" in prompt

    def test_looks_like_file_path_valid(self, fixer):
        """Test checking if string looks like file path - valid cases"""
        assert fixer._looks_like_file_path("src/main.py")
        assert fixer._looks_like_file_path("path/to/file.java")
        assert fixer._looks_like_file_path("app/models/user.js")

    def test_looks_like_file_path_invalid(self, fixer):
        """Test checking if string looks like file path - invalid cases"""
        assert not fixer._looks_like_file_path("just_a_string")
        assert not fixer._looks_like_file_path("no-slash.py")
        assert not fixer._looks_like_file_path("wrong/extension.xyz")

    def test_try_stored_file_path_success(self, fixer, tmp_path, sample_code_block):
        """Test getting file from stored path - success"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        fix = FixSuggestion(
            issue_key="test",
            file_path="test.py",
            original_code="",
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="gpt-4",
            fixed_code_blocks=[sample_code_block]
        )

        result = fixer._try_stored_file_path(fix, tmp_path)
        assert result is not None

    def test_try_stored_file_path_not_exists(self, fixer, tmp_path,sample_code_block):
        """Test getting file from stored path - file doesn't exist"""
        fix = FixSuggestion(
            issue_key="test",
            file_path="nonexistent.py",
            original_code="",
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="gpt-4",
            fixed_code_blocks=[sample_code_block]
        )

        result = fixer._try_stored_file_path(fix, tmp_path)
        assert result is None

    def test_try_extract_from_issue_key_success(self, fixer, tmp_path,sample_code_block):
        """Test extracting file from issue key - success"""
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")

        fix = FixSuggestion(
            issue_key="project:src/test.py:S1234",
            file_path=None,
            original_code="",
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="gpt-4",
            fixed_code_blocks=[sample_code_block],
        )

        result = fixer._try_extract_from_issue_key(fix, tmp_path)
        assert result is not None

    def test_try_extract_from_issue_key_no_colon(self, fixer, tmp_path,sample_code_block):
        """Test extracting file from issue key - no colon separator"""
        fix = FixSuggestion(
            issue_key="simple-key",
            file_path=None,
            original_code="",
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="gpt-4",
            fixed_code_blocks=[sample_code_block]
        )

        result = fixer._try_extract_from_issue_key(fix, tmp_path)
        assert result is None

    def test_find_files_with_content_success(self, fixer, tmp_path):
        """Test finding files containing specific content"""
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("def target_function():\n    pass")
        file2.write_text("def other_function():\n    pass")

        results = fixer._find_files_with_content(tmp_path, "target_function")

        assert len(results) >= 1
        assert any("file1.py" in str(f) for f in results)

    def test_find_files_with_content_not_found(self, fixer, tmp_path):
        """Test finding files with content not present"""
        file1 = tmp_path / "file1.py"
        file1.write_text("def some_function():\n    pass")

        results = fixer._find_files_with_content(tmp_path, "nonexistent_content")

        assert len(results) == 0

    def test_find_files_with_content_unicode_error(self, fixer, tmp_path):
        """Test finding files handles unicode errors gracefully"""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b'\x80\x81\x82')

        # Should not crash on binary files
        results = fixer._find_files_with_content(tmp_path, "text")
        assert isinstance(results, list)

    def test_check_bracket_balance_complex(self, fixer):
        """Test bracket balance checking with complex code"""
        # Balanced
        assert fixer._check_bracket_balance("def f(): return {1: [2, 3]}")

        # Unbalanced - missing closing
        assert not fixer._check_bracket_balance("def f(): return {1: [2, 3]")

        # Unbalanced - extra closing
        assert not fixer._check_bracket_balance("def f(): return {1: [2, 3]]}")

    def test_check_no_duplicate_definitions_no_duplicates(self, fixer):
        """Test checking for duplicate definitions - none found"""
        content = """
def func1():
    pass

def func2():
    pass

class MyClass:
    pass
"""
        assert fixer._check_no_duplicate_definitions(content, ".py")

    def test_check_no_duplicate_definitions_has_duplicates(self, fixer):
        """Test checking for duplicate definitions - duplicates found"""
        content = """
def duplicate():
    pass

def other():
    pass

def duplicate():
    pass
"""
        assert not fixer._check_no_duplicate_definitions(content, ".py")

    def test_check_no_duplicate_definitions_non_python(self, fixer):
        """Test checking duplicate definitions for non-Python file"""
        content = "function duplicate() {}\nfunction duplicate() {}"
        # Should return True (only checks Python)
        assert fixer._check_no_duplicate_definitions(content, ".js")

    def test_apply_indentation_to_fix_empty(self, fixer):
        """Test applying indentation to empty code"""
        result = fixer.apply_indentation_to_fix("", "    ")
        assert result == ""

    def test_apply_indentation_to_fix_multiline(self, fixer):
        """Test applying indentation to multiline code"""
        fixed_code = "def foo():\npass\nreturn 1"
        result = fixer.apply_indentation_to_fix(fixed_code, "    ")

        lines = result.split("\n")
        assert all(line.startswith("    ") or not line.strip() for line in lines)

    def test_get_file_from_fix_all_strategies(self, fixer, tmp_path, sample_code_block):
        """Test all strategies for getting file from fix"""
        test_file = tmp_path / "target.py"
        test_file.write_text("def target_function():\n    pass")

        # Test strategy 3: find by content
        fix = FixSuggestion(
            issue_key="unknown",
            file_path=None,
            original_code="def target_function():",
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="gpt-4",
            fixed_code_blocks=[sample_code_block],
        )

        with patch.object(fixer, '_find_files_with_content') as mock_find:
            mock_find.return_value = [test_file]
            result = fixer._get_file_from_fix(fix, tmp_path)
            assert result is not None

    def test_extract_fields_from_parsed_json_all_fields(self, fixer):
        """Test extracting all fields from parsed JSON"""
        fix_data = {
            "FIXED_SELECTION": "fixed code",
            "NEW_HELPER_CODE": "helper code",
            "PLACEMENT": "GLOBAL_TOP",
            "EXPLANATION": "Explanation text",
            "CONFIDENCE": "0.88"
        }

        result = fixer._extract_fields_from_parsed_json(fix_data)

        assert result is not None
        assert result["fixed_code"] == "fixed code"
        assert result["helper_code"] == "helper code"
        assert result["placement_helper"] == "GLOBAL_TOP"
        assert result["confidence"] == 0.88

    def test_extract_fields_from_parsed_json_minimal(self, fixer):
        """Test extracting fields with minimal data"""
        fix_data = {
            "FIXED_SELECTION": "code"
        }

        result = fixer._extract_fields_from_parsed_json(fix_data)

        assert result is not None
        assert result["fixed_code"] == "code"
        assert result["helper_code"] == ""
        assert result["placement_helper"] == "SIBLING"

    def test_extract_fields_from_parsed_json_invalid_placement(self, fixer):
        """Test extracting fields with invalid placement"""
        fix_data = {
            "FIXED_SELECTION": "code",
            "PLACEMENT": "INVALID_PLACEMENT"
        }

        result = fixer._extract_fields_from_parsed_json(fix_data)

        assert result["placement_helper"] == "SIBLING"  # Default



    def test_apply_regex_patterns_all_patterns(self, fixer):
        """Test applying all regex patterns"""
        content = '''
{
    "FIXED_SELECTION": "fixed code here",
    "NEW_HELPER_CODE": "helper code",
    "PLACEMENT": "SIBLING",
    "EXPLANATION": "Detailed explanation",
    "CONFIDENCE": 0.92
}
'''

        results = fixer._apply_regex_patterns(content)

        assert results["FIXED_SELECTION"] == "fixed code here"
        assert results["NEW_HELPER_CODE"] == "helper code"
        assert results["PLACEMENT"] == "SIBLING"

    def test_get_match_value_with_groups(self, fixer):
        """Test getting match value with multiple groups"""


        # Test with first group match
        match = re.search(r'"key"\s*:\s*"(value1)"|\'key\'\s*:\s*\'(value2)\'', '"key": "value1"')
        result = fixer._get_match_value(match)
        assert result == "value1"

        # Test with second group match
        match = re.search(r'"key"\s*:\s*"(value1)"|\'key\'\s*:\s*\'(value2)\'', "'key': 'value2'")
        result = fixer._get_match_value(match)
        assert "value" in result

    def test_process_match_value_confidence(self, fixer):
        """Test processing confidence value"""
        result = fixer._process_match_value("CONFIDENCE", "0.75")
        assert result == 0.75

        result = fixer._process_match_value("CONFIDENCE", None)
        assert result == 0.5

    def test_process_match_value_placement_default(self, fixer):
        """Test processing placement with default"""
        result = fixer._process_match_value("PLACEMENT", None)
        assert result == "SIBLING"

    def test_validate_results_missing_fixed_selection(self, fixer):
        """Test validating results with missing FIXED_SELECTION"""
        results = {
            "FIXED_SELECTION": "",
            "EXPLANATION": "text",
            "CONFIDENCE": 0.8,
            "NEW_HELPER_CODE": "",
            "PLACEMENT": "SIBLING"
        }

        result = fixer._validate_results(results)
        assert result is None

    def test_validate_results_confidence_bounds(self, fixer):
        """Test confidence is bounded between 0 and 1"""
        results = {
            "IMPORT_BLOCK":"",
            "FIXED_SELECTION": "code",
            "CONFIDENCE": 1.5,  # Over 1
            "NEW_HELPER_CODE": "",
            "PLACEMENT": "SIBLING",
            "EXPLANATION": "text"
        }

        result = fixer._validate_results(results)
        assert result["confidence"] == 1.0

        results["CONFIDENCE"] = -0.5  # Under 0
        result = fixer._validate_results(results)
        assert result["confidence"] == 0.0


# ============================================================================
# TEST CLASS: EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_extract_fix_from_response_malformed_json(self, fixer):
        """Test extracting fix from malformed JSON"""
        content = "{FIXED_SELECTION: 'missing quotes', incomplete"

        # Should fall back to regex
        result = fixer._extract_fix_from_response(content)
        # May return None or partial result
        assert result is None or isinstance(result, dict)

    def test_extract_fix_from_response_with_extra_text(self, fixer):
        """Test extracting fix with extra text before/after JSON"""
        content = '''
Here is the fix:

```json
{
    "FIXED_SELECTION": "def fixed():\n    pass",
    "EXPLANATION": "Fixed it",
    "CONFIDENCE": 0.9
}
```

Hope this helps!
'''

        result = fixer._extract_fix_from_response(content)
        assert result is None or (result and "fixed" in result.get("fixed_code", ""))

    @pytest.mark.skip(reason="Need update")
    def test_call_llm_list_with_empty_issues(self, fixer):
        """Test calling LLM with empty issues list"""
        context = {"context": "code", "start_line": 1, "end_line": 5}

        # Should handle gracefully
        result = fixer._call_llm_list([], context, ".py", {}, "")
        # Implementation may vary
        assert result is None or isinstance(result, dict)

    def test_generate_fix_by_file_empty_issues(self, fixer, tmp_path):
        """Test generating fix with empty issues list"""
        result = fixer.generate_fix_by_file([], tmp_path,tmp_path, {})
        assert result is None

    def test_apply_fixes_to_file_exception(self, fixer, tmp_path,sample_code_block):
        """Test handling exception during fix application"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")

        fix = FixSuggestion(
            issue_key="test",
            file_path="test.py",
            original_code="content",
            fixed_code="new content",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="gpt-4",
            fixed_code_blocks=[sample_code_block]
        )

        # Make file read-only to cause exception
        test_file.chmod(0o444)

        try:
            success, results = fixer._apply_fixes_to_file(test_file, [fix], dry_run=False)
            # Should handle gracefully
            assert isinstance(success, bool)
        finally:
            test_file.chmod(0o644)

    def test_write_explaination_io_error(self, fixer, tmp_path):
        """Test handling IO error during explanation writing"""
        file_md = tmp_path / "readonly" / "output.md"
        file_md.parent.mkdir()
        file_md.parent.chmod(0o444)

        issue = SonarIssue(
            key="test",
            rule="python:S1234",
            severity="MAJOR",
            component="test.py",
            project="test",
            line=1,
            message="Test",
            type="CODE_SMELL",
            status="OPEN",
            first_line=1,
            last_line=5,
            file="test.py"
        )

        try:
            with pytest.raises((PermissionError, OSError)):
                fixer.write_explaination(file_md, {}, [issue], "")
        finally:
            file_md.parent.chmod(0o755)

    def test_try_find_by_content_short_code(self, fixer, tmp_path, sample_code_block):
        """Test finding file by content with very short code"""
        fix = FixSuggestion(
            issue_key="test",
            original_code="x",  # Too short
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="gpt-4",

        fixed_code_blocks = [sample_code_block]
        )

        result = fixer._try_find_by_content(fix, tmp_path)
        assert result is None

    def test_context_extraction_line_out_of_bounds(self, fixer):
        """Test context extraction with line number out of bounds"""
        lines = ["line 1\n", "line 2\n"]


        context = fixer._extract_context(lines, 100, 100,[100] ,context_lines=5)

        # Should handle gracefully
        assert context["context"] == ""



    def test_create_backup_permission_error(self, fixer, tmp_path):
        """Test backup creation with permission issues"""
        # Create directory structure
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("content")

        # Make parent read-only
        tmp_path.chmod(0o444)

        try:
            with pytest.raises((PermissionError, OSError)):
                fixer._create_backup(source)
        finally:
            tmp_path.chmod(0o755)


class TestGetContentRange:
    """Test suite for get_content_range function."""


    # ==================== UNIT TESTS ====================

    def test_exact_match_scenario(self, temp_dir, sample_code):
        """Test Case 1: Exact match - content exists at same lines."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(sample_code)
        actual_file.write_text(sample_code)

        line_range = {
            'first_line': 4,
            'last_line': 6,
            'problem_lines': [5]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 4
        assert result['last_line'] == 6
        assert result['problem_lines'] == [5]
        assert result['confidence'] == 1.0
        assert result['match_type'] == 'exact'

    def test_code_shifted_down(self, temp_dir, sample_code):
        """Test Case 2: Code moved - lines shifted down by adding comments."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(sample_code)

        # Add comments at the beginning
        modified_code = "# Header comment 1\n# Header comment 2\n\n" + sample_code
        actual_file.write_text(modified_code)

        line_range = {
            'first_line': 4,
            'last_line': 6,
            'problem_lines': [5]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 7  # Shifted by 3 lines
        assert result['last_line'] == 9
        assert result['problem_lines'] == [8]  # Adjusted problem line
        assert result['confidence'] == 1.0
        assert result['match_type'] == 'exact'

    def test_code_shifted_up(self, temp_dir, sample_code):
        """Test Case 3: Code moved - lines shifted up by removing content."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(sample_code)

        # Remove first two lines
        lines = sample_code.split('\n')
        modified_code = '\n'.join(lines[2:])
        actual_file.write_text(modified_code)

        line_range = {
            'first_line': 4,
            'last_line': 6,
            'problem_lines': [5]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 2  # Shifted up by 2 lines
        assert result['last_line'] == 4
        assert result['problem_lines'] == [3]
        assert result['confidence'] == 1.0

    def test_duplicate_code_blocks(self, temp_dir):
        """Test Case 4: Duplicate code - same code appears multiple times."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        code_with_duplicate = """def function_one():
    x = 1
    y = 2
    return x + y

def function_two():
    x = 1
    y = 2
    return x + y

def function_three():
    x = 1
    y = 2
    return x * y
"""

        tmp_file.write_text(code_with_duplicate)
        actual_file.write_text(code_with_duplicate)

        # Looking for lines 2-3 which appear in multiple functions
        line_range = {
            'first_line': 2,
            'last_line': 3,
            'problem_lines': [2, 3]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        # Should find first occurrence (with context matching)
        assert result['first_line'] == 2
        assert result['last_line'] == 3
        assert result['confidence'] >= 0.9

    def test_single_line_multiple_occurrences(self, temp_dir):
        """Test Case 5: Single line appears multiple times."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        code = """def func1():
    return True

def func2():
    return True

def func3():
    return False
"""

        tmp_file.write_text(code)
        actual_file.write_text(code)

        line_range = {
            'first_line': 2,
            'last_line': 2,
            'problem_lines': [2]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        # Should find closest match to original line 2
        assert result['first_line'] == 2
        assert result['problem_lines'] == [2]

    def test_modified_code_fuzzy_match(self, temp_dir, sample_code):
        """Test Case 6: Code slightly modified (comments added)."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(sample_code)

        # Modify the code slightly
        modified = sample_code.replace(
            "def calculate_sum(a, b):",
            "def calculate_sum(a, b):  # Modified with comment"
        )
        actual_file.write_text(modified)


        line_range = {
            'first_line': 4,
            'last_line': 6,
            'problem_lines': [5]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        # Should still find it with fuzzy matching or context
        assert result is not None
        assert result['first_line'] is not None
        assert result['confidence'] >= 0.65

    def test_content_not_found(self, temp_dir, sample_code):
        """Test Case 7: Content completely removed/not found."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(sample_code)

        # Completely different content
        actual_file.write_text("def different_function():\n    pass\n")

        line_range = {
            'first_line': 4,
            'last_line': 6,
            'problem_lines': [5]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] is None
        assert result['last_line'] is None
        assert result['confidence'] == 0.0
        assert result['match_type'] == 'not_found'
        assert 'error' in result

    def test_multiline_range(self, temp_dir, sample_code):
        """Test Case 8: Multi-line range (class with multiple methods)."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(sample_code)
        actual_file.write_text(sample_code)

        line_range = {
            'first_line': 12,
            'last_line': 19,
            'problem_lines': [14, 17]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 12
        assert result['last_line'] == 19
        assert 14 in result['problem_lines']
        assert 17 in result['problem_lines']

    def test_edge_case_first_line(self, temp_dir):
        """Test Case 9: Edge case - first line of file."""
        code = """import os
import sys

def main():
    pass
"""

        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(code)
        actual_file.write_text(code)

        line_range = {
            'first_line': 1,
            'last_line': 1,
            'problem_lines': [1]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 1
        assert result['problem_lines'] == [1]

    def test_edge_case_last_line(self, temp_dir):
        """Test Case 10: Edge case - last line of file."""
        code = """def main():
    pass
    return True
"""

        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(code)
        actual_file.write_text(code)

        line_range = {
            'first_line': 3,
            'last_line': 3,
            'problem_lines': [3]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 3
        assert result['last_line'] == 3

    def test_empty_file(self, temp_dir):
        """Test Case 11: Edge case - empty actual file."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("def test():\n    pass\n")
        actual_file.write_text("")

        line_range = {
            'first_line': 1,
            'last_line': 2,
            'problem_lines': [1]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['match_type'] == 'not_found'

    def test_whitespace_differences(self, temp_dir):
        """Test Case 12: Code with different whitespace/indentation."""
        tmp_code = """def test():
    x = 1
    y = 2
    return x + y
"""

        # Different indentation
        actual_code = """def test():
        x = 1
        y = 2
        return x + y
"""

        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(tmp_code)
        actual_file.write_text(actual_code)

        line_range = {
            'first_line': 2,
            'last_line': 3,
            'problem_lines': [2]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        # Should not find exact match due to whitespace
        assert result is not None
        # Might find with fuzzy matching or not at all
        if result['match_type'] != 'not_found':
            assert result['confidence'] < 1.0

    def test_unicode_content(self, temp_dir):
        """Test Case 13: Files with unicode characters."""
        code = """def greet():
    message = "Hello 世界! 🌍"
    return message
"""

        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(code, encoding='utf-8')
        actual_file.write_text(code, encoding='utf-8')

        line_range = {
            'first_line': 2,
            'last_line': 2,
            'problem_lines': [2]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 2
        assert result['match_type'] == 'exact'

    def test_very_long_file(self, temp_dir):
        """Test Case 14: Performance - large file with many lines."""
        # Generate a large file
        lines = []
        for i in range(1000):
            lines.append(f"def function_{i}():\n")
            lines.append(f"    x = {i}\n")
            lines.append(f"    return x\n")
            lines.append("\n")

        large_code = "".join(lines)

        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(large_code)
        actual_file.write_text(large_code)

        # Look for something in the middle
        line_range = {
            'first_line': 502,
            'last_line': 503,
            'problem_lines': [502]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 502
        assert result['confidence'] == 1.0

    # ==================== ERROR HANDLING TESTS ====================

    def test_file_not_found_tmp(self, temp_dir):
        """Test Case 15: Error - temporary file doesn't exist."""
        tmp_file = temp_dir / "nonexistent.py"
        actual_file = temp_dir / "actual.py"
        actual_file.write_text("code")

        line_range = {'first_line': 1, 'last_line': 1, 'problem_lines': [1]}

        with pytest.raises(FileNotFoundError, match="Temporary file not found"):
            get_content_range(tmp_file, line_range, actual_file)

    def test_file_not_found_actual(self, temp_dir):
        """Test Case 16: Error - actual file doesn't exist."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "nonexistent.py"
        tmp_file.write_text("code")

        line_range = {'first_line': 1, 'last_line': 1, 'problem_lines': [1]}

        with pytest.raises(FileNotFoundError, match="Actual file not found"):
            get_content_range(tmp_file, line_range, actual_file)

    def test_invalid_line_range_missing_keys(self, temp_dir):
        """Test Case 17: Error - line_range missing required keys."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("code")
        actual_file.write_text("code")

        invalid_range = {'problem_lines': [1]}  # Missing first_line and last_line

        with pytest.raises(ValueError, match="must contain 'first_line' and 'last_line'"):
            get_content_range(tmp_file, invalid_range, actual_file)

    def test_out_of_bounds_line_range(self, temp_dir):
        """Test Case 18: Error - line range out of bounds."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("line1\nline2\n")
        actual_file.write_text("line1\nline2\n")

        out_of_bounds = {'first_line': 5, 'last_line': 10, 'problem_lines': [5]}

        with pytest.raises(ValueError, match="out of bounds"):
            get_content_range(tmp_file, out_of_bounds, actual_file)

    def test_negative_line_numbers(self, temp_dir):
        """Test Case 19: Error - negative line numbers."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("line1\nline2\n")
        actual_file.write_text("line1\nline2\n")

        invalid_range = {'first_line': -1, 'last_line': 2, 'problem_lines': [1]}

        with pytest.raises(ValueError, match="out of bounds"):
            get_content_range(tmp_file, invalid_range, actual_file)

    def test_first_line_greater_than_last_line(self, temp_dir):
        """Test Case 20: Error - first_line > last_line."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("line1\nline2\nline3\n")
        actual_file.write_text("line1\nline2\nline3\n")

        invalid_range = {'first_line': 3, 'last_line': 1, 'problem_lines': [3]}

        # This should return None as the slice will be empty
        result = get_content_range(tmp_file, invalid_range, actual_file)
        assert result is None

    # ==================== INTEGRATION TESTS ====================

    def test_real_world_scenario_refactoring(self, temp_dir):
        """Test Case 21: Real scenario - code refactored with new structure."""
        original = """def process_data(data):
    cleaned = clean(data)
    validated = validate(cleaned)
    return validated

def clean(data):
    return data.strip()

def validate(data):
    return len(data) > 0
"""

        refactored = """class DataProcessor:
    def process(self, data):
        cleaned = self.clean(data)
        validated = self.validate(cleaned)
        return validated

    def clean(self, data):
        return data.strip()

    def validate(self, data):
        return len(data) > 0
"""

        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(original)
        actual_file.write_text(refactored)

        # Looking for the clean function
        line_range = {'first_line': 7, 'last_line': 8, 'problem_lines': [8]}

        result = get_content_range(tmp_file, line_range, actual_file)

        # Should find it with context or fuzzy matching
        assert result is not None
        if result['match_type'] != 'not_found':
            assert result['confidence'] >= 0.7

    def test_problem_lines_adjustment(self, temp_dir):
        """Test Case 22: Verify problem_lines are correctly adjusted."""
        code = """def func():
    line2
    line3
    line4
    line5
"""

        shifted_code = "# comment\n# comment\n" + code

        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text(code)
        actual_file.write_text(shifted_code)

        line_range = {
            'first_line': 2,
            'last_line': 4,
            'problem_lines': [2, 3, 4]
        }

        result = get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        # Lines should be shifted by 2
        assert result['first_line'] == 4
        assert result['last_line'] == 6
        assert set(result['problem_lines']) == {4, 5, 6}

    def test_find_exact_match_found(self):
        """Test _find_exact_match when content exists."""
        target = ["line1\n", "line2\n"]
        actual = ["other\n", "line1\n", "line2\n", "more\n"]

        result = _find_exact_match(target, actual)

        assert result is not None
        assert result['start'] == 1
        assert result['end'] == 3

    def test_find_exact_match_not_found(self):
        """Test _find_exact_match when content doesn't exist."""
        target = ["line1\n", "line2\n"]
        actual = ["other\n", "different\n"]

        result = _find_exact_match(target, actual)

        assert result is None

    def test_find_all_single_line_matches(self):
        """Test finding all occurrences of a single line."""
        target = "return True\n"
        actual = ["def f1():\n", "return True\n", "def f2():\n", "return True\n", "def f3():\n"]

        matches = _find_all_single_line_matches(target, actual, original_line_num=2)

        assert len(matches) == 2
        assert matches[0] == 1  # Closest to line 2
        assert matches[1] == 3

    def test_create_result(self):
        """Test result dictionary creation."""
        match = {'start': 5, 'end': 8}
        problem_lines_tmp = [3, 4]
        first_line_tmp = 2

        result = _create_result(match, problem_lines_tmp, first_line_tmp, 0.95, 'context')

        assert result['first_line'] == 6
        assert result['last_line'] == 8
        assert result['confidence'] == 0.95
        assert result['match_type'] == 'context'
        # Problem lines adjusted: offset = 6 - 2 = 4, so [3+4, 4+4] = [7, 8]
        assert set(result['problem_lines']) == {7, 8}

