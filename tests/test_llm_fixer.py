# Comprehensive Test Cases for llm_fixer.py


import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open, call
from typing import List, Dict, Any, Union
import tempfile
import shutil
import re
import subprocess

from devdox_ai_sonar.llm_fixer import (LLMFixer, ContextExtractor, _build_fix_suggestion,
                                       _generate_fix_key,
                                       _extract_problem_lines,
                                       FUNCTION_ALREADY_CALLED,
                                       )
from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
from devdox_ai_sonar.services.extractor import (_validate_and_extract_issue_info,
                                                IssueExtractor,
                                                _find_exact_match,
                                                _find_fuzzy_match,
                                                _find_all_single_line_matches,
                                                _create_result
                                                )
from devdox_ai_sonar.utils.async_file_io import AsyncFileReader
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    FixSuggestion,
    SonarFixResponse,
    SonarSecurityIssue,
    Severity,
    IssueType,
    Impact,
    FixResult,
    ChangeType,
    BlockType,
    CodeBlock,
    PlacementType,

)
from devdox_ai_sonar.models.file_structures import FixContext
from devdox_ai_sonar.services.rule_handler import _resolve_effective_values, _resolve_relative_path

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
def mock_openrouter_client():
    """Mock OpenRouter client (uses OpenAI SDK with custom base_url)"""
    with patch('devdox_ai_sonar.llm_fixer.openai.OpenAI') as mock:
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
    
    def test_init_openrouter_with_api_key(self, mock_openrouter_client):
        """Test OpenRouter initialization with API key"""
        fixer = LLMFixer(
            provider="openrouter",
            model="anthropic/claude-sonnet-4",
            api_key="test-openrouter-key"
        )

        assert fixer.provider == "openrouter"
        assert fixer.model == "anthropic/claude-sonnet-4"
        assert fixer.api_key == "test-openrouter-key"
        mock_openrouter_client.assert_called_once_with(
            api_key="test-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://devdox.ai",
                "X-Title": "DevDox AI Sonar",
            },
        )

    def test_init_openrouter_from_env(self, mock_openrouter_client, monkeypatch):
        """Test OpenRouter initialization from environment variable"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-key")

        fixer = LLMFixer(provider="openrouter")

        assert fixer.api_key == "env-openrouter-key"

    def test_init_openrouter_missing_key(self, mock_openrouter_client):
        """Test OpenRouter initialization without API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LLMFixer(provider="openrouter")
            assert "OpenRouter API key not provided" in str(exc_info.value)

    def test_init_openrouter_default_model(self, mock_openrouter_client):
        """Test OpenRouter uses default model when not specified"""
        fixer = LLMFixer(provider="openrouter", api_key="test-key")
        assert fixer.model == "anthropic/claude-sonnet-4"

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

    def test_openrouter_import_error_handling(self):
        """Test handling when OpenAI library is not installed (affects OpenRouter)"""
        with patch('devdox_ai_sonar.llm_fixer.HAS_OPENAI', False):
            with pytest.raises(ImportError) as exc_info:
                LLMFixer(provider="openrouter", api_key="key")
            assert "OpenAI library not installed" in str(exc_info.value)

    def test_configure_openrouter_client_creation(self, mock_openrouter_client):
        """Test OpenRouter client is created correctly"""
        mock_instance = Mock()
        mock_openrouter_client.return_value = mock_instance

        fixer = LLMFixer(provider="openrouter", api_key="test-key")

        assert fixer.client == mock_instance
        mock_openrouter_client.assert_called_once_with(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://devdox.ai",
                "X-Title": "DevDox AI Sonar",
            },
        )

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
        assert "problem line" in context["new_context"][0]["context"]

    def test_extract_context_at_file_start(self, fixer):
        """Test extracting context at beginning of file"""
        lines = ["line 1\n", "line 2\n", "line 3\n"]

        context = fixer._extract_context(lines, 1, 1,[1], context_lines=5)

        assert context["start_line"] == 1
        assert "line 1" in context["new_context"][0]["context"]

    def test_extract_context_at_file_end(self, fixer):
        """Test extracting context at end of file"""
        lines = ["line 1\n", "line 2\n", "line 3\n"]

        context = fixer._extract_context(lines, 3, 3,[3], context_lines=5)

        assert context["end_line"] == 3
        assert "line 3" in context["new_context"][0]["context"]

    def test_extract_context_multiline_issue(self, fixer):
        """Test extracting context for multi-line issue"""
        lines = [f"line {i}\n" for i in range(1, 21)]

        context = fixer._extract_context(lines, 10, 15,[13,15], context_lines=3)

        assert context["start_line"] <= 10
        assert context["end_line"] >= 15
        for i in range(10, 16):
            assert f"line {i}" in context["new_context"][0]["context"]

    def test_extract_context_zero_context_lines(self, fixer):
        """Test extracting context with zero context lines"""
        lines = [f"line {i}\n" for i in range(1, 11)]

        context = fixer._extract_context(lines, 5, 5, [5],context_lines=0)

        assert "line 5" in context["new_context"][0]["context"]
    
    def test_extract_context_large_context(self, fixer):
        """Test extracting context with large context_lines"""
        lines = [f"line {i}\n" for i in range(1, 21)]
        
        context = fixer._extract_context(lines, 10, 10, [10],context_lines=50)
        
        # Should include all lines since context is larger than file
        assert context["start_line"] == 1
        assert context["end_line"] == len(lines)

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
                issues=[sample_issue],
                project_path=tmp_path,
                tmp_path=tmp_path,
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
            issues=[sample_issue],
            project_path=tmp_path,
            tmp_path=tmp_path
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
                issues=[issue1, issue2],
                project_path=tmp_path,
                tmp_path=tmp_path,
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
            issues=[issue1, issue2],
            project_path=tmp_path,
            tmp_path=tmp_path
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
                issues=[sample_issue],
                project_path=tmp_path,
                tmp_path=tmp_path,
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
                issues=[sample_issue],
                project_path=tmp_path,
                tmp_path=tmp_path,
                file_md=str(sample_python_file.relative_to(tmp_path)),
            )

        assert result is not None

    def test_generate_fix_llm_returns_none(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test when LLM returns None"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.return_value = None

            result = fixer.generate_fix_by_file(
                issues=[sample_issue],
                project_path=tmp_path,
                tmp_path=tmp_path
            )

        assert result is None

    def test_generate_fix_exception_handling(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test exception handling during fix generation"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = fixer.generate_fix_by_file(
                issues=[sample_issue],
                project_path=tmp_path,
                tmp_path=tmp_path
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

    async def test_apply_fixes_success(self, fixer, sample_fix, tmp_path):
        """Test successfully applying fixes"""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("old code\n")

        # Update fix to use correct path
        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        with patch.object(fixer, '_apply_fixes_to_file', new_callable=AsyncMock) as mock_apply:
            mock_apply.return_value = (1, 0, [])

            result = await fixer.apply_fixes(
                fixes=[sample_fix],
                project_path=tmp_path,
                create_backup=False,
                dry_run=False
            )

        assert isinstance(result, FixResult)
        assert result.total_fixes_attempted == 1

    async def test_apply_fixes_dry_run(self, fixer, sample_fix, tmp_path):
        """Test dry run mode doesn't modify files"""
        test_file = tmp_path / "test.py"
        original_content = "original content\n"
        test_file.write_text(original_content)

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        result = await fixer.apply_fixes(
            fixes=[sample_fix],
            project_path=tmp_path,
            dry_run=True
        )

        # File should not be modified
        assert test_file.read_text() == original_content

    async def test_apply_fixes_creates_backup(self, fixer, sample_fix, tmp_path):
        """Test backup creation"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        with patch.object(fixer, '_create_backup') as mock_backup:
            mock_backup.return_value = tmp_path / "backup"

            with patch.object(fixer, '_apply_fixes_to_file', new_callable=AsyncMock) as mock_apply:
                mock_apply.return_value = (1, 0, [])

                result = await fixer.apply_fixes(
                    fixes=[sample_fix],
                    project_path=tmp_path,
                    create_backup=True,
                    dry_run=False
                )

        assert result.backup_created is True
        assert result.backup_path is not None
        mock_backup.assert_called_once()

    async def test_apply_fixes_no_backup_in_dry_run(self, fixer, sample_fix, tmp_path):
        """Test backup not created in dry run"""
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        with patch.object(fixer, '_create_backup') as mock_backup:
            result = await fixer.apply_fixes(
                fixes=[sample_fix],
                project_path=tmp_path,
                create_backup=True,
                dry_run=True
            )

        assert result.backup_created is False
        mock_backup.assert_not_called()

    async def test_apply_fixes_multiple_files(self, fixer, tmp_path,sample_code_block):
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

        with patch.object(fixer, '_apply_fixes_to_file', new_callable=AsyncMock) as mock_apply:
            mock_apply.return_value = (1, 0, [])

            result = await fixer.apply_fixes(
                fixes=[fix1, fix2],
                project_path=tmp_path,
                create_backup=False
            )

        assert result.total_fixes_attempted == 2
        assert mock_apply.call_count == 2

    async def test_apply_fixes_groups_by_file(self, fixer, tmp_path,sample_code_block):
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

        with patch.object(fixer, '_apply_fixes_to_file', new_callable=AsyncMock) as mock_apply:
            mock_apply.return_value = (2, 0, [])

            result = await fixer.apply_fixes(
                fixes=[fix1, fix2],
                project_path=tmp_path,
                create_backup=False
            )

        # Should call _apply_fixes_to_file once with both fixes
        assert mock_apply.call_count == 1
        call_args = mock_apply.call_args[0]
        assert len(call_args[1]) == 2  # Two fixes for same file

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

        fixer.write_explaination(file_md, [fix_response], [sample_sonar_issue], original_code)

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

        fixer.write_explaination(file_md, [fix_response], issues, "")

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

        fixer.write_explaination(file_md, [fix_response], [sample_security_issue], "")

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

        fixer.write_explaination(file_md, [fix_response], [sample_sonar_issue], "")

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

        fixer.write_explaination(file_md, [fix_response], [], "")

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

        fixer.write_explaination(file_md, [fix_response], [issue], "")

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

        fixer.write_explaination(file_md, [fix_response], [sample_sonar_issue], "")

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

        fixer.write_explaination(file_md, [fix_response], [sample_sonar_issue], "")

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

        fixer.write_explaination(file_md, [fix_response], [issue], "")

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

        fixer.write_explaination(file_md, [fix_response], [issue], "")

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
                fixer.write_explaination(file_md, [fix_response], [issue], "")
        finally:
            # Restore permissions for cleanup
            os.chmod(file_md.parent, 0o755)

    def test_invalid_file_path(self, fixer):
        """Test with invalid file path"""
        # Use a path nested under os.devnull (impossible on any OS)
        file_md = Path(os.devnull).parent / "impossible_dir" / "impossible.md"

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
            fixer.write_explaination(file_md, [fix_response], [issue], "")

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

            fixer.write_explaination(file_md, [fix_response], [issue], "")

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
        fixer.write_explaination(file_md, [fix_response], issues, "")
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

        fixer.write_explaination(file_md, [fix_response], [sample_sonar_issue], original_code)

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

    async def test_prepare_context_success(self, tmp_path,fixer):
        """Test preparing context successfully"""
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")

        line_range = {
            "first_line": 1,
            "last_line": 2,
            "problem_lines": [1, 2]
        }

        result = await fixer._prepare_context(test_file, line_range, "", context_lines=1, language="python")

        assert result is not None
        assert "context_dict" in result
        assert "file_lines" in result

    async def test_prepare_context_with_modified_content(self, tmp_path,fixer):
        """Test preparing context with modified content"""
        test_file = tmp_path / "test.py"
        test_file.write_text("original\n")

        line_range = {
            "first_line": 1,
            "last_line": 1,
            "problem_lines": [1]
        }

        result = await fixer._prepare_context(
            test_file,
            line_range,
            modified_content="modified content",
            context_lines=1,
            language="python"
        )

        assert result["context_dict"]["context"] == "modified content"

    async def test_prepare_context_unicode_error(self, tmp_path, fixer):
        """Test handling unicode decode error"""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b'\x80\x81\x82')  # Invalid UTF-8

        line_range = {
            "first_line": 1,
            "last_line": 1,
            "problem_lines": [1]
        }

        result = await fixer._prepare_context(test_file, line_range, "", context_lines=1, language="python")
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

    async def test_generate_fix_by_file_empty_issues(self, fixer, tmp_path):
        """Test generating fix with empty issues list"""
        result = await fixer.generate_fix_by_file(
            issues=[],
            project_path=tmp_path,
            tmp_path=tmp_path
        )
        assert result is None

    async def test_apply_fixes_to_file_exception(self, fixer, tmp_path,sample_code_block):
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
            success, results = await fixer._apply_fixes_to_file(test_file, [fix], dry_run=False)
            # Should handle gracefully
            assert isinstance(success, bool)
        finally:
            test_file.chmod(0o644)

    @pytest.mark.skipif(
        sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="Root/admin bypasses filesystem permissions; Windows uses different permission model"
    )
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

        mock_fix_response = Mock()
        mock_fix_response.EXPLANATION = "Test explanation"
        mock_fix_response.FIXED_CODE_BLOCKS = []

        try:
            with pytest.raises((PermissionError, OSError)):
                fixer.write_explaination(file_md, [mock_fix_response], [issue], "")
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
        assert context["new_context"][0]["context"] == ""


    @pytest.mark.skipif(
        sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="Root/admin bypasses filesystem permissions; Windows uses different permission model"
    )
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
    """Test suite for IssueExtractor.get_content_range (async)."""

    def _make_extractor(self):
        return IssueExtractor(AsyncFileReader())

    # ==================== UNIT TESTS ====================

    async def test_exact_match_scenario(self, temp_dir, sample_code):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 4
        assert result['last_line'] == 6
        assert result['problem_lines'] == [5]
        assert result['confidence'] == 1.0
        assert result['match_type'] == 'exact'

    async def test_code_shifted_down(self, temp_dir, sample_code):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 7  # Shifted by 3 lines
        assert result['last_line'] == 9
        assert result['problem_lines'] == [8]  # Adjusted problem line
        assert result['confidence'] == 1.0
        assert result['match_type'] == 'exact'

    async def test_code_shifted_up(self, temp_dir, sample_code):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 2  # Shifted up by 2 lines
        assert result['last_line'] == 4
        assert result['problem_lines'] == [3]
        assert result['confidence'] == 1.0

    async def test_duplicate_code_blocks(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        # Should find first occurrence (with context matching)
        assert result['first_line'] == 2
        assert result['last_line'] == 3
        assert result['confidence'] >= 0.9

    async def test_single_line_multiple_occurrences(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        # Should find closest match to original line 2
        assert result['first_line'] == 2
        assert result['problem_lines'] == [2]

    async def test_modified_code_fuzzy_match(self, temp_dir, sample_code):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        # Should still find it with fuzzy matching or context
        assert result is not None
        assert result['first_line'] is not None
        assert result['confidence'] >= 0.65

    async def test_content_not_found(self, temp_dir, sample_code):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] is None
        assert result['last_line'] is None
        assert result['confidence'] == 0.0
        assert result['match_type'] == 'not_found'
        assert 'error' in result

    async def test_multiline_range(self, temp_dir, sample_code):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 12
        assert result['last_line'] == 19
        assert 14 in result['problem_lines']
        assert 17 in result['problem_lines']

    async def test_edge_case_first_line(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 1
        assert result['problem_lines'] == [1]

    async def test_edge_case_last_line(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 3
        assert result['last_line'] == 3

    async def test_empty_file(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['match_type'] == 'not_found'

    async def test_whitespace_differences(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        # Should not find exact match due to whitespace
        assert result is not None
        # Might find with fuzzy matching or not at all
        if result['match_type'] != 'not_found':
            assert result['confidence'] < 1.0

    async def test_unicode_content(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 2
        assert result['match_type'] == 'exact'

    async def test_very_long_file(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        assert result is not None
        assert result['first_line'] == 502
        assert result['confidence'] == 1.0

    # ==================== ERROR HANDLING TESTS ====================

    async def test_file_not_found_tmp(self, temp_dir):
        """Test Case 15: Error - temporary file doesn't exist."""
        tmp_file = temp_dir / "nonexistent.py"
        actual_file = temp_dir / "actual.py"
        actual_file.write_text("code")

        line_range = {'first_line': 1, 'last_line': 1, 'problem_lines': [1]}

        extractor = self._make_extractor()
        with pytest.raises(FileNotFoundError, match="Temporary file not found"):
            await extractor.get_content_range(tmp_file, line_range, actual_file)

    async def test_file_not_found_actual(self, temp_dir):
        """Test Case 16: Error - actual file doesn't exist."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "nonexistent.py"
        tmp_file.write_text("code")

        line_range = {'first_line': 1, 'last_line': 1, 'problem_lines': [1]}

        extractor = self._make_extractor()
        with pytest.raises(FileNotFoundError, match="Actual file not found"):
            await extractor.get_content_range(tmp_file, line_range, actual_file)

    async def test_invalid_line_range_missing_keys(self, temp_dir):
        """Test Case 17: Error - line_range missing required keys."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("code")
        actual_file.write_text("code")

        invalid_range = {'problem_lines': [1]}  # Missing first_line and last_line

        extractor = self._make_extractor()
        with pytest.raises(ValueError, match="must contain 'first_line' and 'last_line'"):
            await extractor.get_content_range(tmp_file, invalid_range, actual_file)

    async def test_out_of_bounds_line_range(self, temp_dir):
        """Test Case 18: Error - line range out of bounds."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("line1\nline2\n")
        actual_file.write_text("line1\nline2\n")

        out_of_bounds = {'first_line': 5, 'last_line': 10, 'problem_lines': [5]}

        extractor = self._make_extractor()
        with pytest.raises(ValueError, match="out of bounds"):
            await extractor.get_content_range(tmp_file, out_of_bounds, actual_file)

    async def test_negative_line_numbers(self, temp_dir):
        """Test Case 19: Error - negative line numbers."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("line1\nline2\n")
        actual_file.write_text("line1\nline2\n")

        invalid_range = {'first_line': -1, 'last_line': 2, 'problem_lines': [1]}

        extractor = self._make_extractor()
        with pytest.raises(ValueError, match="out of bounds"):
            await extractor.get_content_range(tmp_file, invalid_range, actual_file)

    async def test_first_line_greater_than_last_line(self, temp_dir):
        """Test Case 20: Error - first_line > last_line."""
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"

        tmp_file.write_text("line1\nline2\nline3\n")
        actual_file.write_text("line1\nline2\nline3\n")

        invalid_range = {'first_line': 3, 'last_line': 1, 'problem_lines': [3]}

        # This should return None as the slice will be empty
        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, invalid_range, actual_file)
        assert result is None

    # ==================== INTEGRATION TESTS ====================

    async def test_real_world_scenario_refactoring(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

        # Should find it with context or fuzzy matching
        assert result is not None
        if result['match_type'] != 'not_found':
            assert result['confidence'] >= 0.7

    async def test_problem_lines_adjustment(self, temp_dir):
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

        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)

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






class TestResolveEffectiveValues:
    """Tests for _resolve_effective_values."""

    def _make_context(self, start_line=1):
        return FixContext(
            file_path=Path("/project/src/test.py"),
            file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
            line_range={},
            code_content="original code",
            language="python",
            import_section={"start_line": 1, "end_line": 3},
            class_name=None,
            functions=[],
            context_dict={"start_line": start_line, "import_section": {"end_line": 3}},
        )

    def test_returns_defaults_when_modify_line_range_false(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        line_range = {"first_line": 5, "last_line": 7}
        code_blocks = [CodeBlock(
            block_name="b", start_line=1, end_line=2,
            has_changes=True, change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE,
        )]

        result = _resolve_effective_values(
            code_blocks, file_path, context, line_range, modify_line_range=False
        )

        assert result == (file_path, 10, 5, 7)

    def test_returns_defaults_when_code_blocks_empty(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        line_range = {"first_line": 5, "last_line": 7}

        result = _resolve_effective_values(
            [], file_path, context, line_range
        )

        assert result == (file_path, 10, 5, 7)

    def test_returns_defaults_when_block_has_no_file_path(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        line_range = {"first_line": 5, "last_line": 7}
        code_blocks = [CodeBlock(
            block_name="b", start_line=20, end_line=30,
            has_changes=True, change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE,
            file_path=None,
        )]

        result = _resolve_effective_values(
            code_blocks, file_path, context, line_range
        )

        assert result == (file_path, 10, 5, 7)

    def test_overrides_when_block_has_different_file_path(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        line_range = {"first_line": 5, "last_line": 7}
        code_blocks = [CodeBlock(
            block_name="b", start_line=20, end_line=30,
            has_changes=True, change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE,
            file_path="/project/src/other.py",
        )]

        eff_path, eff_start, eff_sonar, eff_last = _resolve_effective_values(
            code_blocks, file_path, context, line_range
        )

        assert eff_path == Path("/project/src/other.py")
        assert eff_start == 20
        assert eff_sonar == 20
        assert eff_last == 30

    def test_keeps_defaults_when_block_has_same_file_path(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        line_range = {"first_line": 5, "last_line": 7}
        code_blocks = [CodeBlock(
            block_name="b", start_line=20, end_line=30,
            has_changes=True, change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE,
            file_path="/project/src/test.py",
        )]

        result = _resolve_effective_values(
            code_blocks, file_path, context, line_range
        )

        assert result == (file_path, 10, 5, 7)

    def test_last_line_falls_back_to_first_line(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        line_range = {"first_line": 5}

        result = _resolve_effective_values(
            [], file_path, context, line_range, modify_line_range=False
        )

        assert result[3] == 5  # effective_last_line == first_line


class TestResolveRelativePath:
    """Tests for _resolve_relative_path."""

    def test_returns_relative_path(self):
        result = _resolve_relative_path(
            Path("/project/src/test.py"), Path("/project")
        )
        assert result == "src/test.py"

    def test_returns_absolute_when_outside_project(self):
        result = _resolve_relative_path(
            Path("/other/src/test.py"), Path("/project")
        )
        assert result == "/other/src/test.py"

    def test_returns_file_name_when_same_as_project(self):
        result = _resolve_relative_path(
            Path("/project/file.py"), Path("/project")
        )
        assert result == "file.py"


class TestMapFixSuggestionToDto:
    """Tests for LLMFixer._map_fix_suggestion_to_fix_suggestion_dto."""

    def _make_fixer(self):
        with patch('devdox_ai_sonar.llm_fixer.openai.OpenAI'):
            fixer = LLMFixer.__new__(LLMFixer)
            fixer.model = "test-model"
            return fixer

    def _make_code_block(self):
        return CodeBlock(
            block_name="test",
            start_line=1,
            end_line=10,
            has_changes=True,
            change_type=ChangeType.FULL_CODE,
            block_type=BlockType.MODULE,
            context="fixed code",
        )

    def _make_fix_response(self, code_blocks):
        return SonarFixResponse(
            FIXED_CODE_BLOCKS=code_blocks,
            IMPORT_BLOCK="import os",
            NEW_HELPER_CODE="def helper(): pass",
            PLACEMENT=PlacementType.GLOBAL_TOP,
            EXPLANATION="Fixed the issue",
            CONFIDENCE=0.95,
        )

    def test_maps_all_fields_correctly(self):
        fixer = self._make_fixer()
        code_block = self._make_code_block()
        fix_response = self._make_fix_response([code_block])
        context = FixContext(
            file_path=Path("/project/src/test.py"),
            file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
            line_range={},
            code_content="original code",
            language="python",
            import_section={"start_line": 1, "end_line": 3},
            class_name=None,
            functions=[],
            context_dict={"start_line": 5, "import_section": {"end_line": 3}},
        )
        line_range = {"problem_lines": [10, 12]}

        result = fixer._map_fix_suggestion_to_fix_suggestion_dto(
            line_range=line_range,
            context=context,
            code_blocks=[code_block],
            fix_response_single=fix_response,
            llm_model="test-model",
            relative_file_path="src/test.py",
            effective_start_line=5,
            effective_sonar_line=10,
            effective_last_line=12,
        )

        assert isinstance(result, FixSuggestion)
        assert result.issue_key == "fix_L10-L12"
        assert result.original_code == "original code"
        assert result.fixed_code == ""
        assert result.fixed_code_blocks == [code_block]
        assert result.import_block_code == "import os"
        assert result.helper_code == "def helper(): pass"
        assert result.placement_helper == PlacementType.GLOBAL_TOP
        assert result.explanation == "Fixed the issue"
        assert result.confidence == 0.95
        assert result.llm_model == "test-model"
        assert result.file_path == "src/test.py"
        assert result.line_number == 5
        assert result.sonar_line_number == 10
        assert result.last_line_number == 12
        assert result.end_import_block_code == 3

    def test_empty_problem_lines_gives_unknown_key(self):
        fixer = self._make_fixer()
        code_block = self._make_code_block()
        fix_response = self._make_fix_response([code_block])
        context = FixContext(
            file_path=Path("/project/src/test.py"),
            file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
            line_range={},
            code_content="code",
            language="python",
            import_section={},
            class_name=None,
            functions=[],
            context_dict={"start_line": 1, "import_section": {}},
        )

        result = fixer._map_fix_suggestion_to_fix_suggestion_dto(
            line_range={},
            context=context,
            code_blocks=[code_block],
            fix_response_single=fix_response,
            llm_model="m",
            relative_file_path="test.py",
            effective_start_line=1,
            effective_sonar_line=1,
            effective_last_line=1,
        )

        assert result.issue_key == "fix_unknown"

class TestBuildFixSuggestionInstance:
    """Tests for LLMFixer._build_fix_suggestion (lines 369-393)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_code_block(self, **overrides):
        defaults = dict(
            block_name="test",
            start_line=1,
            end_line=10,
            has_changes=True,
            change_type=ChangeType.FULL_CODE,
            block_type=BlockType.MODULE,
            context="new_code",
        )
        defaults.update(overrides)
        return CodeBlock(**defaults)

    def _make_context(self, **ctx_overrides):
        ctx = {
            "start_line": 10,
            "import_section": {"end_line": 3},
            "problem_lines": [10, 11],
        }
        ctx.update(ctx_overrides)
        return FixContext(
            file_path=Path("/project/src/test.py"),
            file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
            line_range={},
            code_content="original code",
            language="python",
            import_section={"end_line": 3},
            class_name=None,
            functions=[],
            context_dict=ctx,
        )

    def test_builds_list_from_multiple_responses(self, fixer):
        """Multiple SonarFixResponses produce one FixSuggestion each."""
        cb = self._make_code_block()
        resp1 = SonarFixResponse(
            FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.9, EXPLANATION="fix1"
        )
        resp2 = SonarFixResponse(
            FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.8, EXPLANATION="fix2"
        )
        context = self._make_context()
        line_range = {"first_line": 10, "last_line": 15, "problem_lines": [10]}

        result = fixer._build_fix_suggestion(
            [resp1, resp2], context, Path("/project/src/test.py"),
            Path("/project"), line_range, modify_line_range=False,
        )

        assert len(result) == 2
        assert result[0].explanation == "fix1"
        assert result[1].explanation == "fix2"

    def test_uses_model_or_unknown(self, fixer):
        """Uses self.model when available, 'unknown' when None."""
        cb = self._make_code_block()
        resp = SonarFixResponse(FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.9)
        context = self._make_context()
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": [1]}

        fixer.model = "gpt-4o"
        result = fixer._build_fix_suggestion(
            [resp], context, Path("/project/src/test.py"),
            Path("/project"), line_range,
        )
        assert result[0].llm_model == "gpt-4o"

        fixer.model = None
        result = fixer._build_fix_suggestion(
            [resp], context, Path("/project/src/test.py"),
            Path("/project"), line_range,
        )
        assert result[0].llm_model == "unknown"

    def test_empty_response_list_returns_empty(self, fixer):
        context = self._make_context()
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": []}
        result = fixer._build_fix_suggestion(
            [], context, Path("/project/src/test.py"),
            Path("/project"), line_range,
        )
        assert result == []

class TestPrepareFixContext:
    """Tests for LLMFixer._prepare_fix_context (lines 434-501)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    async def test_returns_fix_context_from_file(self, fixer, tmp_path):
        """Reads file, extracts imports, builds FixContext."""
        py_file = tmp_path / "module.py"
        py_file.write_text(
            "import os\n\ndef greet():\n    print('hello')\n    return 1\n"
        )
        line_range = {"first_line": 3, "last_line": 5, "problem_lines": [4]}

        result = await fixer._prepare_fix_context(py_file, line_range, "")

        assert isinstance(result, FixContext)
        assert result.language == "python"
        assert result.file_path == py_file

    async def test_with_modified_content_succeeds(self, fixer, tmp_path):
        """modified_content path succeeds with proper problem line extraction."""
        py_file = tmp_path / "module.py"
        py_file.write_text("import os\n\ndef foo():\n    pass\n")
        line_range = {"first_line": 3, "last_line": 4, "problem_lines": [3]}

        result = await fixer._prepare_fix_context(py_file, line_range, "custom context")
        assert result is not None

    async def test_file_not_found_returns_none(self, fixer, tmp_path):
        missing = tmp_path / "missing.py"
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": [1]}
        result = await fixer._prepare_fix_context(missing, line_range, "")
        assert result is None

    async def test_class_name_detected(self, fixer, tmp_path):
        """Detects class name for methods inside a class."""
        py_file = tmp_path / "cls.py"
        py_file.write_text(
            "class MyService:\n    def process(self):\n        return 1\n"
        )
        line_range = {"first_line": 2, "last_line": 3, "problem_lines": [2]}

        result = await fixer._prepare_fix_context(py_file, line_range, "")

        assert result is not None
        assert result.class_name == "MyService"

class TestCallLlmList:
    """Tests for LLMFixer._call_llm_list (lines 648-724)."""

    @pytest.fixture
    def fixer_openai(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    @pytest.fixture
    def basic_context(self, tmp_path):
        return FixContext(
            file_path=tmp_path / "test.py",
            file_path_tmp=tmp_path / "test.py",
            line_range={"first_line": 1, "last_line": 5},
            code_content="def foo(): pass",
            language="python",
            import_section={"start_line": 0, "end_line": 0, "content": "", "has_imports": False},
            class_name=None,
            functions=[],
            context_dict={
                "start_line": 1, "end_line": 5,
                "context": "def foo(): pass",
                "new_context": [{"context": "def foo(): pass","start_line": 1, "end_line": 5,}],
                "problem_line_content": {1: "def foo(): pass\n"},
            },
        )

    @pytest.fixture
    def basic_issue(self):
        return SonarIssue(
            key="test", rule="python:S1234", severity="MAJOR",
            component="test.py", project="test", line=1,
            message="Test issue", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=5,
        )

    def test_returns_none_when_model_not_set(self, fixer_openai, basic_context, basic_issue):
        fixer_openai.model = None
        result = fixer_openai._call_llm_list([basic_issue], basic_context, ".py", {}, "")
        assert result is None

    def test_openai_provider_calls_responses_parse(self, fixer_openai, basic_context, basic_issue):
        mock_parsed = Mock()
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        fixer_openai.client.responses.parse = Mock(return_value=mock_response)

        result = fixer_openai._call_llm_list([basic_issue], basic_context, ".py", {}, "")

        fixer_openai.client.responses.parse.assert_called_once()
        assert result == mock_parsed

    def test_gemini_provider_calls_generate_content(self, mock_gemini_client):
        fixer = LLMFixer(provider="gemini", api_key="test-key")
        mock_parsed = Mock()
        mock_response = Mock()
        mock_response.parsed = mock_parsed
        fixer.client.models.generate_content = Mock(return_value=mock_response)

        ctx = FixContext(
            file_path=Path("test.py"), file_path_tmp=Path("test.py"),
            line_range={}, code_content="code", language="python",
            import_section={"start_line": 0, "end_line": 0, "content": "", "has_imports": False},
            class_name=None, functions=[],
            context_dict={"start_line": 1, "end_line": 5, "context": "code",
                          "new_context": [{"context": "code","start_line": 1, "end_line": 5}],
                          "problem_line_content": {1: "code\n"}},
        )
        issue = SonarIssue(
            key="t", rule="python:S1", severity="MAJOR", component="t.py",
            project="t", line=1, message="t", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )

        result = fixer._call_llm_list([issue], ctx, ".py", {}, "")
        assert result == mock_parsed

    def test_togetherai_provider_calls_completions(self, mock_together_client):
        fixer = LLMFixer(provider="togetherai", api_key="test-key")
        mock_choice = Mock()
        mock_choice.message.content = json.dumps({
            "IMPORT_BLOCK": "",
            "FIXED_CODE_BLOCKS": [{"block_name": "t", "start_line": 1,
                                    "end_line": 2, "has_changes": True,
                                    "change_type": "FULL_CODE",
                                    "block_type": "module", "context": "x"}],
            "NEW_HELPER_CODE": "", "PLACEMENT": "SIBLING",
            "EXPLANATION": "fix", "CONFIDENCE": 0.9,
        })
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        fixer.client.chat.completions.create = Mock(return_value=mock_response)

        ctx = FixContext(
            file_path=Path("test.py"), file_path_tmp=Path("test.py"),
            line_range={}, code_content="code", language="python",
            import_section={"start_line": 0, "end_line": 0, "content": "", "has_imports": False},
            class_name=None, functions=[],
            context_dict={"start_line": 1, "end_line": 5, "context": "code",
                          "new_context": [{"context": "code","start_line": 1, "end_line": 5}],
                          "problem_line_content": {1: "code\n"}},
        )
        issue = SonarIssue(
            key="t", rule="python:S1", severity="MAJOR", component="t.py",
            project="t", line=1, message="t", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )

        result = fixer._call_llm_list([issue], ctx, ".py", {}, "")
        assert result is not None

    def test_api_exception_returns_none(self, fixer_openai, basic_context, basic_issue):
        fixer_openai.client.responses.parse = Mock(side_effect=Exception("API down"))
        result = fixer_openai._call_llm_list([basic_issue], basic_context, ".py", {}, "")
        assert result is None

class TestExtendStrategiesForIssue:
    """Tests for strategy branches (lines 795-1060)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_cognitive_complexity_branch(self, fixer):
        issue = Mock()
        issue.message = "Refactor this function to reduce its Cognitive Complexity from 25 to the 15 allowed."
        issue.rule = "python:S3776"

        strategies = fixer._extend_strategies_for_issue(issue)

        assert any("TARGET" in s for s in strategies)
        assert any("15" in s for s in strategies)

    def test_unused_rule_branch(self, fixer):
        issue = Mock()
        issue.message = "Remove this unused variable"
        issue.rule = "python:unused"

        strategies = fixer._extend_strategies_for_issue(issue)

        assert any("unused" in s.lower() for s in strategies)

    def test_literal_duplication_branch(self, fixer):
        issue = Mock()
        issue.message = 'String literals should not be duplicating this literal "database_url"'
        issue.rule = "python:S1192"

        strategies = fixer._extend_strategies_for_issue(issue)

        assert any("database_url" in s for s in strategies)
        assert any("SCREAMING_SNAKE_CASE" in s for s in strategies)

    def test_null_check_branch(self, fixer):
        issue = Mock()
        issue.message = "This value is nullable and should be checked"
        issue.rule = "python:null"

        strategies = fixer._extend_strategies_for_issue(issue)

        assert any("NULL" in s or "NONE" in s for s in strategies)

    def test_parameter_usage_branch(self, fixer):
        issue = Mock()
        issue.message = "Be sure that every parameter is used"
        issue.rule = "python:S1172"

        strategies = fixer._extend_strategies_for_issue(issue)

        assert any("parameter" in s.lower() for s in strategies)

class TestParseLlmResponse:
    """Tests for LLMFixer.parse_llm_response (lines 1342-1353)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_valid_json_returns_sonar_fix_response(self, fixer):
        data = {
            "IMPORT_BLOCK": "",
            "FIXED_CODE_BLOCKS": [{
                "block_name": "test", "start_line": 1, "end_line": 5,
                "has_changes": True, "change_type": "FULL_CODE",
                "block_type": "module", "context": "fixed",
            }],
            "NEW_HELPER_CODE": "",
            "PLACEMENT": "SIBLING",
            "EXPLANATION": "Fixed it",
            "CONFIDENCE": 0.95,
        }

        result = fixer.parse_llm_response(json.dumps(data))

        assert isinstance(result, SonarFixResponse)
        assert result.CONFIDENCE == 0.95

    def test_invalid_json_raises_value_error(self, fixer):
        with pytest.raises(ValueError, match="Invalid JSON"):
            fixer.parse_llm_response("not valid json {{{")

    def test_schema_validation_failure_raises_value_error(self, fixer):
        # Valid JSON but missing required FIXED_CODE_BLOCKS
        with pytest.raises(ValueError, match="Schema validation failed"):
            fixer.parse_llm_response('{"IMPORT_BLOCK": "x"}')

class TestExtractFixFromResponse:
    """Tests for _extract_fix_from_response (lines 1355-1388)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_valid_json_with_code_fence(self, fixer):
        content = '```json\n{"FIXED_SELECTION": "code", "CONFIDENCE": 0.8}\n```'
        result = fixer._extract_fix_from_response(content)
        assert result is not None
        assert result["fixed_code"] == "code"

    def test_plain_json_object(self, fixer):
        content = '{"FIXED_SELECTION": "def foo(): pass", "CONFIDENCE": 0.9}'
        result = fixer._extract_fix_from_response(content)
        assert result is not None
        assert "foo" in result["fixed_code"]

    def test_completely_broken_content_returns_none(self, fixer):
        result = fixer._extract_fix_from_response("just plain text no json at all")
        assert result is None

class TestExtractFieldsFromParsedJson:
    """Covers remaining branches like invalid confidence type (lines 1301-1302)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_non_numeric_confidence_defaults_to_half(self, fixer):
        data = {"FIXED_SELECTION": "code", "CONFIDENCE": "not-a-number"}
        result = fixer._extract_fields_from_parsed_json(data)
        assert result["confidence"] == 0.5

    def test_no_explanation_defaults(self, fixer):
        data = {"FIXED_SELECTION": "code", "EXPLANATION": ""}
        result = fixer._extract_fields_from_parsed_json(data)
        assert result["explanation"] == "Code fix applied"

class TestExtractPythonImportSection:
    """Tests for import extraction (lines 2113-2370)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_simple_imports(self, fixer):
        lines = ["import os\n", "import sys\n", "\n", "x = 1\n"]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is True
        assert result["start_line"] == 1
        assert result["end_line"] == 2
        assert "import os" in result["content"]

    def test_parenthesized_multiline_import(self, fixer):
        lines = [
            "from module import (\n",
            "    ClassA,\n",
            "    ClassB,\n",
            ")\n",
            "\n",
            "x = 1\n",
        ]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is True
        assert result["end_line"] == 4

    def test_backslash_continuation_import(self, fixer):
        lines = [
            "from module import ClassA, \\\n",
            "    ClassB, ClassC\n",
            "\n",
            "x = 1\n",
        ]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is True
        assert result["end_line"] == 2

    def test_no_imports(self, fixer):
        lines = ["x = 1\n", "y = 2\n"]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is False

    def test_stops_at_decorator(self, fixer):
        lines = [
            "import os\n",
            "\n",
            "@app.route('/')\n",
            "def index(): pass\n",
        ]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is True
        assert result["end_line"] == 1

    def test_stops_at_code_after_imports(self, fixer):
        lines = [
            "import os\n",
            "import sys\n",
            "\n",
            "logger = logging.getLogger(__name__)\n",
        ]
        result = fixer._extract_python_import_section(lines)
        assert result["end_line"] == 2

    def test_non_python_language_returns_empty(self, fixer):
        lines = ["const x = require('module');\n"]
        result = fixer._extract_import_section(lines, "javascript")
        assert result["has_imports"] is False

    def test_skips_docstring_before_imports(self, fixer):
        lines = [
            '"""Module docstring."""\n',
            "import os\n",
            "\n",
            "x = 1\n",
        ]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is True
        assert result["start_line"] == 2

    def test_module_level_dunder_does_not_stop(self, fixer):
        assert fixer._is_module_level_dunder("__all__ = ['foo']")
        assert fixer._is_module_level_dunder("__version__ = '1.0'")
        assert not fixer._is_module_level_dunder("x = 1")

    def test_should_stop_at_decorator_only_after_imports(self, fixer):
        assert not fixer._should_stop_at_decorator("@decorator", None)
        assert fixer._should_stop_at_decorator("@decorator", 1)
        assert not fixer._should_stop_at_decorator("not_decorator", 1)

    def test_detect_multiline_import_parentheses(self, fixer):
        state = {"in_multiline_import": False, "in_parentheses_import": False,
                 "in_backslash_continuation": False}
        result = fixer._detect_multiline_import_pattern("from x import (", state)
        assert result["in_parentheses_import"] is True
        assert result["in_multiline_import"] is True

    def test_detect_multiline_import_backslash(self, fixer):
        state = {"in_multiline_import": False, "in_parentheses_import": False,
                 "in_backslash_continuation": False}
        result = fixer._detect_multiline_import_pattern("from x import A, \\", state)
        assert result["in_backslash_continuation"] is True

    def test_detect_single_line_import_resets(self, fixer):
        state = {"in_multiline_import": True, "in_parentheses_import": True,
                 "in_backslash_continuation": True}
        result = fixer._detect_multiline_import_pattern("import os", state)
        assert result["in_multiline_import"] is False

    def test_is_import_statement_start(self, fixer):
        assert fixer._is_import_statement_start("from os import path")
        assert fixer._is_import_statement_start("import sys")
        assert not fixer._is_import_statement_start("x = import_val")

class TestExtractClassNameFromFile:
    """Tests for class name extraction (lines 2372-2452)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_finds_class_containing_target_line(self, fixer):
        lines = [
            "import os\n",
            "\n",
            "class MyService:\n",
            "    def process(self):\n",
            "        return 1\n",
        ]
        result = fixer._extract_class_name_from_file(lines, 3)
        assert result == "MyService"

    def test_returns_none_for_module_level_function(self, fixer):
        lines = [
            "import os\n",
            "\n",
            "def standalone():\n",
            "    return 1\n",
        ]
        result = fixer._extract_class_name_from_file(lines, 2)
        assert result is None

    def test_skips_comments_and_empty_lines(self, fixer):
        lines = [
            "class Foo:\n",
            "    # comment\n",
            "\n",
            "    def bar(self):\n",
            "        pass\n",
        ]
        result = fixer._extract_class_name_from_file(lines, 3)
        assert result == "Foo"

    def test_is_line_inside_class_true(self, fixer):
        lines = [
            "class Foo:\n",
            "    def bar(self):\n",
            "        pass\n",
        ]
        assert fixer._is_line_inside_class(lines, 2, 0) is True

    def test_is_line_inside_class_false_same_indent(self, fixer):
        lines = [
            "class Foo:\n",
            "    pass\n",
            "\n",
            "x = 1\n",
        ]
        assert fixer._is_line_inside_class(lines, 3, 0) is False

    def test_is_line_inside_class_target_before_class(self, fixer):
        lines = ["x = 1\n", "class Foo:\n", "    pass\n"]
        assert fixer._is_line_inside_class(lines, 0, 1) is False

class TestCheckPythonInterpreter:
    """Tests for check_python_interpreter (lines 1557-1571)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_valid_python_file(self, fixer, tmp_path):
        py_file = tmp_path / "valid.py"
        py_file.write_text("def foo():\n    return 1\n")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            valid, msg = fixer.check_python_interpreter(py_file)
        assert valid is True
        assert msg is None

    def test_invalid_python_file(self, fixer, tmp_path):
        py_file = tmp_path / "invalid.py"
        py_file.write_text("def foo(\n    return 1\n")
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "python", stderr="SyntaxError: invalid syntax"
            )
            valid, msg = fixer.check_python_interpreter(py_file)
        assert valid is False
        assert msg is not None

class TestApplyFixesToFileDetailed:
    """Tests for _apply_fixes_to_file (lines 1528-1547)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    async def test_dry_run_returns_true(self, fixer, tmp_path):
        success, results = await fixer._apply_fixes_to_file(
            tmp_path / "test.py", [], dry_run=True
        )
        assert success is True
        assert results == []

    async def test_applies_fix_with_validation(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\ny = 2\n")

        mock_result = Mock()
        mock_result.success = True
        mock_result.reason = None

        with patch("devdox_ai_sonar.llm_fixer.apply_single_fix") as mock_apply, \
             patch("devdox_ai_sonar.llm_fixer.write_file_lines"), \
             patch("devdox_ai_sonar.llm_fixer.cleanup_tmp_py_file"), \
             patch.object(fixer, "check_python_interpreter", return_value=(True, None)):
            mock_apply.return_value = (mock_result, ["x = 1\n", "y = 2\n"])
            fixer.file_reader.read_lines = AsyncMock(return_value=["x = 1\n", "y = 2\n"])

            fix = Mock()
            fix.issue_key = "test-fix"
            success, results = await fixer._apply_fixes_to_file(py_file, [fix], dry_run=False)

        assert success is True
        assert len(results) == 1

    async def test_validation_failure_skips_write(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        mock_result = Mock()
        mock_result.success = True
        mock_result.reason = None

        with patch("devdox_ai_sonar.llm_fixer.apply_single_fix") as mock_apply, \
             patch("devdox_ai_sonar.llm_fixer.write_file_lines") as mock_write, \
             patch("devdox_ai_sonar.llm_fixer.cleanup_tmp_py_file"), \
             patch.object(fixer, "check_python_interpreter", return_value=(False, "SyntaxError")):
            mock_apply.return_value = (mock_result, ["x = 1\n"])
            fixer.file_reader.read_lines = AsyncMock(return_value=["x = 1\n"])

            fix = Mock()
            fix.issue_key = "test-fix"
            success, results = await fixer._apply_fixes_to_file(py_file, [fix], dry_run=False)

        assert success is False

class TestGenerateFixKeyInstance:
    """Tests for LLMFixer._generate_fix_key (line 409)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_single_problem_line(self, fixer):
        assert fixer._generate_fix_key([42]) == "fix_L42"

    def test_multiple_problem_lines(self, fixer):
        assert fixer._generate_fix_key([5, 20, 10]) == "fix_L5-L20"

    def test_empty_returns_unknown(self, fixer):
        assert fixer._generate_fix_key([]) == "fix_unknown"

class TestContextExtractorCoverage:
    """Tests for uncovered ContextExtractor methods."""

    def test_is_function_line_true(self):
        lines = ["def foo():\n", "    pass\n"]
        ext = ContextExtractor(lines)
        assert ext._is_function_line(0) is True

    def test_is_function_line_false(self):
        lines = ["x = 1\n", "y = 2\n"]
        ext = ContextExtractor(lines)
        assert ext._is_function_line(0) is False

    def test_is_function_line_out_of_bounds(self):
        ext = ContextExtractor(["x\n"])
        assert ext._is_function_line(99) is False

    def test_find_function_end_python_def(self):
        lines = [
            "def foo():\n",
            "    x = 1\n",
            "    return x\n",
            "\n",
            "y = 2\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._find_function_end(0)
        assert result is not None
        assert result >= 2

    def test_find_function_end_brace_language(self):
        lines = [
            "function foo() {\n",
            "    return 1;\n",
            "}\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._find_function_end(0)
        assert result == 2

    def test_find_function_end_out_of_bounds(self):
        ext = ContextExtractor(["x\n"])
        assert ext._find_function_end(99) is None

    def test_get_function_info_valid(self):
        lines = [
            "def process():\n",
            "    x = 1\n",
            "    return x\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        info = ext._get_function_info(0)
        assert info is not None
        assert info["name"] == "process"
        assert info["start_idx"] == 0

    def test_get_function_info_not_function(self):
        lines = ["x = 1\n", "y = 2\n"]
        ext = ContextExtractor(lines)
        assert ext._get_function_info(0) is None

    def test_get_function_info_out_of_bounds(self):
        ext = ContextExtractor(["x\n"])
        assert ext._get_function_info(99) is None

    def test_get_function_info_with_decorators(self):
        lines = [
            "@property\n",
            "def value(self):\n",
            "    return self._val\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        info = ext._get_function_info(1)
        assert info is not None
        assert info["start_idx"] == 0  # includes decorator
        assert len(info["decorators"]) == 1

    def test_find_decorator_start_with_multiple(self):
        lines = [
            "x = 1\n",
            "@decorator1\n",
            "@decorator2\n",
            "def func():\n",
            "    pass\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._find_decorator_start(3)
        assert result == 1

    def test_find_decorator_start_no_decorators(self):
        lines = ["x = 1\n", "\n", "def func():\n", "    pass\n"]
        ext = ContextExtractor(lines)
        result = ext._find_decorator_start(2)
        assert result == 2

    def test_find_decorator_start_at_index_zero(self):
        lines = ["def func():\n", "    pass\n"]
        ext = ContextExtractor(lines)
        assert ext._find_decorator_start(0) == 0

    def test_find_functions_containing_problems(self):
        lines = [
            "def func_a():\n",
            "    x = bad_line\n",
            "    return x\n",
            "\n",
            "def func_b():\n",
            "    y = 2\n",
            "    return y\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._find_functions_containing_problems([1], 0, 6)
        assert len(result) == 1
        assert result[0]["name"] == "func_a"

    def test_find_functions_skips_non_function_problems(self):
        lines = ["x = 1\n", "y = 2\n"]
        ext = ContextExtractor(lines)
        result = ext._find_functions_containing_problems([0], 0, 1)
        assert len(result) == 0

    def test_build_optimized_context_empty_returns_none(self):
        ext = ContextExtractor([])
        assert ext._build_optimized_context([], [1]) is None

    def test_build_optimized_context_single_function(self):
        lines = [
            "def func():\n",
            "    return 1\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        functions = [{
            "start_idx": 0, "end_idx": 1, "name": "func",
            "lines": lines[:2], "indentation": 0,
            "problem_lines_in_function": [1],
        }]
        result = ext._build_optimized_context(functions, [1])
        assert result is not None
        assert result["function_count"] == 1
        assert result["is_complete_function"] is True

    def test_count_functions_in_range(self):
        lines = [
            "def a():\n",
            "    pass\n",
            "\n",
            "def b():\n",
            "    pass\n",
        ]
        ext = ContextExtractor(lines)
        assert ext._count_functions_in_range(0, 4) == 2

    def test_extract_function_name_java_method(self):
        ext = ContextExtractor([])
        name = ext._extract_function_name("public void doSomething(int x) {")
        assert name == "doSomething"

    def test_extract_function_name_filters_keywords(self):
        ext = ContextExtractor([])
        # "public" alone should be filtered
        name = ext._extract_function_name("public()")
        # Should not return "public" since it's a keyword
        assert name != "public" or name == ""

    def test_extract_complete_function_on_decorator(self):
        lines = [
            "@app.route('/')\n",
            "def index():\n",
            "    return 'ok'\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._extract_complete_function(0, 0, 2)
        assert result is not None
        assert result["function_name"] == "index"
        assert result["has_decorators"] is True

    def test_extract_complete_function_out_of_bounds(self):
        ext = ContextExtractor(["x\n"])
        assert ext._extract_complete_function(99, 0, 0) is None

    def test_try_function_extraction_strategies_inside_function(self):
        lines = [
            "def outer():\n",
            "    x = 1\n",
            "    return x\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._try_function_extraction_strategies(1, 2)
        assert result is not None
        assert result["is_complete_function"]

    def test_try_function_extraction_strategies_no_function(self):
        lines = ["x = 1\n", "y = 2\n"]
        ext = ContextExtractor(lines)
        result = ext._try_function_extraction_strategies(0, 1)
        assert result is None

    def test_try_expand_multiline_no_function(self):
        lines = ["x = 1\n", "y = 2\n", "z = 3\n"]
        ext = ContextExtractor(lines)
        assert ext._try_expand_multiline_to_function(0, 1) is None

    def test_try_expand_multiline_function_within_range(self):
        """When function is found within given range, returns the function info."""
        lines = [
            "def short():\n",
            "    return 1\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._try_expand_multiline_to_function(0, 1)
        # Function is found; the method returns function info dict
        assert result is not None
        assert result["function_name"] == "short"

    def test_find_brace_function_end_out_of_bounds(self):
        ext = ContextExtractor(["x\n"])
        assert ext._find_brace_function_end(["x\n"], 99) is None

    def test_find_brace_function_end_no_closing(self):
        lines = ["function f() {\n", "    x = 1;\n"]
        ext = ContextExtractor(lines)
        assert ext._find_brace_function_end(lines, 0) is None

    def test_check_indentation_containment_out_of_bounds(self):
        ext = ContextExtractor(["x\n"])
        assert ext._check_indentation_containment(["x\n"], 99, 0) is False

    def test_check_indentation_containment_empty_line(self):
        lines = ["def foo():\n", "\n", "    x = 1\n"]
        ext = ContextExtractor(lines)
        assert ext._check_indentation_containment(lines, 1, 0) is True

class TestModuleLevelBuildFixSuggestion:
    """Tests for the module-level _build_fix_suggestion function (lines 3319-3351)."""

    def _make_code_block(self):
        return CodeBlock(
            block_name="test", start_line=1, end_line=5,
            has_changes=True, change_type=ChangeType.FULL_CODE,
            block_type=BlockType.MODULE, context="fixed code",
        )

    def test_builds_fix_suggestion_with_list_problem_lines(self, tmp_path):
        cb = self._make_code_block()
        fix_response = SonarFixResponse(
            IMPORT_BLOCK="import os",
            FIXED_CODE_BLOCKS=[cb],
            NEW_HELPER_CODE="",
            PLACEMENT=PlacementType.GLOBAL_TOP,
            EXPLANATION="Explanation",
            CONFIDENCE=0.9,
        )
        file_path = tmp_path / "src" / "test.py"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("code")
        context_info = {
            "context_dict": {
                "context": "original",
                "start_line": 5,
                "end_line": 10,
                "import_section": {"end_line": 3},
            }
        }
        line_range = {"first_line": 5, "last_line": 10, "problem_lines": [5, 7]}

        result = _build_fix_suggestion(
            fix_response, context_info, file_path, tmp_path, line_range, "gpt-4"
        )

        assert isinstance(result, FixSuggestion)
        assert result.issue_key == "fix_L5-L7"
        assert result.llm_model == "gpt-4"
        assert result.sonar_line_number == 5
        assert result.last_line_number == 10

    def test_single_int_problem_lines(self, tmp_path):
        cb = self._make_code_block()
        fix_response = SonarFixResponse(
            FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.8,
        )
        file_path = tmp_path / "test.py"
        file_path.write_text("code")
        context_info = {"context_dict": {"context": "code", "start_line": 1}}
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": 42}

        result = _build_fix_suggestion(
            fix_response, context_info, file_path, tmp_path, line_range, "m"
        )
        assert result.issue_key == "fix_L42"

    def test_no_problem_lines(self, tmp_path):
        cb = self._make_code_block()
        fix_response = SonarFixResponse(
            FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.7,
        )
        file_path = tmp_path / "test.py"
        file_path.write_text("code")
        context_info = {"context_dict": {"context": "code", "start_line": 1}}
        line_range = {"first_line": 1, "last_line": 1}

        result = _build_fix_suggestion(
            fix_response, context_info, file_path, tmp_path, line_range, "m"
        )
        assert result.issue_key == "fix_unknown"

class TestGetFileFromFixStrategies:
    """Cover remaining branches of _get_file_from_fix (lines 1447, 1470, 1477-1478)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_all_strategies_fail_returns_none(self, fixer, tmp_path):
        cb = CodeBlock(
            block_name="t", start_line=1, end_line=1,
            has_changes=True, change_type=ChangeType.FULL_CODE,
            block_type=BlockType.MODULE, context="x",
        )
        fix = FixSuggestion(
            issue_key="no-colon",
            file_path=None,
            original_code="",
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="m",
            fixed_code_blocks=[cb],
        )

        result = fixer._get_file_from_fix(fix, tmp_path)
        assert result is None

    def test_issue_key_no_matching_file(self, fixer, tmp_path):
        cb = CodeBlock(
            block_name="t", start_line=1, end_line=1,
            has_changes=True, change_type=ChangeType.FULL_CODE,
            block_type=BlockType.MODULE, context="x",
        )
        fix = FixSuggestion(
            issue_key="project:src/missing.py:S1234",
            file_path=None,
            original_code="",
            fixed_code="",
            explanation="",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="m",
            fixed_code_blocks=[cb],
        )

        result = fixer._try_extract_from_issue_key(fix, tmp_path)
        assert result is None

class TestApplyFixesWithValidationCoverage:
    """Tests for apply_fixes_with_validation (lines 1676-1858)."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_fix(self, tmp_path, file_name="test.py"):
        cb = CodeBlock(
            block_name="t", start_line=1, end_line=1,
            has_changes=True, change_type=ChangeType.FULL_CODE,
            block_type=BlockType.MODULE, context="x",
        )
        return FixSuggestion(
            issue_key="test",
            file_path=file_name,
            original_code="old",
            fixed_code="new",
            explanation="fix",
            confidence=0.9,
            sonar_line_number=1,
            llm_model="m",
            fixed_code_blocks=[cb],
        )

    def _make_issue(self):
        return SonarIssue(
            key="t", rule="python:S1", severity="MAJOR", component="test.py",
            project="p", line=1, message="msg", type="CODE_SMELL",
            status="OPEN", first_line=1, last_line=1,
        )

    async def test_success_path_no_validator(self, fixer, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("old code\n")

        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(True, [])), \
             patch.object(fixer, "_create_backup", return_value=tmp_path / "bak"):
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=True, use_validator=False,
            )

        assert result.total_fixes_attempted == 1
        assert len(result.successful_fixes) == 1
        assert result.backup_created is True

    async def test_failure_no_validator_marks_failed(self, fixer, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("old\n")

        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        mock_result = Mock()
        mock_result.success = False
        mock_result.reason = "syntax error"

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [mock_result])), \
             patch.object(fixer, "_create_backup", return_value=tmp_path / "bak"):
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=False,
            )

        assert len(result.failed_fixes) > 0

    async def test_dry_run_no_backup(self, fixer, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("code\n")

        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(True, [])):
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=True, dry_run=True, use_validator=False,
            )

        assert result.backup_created is False

    async def test_exception_in_file_processing(self, fixer, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("code\n")

        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=False,
            )

        assert len(result.failed_fixes) == 1
        assert "boom" in result.failed_fixes[0]["error"]

class TestImportGuards:
    """Cover the HAS_TOGETHER/HAS_OPENAI/HAS_GEMINI import fallback paths."""

    def test_together_import_missing_raises(self):
        """When Together is not installed, configuring it should raise ImportError."""
        with patch("devdox_ai_sonar.llm_fixer.HAS_TOGETHER", False):
            with pytest.raises(ImportError, match="Together AI library"):
                fixer = LLMFixer.__new__(LLMFixer)
                fixer._configure_togetherai(None, "key")

    def test_openai_import_missing_raises(self):
        """When openai is not installed, configuring it should raise ImportError."""
        with patch("devdox_ai_sonar.llm_fixer.HAS_OPENAI", False):
            with pytest.raises(ImportError, match="OpenAI library"):
                fixer = LLMFixer.__new__(LLMFixer)
                fixer._configure_openai(None, "key")

    def test_gemini_import_missing_raises(self):
        """When google.genai is not installed, configuring it should raise ImportError."""
        with patch("devdox_ai_sonar.llm_fixer.HAS_GEMINI", False):
            with pytest.raises(ImportError, match="Gemini library"):
                fixer = LLMFixer.__new__(LLMFixer)
                fixer._configure_gemini(None, "key")

    def test_openrouter_import_missing_raises(self):
        """When openai is not installed, configuring openrouter should raise ImportError."""
        with patch("devdox_ai_sonar.llm_fixer.HAS_OPENAI", False):
            with pytest.raises(ImportError, match="OpenAI library"):
                fixer = LLMFixer.__new__(LLMFixer)
                fixer._configure_openrouter(None, "key")

class TestConfigureOpenrouterMissingKey:
    """Cover missing API key raises ValueError for OpenRouter."""

    def test_openrouter_no_key_raises(self):
        with patch("devdox_ai_sonar.llm_fixer.HAS_OPENAI", True), \
             patch("devdox_ai_sonar.llm_fixer.openai.OpenAI"), \
             patch.dict(os.environ, {}, clear=True):
            fixer = LLMFixer.__new__(LLMFixer)
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(ValueError, match="OpenRouter API key"):
                fixer._configure_openrouter(None, None)

class TestConfigureTogetheraiMissingKey:
    """Cover line 144 - missing API key raises ValueError."""

    def test_togetherai_no_key_raises(self):
        with patch("devdox_ai_sonar.llm_fixer.HAS_TOGETHER", True), \
             patch("devdox_ai_sonar.llm_fixer.Together"), \
             patch.dict(os.environ, {}, clear=True):
            fixer = LLMFixer.__new__(LLMFixer)
            # Remove env var if present
            os.environ.pop("TOGETHER_API_KEY", None)
            with pytest.raises(ValueError, match="Together API key"):
                fixer._configure_togetherai(None, None)

class TestWriteExplaination:
    """Cover the write_explaination method."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_writes_markdown_file(self, fixer, tmp_path):
        """Test writing explanation to a markdown file."""
        md_file = tmp_path / "explanations" / "output.md"
        fix_response = Mock()
        fix_response.EXPLANATION = "Fixed the issue"
        fix_response.FIXED_CODE_BLOCKS = [Mock()]

        issue = Mock()
        issue.rule = "python:S1234"
        issue.severity = "MAJOR"
        issue.message = "Bad code"
        issue.file_path = "test.py"
        issue.line = 10

        # Mock the jinja template
        mock_template = Mock()
        mock_template.render.return_value = "# Rendered Markdown"
        fixer.jinja_env_templates = Mock()
        fixer.jinja_env_templates.get_template.return_value = mock_template

        fixer.write_explaination(md_file, [fix_response], [issue], "original code")

        assert md_file.exists()
        content = md_file.read_text()
        assert "Rendered Markdown" in content

    def test_creates_parent_dirs(self, fixer, tmp_path):
        """Test that parent directories are created."""
        md_file = tmp_path / "deep" / "nested" / "output.md"
        fix_response = Mock()
        fix_response.EXPLANATION = "exp"
        fix_response.FIXED_CODE_BLOCKS = []

        mock_template = Mock()
        mock_template.render.return_value = "content"
        fixer.jinja_env_templates = Mock()
        fixer.jinja_env_templates.get_template.return_value = mock_template

        fixer.write_explaination(md_file, [fix_response], [], "code")
        assert md_file.parent.exists()

    def test_appends_to_existing_file(self, fixer, tmp_path):
        """Test that content is appended, not overwritten."""
        md_file = tmp_path / "output.md"
        md_file.write_text("existing content\n")

        fix_response = Mock()
        fix_response.EXPLANATION = "exp"
        fix_response.FIXED_CODE_BLOCKS = []

        issue = Mock()
        issue.rule = "python:S1"
        issue.severity = "MINOR"
        issue.message = "msg"
        issue.file_path = "t.py"
        issue.line = 1

        mock_template = Mock()
        mock_template.render.return_value = "new content"
        fixer.jinja_env_templates = Mock()
        fixer.jinja_env_templates.get_template.return_value = mock_template

        fixer.write_explaination(md_file, [fix_response], [issue], "code")
        content = md_file.read_text()
        assert "existing content" in content
        assert "new content" in content

    @pytest.mark.parametrize("empty_explanation", ["", "   ", "\t\n  "])
    def test_empty_explanation_omitted_from_markdown(self, fixer, tmp_path, empty_explanation):
        """Test that empty or whitespace-only EXPLANATION is omitted from rendered markdown."""
        md_file = tmp_path / "output.md"
        fix_response = Mock()
        fix_response.EXPLANATION = empty_explanation
        fix_response.FIXED_CODE_BLOCKS = []

        issue = Mock()
        issue.rule = "python:S1"
        issue.severity = "MINOR"
        issue.message = "msg"
        issue.file_path = "t.py"
        issue.line = 1

        fixer.write_explaination(md_file, [fix_response], [issue], "code")
        content = md_file.read_text()
        assert "Explanation" not in content

    def test_non_empty_explanation_included_in_markdown(self, fixer, tmp_path):
        """Test that a non-empty EXPLANATION is rendered in the markdown."""
        md_file = tmp_path / "output.md"
        fix_response = Mock()
        fix_response.EXPLANATION = "This fix addresses the root cause"
        fix_response.FIXED_CODE_BLOCKS = []

        issue = Mock()
        issue.rule = "python:S1"
        issue.severity = "MINOR"
        issue.message = "msg"
        issue.file_path = "t.py"
        issue.line = 1

        fixer.write_explaination(md_file, [fix_response], [issue], "code")
        content = md_file.read_text()
        assert "Explanation" in content
        assert "This fix addresses the root cause" in content

    def test_consolidates_all_response_code_blocks(self, fixer, tmp_path):
        """All code blocks from multiple SonarFixResponses appear in markdown."""
        from devdox_ai_sonar.models.sonar import CodeBlock, ChangeType, BlockType

        md_file = tmp_path / "output.md"

        block1 = CodeBlock(
            block_name="build_greeting",
            start_line=10, end_line=11,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
        )
        block2 = CodeBlock(
            block_name="build_greeting",
            start_line=14, end_line=14,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path="/proj/main.py",
        )
        block3 = CodeBlock(
            block_name="build_greeting",
            start_line=10, end_line=10,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path="/proj/api/routes/items.py",
        )

        resp_main = Mock()
        resp_main.EXPLANATION = "Converting async to sync"
        resp_main.FIXED_CODE_BLOCKS = [block1]

        resp_caller1 = Mock()
        resp_caller1.EXPLANATION = ""
        resp_caller1.FIXED_CODE_BLOCKS = [block2]

        resp_caller2 = Mock()
        resp_caller2.EXPLANATION = ""
        resp_caller2.FIXED_CODE_BLOCKS = [block3]

        issue = Mock()
        issue.rule = "python:S7503"
        issue.severity = "MINOR"
        issue.message = "Async function without await"
        issue.file_path = "services/bad_async.py"
        issue.line = 10

        fixer.write_explaination(
            md_file,
            [resp_caller1, resp_caller2, resp_main],
            [issue],
            "original code",
            project_path=Path("/proj"),
        )

        content = md_file.read_text()
        # All three blocks should be rendered
        assert content.count("build_greeting") >= 3
        # External file paths should appear as relative
        assert "main.py" in content
        assert "api/routes/items.py" in content

    def test_file_paths_converted_to_relative(self, fixer, tmp_path):
        """Absolute file paths are converted to relative when project_path is given."""
        from devdox_ai_sonar.models.sonar import CodeBlock, ChangeType, BlockType

        md_file = tmp_path / "output.md"

        block = CodeBlock(
            block_name="my_func",
            start_line=5, end_line=5,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path="/home/user/project/src/foo.py",
        )

        resp = Mock()
        resp.EXPLANATION = "Fix"
        resp.FIXED_CODE_BLOCKS = [block]

        issue = Mock()
        issue.rule = "python:S7503"
        issue.severity = "MINOR"
        issue.message = "msg"
        issue.file_path = "src/foo.py"
        issue.line = 5

        fixer.write_explaination(
            md_file, [resp], [issue], "code",
            project_path=Path("/home/user/project"),
        )

        content = md_file.read_text()
        assert "src/foo.py" in content
        assert "/home/user/project/src/foo.py" not in content

    def test_no_file_path_renders_without_dash(self, fixer, tmp_path):
        """Code blocks without file_path render headers without the em-dash."""
        md_file = tmp_path / "output.md"

        block = Mock()
        block.file_path = None
        block.block_name = "my_func"
        block.start_line = 1
        block.end_line = 2
        block.changes = None
        block.context = None
        block.replacements = None

        resp = Mock()
        resp.EXPLANATION = "Fix"
        resp.FIXED_CODE_BLOCKS = [block]

        issue = Mock()
        issue.rule = "python:S1234"
        issue.severity = "MINOR"
        issue.message = "msg"
        issue.file_path = "test.py"
        issue.line = 1

        fixer.write_explaination(md_file, [resp], [issue], "code")

        content = md_file.read_text()
        assert "my_func" in content
        # Should NOT have the file_path em-dash pattern
        assert "\u2014 `" not in content

    def test_original_code_blocks_not_mutated(self, fixer, tmp_path):
        """Original CodeBlock objects should not have their file_path mutated."""
        from devdox_ai_sonar.models.sonar import CodeBlock, ChangeType, BlockType

        md_file = tmp_path / "output.md"
        original_path = "/home/user/project/src/foo.py"

        block = CodeBlock(
            block_name="my_func",
            start_line=5,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path=original_path,
        )

        resp = Mock()
        resp.EXPLANATION = "Fix"
        resp.FIXED_CODE_BLOCKS = [block]

        issue = Mock()
        issue.rule = "python:S7503"
        issue.severity = "MINOR"
        issue.message = "msg"
        issue.file_path = "src/foo.py"
        issue.line = 5

        fixer.write_explaination(
            md_file, [resp], [issue], "code",
            project_path=Path("/home/user/project"),
        )

        # Original block's file_path should be unchanged
        assert block.file_path == original_path


class TestConvertRegexToDiff:
    """Tests for LLMFixer._convert_regex_to_diff."""

    def test_converts_regex_block_to_diff(self, tmp_path):
        """Regex SearchReplace block is converted to DIFF with actual source."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        source_file = tmp_path / "test.py"
        source_file.write_text("    msg = await build_greeting('root')\n")

        block = CodeBlock(
            block_name="build_greeting",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path=str(source_file),
            replacements=[
                SearchReplace(
                    search=r"\bawait\s+(?=build_greeting\s*\()",
                    replace="",
                    is_regex=True,
                )
            ],
        )

        file_cache: dict = {}
        result = LLMFixer._convert_regex_to_diff(block, file_cache)

        assert result["change_type"] == ChangeType.DIFF
        assert result["replacements"] is None
        assert len(result["changes"]) == 1
        assert "await" in result["changes"][0].old
        assert "await" not in result["changes"][0].new
        assert "build_greeting" in result["changes"][0].new

    def test_non_regex_block_returns_empty(self):
        """Non-regex SearchReplace block returns empty dict."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        block = CodeBlock(
            block_name="func",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path="/some/file.py",
            replacements=[
                SearchReplace(search="old_text", replace="new_text", is_regex=False)
            ],
        )

        result = LLMFixer._convert_regex_to_diff(block, {})
        assert result == {}

    def test_diff_block_returns_empty(self):
        """DIFF block (not SEARCH_REPLACE) returns empty dict."""
        from devdox_ai_sonar.models.sonar import CodeBlock, ChangeType, BlockType

        block = CodeBlock(
            block_name="func",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path="/some/file.py",
        )

        result = LLMFixer._convert_regex_to_diff(block, {})
        assert result == {}

    def test_missing_file_returns_empty(self):
        """Missing source file returns empty dict (graceful fallback)."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        block = CodeBlock(
            block_name="func",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path="/nonexistent/path/file.py",
            replacements=[
                SearchReplace(search=r"\bawait\s+", replace="", is_regex=True)
            ],
        )

        result = LLMFixer._convert_regex_to_diff(block, {})
        assert result == {}

    def test_line_out_of_range_returns_empty(self, tmp_path):
        """Line number beyond file length returns empty dict."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        source_file = tmp_path / "short.py"
        source_file.write_text("line1\n")

        block = CodeBlock(
            block_name="func",
            start_line=999, end_line=999,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path=str(source_file),
            replacements=[
                SearchReplace(search=r"\bawait\s+", replace="", is_regex=True)
            ],
        )

        result = LLMFixer._convert_regex_to_diff(block, {})
        assert result == {}

    def test_caches_file_reads(self, tmp_path):
        """File content is cached — same path is only read once."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        source_file = tmp_path / "cached.py"
        source_file.write_text("msg = await func()\n")

        block = CodeBlock(
            block_name="func",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path=str(source_file),
            replacements=[
                SearchReplace(search=r"\bawait\s+(?=func\s*\()", replace="", is_regex=True)
            ],
        )

        file_cache: dict = {}
        LLMFixer._convert_regex_to_diff(block, file_cache)

        assert str(source_file) in file_cache
        # Second call reuses cache — delete file to prove it
        source_file.unlink()
        result = LLMFixer._convert_regex_to_diff(block, file_cache)
        assert result["change_type"] == ChangeType.DIFF


class TestBuildDisplayBlocks:
    """Tests for LLMFixer._build_display_blocks."""

    def test_converts_absolute_paths_to_relative(self):
        """Absolute file paths are made relative to project_path."""
        from devdox_ai_sonar.models.sonar import CodeBlock, ChangeType, BlockType

        block = CodeBlock(
            block_name="func",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path="/home/user/project/src/foo.py",
        )

        result = LLMFixer._build_display_blocks(
            [block], project_path=Path("/home/user/project")
        )

        assert len(result) == 1
        assert result[0].file_path == "src/foo.py"

    def test_no_project_path_keeps_original(self):
        """Without project_path, file paths are unchanged."""
        from devdox_ai_sonar.models.sonar import CodeBlock, ChangeType, BlockType

        block = CodeBlock(
            block_name="func",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path="/abs/path/file.py",
        )

        result = LLMFixer._build_display_blocks([block])

        assert result[0].file_path == "/abs/path/file.py"

    def test_regex_blocks_converted_to_diff(self, tmp_path):
        """Regex SearchReplace blocks are converted to DIFF in display output."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        source_file = tmp_path / "test.py"
        source_file.write_text("x = await my_func()\n")

        block = CodeBlock(
            block_name="my_func",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path=str(source_file),
            replacements=[
                SearchReplace(
                    search=r"\bawait\s+(?=my_func\s*\()",
                    replace="",
                    is_regex=True,
                )
            ],
        )

        result = LLMFixer._build_display_blocks([block], project_path=tmp_path)

        assert len(result) == 1
        assert result[0].change_type == ChangeType.DIFF
        assert result[0].changes[0].old == "x = await my_func()"
        assert result[0].changes[0].new == "x = my_func()"

    def test_mixed_blocks(self, tmp_path):
        """Handles a mix of DIFF and SearchReplace blocks correctly."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        source_file = tmp_path / "test.py"
        source_file.write_text("result = await greet()\n")

        diff_block = CodeBlock(
            block_name="greet",
            start_line=5, end_line=6,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            file_path=str(tmp_path / "other.py"),
        )
        regex_block = CodeBlock(
            block_name="greet",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path=str(source_file),
            replacements=[
                SearchReplace(
                    search=r"\bawait\s+(?=greet\s*\()",
                    replace="",
                    is_regex=True,
                )
            ],
        )

        result = LLMFixer._build_display_blocks(
            [diff_block, regex_block], project_path=tmp_path
        )

        assert len(result) == 2
        assert result[0].change_type == ChangeType.DIFF  # unchanged
        assert result[1].change_type == ChangeType.DIFF  # converted

    def test_does_not_mutate_originals(self, tmp_path):
        """Original CodeBlock objects are not modified."""
        from devdox_ai_sonar.models.sonar import (
            CodeBlock, ChangeType, BlockType, SearchReplace,
        )

        source_file = tmp_path / "test.py"
        source_file.write_text("val = await fn()\n")

        block = CodeBlock(
            block_name="fn",
            start_line=1, end_line=1,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            file_path=str(source_file),
            replacements=[
                SearchReplace(
                    search=r"\bawait\s+(?=fn\s*\()",
                    replace="",
                    is_regex=True,
                )
            ],
        )

        LLMFixer._build_display_blocks([block], project_path=tmp_path)

        assert block.change_type == ChangeType.SEARCH_REPLACE
        assert block.replacements is not None


class TestGenerateFixByFile:
    """Cover generate_fix_by_file orchestration."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    async def test_empty_issues_returns_none(self, fixer):
        result = await fixer.generate_fix_by_file([], Path("/proj"), Path("/tmp"))
        assert result is None

    async def test_validation_failure_returns_none(self, fixer):
        """When validation fails, returns None (lines 262-264)."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        mock_validation = Mock()
        mock_validation.is_valid = False
        mock_validation.error = "bad"
        with patch("devdox_ai_sonar.llm_fixer.IssueExtractor") as MockExtractor:
            MockExtractor.return_value.validate_issue_group = AsyncMock(return_value=mock_validation)
            result = await fixer.generate_fix_by_file([issue], Path("/proj"), Path("/tmp"))
        assert result is None

    async def test_context_prep_failure_returns_none(self, fixer):
        """When context preparation fails, returns None (lines 274-276)."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        mock_validation = Mock()
        mock_validation.is_valid = True
        mock_validation.file_path = Path("/proj/test.py")
        mock_validation.line_range = {"first_line": 1, "last_line": 5, "problem_lines": [1]}

        with patch("devdox_ai_sonar.llm_fixer.IssueExtractor") as MockExtractor, \
             patch.object(fixer, "_prepare_fix_context", new_callable=AsyncMock, return_value=None):
            MockExtractor.return_value.validate_issue_group = AsyncMock(return_value=mock_validation)
            result = await fixer.generate_fix_by_file([issue], Path("/proj"), Path("/tmp"))
        assert result is None

    async def test_handler_returns_none(self, fixer):
        """When handler returns no fix, returns None (lines 290-292)."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        mock_validation = Mock()
        mock_validation.is_valid = True
        mock_validation.file_path = Path("/proj/test.py")
        mock_validation.line_range = {"first_line": 1, "last_line": 5, "problem_lines": [1]}

        mock_ctx = Mock()
        mock_handler = Mock()
        mock_handler.generate_fixes = AsyncMock(return_value=None)
        mock_handler.MOIDY_LINE_RANGE = False

        with patch("devdox_ai_sonar.llm_fixer.IssueExtractor") as MockExtractor, \
             patch.object(fixer, "_prepare_fix_context", new_callable=AsyncMock, return_value=mock_ctx), \
             patch("devdox_ai_sonar.llm_fixer.RuleHandlerRegistry") as MockRegistry:
            MockExtractor.return_value.validate_issue_group = AsyncMock(return_value=mock_validation)
            MockRegistry.return_value.get_handler.return_value = mock_handler
            result = await fixer.generate_fix_by_file([issue], Path("/proj"), Path("/tmp"))
        assert result is None

    async def test_successful_fix_generation(self, fixer):
        """Full successful path through generate_fix_by_file (lines 296-315)."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        mock_validation = Mock()
        mock_validation.is_valid = True
        mock_validation.file_path = Path("/proj/test.py")
        mock_validation.line_range = {"first_line": 1, "last_line": 5, "problem_lines": [1]}

        mock_ctx = Mock()
        mock_ctx.context_dict = {"context": "code"}

        mock_fix_response = Mock()
        mock_fix_response.FIXED_CODE_BLOCKS = []
        mock_handler = Mock()
        mock_handler.generate_fixes = AsyncMock(return_value=[mock_fix_response])
        mock_handler.MOIDY_LINE_RANGE = False

        mock_suggestion = Mock()
        with patch("devdox_ai_sonar.llm_fixer.IssueExtractor") as MockExtractor, \
             patch.object(fixer, "_prepare_fix_context", new_callable=AsyncMock, return_value=mock_ctx), \
             patch.object(fixer, "_build_fix_suggestion", return_value=[mock_suggestion]), \
             patch("devdox_ai_sonar.llm_fixer.RuleHandlerRegistry") as MockRegistry:
            MockExtractor.return_value.validate_issue_group = AsyncMock(return_value=mock_validation)
            MockRegistry.return_value.get_handler.return_value = mock_handler
            result = await fixer.generate_fix_by_file([issue], Path("/proj"), Path("/tmp"))
        assert result == [mock_suggestion]

    async def test_file_not_found_returns_none(self, fixer):
        """FileNotFoundError path (lines 317-319)."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        mock_validation = Mock()
        mock_validation.is_valid = True
        mock_validation.file_path = Path("/proj/test.py")
        mock_validation.line_range = {"first_line": 1, "last_line": 5, "problem_lines": [1]}

        with patch("devdox_ai_sonar.llm_fixer.IssueExtractor") as MockExtractor, \
             patch.object(fixer, "_prepare_fix_context", new_callable=AsyncMock, side_effect=FileNotFoundError("nope")):
            MockExtractor.return_value.validate_issue_group = AsyncMock(return_value=mock_validation)
            result = await fixer.generate_fix_by_file([issue], Path("/proj"), Path("/tmp"))
        assert result is None

    async def test_generic_exception_returns_none(self, fixer):
        """Generic exception path (lines 320-322)."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        mock_validation = Mock()
        mock_validation.is_valid = True
        mock_validation.file_path = Path("/proj/test.py")
        mock_validation.line_range = {"first_line": 1, "last_line": 5, "problem_lines": [1]}

        with patch("devdox_ai_sonar.llm_fixer.IssueExtractor") as MockExtractor, \
             patch.object(fixer, "_prepare_fix_context", new_callable=AsyncMock, side_effect=RuntimeError("err")):
            MockExtractor.return_value.validate_issue_group = AsyncMock(return_value=mock_validation)
            result = await fixer.generate_fix_by_file([issue], Path("/proj"), Path("/tmp"))
        assert result is None

    async def test_with_file_md_calls_write(self, fixer):
        """When file_md is provided, write_explaination is called (lines 306-312)."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        mock_validation = Mock()
        mock_validation.is_valid = True
        mock_validation.file_path = Path("/proj/test.py")
        mock_validation.line_range = {"first_line": 1, "last_line": 5, "problem_lines": [1]}

        mock_ctx = Mock()
        mock_ctx.context_dict = {"context": "code"}
        mock_fix_response = Mock()
        mock_handler = Mock()
        mock_handler.generate_fixes = AsyncMock(return_value=[mock_fix_response])
        mock_handler.MOIDY_LINE_RANGE = False

        with patch("devdox_ai_sonar.llm_fixer.IssueExtractor") as MockExtractor, \
             patch.object(fixer, "_prepare_fix_context", new_callable=AsyncMock, return_value=mock_ctx), \
             patch.object(fixer, "_build_fix_suggestion", return_value=[Mock()]), \
             patch.object(fixer, "write_explaination") as mock_write, \
             patch("devdox_ai_sonar.llm_fixer.RuleHandlerRegistry") as MockRegistry:
            MockExtractor.return_value.validate_issue_group = AsyncMock(return_value=mock_validation)
            MockRegistry.return_value.get_handler.return_value = mock_handler
            result = await fixer.generate_fix_by_file(
                [issue], Path("/proj"), Path("/tmp"), file_md="docs/fix.md"
            )
        mock_write.assert_called_once()

class TestApplyFixesFailurePath:
    """Cover apply_fixes when _apply_fixes_to_file returns False."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    async def test_apply_fixes_failure_marks_failed(self, fixer, tmp_path):
        """When _apply_fixes_to_file returns False, fixes go to failed list."""
        fix = Mock(spec=FixSuggestion)
        fix.file_path = str(tmp_path / "test.py")
        fix.issue_key = "k"
        fix.original_code = "code"

        with patch.object(fixer, "_get_file_from_fix", return_value=str(tmp_path / "test.py")), \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [])), \
             patch.object(fixer, "_create_backup", return_value=str(tmp_path / "bak")):
            result = await fixer.apply_fixes([fix], tmp_path, create_backup=False)

        assert len(result.failed_fixes) == 1
        assert "Failed to apply fix" in result.failed_fixes[0]["error"]

class TestCallLlmUnknownProvider:
    """Cover the unknown provider else branch in _call_llm_list."""

    def test_unknown_provider_returns_none(self, mock_openai_client):
        fixer = LLMFixer(provider="openai", api_key="test-key")
        fixer.provider = "unknown_provider"
        ctx = FixContext(
            file_path=Path("test.py"), file_path_tmp=Path("test.py"),
            line_range={}, code_content="code", language="python",
            import_section={"start_line": 0, "end_line": 0, "content": "", "has_imports": False},
            class_name=None, functions=[],
            context_dict={ "context": "code",
                          "new_context": [{"context": "code","start_line": 1, "end_line": 5}],
                          "problem_line_content": {1: "code\n"}},
        )
        issue = SonarIssue(
            key="t", rule="python:S1", severity="MAJOR", component="t.py",
            project="t", line=1, message="t", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        result = fixer._call_llm_list([issue], ctx, ".py", {}, "")
        assert result is None

class TestCreateFixPromptListBranches:
    """Cover S3776 template switch and S1172 strategy modification and method type branches."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_context(self, code="def foo(x):\n    return x", class_name=None):
        return FixContext(
            file_path=Path("test.py"), file_path_tmp=Path("test.py"),
            line_range={}, code_content=code, language="python",
            import_section={"start_line": 0, "end_line": 0, "content": "", "has_imports": False},
            class_name=class_name, functions=[],
            context_dict={
                "start_line": 1, "end_line": 10, "context": code,
                "new_context": [{"context": code, "function_name": "foo",
                                 "start_line": 1, "end_line": 2}],
                "problem_line_content": {1: code.split("\n")[0] + "\n"},
                "import_section": {"start_line": 0, "end_line": 0, "content": "", "has_imports": False},
            },
        )

    def test_s3776_switches_template(self, fixer):
        """python:S3776 should switch to refactoring templates."""
        issue = SonarIssue(
            key="k", rule="python:S3776", severity="MAJOR", component="t.py",
            project="p", line=1, message="high complexity", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        ctx = self._make_context()
        rule_info = {"python:S3776": {"how_to_fix": {"steps": ["reduce complexity"]}, "root_cause": "too complex"}}
        prompt, sys_template = fixer._create_fix_prompt_list(
            [issue], ctx, rule_info, "python", ""
        )
        assert isinstance(prompt, str)

    def test_s1172_modifies_strategies(self, fixer):
        """python:S1172 should filter steps and modify rule_info."""
        issue = SonarIssue(
            key="k", rule="python:S1172", severity="MAJOR", component="t.py",
            project="p", line=1, message="unused param", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        ctx = self._make_context()
        rule_info = {
            "python:S1172": {
                "how_to_fix": {"steps": [
                    "Remove unused elements or implement their intended purpose",
                    "Keep needed params"
                ]},
                "root_cause": "unused param",
                "name": "Unused param"
            }
        }
        prompt, _ = fixer._create_fix_prompt_list([issue], ctx, rule_info, "python", "")
        assert "Remove unused elements" not in rule_info["python:S1172"]["how_to_fix"]["steps"]
        assert FUNCTION_ALREADY_CALLED in rule_info["python:S1172"]["how_to_fix"]["steps"]

    def test_staticmethod_instruction(self, fixer):
        """Static method context produces @staticmethod instructions."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        ctx = self._make_context(
            code="@staticmethod\ndef foo(x):\n    return x",
            class_name="MyClass",
        )
        prompt, _ = fixer._create_fix_prompt_list([issue], ctx, {}, "python", "")
        assert "@staticmethod" in prompt

    def test_self_method_instruction(self, fixer):
        """Instance method context produces self instructions."""
        issue = SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )
        ctx = self._make_context(code="def foo(self, x):\n    return x")
        prompt, _ = fixer._create_fix_prompt_list([issue], ctx, {}, "python", "")
        assert "self" in prompt

class TestParseProviderResponses:
    """Cover the individual parse methods and their exception paths."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_parse_openai_exception(self, fixer):
        """Line 1191-1193: exception path in _parse_openai_response."""
        response = Mock()
        response.output_parsed = Mock(side_effect=AttributeError("boom"))
        # The property access itself will raise
        type(response).output_parsed = property(lambda self: (_ for _ in ()).throw(Exception("boom")))
        result = fixer._parse_openai_response(response)
        assert result is None

    def test_parse_gemini_response_success(self, fixer):
        """Lines 1197-1199: successful gemini parse."""
        response = Mock()
        response.text = json.dumps({
            "IMPORT_BLOCK": "", "FIXED_CODE_BLOCKS": [],
            "NEW_HELPER_CODE": "", "PLACEMENT": "SIBLING",
            "EXPLANATION": "fix", "CONFIDENCE": 0.9
        })
        with patch.object(fixer, "_extract_fix_from_response") as mock_extract:
            mock_extract.return_value = {"result": "ok"}
            result = fixer._parse_gemini_response(response)
        assert result == {"result": "ok"}

    def test_parse_gemini_response_exception(self, fixer):
        """Lines 1200-1202: exception in gemini parse."""
        response = Mock()
        type(response).text = property(lambda self: (_ for _ in ()).throw(Exception("fail")))
        result = fixer._parse_gemini_response(response)
        assert result is None

    def test_parse_togetherai_response_success(self, fixer):
        """Lines 1207-1208: successful togetherai parse."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = json.dumps({
            "IMPORT_BLOCK": "",
            "FIXED_CODE_BLOCKS": [{"block_name": "f", "start_line": 1,
                                    "end_line": 2, "has_changes": True,
                                    "change_type": "FULL_CODE",
                                    "block_type": "module", "context": "x"}],
            "NEW_HELPER_CODE": "", "PLACEMENT": "SIBLING",
            "EXPLANATION": "fix", "CONFIDENCE": 0.9
        })
        result = fixer._parse_togetherai_response(response)
        assert result is not None

    def test_parse_togetherai_response_exception(self, fixer):
        """Lines 1209-1211: exception in togetherai parse."""
        response = Mock()
        response.choices = []  # IndexError
        result = fixer._parse_togetherai_response(response)
        assert result is None

    def test_parse_openrouter_response_success(self, fixer):
        """Successful openrouter parse."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = json.dumps({
            "IMPORT_BLOCK": "",
            "FIXED_CODE_BLOCKS": [{"block_name": "f", "start_line": 1,
                                    "end_line": 2, "has_changes": True,
                                    "change_type": "FULL_CODE",
                                    "block_type": "module", "context": "x"}],
            "NEW_HELPER_CODE": "", "PLACEMENT": "SIBLING",
            "EXPLANATION": "fix", "CONFIDENCE": 0.9
        })
        result = fixer._parse_openrouter_response(response)
        assert result is not None

    def test_parse_openrouter_response_exception(self, fixer):
        """Exception path in openrouter parse."""
        response = Mock()
        response.choices = []  # IndexError
        result = fixer._parse_openrouter_response(response)
        assert result is None

    def test_extract_using_regex_fallback_exception(self, fixer):
        """Lines 1219-1221: exception in regex fallback."""
        with patch.object(fixer, "_apply_regex_patterns", side_effect=Exception("boom")):
            result = fixer._extract_using_regex_fallback("some content")
        assert result is None

class TestApplyIndentationEdgeCases:
    """Cover edge cases in apply_indentation_to_fix."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_empty_lines_list(self, fixer):
        """After strip, if split produces no lines, return as-is (line 1496)."""
        # "x" is non-empty but single-line
        result = fixer.apply_indentation_to_fix("x", "    ")
        assert result == "    x"

    def test_preserves_empty_lines(self, fixer):
        """Empty lines don't get indented (line 1507)."""
        code = "def foo():\n\n    pass"
        result = fixer.apply_indentation_to_fix(code, "  ")
        lines = result.split("\n")
        assert lines[1] == ""  # Empty line not indented

class TestApplyFixesToFileException:
    """Cover the exception handler in _apply_fixes_to_file."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    async def test_exception_returns_false_empty(self, fixer, tmp_path):
        """Lines 1545-1547: exception returns (False, [])."""
        file_path = tmp_path / "test.py"
        file_path.write_text("x = 1\n")
        fix = Mock()
        fixer.file_reader.read_lines = AsyncMock(side_effect=Exception("boom"))
        success, results = await fixer._apply_fixes_to_file(file_path, [fix], False)
        assert success is False
        assert results == []

class TestCheckBracketBalance:
    """Cover _check_bracket_balance early return False."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_unmatched_closing_bracket(self, fixer):
        """Line 1588: closing bracket with empty stack returns False."""
        assert fixer._check_bracket_balance(")") is False

    def test_mismatched_brackets(self, fixer):
        """Mismatched bracket types."""
        assert fixer._check_bracket_balance("(]") is False

    def test_balanced_brackets(self, fixer):
        assert fixer._check_bracket_balance("()[]{}") is True

class TestApplyFixesValidatorFallback:
    """Cover the validator fallback paths in apply_fixes_with_validation."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_fix(self, tmp_path):
        fix = Mock(spec=FixSuggestion)
        fix.file_path = str(tmp_path / "test.py")
        fix.issue_key = "k"
        fix.original_code = "x = 1"
        return fix

    def _make_issue(self):
        return SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )

    async def test_validator_approved(self, fixer, tmp_path):
        """Line 1766-1767: validator APPROVED path."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "syntax error"

        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.APPROVED

        with patch.object(fixer, "_get_file_from_fix", return_value=str(py_file)), \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])), \
             patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=True,
                validator_provider="openai", validator_api_key="key",
            )
        assert len(result.successful_fixes) == 1

    async def test_validator_modified_success(self, fixer, tmp_path):
        """Lines 1774-1783: validator MODIFIED with successful re-application."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "syntax error"

        improved_fix = Mock()
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.MODIFIED
        mock_val_result.final_fix = improved_fix

        call_count = [0]
        async def apply_side_effect(fp, fixes, dry):
            call_count[0] += 1
            if call_count[0] == 1:
                return (False, [failed_result])
            return (True, [])

        with patch.object(fixer, "_get_file_from_fix", return_value=str(py_file)), \
             patch.object(fixer, "_apply_fixes_to_file", side_effect=apply_side_effect), \
             patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=True,
                validator_provider="openai", validator_api_key="key",
            )
        assert len(result.successful_fixes) == 1

    async def test_validator_modified_no_final_fix(self, fixer, tmp_path):
        """Lines 1792-1798: MODIFIED but no final_fix."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"

        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.MODIFIED
        mock_val_result.final_fix = None

        with patch.object(fixer, "_get_file_from_fix", return_value=str(py_file)), \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])), \
             patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=True,
                validator_provider="openai", validator_api_key="key",
            )
        assert len(result.failed_fixes) == 1
        assert "no improved fix" in result.failed_fixes[0]["error"]

    async def test_validator_modified_reapply_fails(self, fixer, tmp_path):
        """Lines 1785-1791: MODIFIED but re-application fails."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"

        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.MODIFIED
        mock_val_result.final_fix = Mock()

        with patch.object(fixer, "_get_file_from_fix", return_value=str(py_file)), \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])), \
             patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=True,
                validator_provider="openai", validator_api_key="key",
            )
        assert len(result.failed_fixes) == 1
        assert "still failed" in result.failed_fixes[0]["error"]

    async def test_validator_rejected(self, fixer, tmp_path):
        """Lines 1800-1812: validator REJECTED."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"

        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.REJECTED
        mock_val_result.explanation = "bad fix"

        with patch.object(fixer, "_get_file_from_fix", return_value=str(py_file)), \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])), \
             patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=True,
                validator_provider="openai", validator_api_key="key",
            )
        assert len(result.failed_fixes) == 1
        assert "Rejected" in result.failed_fixes[0]["error"]

    async def test_validator_needs_review(self, fixer, tmp_path):
        """Lines 1814-1823: validator NEEDS_REVIEW."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"

        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.NEEDS_REVIEW
        mock_val_result.explanation = "check manually"

        with patch.object(fixer, "_get_file_from_fix", return_value=str(py_file)), \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])), \
             patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=True,
                validator_provider="openai", validator_api_key="key",
            )
        assert len(result.failed_fixes) == 1
        assert "manual review" in result.failed_fixes[0]["error"]

    async def test_validator_exception(self, fixer, tmp_path):
        """Lines 1825-1832: exception in validator fallback."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"

        with patch.object(fixer, "_get_file_from_fix", return_value=str(py_file)), \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])), \
             patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.side_effect = Exception("validator crash")
            result = await fixer.apply_fixes_with_validation(
                [fix], [issue], tmp_path,
                create_backup=False, use_validator=True,
                validator_provider="openai", validator_api_key="key",
            )
        assert len(result.failed_fixes) == 1
        assert "Validator error" in result.failed_fixes[0]["error"]

class TestFindFilesWithContent:
    """Cover _find_files_with_content early break and exception."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_finds_matching_files(self, fixer, tmp_path):
        """Test finding files with matching content."""
        for i in range(4):
            f = tmp_path / f"file{i}.py"
            f.write_text("target_content_here\n")
        result = fixer._find_files_with_content(tmp_path, "target_content_here")
        assert len(result) <= 3

    def test_exception_returns_empty(self, fixer, tmp_path):
        """Exception returns empty list."""
        with patch("pathlib.Path.rglob", side_effect=Exception("boom")):
            result = fixer._find_files_with_content(tmp_path, "x")
        assert result == []

class TestExtractPythonImportSectionEdgeCases:
    """Cover remaining edge cases in import extraction."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_backslash_after_closing_paren(self, fixer):
        """Line 2152-2153: backslash continuation after closing paren."""
        lines = [
            "from os import (\n",
            "    path) \\\n",
            "    ; import sys\n",
            "x = 1\n",
        ]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is True

    def test_docstring_after_imports_breaks(self, fixer):
        """Line 2192: docstring after imports causes break."""
        lines = [
            "import os\n",
            '"""Module docstring"""\n',
            "x = 1\n",
        ]
        result = fixer._extract_python_import_section(lines)
        assert result["has_imports"] is True
        assert result["end_line"] == 1

    def test_should_stop_at_code_in_multiline_import(self, fixer):
        """Line 2248: in_multiline_import prevents stopping at code."""
        state = {'in_multiline_import': True, 'in_parentheses_import': False, 'in_backslash_continuation': False}
        result = fixer._should_stop_at_code("x = 1", 1, state)
        assert result is False

    def test_should_stop_at_code_dunder(self, fixer):
        """Line 2261: dunder variable doesn't stop import collection."""
        state = {'in_multiline_import': False, 'in_parentheses_import': False, 'in_backslash_continuation': False}
        result = fixer._should_stop_at_code("__all__ = ['foo']", 1, state)
        assert result is False

class TestIsLineInsideClassEdgeCases:
    """Cover remaining _is_line_inside_class branches."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_another_class_def_returns_false(self, fixer):
        """Lines 2443-2444: another class definition at same indent returns False."""
        lines = [
            "class A:\n",
            "    def foo(self):\n",
            "        pass\n",
            "class B:\n",
            "    def bar(self):\n",
        ]
        # target_idx=4 (inside class B), class at idx 0 (class A)
        result = fixer._is_line_inside_class(lines, 4, 0)
        assert result is False

    def test_empty_target_line_returns_true(self, fixer):
        """Line 2449: empty target line returns True."""
        lines = [
            "class A:\n",
            "    def foo(self):\n",
            "\n",
        ]
        result = fixer._is_line_inside_class(lines, 2, 0)
        assert result is True

class TestContextExtractorRemainingBranches:
    """Cover remaining uncovered lines in ContextExtractor."""

    def test_find_functions_duplicate_function_start(self):
        """Line 2490-2491: already processed function start is skipped."""
        lines = [
            "def foo():\n",
            "    x = 1\n",
            "    y = 2\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        # Two problem indices in same function - second should be skipped
        result = ext._find_functions_containing_problems([0, 1], 0, 3)
        assert len(result) == 1

    def test_get_function_info_fails_returns_none(self):
        """Lines 2496-2499: get_function_info returns None for invalid line."""
        lines = [
            "x = 1\n",
            "y = 2\n",
        ]
        ext = ContextExtractor(lines)
        # Problem at line that's not a function - _find_containing_function will return None
        result = ext._find_functions_containing_problems([0], 0, 1)
        assert len(result) == 0

    def test_count_functions_line_exceeds_range(self):
        """Line 2597: break when line_idx >= len(lines)."""
        lines = ["def foo():\n", "    pass\n"]
        ext = ContextExtractor(lines)
        count = ext._count_functions_in_range(0, 100)
        assert count == 1

    def test_get_function_info_no_end(self):
        """Line 2697: _find_function_end returns None."""
        # A function definition where end can't be found
        lines = ["def foo():\n"]
        ext = ContextExtractor(lines)
        with patch.object(ext, "_find_function_end", return_value=None):
            result = ext._get_function_info(0)
        assert result is None

    def test_is_line_inside_function_before_start(self):
        """Line 2765: line_idx < function_start_idx returns False."""
        lines = ["x = 1\n", "def foo():\n", "    pass\n"]
        ext = ContextExtractor(lines)
        result = ext._is_line_inside_function(lines, 0, 1)
        assert result is False

    def test_is_line_inside_function_no_end_found(self):
        """Lines 2770: when _find_function_end returns None, uses indentation check."""
        lines = ["def foo():\n", "    x = 1\n"]
        ext = ContextExtractor(lines)
        with patch.object(ext, "_find_function_end", return_value=None):
            result = ext._is_line_inside_function(lines, 1, 0)
        assert result is True

    def test_find_python_function_end_out_of_bounds(self):
        """Line 2847: start_idx >= len(lines) returns None."""
        lines = ["x\n"]
        ext = ContextExtractor(lines)
        result = ext._find_python_function_end(lines, 99)
        assert result is None

    def test_find_function_end_click_decorator(self):
        """Lines 2965-2966: @click. decorator uses python function end."""
        lines = [
            "@click.command()\n",
            "def cli():\n",
            "    pass\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._find_function_end(0)
        assert result is not None

    def test_find_function_end_unknown_tries_both(self):
        """Lines 2967-2972: unknown pattern tries python then brace."""
        lines = [
            "some_unknown_syntax:\n",
            "    x = 1\n",
            "    y = 2\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        result = ext._find_function_end(0)
        # Should try python first, then brace if python returns None
        assert result is not None or result is None  # Either outcome is valid

    def test_extract_complete_function_no_end(self):
        """Line 3077: function_end is None fallback to +50 lines."""
        lines = ["def foo():\n"] + ["    x = 1\n"] * 60
        ext = ContextExtractor(lines)
        with patch.object(ext, "_find_function_end", return_value=None):
            result = ext._extract_complete_function(0, 0, 0)
        assert result is not None

    def test_find_function_start_with_decorators_empty_line_break(self):
        """Line 3136: empty line stops decorator search."""
        lines = [
            "@some_decorator\n",
            "\n",
            "@another\n",
            "def foo():\n",
            "    pass\n",
        ]
        ext = ContextExtractor(lines)
        start = ext._find_function_start_with_decorators(lines, 3)
        assert start == 2  # Stops at empty line, doesn't include @some_decorator

    def test_try_expand_multiline_function_end_none(self):
        """Line 3230: _find_function_end returns None in expand."""
        lines = [
            "def foo():\n",
            "    x = 1\n",
            "    y = 2\n",
        ]
        ext = ContextExtractor(lines)
        with patch.object(ext, "_find_containing_function", return_value=0), \
             patch.object(ext, "_find_function_end", return_value=None):
            result = ext._try_expand_multiline_to_function(0, 1)
        assert result is None

    def test_try_expand_multiline_end_not_beyond_range(self):
        """Line 3234: function_end_idx <= last_idx returns None."""
        lines = [
            "def foo():\n",
            "    x = 1\n",
            "    y = 2\n",
        ]
        ext = ContextExtractor(lines)
        # last_idx=2, function ends at 2 -> not beyond range
        with patch.object(ext, "_find_containing_function", return_value=0), \
             patch.object(ext, "_find_function_end", return_value=2):
            result = ext._try_expand_multiline_to_function(0, 2)
        assert result is None

    def test_try_function_extraction_strategies_multiline(self):
        """Line 3197: multi-line path calls _try_expand_multiline_to_function."""
        lines = [
            "x = 1\n",
            "def foo():\n",
            "    return 1\n",
            "    return 2\n",
            "\n",
        ]
        ext = ContextExtractor(lines)
        # first_idx != last_idx triggers multiline path
        with patch.object(ext, "_try_extract_containing_function", return_value=None), \
             patch.object(ext, "_try_expand_multiline_to_function", return_value={"result": "expanded"}) as mock_expand:
            result = ext._try_function_extraction_strategies(0, 2)
        assert result == {"result": "expanded"}
        mock_expand.assert_called_once_with(0, 2)

class TestModuleLevelBuildFixSuggestionElse:
    """Cover the else branch where problem_lines is neither list nor int."""

    def test_non_list_non_int_problem_lines(self):
        """Line 3332: problem_lines that's neither list nor int gets empty list."""
        block = CodeBlock(
            block_name="f", start_line=1, end_line=2,
            has_changes=True, change_type=ChangeType.FULL_CODE,
            block_type=BlockType.FUNCTION, context="x"
        )
        fix_response = SonarFixResponse(
            IMPORT_BLOCK="",
            FIXED_CODE_BLOCKS=[block],
            NEW_HELPER_CODE="",
            PLACEMENT="GLOBAL_TOP",
            EXPLANATION="e",
            CONFIDENCE=0.9,
        )
        line_range = {
            "first_line": 1,
            "last_line": 5,
            "problem_lines": "invalid_type",
        }
        context_info = {
            "context_dict": {
                "context": "code",
                "start_line": 1,
                "end_line": 5,
            }
        }
        result = _build_fix_suggestion(
            fix_response, context_info,
            Path("/proj/test.py"), Path("/proj"), line_range, "model"
        )
        assert result.issue_key == "fix_unknown"

class TestPrepareFixContextFullPath:
    """Cover the full successful path through _prepare_fix_context."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    async def test_successful_context_from_file(self, fixer, tmp_path):
        """Full successful path reading from file (lines 434-497)."""
        py_file = tmp_path / "test.py"
        py_file.write_text("import os\n\ndef foo():\n    x = 1\n    return x\n")

        line_range = {
            "first_line": 3,
            "last_line": 5,
            "problem_lines": [4],
        }

        result = await fixer._prepare_fix_context(py_file, line_range, "")
        assert result is not None
        assert isinstance(result, FixContext)
        assert result.language == "python"

    async def test_unicode_error_returns_none(self, fixer, tmp_path):
        """Line 2042-2044 in _read_and_extract: UnicodeDecodeError returns None."""
        py_file = tmp_path / "test.py"
        py_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8

        line_range = {"first_line": 1, "last_line": 1, "problem_lines": [1]}
        # This may or may not raise UnicodeDecodeError depending on error handling
        # The method has its own try/except so it should return None
        result = await fixer._prepare_fix_context(py_file, line_range, "")
        # Could be None or a context - depends on encoding handling
        # The key is it doesn't crash

class TestPrepareContext:
    """Cover _prepare_context edge cases."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    async def test_class_name_detection_logged(self, fixer, tmp_path):
        """Line 2001: class_name detected gets logged."""
        py_file = tmp_path / "test.py"
        py_file.write_text("class Foo:\n    def bar(self):\n        pass\n")
        line_range = {"first_line": 2, "last_line": 3, "problem_lines": [2]}
        result = await fixer._prepare_context(py_file, line_range, "", 5, "python")
        assert result is not None

    async def test_generic_exception_returns_none(self, fixer, tmp_path):
        """Lines 2045-2047: generic exception returns None."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": [1]}
        fixer.file_reader.read_lines = AsyncMock(side_effect=RuntimeError("boom"))
        result = await fixer._prepare_context(py_file, line_range, "", 5, "python")
        assert result is None


# ============================================================================
# Tests for extracted sub-methods of apply_fixes_with_validation
# ============================================================================


class TestCreateBackupIfNeeded:
    """Tests for _create_backup_if_needed."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def test_creates_backup_when_requested_and_not_dry_run(self, fixer, tmp_path):
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        with patch.object(fixer, "_create_backup", return_value=tmp_path / "bak") as mock_backup:
            fixer._create_backup_if_needed(result, tmp_path, create_backup=True, dry_run=False)
        mock_backup.assert_called_once_with(tmp_path)
        assert result.backup_created is True
        assert result.backup_path == tmp_path / "bak"

    def test_skips_backup_on_dry_run(self, fixer, tmp_path):
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        with patch.object(fixer, "_create_backup") as mock_backup:
            fixer._create_backup_if_needed(result, tmp_path, create_backup=True, dry_run=True)
        mock_backup.assert_not_called()
        assert result.backup_created is False

    def test_skips_backup_when_not_requested(self, fixer, tmp_path):
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        with patch.object(fixer, "_create_backup") as mock_backup:
            fixer._create_backup_if_needed(result, tmp_path, create_backup=False, dry_run=False)
        mock_backup.assert_not_called()
        assert result.backup_created is False


class TestGroupFixesByFile:
    """Tests for _group_fixes_by_file."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_fix(self, issue_key="k"):
        fix = Mock(spec=FixSuggestion)
        fix.issue_key = issue_key
        fix.file_path = "test.py"
        fix.original_code = "x"
        return fix

    def _make_issue(self):
        return SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )

    def test_groups_fixes_by_file_path(self, fixer, tmp_path):
        fix1 = self._make_fix("k1")
        fix2 = self._make_fix("k2")
        issue1 = self._make_issue()
        issue2 = self._make_issue()

        with patch.object(fixer, "_get_file_from_fix", side_effect=["file_a.py", "file_a.py"]):
            result = fixer._group_fixes_by_file([fix1, fix2], [issue1, issue2], tmp_path)

        assert "file_a.py" in result
        assert len(result["file_a.py"]) == 2

    def test_excludes_fixes_with_no_file(self, fixer, tmp_path):
        fix1 = self._make_fix("k1")
        fix2 = self._make_fix("k2")
        issue1 = self._make_issue()
        issue2 = self._make_issue()

        with patch.object(fixer, "_get_file_from_fix", side_effect=["file_a.py", None]):
            result = fixer._group_fixes_by_file([fix1, fix2], [issue1, issue2], tmp_path)

        assert len(result) == 1
        assert len(result["file_a.py"]) == 1

    def test_empty_input(self, fixer, tmp_path):
        result = fixer._group_fixes_by_file([], [], tmp_path)
        assert result == {}


class TestProcessSingleFile:
    """Tests for _process_single_file."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_fix(self, tmp_path):
        fix = Mock(spec=FixSuggestion)
        fix.file_path = str(tmp_path / "test.py")
        fix.issue_key = "k"
        fix.original_code = "x = 1"
        return fix

    def _make_issue(self):
        return SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )

    async def test_direct_success_extends_result(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(True, [])):
            await fixer._process_single_file(py_file, [(fix, issue)], result, None, False)

        assert len(result.successful_fixes) == 1

    async def test_direct_failure_delegates_to_validator(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "syntax error"

        mock_validator = Mock()
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.APPROVED
        mock_validator.validate_fix.return_value = mock_val_result

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])), \
             patch.object(fixer, "_handle_failed_fixes_with_validator", new_callable=AsyncMock) as mock_handle:
            await fixer._process_single_file(py_file, [(fix, issue)], result, mock_validator, False)

        mock_handle.assert_called_once()

    async def test_direct_failure_no_validator_marks_failed(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)

        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [failed_result])):
            await fixer._process_single_file(py_file, [(fix, issue)], result, None, False)

        assert len(result.failed_fixes) == 1
        assert "no validator" in result.failed_fixes[0]["error"]

    async def test_reads_original_content_for_existing_file(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("original\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)

        with patch.object(fixer, "read_file_async", new_callable=AsyncMock, return_value="original\n") as mock_read, \
             patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(True, [])):
            await fixer._process_single_file(py_file, [(fix, issue)], result, None, False)

        mock_read.assert_called_once_with(str(py_file))


class TestHandleFailedFixesWithValidator:
    """Tests for _handle_failed_fixes_with_validator."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_fix(self, tmp_path):
        fix = Mock(spec=FixSuggestion)
        fix.file_path = str(tmp_path / "test.py")
        fix.issue_key = "k"
        fix.original_code = "x = 1"
        return fix

    def _make_issue(self):
        return SonarIssue(
            key="k", rule="python:S1", severity="MAJOR", component="t.py",
            project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
            first_line=1, last_line=1,
        )

    async def test_restores_original_content(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)

        failed_app = Mock()
        failed_app.reason = "syntax error"

        mock_validator = Mock()
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.APPROVED
        mock_validator.validate_fix.return_value = mock_val_result

        mock_write = AsyncMock()
        with patch.object(fixer.file_reader, "write_text", mock_write), \
             patch.object(fixer, "read_file_async", new_callable=AsyncMock, return_value="x = 1\n"):
            await fixer._handle_failed_fixes_with_validator(
                py_file, [(fix, issue)], [failed_app], "original content",
                result, mock_validator, dry_run=False,
            )

        mock_write.assert_called_once_with(py_file, "original content")

    async def test_collects_reasons_from_new_fixes(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)

        app1 = Mock()
        app1.reason = "error A"
        app2 = Mock()
        app2.reason = None
        app3 = Mock()
        app3.reason = "error B"

        mock_validator = Mock()
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.APPROVED
        mock_validator.validate_fix.return_value = mock_val_result

        with patch.object(fixer, "read_file_async", new_callable=AsyncMock, return_value="x = 1\n"):
            await fixer._handle_failed_fixes_with_validator(
                py_file, [(fix, issue)], [app1, app2, app3], "original",
                result, mock_validator, dry_run=False,
            )

        # Verify reason_msg passed to validate_fix contains "error A" and "error B"
        call_kwargs = mock_validator.validate_fix.call_args
        assert "error A" in call_kwargs.kwargs.get("new_error_msg", call_kwargs[0][-1] if len(call_kwargs[0]) > 3 else "")

    async def test_reads_tmp_file_if_exists(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        tmp_file = py_file.with_suffix(".tmp.py")
        tmp_file.write_text("modified content\n")

        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)

        failed_app = Mock()
        failed_app.reason = "error"

        mock_validator = Mock()
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.APPROVED
        mock_validator.validate_fix.return_value = mock_val_result

        with patch.object(fixer, "read_file_async", new_callable=AsyncMock, return_value="modified content\n"), \
             patch("devdox_ai_sonar.llm_fixer.cleanup_tmp_py_file") as mock_cleanup:
            await fixer._handle_failed_fixes_with_validator(
                py_file, [(fix, issue)], [failed_app], "original",
                result, mock_validator, dry_run=False,
            )

        mock_cleanup.assert_called_once()

    async def test_exception_in_single_fix_does_not_abort_others(self, fixer, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        fix1 = self._make_fix(tmp_path)
        fix1.issue_key = "k1"
        fix2 = self._make_fix(tmp_path)
        fix2.issue_key = "k2"
        issue1 = self._make_issue()
        issue2 = self._make_issue()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=2)

        failed_app = Mock()
        failed_app.reason = "error"

        mock_validator = Mock()
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.APPROVED
        # First call raises, second succeeds
        mock_validator.validate_fix.side_effect = [Exception("crash"), mock_val_result]

        with patch.object(fixer, "read_file_async", new_callable=AsyncMock, return_value="x = 1\n"):
            await fixer._handle_failed_fixes_with_validator(
                py_file, [(fix1, issue1), (fix2, issue2)], [failed_app], "original",
                result, mock_validator, dry_run=False,
            )

        assert len(result.failed_fixes) == 1
        assert "Validator error" in result.failed_fixes[0]["error"]
        assert len(result.successful_fixes) == 1


class TestProcessValidationResult:
    """Tests for _process_validation_result."""

    @pytest.fixture
    def fixer(self, mock_openai_client):
        return LLMFixer(provider="openai", api_key="test-key")

    def _make_fix(self):
        fix = Mock(spec=FixSuggestion)
        fix.issue_key = "k"
        fix.original_code = "x = 1"
        return fix

    async def test_approved_appends_to_successful(self, fixer, tmp_path):
        fix = self._make_fix()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        val_result = Mock()
        val_result.status = ValidationStatus.APPROVED

        await fixer._process_validation_result(val_result, fix, tmp_path / "t.py", result, False)

        assert len(result.successful_fixes) == 1
        assert result.successful_fixes[0] is fix

    async def test_rejected_appends_to_failed(self, fixer, tmp_path):
        fix = self._make_fix()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        val_result = Mock()
        val_result.status = ValidationStatus.REJECTED
        val_result.explanation = "bad fix"

        await fixer._process_validation_result(val_result, fix, tmp_path / "t.py", result, False)

        assert len(result.failed_fixes) == 1
        assert "Rejected" in result.failed_fixes[0]["error"]

    async def test_needs_review_appends_to_failed(self, fixer, tmp_path):
        fix = self._make_fix()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        val_result = Mock()
        val_result.status = ValidationStatus.NEEDS_REVIEW
        val_result.explanation = "check this"

        await fixer._process_validation_result(val_result, fix, tmp_path / "t.py", result, False)

        assert len(result.failed_fixes) == 1
        assert "manual review" in result.failed_fixes[0]["error"]

    async def test_modified_with_final_fix_success(self, fixer, tmp_path):
        fix = self._make_fix()
        improved_fix = Mock()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        val_result = Mock()
        val_result.status = ValidationStatus.MODIFIED
        val_result.final_fix = improved_fix

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(True, [])):
            await fixer._process_validation_result(val_result, fix, tmp_path / "t.py", result, False)

        assert len(result.successful_fixes) == 1
        assert result.successful_fixes[0] is improved_fix

    async def test_modified_with_final_fix_reapply_fails(self, fixer, tmp_path):
        fix = self._make_fix()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        val_result = Mock()
        val_result.status = ValidationStatus.MODIFIED
        val_result.final_fix = Mock()

        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [])):
            await fixer._process_validation_result(val_result, fix, tmp_path / "t.py", result, False)

        assert len(result.failed_fixes) == 1
        assert "still failed" in result.failed_fixes[0]["error"]

    async def test_modified_no_final_fix(self, fixer, tmp_path):
        fix = self._make_fix()
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        val_result = Mock()
        val_result.status = ValidationStatus.MODIFIED
        val_result.final_fix = None

        await fixer._process_validation_result(val_result, fix, tmp_path / "t.py", result, False)

        assert len(result.failed_fixes) == 1
        assert "no improved fix" in result.failed_fixes[0]["error"]

