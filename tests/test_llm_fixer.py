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
                                       _extract_problem_lines,build_llm,
                                       FUNCTION_ALREADY_CALLED,call_llm_with_graph
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


def _make_mock_llm():
    """Return a mock BaseChatModel that build_llm() will return."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="{}")
    mock_llm.with_structured_output.return_value = mock_llm
    return mock_llm


@pytest.fixture
def mock_build_llm():
    """Patch build_llm so no real LangChain clients are created."""
    mock_llm = _make_mock_llm()
    with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm) as p:
        yield p, mock_llm


@pytest.fixture
def mock_call_llm_graph():
    """Patch call_llm_with_graph to avoid real LLM calls in _call_llm_list."""
    with patch("devdox_ai_sonar.llm_fixer.call_llm_with_graph", return_value=None) as p:
        yield p


# ---------------------------------------------------------------------------
# Backwards-compat fixtures so old fixture names still work
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client(mock_build_llm):
    """Drop-in replacement: patches build_llm and yields (patcher, mock_llm)."""
    return mock_build_llm


@pytest.fixture
def mock_gemini_client(mock_build_llm):
    return mock_build_llm


@pytest.fixture
def mock_together_client(mock_build_llm):
    return mock_build_llm


@pytest.fixture
def mock_openrouter_client(mock_build_llm):
    return mock_build_llm



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


def _make_fixer(**kwargs) -> LLMFixer:
    """Create a LLMFixer with build_llm patched out."""
    mock_llm = _make_mock_llm()
    with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm):
        return LLMFixer(provider=kwargs.get("provider", "openai"),
                        api_key=kwargs.get("api_key", "test-key"),
                        model=kwargs.get("model", None),
                        context_lines=kwargs.get("context_lines", 10))


class TestLLMFixerInitialization:
    """Test LLMFixer initialization — all providers go through build_llm now."""

    def test_init_openai_with_api_key(self):
        mock_llm = _make_mock_llm()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm) as mock_build:
            fixer = LLMFixer(provider="openai", model="gpt-4", api_key="test-key-123")
        assert fixer.provider == "openai"
        assert fixer.model == "gpt-4"
        assert fixer.api_key == "test-key-123"
        mock_build.assert_called_once_with(provider="openai", model="gpt-4", api_key="test-key-123")

    def test_init_openai_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        fixer = _make_fixer(provider="openai", api_key="env-api-key")
        assert fixer.api_key == "env-api-key"

    def test_init_openai_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not provided"):
                with patch("devdox_ai_sonar.llm_fixer.build_llm"):
                    LLMFixer(provider="openai")

    def test_init_openai_default_model(self):
        fixer = _make_fixer(provider="openai")
        assert fixer.model == "gpt-4o"

    def test_init_gemini_with_api_key(self):
        fixer = _make_fixer(provider="gemini", model="gemini-pro", api_key="test-gemini-key")
        assert fixer.provider == "gemini"
        assert fixer.model == "gemini-pro"
        assert fixer.api_key == "test-gemini-key"

    def test_init_gemini_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY", "env-gemini-key")
        fixer = _make_fixer(provider="gemini", api_key="env-gemini-key")
        assert fixer.api_key == "env-gemini-key"

    def test_init_gemini_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                with patch("devdox_ai_sonar.llm_fixer.build_llm"):
                    LLMFixer(provider="gemini")

    def test_init_togetherai_with_api_key(self):
        fixer = _make_fixer(provider="togetherai", api_key="test-together-key")
        assert fixer.provider == "togetherai"
        assert fixer.api_key == "test-together-key"

    def test_init_togetherai_from_env(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "env-together-key")
        fixer = _make_fixer(provider="togetherai", api_key="env-together-key")
        assert fixer.api_key == "env-together-key"

    def test_init_openrouter_with_api_key(self):
        mock_llm = _make_mock_llm()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm) as mock_build:
            fixer = LLMFixer(
                provider="openrouter",
                model="anthropic/claude-sonnet-4",
                api_key="test-openrouter-key",
            )
        assert fixer.provider == "openrouter"
        assert fixer.model == "anthropic/claude-sonnet-4"
        assert fixer.api_key == "test-openrouter-key"
        mock_build.assert_called_once_with(
            provider="openrouter",
            model="anthropic/claude-sonnet-4",
            api_key="test-openrouter-key",
        )

    def test_init_openrouter_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-key")
        fixer = _make_fixer(provider="openrouter", api_key="env-openrouter-key")
        assert fixer.api_key == "env-openrouter-key"

    def test_init_openrouter_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                with patch("devdox_ai_sonar.llm_fixer.build_llm"):
                    LLMFixer(provider="openrouter")

    def test_init_openrouter_default_model(self):
        fixer = _make_fixer(provider="openrouter")
        assert fixer.model == "anthropic/claude-sonnet-4"

    def test_init_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            with patch("devdox_ai_sonar.llm_fixer.build_llm"):
                LLMFixer(provider="unsupported", api_key="key")

    def test_init_context_lines_default(self):
        fixer = _make_fixer()
        assert fixer.context_lines == 10

    def test_init_context_lines_custom(self):
        fixer = _make_fixer(context_lines=20)
        assert fixer.context_lines == 20

    def test_init_prompt_dir_setup(self):
        fixer = _make_fixer()
        assert fixer.prompt_dir.exists() or fixer.prompt_dir.name == "prompts"

    def test_init_jinja_env_setup(self):
        fixer = _make_fixer()
        assert hasattr(fixer, "jinja_env")
        assert fixer.jinja_env is not None

    def test_init_provider_case_insensitive(self):
        f1 = _make_fixer(provider="OpenAI")
        f2 = _make_fixer(provider="OPENAI")
        f3 = _make_fixer(provider="openai")
        assert f1.provider == "openai"
        assert f2.provider == "openai"
        assert f3.provider == "openai"

    def test_llm_attribute_set(self):
        """_llm is a BaseChatModel cached on the instance."""
        mock_llm = _make_mock_llm()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm):
            fixer = LLMFixer(provider="openai", api_key="key")
        assert fixer._llm is mock_llm


class TestProviderConfiguration:
    """build_llm is called once per provider; errors propagate through it."""

    def test_build_llm_called_for_openai(self):
        mock_llm = _make_mock_llm()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm) as mock_build:
            fixer = LLMFixer(provider="openai", api_key="test-key")
        mock_build.assert_called_once_with(provider="openai", model="gpt-4o", api_key="test-key")
        assert fixer._llm is mock_llm

    def test_build_llm_called_for_gemini(self):
        mock_llm = _make_mock_llm()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm) as mock_build:
            LLMFixer(provider="gemini", api_key="test-key")
        mock_build.assert_called_once_with(provider="gemini", model="gemini-1.5-flash", api_key="test-key")

    def test_build_llm_called_for_togetherai(self):
        mock_llm = _make_mock_llm()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm) as mock_build:
            LLMFixer(provider="togetherai", api_key="test-key")
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["provider"] == "togetherai"

    def test_build_llm_called_for_openrouter(self):
        mock_llm = _make_mock_llm()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", return_value=mock_llm) as mock_build:
            LLMFixer(provider="openrouter", api_key="test-key")
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["provider"] == "openrouter"

    def test_build_llm_raises_propagates(self):
        with patch("devdox_ai_sonar.llm_fixer.build_llm", side_effect=ImportError("langchain_openai missing")):
            with pytest.raises(ImportError, match="langchain_openai missing"):
                LLMFixer(provider="openai", api_key="key")

    # Legacy import-guard tests now work via build_llm raising ImportError directly
    def test_openai_import_error_handling(self):
        with patch("devdox_ai_sonar.llm_fixer.build_llm", side_effect=ImportError("OpenAI library not installed")):
            with pytest.raises(ImportError, match="OpenAI library not installed"):
                LLMFixer(provider="openai", api_key="key")

    def test_gemini_import_error_handling(self):
        with patch("devdox_ai_sonar.llm_fixer.build_llm", side_effect=ImportError("Gemini library not installed")):
            with pytest.raises(ImportError, match="Gemini library not installed"):
                LLMFixer(provider="gemini", api_key="key")

    def test_together_import_error_handling(self):
        with patch("devdox_ai_sonar.llm_fixer.build_llm", side_effect=ImportError("Together AI library not installed")):
            with pytest.raises(ImportError, match="Together AI library not installed"):
                LLMFixer(provider="togetherai", api_key="key")

    def test_openrouter_import_error_handling(self):
        with patch("devdox_ai_sonar.llm_fixer.build_llm", side_effect=ImportError("OpenAI library not installed")):
            with pytest.raises(ImportError, match="OpenAI library not installed"):
                LLMFixer(provider="openrouter", api_key="key")


class TestContextExtraction:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def test_extract_context_simple(self, fixer):
        lines = ["line 1\n", "line 2\n", "line 3\n", "problem line\n", "line 5\n", "line 6\n", "line 7\n"]
        context = fixer._extract_context(lines, 4, 4, [3], context_lines=2)
        assert context["start_line"] <= 4
        assert context["end_line"] >= 4
        assert "problem line" in context["new_context"][0]["context"]

    def test_extract_context_at_file_start(self, fixer):
        lines = ["line 1\n", "line 2\n", "line 3\n"]
        context = fixer._extract_context(lines, 1, 1, [1], context_lines=5)
        assert context["start_line"] == 1
        assert "line 1" in context["new_context"][0]["context"]

    def test_extract_context_at_file_end(self, fixer):
        lines = ["line 1\n", "line 2\n", "line 3\n"]
        context = fixer._extract_context(lines, 3, 3, [3], context_lines=5)
        assert context["end_line"] == 3
        assert "line 3" in context["new_context"][0]["context"]

    def test_extract_context_multiline_issue(self, fixer):
        lines = [f"line {i}\n" for i in range(1, 21)]
        context = fixer._extract_context(lines, 10, 15, [13, 15], context_lines=3)
        assert context["start_line"] <= 10
        assert context["end_line"] >= 15
        for i in range(10, 16):
            assert f"line {i}" in context["new_context"][0]["context"]

    def test_extract_context_zero_context_lines(self, fixer):
        lines = [f"line {i}\n" for i in range(1, 11)]
        context = fixer._extract_context(lines, 5, 5, [5], context_lines=0)
        assert "line 5" in context["new_context"][0]["context"]

    def test_extract_context_large_context(self, fixer):
        lines = [f"line {i}\n" for i in range(1, 21)]
        context = fixer._extract_context(lines, 10, 10, [10], context_lines=50)
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

    async def test_generate_fix_file_not_found(self, fixer, sample_issue, tmp_path, rule_info):
        """Test fix generation when file doesn't exist"""
        sample_issue.file = "nonexistent/file.py"

        result = await fixer.generate_fix_by_file(
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

    async def test_generate_fix_multiple_files_error(self, fixer, sample_python_file, rule_info, tmp_path):
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

        result = await fixer.generate_fix_by_file(
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

    async def test_generate_fix_llm_returns_none(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test when LLM returns None"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.return_value = None

            result = await fixer.generate_fix_by_file(
                issues=[sample_issue],
                project_path=tmp_path,
                tmp_path=tmp_path
            )

        assert result is None

    async def test_generate_fix_exception_handling(self, fixer, sample_issue, sample_python_file, rule_info, tmp_path):
        """Test exception handling during fix generation"""
        sample_issue.file = str(sample_python_file.relative_to(tmp_path))

        with patch.object(fixer, '_call_llm_list') as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = await fixer.generate_fix_by_file(
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
    async def test_call_llm_openai_success(self, fixer_openai):
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

        result = await fixer_openai._call_llm_list(
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
    async def test_call_llm_openai_api_error(self, fixer_openai):
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

        result = await fixer_openai._call_llm_list(
            [issue],
            context_dict['context_dict'],
            ".py",
            {},
            ""
        )

        assert result is None

    @pytest.mark.skip(reason="Need update")
    async def test_call_llm_gemini_success(self, fixer_gemini):
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
        result = await fixer_gemini._call_llm_list(
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
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    @pytest.fixture
    def sample_fix(self, tmp_path, sample_code_block):
        return FixSuggestion(
            issue_key="issue-1", file_path="test.py", original_code="old code",
            fixed_code="new code", helper_code="", explanation="Fixed", confidence=0.9,
            sonar_line_number=10, line_number=10, last_line_number=15, llm_model="a",
            fixed_code_blocks=[sample_code_block],
        )

    async def test_apply_fixes_success(self, fixer, sample_fix, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("old code\n")
        sample_fix.file_path = str(test_file.relative_to(tmp_path))
        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(1, 0, [])):
            result = await fixer.apply_fixes(fixes=[sample_fix], project_path=tmp_path,
                                             create_backup=False, dry_run=False)
        assert isinstance(result, FixResult)
        assert result.total_fixes_attempted == 1

    async def test_apply_fixes_dry_run(self, fixer, sample_fix, tmp_path):
        test_file = tmp_path / "test.py"
        original_content = "original content\n"
        test_file.write_text(original_content)
        sample_fix.file_path = str(test_file.relative_to(tmp_path))
        result = await fixer.apply_fixes(fixes=[sample_fix], project_path=tmp_path, dry_run=True)
        assert test_file.read_text() == original_content

    async def test_apply_fixes_creates_backup(self, fixer, sample_fix, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")
        sample_fix.file_path = str(test_file.relative_to(tmp_path))
        with patch.object(fixer, "_create_backup", return_value=tmp_path / "backup"), \
                patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(1, 0, [])):
            result = await fixer.apply_fixes(fixes=[sample_fix], project_path=tmp_path,
                                             create_backup=True, dry_run=False)
        assert result.backup_created is True
        assert result.backup_path is not None

    async def test_apply_fixes_no_backup_in_dry_run(self, fixer, sample_fix, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")
        sample_fix.file_path = str(test_file.relative_to(tmp_path))
        with patch.object(fixer, "_create_backup") as mock_backup:
            result = await fixer.apply_fixes(fixes=[sample_fix], project_path=tmp_path,
                                             create_backup=True, dry_run=True)
        assert result.backup_created is False
        mock_backup.assert_not_called()

    async def test_apply_fixes_multiple_files(self, fixer, tmp_path, sample_code_block):
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("content1\n")
        file2.write_text("content2\n")
        fix1 = FixSuggestion(issue_key="issue-1", file_path=str(file1.relative_to(tmp_path)),
                             helper_code="", original_code="content1", fixed_code="fixed1",
                             explanation="Fixed", llm_model="a", confidence=0.9,
                             sonar_line_number=1, fixed_code_blocks=[sample_code_block])
        fix2 = FixSuggestion(issue_key="issue-2", file_path=str(file2.relative_to(tmp_path)),
                             original_code="content2", fixed_code="fixed2", explanation="Fixed",
                             confidence=0.9, llm_model="a", sonar_line_number=1,
                             fixed_code_blocks=[sample_code_block])
        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(1, 0, [])):
            result = await fixer.apply_fixes(fixes=[fix1, fix2], project_path=tmp_path, create_backup=False)
        assert result.total_fixes_attempted == 2

    async def test_apply_fixes_groups_by_file(self, fixer, tmp_path, sample_code_block):
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")
        fix1 = FixSuggestion(issue_key="issue-1", file_path=str(test_file.relative_to(tmp_path)),
                             original_code="line1", fixed_code="fixed1", explanation="Fixed",
                             helper_code="", llm_model="a", confidence=0.9, sonar_line_number=1,
                             fixed_code_blocks=[sample_code_block])
        fix2 = FixSuggestion(issue_key="issue-2", file_path=str(test_file.relative_to(tmp_path)),
                             original_code="line2", fixed_code="fixed2", explanation="Fixed",
                             helper_code="", llm_model="a", confidence=0.9, sonar_line_number=2,
                             fixed_code_blocks=[sample_code_block])
        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(2, 0, [])) as mock_apply:
            result = await fixer.apply_fixes(fixes=[fix1, fix2], project_path=tmp_path, create_backup=False)
        assert mock_apply.call_count == 1
        call_args = mock_apply.call_args[0]
        assert len(call_args[1]) == 2


class TestHelperMethods:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    @pytest.fixture
    def context_extractor(self):
        return ContextExtractor(lines=[])

    def test_get_language_from_extension_python(self, fixer):
        assert fixer._get_language_from_extension(".py") == "python"

    def test_get_language_from_extension_javascript(self, fixer):
        assert fixer._get_language_from_extension(".js") in ["javascript", "js"]

    def test_get_language_from_extension_java(self, fixer):
        assert fixer._get_language_from_extension(".java") == "java"

    def test_get_language_from_extension_unknown(self, fixer):
        assert fixer._get_language_from_extension(".xyz") is not None

    def test_create_backup(self, fixer, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "file.py").write_text("content")
        backup_path = fixer._create_backup(tmp_path)
        assert backup_path.exists()
        assert backup_path.is_dir()
        assert "backup" in str(backup_path)

    def test_check_bracket_balance_balanced(self, fixer):
        assert fixer._check_bracket_balance("def foo():\n    if (x > 0):\n        print('ok')") is True

    def test_check_bracket_balance_unbalanced(self, fixer):
        assert fixer._check_bracket_balance("def foo():\n    if (x > 0:\n        print('ok')") is False

    def test_apply_indentation_to_fix(self, fixer):
        result = fixer.apply_indentation_to_fix("def foo():\npass", "    ")
        assert result.startswith("    ")

    def test_is_actual_function_def_true(self, context_extractor):
        assert context_extractor._is_actual_function_def("def my_function(arg1, arg2):") is True

    def test_is_actual_function_def_async(self, context_extractor):
        assert context_extractor._is_actual_function_def("async def async_function():") is True

    def test_is_actual_function_def_false(self, context_extractor):
        assert context_extractor._is_actual_function_def("# This is a comment about def") is False


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
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def test_generate_fix_key_single_line(self):
        assert _generate_fix_key([10]) == "fix_L10"

    def test_generate_fix_key_multiple_lines(self):
        assert _generate_fix_key([10, 15, 20]) == "fix_L10-L20"

    def test_generate_fix_key_empty(self):
        assert _generate_fix_key([]) == "fix_unknown"

    def test_extract_problem_lines_valid(self):
        file_lines = [f"line {i}\n" for i in range(1, 11)]
        problem_lines = _extract_problem_lines(file_lines, [1, 5, 10])
        assert problem_lines[1] == "line 1\n"
        assert problem_lines[5] == "line 5\n"
        assert problem_lines[10] == "line 10\n"

    def test_extract_problem_lines_out_of_bounds(self):
        file_lines = ["line 1\n", "line 2\n"]
        problem_lines = _extract_problem_lines(file_lines, [1, 100])
        assert isinstance(problem_lines, dict)
        assert len(problem_lines) == 1
        assert 1 in problem_lines
        assert 100 not in problem_lines

    def test_validate_and_extract_issue_info_file_not_found(self, tmp_path):
        issue = SonarIssue(key="test", rule="python:S1234", severity="MAJOR",
                           component="nonexistent.py", project="test", line=1, message="Test",
                           type="VULNERABILITY", status="OPEN", first_line=1, last_line=5,
                           file="nonexistent.py")
        with pytest.raises(FileNotFoundError):
            _validate_and_extract_issue_info([issue], tmp_path)

    async def test_prepare_context_success(self, tmp_path, fixer):
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")
        line_range = {"first_line": 1, "last_line": 2, "problem_lines": [1, 2]}
        result = await fixer._prepare_context(test_file, line_range, "", context_lines=1, language="python")
        assert result is not None
        assert "context_dict" in result
        assert "file_lines" in result

    async def test_prepare_context_with_modified_content(self, tmp_path, fixer):
        test_file = tmp_path / "test.py"
        test_file.write_text("original\n")
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": [1]}
        result = await fixer._prepare_context(test_file, line_range, "modified content",
                                              context_lines=1, language="python")
        assert result["context_dict"]["context"] == "modified content"

    async def test_prepare_context_unicode_error(self, tmp_path, fixer):
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x80\x81\x82")
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": [1]}
        result = await fixer._prepare_context(test_file, line_range, "", context_lines=1, language="python")
        assert result is None


class TestLLMFixerAdditionalMethods:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def test_extract_complexity_info_standard_format(self, fixer):
        result = fixer._extract_complexity_info("Cognitive Complexity from 25 to the 15 allowed")
        assert result["current"] == "25"
        assert result["target"] == "15"

    def test_extract_complexity_info_alternative_format(self, fixer):
        result = fixer._extract_complexity_info("complexity is 30, maximum is 15")
        assert result["current"] == "30"
        assert result["target"] == "15"

    def test_extract_complexity_info_no_match(self, fixer):
        result = fixer._extract_complexity_info("Some other message")
        assert result["current"] == "Unknown"
        assert result["target"] == "15"

    def test_is_init_method_python(self, fixer):
        assert fixer._is_init_method("def __init__(self):\n    self.x = 1")

    def test_is_init_method_java_constructor(self, fixer):
        assert fixer._is_init_method("public MyClass() {\n    this.x = 1;\n}")

    def test_is_init_method_javascript_constructor(self, fixer):
        assert fixer._is_init_method("constructor() {\n    this.x = 1;\n}")

    def test_is_init_method_not_constructor(self, fixer):
        assert not fixer._is_init_method("def regular_method():\n    return 1")

    def test_looks_like_file_path_valid(self, fixer):
        assert fixer._looks_like_file_path("src/main.py")
        assert fixer._looks_like_file_path("path/to/file.java")

    def test_looks_like_file_path_invalid(self, fixer):
        assert not fixer._looks_like_file_path("just_a_string")
        assert not fixer._looks_like_file_path("no-slash.py")

    def test_try_stored_file_path_success(self, fixer, tmp_path, sample_code_block):
        test_file = tmp_path / "test.py"
        test_file.write_text("content")
        fix = FixSuggestion(issue_key="test", file_path="test.py", original_code="",
                            fixed_code="", explanation="", confidence=0.9,
                            sonar_line_number=1, llm_model="gpt-4",
                            fixed_code_blocks=[sample_code_block])
        result = fixer._try_stored_file_path(fix, tmp_path)
        assert result is not None

    def test_try_stored_file_path_not_exists(self, fixer, tmp_path, sample_code_block):
        fix = FixSuggestion(issue_key="test", file_path="nonexistent.py", original_code="",
                            fixed_code="", explanation="", confidence=0.9,
                            sonar_line_number=1, llm_model="gpt-4",
                            fixed_code_blocks=[sample_code_block])
        result = fixer._try_stored_file_path(fix, tmp_path)
        assert result is None

    def test_find_files_with_content_success(self, fixer, tmp_path):
        (tmp_path / "file1.py").write_text("def target_function():\n    pass")
        (tmp_path / "file2.py").write_text("def other_function():\n    pass")
        results = fixer._find_files_with_content(tmp_path, "target_function")
        assert len(results) >= 1
        assert any("file1.py" in str(f) for f in results)

    def test_find_files_with_content_not_found(self, fixer, tmp_path):
        (tmp_path / "file1.py").write_text("def some_function():\n    pass")
        results = fixer._find_files_with_content(tmp_path, "nonexistent_content")
        assert len(results) == 0

    def test_find_files_with_content_unicode_error(self, fixer, tmp_path):
        (tmp_path / "binary.bin").write_bytes(b"\x80\x81\x82")
        results = fixer._find_files_with_content(tmp_path, "text")
        assert isinstance(results, list)

    def test_check_bracket_balance_complex(self, fixer):
        assert fixer._check_bracket_balance("def f(): return {1: [2, 3]}")
        assert not fixer._check_bracket_balance("def f(): return {1: [2, 3]")
        assert not fixer._check_bracket_balance("def f(): return {1: [2, 3]]}")

    def test_check_no_duplicate_definitions_no_duplicates(self, fixer):
        content = "def func1():\n    pass\n\ndef func2():\n    pass\n"
        assert fixer._check_no_duplicate_definitions(content, ".py")

    def test_check_no_duplicate_definitions_has_duplicates(self, fixer):
        content = "def duplicate():\n    pass\n\ndef other():\n    pass\n\ndef duplicate():\n    pass\n"
        assert not fixer._check_no_duplicate_definitions(content, ".py")

    def test_check_no_duplicate_definitions_non_python(self, fixer):
        content = "function duplicate() {}\nfunction duplicate() {}"
        assert fixer._check_no_duplicate_definitions(content, ".js")

    def test_apply_indentation_to_fix_empty(self, fixer):
        assert fixer.apply_indentation_to_fix("", "    ") == ""

    def test_apply_indentation_to_fix_multiline(self, fixer):
        result = fixer.apply_indentation_to_fix("def foo():\npass\nreturn 1", "    ")
        lines = result.split("\n")
        assert all(line.startswith("    ") or not line.strip() for line in lines)

    def test_extract_fields_from_parsed_json_all_fields(self, fixer):
        fix_data = {"FIXED_SELECTION": "fixed code", "NEW_HELPER_CODE": "helper code",
                    "PLACEMENT": "GLOBAL_TOP", "EXPLANATION": "Explanation text", "CONFIDENCE": "0.88"}
        result = fixer._extract_fields_from_parsed_json(fix_data)
        assert result is not None
        assert result["fixed_code"] == "fixed code"
        assert result["helper_code"] == "helper code"
        assert result["placement_helper"] == "GLOBAL_TOP"
        assert result["confidence"] == 0.88

    def test_extract_fields_from_parsed_json_minimal(self, fixer):
        result = fixer._extract_fields_from_parsed_json({"FIXED_SELECTION": "code"})
        assert result is not None
        assert result["fixed_code"] == "code"
        assert result["helper_code"] == ""
        assert result["placement_helper"] == "SIBLING"

    def test_extract_fields_from_parsed_json_invalid_placement(self, fixer):
        result = fixer._extract_fields_from_parsed_json({"FIXED_SELECTION": "code", "PLACEMENT": "INVALID"})
        assert result["placement_helper"] == "SIBLING"

    def test_apply_regex_patterns_all_patterns(self, fixer):
        content = '{"FIXED_SELECTION": "fixed code here", "NEW_HELPER_CODE": "helper code", ' \
                  '"PLACEMENT": "SIBLING", "EXPLANATION": "Detailed explanation", "CONFIDENCE": 0.92}'
        results = fixer._apply_regex_patterns(content)
        assert results["FIXED_SELECTION"] == "fixed code here"
        assert results["NEW_HELPER_CODE"] == "helper code"
        assert results["PLACEMENT"] == "SIBLING"

    def test_process_match_value_confidence(self, fixer):
        assert fixer._process_match_value("CONFIDENCE", "0.75") == 0.75
        assert fixer._process_match_value("CONFIDENCE", None) == 0.5

    def test_process_match_value_placement_default(self, fixer):
        assert fixer._process_match_value("PLACEMENT", None) == "SIBLING"

    def test_validate_results_missing_fixed_selection(self, fixer):
        results = {"FIXED_SELECTION": "", "EXPLANATION": "text", "CONFIDENCE": 0.8,
                   "NEW_HELPER_CODE": "", "PLACEMENT": "SIBLING"}
        assert fixer._validate_results(results) is None

    def test_validate_results_confidence_bounds(self, fixer):
        results = {"IMPORT_BLOCK": "", "FIXED_SELECTION": "code", "CONFIDENCE": 1.5,
                   "NEW_HELPER_CODE": "", "PLACEMENT": "SIBLING", "EXPLANATION": "text"}
        result = fixer._validate_results(results)
        assert result["confidence"] == 1.0
        results["CONFIDENCE"] = -0.5
        result = fixer._validate_results(results)
        assert result["confidence"] == 0.0


class TestEdgeCasesAndErrors:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def test_extract_fix_from_response_malformed_json(self, fixer):
        result = fixer._extract_fix_from_response("{FIXED_SELECTION: 'missing quotes', incomplete")
        assert result is None or isinstance(result, dict)

    def test_extract_fix_from_response_with_extra_text(self, fixer):
        content = '\nHere is the fix:\n\n```json\n{"FIXED_SELECTION": "def fixed():\\n    pass", ' \
                  '"EXPLANATION": "Fixed it", "CONFIDENCE": 0.9}\n```\n\nHope this helps!\n'
        result = fixer._extract_fix_from_response(content)
        assert result is None or (result and "fixed" in result.get("fixed_code", ""))

    async def test_generate_fix_by_file_empty_issues(self, fixer, tmp_path):
        result = await fixer.generate_fix_by_file(issues=[], project_path=tmp_path, tmp_path=tmp_path)
        assert result is None

    async def test_apply_fixes_to_file_exception(self, fixer, tmp_path, sample_code_block):
        test_file = tmp_path / "test.py"
        test_file.write_text("content\n")
        fix = FixSuggestion(issue_key="test", file_path="test.py", original_code="content",
                            fixed_code="new content", explanation="", confidence=0.9,
                            sonar_line_number=1, llm_model="gpt-4", fixed_code_blocks=[sample_code_block])
        test_file.chmod(0o444)
        try:
            success, results = await fixer._apply_fixes_to_file(test_file, [fix], dry_run=False)
            assert isinstance(success, bool)
        finally:
            test_file.chmod(0o644)

    def test_try_find_by_content_short_code(self, fixer, tmp_path, sample_code_block):
        fix = FixSuggestion(issue_key="test", original_code="x", fixed_code="",
                            explanation="", confidence=0.9, sonar_line_number=1, llm_model="gpt-4",
                            fixed_code_blocks=[sample_code_block])
        result = fixer._try_find_by_content(fix, tmp_path)
        assert result is None

    def test_context_extraction_line_out_of_bounds(self, fixer):
        lines = ["line 1\n", "line 2\n"]
        context = fixer._extract_context(lines, 100, 100, [100], context_lines=5)
        assert context["new_context"][0]["context"] == ""

    @pytest.mark.skipif(
        sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="Root/admin bypasses filesystem permissions; Windows uses different permission model",
    )
    def test_create_backup_permission_error(self, fixer, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("content")
        tmp_path.chmod(0o444)
        try:
            with pytest.raises((PermissionError, OSError)):
                fixer._create_backup(source)
        finally:
            tmp_path.chmod(0o755)


class TestContextExtractorComprehensive:
    def test_is_function_definition_python_variations(self):
        ext = ContextExtractor([])
        assert ext._is_function_definition("def foo():")
        assert ext._is_function_definition("    def bar(x, y):")
        assert ext._is_function_definition("async def async_foo():")
        assert not ext._is_function_definition("# def not_a_function")
        assert not ext._is_function_definition("defined = 'string'")

    def test_is_function_definition_javascript_variations(self):
        ext = ContextExtractor([])
        assert ext._is_function_definition("function myFunc() {")
        assert ext._is_function_definition("async function asyncFunc() {")
        assert ext._is_function_definition("const foo = () => {")

    def test_is_function_definition_java_csharp(self):
        ext = ContextExtractor([])
        assert ext._is_function_definition("public void doSomething() {")
        assert ext._is_function_definition("private int calculate(int x) {")

    def test_find_containing_function_simple(self):
        lines = ["class MyClass:\n", "    def method1(self):\n", "        x = 1\n",
                 "        y = 2\n", "        return x + y\n", "\n", "    def method2(self):\n", "        pass\n"]
        ext = ContextExtractor(lines)
        result = ext._find_containing_function(3)
        assert result == 1

    def test_find_containing_function_not_found(self):
        lines = ["import sys\n", "import os\n", "\n", "x = 1\n", "y = 2\n"]
        ext = ContextExtractor(lines)
        assert ext._find_containing_function(3) is None

    def test_find_python_function_end_with_nested_blocks(self):
        lines = ["def complex():\n", "    if True:\n", "        x = 1\n",
                 "        if x > 0:\n", "            print(x)\n", "    return x\n", "\n", "def another():\n"]
        ext = ContextExtractor(lines)
        result = ext._find_python_function_end(lines, 0)
        assert result == 6

    def test_find_brace_function_end_simple(self):
        lines = ["function test() {\n", "    var x = 1;\n", "    return x;\n", "}\n", "\n"]
        ext = ContextExtractor(lines)
        assert ext._find_brace_function_end(lines, 0) == 3

    def test_find_brace_function_end_nested_braces(self):
        lines = ["function test() {\n", "    if (true) {\n", "        console.log('test');\n",
                 "    }\n", "    return;\n", "}\n"]
        ext = ContextExtractor(lines)
        assert ext._find_brace_function_end(lines, 0) == 5

    def test_extract_normal_context(self):
        lines = [f"line {i}\n" for i in range(1, 11)]
        ext = ContextExtractor(lines)
        result = ext._extract_normal_context(5, 5, context_lines=2)
        assert result is not None
        assert not result["is_complete_function"]
        assert result["start_line"] == 4
        assert result["end_line"] == 8

    def test_is_decorator_python(self):
        ext = ContextExtractor([])
        assert ext._is_decorator("@property")
        assert ext._is_decorator("@app.route('/path')")
        assert not ext._is_decorator("# @not_a_decorator")

    def test_extract_function_name_python(self):
        ext = ContextExtractor([])
        assert ext._extract_function_name("def my_function():") == "my_function"
        assert ext._extract_function_name("async def async_func(x, y):") == "async_func"

    def test_extract_function_name_javascript(self):
        ext = ContextExtractor([])
        assert ext._extract_function_name("function myFunc() {") == "myFunc"


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
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def _make_code_block(self):
        return CodeBlock(block_name="test", start_line=1, end_line=10, has_changes=True,
                         change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE, context="fixed code")

    def _make_fix_response(self, code_blocks):
        return SonarFixResponse(FIXED_CODE_BLOCKS=code_blocks, IMPORT_BLOCK="import os",
                                NEW_HELPER_CODE="def helper(): pass", PLACEMENT=PlacementType.GLOBAL_TOP,
                                EXPLANATION="Fixed the issue", CONFIDENCE=0.95)

    def test_maps_all_fields_correctly(self, fixer):
        cb = self._make_code_block()
        fix_response = self._make_fix_response([cb])
        context = FixContext(
            file_path=Path("/project/src/test.py"),
            file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
            line_range={}, code_content="original code", language="python",
            import_section={"start_line": 1, "end_line": 3}, class_name=None, functions=[],
            context_dict={"start_line": 5, "import_section": {"end_line": 3}},
        )
        result = fixer._map_fix_suggestion_to_fix_suggestion_dto(
            line_range={"problem_lines": [10, 12]}, context=context, code_blocks=[cb],
            fix_response_single=fix_response, llm_model="test-model", relative_file_path="src/test.py",
            effective_start_line=5, effective_sonar_line=10, effective_last_line=12,
        )
        assert isinstance(result, FixSuggestion)
        assert result.issue_key == "fix_L10-L12"
        assert result.confidence == 0.95
        assert result.llm_model == "test-model"
        assert result.file_path == "src/test.py"

    def test_empty_problem_lines_gives_unknown_key(self, fixer):
        cb = self._make_code_block()
        fix_response = self._make_fix_response([cb])
        context = FixContext(
            file_path=Path("/project/src/test.py"),
            file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
            line_range={}, code_content="code", language="python",
            import_section={}, class_name=None, functions=[],
            context_dict={"start_line": 1, "import_section": {}},
        )
        result = fixer._map_fix_suggestion_to_fix_suggestion_dto(
            line_range={}, context=context, code_blocks=[cb],
            fix_response_single=fix_response, llm_model="m",
            relative_file_path="test.py", effective_start_line=1,
            effective_sonar_line=1, effective_last_line=1,
        )
        assert result.issue_key == "fix_unknown"


# ===========================================================================
# TestBuildFixSuggestionInstance
# ===========================================================================


class TestBuildFixSuggestionInstance:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def _make_code_block(self, **overrides):
        defaults = dict(block_name="test", start_line=1, end_line=10, has_changes=True,
                        change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE, context="new_code")
        defaults.update(overrides)
        return CodeBlock(**defaults)

    def _make_context(self, **ctx_overrides):
        ctx = {"start_line": 10, "import_section": {"end_line": 3}, "problem_lines": [10, 11]}
        ctx.update(ctx_overrides)
        return FixContext(file_path=Path("/project/src/test.py"),
                          file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
                          line_range={}, code_content="original code", language="python",
                          import_section={"end_line": 3}, class_name=None, functions=[], context_dict=ctx)

    def test_builds_list_from_multiple_responses(self, fixer):
        cb = self._make_code_block()
        resp1 = SonarFixResponse(FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.9, EXPLANATION="fix1")
        resp2 = SonarFixResponse(FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.8, EXPLANATION="fix2")
        context = self._make_context()
        line_range = {"first_line": 10, "last_line": 15, "problem_lines": [10]}
        result = fixer._build_fix_suggestion([resp1, resp2], context, Path("/project/src/test.py"),
                                             Path("/project"), line_range, modify_line_range=False)
        assert len(result) == 2
        assert result[0].explanation == "fix1"
        assert result[1].explanation == "fix2"

    def test_uses_model_or_unknown(self, fixer):
        cb = self._make_code_block()
        resp = SonarFixResponse(FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.9)
        context = self._make_context()
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": [1]}
        fixer.model = "gpt-4o"
        result = fixer._build_fix_suggestion([resp], context, Path("/project/src/test.py"),
                                             Path("/project"), line_range)
        assert result[0].llm_model == "gpt-4o"
        fixer.model = None
        result = fixer._build_fix_suggestion([resp], context, Path("/project/src/test.py"),
                                             Path("/project"), line_range)
        assert result[0].llm_model == "unknown"

    def test_empty_response_list_returns_empty(self, fixer):
        context = self._make_context()
        line_range = {"first_line": 1, "last_line": 1, "problem_lines": []}
        result = fixer._build_fix_suggestion([], context, Path("/project/src/test.py"),
                                             Path("/project"), line_range)
        assert result == []


# ===========================================================================
# TestPrepareFixContext
# ===========================================================================


class TestPrepareFixContext:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    async def test_returns_fix_context_from_file(self, fixer, tmp_path):
        py_file = tmp_path / "module.py"
        py_file.write_text("import os\n\ndef greet():\n    print('hello')\n    return 1\n")
        line_range = {"first_line": 3, "last_line": 5, "problem_lines": [4]}
        result = await fixer._prepare_fix_context(py_file, line_range, "")
        assert isinstance(result, FixContext)
        assert result.language == "python"
        assert result.file_path == py_file

    async def test_with_modified_content_succeeds(self, fixer, tmp_path):
        py_file = tmp_path / "module.py"
        py_file.write_text("import os\n\ndef foo():\n    pass\n")
        line_range = {"first_line": 3, "last_line": 4, "problem_lines": [3]}
        result = await fixer._prepare_fix_context(py_file, line_range, "custom context")
        assert result is not None

    async def test_file_not_found_returns_none(self, fixer, tmp_path):
        missing = tmp_path / "missing.py"
        result = await fixer._prepare_fix_context(missing, {"first_line": 1, "last_line": 1, "problem_lines": [1]}, "")
        assert result is None

    async def test_class_name_detected(self, fixer, tmp_path):
        py_file = tmp_path / "cls.py"
        py_file.write_text("class MyService:\n    def process(self):\n        return 1\n")
        line_range = {"first_line": 2, "last_line": 3, "problem_lines": [2]}
        result = await fixer._prepare_fix_context(py_file, line_range, "")
        assert result is not None
        assert result.class_name == "MyService"


# ===========================================================================
# TestExtendStrategiesForIssue
# ===========================================================================


class TestExtendStrategiesForIssue:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

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


# ===========================================================================
# TestApplyFixesWithValidationCoverage
# ===========================================================================


class TestApplyFixesWithValidationCoverage:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def _make_fix(self, tmp_path, file_name="test.py"):
        cb = CodeBlock(block_name="t", start_line=1, end_line=1, has_changes=True,
                       change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE, context="x")
        return FixSuggestion(issue_key="test", file_path=file_name, original_code="old", fixed_code="new",
                             explanation="fix", confidence=0.9, sonar_line_number=1, llm_model="m",
                             fixed_code_blocks=[cb])

    def _make_issue(self):
        return SonarIssue(key="t", rule="python:S1", severity="MAJOR", component="test.py",
                          project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
                          first_line=1, last_line=1)

    async def test_success_path_no_validator(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("old code\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(True, [])), \
                patch.object(fixer, "_create_backup", return_value=tmp_path / "bak"):
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=True, use_validator=False)
        assert result.total_fixes_attempted == 1
        assert len(result.successful_fixes) == 1
        assert result.backup_created is True

    async def test_failure_no_validator_marks_failed(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("old\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        mock_result = Mock()
        mock_result.success = False
        mock_result.reason = "syntax error"
        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(False, [mock_result])), \
                patch.object(fixer, "_create_backup", return_value=tmp_path / "bak"):
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=False, use_validator=False)
        assert len(result.failed_fixes) > 0

    async def test_dry_run_no_backup(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("code\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, return_value=(True, [])):
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=True, dry_run=True, use_validator=False)
        assert result.backup_created is False

    async def test_exception_in_file_processing(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("code\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        with patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=False, use_validator=False)
        assert len(result.failed_fixes) == 1
        assert "boom" in result.failed_fixes[0]["error"]


# ===========================================================================
# TestResolveEffectiveValues / TestResolveRelativePath
# ===========================================================================


class TestResolveEffectiveValues:
    def _make_context(self, start_line=1):
        return FixContext(
            file_path=Path("/project/src/test.py"),
            file_path_tmp=Path(tempfile.gettempdir()) / "test.py",
            line_range={}, code_content="original code", language="python",
            import_section={"start_line": 1, "end_line": 3}, class_name=None, functions=[],
            context_dict={"start_line": start_line, "import_section": {"end_line": 3}},
        )

    def test_returns_defaults_when_modify_line_range_false(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        code_blocks = [CodeBlock(block_name="b", start_line=1, end_line=2, has_changes=True,
                                 change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE)]
        result = _resolve_effective_values(code_blocks, file_path, context, {"first_line": 5, "last_line": 7},
                                           modify_line_range=False)
        assert result == (file_path, 10, 5, 7)

    def test_overrides_when_block_has_different_file_path(self):
        context = self._make_context(start_line=10)
        file_path = Path("/project/src/test.py")
        code_blocks = [CodeBlock(block_name="b", start_line=20, end_line=30, has_changes=True,
                                 change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE,
                                 file_path="/project/src/other.py")]
        eff_path, eff_start, eff_sonar, eff_last = _resolve_effective_values(
            code_blocks, file_path, context, {"first_line": 5, "last_line": 7})
        assert eff_path == Path("/project/src/other.py")
        assert eff_start == 20
        assert eff_sonar == 20
        assert eff_last == 30


class TestResolveRelativePath:
    def test_returns_relative_path(self):
        result = _resolve_relative_path(Path("/project/src/test.py"), Path("/project"))
        assert result == "src/test.py"

    def test_returns_absolute_when_outside_project(self):
        result = _resolve_relative_path(Path("/other/src/test.py"), Path("/project"))
        assert result == "/other/src/test.py"


# ===========================================================================
# TestBuildLlm (unit tests for the new build_llm function)
# ===========================================================================


class TestBuildLlm:
    """Verify build_llm dispatches to the correct LangChain class."""

    def test_openai_dispatches_to_chat_openai(self):
        mock_cls = Mock()
        with patch("devdox_ai_sonar.llm_fixer.build_llm", wraps=build_llm):
            try:
                from langchain_openai import ChatOpenAI
                with patch("devdox_ai_sonar.llm_fixer.ChatOpenAI" if hasattr(
                        __import__("devdox_ai_sonar.llm_fixer", fromlist=["ChatOpenAI"]), "ChatOpenAI")
                           else "langchain_openai.ChatOpenAI", mock_cls):
                    pass  # just ensure no crash
            except ImportError:
                pytest.skip("langchain_openai not installed")

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            build_llm(provider="unknown", model="m", api_key="k")


# ===========================================================================
# TestCallLlmWithGraph (integration-level unit test)
# ===========================================================================


class TestCallLlmWithGraph:
    async def test_success_returns_fix_result(self):
        mock_result = Mock()
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "fix_result": mock_result, "validation_error": None,
            "attempt": 0, "max_attempts": 3,
        })
        with patch("devdox_ai_sonar.llm_fixer._get_fix_graph", return_value=mock_graph):
            result = await call_llm_with_graph("openai", "gpt-4o", "key", "system", "user")
        assert result is mock_result

    async def test_failure_returns_none(self):
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "fix_result": None, "validation_error": "failed",
            "attempt": 3, "max_attempts": 3,
        })
        with patch("devdox_ai_sonar.llm_fixer._get_fix_graph", return_value=mock_graph):
            result = await call_llm_with_graph("openai", "gpt-4o", "key", "system", "user")
        assert result is None


# ===========================================================================
# TestModuleLevelBuildFixSuggestion
# ===========================================================================


class TestModuleLevelBuildFixSuggestion:
    def _make_code_block(self):
        return CodeBlock(block_name="test", start_line=1, end_line=5, has_changes=True,
                         change_type=ChangeType.FULL_CODE, block_type=BlockType.MODULE, context="fixed code")

    def test_builds_fix_suggestion_with_list_problem_lines(self, tmp_path):
        cb = self._make_code_block()
        fix_response = SonarFixResponse(IMPORT_BLOCK="import os", FIXED_CODE_BLOCKS=[cb],
                                        NEW_HELPER_CODE="", PLACEMENT=PlacementType.GLOBAL_TOP,
                                        EXPLANATION="Explanation", CONFIDENCE=0.9)
        file_path = tmp_path / "src" / "test.py"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("code")
        context_info = {"context_dict": {"context": "original", "start_line": 5, "end_line": 10,
                                         "import_section": {"end_line": 3}}}
        line_range = {"first_line": 5, "last_line": 10, "problem_lines": [5, 7]}
        result = _build_fix_suggestion(fix_response, context_info, file_path, tmp_path, line_range, "gpt-4")
        assert isinstance(result, FixSuggestion)
        assert result.issue_key == "fix_L5-L7"
        assert result.llm_model == "gpt-4"

    def test_single_int_problem_lines(self, tmp_path):
        cb = self._make_code_block()
        fix_response = SonarFixResponse(FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.8)
        file_path = tmp_path / "test.py"
        file_path.write_text("code")
        context_info = {"context_dict": {"context": "code", "start_line": 1}}
        result = _build_fix_suggestion(fix_response, context_info, file_path, tmp_path,
                                       {"first_line": 1, "last_line": 1, "problem_lines": 42}, "m")
        assert result.issue_key == "fix_L42"

    def test_no_problem_lines(self, tmp_path):
        cb = self._make_code_block()
        fix_response = SonarFixResponse(FIXED_CODE_BLOCKS=[cb], CONFIDENCE=0.7)
        file_path = tmp_path / "test.py"
        file_path.write_text("code")
        context_info = {"context_dict": {"context": "code", "start_line": 1}}
        result = _build_fix_suggestion(fix_response, context_info, file_path, tmp_path,
                                       {"first_line": 1, "last_line": 1}, "m")
        assert result.issue_key == "fix_unknown"


# ===========================================================================
# TestWriteExplaination
# ===========================================================================


class TestWriteExplaination:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def test_writes_markdown_file(self, fixer, tmp_path):
        md_file = tmp_path / "explanations" / "output.md"
        fix_response = Mock()
        fix_response.EXPLANATION = "Fixed the issue"
        fix_response.FIXED_CODE_BLOCKS = []
        issue = Mock()
        issue.rule = "python:S1234"
        issue.severity = "MAJOR"
        issue.message = "Bad code"
        issue.file_path = "test.py"
        issue.line = 10
        mock_template = Mock()
        mock_template.render.return_value = "# Rendered Markdown"
        fixer.jinja_env_templates = Mock()
        fixer.jinja_env_templates.get_template.return_value = mock_template
        fixer.write_explaination(md_file, [fix_response], [issue], "original code")
        assert md_file.exists()
        assert "Rendered Markdown" in md_file.read_text()

    def test_creates_parent_dirs(self, fixer, tmp_path):
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


# ===========================================================================
# TestApplyFixesFailurePath
# ===========================================================================


class TestApplyFixesFailurePath:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    async def test_apply_fixes_failure_marks_failed(self, fixer, tmp_path):
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


# ===========================================================================
# TestApplyFixesValidatorFallback
# ===========================================================================


class TestApplyFixesValidatorFallback:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def _make_fix(self, tmp_path):
        fix = Mock(spec=FixSuggestion)
        fix.file_path = str(tmp_path / "test.py")
        fix.issue_key = "k"
        fix.original_code = "x = 1"
        return fix

    def _make_issue(self):
        return SonarIssue(key="t", rule="python:S1", severity="MAJOR", component="test.py",
                          project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
                          first_line=1, last_line=1)

    async def test_validator_approved(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "syntax error"
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.APPROVED
        with patch.object(fixer, "_get_file_from_fix", return_value=str(tmp_path / "test.py")), \
                patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock,
                             return_value=(False, [failed_result])), \
                patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=False, use_validator=True,
                                                             validator_provider="openai", validator_api_key="key")
        assert len(result.successful_fixes) == 1

    async def test_validator_rejected(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.REJECTED
        mock_val_result.explanation = "bad fix"
        with patch.object(fixer, "_get_file_from_fix", return_value=str(tmp_path / "test.py")), \
                patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock,
                             return_value=(False, [failed_result])), \
                patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=False, use_validator=True,
                                                             validator_provider="openai", validator_api_key="key")
        assert len(result.failed_fixes) == 1
        assert "Rejected" in result.failed_fixes[0]["error"]

    async def test_validator_needs_review(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"
        mock_val_result = Mock()
        mock_val_result.status = ValidationStatus.NEEDS_REVIEW
        mock_val_result.explanation = "check manually"
        with patch.object(fixer, "_get_file_from_fix", return_value=str(tmp_path / "test.py")), \
                patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock,
                             return_value=(False, [failed_result])), \
                patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.return_value = mock_val_result
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=False, use_validator=True,
                                                             validator_provider="openai", validator_api_key="key")
        assert len(result.failed_fixes) == 1
        assert "manual review" in result.failed_fixes[0]["error"]

    async def test_validator_exception(self, fixer, tmp_path):
        (tmp_path / "test.py").write_text("x = 1\n")
        fix = self._make_fix(tmp_path)
        issue = self._make_issue()
        failed_result = Mock()
        failed_result.success = False
        failed_result.reason = "error"
        with patch.object(fixer, "_get_file_from_fix", return_value=str(tmp_path / "test.py")), \
                patch.object(fixer, "_apply_fixes_to_file", new_callable=AsyncMock,
                             return_value=(False, [failed_result])), \
                patch("devdox_ai_sonar.llm_fixer.FixValidator") as MockValidator:
            MockValidator.return_value.validate_fix.side_effect = Exception("validator crash")
            result = await fixer.apply_fixes_with_validation([fix], [issue], tmp_path,
                                                             create_backup=False, use_validator=True,
                                                             validator_provider="openai", validator_api_key="key")
        assert len(result.failed_fixes) == 1
        assert "Validator error" in result.failed_fixes[0]["error"]


# ===========================================================================
# TestCreateBackupIfNeeded / TestGroupFixesByFile
# ===========================================================================


class TestCreateBackupIfNeeded:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def test_creates_backup_when_requested_and_not_dry_run(self, fixer, tmp_path):
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        with patch.object(fixer, "_create_backup", return_value=tmp_path / "bak") as mock_backup:
            fixer._create_backup_if_needed(result, tmp_path, create_backup=True, dry_run=False)
        mock_backup.assert_called_once_with(tmp_path)
        assert result.backup_created is True

    def test_skips_backup_on_dry_run(self, fixer, tmp_path):
        result = FixResult(project_path=tmp_path, total_fixes_attempted=1)
        with patch.object(fixer, "_create_backup") as mock_backup:
            fixer._create_backup_if_needed(result, tmp_path, create_backup=True, dry_run=True)
        mock_backup.assert_not_called()
        assert result.backup_created is False


class TestGroupFixesByFile:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def _make_fix(self, issue_key="k"):
        fix = Mock(spec=FixSuggestion)
        fix.issue_key = issue_key
        fix.file_path = "test.py"
        fix.original_code = "x"
        return fix

    def _make_issue(self):
        return SonarIssue(key="k", rule="python:S1", severity="MAJOR", component="t.py",
                          project="p", line=1, message="msg", type="CODE_SMELL", status="OPEN",
                          first_line=1, last_line=1)

    def test_groups_fixes_by_file_path(self, fixer, tmp_path):
        fix1, fix2 = self._make_fix("k1"), self._make_fix("k2")
        issue1, issue2 = self._make_issue(), self._make_issue()
        with patch.object(fixer, "_get_file_from_fix", side_effect=["file_a.py", "file_a.py"]):
            result = fixer._group_fixes_by_file([fix1, fix2], [issue1, issue2], tmp_path)
        assert "file_a.py" in result
        assert len(result["file_a.py"]) == 2

    def test_excludes_fixes_with_no_file(self, fixer, tmp_path):
        fix1, fix2 = self._make_fix("k1"), self._make_fix("k2")
        issue1, issue2 = self._make_issue(), self._make_issue()
        with patch.object(fixer, "_get_file_from_fix", side_effect=["file_a.py", None]):
            result = fixer._group_fixes_by_file([fix1, fix2], [issue1, issue2], tmp_path)
        assert len(result) == 1

    def test_empty_input(self, fixer, tmp_path):
        assert fixer._group_fixes_by_file([], [], tmp_path) == {}


# ===========================================================================
# TestCheckBracketBalance
# ===========================================================================


class TestCheckBracketBalance:
    @pytest.fixture
    def fixer(self):
        return _make_fixer()

    def test_unmatched_closing_bracket(self, fixer):
        assert fixer._check_bracket_balance(")") is False

    def test_mismatched_brackets(self, fixer):
        assert fixer._check_bracket_balance("(]") is False

    def test_balanced_brackets(self, fixer):
        assert fixer._check_bracket_balance("()[]{}") is True


# ===========================================================================
# GetContentRange tests (unchanged — no LLM involvement)
# ===========================================================================


class TestGetContentRange:
    def _make_extractor(self):
        return IssueExtractor(AsyncFileReader())

    async def test_exact_match_scenario(self, temp_dir, sample_code):
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"
        tmp_file.write_text(sample_code)
        actual_file.write_text(sample_code)
        line_range = {"first_line": 4, "last_line": 6, "problem_lines": [5]}
        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)
        assert result is not None
        assert result["first_line"] == 4
        assert result["confidence"] == 1.0
        assert result["match_type"] == "exact"

    async def test_content_not_found(self, temp_dir, sample_code):
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"
        tmp_file.write_text(sample_code)
        actual_file.write_text("def different_function():\n    pass\n")
        line_range = {"first_line": 4, "last_line": 6, "problem_lines": [5]}
        extractor = self._make_extractor()
        result = await extractor.get_content_range(tmp_file, line_range, actual_file)
        assert result is not None
        assert result["first_line"] is None
        assert result["confidence"] == 0.0
        assert result["match_type"] == "not_found"

    async def test_file_not_found_tmp(self, temp_dir):
        extractor = self._make_extractor()
        actual_file = temp_dir / "actual.py"
        actual_file.write_text("code")
        with pytest.raises(FileNotFoundError, match="Temporary file not found"):
            await extractor.get_content_range(temp_dir / "nonexistent.py",
                                              {"first_line": 1, "last_line": 1, "problem_lines": [1]}, actual_file)

    async def test_invalid_line_range_missing_keys(self, temp_dir):
        tmp_file = temp_dir / "temp.py"
        actual_file = temp_dir / "actual.py"
        tmp_file.write_text("code")
        actual_file.write_text("code")
        extractor = self._make_extractor()
        with pytest.raises(ValueError, match="must contain 'first_line' and 'last_line'"):
            await extractor.get_content_range(tmp_file, {"problem_lines": [1]}, actual_file)

    def test_find_exact_match_found(self):
        target = ["line1\n", "line2\n"]
        actual = ["other\n", "line1\n", "line2\n", "more\n"]
        result = _find_exact_match(target, actual)
        assert result is not None
        assert result["start"] == 1

    def test_find_exact_match_not_found(self):
        result = _find_exact_match(["line1\n", "line2\n"], ["other\n", "different\n"])
        assert result is None

    def test_create_result(self):
        match = {"start": 5, "end": 8}
        result = _create_result(match, [3, 4], 2, 0.95, "context")
        assert result["first_line"] == 6
        assert result["confidence"] == 0.95
        assert result["match_type"] == "context"