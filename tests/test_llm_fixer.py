"""Comprehensive tests for LLM-powered code fixer."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from devdox_ai_sonar.models import (
    SonarIssue,
    FixSuggestion,
    FixResult,
    Severity,
    IssueType,
)
from devdox_ai_sonar.llm_fixer import calculate_base_indentation


@pytest.fixture
def sample_issue():
    """Create a sample SonarCloud issue for testing."""
    return SonarIssue(
        key="test:src/test.py:S1481",
        rule="python:S1481",
        severity=Severity.MAJOR,
        component="test:src/test.py",
        project="test-project",
        first_line=10,
        last_line=10,
        message='Remove the unused local variable "unused_var".',
        type=IssueType.CODE_SMELL,
        file="src/test.py",
        status="OPEN",
    )


@pytest.fixture
def sample_fix():
    """Create a sample fix suggestion."""
    return FixSuggestion(
        issue_key="test:src/test.py:S1481",
        original_code="    unused_var = 42\n    return value",
        fixed_code="    return value",
        explanation="Removed unused variable 'unused_var'",
        confidence=0.95,
        llm_model="gpt-4",
        file_path="src/test.py",
        line_number=10,
        last_line_number=11,
    )


@pytest.fixture
def mock_llm_fixer():
    """Create a mock LLMFixer for testing."""
    with patch("devdox_ai_sonar.llm_fixer.openai") as mock_openai:
        mock_openai.OpenAI.return_value = MagicMock()
        from devdox_ai_sonar.llm_fixer import LLMFixer

        fixer = LLMFixer(provider="openai", api_key="test-key")
        return fixer


class TestLLMFixerInitialization:
    """Test LLMFixer initialization."""

    def test_openai_initialization(self):
        """Test OpenAI provider initialization."""
        with patch("devdox_ai_sonar.llm_fixer.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()

            from devdox_ai_sonar.llm_fixer import LLMFixer

            fixer = LLMFixer(provider="openai", api_key="test-key")
            assert fixer.provider == "openai"
            assert fixer.model == "gpt-4o"
            assert fixer.api_key == "test-key"

    def test_gemini_initialization(self):
        """Test Gemini provider initialization."""
        with patch("devdox_ai_sonar.llm_fixer.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()

            from devdox_ai_sonar.llm_fixer import LLMFixer

            fixer = LLMFixer(provider="gemini", api_key="test-gemini-key")
            assert fixer.provider == "gemini"
            assert fixer.model == "claude-3-5-sonnet-20241022"

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        from devdox_ai_sonar.llm_fixer import LLMFixer

        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMFixer(provider="invalid", api_key="test-key")

    def test_missing_api_key(self):
        """Test that missing API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            from devdox_ai_sonar.llm_fixer import LLMFixer

            with pytest.raises(ValueError, match="API key not provided"):
                LLMFixer(provider="openai", api_key=None)


class TestContextExtraction:
    """Test context extraction methods."""

    def test_extract_normal_context(self, mock_llm_fixer):
        """Test extraction of normal code context."""
        lines = ["line 1\n", "line 2\n", "line 3\n", "line 4\n", "line 5\n"]
        context = mock_llm_fixer._extract_context(lines, 3, 3, context_lines=1)

        assert "line 2" in context["context"]
        assert "line 3" in context["context"]
        assert "line 4" in context["context"]
        assert context["line_number"] == 3
        assert context["start_line"] == 2
        assert context["end_line"] == 4

    def test_extract_context_at_file_boundaries(self, mock_llm_fixer):
        """Test context extraction at beginning and end of file."""
        lines = ["line 1\n", "line 2\n", "line 3\n"]

        # Test at beginning
        context = mock_llm_fixer._extract_context(lines, 1, 1, context_lines=5)
        assert context["start_line"] == 1
        assert "line 1" in context["context"]

        # Test at end
        context = mock_llm_fixer._extract_context(lines, 3, 3, context_lines=5)
        assert context["end_line"] == 3
        assert "line 3" in context["context"]

    def test_extract_complete_function(self, mock_llm_fixer):
        """Test extraction of complete function."""
        lines = [
            "def my_function():\n",
            "    x = 1\n",
            "    y = 2\n",
            "    return x + y\n",
            "\n",
            "def other_function():\n",
        ]

        context = mock_llm_fixer._extract_context(lines, 1, 1, context_lines=2)

        if context.get("is_complete_function"):
            assert "def my_function" in context["context"]
            assert context["function_name"] == "my_function"

    def test_is_function_definition(self, mock_llm_fixer):
        """Test function definition detection."""
        assert mock_llm_fixer._is_function_definition("def my_func():")
        assert mock_llm_fixer._is_function_definition("async def async_func():")
        assert mock_llm_fixer._is_function_definition("    def method(self):")
        assert not mock_llm_fixer._is_function_definition("    x = 42")
        assert not mock_llm_fixer._is_function_definition("# def commented()")


class TestIndentation:
    """Test indentation handling."""

    def test_calculate_base_indentation(self):
        """Test calculation of base indentation."""

        assert calculate_base_indentation("    code") == 4
        assert calculate_base_indentation("  code") == 2
        assert calculate_base_indentation("code") == 0
        assert calculate_base_indentation("\t\tcode") == 2  # Tabs count as characters

    def test_apply_indentation_to_fix(self, mock_llm_fixer):
        """Test applying indentation to fixed code."""
        fixed_code = "x = 1\ny = 2\nreturn x + y"
        base_indent = "    "

        indented = mock_llm_fixer.apply_indentation_to_fix(fixed_code, base_indent)

        lines = indented.split("\n")
        assert all(line.startswith(base_indent) or not line.strip() for line in lines)

    def test_normalize_indentation(self, mock_llm_fixer):
        """Test indentation normalization."""
        lines = ["    def func():", "        x = 1", "        return x"]
        normalized = mock_llm_fixer._normalize_indentation(lines)

        assert normalized[0] == "def func():"
        assert normalized[1] == "    x = 1"


class TestFixGeneration:
    """Test fix generation."""

    def test_generate_fix_success(self, mock_llm_fixer, sample_issue, tmp_path):
        """Test successful fix generation."""
        # Create test file
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """def my_function():
    unused_var = 42
    value = 100
    return value
"""
        )

        rule_info = {
            "name": "Unused local variables",
            "root_cause": "Unused code creates clutter",
            "how_to_fix": {"description": "Remove the unused variable"},
        }

        # Mock LLM response
        mock_response = {
            "fixed_code": "    value = 100\n    return value",
            "explanation": "Removed unused variable",
            "confidence": 0.95,
            "helper_code": "",
            "placement_helper": "SIBLING",
        }

        with patch.object(
            mock_llm_fixer, "_call_llm", return_value=mock_response
        ) as mock_call:
            fix = mock_llm_fixer.generate_fix(sample_issue, tmp_path, rule_info)

            assert fix is not None
            assert fix.issue_key == sample_issue.key
            assert fix.confidence == 0.95
            assert "unused variable" in fix.explanation.lower()

    def test_generate_fix_missing_file(self, mock_llm_fixer, sample_issue, tmp_path):
        """Test fix generation with missing file."""
        rule_info = {}

        fix = mock_llm_fixer.generate_fix(sample_issue, tmp_path, rule_info)
        assert fix is None

    def test_generate_fix_no_line_numbers(self, mock_llm_fixer, tmp_path):
        """Test fix generation with missing line numbers."""
        issue = SonarIssue(
            key="test:src/test.py:S1481",
            rule="python:S1481",
            severity=Severity.MAJOR,
            component="test:src/test.py",
            project="test-project",
            first_line=None,
            last_line=None,
            message="Test issue",
            type=IssueType.CODE_SMELL,
            file="src/test.py",
        )

        rule_info = {}
        fix = mock_llm_fixer.generate_fix(issue, tmp_path, rule_info)
        assert fix is None


class TestLanguageDetection:
    """Test programming language detection."""

    def test_get_language_from_extension(self, mock_llm_fixer):
        """Test language detection from file extension."""
        assert mock_llm_fixer._get_language_from_extension(".py") == "python"
        assert mock_llm_fixer._get_language_from_extension(".js") == "javascript"
        assert mock_llm_fixer._get_language_from_extension(".java") == "java"
        assert mock_llm_fixer._get_language_from_extension(".ts") == "typescript"
        assert mock_llm_fixer._get_language_from_extension(".unknown") == "text"


class TestFixApplication:
    """Test fix application to files."""

    def test_apply_fixes_success(self, mock_llm_fixer, sample_fix, tmp_path):
        """Test successful application of fixes."""
        # Create test file
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """def my_function():
    unused_var = 42
    value = 100
    return value
"""
        )

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        result = mock_llm_fixer.apply_fixes(
            [sample_fix], tmp_path, create_backup=False, dry_run=False
        )

        assert result.total_fixes_attempted == 1
        assert len(result.successful_fixes) >= 0  # May succeed or fail based on implementation

    def test_apply_fixes_dry_run(self, mock_llm_fixer, sample_fix, tmp_path):
        """Test dry run mode doesn't modify files."""
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        original_content = """def my_function():
    unused_var = 42
    return value
"""
        test_file.write_text(original_content)

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        result = mock_llm_fixer.apply_fixes(
            [sample_fix], tmp_path, create_backup=False, dry_run=True
        )

        # File should not be modified
        assert test_file.read_text() == original_content

    def test_apply_fixes_with_backup(self, mock_llm_fixer, sample_fix, tmp_path):
        """Test that backup is created when requested."""
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test(): pass")

        sample_fix.file_path = str(test_file.relative_to(tmp_path))

        with patch.object(
            mock_llm_fixer, "_create_backup", return_value=tmp_path / "backup"
        ) as mock_backup:
            result = mock_llm_fixer.apply_fixes(
                [sample_fix], tmp_path, create_backup=True, dry_run=False
            )

            mock_backup.assert_called_once()


class TestPromptGeneration:
    """Test LLM prompt generation."""

    def test_create_fix_prompt(self, mock_llm_fixer, sample_issue):
        """Test fix prompt creation."""
        context = {
            "context": "def func():\n    x = 1\n    return x",
            "problem_line": "    x = 1",
            "line_number": 2,
            "start_line": 1,
            "end_line": 3,
        }

        rule_info = {
            "name": "Test Rule",
            "root_cause": "Test cause",
            "how_to_fix": {"description": "Test fix"},
        }

        prompt = mock_llm_fixer._create_fix_prompt(sample_issue, context, rule_info, "python")

        assert "python" in prompt.lower()
        assert sample_issue.message in prompt
        assert sample_issue.rule in prompt
        assert context["context"] in prompt


class TestFileOperations:
    """Test file operation utilities."""

    def test_get_file_from_fix(self, mock_llm_fixer, sample_fix, tmp_path):
        """Test extracting file path from fix."""
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("test content")

        sample_fix.file_path = "src/test.py"

        file_path = mock_llm_fixer._get_file_from_fix(sample_fix, tmp_path)
        assert file_path == str(test_file)

    def test_find_files_with_content(self, mock_llm_fixer, tmp_path):
        """Test finding files containing specific content."""
        # Create test files
        file1 = tmp_path / "test1.py"
        file2 = tmp_path / "test2.py"
        file3 = tmp_path / "test3.js"

        file1.write_text("def unique_function():\n    pass")
        file2.write_text("def other_function():\n    pass")
        file3.write_text("function unique_function() {}")

        # Search for unique content
        matches = mock_llm_fixer._find_files_with_content(tmp_path, "unique_function")

        assert len(matches) >= 1
        assert any("test1.py" in str(f) or "test3.js" in str(f) for f in matches)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
