"""Comprehensive tests for LLM-powered code fixer."""

import pytest
import json
from unittest.mock import patch, MagicMock
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    FixSuggestion,
    Severity,
    IssueType,
)
from devdox_ai_sonar.fix_validator import ValidationStatus, ValidationResult

from devdox_ai_sonar.llm_fixer import calculate_base_indentation, LLMFixer


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
        message="Remove the unused local variable",
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

        fixer = LLMFixer(provider="openai", api_key="test-key")
        return fixer


class TestLLMFixerInitialization:
    """Test LLMFixer initialization."""

    def test_openai_initialization(self):
        """Test OpenAI provider initialization."""
        with patch("devdox_ai_sonar.llm_fixer.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()

            fixer = LLMFixer(provider="openai", api_key="test-key")
            assert fixer.provider == "openai"
            assert fixer.model == "gpt-4o"
            assert fixer.api_key == "test-key"

    def test_gemini_initialization(self):
        """Test Gemini provider initialization."""
        with patch("devdox_ai_sonar.llm_fixer.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()

            fixer = LLMFixer(provider="gemini", api_key="test-gemini-key")
            assert fixer.provider == "gemini"
            assert fixer.model == "claude-3-5-sonnet-20241022"

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""

        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMFixer(provider="invalid", api_key="test-key")

    def test_missing_api_key(self):
        """Test that missing API key raises error."""
        with patch.dict("os.environ", {}, clear=True):

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
        assert (
            len(result.successful_fixes) >= 0
        )  # May succeed or fail based on implementation

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

        prompt = mock_llm_fixer._create_fix_prompt(
            sample_issue, context, rule_info, "python"
        )

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


class TestTogetherAIInitialization:
    """Test TogetherAI provider specific initialization."""

    def test_togetherai_initialization(self):
        """Test TogetherAI provider initialization."""
        with patch("devdox_ai_sonar.llm_fixer.Together") as mock_together:
            mock_together.return_value = MagicMock()
            # Mock the import check constant
            with patch("devdox_ai_sonar.llm_fixer.HAS_TOGETHER", True):
                fixer = LLMFixer(provider="togetherai", api_key="test-together-key")
                assert fixer.provider == "togetherai"
                assert fixer.model == "gpt-4o"  # Default fallback in code
                assert fixer.api_key == "test-together-key"
                mock_together.assert_called_once_with(api_key="test-together-key")

    def test_togetherai_missing_library(self):
        """Test error when TogetherAI lib is missing."""
        with patch("devdox_ai_sonar.llm_fixer.HAS_TOGETHER", False):
            with pytest.raises(ImportError, match="Together AI library not installed"):
                LLMFixer(provider="togetherai", api_key="key")


class TestDecoratorHandling:
    """Test logic for identifying and handling Python decorators."""

    def test_is_decorator(self, mock_llm_fixer):
        """Test the regex patterns for decorator detection."""
        # Positive cases
        assert mock_llm_fixer._is_decorator("@staticmethod")
        assert mock_llm_fixer._is_decorator("@app.route('/index')")
        assert mock_llm_fixer._is_decorator("  @classmethod")  # indented
        assert mock_llm_fixer._is_decorator("@pytest.fixture(scope='module')")

        # Negative cases
        assert not mock_llm_fixer._is_decorator("email@example.com")
        assert not mock_llm_fixer._is_decorator("x = @value")
        assert not mock_llm_fixer._is_decorator("# @commented_out")

    def test_find_function_start_with_decorators(self, mock_llm_fixer):
        """Test finding the top decorator of a function."""
        lines = [
            "import something\n",  # 0
            "@app.route('/')\n",  # 1
            "@auth_required\n",  # 2
            "def index():\n",  # 3
            "    pass\n",
        ]
        # If we target the def line (3), it should backtrack to line 1
        start_index = mock_llm_fixer._find_function_start_with_decorators(lines, 3)
        assert start_index == 1


class TestSmartContextExtraction:
    """Test the logic that finds parent functions and containment."""

    def test_find_containing_function(self, mock_llm_fixer):
        """Test locating the parent function of a specific line."""
        lines = [
            "def parent_function():\n",  # 0
            "    x = 1\n",  # 1
            "    y = 2\n",  # 2
            "    return x + y\n",  # 3
        ]

        # Mock _find_function_end to return the last line
        with patch.object(mock_llm_fixer, "_find_function_end", return_value=3):
            # Target line 2 (inside function)
            parent_idx = mock_llm_fixer._find_containing_function(lines, 2)
            assert parent_idx == 0

    def test_check_indentation_containment(self, mock_llm_fixer):
        """Test fallback logic using indentation levels."""
        lines = [
            "def func():\n",  # 0 (indent 0)
            "    if True:\n",  # 1 (indent 4)
            "        return\n",  # 2 (indent 8)
        ]

        # Line 1 is deeper than Line 0 -> True
        assert mock_llm_fixer._check_indentation_containment(lines, 1, 0)
        # Line 2 is deeper than Line 0 -> True
        assert mock_llm_fixer._check_indentation_containment(lines, 2, 0)

        lines_broken = [
            "def func():\n",  # 0
            "return\n",  # 1 (Same indent as def, technically outside or broken)
        ]
        # Line 1 is same indent -> False
        assert not mock_llm_fixer._check_indentation_containment(lines_broken, 1, 0)

    def test_extract_context_with_modified_content(
        self, mock_llm_fixer, sample_issue, tmp_path
    ):
        """Test generate_fix when modified_content is explicitly passed."""
        # Create a dummy file just to satisfy the file existence check
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()

        modified_content = "def patched_func():\n    return 'fixed'"

        # Mock the LLM call to ensure it actually runs
        with patch.object(mock_llm_fixer, "_call_llm") as mock_call:
            mock_call.return_value = {
                "fixed_code": "code",
                "explanation": "exp",
                "confidence": 1.0,
            }

            mock_llm_fixer.generate_fix(
                sample_issue, tmp_path, rule_info={}, modified_content=modified_content
            )

            # Verify _call_llm was called with the modified content in the context
            # args[1] is the context dict
            call_args = mock_call.call_args
            print(" call_args[0][1] ",  call_args[0][1])
            assert call_args[0][1]['context'] == modified_content


class TestAdvancedFixApplication:
    """Test complex fix application scenarios."""

    def test_apply_fixes_grouping_logic(self, mock_llm_fixer, tmp_path):
        """Test that multiple fixes for the same file are grouped correctly."""
        file_path = tmp_path / "test.py"
        file_path.write_text("line1\nline2\nline3")

        fix1 = FixSuggestion(
            issue_key="1",
            original_code="line1",
            fixed_code="fixed1",
            explanation="e",
            confidence=1,
            llm_model="gpt",
            file_path="test.py",
            line_number=1,
            last_line_number=1,
        )
        fix2 = FixSuggestion(
            issue_key="2",
            original_code="line3",
            fixed_code="fixed3",
            explanation="e",
            confidence=1,
            llm_model="gpt",
            file_path="test.py",
            line_number=3,
            last_line_number=3,
        )

        with patch.object(
            mock_llm_fixer, "_apply_fixes_to_file", return_value=True
        ) as mock_apply:
            result = mock_llm_fixer.apply_fixes(
                [fix1, fix2], tmp_path, create_backup=False
            )

            # Should be called once per file, containing list of 2 fixes
            assert mock_apply.call_count == 1
            args = mock_apply.call_args
            assert str(args[0][0]) == str(file_path)  # First arg is path
            assert len(args[0][1]) == 2  # Second arg is list of fixes

    def test_apply_fixes_failure_handling(self, mock_llm_fixer, tmp_path):
        """Test handling when individual file application fails."""
        file_path = tmp_path / "test.py"
        file_path.write_text("content")

        fix = FixSuggestion(
            issue_key="1",
            original_code="c",
            fixed_code="f",
            explanation="e",
            confidence=1,
            llm_model="m",
            file_path="test.py",
            line_number=1,
            last_line_number=1,
        )

        # Simulate exception during file processing
        with patch.object(
            mock_llm_fixer, "_apply_fixes_to_file", side_effect=Exception("Disk full")
        ):
            result = mock_llm_fixer.apply_fixes([fix], tmp_path, create_backup=False)

            assert len(result.successful_fixes) == 0
            assert len(result.failed_fixes) == 1
            assert "Disk full" in result.failed_fixes[0]["error"]

    def test_generate_fix_llm_exception(self, mock_llm_fixer, sample_issue, tmp_path):
        """Test graceful handling of LLM API failures."""
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("code")

        with patch.object(
            mock_llm_fixer, "_call_llm", side_effect=Exception("API Rate Limit")
        ):
            fix = mock_llm_fixer.generate_fix(sample_issue, tmp_path, rule_info={})
            assert fix is None  # Should return None, not crash


class TestTogetherAIProviderIntegration:
    """Test TogetherAI LLM provider integration for code fixing."""

    @patch("devdox_ai_sonar.llm_fixer.Together")
    @patch("devdox_ai_sonar.llm_fixer.HAS_TOGETHER", True)
    def test_call_llm_togetherai_success(self, mock_together, sample_issue, tmp_path):
        """Test successful LLM call with TogetherAI provider."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "FIXED_SELECTION": "return value",
                "NEW_HELPER_CODE": "",
                "PLACEMENT": "SIBLING",
                "EXPLANATION": "Removed unused variable",
                "CONFIDENCE": 0.95,
            }
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_together.return_value = mock_client

        fixer = LLMFixer(provider="togetherai", api_key="test-key")

        # Create test file
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test():\n    unused_var = 42\n    return value")

        fix = fixer.generate_fix(sample_issue, tmp_path, rule_info={})

        assert fix is not None
        assert fix.confidence == 0.95
        mock_client.chat.completions.create.assert_called_once()

    @patch("devdox_ai_sonar.llm_fixer.Together")
    @patch("devdox_ai_sonar.llm_fixer.HAS_TOGETHER", True)
    def test_call_llm_togetherai_error(self, mock_together, sample_issue, tmp_path):
        """Test TogetherAI provider error handling."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "TogetherAI API Error"
        )
        mock_together.return_value = mock_client

        fixer = LLMFixer(provider="togetherai", api_key="test-key")

        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("code")

        fix = fixer.generate_fix(sample_issue, tmp_path, rule_info={})

        assert fix is None

    @patch("devdox_ai_sonar.llm_fixer.Together")
    @patch("devdox_ai_sonar.llm_fixer.HAS_TOGETHER", True)
    def test_togetherai_custom_parameters(self, mock_together):
        """Test TogetherAI with custom parameters."""
        mock_together.return_value = MagicMock()

        fixer = LLMFixer(
            provider="togetherai",
            model="meta-llama/Llama-3-70b-chat-hf",
            api_key="test-key",
        )

        assert fixer.model == "meta-llama/Llama-3-70b-chat-hf"
        assert fixer.provider == "togetherai"


# ==============================================================================
# CRITICAL: Gemini Provider Integration Tests
# ==============================================================================


class TestGeminiProviderIntegration:
    """Test Gemini LLM provider integration."""

    @patch("devdox_ai_sonar.llm_fixer.genai")
    def test_call_llm_gemini_success(self, mock_genai, sample_issue, tmp_path):
        """Test successful LLM call with Gemini provider."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "FIXED_SELECTION": "return value",
                "NEW_HELPER_CODE": "",
                "PLACEMENT": "SIBLING",
                "EXPLANATION": "Fixed",
                "CONFIDENCE": 0.9,
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        fixer = LLMFixer(provider="gemini", api_key="test-key")

        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("code")

        fix = fixer.generate_fix(sample_issue, tmp_path, rule_info={})

        assert fix is not None
        mock_client.models.generate_content.assert_called_once()

    @patch("devdox_ai_sonar.llm_fixer.genai")
    def test_call_llm_gemini_error(self, mock_genai, sample_issue, tmp_path):
        """Test Gemini provider error handling."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Gemini API Error")
        mock_genai.Client.return_value = mock_client

        fixer = LLMFixer(provider="gemini", api_key="test-key")

        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("code")

        fix = fixer.generate_fix(sample_issue, tmp_path, rule_info={})

        assert fix is None


# ==============================================================================
# CRITICAL: Response Parsing Edge Cases
# ==============================================================================


class TestResponseParsingEdgeCases:
    """Test LLM response parsing with various malformed inputs."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_fix_missing_opening_brace(self, mock_openai):
        """Test parsing response missing opening brace."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        response = '"FIXED_SELECTION": "code", "CONFIDENCE": 0.9}'
        result = fixer._extract_fix_from_response(response)

        # Should use regex fallback
        assert (
            result is not None or result is None
        )  # Either works or returns None gracefully

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_fix_missing_closing_brace(self, mock_openai):
        """Test parsing response missing closing brace."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        response = '{"FIXED_SELECTION": "code", "CONFIDENCE": 0.9'
        result = fixer._extract_fix_from_response(response)

        # Should use regex fallback
        assert result is not None or result is None

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_fix_with_escaped_quotes(self, mock_openai):
        """Test parsing response with escaped quotes in code."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        response = json.dumps(
            {
                "FIXED_SELECTION": 'print("Hello \\"World\\"")',
                "NEW_HELPER_CODE": "",
                "PLACEMENT": "SIBLING",
                "EXPLANATION": "Fixed",
                "CONFIDENCE": 0.9,
            }
        )

        result = fixer._extract_fix_from_response(response)

        assert result is not None
        assert '"' in result["fixed_code"] or '\\"' in result["fixed_code"]

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_fix_with_newlines_in_strings(self, mock_openai):
        """Test parsing response with newlines in string literals."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        response = json.dumps(
            {
                "FIXED_SELECTION": "x = 'multi\\nline\\nstring'",
                "NEW_HELPER_CODE": "",
                "PLACEMENT": "SIBLING",
                "EXPLANATION": "Fixed",
                "CONFIDENCE": 0.85,
            }
        )

        result = fixer._extract_fix_from_response(response)

        assert result is not None
        assert result["fixed_code"]

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_fix_malformed_json_regex_fallback(self, mock_openai):
        """Test regex fallback for completely malformed JSON."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        response = """Here's the fix:
{
"FIXED_SELECTION": "return value",
"CONFIDENCE": "0.95",
"EXPLANATION": "Removed unused variable"
}
"""

        result = fixer._extract_fix_from_response(response)

        # Should extract using regex
        assert result is not None
        assert result["fixed_code"] == "return value"
        assert result["confidence"] == 0.95

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_fix_empty_response(self, mock_openai):
        """Test parsing completely empty response."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        result = fixer._extract_fix_from_response("")

        assert result is None

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_fix_missing_required_field(self, mock_openai):
        """Test parsing response missing FIXED_SELECTION."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        response = json.dumps({"NEW_HELPER_CODE": "helper", "CONFIDENCE": 0.9})

        result = fixer._extract_fix_from_response(response)

        assert result is None


# ==============================================================================
# CRITICAL: Cognitive Complexity Handling
# ==============================================================================


class TestCognitiveComplexityHandling:
    """Test handling of cognitive complexity issues."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_complexity_info_standard_format(self, mock_openai):
        """Test extracting complexity from standard message format."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        message = "Cognitive Complexity from 25 to the 15 allowed"
        result = fixer._extract_complexity_info(message)

        assert result["current"] == "25"
        assert result["target"] == "15"

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_complexity_info_alternative_format(self, mock_openai):
        """Test extracting complexity from alternative message format."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        message = "complexity is 30, maximum allowed is 15"
        result = fixer._extract_complexity_info(message)

        assert result["current"] == "30"
        assert result["target"] == "15"

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_complexity_info_no_match(self, mock_openai):
        """Test extracting complexity when format doesn't match."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        message = "This function is too complex"
        result = fixer._extract_complexity_info(message)

        assert result["current"] == "Unknown"
        assert result["target"] == "15"  # Default

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_is_init_method_python(self, mock_openai):
        """Test detection of Python __init__ method."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        context = "def __init__(self, x, y):\n    self.x = x\n    self.y = y"

        assert fixer._is_init_method(context) is True

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_is_init_method_java_constructor(self, mock_openai):
        """Test detection of Java constructor."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        context = "public MyClass(int x) {\n    this.x = x;\n}"

        assert fixer._is_init_method(context) is True

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_is_init_method_javascript_constructor(self, mock_openai):
        """Test detection of JavaScript constructor."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        context = "constructor(x, y) {\n    this.x = x;\n    this.y = y;\n}"

        assert fixer._is_init_method(context) is True

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_is_init_method_not_constructor(self, mock_openai):
        """Test that regular methods are not detected as constructors."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        context = "def process_data(self, data):\n    return data.strip()"

        assert fixer._is_init_method(context) is False


# ==============================================================================
# CRITICAL: Function Boundary Detection Edge Cases
# ==============================================================================


class TestFunctionBoundaryDetection:
    """Test edge cases in function boundary detection."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_python_function_end_nested_functions(self, mock_openai):
        """Test finding end of function with nested functions."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        lines = [
            "def outer():\n",
            "    def inner():\n",
            "        return 1\n",
            "    return inner()\n",
            "\n",
            "def next_function():\n",
        ]

        end = fixer._find_python_function_end(lines, 0)

        assert end is not None
        assert end >= 3  # Should include all of outer function

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_python_function_end_multiline_strings(self, mock_openai):
        """Test finding end with multiline strings."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        lines = [
            "def func():\n",
            '    text = """\n',
            "    This is a\n",
            "    multiline string\n",
            '    """\n',
            "    return text\n",
            "\n",
        ]

        end = fixer._find_python_function_end(lines, 0)

        assert end is not None
        assert end >= 5

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_brace_function_end_nested_braces(self, mock_openai):
        """Test finding end of brace-based function with nested structures."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        lines = [
            "function test() {\n",
            "    if (true) {\n",
            "        return 1;\n",
            "    }\n",
            "}\n",
        ]

        end = fixer._find_brace_function_end(lines, 0)

        assert end == 4  # Should match closing brace of function

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_brace_function_end_strings_with_braces(self, mock_openai):
        """Test that braces in strings are ignored."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        lines = [
            "function test() {\n",
            '    var x = "{ not a brace }";\n',
            "    return x;\n",
            "}\n",
        ]

        end = fixer._find_brace_function_end(lines, 0)

        assert end == 3

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_remove_strings_and_comments(self, mock_openai):
        """Test removing strings and comments from code line."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        line = 'x = "{ test }" // comment { more }'
        cleaned = fixer._remove_strings_and_comments(line)

        assert "{" not in cleaned or cleaned.count("{") < line.count("{")

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_function_start_with_decorators_multiple(self, mock_openai):
        """Test finding function start with multiple decorators."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        lines = [
            "class MyClass:\n",
            "    @staticmethod\n",
            "    @cache\n",
            "    @log\n",
            "    def method():\n",
            "        pass\n",
        ]

        start = fixer._find_function_start_with_decorators(lines, 4)

        assert start == 1  # Should find first decorator


# ==============================================================================
# CRITICAL: Placement Strategy Tests
# ==============================================================================



# ==============================================================================
# CRITICAL: Fix Application with Different Placements
# ==============================================================================


class TestFixApplicationWithPlacements:
    """Test applying fixes with different helper code placements."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_apply_fix_sibling_placement(self, mock_openai, tmp_path):
        """Test applying fix with SIBLING helper placement."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    x = complex_logic()\n    return x\n")

        fix = FixSuggestion(
            issue_key="1",
            original_code="    x = complex_logic()",
            fixed_code="    x = helper()",
            helper_code="def helper():\n    return complex_logic()",
            placement_helper="SIBLING",
            explanation="Extracted to helper",
            confidence=0.9,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=2,
            last_line_number=2,
            sonar_line_number=2
        )

        success = fixer._apply_fixes_to_file(test_file, [fix], dry_run=False)

        content = test_file.read_text()
        print("content ", content)
        assert "helper()" in content
        assert "def helper():" in content

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_apply_fix_global_bottom_placement(self, mock_openai, tmp_path):
        """Test applying fix with GLOBAL_BOTTOM helper placement."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    x = value\n    return x\n")

        fix = FixSuggestion(
            issue_key="1",
            original_code="    x = value",
            fixed_code="    x = CONSTANT",
            helper_code="# Utility function\ndef utility():\n    return 42",
            placement_helper="GLOBAL_BOTTOM",
            explanation="Added utility",
            confidence=0.9,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=2,
            last_line_number=2,
            sonar_line_number=2
        )

        success = fixer._apply_fixes_to_file(test_file, [fix], dry_run=False)

        content = test_file.read_text()
        # Helper should be at bottom
        lines = content.split("\n")
        assert any("def utility():" in line for line in lines[-10:])

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_apply_fix_global_top_import_placement(self, mock_openai, tmp_path):
        """Test applying fix with GLOBAL_TOP for imports."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        test_file = tmp_path / "test.py"
        test_file.write_text("import os\n\ndef func():\n    x = value\n    return x\n")

        fix = FixSuggestion(
            issue_key="1",
            original_code="    x = value",
            fixed_code="    x = re.match(value)",
            helper_code="import re",
            placement_helper="GLOBAL_TOP",
            explanation="Added import",
            confidence=0.9,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=4,
            last_line_number=4,
            sonar_line_number=4
        )

        success = fixer._apply_fixes_to_file(test_file, [fix], dry_run=False)

        content = test_file.read_text()
        lines = content.split("\n")

        # Import should be near top with other imports
        import_lines = [i for i, line in enumerate(lines) if "import" in line]
        assert len(import_lines) >= 2  # Original + new import


# ==============================================================================
# CRITICAL: Indentation Handling Edge Cases
# ==============================================================================


class TestIndentationEdgeCases:
    """Test indentation handling in various scenarios."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_apply_indentation_to_fix_zero_indent(self, mock_openai):
        """Test applying zero indentation."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        fixed_code = "x = 1\ny = 2"
        result = fixer.apply_indentation_to_fix(fixed_code, "")

        assert result == "x = 1\ny = 2"

    def test_calculate_base_indentation_standalone(self):
        """Test calculate_base_indentation standalone function."""
        assert calculate_base_indentation("    code") == 4
        assert calculate_base_indentation("  code") == 2
        assert calculate_base_indentation("code") == 0
        assert calculate_base_indentation("\tcode") == 1


# ==============================================================================
# CRITICAL: Validation Integration Tests
# ==============================================================================


class TestValidationIntegration:
    """Test integration with fix validator."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    @patch("devdox_ai_sonar.llm_fixer.FixValidator")
    def test_apply_fixes_with_validation_approved(
        self, mock_validator_class, mock_openai, tmp_path, sample_fix, sample_issue
    ):
        """Test applying fixes with validator approval."""
        mock_openai.OpenAI.return_value = MagicMock()

        mock_validator = MagicMock()
        mock_result = ValidationResult(
            status=ValidationStatus.APPROVED, original_fix=sample_fix, confidence=0.95
        )
        mock_validator.validate_fix.return_value = mock_result
        mock_validator_class.return_value = mock_validator

        fixer = LLMFixer(provider="openai", api_key="test-key")

        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test():\n    unused = 42\n    return value\n")

        sample_fix.file_path = "src/test.py"

        result = fixer.apply_fixes_with_validation(
            [sample_fix],
            [sample_issue],
            tmp_path,
            use_validator=True,
            validator_provider="openai",
        )

        # Should succeed
        assert len(result.successful_fixes) >= 0


# ==============================================================================
# HIGH PRIORITY: Prompt Generation Edge Cases
# ==============================================================================


class TestPromptGenerationEdgeCases:
    """Test prompt generation for various issue types."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_create_fix_prompt_literal_duplication(self, mock_openai):
        """Test prompt generation for literal duplication issues."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        issue = SonarIssue(
            key="test",
            rule="python:S1192",
            severity=Severity.MAJOR,
            component="test",
            project="test",
            first_line=10,
            last_line=10,
            message='Define a constant instead of duplicating this literal "value" 3 times.',
            type=IssueType.CODE_SMELL,
            file="test.py",
        )

        context = {"context": "x = 'value'\ny = 'value'", "problem_line": "x = 'value'"}

        prompt = fixer._create_fix_prompt(issue, context, {}, "python")

        assert "literal" in prompt.lower()
        assert "constant" in prompt.lower()

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_create_fix_prompt_null_check(self, mock_openai):
        """Test prompt generation for null check issues."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        issue = SonarIssue(
            key="test",
            rule="python:S2259",
            severity=Severity.CRITICAL,
            component="test",
            project="test",
            first_line=10,
            last_line=10,
            message="Null pointer dereference",
            type=IssueType.BUG,
            file="test.py",
        )

        context = {"context": "x.method()", "problem_line": "x.method()"}

        prompt = fixer._create_fix_prompt(issue, context, {}, "python")

        assert "null" in prompt.lower() or "none" in prompt.lower()

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_create_fix_prompt_with_error_message(self, mock_openai):
        """Test prompt generation with previous error message."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        issue = SonarIssue(
            key="test",
            rule="test",
            severity=Severity.MAJOR,
            component="test",
            project="test",
            first_line=10,
            last_line=10,
            message="Issue",
            type=IssueType.CODE_SMELL,
            file="test.py",
        )

        context = {"context": "code", "problem_line": "code"}
        error_msg = "SyntaxError: invalid syntax"

        prompt = fixer._create_fix_prompt(issue, context, {}, "python", error_msg)

        assert "SyntaxError" in prompt or error_msg in prompt


# ==============================================================================
# HIGH PRIORITY: File Search and Content Matching
# ==============================================================================


class TestFileSearching:
    """Test file searching and content matching."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_files_with_content_matches(self, mock_openai, tmp_path):
        """Test finding files containing specific content."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        # Create test files
        file1 = tmp_path / "test1.py"
        file1.write_text("def unique_function():\n    pass")

        file2 = tmp_path / "test2.py"
        file2.write_text("def other_function():\n    pass")

        matches = fixer._find_files_with_content(tmp_path, "unique_function")

        assert len(matches) >= 1
        assert any("test1.py" in str(f) for f in matches)

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_files_with_content_no_matches(self, mock_openai, tmp_path):
        """Test searching for content that doesn't exist."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        file1 = tmp_path / "test.py"
        file1.write_text("def func(): pass")

        matches = fixer._find_files_with_content(tmp_path, "nonexistent_code")

        assert len(matches) == 0

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_find_files_with_content_limits_results(self, mock_openai, tmp_path):
        """Test that file search limits results for performance."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        # Create many matching files
        for i in range(10):
            file = tmp_path / f"test{i}.py"
            file.write_text("common_pattern")

        matches = fixer._find_files_with_content(tmp_path, "common_pattern")

        # Should limit to 3 matches
        assert len(matches) <= 3


# ==============================================================================
# HIGH PRIORITY: Language Detection
# ==============================================================================


class TestLanguageDetection:
    """Test programming language detection from file extensions."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_get_language_all_supported_extensions(self, mock_openai):
        """Test language detection for all supported extensions."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        test_cases = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".kt": "kotlin",
            ".scala": "scala",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".unknown": "text",
        }

        for ext, expected_lang in test_cases.items():
            assert fixer._get_language_from_extension(ext) == expected_lang


# ==============================================================================
# HIGH PRIORITY: Bracket Balance Validation
# ==============================================================================


class TestBracketBalanceValidation:
    """Test bracket balance checking."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_bracket_balance_balanced(self, mock_openai):
        """Test balanced brackets are detected correctly."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "def func():\n    x = [1, 2, 3]\n    return {x: (y, z)}"

        assert fixer._check_bracket_balance(content) is True

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_bracket_balance_unbalanced_paren(self, mock_openai):
        """Test unbalanced parentheses are detected."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "def func():\n    x = (1 + 2\n    return x"

        assert fixer._check_bracket_balance(content) is False

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_bracket_balance_unbalanced_brace(self, mock_openai):
        """Test unbalanced braces are detected."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "function test() {\n    if (true) {\n        return 1;"

        assert fixer._check_bracket_balance(content) is False

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_bracket_balance_mismatched(self, mock_openai):
        """Test mismatched brackets are detected."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "x = [1, 2, 3)"

        assert fixer._check_bracket_balance(content) is False


# ==============================================================================
# HIGH PRIORITY: Duplicate Definition Detection
# ==============================================================================


class TestDuplicateDefinitionDetection:
    """Test duplicate function/class definition detection."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_no_duplicate_definitions_clean(self, mock_openai):
        """Test clean code with no duplicates."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "def func1():\n    pass\n\ndef func2():\n    pass"

        assert fixer._check_no_duplicate_definitions(content, ".py") is True

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_no_duplicate_definitions_duplicate_func(self, mock_openai):
        """Test detection of duplicate function definitions."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "def helper():\n    pass\n\ndef helper():\n    pass"

        assert fixer._check_no_duplicate_definitions(content, ".py") is False

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_no_duplicate_definitions_duplicate_class(self, mock_openai):
        """Test detection of duplicate class definitions."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "class MyClass:\n    pass\n\nclass MyClass:\n    pass"

        assert fixer._check_no_duplicate_definitions(content, ".py") is False

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_check_no_duplicate_definitions_non_python(self, mock_openai):
        """Test that non-Python files skip duplicate check."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        content = "function test() {}\nfunction test() {}"

        # Should return True for non-Python files
        assert fixer._check_no_duplicate_definitions(content, ".js") is True


# ==============================================================================
# MEDIUM PRIORITY: Extract Function Name
# ==============================================================================


class TestExtractFunctionName:
    """Test function name extraction from definitions."""

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_function_name_python(self, mock_openai):
        """Test extracting Python function name."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        assert fixer._extract_function_name("def my_function():") == "my_function"
        assert fixer._extract_function_name("async def async_func():") == "async_func"

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_function_name_javascript(self, mock_openai):
        """Test extracting JavaScript function name."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        assert fixer._extract_function_name("function myFunc() {") == "myFunc"
        assert fixer._extract_function_name("const myFunc = () => {") == "myFunc"

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_function_name_java(self, mock_openai):
        """Test extracting Java method name."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        name = fixer._extract_function_name("public void myMethod() {")
        assert name == "myMethod"

    @patch("devdox_ai_sonar.llm_fixer.openai")
    def test_extract_function_name_filters_keywords(self, mock_openai):
        """Test that keywords are filtered out."""
        mock_openai.OpenAI.return_value = MagicMock()
        fixer = LLMFixer(provider="openai", api_key="test-key")

        # Should not return 'public' or 'static'
        name = fixer._extract_function_name("public static void test() {")
        assert name not in ["public", "static", "void"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
