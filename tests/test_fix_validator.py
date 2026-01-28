"""Comprehensive tests for Fix Validator."""

import pytest
from unittest.mock import patch, MagicMock

from devdox_ai_sonar.fix_validator import (
    ValidationStatus,
    ValidationResult,
    FixValidator,
)
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    FixSuggestion,
    Severity,
    IssueType,
    ChangeType,
    BlockType,
    CodeBlock
)


@pytest.fixture
def sample_code_block():
    return CodeBlock(block_name="test",
                     start_line="1",
                     end_line="10",
                     has_changes=True,
                     change_type=ChangeType.FULL_CODE,
                     block_type=BlockType.FUNCTION,
                     context="new_code"
                     )

@pytest.fixture
def sample_issue():
    """Create a sample SonarCloud issue."""
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
    )


@pytest.fixture
def sample_fix(sample_code_block):
    """Create a sample fix suggestion."""
    return FixSuggestion(
        issue_key="test:src/test.py:S1481",
        original_code="    unused_var = 42\n    return value",
        fixed_code="    return value",
        explanation="Removed unused variable",
        confidence=0.95,
        llm_model="gpt-4",
        file_path="src/test.py",
        line_number=10,
        last_line_number=11,
        fixed_code_blocks=[sample_code_block]
    )


@pytest.fixture
def sample_file_content():
    """Sample file content for validation."""
    return """def my_function():
    unused_var = 42
    value = 100
    return value
"""


class TestFixValidatorInitialization:
    """Test FixValidator initialization."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_init_openai_provider(self, mock_openai):
        """Test initialization with OpenAI provider."""
        mock_openai.OpenAI.return_value = MagicMock()

        validator = FixValidator(provider="openai", api_key="test-key")

        assert validator.provider == "openai"
        assert validator.model == "gpt-4o"
        assert validator.api_key == "test-key"

    @patch("devdox_ai_sonar.fix_validator.HAS_OPENAI", False)
    def test_init_openai_missing_library(self):
        """Test initialization fails when OpenAI library missing."""

        with pytest.raises(ImportError, match="OpenAI library not installed"):
            FixValidator(provider="openai", api_key="test-key")

    @patch("devdox_ai_sonar.fix_validator.genai")
    def test_init_gemini_provider(self, mock_genai):
        """Test initialization with Gemini provider."""
        mock_genai.Client.return_value = MagicMock()

        validator = FixValidator(provider="gemini", api_key="test-key")

        assert validator.provider == "gemini"
        assert validator.model == "gemini-1.5-flash"

    @patch("devdox_ai_sonar.fix_validator.HAS_GEMINI", False)
    def test_init_gemini_missing_library(self):
        """Test initialization fails when Gemini library missing."""

        with pytest.raises(ImportError, match="Gemini library not installed"):
            FixValidator(provider="gemini", api_key="test-key")

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider."""

        with pytest.raises(ValueError, match="Unsupported provider"):
            FixValidator(provider="invalid", api_key="test-key")

    @patch.dict("os.environ", {}, clear=True)
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_init_missing_api_key(self, mock_openai):
        """Test initialization fails when API key missing."""

        with pytest.raises(ValueError, match="API key not provided"):
            FixValidator(provider="openai", api_key=None)

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_init_custom_model(self, mock_openai):
        """Test initialization with custom model."""
        mock_openai.OpenAI.return_value = MagicMock()

        validator = FixValidator(
            provider="openai", model="gpt-4-turbo", api_key="test-key"
        )

        assert validator.model == "gpt-4-turbo"

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_init_custom_confidence_threshold(self, mock_openai):
        """Test initialization with custom confidence threshold."""
        mock_openai.OpenAI.return_value = MagicMock()

        validator = FixValidator(
            provider="openai", api_key="test-key", min_confidence_threshold=0.8
        )

        assert validator.min_confidence_threshold == 0.8

@pytest.mark.skip(reason="Need update")
class TestValidateFix:
    """Test fix validation."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_approved(
        self, mock_openai, sample_fix, sample_issue, sample_file_content, sample_code_block
    ):
        """Test validation when fix is approved."""

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
"IMPROVED_FIX": "",

"CONFIDENCE": "0.95",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"The fix correctly removes the unused variable."
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.confidence == 0.95
        assert "correctly removes" in result.explanation

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_rejected(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test validation when fix is rejected."""

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """

{
"IMPROVED_FIX": "",

"CONFIDENCE": "0.3",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"This fix would break the code."
}

"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key",model="gpt-4o")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW
        assert result.confidence <= 0.5

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_modified(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test validation when fix is modified."""

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
"IMPROVED_FIX": "return value",

"CONFIDENCE": "0.85",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"Added comment for clarity"
}

"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_fix is not None
        assert "return value" in result.modified_fix.fixed_code

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_needs_review(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test validation when fix needs review."""

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
STATUS: NEEDS_REVIEW

CONFIDENCE: 0.5

VALIDATION_NOTES:
Uncertain about side effects.

CONCERNS:
- May affect other code
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_below_confidence_threshold(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test that low confidence approved fixes become NEEDS_REVIEW."""

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
STATUS: APPROVED

CONFIDENCE: 0.6

VALIDATION_NOTES:
Fix looks okay but not confident.

CONCERNS:
None
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(
            provider="openai", api_key="test-key", min_confidence_threshold=0.7
        )
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW


class TestExtractValidationContext:
    """Test context extraction for validation."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_extract_context(self, mock_openai):
        """Test extracting broader context for validation."""

        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = """def function1():
    x = 1
    return x

def function2():
    unused = 42
    return 0
"""

        context = validator._extract_validation_context(
            file_content, 6, 6, context_lines=20
        )

        assert "function2" in context["full_context"]
        assert context["issue_start"] == 6




class TestValidationResultProperties:
    """Test ValidationResult properties."""

    def test_should_apply_approved(self, sample_fix):
        """Test should_apply for approved fix."""

        result = ValidationResult(
            status=ValidationStatus.APPROVED, original_fix=sample_fix, confidence=0.9
        )

        assert result.should_apply is True

    def test_should_apply_modified(self, sample_fix, sample_code_block):
        """Test should_apply for modified fix."""

        modified_fix = FixSuggestion(
            issue_key=sample_fix.issue_key,
            original_code=sample_fix.original_code,
            fixed_code="# improved code",
            explanation="Better fix",
            confidence=0.9,
            llm_model="gpt-4",
            file_path=sample_fix.file_path,
            line_number=sample_fix.line_number,
            last_line_number=sample_fix.last_line_number,
            fixed_code_blocks=[sample_code_block],
        )

        result = ValidationResult(
            status=ValidationStatus.MODIFIED,
            original_fix=sample_fix,
            modified_fix=modified_fix,
            confidence=0.85,
        )

        assert result.should_apply is True
        assert result.final_fix == modified_fix

    def test_should_apply_rejected(self, sample_fix):
        """Test should_apply for rejected fix."""

        result = ValidationResult(
            status=ValidationStatus.REJECTED, original_fix=sample_fix, confidence=0.3
        )

        assert result.should_apply is False

    def test_should_apply_needs_review(self, sample_fix):
        """Test should_apply for fix needing review."""

        result = ValidationResult(
            status=ValidationStatus.NEEDS_REVIEW,
            original_fix=sample_fix,
            confidence=0.5,
        )

        assert result.should_apply is False

    def test_final_fix_original(self, sample_fix):
        """Test final_fix returns original when not modified."""

        result = ValidationResult(
            status=ValidationStatus.APPROVED, original_fix=sample_fix, confidence=0.9
        )

        assert result.final_fix == sample_fix


class TestErrorHandling:
    """Test error handling in validation."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_file_not_found(self, mock_openai, sample_fix, sample_issue):
        """Test validation when file doesn't exist."""

        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        # File content is required parameter
        result = validator.validate_fix(sample_fix, sample_issue, "")

        # Should handle gracefully
        assert result is not None

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_llm_error(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test validation when LLM call fails."""

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW


class TestTogetherAIInitializationValidator:
    """Test TogetherAI provider specific initialization for FixValidator."""

    def test_togetherai_initialization(self):
        """Test TogetherAI provider initialization."""
        # Mock Together client and set the HAS_TOGETHER flag to True
        with patch("devdox_ai_sonar.fix_validator.Together") as mock_together:
            with patch("devdox_ai_sonar.fix_validator.HAS_TOGETHER", True):
                mock_together.return_value = MagicMock()
                fixer = FixValidator(provider="togetherai", api_key="test-together-key")
                assert fixer.provider == "togetherai"
                assert fixer.model == "gpt-4o"
                assert fixer.api_key == "test-together-key"
                mock_together.assert_called_once()

    def test_togetherai_missing_library(self):
        """Test error when TogetherAI lib is missing."""
        with patch("devdox_ai_sonar.fix_validator.HAS_TOGETHER", False):
            with pytest.raises(ImportError, match="Together AI library not installed"):
                FixValidator(provider="togetherai", api_key="key")

    @patch.dict("os.environ", {"TOGETHER_API_KEY": "env-key"}, clear=True)
    def test_togetherai_api_key_from_env(self):
        """Test API key is loaded from environment variables."""
        with patch("devdox_ai_sonar.fix_validator.Together"):
            with patch("devdox_ai_sonar.fix_validator.HAS_TOGETHER", True):
                fixer = FixValidator(provider="togetherai", api_key=None)
                assert fixer.api_key == "env-key"


class TestPromptGeneration:
    """Test the content and structure of the LLM validation prompt."""

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_create_validation_prompt_content(
        self, mock_openai, sample_fix, sample_issue
    ):
        """Test that all required issue and fix details are present in the prompt."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        context = {
            "full_context": "context lines...",
            "problem_lines": "problem line",
            "start_line": 5,
            "end_line": 15,
            "issue_start": 10,
            "issue_end": 10,
        }

        prompt = validator._create_validation_prompt(sample_fix, sample_issue, context,"")

        # Check Issue details
        assert sample_issue.rule in prompt
        assert sample_issue.message in prompt
        # assert str(sample_issue.severity) in prompt
        # assert f"{sample_issue.first_line}-{sample_issue.last_line}" in prompt

        # Check Fix details
        assert f"{sample_fix.confidence:.2f}" in prompt

        assert sample_fix.fixed_code in prompt

        # Check Context details
        assert "context lines..." in prompt
        assert "5-15" in prompt
        assert "Your Task:" in prompt

        assert "IMPROVED_FIX" in prompt


class TestAdvancedContextExtraction:
    """Test context extraction for validation, particularly boundary conditions."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_extract_context_at_file_start(self, mock_openai):
        """Test context extraction at the very beginning of the file."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6"

        # Issue on line 1 (first_line=1, last_line=1). context_lines=20 (maximum)
        context = validator._extract_validation_context(
            file_content, 1, 1, context_lines=20
        )

        # Should start at line 1 and end at the end of the file (line 6)
        assert context["start_line"] == 1
        assert context["end_line"] == 6
        assert context["issue_start"] == 1
        assert context["full_context"] == file_content

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_extract_context_at_file_end(self, mock_openai):
        """Test context extraction at the very end of the file."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6"
        lines = file_content.split("\n")

        # Issue on line 6 (first_line=6, last_line=6). context_lines=20
        context = validator._extract_validation_context(
            file_content, 6, 6, context_lines=20
        )

        # Should start at line 1 and end at the end of the file (line 6)
        assert context["start_line"] == 1
        assert context["end_line"] == 6
        assert context["issue_start"] == 6
        assert context["full_context"] == file_content

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_extract_context_multi_line_issue(self, mock_openai):
        """Test extraction for an issue spanning multiple lines."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = (
            "line 1\nline 2 (start issue)\nline 3\nline 4 (end issue)\nline 5\nline 6"
        )

        # Issue spans lines 2 to 4. context_lines=1
        context = validator._extract_validation_context(
            file_content, 2, 4, context_lines=1
        )

        # Start Index: max(0, 2-1-1) = 0. Start Line: 1
        # End Index: min(6, 4-1+1+1) = 5. End Line: 5
        assert context["start_line"] == 1
        assert context["end_line"] == 5
        assert context["issue_start"] == 2
        assert context["issue_end"] == 4
        # Full Context should be lines 1 through 5
        assert context["full_context"].count("\n") == 4


class TestGeminiProviderIntegration:
    """Test Gemini LLM provider integration."""

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.genai")
    def test_call_llm_validator_gemini_success(
        self, mock_genai, sample_fix, sample_issue, sample_file_content
    ):
        """Test successful LLM call with Gemini provider."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """{
"IMPROVED_FIX": "",

"CONFIDENCE": "0.95",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"The fix correctly removes the unused variable."
}
"""
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        validator = FixValidator(provider="gemini", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.confidence == 0.95
        mock_client.models.generate_content.assert_called_once()

    @patch("devdox_ai_sonar.fix_validator.genai")
    def test_call_llm_validator_gemini_error(
        self, mock_genai, sample_fix, sample_issue, sample_file_content
    ):
        """Test Gemini provider error handling."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Gemini API Error")
        mock_genai.Client.return_value = mock_client

        validator = FixValidator(provider="gemini", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW
        assert "Validation failed" in result.explanation

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.genai")
    def test_call_llm_validator_gemini_empty_response(
        self, mock_genai, sample_fix, sample_issue, sample_file_content
    ):
        """Test handling of empty response from Gemini."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        validator = FixValidator(provider="gemini", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW


class TestTogetherAIProviderIntegration:
    """Test TogetherAI LLM provider integration."""

    @patch("devdox_ai_sonar.fix_validator.Together")
    @patch("devdox_ai_sonar.fix_validator.HAS_TOGETHER", True)
    def test_call_llm_validator_togetherai_success(
        self, mock_together, sample_fix, sample_issue, sample_file_content
    ):
        """Test successful LLM call with TogetherAI provider."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
STATUS: APPROVED
CONFIDENCE: 0.92
VALIDATION_NOTES: Good fix
CONCERNS: None
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_together.return_value = mock_client

        validator = FixValidator(provider="togetherai", api_key="test-key")

        # TogetherAI currently not implemented in _call_llm_validator
        # This test documents expected behavior
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should handle gracefully even if not implemented
        assert result is not None

    @patch("devdox_ai_sonar.fix_validator.Together")
    @patch("devdox_ai_sonar.fix_validator.HAS_TOGETHER", True)
    def test_togetherai_with_custom_model(self, mock_together):
        """Test TogetherAI initialization with custom model."""
        mock_together.return_value = MagicMock()

        validator = FixValidator(
            provider="togetherai",
            model="meta-llama/Llama-3-70b-chat-hf",
            api_key="test-key",
        )

        assert validator.model == "meta-llama/Llama-3-70b-chat-hf"
        assert validator.provider == "togetherai"


class TestModifiedStatusEdgeCases:
    """Test MODIFIED status parsing edge cases."""

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_modified_without_improved_code(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test MODIFIED status without improved code block → should become NEEDS_REVIEW."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
STATUS: MODIFIED

CONFIDENCE: 0.85

VALIDATION_NOTES:
Fix needs improvement but no code provided.

CONCERNS:
None

IMPROVED_EXPLANATION:
Should have better error handling
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should fall back to NEEDS_REVIEW when MODIFIED but no improved fix
        assert result.status == ValidationStatus.NEEDS_REVIEW

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_modified_with_malformed_code_block(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test MODIFIED with malformed code block."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
STATUS: MODIFIED
CONFIDENCE: 0.85
VALIDATION_NOTES: Improved
CONCERNS: None

IMPROVED_FIX:
```python
    # Missing closing backticks
    return value
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should handle gracefully
        assert result is not None

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_modified_with_language_specifier(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test MODIFIED with language specifier in code block."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
"IMPROVED_FIX": "return value",

"CONFIDENCE": "0.9",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"Added clear comment"
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_fix is not None
        assert "return value" in result.modified_fix.fixed_code

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_modified_with_multiple_code_blocks(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test MODIFIED with multiple code blocks (should use first one)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
        {
"IMPROVED_FIX": "return different_value",

"CONFIDENCE": "0.9",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"The fix correctly removes the unused variable."
}

"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        # Should extract first code block
        assert "different_value" in result.modified_fix.fixed_code


# ==============================================================================
# CRITICAL: Regex Parsing Edge Cases
# ==============================================================================


class TestRegexParsingEdgeCases:
    """Test parsing edge cases in validation responses."""

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_missing_explanation_notes(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test response with missing VALIDATION_NOTES section."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content =  """{
"IMPROVED_FIX": "",

"CONFIDENCE": "0.95",

"PLACEMENT":"SIBLING"
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.explanation == "Code fix applied"

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_missing_confidence(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test response with missing CONFIDENCE field."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
        {
"IMPROVED_FIX": "",


"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":""
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)
        print("result ", result)
        # Should default to 0.0 as confidence is required
        assert result.confidence == 0.0


# ==============================================================================
# CRITICAL: Confidence Boundary Conditions
# ==============================================================================


class TestConfidenceBoundaryConditions:
    """Test confidence value edge cases."""

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_confidence_above_one(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test confidence value > 1.0 is clamped."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
{
"IMPROVED_FIX": "",

"CONFIDENCE": "1.5",

"PLACEMENT":"",
"IMPROVED_EXPLANATION":""
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should clamp to 1.0
        assert result.confidence == 1.0

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_confidence_below_zero(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test confidence value < 0.0 is clamped."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
STATUS: REJECTED
CONFIDENCE: -0.2
VALIDATION_NOTES: Bad fix
CONCERNS: Multiple issues
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should clamp to 0.0
        assert result.confidence == 0.0

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_confidence_invalid_format(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test non-numeric confidence value."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """
{
"IMPROVED_FIX": "",

"CONFIDENCE": "high",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":""
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should default to 0
        assert result.confidence == 0.0

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_confidence_exactly_threshold(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test confidence exactly at threshold."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
"IMPROVED_FIX": "",

"CONFIDENCE": "0.7",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"The fix correctly removes the unused variable."
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(
            provider="openai", api_key="test-key", min_confidence_threshold=0.7
        )
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Exactly at threshold should pass
        assert result.status == ValidationStatus.MODIFIED


# ==============================================================================
# HIGH PRIORITY: Validation Response Edge Cases
# ==============================================================================

@pytest.mark.skip(reason="Need update")
class TestValidationResponseEdgeCases:
    """Test edge cases in validation responses."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_completely_empty_response(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test completely empty LLM response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_response_with_only_status(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test response with only STATUS field."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{ "IMPROVED_FIX": "", "CONFIDENCE": "0.7", "PLACEMENT":"SIBLING", "IMPROVED_EXPLANATION":"The fix correctly removes the unused variable."}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)
        # Should handle gracefully with defaults
        assert result.status == ValidationStatus.MODIFIED
        assert result.confidence == 0.7

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_response_with_extra_fields(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test response with unexpected extra fields."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
"IMPROVED_FIX": "",

"CONFIDENCE": "0.9",

"PLACEMENT":"SIBLING",
"IMPROVED_EXPLANATION":"",
"EXTRA_FIELD": " This shouldn't break anything",
"ANOTHER_UNEXPECTED": "Field"
}
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should ignore extra fields
        assert result.status == ValidationStatus.MODIFIED


# ==============================================================================
# HIGH PRIORITY: Context Extraction Boundary Conditions
# ==============================================================================


class TestContextExtractionBoundaries:
    """Test context extraction edge cases."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_single_line_file(self, mock_openai):
        """Test context extraction for single-line file."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = "single line"
        context = validator._extract_validation_context(
            file_content, 1, 1, context_lines=5
        )

        assert context["start_line"] == 1
        assert context["end_line"] == 1
        assert context["full_context"] == "single line"

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_empty_file(self, mock_openai):
        """Test context extraction for empty file."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = ""
        context = validator._extract_validation_context(
            file_content, 1, 1, context_lines=5
        )

        assert context["start_line"] == 1
        assert context["full_context"] == ""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_very_long_file(self, mock_openai):
        """Test context extraction doesn't load entire huge file."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        # Create 10000 line file
        file_content = "\n".join([f"line {i}" for i in range(10000)])

        # Issue on line 5000
        context = validator._extract_validation_context(
            file_content, 5000, 5000, context_lines=10
        )

        # Should only extract lines 4990-5010 (20 lines context)
        lines_in_context = context["full_context"].count("\n")
        assert lines_in_context <= 21  # Issue line + 20 context lines

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_context_with_zero_lines(self, mock_openai):
        """Test context extraction with zero context lines."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = "line 1\nline 2\nline 3\nline 4\nline 5"

        context = validator._extract_validation_context(
            file_content, 3, 3, context_lines=0
        )

        # Should only get the exact issue line
        assert context["problem_lines"] == "line 3"


# ==============================================================================
# HIGH PRIORITY: Batch Validation Scenarios
# ==============================================================================



class TestAdditionalEdgeCases:
    """Test additional edge cases for comprehensive coverage."""

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_llm_returns_none(
        self, mock_openai, sample_fix, sample_issue, sample_file_content
    ):
        """Test when _call_llm_validator returns None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = None
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW
        assert "Validation failed" in result.explanation

    @pytest.mark.skip(reason="Need update")
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_parse_response_exception(self, mock_openai, sample_fix):
        """Test _parse_validation_response with exception."""
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        # Malformed response that causes parsing error
        response_text = None  # This should cause an error

        result = validator._parse_validation_response(
            response_text, sample_fix
        )

        assert result.status == ValidationStatus.NEEDS_REVIEW
        assert "Failed to parse" in result.explanation

    def test_validation_result_default_concerns(self, sample_fix):
        """Test ValidationResult with no concerns provided."""
        result = ValidationResult(
            status=ValidationStatus.APPROVED, original_fix=sample_fix, confidence=0.9
        )

        assert result.concerns == []

    def test_validation_result_custom_concerns(self, sample_fix):
        """Test ValidationResult with custom concerns."""
        concerns = ["Issue 1", "Issue 2"]
        result = ValidationResult(
            status=ValidationStatus.NEEDS_REVIEW,
            original_fix=sample_fix,
            concerns=concerns,
            confidence=0.5,
        )

        assert len(result.concerns) == 2
        assert result.concerns == concerns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
