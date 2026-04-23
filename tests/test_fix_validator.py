"""Comprehensive tests for Fix Validator."""

import pytest
from unittest.mock import patch, MagicMock

from devdox_ai_sonar.fix_validator import (
    ValidationStatus,
    ValidationResult,
    FixValidator,
    _format_block_header,
    _format_full_code_content,
    _format_diff_content,
    _format_search_replace_content,
)
from devdox_ai_sonar.llm.profile import LLMProfile
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    FixSuggestion,
    Severity,
    IssueType,
    ChangeType,
    ChangeAction,
    BlockType,
    CodeBlock,
    LineChange,
    SearchReplace,
)


@pytest.fixture
def sample_code_block():
    return CodeBlock(
        block_name="test",
        start_line="1",
        end_line="10",
        has_changes=True,
        change_type=ChangeType.FULL_CODE,
        block_type=BlockType.FUNCTION,
        context="new_code",
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
        fixed_code_blocks=[sample_code_block],
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
    """Tests for FixValidator's profile-based constructor."""

    def test_init_sets_profile_and_conveniences(self):
        profile = LLMProfile(name="prod", model="openai/gpt-4o", api_key="sk-x")
        validator = FixValidator(profile=profile)

        assert validator.profile is profile
        assert validator.model == "openai/gpt-4o"
        assert validator.api_key == "sk-x"
        assert validator.provider == "openai"  # derived from prefix

    def test_init_constructs_openhands_client(self):
        profile = LLMProfile(
            name="vllm",
            model="openai/my-model",
            api_key="k",
            base_url="https://vllm.example.com",
        )
        with patch("devdox_ai_sonar.fix_validator.LLM") as mock_llm:
            FixValidator(profile=profile)
            kwargs = mock_llm.call_args.kwargs
            assert kwargs["model"] == "openai/my-model"
            assert kwargs["api_key"] == "k"
            assert kwargs["base_url"] == "https://vllm.example.com"

    def test_init_empty_api_key_forwarded_as_none(self):
        profile = LLMProfile(name="ollama", model="ollama/llama3", api_key="")
        with patch("devdox_ai_sonar.fix_validator.LLM") as mock_llm:
            FixValidator(profile=profile)
            assert mock_llm.call_args.kwargs["api_key"] is None

    def test_init_custom_confidence_threshold(self):
        profile = LLMProfile(name="p", model="openai/gpt-4o", api_key="k")
        with patch("devdox_ai_sonar.fix_validator.LLM"):
            validator = FixValidator(profile=profile, min_confidence_threshold=0.9)
        assert validator.min_confidence_threshold == 0.9


class TestValidateFix:
    """Test fix validation."""
    @pytest.mark.skip(
        reason=(
            "validate_fix path now routes through "
            "openhands.sdk.LLM.completion. The legacy mock shape "
            "(mock_client.chat.completions.create) does not line up "
            "with the new path; these tests need to be re-authored "
            "to patch devdox_ai_sonar.fix_validator.LLM with a mock "
            "whose .completion returns the desired response."
        )
    )
    def test_validate_fix_approved(
        self,
        sample_fix,
        sample_issue,
        sample_file_content,
        sample_code_block,
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.confidence == 0.95
        assert "correctly removes" in result.explanation
    @pytest.mark.skip(
        reason=(
            "validate_fix path now routes through "
            "openhands.sdk.LLM.completion. The legacy mock shape "
            "(mock_client.chat.completions.create) does not line up "
            "with the new path; these tests need to be re-authored "
            "to patch devdox_ai_sonar.fix_validator.LLM with a mock "
            "whose .completion returns the desired response."
        )
    )
    def test_validate_fix_rejected(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW
        assert result.confidence <= 0.5
    @pytest.mark.skip(
        reason=(
            "validate_fix path now routes through "
            "openhands.sdk.LLM.completion. The legacy mock shape "
            "(mock_client.chat.completions.create) does not line up "
            "with the new path; these tests need to be re-authored "
            "to patch devdox_ai_sonar.fix_validator.LLM with a mock "
            "whose .completion returns the desired response."
        )
    )
    def test_validate_fix_modified(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_fix is not None
        assert "return value" in result.modified_fix.fixed_code
    @pytest.mark.skip(
        reason=(
            "validate_fix path now routes through "
            "openhands.sdk.LLM.completion. The legacy mock shape "
            "(mock_client.chat.completions.create) does not line up "
            "with the new path; these tests need to be re-authored "
            "to patch devdox_ai_sonar.fix_validator.LLM with a mock "
            "whose .completion returns the desired response."
        )
    )
    def test_validate_fix_needs_review(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW
    @pytest.mark.skip(
        reason=(
            "validate_fix path now routes through "
            "openhands.sdk.LLM.completion. The legacy mock shape "
            "(mock_client.chat.completions.create) does not line up "
            "with the new path; these tests need to be re-authored "
            "to patch devdox_ai_sonar.fix_validator.LLM with a mock "
            "whose .completion returns the desired response."
        )
    )
    def test_validate_fix_below_confidence_threshold(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(
            provider="openai", api_key="test-key", min_confidence_threshold=0.7
        )
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW


class TestExtractValidationContext:
    """Test context extraction for validation."""
    def test_extract_context(self):
        """Test extracting broader context for validation."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

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
    def test_validate_fix_file_not_found(self, sample_fix, sample_issue):
        """Test validation when file doesn't exist."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

        # File content is required parameter
        result = validator.validate_fix(sample_fix, sample_issue, "")

        # Should handle gracefully
        assert result is not None

    @pytest.mark.skip(reason="Need update")
    def test_validate_fix_llm_error(
        self, sample_fix, sample_issue, sample_file_content
    ):
        """Test validation when LLM call fails."""

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW


@pytest.mark.skip(
    reason=(
        "Post-OpenHands migration: fix_validator.Together and the "
        "HAS_TOGETHER guard are gone.  TogetherAI key / init now flows "
        "through openhands.sdk.LLM rather than a direct Together client."
    )
)
class TestPromptGeneration:
    """Test the content and structure of the LLM validation prompt."""

    @pytest.mark.skip(reason="Need update")
    def test_create_validation_prompt_content(
        self, sample_fix, sample_issue
    ):
        """Test that all required issue and fix details are present in the prompt."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

        context = {
            "full_context": "context lines...",
            "problem_lines": "problem line",
            "start_line": 5,
            "end_line": 15,
            "issue_start": 10,
            "issue_end": 10,
        }

        prompt = validator._create_validation_prompt(
            sample_fix, sample_issue, context, ""
        )

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
    def test_extract_context_at_file_start(self):
        """Test context extraction at the very beginning of the file."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

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
    def test_extract_context_at_file_end(self):
        """Test context extraction at the very end of the file."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

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
    def test_extract_context_multi_line_issue(self):
        """Test extraction for an issue spanning multiple lines."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

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


class TestModifiedStatusEdgeCases:
    """Test MODIFIED status parsing edge cases."""

    @pytest.mark.skip(reason="Need update")
    def test_modified_without_improved_code(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should fall back to NEEDS_REVIEW when MODIFIED but no improved fix
        assert result.status == ValidationStatus.NEEDS_REVIEW
    def test_modified_with_malformed_code_block(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should handle gracefully
        assert result is not None
    @pytest.mark.skip(
        reason=(
            "validate_fix path now routes through "
            "openhands.sdk.LLM.completion. The legacy mock shape "
            "(mock_client.chat.completions.create) does not line up "
            "with the new path; these tests need to be re-authored "
            "to patch devdox_ai_sonar.fix_validator.LLM with a mock "
            "whose .completion returns the desired response."
        )
    )
    def test_modified_with_language_specifier(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_fix is not None
        assert "return value" in result.modified_fix.fixed_code

    @pytest.mark.skip(reason="Need update")
    def test_modified_with_multiple_code_blocks(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
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
    def test_missing_explanation_notes(
        self, sample_fix, sample_issue, sample_file_content
    ):
        """Test response with missing VALIDATION_NOTES section."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = """{
"IMPROVED_FIX": "",

"CONFIDENCE": "0.95",

"PLACEMENT":"SIBLING"
}
"""
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.explanation == "Code fix applied"

    @pytest.mark.skip(reason="Need update")
    def test_missing_confidence(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
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
    def test_confidence_above_one(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should clamp to 1.0
        assert result.confidence == 1.0

    @pytest.mark.skip(reason="Need update")
    def test_confidence_below_zero(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should clamp to 0.0
        assert result.confidence == 0.0

    @pytest.mark.skip(reason="Need update")
    def test_confidence_invalid_format(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should default to 0
        assert result.confidence == 0.0
    @pytest.mark.skip(
        reason=(
            "validate_fix path now routes through "
            "openhands.sdk.LLM.completion. The legacy mock shape "
            "(mock_client.chat.completions.create) does not line up "
            "with the new path; these tests need to be re-authored "
            "to patch devdox_ai_sonar.fix_validator.LLM with a mock "
            "whose .completion returns the desired response."
        )
    )
    def test_confidence_exactly_threshold(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

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
    def test_completely_empty_response(
        self, sample_fix, sample_issue, sample_file_content
    ):
        """Test completely empty LLM response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW
    def test_response_with_only_status(
        self, sample_fix, sample_issue, sample_file_content
    ):
        """Test response with only STATUS field."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{ "IMPROVED_FIX": "", "CONFIDENCE": "0.7", "PLACEMENT":"SIBLING", "IMPROVED_EXPLANATION":"The fix correctly removes the unused variable."}'
        )
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)
        # Should handle gracefully with defaults
        assert result.status == ValidationStatus.MODIFIED
        assert result.confidence == 0.7
    def test_response_with_extra_fields(
        self, sample_fix, sample_issue, sample_file_content
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
        mock_client.completion.return_value = mock_response

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        # Should ignore extra fields
        assert result.status == ValidationStatus.MODIFIED


# ==============================================================================
# HIGH PRIORITY: Context Extraction Boundary Conditions
# ==============================================================================


class TestContextExtractionBoundaries:
    """Test context extraction edge cases."""
    def test_single_line_file(self):
        """Test context extraction for single-line file."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

        file_content = "single line"
        context = validator._extract_validation_context(
            file_content, 1, 1, context_lines=5
        )

        assert context["start_line"] == 1
        assert context["end_line"] == 1
        assert context["full_context"] == "single line"
    def test_empty_file(self):
        """Test context extraction for empty file."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

        file_content = ""
        context = validator._extract_validation_context(
            file_content, 1, 1, context_lines=5
        )

        assert context["start_line"] == 1
        assert context["full_context"] == ""
    def test_very_long_file(self):
        """Test context extraction doesn't load entire huge file."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

        # Create 10000 line file
        file_content = "\n".join([f"line {i}" for i in range(10000)])

        # Issue on line 5000
        context = validator._extract_validation_context(
            file_content, 5000, 5000, context_lines=10
        )

        # Should only extract lines 4990-5010 (20 lines context)
        lines_in_context = context["full_context"].count("\n")
        assert lines_in_context <= 21  # Issue line + 20 context lines
    def test_context_with_zero_lines(self):
        """Test context extraction with zero context lines."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

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
    def test_llm_returns_none(
        self, sample_fix, sample_issue, sample_file_content
    ):
        """Test when _call_llm_validator returns None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = None

        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW
        assert "Validation failed" in result.explanation

    @pytest.mark.skip(reason="Need update")
    def test_parse_response_exception(self, sample_fix):
        """Test _parse_validation_response with exception."""
        validator = FixValidator(profile=LLMProfile(name="openai", model="openai/gpt-4o", api_key='test-key'))

        # Malformed response that causes parsing error
        response_text = None  # This should cause an error

        result = validator._parse_validation_response(response_text, sample_fix)

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


class TestFormatBlockHeader:
    """Test _format_block_header helper."""

    def test_formats_header_with_block_info(self):
        block = CodeBlock(
            block_name="my_function",
            start_line=10,
            end_line=25,
            has_changes=True,
            change_type=ChangeType.FULL_CODE,
            block_type=BlockType.FUNCTION,
        )
        result = _format_block_header(1, block)

        assert "Block 1: my_function" in result
        assert "Lines 10-25" in result
        assert "Type: function" in result
        assert "Change Type: FULL_CODE" in result
        assert "Has Changes: True" in result

    def test_formats_header_with_no_changes(self):
        block = CodeBlock(
            block_name="unchanged",
            start_line=1,
            end_line=5,
            has_changes=False,
            change_type=ChangeType.DIFF,
            block_type=BlockType.CLASS,
        )
        result = _format_block_header(3, block)

        assert "Block 3: unchanged" in result
        assert "Has Changes: False" in result
        assert "Type: class" in result


class TestFormatFullCodeContent:
    """Test _format_full_code_content helper."""

    def test_wraps_context_in_code_fence(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.FULL_CODE,
            block_type=BlockType.FUNCTION,
            context="def func():\n    return 42",
        )
        result = _format_full_code_content(block)

        assert result.startswith("```python\n")
        assert result.endswith("\n```\n")
        assert "def func():\n    return 42" in result


class TestFormatDiffContent:
    """Test _format_diff_content helper."""

    def test_formats_replace_action(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            changes=[
                LineChange(
                    line=3,
                    action=ChangeAction.REPLACE,
                    old="x = 1",
                    new="x = 2",
                )
            ],
        )
        result = _format_diff_content(block)

        assert "Changes:\n" in result
        assert "Line 3 (REPLACE):" in result
        assert "- Old: x = 1" in result
        assert "+ New: x = 2" in result

    def test_formats_insert_action(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            changes=[
                LineChange(
                    line=4,
                    action=ChangeAction.INSERT,
                    new="    logger.info('done')",
                )
            ],
        )
        result = _format_diff_content(block)

        assert "Line 4 (INSERT):" in result
        assert "+ New:     logger.info('done')" in result
        assert "Old" not in result

    def test_formats_delete_action(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            changes=[
                LineChange(
                    line=2,
                    action=ChangeAction.DELETE,
                    old="unused = 42",
                )
            ],
        )
        result = _format_diff_content(block)

        assert "Line 2 (DELETE):" in result
        assert "- Old: unused = 42" in result
        assert "New" not in result

    def test_formats_multiple_changes(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=10,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            changes=[
                LineChange(line=2, action=ChangeAction.DELETE, old="old_line"),
                LineChange(line=5, action=ChangeAction.REPLACE, old="a", new="b"),
                LineChange(line=8, action=ChangeAction.INSERT, new="new_line"),
            ],
        )
        result = _format_diff_content(block)

        assert "Line 2 (DELETE):" in result
        assert "Line 5 (REPLACE):" in result
        assert "Line 8 (INSERT):" in result

    def test_empty_changes_list(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            changes=[],
        )
        result = _format_diff_content(block)

        assert result == "Changes:\n\n"


class TestFormatSearchReplaceContent:
    """Test _format_search_replace_content helper."""

    def test_formats_simple_replacement(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            replacements=[
                SearchReplace(search="foo", replace="bar"),
            ],
        )
        result = _format_search_replace_content(block)

        assert "Search/Replace Operations:\n" in result
        assert "Operation 1" in result
        assert "(all occurrences)" in result
        assert "Search:  'foo'" in result
        assert "Replace: 'bar'" in result
        assert "(REGEX)" not in result

    def test_formats_regex_replacement(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            replacements=[
                SearchReplace(search=r"\bawait\s+", replace="", is_regex=True, count=1),
            ],
        )
        result = _format_search_replace_content(block)

        assert "(REGEX)" in result
        assert "(count: 1)" in result

    def test_formats_multiple_replacements(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=10,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            replacements=[
                SearchReplace(search="a", replace="b"),
                SearchReplace(search="c", replace="d", is_regex=True, count=2),
            ],
        )
        result = _format_search_replace_content(block)

        assert "Operation 1" in result
        assert "Operation 2" in result
        assert "(all occurrences)" in result
        assert "(count: 2)" in result

    def test_empty_replacements_list(self):
        block = CodeBlock(
            block_name="func",
            start_line=1,
            end_line=5,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.FUNCTION,
            replacements=[],
        )
        result = _format_search_replace_content(block)

        assert result == "Search/Replace Operations:\n\n"


class TestTryExtractJson:
    """Tests for _try_extract_json markdown fence stripping."""

    def test_returns_none_for_empty_string(self):
        assert FixValidator._try_extract_json("") is None

    def test_returns_none_for_none(self):
        assert FixValidator._try_extract_json(None) is None

    def test_strips_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert FixValidator._try_extract_json(text) == '{"key": "value"}'

    def test_strips_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert FixValidator._try_extract_json(text) == '{"key": "value"}'

    def test_strips_surrounding_whitespace(self):
        text = '  \n```json\n{"key": "value"}\n```\n  '
        assert FixValidator._try_extract_json(text) == '{"key": "value"}'

    def test_returns_text_unchanged_without_fences(self):
        text = '{"key": "value"}'
        assert FixValidator._try_extract_json(text) == '{"key": "value"}'

    def test_fence_without_newline(self):
        text = '```{"key": "value"}```'
        result = FixValidator._try_extract_json(text)
        assert "key" in result


class TestBackfillMissingContext:
    """Tests for FixValidator._backfill_missing_context."""

    @staticmethod
    def _make_block(**kwargs):
        """Helper to create a CodeBlock with sensible defaults."""
        defaults = {
            "block_name": "test_block",
            "block_type": BlockType.FUNCTION,
        }
        defaults.update(kwargs)
        return CodeBlock(**defaults)

    def test_copies_context_from_matching_original_block(self):
        """FULL_CODE block with no context gets context from the matching original."""
        original = self._make_block(
            start_line=10, end_line=20, has_changes=True,
            change_type=ChangeType.FULL_CODE, context="def foo():\n    pass",
        )
        validated = self._make_block(
            start_line=10, end_line=20, has_changes=True,
            change_type=ChangeType.FULL_CODE, context=None,
        )

        FixValidator._backfill_missing_context([validated], [original])

        assert validated.context == "def foo():\n    pass"

    def test_skips_block_with_existing_context(self):
        """Block that already has context is left untouched."""
        original = self._make_block(
            start_line=10, end_line=20, has_changes=True,
            change_type=ChangeType.FULL_CODE, context="original context",
        )
        validated = self._make_block(
            start_line=10, end_line=20, has_changes=True,
            change_type=ChangeType.FULL_CODE, context="already set",
        )

        FixValidator._backfill_missing_context([validated], [original])

        assert validated.context == "already set"

    def test_skips_block_without_changes(self):
        """Block with has_changes=False is left untouched."""
        validated = self._make_block(
            start_line=10, end_line=20, has_changes=False,
            change_type=ChangeType.FULL_CODE, context=None,
        )

        FixValidator._backfill_missing_context([validated], [])

        assert validated.context is None

    def test_skips_non_full_code_block(self):
        """DIFF block is left untouched even if context is None."""
        validated = self._make_block(
            start_line=10, end_line=20, has_changes=True,
            change_type=ChangeType.DIFF, context=None,
        )

        FixValidator._backfill_missing_context([validated], [])

        assert validated.context is None

    def test_no_matching_original_prints_warning(self, capsys):
        """When no original block matches, a warning is printed."""
        validated = self._make_block(
            start_line=99, end_line=105, has_changes=True,
            change_type=ChangeType.FULL_CODE, context=None,
        )

        FixValidator._backfill_missing_context([validated], [])

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "99" in captured.out

    def test_multiple_blocks_handled(self):
        """Multiple validated blocks are each processed independently."""
        originals = [
            self._make_block(start_line=1, end_line=5, has_changes=True,
                             change_type=ChangeType.FULL_CODE, context="ctx1"),
            self._make_block(start_line=10, end_line=15, has_changes=True,
                             change_type=ChangeType.FULL_CODE, context="ctx2"),
        ]
        validated = [
            self._make_block(start_line=1, end_line=5, has_changes=True,
                             change_type=ChangeType.FULL_CODE, context=None),
            self._make_block(start_line=10, end_line=15, has_changes=True,
                             change_type=ChangeType.FULL_CODE, context=None),
        ]

        FixValidator._backfill_missing_context(validated, originals)

        assert validated[0].context == "ctx1"
        assert validated[1].context == "ctx2"

    def test_empty_blocks_list(self):
        """Empty blocks list is a no-op."""
        FixValidator._backfill_missing_context([], [])


class TestFormatCodeBlocksForValidation:
    """Tests for FixValidator._format_code_blocks_for_validation."""

    @staticmethod
    def _make_block(**kwargs):
        defaults = {
            "block_name": "test_block",
            "block_type": BlockType.FUNCTION,
            "start_line": 1,
            "end_line": 10,
            "has_changes": True,
            "change_type": ChangeType.FULL_CODE,
            "context": "def foo():\n    pass",
        }
        defaults.update(kwargs)
        return CodeBlock(**defaults)

    def _make_validator(self):
        with patch("devdox_ai_sonar.fix_validator.LLM"):
            return FixValidator(profile=LLMProfile(name="openrouter", model="openrouter/anthropic/claude-sonnet-4", api_key='test-key'))

    def test_empty_list_returns_empty_string_and_zeros(self):
        validator = self._make_validator()
        text, start, end = validator._format_code_blocks_for_validation([])
        assert text == ""
        assert start == 0
        assert end == 0

    def test_single_block_returns_its_line_range(self):
        validator = self._make_validator()
        block = self._make_block(start_line=5, end_line=15)
        text, start, end = validator._format_code_blocks_for_validation([block])
        assert start == 5
        assert end == 15
        assert isinstance(text, str)
        assert len(text) > 0

    def test_multiple_blocks_returns_min_start_max_end(self):
        validator = self._make_validator()
        blocks = [
            self._make_block(start_line=20, end_line=30),
            self._make_block(start_line=5, end_line=50),
            self._make_block(start_line=10, end_line=25),
        ]
        text, start, end = validator._format_code_blocks_for_validation(blocks)
        assert start == 5
        assert end == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
