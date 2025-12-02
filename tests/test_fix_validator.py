"""Comprehensive tests for Fix Validator."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from devdox_ai_sonar.models import (
    SonarIssue,
    FixSuggestion,
    Severity,
    IssueType,
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
def sample_fix():
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
        
        from devdox_ai_sonar.fix_validator import FixValidator

        validator = FixValidator(provider="openai", api_key="test-key")
        
        assert validator.provider == "openai"
        assert validator.model == "gpt-4o"
        assert validator.api_key == "test-key"

    @patch("devdox_ai_sonar.fix_validator.HAS_OPENAI", False)
    def test_init_openai_missing_library(self):
        """Test initialization fails when OpenAI library missing."""
        from devdox_ai_sonar.fix_validator import FixValidator
        
        with pytest.raises(ImportError, match="OpenAI library not installed"):
            FixValidator(provider="openai", api_key="test-key")

    @patch("devdox_ai_sonar.fix_validator.genai")
    def test_init_gemini_provider(self, mock_genai):
        """Test initialization with Gemini provider."""
        mock_genai.Client.return_value = MagicMock()
        
        from devdox_ai_sonar.fix_validator import FixValidator

        validator = FixValidator(provider="gemini", api_key="test-key")
        
        assert validator.provider == "gemini"
        assert validator.model == "claude-3-5-sonnet-20241022"

    @patch("devdox_ai_sonar.fix_validator.HAS_GEMINI", False)
    def test_init_gemini_missing_library(self):
        """Test initialization fails when Gemini library missing."""
        from devdox_ai_sonar.fix_validator import FixValidator
        
        with pytest.raises(ImportError, match="Gemini library not installed"):
            FixValidator(provider="gemini", api_key="test-key")

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider."""
        from devdox_ai_sonar.fix_validator import FixValidator
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            FixValidator(provider="invalid", api_key="test-key")

    @patch.dict("os.environ", {}, clear=True)
    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_init_missing_api_key(self, mock_openai):
        """Test initialization fails when API key missing."""
        from devdox_ai_sonar.fix_validator import FixValidator
        
        with pytest.raises(ValueError, match="API key not provided"):
            FixValidator(provider="openai", api_key=None)

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_init_custom_model(self, mock_openai):
        """Test initialization with custom model."""
        mock_openai.OpenAI.return_value = MagicMock()
        
        from devdox_ai_sonar.fix_validator import FixValidator

        validator = FixValidator(
            provider="openai",
            model="gpt-4-turbo",
            api_key="test-key"
        )
        
        assert validator.model == "gpt-4-turbo"

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_init_custom_confidence_threshold(self, mock_openai):
        """Test initialization with custom confidence threshold."""
        mock_openai.OpenAI.return_value = MagicMock()
        
        from devdox_ai_sonar.fix_validator import FixValidator

        validator = FixValidator(
            provider="openai",
            api_key="test-key",
            min_confidence_threshold=0.8
        )
        
        assert validator.min_confidence_threshold == 0.8


class TestValidateFix:
    """Test fix validation."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_approved(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test validation when fix is approved."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
STATUS: APPROVED

CONFIDENCE: 0.95

VALIDATION_NOTES:
The fix correctly removes the unused variable.

CONCERNS:
None
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.APPROVED
        assert result.confidence == 0.95
        assert "correctly removes" in result.validation_notes

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_rejected(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test validation when fix is rejected."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
STATUS: REJECTED

CONFIDENCE: 0.3

VALIDATION_NOTES:
This fix would break the code.

CONCERNS:
- Removes necessary variable
- Missing error handling
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.REJECTED
        assert len(result.concerns) > 0

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_modified(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test validation when fix is modified."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
STATUS: MODIFIED

CONFIDENCE: 0.85

VALIDATION_NOTES:
Fix is good but can be improved.

CONCERNS:
None

IMPROVED_FIX:
```python
    # Better implementation
    return value
```

IMPROVED_EXPLANATION:
Added comment for clarity
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.MODIFIED
        assert result.modified_fix is not None
        assert "Better implementation" in result.modified_fix.fixed_code

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_needs_review(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test validation when fix needs review."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
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
    def test_validate_fix_below_confidence_threshold(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test that low confidence approved fixes become NEEDS_REVIEW."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
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
            provider="openai",
            api_key="test-key",
            min_confidence_threshold=0.7
        )
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW


class TestExtractValidationContext:
    """Test context extraction for validation."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_extract_context(self, mock_openai):
        """Test extracting broader context for validation."""
        from devdox_ai_sonar.fix_validator import FixValidator
        
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        file_content = """def function1():
    x = 1
    return x

def function2():
    unused = 42
    return 0
"""
        
        context = validator._extract_validation_context(file_content, 6, 6, context_lines=20)

        assert "function2" in context["full_context"]
        assert context["issue_start"] == 6


class TestParseValidationResponse:
    """Test parsing validation responses."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_parse_response_with_all_fields(self, mock_openai, sample_fix, sample_issue):
        """Test parsing response with all fields present."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        response_text = """
STATUS: APPROVED

CONFIDENCE: 0.9

VALIDATION_NOTES:
Excellent fix.

CONCERNS:
- Minor style issue
"""
        
        result = validator._parse_validation_response(response_text, sample_fix, sample_issue)

        assert result.status == ValidationStatus.APPROVED
        assert result.confidence == 0.9
        assert len(result.concerns) > 0

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_parse_response_missing_fields(self, mock_openai, sample_fix, sample_issue):
        """Test parsing response with missing fields."""
        from devdox_ai_sonar.fix_validator import FixValidator
        
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        response_text = "Invalid response"
        
        result = validator._parse_validation_response(response_text, sample_fix, sample_issue)

        # Should default to NEEDS_REVIEW
        assert result is not None


class TestValidationResultProperties:
    """Test ValidationResult properties."""

    def test_should_apply_approved(self, sample_fix):
        """Test should_apply for approved fix."""
        from devdox_ai_sonar.fix_validator import ValidationResult, ValidationStatus
        
        result = ValidationResult(
            status=ValidationStatus.APPROVED,
            original_fix=sample_fix,
            confidence=0.9
        )

        assert result.should_apply is True

    def test_should_apply_modified(self, sample_fix):
        """Test should_apply for modified fix."""
        from devdox_ai_sonar.fix_validator import ValidationResult, ValidationStatus
        
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
        )

        result = ValidationResult(
            status=ValidationStatus.MODIFIED,
            original_fix=sample_fix,
            modified_fix=modified_fix,
            confidence=0.85
        )

        assert result.should_apply is True
        assert result.final_fix == modified_fix

    def test_should_apply_rejected(self, sample_fix):
        """Test should_apply for rejected fix."""
        from devdox_ai_sonar.fix_validator import ValidationResult, ValidationStatus
        
        result = ValidationResult(
            status=ValidationStatus.REJECTED,
            original_fix=sample_fix,
            confidence=0.3
        )

        assert result.should_apply is False

    def test_should_apply_needs_review(self, sample_fix):
        """Test should_apply for fix needing review."""
        from devdox_ai_sonar.fix_validator import ValidationResult, ValidationStatus
        
        result = ValidationResult(
            status=ValidationStatus.NEEDS_REVIEW,
            original_fix=sample_fix,
            confidence=0.5
        )

        assert result.should_apply is False

    def test_final_fix_original(self, sample_fix):
        """Test final_fix returns original when not modified."""
        from devdox_ai_sonar.fix_validator import ValidationResult, ValidationStatus
        
        result = ValidationResult(
            status=ValidationStatus.APPROVED,
            original_fix=sample_fix,
            confidence=0.9
        )

        assert result.final_fix == sample_fix


class TestBatchValidation:
    """Test batch validation functionality."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fixes_batch(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test validating multiple fixes in batch."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
STATUS: APPROVED
CONFIDENCE: 0.9
VALIDATION_NOTES: Good fix
CONCERNS: None
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        
        fixes_data = [
            (sample_fix, sample_issue, sample_file_content),
            (sample_fix, sample_issue, sample_file_content),
        ]

        results = validator.validate_fixes_batch(fixes_data)

        assert len(results) == 2
        assert all(r.status == ValidationStatus.APPROVED for r in results)

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fixes_batch_stop_on_rejection(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test stopping batch validation on rejection."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        
        # First call returns REJECTED
        response1 = MagicMock()
        response1.choices[0].message.content = """
STATUS: REJECTED
CONFIDENCE: 0.3
VALIDATION_NOTES: Bad fix
CONCERNS: Major issues
"""
        
        mock_client.chat.completions.create.return_value = response1
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        
        fixes_data = [
            (sample_fix, sample_issue, sample_file_content),
            (sample_fix, sample_issue, sample_file_content),
        ]

        results = validator.validate_fixes_batch(fixes_data, stop_on_rejection=True)

        # Should stop after first rejection
        assert len(results) == 1
        assert results[0].status == ValidationStatus.REJECTED


class TestConvenienceFunction:
    """Test validate_fixes_with_agent convenience function."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fixes_with_agent(self, mock_openai, sample_fix, sample_issue, tmp_path):
        """Test convenience function for validating fixes."""
        from devdox_ai_sonar.fix_validator import validate_fixes_with_agent, ValidationStatus
        
        # Create test file
        test_file = tmp_path / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def test(): pass")

        sample_fix.file_path = "src/test.py"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
STATUS: APPROVED
CONFIDENCE: 0.9
VALIDATION_NOTES: Good
CONCERNS: None
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        results = validate_fixes_with_agent(
            [sample_fix],
            [sample_issue],
            tmp_path,
            provider="openai",
            api_key="test-key"
        )

        assert len(results) == 1


class TestErrorHandling:
    """Test error handling in validation."""

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_file_not_found(self, mock_openai, sample_fix, sample_issue):
        """Test validation when file doesn't exist."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_openai.OpenAI.return_value = MagicMock()
        validator = FixValidator(provider="openai", api_key="test-key")

        # File content is required parameter
        result = validator.validate_fix(sample_fix, sample_issue, "")

        # Should handle gracefully
        assert result is not None

    @patch("devdox_ai_sonar.fix_validator.openai")
    def test_validate_fix_llm_error(self, mock_openai, sample_fix, sample_issue, sample_file_content):
        """Test validation when LLM call fails."""
        from devdox_ai_sonar.fix_validator import FixValidator, ValidationStatus
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client

        validator = FixValidator(provider="openai", api_key="test-key")
        result = validator.validate_fix(sample_fix, sample_issue, sample_file_content)

        assert result.status == ValidationStatus.NEEDS_REVIEW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
