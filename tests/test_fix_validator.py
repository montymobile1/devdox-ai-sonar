"""
Comprehensive unit tests for fix_validator.py

Test Coverage:
- ValidationStatus enum
- ValidationResult class
- FixValidator class
- Helper functions
- Edge cases and error handling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from devdox_ai_sonar.fix_validator import (
    ValidationStatus,
    ValidationResult,
    FixValidator,
    validate_fixes_with_agent
)
from devdox_ai_sonar.models import FixSuggestion, SonarIssue, Severity, IssueType


class TestValidationStatus(unittest.TestCase):
    """Test ValidationStatus enum."""

    def test_validation_status_values(self):
        """Test all validation status values exist."""
        self.assertEqual(ValidationStatus.APPROVED.value, "APPROVED")
        self.assertEqual(ValidationStatus.MODIFIED.value, "MODIFIED")
        self.assertEqual(ValidationStatus.REJECTED.value, "REJECTED")
        self.assertEqual(ValidationStatus.NEEDS_REVIEW.value, "NEEDS_REVIEW")

    def test_validation_status_membership(self):
        """Test membership checks."""
        statuses = [ValidationStatus.APPROVED, ValidationStatus.MODIFIED,
                    ValidationStatus.REJECTED, ValidationStatus.NEEDS_REVIEW]
        self.assertEqual(len(statuses), 4)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult class."""

    def setUp(self):
        """Set up test fixtures."""
        self.fix = FixSuggestion(
            issue_key="TEST-123",
            original_code="x = 1",
            fixed_code="x = 2",
            explanation="Fix value",
            confidence=0.9,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=10
        )

    def test_validation_result_approved(self):
        """Test approved validation result."""
        result = ValidationResult(
            status=ValidationStatus.APPROVED,
            original_fix=self.fix,
            validation_notes="Good fix",
            confidence=0.95
        )

        self.assertEqual(result.status, ValidationStatus.APPROVED)
        self.assertEqual(result.final_fix, self.fix)
        self.assertTrue(result.should_apply)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.validation_notes, "Good fix")

    def test_validation_result_modified(self):
        """Test modified validation result."""
        modified_fix = FixSuggestion(
            issue_key="TEST-123",
            original_code="x = 1",
            fixed_code="x = calculate_value()",
            explanation="Better approach",
            confidence=0.95,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=10
        )

        result = ValidationResult(
            status=ValidationStatus.MODIFIED,
            original_fix=self.fix,
            modified_fix=modified_fix,
            validation_notes="Improved fix",
            confidence=0.9
        )

        self.assertEqual(result.status, ValidationStatus.MODIFIED)
        self.assertEqual(result.final_fix, modified_fix)
        self.assertTrue(result.should_apply)

    def test_validation_result_rejected(self):
        """Test rejected validation result."""
        result = ValidationResult(
            status=ValidationStatus.REJECTED,
            original_fix=self.fix,
            validation_notes="Unsafe fix",
            concerns=["Security risk", "Logic error"],
            confidence=0.2
        )

        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertFalse(result.should_apply)
        self.assertIn("Security risk", result.concerns)
        self.assertIn("Logic error", result.concerns)

    def test_validation_result_needs_review(self):
        """Test needs review validation result."""
        result = ValidationResult(
            status=ValidationStatus.NEEDS_REVIEW,
            original_fix=self.fix,
            validation_notes="Uncertain",
            confidence=0.5
        )

        self.assertEqual(result.status, ValidationStatus.NEEDS_REVIEW)
        self.assertFalse(result.should_apply)

    def test_should_apply_logic(self):
        """Test should_apply property logic."""
        # Should apply for APPROVED
        result_approved = ValidationResult(
            status=ValidationStatus.APPROVED,
            original_fix=self.fix,
            confidence=0.9
        )
        self.assertTrue(result_approved.should_apply)

        # Should apply for MODIFIED
        result_modified = ValidationResult(
            status=ValidationStatus.MODIFIED,
            original_fix=self.fix,
            confidence=0.9
        )
        self.assertTrue(result_modified.should_apply)

        # Should NOT apply for REJECTED
        result_rejected = ValidationResult(
            status=ValidationStatus.REJECTED,
            original_fix=self.fix,
            confidence=0.2
        )
        self.assertFalse(result_rejected.should_apply)

        # Should NOT apply for NEEDS_REVIEW
        result_review = ValidationResult(
            status=ValidationStatus.NEEDS_REVIEW,
            original_fix=self.fix,
            confidence=0.5
        )
        self.assertFalse(result_review.should_apply)


class TestFixValidatorInitialization(unittest.TestCase):
    """Test FixValidator initialization."""

    @patch('devdox_ai_sonar.fix_validator.openai.OpenAI')
    def test_init_openai(self, mock_openai):
        """Test initialization with OpenAI provider."""
        validator = FixValidator(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            min_confidence_threshold=0.8
        )

        self.assertEqual(validator.provider, "openai")
        self.assertEqual(validator.model, "gpt-4o")
        self.assertEqual(validator.min_confidence_threshold, 0.8)
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch('devdox_ai_sonar.fix_validator.HAS_OPENAI', False)
    def test_init_openai_missing_library(self):
        """Test initialization fails when OpenAI library missing."""
        with self.assertRaises(ImportError) as context:
            FixValidator(provider="openai")
        self.assertIn("OpenAI library not installed", str(context.exception))

    @patch('devdox_ai_sonar.fix_validator.openai.OpenAI')
    @patch('devdox_ai_sonar.fix_validator.os.getenv')
    def test_init_openai_env_api_key(self, mock_getenv, mock_openai):
        """Test initialization with API key from environment."""
        mock_getenv.return_value = "env-api-key"

        validator = FixValidator(provider="openai")

        self.assertEqual(validator.api_key, "env-api-key")
        mock_getenv.assert_called_with("OPENAI_API_KEY")

    @patch('devdox_ai_sonar.fix_validator.openai.OpenAI')
    def test_init_openai_missing_api_key(self, mock_openai):
        """Test initialization fails without API key."""
        with patch('devdox_ai_sonar.fix_validator.os.getenv', return_value=None):
            with self.assertRaises(ValueError) as context:
                FixValidator(provider="openai")
            self.assertIn("API key not provided", str(context.exception))

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider."""
        with self.assertRaises(ValueError) as context:
            FixValidator(provider="invalid_provider", api_key="test")
        self.assertIn("Unsupported provider", str(context.exception))


class TestFixValidatorExtractContext(unittest.TestCase):
    """Test context extraction."""

    def setUp(self):
        """Set up test validator."""
        with patch('devdox_ai_sonar.fix_validator.openai.OpenAI'):
            self.validator = FixValidator(provider="openai", api_key="test")

    def test_extract_context_basic(self):
        """Test basic context extraction."""
        file_content = "\n".join([
            "line 1",
            "line 2",
            "line 3",
            "line 4",  # Problem line
            "line 5",
            "line 6",
            "line 7"
        ])

        context = self.validator._extract_validation_context(
            file_content, 4, 4, context_lines=2
        )

        self.assertIn("line 2", context["full_context"])
        self.assertIn("line 3", context["full_context"])
        self.assertIn("line 4", context["full_context"])
        self.assertIn("line 5", context["full_context"])
        self.assertIn("line 6", context["full_context"])
        self.assertEqual(context["problem_lines"], "line 4")
        self.assertEqual(context["issue_start"], 4)
        self.assertEqual(context["issue_end"], 4)

    def test_extract_context_at_file_start(self):
        """Test context extraction at start of file."""
        file_content = "\n".join(["line 1", "line 2", "line 3", "line 4"])

        context = self.validator._extract_validation_context(
            file_content, 1, 1, context_lines=5
        )

        self.assertEqual(context["start_line"], 1)
        self.assertIn("line 1", context["full_context"])

    def test_extract_context_at_file_end(self):
        """Test context extraction at end of file."""
        file_content = "\n".join(["line 1", "line 2", "line 3", "line 4"])

        context = self.validator._extract_validation_context(
            file_content, 4, 4, context_lines=5
        )

        self.assertIn("line 4", context["full_context"])
        self.assertEqual(context["end_line"], 4)

    def test_extract_context_multi_line_issue(self):
        """Test context extraction for multi-line issue."""
        file_content = "\n".join([
            "line 1",
            "line 2",
            "line 3",  # Issue start
            "line 4",
            "line 5",  # Issue end
            "line 6"
        ])

        context = self.validator._extract_validation_context(
            file_content, 3, 5, context_lines=1
        )

        self.assertIn("line 3", context["problem_lines"])
        self.assertIn("line 4", context["problem_lines"])
        self.assertIn("line 5", context["problem_lines"])


class TestFixValidatorPromptCreation(unittest.TestCase):
    """Test validation prompt creation."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('devdox_ai_sonar.fix_validator.openai.OpenAI'):
            self.validator = FixValidator(provider="openai", api_key="test")

        self.fix = FixSuggestion(
            issue_key="TEST-123",
            original_code="x = 1",
            fixed_code="x = 2",
            explanation="Update value",
            confidence=0.8,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=10
        )

        self.issue = SonarIssue(
            key="TEST-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="test.py",
            project="test-project",
            first_line=10,
            last_line=10,
            message="Fix this issue",
            type=IssueType.CODE_SMELL
        )


class TestFixValidatorResponseParsing(unittest.TestCase):
    """Test validation response parsing."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('devdox_ai_sonar.fix_validator.openai.OpenAI'):
            self.validator = FixValidator(provider="openai", api_key="test")

        self.fix = FixSuggestion(
            issue_key="TEST-123",
            original_code="x = 1",
            fixed_code="x = 2",
            explanation="Update value",
            confidence=0.8,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=10
        )

        self.issue = SonarIssue(
            key="TEST-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="test.py",
            project="test-project",
            first_line=10,
            last_line=10,
            message="Fix this issue",
            type=IssueType.CODE_SMELL
        )

    def test_parse_approved_response(self):
        """Test parsing approved validation response."""
        response_text = """
STATUS: APPROVED

CONFIDENCE: 0.95

VALIDATION_NOTES:
This fix correctly updates the value and follows best practices.

CONCERNS:
None
"""

        result = self.validator._parse_validation_response(
            response_text, self.fix, self.issue
        )

        self.assertEqual(result.status, ValidationStatus.APPROVED)
        self.assertEqual(result.confidence, 0.95)
        self.assertIn("correctly updates", result.validation_notes)
        self.assertEqual(len(result.concerns), 0)

    def test_parse_modified_response(self):
        """Test parsing modified validation response."""
        response_text = """
STATUS: MODIFIED

CONFIDENCE: 0.90

VALIDATION_NOTES:
The fix works but can be improved.

CONCERNS:
- Could use more descriptive variable name

IMPROVED_FIX:
```python
x = calculate_value()
```

IMPROVED_EXPLANATION:
Using a function is better than hardcoding the value.
"""

        result = self.validator._parse_validation_response(
            response_text, self.fix, self.issue
        )

        self.assertEqual(result.status, ValidationStatus.MODIFIED)
        self.assertEqual(result.confidence, 0.90)
        self.assertIsNotNone(result.modified_fix)
        self.assertIn("calculate_value", result.modified_fix.fixed_code)
        self.assertIn("descriptive variable name", result.concerns[0])

    def test_parse_rejected_response(self):
        """Test parsing rejected validation response."""
        response_text = """
STATUS: REJECTED

CONFIDENCE: 0.30

VALIDATION_NOTES:
This fix introduces a security vulnerability.

CONCERNS:
- SQL injection risk
- No input validation
- Breaks existing functionality
"""

        result = self.validator._parse_validation_response(
            response_text, self.fix, self.issue
        )

        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertEqual(result.confidence, 0.30)
        self.assertEqual(len(result.concerns), 3)
        self.assertIn("SQL injection", result.concerns[0])

    def test_parse_needs_review_response(self):
        """Test parsing needs review validation response."""
        response_text = """
STATUS: NEEDS_REVIEW

CONFIDENCE: 0.60

VALIDATION_NOTES:
Uncertain about this fix. Needs human review.

CONCERNS:
- Unclear impact on system
"""

        result = self.validator._parse_validation_response(
            response_text, self.fix, self.issue
        )

        self.assertEqual(result.status, ValidationStatus.NEEDS_REVIEW)
        self.assertFalse(result.should_apply)

    def test_parse_malformed_response(self):
        """Test parsing malformed response."""
        response_text = "This is not a valid response format"

        result = self.validator._parse_validation_response(
            response_text, self.fix, self.issue
        )

        # Should default to NEEDS_REVIEW on parse error
        self.assertEqual(result.status, ValidationStatus.NEEDS_REVIEW)

    def test_confidence_threshold_enforcement(self):
        """Test that confidence threshold is enforced."""
        validator = FixValidator(
            provider="openai",
            api_key="test",
            min_confidence_threshold=0.8
        )

        response_text = """
STATUS: APPROVED

CONFIDENCE: 0.70

VALIDATION_NOTES:
Good fix but below threshold.

CONCERNS:
None
"""

        result = validator._parse_validation_response(
            response_text, self.fix, self.issue
        )

        # Should be changed to NEEDS_REVIEW due to low confidence
        self.assertEqual(result.status, ValidationStatus.NEEDS_REVIEW)
        self.assertIn("below required threshold", result.validation_notes)


class TestFixValidatorLLMCalls(unittest.TestCase):
    """Test LLM API calls."""

    @patch('devdox_ai_sonar.fix_validator.openai.OpenAI')
    def test_call_llm_openai_success(self, mock_openai_class):
        """Test successful OpenAI API call."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "STATUS: APPROVED\nCONFIDENCE: 0.9"
        mock_client.chat.completions.create.return_value = mock_response

        validator = FixValidator(provider="openai", api_key="test")
        result = validator._call_llm_validator("test prompt")

        self.assertIsNotNone(result)
        self.assertIn("APPROVED", result)
        mock_client.chat.completions.create.assert_called_once()

    @patch('devdox_ai_sonar.fix_validator.openai.OpenAI')
    def test_call_llm_openai_error(self, mock_openai_class):
        """Test OpenAI API call with error."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        validator = FixValidator(provider="openai", api_key="test")
        result = validator._call_llm_validator("test prompt")

        self.assertIsNone(result)


class TestFixValidatorValidateFix(unittest.TestCase):
    """Test fix validation end-to-end."""

    def setUp(self):
        """Set up test fixtures."""
        self.fix = FixSuggestion(
            issue_key="TEST-123",
            original_code="x = 1",
            fixed_code="x = 2",
            explanation="Update value",
            confidence=0.8,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=10
        )

        self.issue = SonarIssue(
            key="TEST-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="test.py",
            project="test-project",
            first_line=10,
            last_line=10,
            message="Fix this issue",
            type=IssueType.CODE_SMELL
        )

        self.file_content = "\n".join([
            "def foo():",
            "    x = 1",
            "    y = 2",
            "    return x + y"
        ])

    @patch('devdox_ai_sonar.fix_validator.openai.OpenAI')
    def test_validate_fix_success(self, mock_openai):
        """Test successful fix validation."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
STATUS: APPROVED
CONFIDENCE: 0.95
VALIDATION_NOTES: Good fix
CONCERNS: None
"""
        mock_client.chat.completions.create.return_value = mock_response

        validator = FixValidator(provider="openai", api_key="test")
        result = validator.validate_fix(self.fix, self.issue, self.file_content)

        self.assertEqual(result.status, ValidationStatus.APPROVED)
        self.assertTrue(result.should_apply)

    @patch('devdox_ai_sonar.fix_validator.openai.OpenAI')
    def test_validate_fix_llm_failure(self, mock_openai):
        """Test fix validation when LLM call fails."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        validator = FixValidator(provider="openai", api_key="test")
        result = validator.validate_fix(self.fix, self.issue, self.file_content)

        self.assertEqual(result.status, ValidationStatus.NEEDS_REVIEW)
        self.assertFalse(result.should_apply)


class TestValidateFixesWithAgent(unittest.TestCase):
    """Test the convenience function for batch validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.fix = FixSuggestion(
            issue_key="TEST-123",
            original_code="x = 1",
            fixed_code="x = 2",
            explanation="Update value",
            confidence=0.8,
            llm_model="gpt-4",
            file_path="test.py",
            line_number=10
        )

        self.issue = SonarIssue(
            key="TEST-123",
            rule="python:S1234",
            severity=Severity.MAJOR,
            component="test.py",
            project="test-project",
            first_line=10,
            last_line=10,
            message="Fix this issue",
            type=IssueType.CODE_SMELL
        )

    @patch('devdox_ai_sonar.fix_validator.FixValidator')
    @patch('builtins.open', new_callable=mock_open, read_data="x = 1\ny = 2")
    @patch('pathlib.Path.exists')
    def test_validate_fixes_with_agent(self, mock_exists, mock_file, mock_validator_class):
        """Test batch validation with agent."""
        mock_exists.return_value = True
        mock_validator = MagicMock()
        mock_validator_class.return_value = mock_validator

        mock_result = ValidationResult(
            status=ValidationStatus.APPROVED,
            original_fix=self.fix,
            confidence=0.9
        )
        mock_validator.validate_fix.return_value = mock_result

        project_path = Path("/test/project")
        results = validate_fixes_with_agent(
            fixes=[self.fix],
            issues=[self.issue],
            project_path=project_path,
            provider="openai",
            api_key="test"
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, ValidationStatus.APPROVED)

    @patch('devdox_ai_sonar.fix_validator.FixValidator')
    @patch('pathlib.Path.exists')
    def test_validate_fixes_file_not_found(self, mock_exists, mock_validator_class):
        """Test batch validation when file not found."""
        mock_exists.return_value = False

        project_path = Path("/test/project")
        results = validate_fixes_with_agent(
            fixes=[self.fix],
            issues=[self.issue],
            project_path=project_path,
            provider="openai",
            api_key="test"
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, ValidationStatus.NEEDS_REVIEW)
        self.assertIn("File not found", results[0].validation_notes)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('devdox_ai_sonar.fix_validator.openai.OpenAI'):
            self.validator = FixValidator(provider="openai", api_key="test")

    def test_empty_file_content(self):
        """Test validation with empty file content."""
        context = self.validator._extract_validation_context("", 1, 1, 10)
        self.assertEqual(context["problem_lines"], "")

    def test_very_large_context(self):
        """Test validation with very large context."""
        large_content = "\n".join([f"line {i}" for i in range(10000)])
        context = self.validator._extract_validation_context(
            large_content, 5000, 5000, 100
        )
        self.assertIsNotNone(context)

    def test_unicode_content(self):
        """Test validation with unicode content."""
        unicode_content = "x = '你好世界'\ny = 'Hello 🌍'"
        context = self.validator._extract_validation_context(
            unicode_content, 1, 2, 5
        )
        self.assertIn("你好世界", context["full_context"])
        self.assertIn("🌍", context["full_context"])

    def test_confidence_clamping(self):
        """Test that confidence values are clamped to [0, 1]."""
        fix = FixSuggestion(
            issue_key="TEST",
            original_code="x=1",
            fixed_code="x=2",
            explanation="test",
            confidence=0.8,
            llm_model="gpt-4"
        )

        # Test confidence > 1.0
        response_text = "STATUS: APPROVED\nCONFIDENCE: 1.5\nVALIDATION_NOTES: test\nCONCERNS: None"
        result = self.validator._parse_validation_response(response_text, fix, Mock())
        self.assertLessEqual(result.confidence, 1.0)

        # Test confidence < 0.0
        response_text = "STATUS: REJECTED\nCONFIDENCE: -0.5\nVALIDATION_NOTES: test\nCONCERNS: None"
        result = self.validator._parse_validation_response(response_text, fix, Mock())
        self.assertGreaterEqual(result.confidence, 0.0)


if __name__ == '__main__':
    unittest.main()