

import pytest
from pathlib import Path
from typing import Union, List

from devdox_ai_sonar.utils.validator import InputValidator
from devdox_ai_sonar.utils.exceptions import ValidationError


# ============================================================================
# PULL REQUEST NUMBER VALIDATION TESTS
# ============================================================================


class TestValidatePullRequestNumber:
    """Tests for validate_pull_request_number method."""

    def test_validate_pull_request_number_valid_int(self):
        """Test validation with valid integer."""
        result = InputValidator.validate_pull_request_number(42)

        assert result == 42
        assert isinstance(result, int)

    def test_validate_pull_request_number_valid_string(self):
        """Test validation with valid string."""
        result = InputValidator.validate_pull_request_number("123")

        assert result == 123
        assert isinstance(result, int)

    def test_validate_pull_request_number_zero(self):
        """Test validation with zero (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_pull_request_number(0)

        assert "positive" in str(exc_info.value).lower()
        assert exc_info.value.field == "pull_request"

    def test_validate_pull_request_number_negative(self):
        """Test validation with negative number."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_pull_request_number(-5)

        assert "positive" in str(exc_info.value).lower()

    def test_validate_pull_request_number_invalid_string(self):
        """Test validation with non-numeric string."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_pull_request_number("abc")

        assert "valid integer" in str(exc_info.value).lower()
        assert exc_info.value.field == "pull_request"

    def test_validate_pull_request_number_float_string(self):
        """Test validation with float string."""
        with pytest.raises(ValidationError):
            InputValidator.validate_pull_request_number("12.5")

    def test_validate_pull_request_number_none(self):
        """Test validation with None."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_pull_request_number(None)

        assert "valid integer" in str(exc_info.value).lower()

    def test_validate_pull_request_number_very_large(self):
        """Test validation with very large number."""
        result = InputValidator.validate_pull_request_number(999999)

        assert result == 999999


# ============================================================================
# BRANCH NAME VALIDATION TESTS
# ============================================================================


class TestValidateBranchName:
    """Tests for validate_branch_name method."""

    def test_validate_branch_name_valid_simple(self):
        """Test validation with simple valid branch name."""
        result = InputValidator.validate_branch_name("main")

        assert result == "main"

    def test_validate_branch_name_valid_with_slashes(self):
        """Test validation with slashes (feature branches)."""
        result = InputValidator.validate_branch_name("feature/new-feature")

        assert result == "feature/new-feature"

    def test_validate_branch_name_valid_with_hyphens(self):
        """Test validation with hyphens."""
        result = InputValidator.validate_branch_name("fix-bug-123")

        assert result == "fix-bug-123"

    def test_validate_branch_name_valid_with_underscores(self):
        """Test validation with underscores."""
        result = InputValidator.validate_branch_name("release_v1.0")

        assert result == "release_v1.0"

    def test_validate_branch_name_trims_whitespace(self):
        """Test that whitespace is trimmed."""
        result = InputValidator.validate_branch_name("  main  ")

        assert result == "main"

    def test_validate_branch_name_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name("")

        assert "cannot be empty" in str(exc_info.value).lower()
        assert exc_info.value.field == "branch"

    def test_validate_branch_name_whitespace_only(self):
        """Test validation with whitespace only."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name("   ")

        assert "cannot be empty" in str(exc_info.value).lower()

    def test_validate_branch_name_with_space(self):
        """Test validation with space (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name("feature branch")

        assert "invalid characters" in str(exc_info.value).lower()
        assert exc_info.value.field == "branch"

    def test_validate_branch_name_with_tilde(self):
        """Test validation with tilde (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name("feature~1")

        assert "invalid characters" in str(exc_info.value).lower()

    def test_validate_branch_name_with_caret(self):
        """Test validation with caret (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature^2")

    def test_validate_branch_name_with_colon(self):
        """Test validation with colon (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature:test")

    def test_validate_branch_name_with_question_mark(self):
        """Test validation with question mark (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature?")

    def test_validate_branch_name_with_asterisk(self):
        """Test validation with asterisk (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature*")

    def test_validate_branch_name_with_bracket(self):
        """Test validation with bracket (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature[test]")

    def test_validate_branch_name_with_backslash(self):
        """Test validation with backslash (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature\\test")

    def test_validate_branch_name_with_tab(self):
        """Test validation with tab (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature\ttest")

    def test_validate_branch_name_with_newline(self):
        """Test validation with newline (invalid)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_branch_name("feature\ntest")

    def test_validate_branch_name_starts_with_dot(self):
        """Test validation with branch starting with dot (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name(".feature")

        assert "cannot start or end with a dot" in str(exc_info.value).lower()

    def test_validate_branch_name_ends_with_dot(self):
        """Test validation with branch ending with dot (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name("feature.")

        assert "cannot start or end with a dot" in str(exc_info.value).lower()

    def test_validate_branch_name_consecutive_dots(self):
        """Test validation with consecutive dots (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name("feature..test")

        assert "consecutive dots" in str(exc_info.value).lower()

    def test_validate_branch_name_single_dot(self):
        """Test validation with single dot (valid)."""
        result = InputValidator.validate_branch_name("v1.0.0")

        assert result == "v1.0.0"


# ============================================================================
# MAX ISSUES VALIDATION TESTS
# ============================================================================


class TestValidateMaxIssues:
    """Tests for validate_max_issues method."""

    def test_validate_max_issues_valid_int(self):
        """Test validation with valid integer."""
        result = InputValidator.validate_max_issues(50, max_limit=100)

        assert result == 50

    def test_validate_max_issues_valid_string(self):
        """Test validation with valid string."""
        result = InputValidator.validate_max_issues("75", max_limit=100)

        assert result == 75

    def test_validate_max_issues_at_limit(self):
        """Test validation at max limit."""
        result = InputValidator.validate_max_issues(100, max_limit=100)

        assert result == 100

    def test_validate_max_issues_exceeds_limit(self):
        """Test validation exceeding limit (should cap)."""
        result = InputValidator.validate_max_issues(150, max_limit=100)

        assert result == 100  # Capped

    def test_validate_max_issues_zero(self):
        """Test validation with zero (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_max_issues(0, max_limit=100)

        assert "must be positive" in str(exc_info.value).lower()

    def test_validate_max_issues_negative(self):
        """Test validation with negative number."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_max_issues(-10, max_limit=100)

        assert "must be positive" in str(exc_info.value).lower()

    def test_validate_max_issues_invalid_string(self):
        """Test validation with invalid string."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_max_issues("abc", max_limit=100)

        assert "valid integer" in str(exc_info.value).lower()

    def test_validate_max_issues_custom_field_name(self):
        """Test validation with custom field name."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_max_issues(-5, max_limit=100, field_name="custom_field")

        assert "custom_field" in str(exc_info.value).lower()
        assert exc_info.value.field == "custom_field"

    def test_validate_max_issues_one(self):
        """Test validation with minimum valid value."""
        result = InputValidator.validate_max_issues(1, max_limit=100)

        assert result == 1


# ============================================================================
# ISSUE TYPES VALIDATION TESTS
# ============================================================================


class TestValidateIssueTypes:
    """Tests for validate_issue_types method."""

    def test_validate_issue_types_valid_list(self):
        """Test validation with valid list."""
        result = InputValidator.validate_issue_types(["BUG", "CODE_SMELL"])

        assert result == ["BUG", "CODE_SMELL"]

    def test_validate_issue_types_valid_string(self):
        """Test validation with comma-separated string."""
        result = InputValidator.validate_issue_types("BUG,CODE_SMELL,VULNERABILITY")

        assert result == ["BUG", "CODE_SMELL", "VULNERABILITY"]

    def test_validate_issue_types_lowercase_normalized(self):
        """Test that lowercase is normalized to uppercase."""
        result = InputValidator.validate_issue_types("bug,code_smell")

        assert result == ["BUG", "CODE_SMELL"]

    def test_validate_issue_types_with_whitespace(self):
        """Test validation with whitespace (trimmed)."""
        result = InputValidator.validate_issue_types("  BUG  ,  CODE_SMELL  ")

        assert result == ["BUG", "CODE_SMELL"]

    def test_validate_issue_types_single_type(self):
        """Test validation with single type."""
        result = InputValidator.validate_issue_types("VULNERABILITY")

        assert result == ["VULNERABILITY"]

    def test_validate_issue_types_all_valid_types(self):
        """Test validation with all valid types."""
        result = InputValidator.validate_issue_types(
            "BUG,VULNERABILITY,CODE_SMELL"
        )

        assert len(result) == 3
        assert set(result) == InputValidator.VALID_ISSUE_TYPES

    def test_validate_issue_types_invalid_type(self):
        """Test validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_issue_types("BUG,INVALID_TYPE")

        assert "invalid issue types" in str(exc_info.value).lower()
        assert "INVALID_TYPE" in str(exc_info.value)

    def test_validate_issue_types_multiple_invalid(self):
        """Test validation with multiple invalid types."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_issue_types("INVALID1,INVALID2")

        assert "invalid issue types" in str(exc_info.value).lower()

    def test_validate_issue_types_empty_string(self):
        """Test validation with empty string."""
        result = InputValidator.validate_issue_types("")

        assert result == []

    def test_validate_issue_types_empty_list(self):
        """Test validation with empty list."""
        result = InputValidator.validate_issue_types([])

        assert result == []


# ============================================================================
# SEVERITIES VALIDATION TESTS
# ============================================================================


class TestValidateSeverities:
    """Tests for validate_severities method."""

    def test_validate_severities_valid_list(self):
        """Test validation with valid list."""
        result = InputValidator.validate_severities(["CRITICAL", "MAJOR"])

        assert result == ["CRITICAL", "MAJOR"]

    def test_validate_severities_valid_string(self):
        """Test validation with comma-separated string."""
        result = InputValidator.validate_severities("BLOCKER,CRITICAL,MAJOR")

        assert result == ["BLOCKER", "CRITICAL", "MAJOR"]

    def test_validate_severities_lowercase_normalized(self):
        """Test that lowercase is normalized to uppercase."""
        result = InputValidator.validate_severities("critical,major,minor")

        assert result == ["CRITICAL", "MAJOR", "MINOR"]

    def test_validate_severities_with_whitespace(self):
        """Test validation with whitespace (trimmed)."""
        result = InputValidator.validate_severities("  CRITICAL  ,  MAJOR  ")

        assert result == ["CRITICAL", "MAJOR"]

    def test_validate_severities_all_valid(self):
        """Test validation with all valid severities."""
        result = InputValidator.validate_severities(
            "BLOCKER,CRITICAL,MAJOR,MINOR,INFO"
        )

        assert len(result) == 5
        assert set(result) == InputValidator.VALID_SEVERITIES

    def test_validate_severities_invalid_severity(self):
        """Test validation with invalid severity."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_severities("CRITICAL,INVALID")

        assert "invalid severities" in str(exc_info.value).lower()
        assert "INVALID" in str(exc_info.value)

    def test_validate_severities_empty_string(self):
        """Test validation with empty string."""
        result = InputValidator.validate_severities("")

        assert result == []

    def test_validate_severities_empty_list(self):
        """Test validation with empty list."""
        result = InputValidator.validate_severities([])

        assert result == []


# ============================================================================
# PROJECT PATH VALIDATION TESTS
# ============================================================================


class TestValidateProjectPath:
    """Tests for validate_project_path method."""

    def test_validate_project_path_valid_directory(self, tmp_path):
        """Test validation with valid directory."""
        result = InputValidator.validate_project_path(tmp_path)

        assert result == tmp_path
        assert isinstance(result, Path)

    def test_validate_project_path_string_input(self, tmp_path):
        """Test validation with string input."""
        result = InputValidator.validate_project_path(str(tmp_path))

        assert result == tmp_path
        assert isinstance(result, Path)

    def test_validate_project_path_not_exists(self):
        """Test validation with non-existent path."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_project_path("/nonexistent/path")

        assert "does not exist" in str(exc_info.value).lower()
        assert exc_info.value.field == "project_path"

    def test_validate_project_path_is_file(self, tmp_path):
        """Test validation with file instead of directory."""
        file_path = tmp_path / "test.txt"
        file_path.touch()

        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_project_path(file_path)

        assert "not a directory" in str(exc_info.value).lower()

    def test_validate_project_path_current_directory(self):
        """Test validation with current directory."""
        result = InputValidator.validate_project_path(".")

        assert result == Path(".")
        assert result.exists()


# ============================================================================
# API TOKEN VALIDATION TESTS
# ============================================================================


class TestValidateApiToken:
    """Tests for validate_api_token method."""

    def test_validate_api_token_valid(self):
        """Test validation with valid token."""
        token = "squ_" + "a" * 40
        result = InputValidator.validate_api_token(token)

        assert result == token

    def test_validate_api_token_with_whitespace(self):
        """Test validation trims whitespace."""
        token = "squ_" + "a" * 40
        result = InputValidator.validate_api_token(f"  {token}  ")

        assert result == token

    def test_validate_api_token_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_api_token("")

        assert "cannot be empty" in str(exc_info.value).lower()
        assert exc_info.value.field == "token"

    def test_validate_api_token_whitespace_only(self):
        """Test validation with whitespace only."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_api_token("   ")

        assert "cannot be empty" in str(exc_info.value).lower()

    def test_validate_api_token_too_short(self):
        """Test validation with token too short."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_api_token("short")

        assert "too short" in str(exc_info.value).lower()
        assert "20 characters" in str(exc_info.value)

    def test_validate_api_token_exactly_20_chars(self):
        """Test validation with token of exactly 20 characters."""
        token = "a" * 20
        result = InputValidator.validate_api_token(token)

        assert result == token

    def test_validate_api_token_19_chars(self):
        """Test validation with 19 characters (too short)."""
        with pytest.raises(ValidationError):
            InputValidator.validate_api_token("a" * 19)

    def test_validate_api_token_custom_provider(self):
        """Test validation with custom provider name."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_api_token("short", provider="GitHub")

        assert "GitHub" in str(exc_info.value)


# ============================================================================
# LLM PROVIDER VALIDATION TESTS
# ============================================================================


class TestValidateConfidence:
    """Tests for validate_confidence method."""

    def test_validate_confidence_valid_zero(self):
        """Test validation with 0.0."""
        result = InputValidator.validate_confidence(0.0)

        assert result == 0.0

    def test_validate_confidence_valid_one(self):
        """Test validation with 1.0."""
        result = InputValidator.validate_confidence(1.0)

        assert result == 1.0

    def test_validate_confidence_valid_middle(self):
        """Test validation with middle value."""
        result = InputValidator.validate_confidence(0.75)

        assert result == 0.75

    def test_validate_confidence_below_zero(self):
        """Test validation with value below 0."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_confidence(-0.1)

        assert "between 0.0 and 1.0" in str(exc_info.value)
        assert exc_info.value.field == "confidence"

    def test_validate_confidence_above_one(self):
        """Test validation with value above 1."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_confidence(1.5)

        assert "between 0.0 and 1.0" in str(exc_info.value)

    def test_validate_confidence_very_small(self):
        """Test validation with very small value."""
        result = InputValidator.validate_confidence(0.0001)

        assert result == 0.0001

    def test_validate_confidence_very_close_to_one(self):
        """Test validation with value very close to 1."""
        result = InputValidator.validate_confidence(0.9999)

        assert result == 0.9999


# ============================================================================
# MODEL NAME VALIDATION TESTS
# ============================================================================


class TestValidateModelName:
    """Tests for validate_model_name method."""

    def test_validate_model_name_valid(self):
        """Test validation with valid model name."""
        result = InputValidator.validate_model_name("gpt-4", "openai")

        assert result == "gpt-4"

    def test_validate_model_name_with_whitespace(self):
        """Test validation trims whitespace."""
        result = InputValidator.validate_model_name("  gpt-4  ", "openai")

        assert result == "gpt-4"

    def test_validate_model_name_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_model_name("", "openai")

        assert "cannot be empty" in str(exc_info.value).lower()
        assert "openai" in str(exc_info.value)
        assert exc_info.value.field == "model"

    def test_validate_model_name_whitespace_only(self):
        """Test validation with whitespace only."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_model_name("   ", "gemini")

        assert "cannot be empty" in str(exc_info.value).lower()
        assert "gemini" in str(exc_info.value)

    def test_validate_model_name_complex_name(self):
        """Test validation with complex model name."""
        result = InputValidator.validate_model_name(
            "claude-3-opus-20240229",
            "anthropic"
        )

        assert result == "claude-3-opus-20240229"


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


class TestValidatorConstants:
    """Tests for InputValidator constants."""

    def test_valid_issue_types_constant(self):
        """Test VALID_ISSUE_TYPES constant."""
        assert "BUG" in InputValidator.VALID_ISSUE_TYPES
        assert "VULNERABILITY" in InputValidator.VALID_ISSUE_TYPES
        assert "CODE_SMELL" in InputValidator.VALID_ISSUE_TYPES
        assert len(InputValidator.VALID_ISSUE_TYPES) == 3

    def test_valid_severities_constant(self):
        """Test VALID_SEVERITIES constant."""
        assert "BLOCKER" in InputValidator.VALID_SEVERITIES
        assert "CRITICAL" in InputValidator.VALID_SEVERITIES
        assert "MAJOR" in InputValidator.VALID_SEVERITIES
        assert "MINOR" in InputValidator.VALID_SEVERITIES
        assert "INFO" in InputValidator.VALID_SEVERITIES
        assert len(InputValidator.VALID_SEVERITIES) == 5


    def test_invalid_branch_chars_constant(self):
        """Test INVALID_BRANCH_CHARS constant."""
        assert '~' in InputValidator.INVALID_BRANCH_CHARS
        assert '^' in InputValidator.INVALID_BRANCH_CHARS
        assert ':' in InputValidator.INVALID_BRANCH_CHARS
        assert '?' in InputValidator.INVALID_BRANCH_CHARS
        assert '*' in InputValidator.INVALID_BRANCH_CHARS
        assert ' ' in InputValidator.INVALID_BRANCH_CHARS


# ============================================================================
# EDGE CASES AND INTEGRATION TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_validate_branch_name_multiple_invalid_chars(self):
        """Test branch validation with multiple invalid characters."""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_branch_name("feat~ure:test")

        # Should report multiple invalid chars
        error_msg = str(exc_info.value)
        assert "invalid characters" in error_msg.lower()

    def test_validate_issue_types_mixed_case_list(self):
        """Test issue types with mixed case in list."""
        result = InputValidator.validate_issue_types(["bug", "CODE_SMELL", "VuLnErAbIlItY"])

        assert result == ["BUG", "CODE_SMELL", "VULNERABILITY"]

    def test_validate_severities_duplicate_values(self):
        """Test severities with duplicate values."""
        result = InputValidator.validate_severities("CRITICAL,MAJOR,CRITICAL")

        # Should handle duplicates gracefully
        assert "CRITICAL" in result
        assert "MAJOR" in result

    def test_validate_project_path_with_symlink(self, tmp_path):
        """Test validation with symbolic link."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()

        # Skip if OS doesn't support symlinks
        try:
            symlink = tmp_path / "link"
            symlink.symlink_to(real_dir)

            result = InputValidator.validate_project_path(symlink)
            assert result.exists()
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this system")

    def test_validation_error_attributes(self):
        """Test ValidationError contains expected attributes."""
        try:
            InputValidator.validate_pull_request_number(-1)
        except ValidationError as e:
            assert hasattr(e, 'field')
            assert hasattr(e, 'message')
            assert e.field == "pull_request"


class TestIntegration:
    """Integration tests for combined validation scenarios."""

