"""
Input validation utilities for devdox-ai-sonar.

This module provides validation functions for user inputs to ensure
data integrity and provide clear error messages.
"""

from typing import Union, List, Set
from pathlib import Path
from enum import Enum

from devdox_ai_sonar.utils.exceptions import ValidationError


class IssueType(Enum):
    """Type of issues to process."""

    REGULAR = "regular"
    SECURITY = "security"


class InputValidator:
    """Validates user inputs with clear error messages."""

    # Constants for validation
    VALID_ISSUE_TYPES: Set[str] = {"BUG", "VULNERABILITY", "CODE_SMELL"}
    VALID_SEVERITIES: Set[str] = {"BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"}

    # Git branch name invalid characters
    INVALID_BRANCH_CHARS: Set[str] = {
        "~",
        "^",
        ":",
        "?",
        "*",
        "[",
        "\\",
        " ",
        "\t",
        "\n",
    }

    @staticmethod
    def validate_pull_request_number(value: Union[str, int]) -> int:
        """
        Validate pull request number is a positive integer.

        Args:
            value: Pull request number to validate

        Returns:
            Validated PR number as integer

        Raises:
            ValidationError: If value is not a positive integer
        """
        try:
            pr_num = int(value)
            if pr_num <= 0:
                raise ValidationError(
                    "Pull request number must be positive",
                    field="pull_request",
                    value=str(value),
                )
            return pr_num
        except (ValueError, TypeError) as e:
            raise ValidationError(
                "Pull request number must be a valid integer",
                field="pull_request",
                value=str(value),
            ) from e

    @staticmethod
    def validate_branch_name(branch: str) -> str:
        """
        Validate Git branch name.

        Args:
            branch: Branch name to validate

        Returns:
            Validated and trimmed branch name

        Raises:
            ValidationError: If branch name is invalid
        """
        if not branch or not branch.strip():
            raise ValidationError("Branch name cannot be empty", field="branch")

        branch = branch.strip()

        # Check for invalid characters
        invalid_found = [
            char for char in InputValidator.INVALID_BRANCH_CHARS if char in branch
        ]
        if invalid_found:
            raise ValidationError(
                f"Branch name contains invalid characters: {', '.join(invalid_found)}",
                field="branch",
                value=branch,
            )

        # Check for invalid patterns
        if branch.startswith(".") or branch.endswith("."):
            raise ValidationError(
                "Branch name cannot start or end with a dot",
                field="branch",
                value=branch,
            )

        if ".." in branch:
            raise ValidationError(
                "Branch name cannot contain consecutive dots",
                field="branch",
                value=branch,
            )

        return branch

    @staticmethod
    def validate_max_issues(
        value: Union[str, int], max_limit: int, field_name: str = "max_issues"
    ) -> int:
        """
        Validate maximum number of issues to fetch.

        Args:
            value: Value to validate
            max_limit: Maximum allowed value
            field_name: Name of the field (for error messages)

        Returns:
            Validated count (capped at max_limit if necessary)

        Raises:
            ValidationError: If value is not a positive integer
        """
        try:
            count = int(value)
            if count <= 0:
                raise ValidationError(
                    f"{field_name} must be positive", field=field_name, value=str(value)
                )

            # Cap at max_limit rather than raising error
            if count > max_limit:
                return max_limit

            return count
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"{field_name} must be a valid integer",
                field=field_name,
                value=str(value),
            ) from e

    @staticmethod
    def validate_issue_types(types: Union[str, List[str]]) -> List[str]:
        """
        Validate issue types are valid SonarCloud types.

        Args:
            types: Issue types to validate (comma-separated string or list)

        Returns:
            List of validated issue types

        Raises:
            ValidationError: If any type is invalid
        """
        if isinstance(types, str):
            types_list = [t.strip().upper() for t in types.split(",") if t.strip()]
        else:
            types_list = [t.strip().upper() for t in types if t.strip()]

        invalid = set(types_list) - InputValidator.VALID_ISSUE_TYPES
        if invalid:
            raise ValidationError(
                f"Invalid issue types: {', '.join(invalid)}. "
                f"Valid types: {', '.join(InputValidator.VALID_ISSUE_TYPES)}",
                field="types",
                value=", ".join(invalid),
            )

        return types_list

    @staticmethod
    def validate_severities(severities: Union[str, List[str]]) -> List[str]:
        """
        Validate severities are valid SonarCloud severities.

        Args:
            severities: Severities to validate (comma-separated string or list)

        Returns:
            List of validated severities

        Raises:
            ValidationError: If any severity is invalid
        """
        if isinstance(severities, str):
            sev_list = [s.strip().upper() for s in severities.split(",") if s.strip()]
        else:
            sev_list = [s.strip().upper() for s in severities if s.strip()]

        invalid = set(sev_list) - InputValidator.VALID_SEVERITIES
        if invalid:
            raise ValidationError(
                f"Invalid severities: {', '.join(invalid)}. "
                f"Valid severities: {', '.join(InputValidator.VALID_SEVERITIES)}",
                field="severities",
                value=", ".join(invalid),
            )

        return sev_list

    @staticmethod
    def validate_project_path(path: Union[str, Path]) -> Path:
        """
        Validate project path exists and is a directory.

        Args:
            path: Project path to validate

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid
        """
        project_path = Path(path) if isinstance(path, str) else path

        if not project_path.exists():
            raise ValidationError(
                f"Project path does not exist: {project_path}",
                field="project_path",
                value=str(project_path),
            )

        if not project_path.is_dir():
            raise ValidationError(
                f"Project path is not a directory: {project_path}",
                field="project_path",
                value=str(project_path),
            )

        return project_path

    @staticmethod
    def validate_api_token(token: str, provider: str = "SonarCloud") -> str:
        """
        Validate API token format.

        Args:
            token: Token to validate
            provider: Provider name (for error messages)

        Returns:
            Validated token

        Raises:
            ValidationError: If token format is invalid
        """
        if not token or not token.strip():
            raise ValidationError(f"{provider} token cannot be empty", field="token")

        token = token.strip()

        # Basic validation - tokens should be reasonably long
        if len(token) < 20:
            raise ValidationError(
                f"{provider} token appears too short (expected at least 20 characters)",
                field="token",
                value=f"{token[:10]}... ({len(token)} chars)",
            )

        return token

    @staticmethod
    def validate_confidence(confidence: float) -> float:
        """
        Validate confidence score is between 0 and 1.

        Args:
            confidence: Confidence score to validate

        Returns:
            Validated confidence score

        Raises:
            ValidationError: If confidence is out of range
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError(
                f"Confidence must be between 0.0 and 1.0, got {confidence}",
                field="confidence",
                value=str(confidence),
            )

        return confidence

    @staticmethod
    def validate_model_name(model: str, provider: str) -> str:
        """
        Validate model name is not empty.

        Args:
            model: Model name to validate
            provider: Provider name (for error messages)

        Returns:
            Validated model name

        Raises:
            ValidationError: If model name is invalid
        """
        if not model or not model.strip():
            raise ValidationError(
                f"Model name cannot be empty for provider {provider}", field="model"
            )

        return model.strip()
