"""
Custom exceptions for devdox-ai-sonar.

This module defines a hierarchy of exceptions for better error handling
and more informative error messages throughout the application.
"""

from typing import Optional


class DevDoxSonarError(Exception):
    """Base exception for all devdox-ai-sonar errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class AuthenticationError(DevDoxSonarError):
    """Raised when authentication with SonarCloud fails."""

    def __init__(self, message: str = "Authentication failed", details: Optional[str] = None):
        super().__init__(message, details)


class ConfigurationError(DevDoxSonarError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        details = f"Field: {field}" if field else None
        super().__init__(message, details)


class SonarCloudAPIError(DevDoxSonarError):
    """Raised when SonarCloud API returns an error."""

    def __init__(
            self,
            message: str,
            status_code: Optional[int] = None,
            response_body: Optional[str] = None
    ):
        self.status_code = status_code
        self.response_body = response_body

        details = []
        if status_code:
            details.append(f"Status Code: {status_code}")
        if response_body:
            details.append(f"Response: {response_body[:200]}")

        super().__init__(message, "\n".join(details) if details else None)


class FixApplicationError(DevDoxSonarError):
    """Raised when fix application fails."""

    def __init__(
            self,
            message: str,
            file_path: str,
            original_error: Optional[Exception] = None
    ):
        self.file_path = file_path
        self.original_error = original_error

        details = f"File: {file_path}"
        if original_error:
            details += f"\nOriginal Error: {str(original_error)}"

        super().__init__(message, details)


class LLMProviderError(DevDoxSonarError):
    """Raised when LLM provider encounters an error."""

    def __init__(
            self,
            message: str,
            provider: str,
            model: Optional[str] = None,
            original_error: Optional[Exception] = None
    ):
        self.provider = provider
        self.model = model
        self.original_error = original_error

        details = f"Provider: {provider}"
        if model:
            details += f"\nModel: {model}"
        if original_error:
            details += f"\nOriginal Error: {str(original_error)}"

        super().__init__(message, details)


class ValidationError(DevDoxSonarError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str, value: Optional[str] = None):
        self.field = field
        self.value = value

        details = f"Field: {field}"
        if value:
            details += f"\nValue: {value}"

        super().__init__(message, details)


class FileNotFoundError(DevDoxSonarError):
    """Raised when a required file is not found."""

    def __init__(self, file_path: str, context: Optional[str] = None):
        self.file_path = file_path
        message = f"File not found: {file_path}"
        super().__init__(message, context)


class RuleNotFoundError(DevDoxSonarError):
    """Raised when a SonarCloud rule is not found."""

    def __init__(self, rule_key: str):
        self.rule_key = rule_key
        message = f"SonarCloud rule not found: {rule_key}"
        super().__init__(message)



class SwitchCommandException(Exception):
    """Exception raised when user wants to switch commands."""
    pass


class ReturnToMenuException(Exception):
    """Exception raised when user wants to return to main menu."""
    pass
