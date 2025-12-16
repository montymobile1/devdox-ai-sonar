"""
Comprehensive tests for devdox_ai_sonar/utils/exceptions.py
Target Coverage: 100%
"""

import pytest
from devdox_ai_sonar.utils.exceptions import (
    DevDoxSonarError,
    ValidationError,
    ConfigurationError,
    SonarCloudAPIError,
    LLMProviderError,
    FixApplicationError
)


class TestDevDoxSonarError:
    """Test base DevDoxSonarError exception"""

    def test_create_devdox_error(self):
        """Should create DevDoxSonarError with message"""
        error = DevDoxSonarError("Test error message")
        assert str(error) == "Test error message"

    def test_devdox_error_is_exception(self):
        """DevDoxSonarError should inherit from Exception"""
        error = DevDoxSonarError("test")
        assert isinstance(error, Exception)

    def test_raise_devdox_error(self):
        """Should be able to raise DevDoxSonarError"""
        with pytest.raises(DevDoxSonarError) as exc_info:
            raise DevDoxSonarError("Test error")
        assert "Test error" in str(exc_info.value)

    def test_devdox_error_empty_message(self):
        """Should handle empty message"""
        error = DevDoxSonarError("")
        assert str(error) == ""

    def test_devdox_error_with_special_characters(self):
        """Should handle special characters in message"""
        error = DevDoxSonarError("Error: αβγ 中文 🔥")
        assert "αβγ" in str(error)
        assert "中文" in str(error)


class TestValidationError:
    """Test ValidationError exception"""

    def test_create_validation_error(self):
        """Should create ValidationError"""
        error = ValidationError("Invalid input","test")
        assert "Invalid input" in str(error)

    def test_validation_error_inherits_devdox_error(self):
        """ValidationError should inherit from DevDoxSonarError"""
        error = ValidationError("test","test")
        assert isinstance(error, DevDoxSonarError)
        assert isinstance(error, Exception)

    def test_raise_validation_error(self):
        """Should be able to raise ValidationError"""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed","test")

    def test_catch_as_devdox_error(self):
        """Should be catchable as DevDoxSonarError"""
        with pytest.raises(DevDoxSonarError):
            raise ValidationError("test","test")

    def test_validation_error_with_field_name(self):
        """Should handle field name in message"""
        error = ValidationError("Invalid value for field 'username'", "username")
        assert "username" in str(error)

    def test_validation_error_detailed_message(self):
        """Should support detailed validation messages"""
        message = "Field 'age' must be between 0 and 120, got -5"
        error = ValidationError(message, "age")
        assert "age" in str(error)
        assert "-5" in str(error)


class TestConfigurationError:
    """Test ConfigurationError exception"""

    def test_create_configuration_error(self):
        """Should create ConfigurationError"""
        error = ConfigurationError("Config file not found")
        assert "Config file not found" in str(error)

    def test_configuration_error_inherits_devdox_error(self):
        """ConfigurationError should inherit from DevDoxSonarError"""
        error = ConfigurationError("test")
        assert isinstance(error, DevDoxSonarError)

    def test_raise_configuration_error(self):
        """Should be able to raise ConfigurationError"""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Missing configuration")

    def test_configuration_error_with_path(self):
        """Should handle path in error message"""
        error = ConfigurationError("Config not found at /path/to/config.toml")
        assert "/path/to/config.toml" in str(error)


class TestSonarCloudAPIError:
    """Test SonarCloudAPIError exception"""

    def test_create_sonarcloud_error(self):
        """Should create SonarCloudAPIError"""
        error = SonarCloudAPIError("API request failed")
        assert "API request failed" in str(error)

    def test_sonarcloud_error_inherits_devdox_error(self):
        """SonarCloudAPIError should inherit from DevDoxSonarError"""
        error = SonarCloudAPIError("test")
        assert isinstance(error, DevDoxSonarError)

    def test_raise_sonarcloud_error(self):
        """Should be able to raise SonarCloudAPIError"""
        with pytest.raises(SonarCloudAPIError):
            raise SonarCloudAPIError("SonarCloud API error")

    def test_sonarcloud_error_with_status_code(self):
        """Should handle HTTP status code in message"""
        error = SonarCloudAPIError("Request failed with status 404")
        assert "404" in str(error)

    def test_sonarcloud_error_with_details(self):
        """Should handle detailed error information"""
        error = SonarCloudAPIError("Invalid project key 'invalid-key'")
        assert "invalid-key" in str(error)


class TestLLMProviderError:
    """Test LLMProviderError exception"""

    def test_create_llm_error(self):
        """Should create LLMProviderError"""
        error = LLMProviderError("Model not available", "gpt-4")
        assert "Model not available" in str(error)

    def test_llm_error_inherits_devdox_error(self):
        """LLMProviderError should inherit from DevDoxSonarError"""
        error = LLMProviderError("test", "test")
        assert isinstance(error, DevDoxSonarError)

    def test_raise_llm_error(self):
        """Should be able to raise LLMProviderError"""
        with pytest.raises(LLMProviderError):
            raise LLMProviderError("LLM API error", "gpt-4")

    def test_llm_error_with_provider(self):
        """Should handle provider name in message"""
        error = LLMProviderError("OpenAI API rate limit exceeded", "openai")
        assert "OpenAI" in str(error)

    def test_llm_error_with_model(self):
        """Should handle model name in message"""
        error = LLMProviderError("Model 'gpt-4' not found", "openai")
        assert "gpt-4" in str(error)


class TestFixApplicationError:
    """Test FixApplicationError exception"""

    def test_create_file_operation_error(self):
        """Should create FixApplicationError"""
        error = FixApplicationError("File not found", "/tmp/test.py")
        assert "File not found" in str(error)

    def test_file_operation_error_inherits_devdox_error(self):
        """FixApplicationError should inherit from DevDoxSonarError"""
        error = FixApplicationError("test","/tmp/test.py")
        assert isinstance(error, DevDoxSonarError)

    def test_raise_file_operation_error(self):
        """Should be able to raise FixApplicationError"""
        with pytest.raises(FixApplicationError):
            raise FixApplicationError("Cannot write file", "/tmp/test.py")

    def test_file_operation_error_with_path(self):
        """Should handle file path in message"""
        error = FixApplicationError("Cannot read /tmp/test.py", "/tmp/test.py")
        assert "/tmp/test.py" in str(error)

    def test_file_operation_error_permission_denied(self):
        """Should handle permission errors"""
        error = FixApplicationError("Permission denied: /etc/config", "/etc/config")
        assert "Permission denied" in str(error)


class TestExceptionInheritance:
    """Test exception inheritance chain"""

    def test_all_inherit_from_devdox_error(self):
        """All custom exceptions should inherit from DevDoxSonarError"""
        exceptions = [
            ValidationError("test","test"),
            ConfigurationError("test"),
            SonarCloudAPIError("test"),
            LLMProviderError("test","test"),
            FixApplicationError("test","tmp/test.py")
        ]

        for exc in exceptions:
            assert isinstance(exc, DevDoxSonarError)

    def test_all_inherit_from_exception(self):
        """All custom exceptions should inherit from Exception"""
        exceptions = [
            DevDoxSonarError("test"),
            ValidationError("test", "test"),
            ConfigurationError("test"),
            SonarCloudAPIError("test"),
            LLMProviderError("test","test"),
            FixApplicationError("test","tmp/test.py")
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)

    def test_inheritance_chain_validation_error(self):
        """Test complete inheritance chain for ValidationError"""
        error = ValidationError("test","test")
        assert isinstance(error, ValidationError)
        assert isinstance(error, DevDoxSonarError)
        assert isinstance(error, Exception)
        assert isinstance(error, BaseException)


class TestExceptionCatching:
    """Test exception catching patterns"""

    def test_catch_specific_exception(self):
        """Should catch specific exception type"""
        with pytest.raises(ValidationError):
            raise   ValidationError("test","test")

    def test_catch_as_base_exception(self):
        """Should catch derived exception as base type"""
        with pytest.raises(DevDoxSonarError):
            raise   ValidationError("test","test")

    def test_catch_multiple_exceptions(self):
        """Should handle multiple exception types"""

        def raise_random_error(error_type):
            if error_type == "validation":
                raise ValidationError("validation error","test")
            elif error_type == "config":
                raise ConfigurationError("config error")

        with pytest.raises((ValidationError, ConfigurationError)):
            raise_random_error("validation")

    def test_exception_info_preserved(self):
        """Should preserve exception information"""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test message","test")

        assert exc_info.type == ValidationError
        assert "Test message" in str(exc_info.value)


class TestExceptionMessages:
    """Test exception message handling"""

    @pytest.mark.parametrize("message", [
        "Simple message",
        "Message with 123 numbers",
        "Message with special chars: @#$%",
        "Multi\nline\nmessage",
        "Very long message " * 100
    ])
    def test_various_message_formats(self, message):
        """Should handle various message formats"""
        error = DevDoxSonarError(message)
        assert message in str(error)

    def test_unicode_in_message(self):
        """Should handle Unicode characters"""
        error = DevDoxSonarError("Error: αβγδ 中文字符 🎉")
        message = str(error)
        assert "αβγδ" in message
        assert "中文字符" in message

    def test_empty_message(self):
        """Should handle empty messages"""
        error = DevDoxSonarError("")
        assert str(error) == ""

    def test_none_message(self):
        """Should handle None as message"""
        try:
            error = DevDoxSonarError(None)
            # Might convert None to string "None"
            assert str(error) in ["None", ""]
        except TypeError:
            # Or might raise TypeError
            pass

# Run with: pytest tests/utils/test_exceptions.py -v --cov=devdox_ai_sonar.utils.exceptions --cov-report=term-missing