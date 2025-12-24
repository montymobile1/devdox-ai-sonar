

import pytest
from typing import List
from dataclasses import dataclass

from devdox_ai_sonar.utils import constant
from devdox_ai_sonar.utils.result import (
    Ok,
    Err,
    Result,
    PromptConfig,
    ConfirmConfig,
)


# ============================================================================
# Test Result Types: Ok and Err
# ============================================================================

class TestOk:
    """Test cases for Ok[T] result type."""

    def test_ok_with_string_value(self):
        """Test Ok with string value."""
        result = Ok(value="success")

        assert result.is_ok() is True
        assert result.is_err() is False
        assert result.unwrap() == "success"

    def test_ok_with_int_value(self):
        """Test Ok with integer value."""
        result = Ok(value=42)

        assert result.is_ok() is True
        assert result.is_err() is False
        assert result.unwrap() == 42

    def test_ok_with_dict_value(self):
        """Test Ok with dictionary value."""
        data = {"key": "value", "count": 10}
        result = Ok(value=data)

        assert result.is_ok() is True
        assert result.unwrap() == data

    def test_ok_with_list_value(self):
        """Test Ok with list value."""
        items = ["item1", "item2", "item3"]
        result = Ok(value=items)

        assert result.unwrap() == items

    def test_ok_with_none_value(self):
        """Test Ok with None value (valid case)."""
        result = Ok(value=None)

        assert result.is_ok() is True
        assert result.unwrap() is None

    def test_ok_with_custom_object(self):
        """Test Ok with custom object."""

        @dataclass
        class User:
            name: str
            age: int

        user = User(name="Alice", age=30)
        result = Ok(value=user)

        assert result.unwrap().name == "Alice"
        assert result.unwrap().age == 30

    def test_ok_equality(self):
        """Test Ok equality comparison."""
        result1 = Ok(value="test")
        result2 = Ok(value="test")
        result3 = Ok(value="different")

        assert result1 == result2
        assert result1 != result3


class TestErr:
    """Test cases for Err[E] error type."""

    def test_err_with_value_error(self):
        """Test Err with ValueError."""
        error = ValueError("Invalid input")
        result = Err(error=error)

        assert result.is_ok() is False
        assert result.is_err() is True
        assert result.error == error

    def test_err_with_runtime_error(self):
        """Test Err with RuntimeError."""
        error = RuntimeError("Something went wrong")
        result = Err(error=error)

        assert result.is_err() is True

    def test_err_unwrap_raises_exception(self):
        """Test that unwrap raises the contained error."""
        error = ValueError("Test error")
        result = Err(error=error)

        with pytest.raises(ValueError, match="Test error"):
            result.unwrap()

    def test_err_unwrap_raises_custom_exception(self):
        """Test unwrap with custom exception."""

        class CustomError(Exception):
            pass

        error = CustomError("Custom error message")
        result = Err(error=error)

        with pytest.raises(CustomError, match="Custom error message"):
            result.unwrap()

    def test_err_with_keyboard_interrupt(self):
        """Test Err with KeyboardInterrupt (BaseException subclass)."""
        # Note: KeyboardInterrupt is not bound by Exception in TypeVar
        # but testing runtime behavior
        error = KeyboardInterrupt()
        result = Err(error=error)

        assert result.is_err() is True

        with pytest.raises(KeyboardInterrupt):
            result.unwrap()

    def test_err_equality(self):
        """Test Err equality comparison."""
        error1 = ValueError("error")
        error2 = ValueError("error")
        error3 = RuntimeError("different")

        result1 = Err(error=error1)
        result2 = Err(error=error2)
        result3 = Err(error=error3)

        # Errors with same message may not be equal (different instances)
        # Testing that comparison works
        assert result1 != result3


class TestResultTypeHints:
    """Test Result type usage patterns."""

    def test_result_union_type_with_ok(self):
        """Test Result union type with Ok."""

        def divide(a: int, b: int) -> Result[float, ZeroDivisionError]:
            if b == 0:
                return Err(error=ZeroDivisionError("Cannot divide by zero"))
            return Ok(value=a / b)

        result = divide(10, 2)
        assert result.is_ok()
        assert result.unwrap() == 5.0

    def test_result_union_type_with_err(self):
        """Test Result union type with Err."""

        def divide(a: int, b: int) -> Result[float, ZeroDivisionError]:
            if b == 0:
                return Err(error=ZeroDivisionError("Cannot divide by zero"))
            return Ok(value=a / b)

        result = divide(10, 0)
        assert result.is_err()

        with pytest.raises(ZeroDivisionError):
            result.unwrap()

    def test_result_pattern_matching_style(self):
        """Test Result with pattern matching style usage."""

        def parse_int(value: str) -> Result[int, ValueError]:
            try:
                return Ok(value=int(value))
            except ValueError as e:
                return Err(error=e)

        # Success case
        result_ok = parse_int("42")
        if result_ok.is_ok():
            assert result_ok.unwrap() == 42

        # Error case
        result_err = parse_int("not_a_number")
        if result_err.is_err():
            with pytest.raises(ValueError):
                result_err.unwrap()


# ============================================================================
# Test PromptConfig
# ============================================================================

class TestPromptConfig:
    """Test cases for PromptConfig dataclass."""

    def test_prompt_config_minimal(self):
        """Test PromptConfig with minimal parameters."""
        config = PromptConfig(message="Enter value:")

        assert config.message == "Enter value:"
        assert config.default is None
        assert config.choices is None
        assert config.allow_switch is True
        assert config.multiple is False

    def test_prompt_config_with_string_default(self):
        """Test PromptConfig with string default."""
        config = PromptConfig(
            message="Enter name:",
            default="John"
        )

        assert config.default == "John"

    def test_prompt_config_with_list_default(self):
        """Test PromptConfig with list default."""
        config = PromptConfig(
            message="Select items:",
            default=["item1", "item2"],
            multiple=True
        )

        assert config.default == ["item1", "item2"]
        assert config.multiple is True

    def test_prompt_config_with_choices(self):
        """Test PromptConfig with choices."""
        choices = ["Option A", "Option B", "Option C"]
        config = PromptConfig(
            message="Select option:",
            choices=choices
        )

        assert config.choices == choices

    def test_prompt_config_allow_switch_disabled(self):
        """Test PromptConfig with switch disabled."""
        config = PromptConfig(
            message="Enter value:",
            allow_switch=False
        )

        assert config.allow_switch is False

    def test_prompt_config_get_display_message_with_switch(self):
        """Test get_display_message with switch enabled."""
        config = PromptConfig(
            message="Enter value:",
            allow_switch=True
        )

        expected = "Enter value:\n[dim](Type '/' to switch commands)[/dim]"
        assert config.get_display_message() == expected

    def test_prompt_config_get_display_message_without_switch(self):
        """Test get_display_message with switch disabled."""
        config = PromptConfig(
            message="Enter value:",
            allow_switch=False
        )

        assert config.get_display_message() == "Enter value:"

    def test_prompt_config_multiple_with_choices(self):
        """Test PromptConfig for multiple selection."""
        config = PromptConfig(
            message="Select languages:",
            choices=["Python", "JavaScript", "Go"],
            default=["Python", "Go"],
            multiple=True
        )

        assert config.multiple is True
        assert config.choices == ["Python", "JavaScript", "Go"]
        assert config.default == ["Python", "Go"]

    def test_prompt_config_equality(self):
        """Test PromptConfig equality."""
        config1 = PromptConfig(message="Test", default="A")
        config2 = PromptConfig(message="Test", default="A")
        config3 = PromptConfig(message="Test", default="B")

        assert config1 == config2
        assert config1 != config3

    def test_prompt_config_with_empty_choices_list(self):
        """Test PromptConfig with empty choices list."""
        config = PromptConfig(
            message="Select:",
            choices=[]
        )

        assert config.choices == []

    def test_prompt_config_switch_trigger_in_message(self):
        """Test display message includes correct switch trigger."""
        # Assuming constant.SWITCH_COMMAND_TRIGGER is defined
        config = PromptConfig(
            message="Choose option:",
            allow_switch=True
        )

        display_message = config.get_display_message()
        assert "Type '/' to switch commands" in display_message or \
               f"Type '{constant.SWITCH_COMMAND_TRIGGER}' to switch commands" in display_message


# ============================================================================
# Test ConfirmConfig
# ============================================================================

class TestConfirmConfig:
    """Test cases for ConfirmConfig dataclass."""

    def test_confirm_config_minimal(self):
        """Test ConfirmConfig with minimal parameters."""
        config = ConfirmConfig(
            message="Continue?",
            default=True,
            allow_switch=True
        )

        assert config.message == "Continue?"
        assert config.default is True
        assert config.allow_switch is True

    def test_confirm_config_default_false(self):
        """Test ConfirmConfig with default False."""
        config = ConfirmConfig(
            message="Delete file?",
            default=False,
            allow_switch=True
        )

        assert config.default is False

    def test_confirm_config_switch_disabled(self):
        """Test ConfirmConfig with switch disabled."""
        config = ConfirmConfig(
            message="Proceed?",
            default=True,
            allow_switch=False
        )

        assert config.allow_switch is False

    def test_confirm_config_get_display_message_with_switch(self):
        """Test get_display_message with switch enabled."""
        config = ConfirmConfig(
            message="Confirm action?",
            default=True,
            allow_switch=True
        )

        expected = "Confirm action?\n[dim](Type '/' to switch commands)[/dim]"
        assert config.get_display_message() == expected

    def test_confirm_config_get_display_message_without_switch(self):
        """Test get_display_message with switch disabled."""
        config = ConfirmConfig(
            message="Confirm action?",
            default=True,
            allow_switch=False
        )

        assert config.get_display_message() == "Confirm action?"

    def test_confirm_config_get_default_choice_yes(self):
        """Test get_default_choice returns 'Yes' when default is True."""
        config = ConfirmConfig(
            message="Continue?",
            default=True,
            allow_switch=True
        )

        assert config.get_default_choice() == "Yes"

    def test_confirm_config_get_default_choice_no(self):
        """Test get_default_choice returns 'No' when default is False."""
        config = ConfirmConfig(
            message="Continue?",
            default=False,
            allow_switch=True
        )

        assert config.get_default_choice() == "No"

    def test_confirm_config_get_questionary_choices_with_switch(self):
        """Test get_questionary_choices includes switch option."""
        config = ConfirmConfig(
            message="Proceed?",
            default=True,
            allow_switch=True
        )

        choices = config.get_questionary_choices()

        assert "Yes" in choices
        assert "No" in choices
        assert f"{constant.SWITCH_COMMAND_TRIGGER} Switch Command" in choices
        assert len(choices) == 3

    def test_confirm_config_get_questionary_choices_without_switch(self):
        """Test get_questionary_choices excludes switch option."""
        config = ConfirmConfig(
            message="Proceed?",
            default=True,
            allow_switch=False
        )

        choices = config.get_questionary_choices()

        assert "Yes" in choices
        assert "No" in choices
        assert len(choices) == 2
        assert f"{constant.SWITCH_COMMAND_TRIGGER} Switch Command" not in choices

    def test_confirm_config_equality(self):
        """Test ConfirmConfig equality."""
        config1 = ConfirmConfig("Test?", True, True)
        config2 = ConfirmConfig("Test?", True, True)
        config3 = ConfirmConfig("Test?", False, True)

        assert config1 == config2
        assert config1 != config3

    def test_confirm_config_all_methods_integration(self):
        """Test all ConfirmConfig methods together."""
        config = ConfirmConfig(
            message="Deploy to production?",
            default=False,
            allow_switch=True
        )

        # Test all methods
        display_msg = config.get_display_message()
        default_choice = config.get_default_choice()
        choices = config.get_questionary_choices()

        assert "Deploy to production?" in display_msg
        assert "Type '/' to switch commands" in display_msg
        assert default_choice == "No"
        assert len(choices) == 3


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_prompt_config_with_special_characters_in_message(self):
        """Test PromptConfig with special characters."""
        config = PromptConfig(
            message="Enter value (1-100):\n> ",
            default="50"
        )

        assert "Enter value (1-100):\n> " in config.message

    def test_prompt_config_with_unicode_characters(self):
        """Test PromptConfig with Unicode characters."""
        config = PromptConfig(
            message="选择选项:",  # Chinese: "Select option:"
            choices=["选项A", "选项B"]
        )

        assert config.message == "选择选项:"
        assert config.choices == ["选项A", "选项B"]

    def test_prompt_config_with_very_long_message(self):
        """Test PromptConfig with very long message."""
        long_message = "A" * 1000
        config = PromptConfig(message=long_message)

        assert len(config.message) == 1000

    def test_confirm_config_with_empty_message(self):
        """Test ConfirmConfig with empty message (valid but unusual)."""
        config = ConfirmConfig(
            message="",
            default=True,
            allow_switch=False
        )

        assert config.message == ""
        assert config.get_display_message() == ""

    def test_ok_with_zero_value(self):
        """Test Ok with zero value (falsy but valid)."""
        result = Ok(value=0)

        assert result.is_ok()
        assert result.unwrap() == 0

    def test_ok_with_empty_string(self):
        """Test Ok with empty string (falsy but valid)."""
        result = Ok(value="")

        assert result.is_ok()
        assert result.unwrap() == ""

    def test_ok_with_empty_list(self):
        """Test Ok with empty list (falsy but valid)."""
        result = Ok(value=[])

        assert result.is_ok()
        assert result.unwrap() == []


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_result_in_prompt_validation_scenario(self):
        """Test Result type in a prompt validation scenario."""

        def validate_choice(
                choice: str,
                valid_choices: List[str]
        ) -> Result[str, ValueError]:
            if choice in valid_choices:
                return Ok(value=choice)
            return Err(error=ValueError(f"Invalid choice: {choice}"))

        config = PromptConfig(
            message="Select option:",
            choices=["A", "B", "C"]
        )

        # Valid choice
        result_ok = validate_choice("A", config.choices)
        assert result_ok.is_ok()
        assert result_ok.unwrap() == "A"

        # Invalid choice
        result_err = validate_choice("D", config.choices)
        assert result_err.is_err()

    def test_prompt_and_confirm_configs_together(self):
        """Test using both config types in workflow."""
        prompt_config = PromptConfig(
            message="Enter deployment target:",
            choices=["staging", "production"],
            default="staging"
        )

        confirm_config = ConfirmConfig(
            message="Proceed with deployment?",
            default=False,
            allow_switch=True
        )

        # Verify both configs work together
        assert prompt_config.choices == ["staging", "production"]
        assert confirm_config.get_default_choice() == "No"
        assert confirm_config.get_questionary_choices() == [
            "Yes",
            "No",
            f"{constant.SWITCH_COMMAND_TRIGGER} Switch Command"
        ]


# ============================================================================
# Pytest Fixtures (if needed for more complex tests)
# ============================================================================

@pytest.fixture
def sample_prompt_config():
    """Fixture providing a sample PromptConfig."""
    return PromptConfig(
        message="Test prompt:",
        default="test_value",
        choices=["option1", "option2"],
        allow_switch=True,
        multiple=False
    )


@pytest.fixture
def sample_confirm_config():
    """Fixture providing a sample ConfirmConfig."""
    return ConfirmConfig(
        message="Test confirmation?",
        default=True,
        allow_switch=True
    )


class TestWithFixtures:
    """Tests using fixtures for reusability."""

    def test_prompt_config_fixture(self, sample_prompt_config):
        """Test using prompt config fixture."""
        assert sample_prompt_config.message == "Test prompt:"
        assert sample_prompt_config.default == "test_value"

    def test_confirm_config_fixture(self, sample_confirm_config):
        """Test using confirm config fixture."""
        assert sample_confirm_config.default is True
        assert sample_confirm_config.get_default_choice() == "Yes"

