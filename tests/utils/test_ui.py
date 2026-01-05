import pytest
from typing import List
from unittest.mock import Mock, patch, MagicMock, call

from devdox_ai_sonar.utils.ui import (
    smart_prompt,
    smart_confirm,
    _check_for_switch_command,
    _prompt_with_rich_fallback,
    _prompt_with_questionary,
    _questionary_text_prompt,
    _questionary_select_prompt,
    _questionary_checkbox_prompt,
    _parse_default_choices,
    _confirm_with_questionary,
    _confirm_with_console_fallback,
    _build_console_prompt,
    _parse_confirmation_result,
)
from devdox_ai_sonar.utils.exceptions import SwitchCommandException
from devdox_ai_sonar.utils import constant
from devdox_ai_sonar.utils.result import PromptConfig, ConfirmConfig


# ============================================================================
# Test _parse_default_choices
# ============================================================================

class TestParseDefaultChoices:
    """Test cases for _parse_default_choices function."""

    def test_parse_with_none(self):
        """Test parsing None returns empty set."""
        result = _parse_default_choices(None)
        assert result == set()

    def test_parse_with_empty_list(self):
        """Test parsing empty list returns empty set."""
        result = _parse_default_choices([])
        assert result == set()

    def test_parse_with_single_item(self):
        """Test parsing single item list."""
        result = _parse_default_choices(["option1"])
        assert result == {"option1"}

    def test_parse_with_multiple_items(self):
        """Test parsing multiple items."""
        result = _parse_default_choices(["option1", "option2", "option3"])
        assert result == {"option1", "option2", "option3"}

    def test_parse_with_whitespace(self):
        """Test parsing items with whitespace."""
        result = _parse_default_choices(["  option1  ", "option2", "  option3"])
        assert result == {"option1", "option2", "option3"}

    def test_parse_with_empty_strings(self):
        """Test parsing filters out empty strings."""
        result = _parse_default_choices(["option1", "", "  ", "option2"])
        assert result == {"option1", "option2"}

    def test_parse_removes_duplicates(self):
        """Test parsing removes duplicates (set behavior)."""
        result = _parse_default_choices(["option1", "option1", "option2"])
        assert result == {"option1", "option2"}
        assert len(result) == 2


# ============================================================================
# Test _check_for_switch_command
# ============================================================================

class TestCheckForSwitchCommand:
    """Test cases for _check_for_switch_command function."""

    def test_no_switch_when_disabled(self):
        """Test no exception when switching is disabled."""
        _check_for_switch_command("/", allow_switch=False)
        # Should not raise

    def test_no_switch_on_none_result(self):
        """Test no exception on None result."""
        _check_for_switch_command(None, allow_switch=True)
        # Should not raise

    def test_switch_on_string_trigger(self):
        """Test raises exception on switch trigger string."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            with pytest.raises(SwitchCommandException):
                _check_for_switch_command("/", allow_switch=True)

    def test_switch_on_string_trigger_with_whitespace(self):
        """Test raises exception on switch trigger with whitespace."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            with pytest.raises(SwitchCommandException):
                _check_for_switch_command("  /  ", allow_switch=True)

    def test_no_switch_on_different_string(self):
        """Test no exception on different string."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            _check_for_switch_command("normal input", allow_switch=True)
            # Should not raise

    def test_switch_on_list_containing_trigger(self):
        """Test raises exception when list contains trigger."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            with pytest.raises(SwitchCommandException):
                _check_for_switch_command(["option1", "/", "option2"], allow_switch=True)

    def test_no_switch_on_list_without_trigger(self):
        """Test no exception when list doesn't contain trigger."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            _check_for_switch_command(["option1", "option2"], allow_switch=True)
            # Should not raise

    def test_empty_string_no_switch(self):
        """Test empty string doesn't trigger switch."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            _check_for_switch_command("", allow_switch=True)
            # Should not raise


# ============================================================================
# Test _build_console_prompt
# ============================================================================

class TestBuildConsolePrompt:
    """Test cases for _build_console_prompt function."""

    def test_prompt_with_default_true(self):
        """Test prompt format when default is True."""
        result = _build_console_prompt("Continue?", True)
        assert result == "Continue? [Y/n]: "

    def test_prompt_with_default_false(self):
        """Test prompt format when default is False."""
        result = _build_console_prompt("Delete?", False)
        assert result == "Delete? [y/N]: "

    def test_prompt_with_empty_message(self):
        """Test prompt with empty message."""
        result = _build_console_prompt("", True)
        assert result == " [Y/n]: "

    def test_prompt_preserves_message_formatting(self):
        """Test prompt preserves message formatting."""
        result = _build_console_prompt("Deploy to production?", True)
        assert "Deploy to production?" in result


# ============================================================================
# Test _parse_confirmation_result
# ============================================================================

class TestParseConfirmationResult:
    """Test cases for _parse_confirmation_result function."""

    def test_parse_none_returns_default_true(self):
        """Test None returns default (True)."""
        result = _parse_confirmation_result(None, True)
        assert result is True

    def test_parse_none_returns_default_false(self):
        """Test None returns default (False)."""
        result = _parse_confirmation_result(None, False)
        assert result is False

    def test_parse_empty_string_returns_default(self):
        """Test empty string returns default."""
        assert _parse_confirmation_result("", True) is True
        assert _parse_confirmation_result("", False) is False

    def test_parse_whitespace_returns_default(self):
        """Test whitespace returns default."""
        assert _parse_confirmation_result("   ", True) is True
        assert _parse_confirmation_result("   ", False) is False

    def test_parse_y_returns_true(self):
        """Test 'y' returns True."""
        result = _parse_confirmation_result("y", False)
        assert result is True

    def test_parse_yes_returns_true(self):
        """Test 'yes' returns True."""
        result = _parse_confirmation_result("yes", False)
        assert result is True

    def test_parse_Y_returns_true(self):
        """Test 'Y' (uppercase) returns True."""
        result = _parse_confirmation_result("Y", False)
        assert result is True

    def test_parse_YES_returns_true(self):
        """Test 'YES' (uppercase) returns True."""
        result = _parse_confirmation_result("YES", False)
        assert result is True

    def test_parse_n_returns_false(self):
        """Test 'n' returns False."""
        result = _parse_confirmation_result("n", True)
        assert result is False

    def test_parse_no_returns_false(self):
        """Test 'no' returns False."""
        result = _parse_confirmation_result("no", True)
        assert result is False

    def test_parse_N_returns_false(self):
        """Test 'N' (uppercase) returns False."""
        result = _parse_confirmation_result("N", True)
        assert result is False

    def test_parse_invalid_input_returns_false(self):
        """Test invalid input returns False."""
        result = _parse_confirmation_result("invalid", True)
        assert result is False


# ============================================================================
# Test questionary helper functions
# ============================================================================

class TestQuestionaryTextPrompt:
    """Test cases for _questionary_text_prompt."""

    @patch('devdox_ai_sonar.utils.ui.questionary.text')
    def test_text_prompt_with_default(self, mock_text):
        """Test text prompt with default value."""
        mock_text.return_value.ask.return_value = "user_input"

        result = _questionary_text_prompt("Enter name:", "John")

        mock_text.assert_called_once_with("Enter name:", default="John")
        assert result == "user_input"

    @patch('devdox_ai_sonar.utils.ui.questionary.text')
    def test_text_prompt_without_default(self, mock_text):
        """Test text prompt without default value."""
        mock_text.return_value.ask.return_value = "user_input"

        result = _questionary_text_prompt("Enter name:", None)

        mock_text.assert_called_once_with("Enter name:", default="")
        assert result == "user_input"

    @patch('devdox_ai_sonar.utils.ui.questionary.text')
    def test_text_prompt_with_empty_default(self, mock_text):
        """Test text prompt with empty string default."""
        mock_text.return_value.ask.return_value = "user_input"

        result = _questionary_text_prompt("Enter name:", "")

        mock_text.assert_called_once_with("Enter name:", default="")
        assert result == "user_input"


class TestQuestionarySelectPrompt:
    """Test cases for _questionary_select_prompt."""

    @patch('devdox_ai_sonar.utils.ui.questionary.select')
    def test_select_prompt_with_default(self, mock_select):
        """Test select prompt with default choice."""
        mock_select.return_value.ask.return_value = "Option B"

        result = _questionary_select_prompt(
            "Choose:",
            ["Option A", "Option B", "Option C"],
            "Option B"
        )

        mock_select.assert_called_once_with(
            "Choose:",
            choices=["Option A", "Option B", "Option C"],
            default="Option B"
        )
        assert result == "Option B"

    @patch('devdox_ai_sonar.utils.ui.questionary.select')
    def test_select_prompt_without_default(self, mock_select):
        """Test select prompt without default choice."""
        mock_select.return_value.ask.return_value = "Option A"

        result = _questionary_select_prompt(
            "Choose:",
            ["Option A", "Option B"],
            None
        )
        mock_select.assert_called_once()
        mock_select.assert_called_once_with(
            "Choose:",
            choices=["Option A", "Option B"],
            default="Option A"
        )
        assert result == "Option A"


class TestQuestionaryCheckboxPrompt:
    """Test cases for _questionary_checkbox_prompt."""

    @patch('devdox_ai_sonar.utils.ui.questionary.checkbox')
    def test_checkbox_prompt_with_defaults(self, mock_checkbox):
        """Test checkbox prompt with default selections."""
        mock_checkbox.return_value.ask.return_value = ["Python", "JavaScript"]

        result = _questionary_checkbox_prompt(
            "Select languages:",
            ["Python", "JavaScript", "Go", "Rust"],
            ["Python", "Go"]
        )

        # Verify checkbox was called
        assert mock_checkbox.called
        call_args = mock_checkbox.call_args

        # Check message
        assert call_args[0][0] == "Select languages:"

        # Check that choices were created correctly
        choices = call_args[1]['choices']
        assert len(choices) == 4

        # Verify correct items are checked
        python_choice = next(c for c in choices if c.value == "Python")
        go_choice = next(c for c in choices if c.value == "Go")
        js_choice = next(c for c in choices if c.value == "JavaScript")

        assert python_choice.checked is True
        assert go_choice.checked is True
        assert js_choice.checked is False

        assert result == ["Python", "JavaScript"]

    @patch('devdox_ai_sonar.utils.ui.questionary.checkbox')
    def test_checkbox_prompt_without_defaults(self, mock_checkbox):
        """Test checkbox prompt without default selections."""
        mock_checkbox.return_value.ask.return_value = ["Option A"]

        result = _questionary_checkbox_prompt(
            "Select options:",
            ["Option A", "Option B", "Option C"],
            None
        )

        assert mock_checkbox.called
        assert result == ["Option A"]

    @patch('devdox_ai_sonar.utils.ui.questionary.checkbox')
    def test_checkbox_prompt_with_empty_defaults(self, mock_checkbox):
        """Test checkbox prompt with empty default list."""
        mock_checkbox.return_value.ask.return_value = []

        result = _questionary_checkbox_prompt(
            "Select options:",
            ["Option A", "Option B"],
            []
        )

        assert result == []


# ============================================================================
# Test _prompt_with_questionary
# ============================================================================

class TestPromptWithQuestionary:
    """Test cases for _prompt_with_questionary function."""

    @patch('devdox_ai_sonar.utils.ui._questionary_text_prompt')
    def test_text_prompt_when_no_choices(self, mock_text_prompt, capsys):
        """Test routes to text prompt when no choices."""
        mock_text_prompt.return_value = "user_input"

        config = PromptConfig(
            message="Enter value:",
            default="default_value",
            choices=None,
            allow_switch=True,
            multiple=False
        )

        result = _prompt_with_questionary(config)

        mock_text_prompt.assert_called_once()
        assert result == "user_input"


    @patch('devdox_ai_sonar.utils.ui._questionary_checkbox_prompt')
    def test_checkbox_prompt_when_multiple(self, mock_checkbox_prompt):
        """Test routes to checkbox prompt when multiple is True."""
        mock_checkbox_prompt.return_value = ["option1", "option2"]

        config = PromptConfig(
            message="Select items:",
            choices=["option1", "option2", "option3"],
            multiple=True
        )

        result = _prompt_with_questionary(config)

        mock_checkbox_prompt.assert_called_once()
        assert result == ["option1", "option2"]

    @patch('devdox_ai_sonar.utils.ui._questionary_select_prompt')
    def test_select_prompt_when_choices_not_multiple(self, mock_select_prompt):
        """Test routes to select prompt when choices exist but not multiple."""
        mock_select_prompt.return_value = "option1"

        config = PromptConfig(
            message="Choose one:",
            choices=["option1", "option2"],
            multiple=False
        )

        result = _prompt_with_questionary(config)

        mock_select_prompt.assert_called_once()
        assert result == "option1"


# ============================================================================
# Test _prompt_with_rich_fallback
# ============================================================================

class TestPromptWithRichFallback:
    """Test cases for _prompt_with_rich_fallback function."""

    @patch('devdox_ai_sonar.utils.ui.console')
    @patch('devdox_ai_sonar.utils.ui.Prompt')
    def test_text_prompt_without_choices(self, mock_prompt_class, mock_console):
        """Test rich fallback for text input."""
        mock_prompt_class.ask.return_value = "user_input"

        config = PromptConfig(
            message="Enter value:",
            default="default",
            choices=None,
            allow_switch=False
        )

        result = _prompt_with_rich_fallback(config)

        mock_prompt_class.ask.assert_called_once_with("Enter value:", default="default")
        assert result == "user_input"

    @patch('devdox_ai_sonar.utils.ui.console')
    @patch('devdox_ai_sonar.utils.ui.Prompt')
    def test_select_prompt_with_choices(self, mock_prompt_class, mock_console):
        """Test rich fallback for select with choices."""
        mock_prompt_class.ask.return_value = "option2"

        config = PromptConfig(
            message="Choose:",
            choices=["option1", "option2"],
            default="option1",
            allow_switch=False
        )

        result = _prompt_with_rich_fallback(config)

        mock_prompt_class.ask.assert_called_once_with(
            "Choose:",
            choices=["option1", "option2"],
            default="option1"
        )
        assert result == "option2"

    @patch('devdox_ai_sonar.utils.ui.console')
    @patch('devdox_ai_sonar.utils.ui.Prompt')
    def test_shows_switch_hint_when_enabled(self, mock_prompt_class, mock_console):
        """Test shows switch command hint when enabled."""
        mock_prompt_class.ask.return_value = "input"

        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            config = PromptConfig(
                message="Enter:",
                allow_switch=True
            )

            _prompt_with_rich_fallback(config)

            mock_console.print.assert_called_with(
                "[dim](Type '/' to switch commands)[/dim]"
            )

    @patch('devdox_ai_sonar.utils.ui.console')
    @patch('devdox_ai_sonar.utils.ui.Prompt')
    def test_shows_multiple_warning(self, mock_prompt_class, mock_console):
        """Test shows warning when multiple selection requested."""
        mock_prompt_class.ask.return_value = "option1"

        config = PromptConfig(
            message="Select:",
            choices=["option1", "option2"],
            multiple=True,
            allow_switch=False
        )

        _prompt_with_rich_fallback(config)

        # Check that warning was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Multiple selection not available" in str(call) for call in calls)


# ============================================================================
# Test _confirm_with_questionary
# ============================================================================

class TestConfirmWithQuestionary:
    """Test cases for _confirm_with_questionary function."""

    @patch('devdox_ai_sonar.utils.ui.questionary.confirm')
    def test_confirm_with_default_yes(self, mock_confirm):
        """Test confirmation with Yes as default."""
        mock_confirm.return_value.ask.return_value = True  # Boolean, not string!

        config = ConfirmConfig(
            message="Continue?",
            default=True,
            allow_switch=True
        )

        result = _confirm_with_questionary(config)

        # Verify confirm was called with correct arguments
        mock_confirm.assert_called_once_with("Continue?", default=True)
        assert result == "yes"

    @patch('devdox_ai_sonar.utils.ui.questionary.confirm')
    def test_confirm_with_default_no(self, mock_confirm):
        """Test confirmation with No as default."""
        mock_confirm.return_value.ask.return_value = False  # Boolean!

        config = ConfirmConfig(
            message="Delete?",
            default=False,
            allow_switch=False
        )

        result = _confirm_with_questionary(config)

        mock_confirm.assert_called_once_with("Delete?", default=False)
        assert result == "no"


    @patch('devdox_ai_sonar.utils.ui.questionary.confirm')
    def test_confirm_basic_functionality(self, mock_confirm):
        """Test confirmation basic functionality."""
        mock_confirm.return_value.ask.return_value = "Yes"
        mock_confirm.return_value.ask.return_value = True

        config = ConfirmConfig(
            message="Proceed?",
            default=True,
            allow_switch=True  # Doesn't affect questionary.confirm
        )

        result = _confirm_with_questionary(config)

        mock_confirm.assert_called_once_with("Proceed?", default=True)
        assert result == "yes"

# ============================================================================
# Test _confirm_with_console_fallback
# ============================================================================

class TestConfirmWithConsoleFallback:
    """Test cases for _confirm_with_console_fallback function."""

    @patch('devdox_ai_sonar.utils.ui.console')
    def test_console_confirm_with_yes_input(self, mock_console):
        """Test console fallback with 'yes' input."""
        mock_console.input.return_value = "yes"

        config = ConfirmConfig(
            message="Continue?",
            default=True,
            allow_switch=False
        )

        result = _confirm_with_console_fallback(config)

        assert result == "yes"
        mock_console.input.assert_called_once()

    @patch('devdox_ai_sonar.utils.ui.console')
    def test_console_confirm_with_no_input(self, mock_console):
        """Test console fallback with 'no' input."""
        mock_console.input.return_value = "no"

        config = ConfirmConfig(
            message="Delete?",
            default=False,
            allow_switch=False
        )

        result = _confirm_with_console_fallback(config)

        assert result == "no"

    @patch('devdox_ai_sonar.utils.ui.console')
    def test_console_confirm_shows_switch_hint(self, mock_console):
        """Test console fallback shows switch hint."""
        mock_console.input.return_value = "yes"

        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            config = ConfirmConfig(
                message="Proceed?",
                default=True,
                allow_switch=True
            )

            _confirm_with_console_fallback(config)

            mock_console.print.assert_called_once_with(
                "[dim](Type '/' to switch commands)[/dim]"
            )

    @patch('devdox_ai_sonar.utils.ui.console')
    def test_console_confirm_strips_whitespace(self, mock_console):
        """Test console fallback strips whitespace from input."""
        mock_console.input.return_value = "  yes  "

        config = ConfirmConfig(
            message="Continue?",
            default=True,
            allow_switch=False
        )

        result = _confirm_with_console_fallback(config)

        assert result == "yes"  # Should be stripped and lowercased


# ============================================================================
# Test smart_prompt (integration)
# ============================================================================

class TestSmartPrompt:
    """Integration tests for smart_prompt function."""

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    def test_smart_prompt_uses_questionary(self, mock_check, mock_questionary):
        """Test smart_prompt uses questionary when available."""
        mock_questionary.return_value = "user_input"

        result = smart_prompt("Enter value:", default="default")

        assert result == "user_input"
        mock_questionary.assert_called_once()
        mock_check.assert_called_once_with("user_input", True)

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._prompt_with_rich_fallback')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    def test_smart_prompt_falls_back_to_rich(self, mock_check, mock_rich, mock_questionary):
        """Test smart_prompt falls back to rich on ImportError."""
        mock_questionary.side_effect = ImportError("questionary not found")
        mock_rich.return_value = "fallback_input"

        result = smart_prompt("Enter value:")

        assert result == "fallback_input"
        mock_rich.assert_called_once()
        mock_check.assert_called_once_with("fallback_input", True)

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    def test_smart_prompt_with_choices(self, mock_check, mock_questionary):
        """Test smart_prompt with choices."""
        mock_questionary.return_value = "option2"

        result = smart_prompt(
            "Choose:",
            choices=["option1", "option2", "option3"],
            default="option1"
        )

        assert result == "option2"

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    def test_smart_prompt_with_multiple(self, mock_check, mock_questionary):
        """Test smart_prompt with multiple selection."""
        mock_questionary.return_value = ["option1", "option3"]

        result = smart_prompt(
            "Select multiple:",
            choices=["option1", "option2", "option3"],
            multiple=True
        )

        assert result == ["option1", "option3"]

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    def test_smart_prompt_raises_switch_exception(self, mock_questionary):
        """Test smart_prompt raises SwitchCommandException."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            mock_questionary.return_value = "/"

            with pytest.raises(SwitchCommandException):
                smart_prompt("Enter value:", allow_switch=True)

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    def test_smart_prompt_with_switch_disabled(self, mock_check, mock_questionary):
        """Test smart_prompt with switch disabled."""
        mock_questionary.return_value = "/"

        result = smart_prompt("Enter value:", allow_switch=False)

        assert result == "/"
        mock_check.assert_called_once_with("/", False)


# ============================================================================
# Test smart_confirm (integration)
# ============================================================================

class TestSmartConfirm:
    """Integration tests for smart_confirm function."""

    @patch('devdox_ai_sonar.utils.ui._confirm_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    @patch('devdox_ai_sonar.utils.ui._parse_confirmation_result')
    def test_smart_confirm_uses_questionary(self, mock_parse, mock_check, mock_questionary):
        """Test smart_confirm uses questionary when available."""
        mock_questionary.return_value = "Yes"
        mock_parse.return_value = True

        result = smart_confirm("Continue?", default=True)

        assert result is True
        mock_questionary.assert_called_once()
        mock_check.assert_called_once_with("Yes", True)
        mock_parse.assert_called_once_with("Yes", True)

    @patch('devdox_ai_sonar.utils.ui._confirm_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._confirm_with_console_fallback')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    @patch('devdox_ai_sonar.utils.ui._parse_confirmation_result')
    def test_smart_confirm_falls_back_to_console(
            self, mock_parse, mock_check, mock_console, mock_questionary
    ):
        """Test smart_confirm falls back to console on ImportError."""
        mock_questionary.side_effect = ImportError("questionary not found")
        mock_console.return_value = "yes"
        mock_parse.return_value = True

        result = smart_confirm("Continue?")

        assert result is True
        mock_console.assert_called_once()



    @patch('devdox_ai_sonar.utils.ui._confirm_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    @patch('devdox_ai_sonar.utils.ui._parse_confirmation_result')
    def test_smart_confirm_with_default_false(self, mock_parse, mock_check, mock_questionary):
        """Test smart_confirm with default False."""
        mock_questionary.return_value = "No"
        mock_parse.return_value = False

        result = smart_confirm("Delete?", default=False)

        assert result is False
        mock_parse.assert_called_once_with("No", False)

    @patch('devdox_ai_sonar.utils.ui._confirm_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    @patch('devdox_ai_sonar.utils.ui._parse_confirmation_result')
    def test_smart_confirm_with_switch_disabled(self, mock_parse, mock_check, mock_questionary):
        """Test smart_confirm with switch disabled."""
        mock_questionary.return_value = "Yes"
        mock_parse.return_value = True

        result = smart_confirm("Proceed?", allow_switch=False)

        assert result is True
        mock_check.assert_called_once_with("Yes", False)


# ============================================================================
# Edge Cases and Error Scenarios
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    def test_smart_prompt_with_none_result(self, mock_check, mock_questionary):
        """Test smart_prompt handles None result (user cancelled)."""
        mock_questionary.return_value = None

        result = smart_prompt("Enter value:")

        assert result is None
        mock_check.assert_called_once_with(None, True)

    @patch('devdox_ai_sonar.utils.ui._prompt_with_questionary')
    @patch('devdox_ai_sonar.utils.ui._check_for_switch_command')
    def test_smart_prompt_with_empty_string(self, mock_check, mock_questionary):
        """Test smart_prompt handles empty string."""
        mock_questionary.return_value = ""

        result = smart_prompt("Enter value:")

        assert result == ""

    def test_parse_default_choices_with_unicode(self):
        """Test parsing choices with Unicode characters."""
        result = _parse_default_choices(["选项1", "选项2", "option3"])
        assert result == {"选项1", "选项2", "option3"}

    @patch('devdox_ai_sonar.utils.ui.console')
    def test_console_fallback_with_unicode_input(self, mock_console):
        """Test console fallback handles Unicode input."""
        mock_console.input.return_value = "是"  # Chinese: "yes"

        config = ConfirmConfig("继续?", True, False)
        result = _confirm_with_console_fallback(config)

        assert result == "是"


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def mock_questionary():
    """Fixture to mock questionary module."""
    with patch('devdox_ai_sonar.utils.ui.questionary') as mock:
        yield mock


@pytest.fixture
def mock_console():
    """Fixture to mock rich console."""
    with patch('devdox_ai_sonar.utils.ui.console') as mock:
        yield mock


@pytest.fixture
def sample_prompt_config():
    """Fixture providing sample PromptConfig."""
    return PromptConfig(
        message="Test prompt",
        default="default_value",
        choices=["choice1", "choice2"],
        allow_switch=True,
        multiple=False
    )


@pytest.fixture
def sample_confirm_config():
    """Fixture providing sample ConfirmConfig."""
    return ConfirmConfig(
        message="Test confirm",
        default=True,
        allow_switch=True
    )


# ============================================================================
# Parametrized Tests
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("input_value,expected", [
        ("y", True),
        ("Y", True),
        ("yes", True),
        ("YES", True),
        ("Yes", True),
        ("n", False),
        ("N", False),
        ("no", False),
        ("NO", False),
        ("No", False),
        ("invalid", False),
        ("", None),  # Will use default
    ])
    def test_parse_confirmation_various_inputs(self, input_value, expected):
        """Test parsing various confirmation inputs."""
        if expected is None:
            # Test with both defaults
            assert _parse_confirmation_result(input_value, True) is True
            assert _parse_confirmation_result(input_value, False) is False
        else:
            result = _parse_confirmation_result(input_value, False)
            assert result == expected

    @pytest.mark.parametrize("allow_switch,should_raise", [
        (True, True),
        (False, False),
    ])
    def test_check_switch_with_various_settings(self, allow_switch, should_raise):
        """Test switch checking with various settings."""
        with patch.object(constant, 'SWITCH_COMMAND_TRIGGER', '/'):
            if should_raise:
                with pytest.raises(SwitchCommandException):
                    _check_for_switch_command("/", allow_switch)
            else:
                _check_for_switch_command("/", allow_switch)
                # Should not raise