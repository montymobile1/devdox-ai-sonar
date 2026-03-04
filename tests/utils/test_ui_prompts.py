"""Tests for the cross-platform UI prompt utilities."""

from unittest.mock import patch, Mock
from devdox_ai_sonar.utils.ui_prompts import select_from_list


class TestSelectFromList:
    """Tests for select_from_list function."""

    def test_empty_list_returns_none(self):
        """Empty choices should return None without prompting."""
        result = select_from_list([], "Pick one")
        assert result is None

    @patch("devdox_ai_sonar.utils.ui_prompts.questionary")
    def test_returns_selected_value(self, mock_q):
        """Should return the string chosen by the user."""
        mock_q.select.return_value.ask.return_value = "anthropic"

        result = select_from_list(["openai", "anthropic"], "Select provider")

        assert result == "anthropic"
        mock_q.select.assert_called_once()

    @patch("devdox_ai_sonar.utils.ui_prompts.questionary")
    def test_cancel_returns_none(self, mock_q):
        """Ctrl+C / cancel should return None."""
        mock_q.select.return_value.ask.return_value = None

        result = select_from_list(["openai"], "Select")

        assert result is None
