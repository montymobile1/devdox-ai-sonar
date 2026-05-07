"""Unit tests for the wiring between the SonarCloud fix agent and its tools.

Covers the two module-level helpers extracted from
``LLMFixer._call_llm_list`` — the ones that were hard to test while embedded
as a closure and an inline list literal:

* ``_build_agent_tools()`` — the ordered tool list passed to ``Agent``.
* ``_handle_agent_event()`` — the callback that renders agent progress to
  the Rich console.

Strategy: call the helpers directly.  Rich's ``Console`` is constructed with
``file=StringIO()`` so we can assert on its output without mocking the Rich
package.  OpenHands' event types are third-party and constructing real
instances would drag in a lot of runtime state, so we stub them with
``MagicMock(spec=...)`` (this is the "patch what you can't inject" carve-out
from the testing guide).  Observation payloads are plain
``SimpleNamespace`` objects because the handler reads them only via
``getattr``.
"""

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent

from devdox_ai_sonar.llm_fixer import (
    _build_agent_tools,
    _handle_agent_event,
)
from devdox_ai_sonar.openhands_tools.fix_at_line import FixAtLineTool


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def capture_console() -> Tuple[Console, StringIO]:
    """A Rich Console whose output can be asserted on."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=240)
    return console, buf


# ============================================================================
# _build_agent_tools()
# ============================================================================


class TestBuildAgentTools:
    """The tool list wired into the SonarCloud fix Agent."""

    def test_returns_four_tools(self):
        assert len(_build_agent_tools()) == 4

    def test_includes_fix_at_line(self):
        names = [t.name for t in _build_agent_tools()]

        assert FixAtLineTool.name in names
        assert "fix_at_line" in names

    def test_expected_tool_names_in_expected_order(self):
        expected = ["terminal", "file_editor", "fix_at_line", "task_tracker"]

        assert [t.name for t in _build_agent_tools()] == expected

    def test_returns_fresh_list_each_call(self):
        """Defensive — callers must not accidentally share list state."""
        first = _build_agent_tools()
        second = _build_agent_tools()

        assert first is not second


# ============================================================================
# _handle_agent_event() — fix_at_line observations
# ============================================================================


class TestHandleAgentEventFixAtLine:
    """The new branch that formats ``fix_at_line`` tool observations."""

    def test_single_line_success_shows_line_label(self, capture_console):
        console, buf = capture_console
        obs = SimpleNamespace(
            text="Replaced line 7",
            start_line=7, end_line=7, is_error=False,
            path="/abs/file.py",
        )
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "fix_at_line"
        event.observation = obs

        _handle_agent_event(event, console, Path("/abs/file.py"), [])

        out = buf.getvalue()
        assert "Edited line 7" in out
        assert "/abs/file.py" in out

    def test_multi_line_success_shows_range_label(self, capture_console):
        console, buf = capture_console
        obs = SimpleNamespace(
            text="Replaced lines 3-5",
            start_line=3, end_line=5, is_error=False,
            path="/abs/file.py",
        )
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "fix_at_line"
        event.observation = obs

        _handle_agent_event(event, console, Path("/abs/file.py"), [])

        assert "Edited lines 3-5" in buf.getvalue()

    def test_error_observation_shows_failure_line(self, capture_console):
        console, buf = capture_console
        obs = SimpleNamespace(
            text="old_block does not match the current content",
            start_line=7, end_line=7, is_error=True,
            path="/abs/file.py",
        )
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "fix_at_line"
        event.observation = obs

        _handle_agent_event(event, console, Path("/abs/file.py"), [])

        out = buf.getvalue()
        assert "fix_at_line failed" in out
        assert "old_block does not match" in out

    def test_missing_line_numbers_fall_back_to_range_label(self, capture_console):
        """Defensive: if an observation somehow arrives without line numbers."""
        console, buf = capture_console
        obs = SimpleNamespace(
            text="?",
            start_line=None, end_line=None, is_error=False,
            path="/abs/file.py",
        )
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "fix_at_line"
        event.observation = obs

        _handle_agent_event(event, console, Path("/abs/file.py"), [])

        assert "Edited range" in buf.getvalue()


# ============================================================================
# _handle_agent_event() — pre-existing branches (regression guard)
# ============================================================================


class TestHandleAgentEventFileEditor:
    """Pre-existing ``file_editor`` handling still works after the refactor."""

    def test_str_replace_command_prints_edited(self, capture_console):
        console, buf = capture_console
        obs = SimpleNamespace(command="str_replace", text="", path="/abs/editor.py")
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "file_editor"
        event.observation = obs

        _handle_agent_event(event, console, Path("/abs/editor.py"), [])

        out = buf.getvalue()
        assert "Edited" in out
        assert "/abs/editor.py" in out

    def test_view_command_prints_view_label(self, capture_console):
        console, buf = capture_console
        obs = SimpleNamespace(command="view", text="", path="/abs/view.py")
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "file_editor"
        event.observation = obs

        _handle_agent_event(event, console, Path("/abs/view.py"), [])

        assert "view" in buf.getvalue()


class TestHandleAgentEventTerminal:
    """Terminal observations use the exit-code path."""

    def test_terminal_success_prints_check_mark_and_exit_zero(self, capture_console):
        console, buf = capture_console
        obs = SimpleNamespace(
            text="SYNTAX_OK",
            metadata=SimpleNamespace(exit_code=0),
        )
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "terminal"
        event.observation = obs

        _handle_agent_event(event, console, Path("/f.py"), [])

        out = buf.getvalue()
        assert "Terminal" in out
        assert "exit 0" in out
        assert "SYNTAX_OK" in out

    def test_terminal_failure_shows_nonzero_exit(self, capture_console):
        console, buf = capture_console
        obs = SimpleNamespace(
            text="SyntaxError: invalid syntax",
            metadata=SimpleNamespace(exit_code=1),
        )
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = "terminal"
        event.observation = obs

        _handle_agent_event(event, console, Path("/f.py"), [])

        assert "exit 1" in buf.getvalue()


class TestHandleAgentEventAgentMessage:
    """Agent-authored messages are appended to the accumulator and printed."""

    def test_agent_text_is_appended_and_printed(self, capture_console):
        console, buf = capture_console
        messages: List[str] = []
        event = MagicMock(spec=MessageEvent)
        event.source = "agent"
        event.to_llm_message.return_value = SimpleNamespace(
            content="Fix applied on line 7."
        )

        _handle_agent_event(event, console, Path("/f.py"), messages)

        assert messages == ["Fix applied on line 7."]
        assert "Fix applied on line 7" in buf.getvalue()

    def test_user_messages_are_ignored(self, capture_console):
        console, buf = capture_console
        messages: List[str] = []
        event = MagicMock(spec=MessageEvent)
        event.source = "user"
        event.to_llm_message.return_value = SimpleNamespace(
            content="Please fix the issue."
        )

        _handle_agent_event(event, console, Path("/f.py"), messages)

        assert messages == []
        assert buf.getvalue() == ""

    def test_whitespace_only_messages_are_dropped(self, capture_console):
        console, _ = capture_console
        messages: List[str] = []
        event = MagicMock(spec=MessageEvent)
        event.source = "agent"
        event.to_llm_message.return_value = SimpleNamespace(content="   \n  ")

        _handle_agent_event(event, console, Path("/f.py"), messages)

        assert messages == []


class TestHandleAgentEventActionEvent:
    """ActionEvents show the tool being invoked and any prefatory thought."""

    def test_action_without_thought_prints_tool_call(self, capture_console):
        console, buf = capture_console
        event = MagicMock(spec=ActionEvent)
        event.tool_name = "fix_at_line"
        event.thought = None

        _handle_agent_event(event, console, Path("/f.py"), [])

        out = buf.getvalue()
        assert "Calling" in out
        assert "fix_at_line" in out

    def test_action_with_thought_prints_thought_and_tool(self, capture_console):
        console, buf = capture_console
        event = MagicMock(spec=ActionEvent)
        event.tool_name = "fix_at_line"
        event.thought = [SimpleNamespace(text="I'll replace line 7 with the constant")]

        _handle_agent_event(event, console, Path("/f.py"), [])

        out = buf.getvalue()
        assert "I'll replace line 7" in out
        assert "fix_at_line" in out
