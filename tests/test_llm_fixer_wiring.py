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
    _block_to_text,
    _build_agent_tools,
    _extract_agent_message_text,
    _extract_first_thought,
    _extract_observation_text,
    _extract_terminal_exit_code,
    _format_fix_at_line_range,
    _handle_agent_event,
    _is_fix_at_line_action,
    _make_event_callback,
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


# ============================================================================
# PURE-LOGIC HELPERS EXTRACTED FROM _handle_agent_event
# ============================================================================
#
# The helpers below are the small extract/format functions produced by the
# cognitive-complexity refactor.  The render/handle helpers stay covered by
# the TestHandleAgentEvent* classes above (transitive coverage through the
# public ``_handle_agent_event`` entry point).


class TestBlockToText:
    """``_block_to_text`` extracts text from a single content block."""

    def test_object_with_text_attribute(self):
        block = SimpleNamespace(text="hello world")

        assert _block_to_text(block) == "hello world"

    def test_dict_with_text_type(self):
        block = {"type": "text", "text": "dict content"}

        assert _block_to_text(block) == "dict content"

    def test_dict_with_non_text_type_returns_empty(self):
        block = {"type": "image", "text": "ignored"}

        assert _block_to_text(block) == ""

    def test_object_with_empty_text_falls_through(self):
        """An object whose .text is empty should fall through to dict check."""
        block = SimpleNamespace(text="")

        assert _block_to_text(block) == ""

    def test_unknown_block_returns_empty(self):
        assert _block_to_text(object()) == ""


class TestExtractAgentMessageText:
    """``_extract_agent_message_text`` normalizes string / list / other content."""

    def _event_with_content(self, content):
        event = MagicMock(spec=MessageEvent)
        event.to_llm_message.return_value = SimpleNamespace(content=content)
        return event

    def test_plain_string_content(self):
        event = self._event_with_content("direct text")

        assert _extract_agent_message_text(event) == "direct text"

    def test_list_of_text_blocks_joined(self):
        event = self._event_with_content([
            SimpleNamespace(text="first "),
            SimpleNamespace(text="second"),
        ])

        assert _extract_agent_message_text(event) == "first second"

    def test_list_with_dict_blocks_joined(self):
        event = self._event_with_content([
            {"type": "text", "text": "A"},
            {"type": "text", "text": "B"},
        ])

        assert _extract_agent_message_text(event) == "AB"

    def test_unknown_content_type_returns_empty(self):
        event = self._event_with_content(42)

        assert _extract_agent_message_text(event) == ""


class TestExtractFirstThought:
    """``_extract_first_thought`` returns the first block's text, truncated."""

    def test_first_block_with_text(self):
        blocks = [SimpleNamespace(text="first thought"), SimpleNamespace(text="second")]

        assert _extract_first_thought(blocks) == "first thought"

    def test_skips_blocks_without_text_attribute(self):
        class Bare:
            pass

        blocks = [Bare(), SimpleNamespace(text="found it")]

        assert _extract_first_thought(blocks) == "found it"

    def test_no_text_blocks_returns_empty(self):
        class Bare:
            pass

        assert _extract_first_thought([Bare(), Bare()]) == ""

    def test_empty_iterable_returns_empty(self):
        assert _extract_first_thought([]) == ""

    def test_truncates_at_120_chars(self):
        long = "x" * 200
        blocks = [SimpleNamespace(text=long)]

        result = _extract_first_thought(blocks)

        assert len(result) == 120


class TestExtractObservationText:
    """``_extract_observation_text`` prefers ``text`` then falls back to ``content``."""

    def test_uses_text_attribute_when_present(self):
        obs = SimpleNamespace(text="primary", content="fallback")

        assert _extract_observation_text(obs) == "primary"

    def test_falls_back_to_content_when_text_empty(self):
        obs = SimpleNamespace(text="", content="fallback")

        assert _extract_observation_text(obs) == "fallback"

    def test_returns_empty_when_neither_present(self):
        assert _extract_observation_text(SimpleNamespace()) == ""

    def test_returns_empty_when_both_empty(self):
        obs = SimpleNamespace(text="", content="")

        assert _extract_observation_text(obs) == ""


class TestExtractTerminalExitCode:
    """``_extract_terminal_exit_code`` reads ``obs.metadata.exit_code`` defensively."""

    def test_returns_exit_code_when_metadata_present(self):
        obs = SimpleNamespace(metadata=SimpleNamespace(exit_code=0))

        assert _extract_terminal_exit_code(obs) == 0

    def test_non_zero_exit_code(self):
        obs = SimpleNamespace(metadata=SimpleNamespace(exit_code=127))

        assert _extract_terminal_exit_code(obs) == 127

    def test_returns_none_when_metadata_missing(self):
        assert _extract_terminal_exit_code(SimpleNamespace()) is None

    def test_returns_none_when_metadata_is_none(self):
        obs = SimpleNamespace(metadata=None)

        assert _extract_terminal_exit_code(obs) is None

    def test_returns_none_when_exit_code_missing_from_metadata(self):
        obs = SimpleNamespace(metadata=SimpleNamespace())

        assert _extract_terminal_exit_code(obs) is None


class TestFormatFixAtLineRange:
    """``_format_fix_at_line_range`` renders an edit range label."""

    def test_single_line_when_start_equals_end(self):
        obs = SimpleNamespace(start_line=7, end_line=7)

        assert _format_fix_at_line_range(obs) == "line 7"

    def test_range_when_start_differs_from_end(self):
        obs = SimpleNamespace(start_line=3, end_line=5)

        assert _format_fix_at_line_range(obs) == "lines 3-5"

    def test_fallback_when_start_missing(self):
        obs = SimpleNamespace(start_line=None, end_line=5)

        assert _format_fix_at_line_range(obs) == "range"

    def test_fallback_when_end_missing(self):
        obs = SimpleNamespace(start_line=3, end_line=None)

        assert _format_fix_at_line_range(obs) == "range"

    def test_fallback_when_both_missing(self):
        assert _format_fix_at_line_range(SimpleNamespace()) == "range"


# ============================================================================
# _is_fix_at_line_action() — narration-detection predicate
# ============================================================================
#
# This predicate is what tells ``_call_llm_list`` whether the agent actually
# invoked the ``fix_at_line`` tool (vs. narrating the call as text). The
# downstream fail-fast on narration depends on it being correct.


class TestIsFixAtLineAction:
    """``_is_fix_at_line_action`` returns True only for ``fix_at_line`` actions."""

    def test_action_event_for_fix_at_line_returns_true(self):
        event = MagicMock(spec=ActionEvent)
        event.tool_name = FixAtLineTool.name

        assert _is_fix_at_line_action(event) is True

    def test_action_event_for_file_editor_returns_false(self):
        event = MagicMock(spec=ActionEvent)
        event.tool_name = "file_editor"

        assert _is_fix_at_line_action(event) is False

    def test_action_event_for_terminal_returns_false(self):
        event = MagicMock(spec=ActionEvent)
        event.tool_name = "terminal"

        assert _is_fix_at_line_action(event) is False

    def test_action_event_for_task_tracker_returns_false(self):
        event = MagicMock(spec=ActionEvent)
        event.tool_name = "task_tracker"

        assert _is_fix_at_line_action(event) is False

    def test_message_event_returns_false_even_if_text_mentions_fix_at_line(self):
        """An agent that *narrates* fix_at_line in chat must NOT be counted.

        This is the regression we're guarding against: narration arrives as
        a MessageEvent, not an ActionEvent, so the predicate must say no.
        """
        event = MagicMock(spec=MessageEvent)
        # Even if some attribute happens to be named ``tool_name`` on a
        # MessageEvent, ``isinstance`` should still gate it out.
        event.tool_name = FixAtLineTool.name

        assert _is_fix_at_line_action(event) is False

    def test_observation_event_returns_false(self):
        """Observations come *after* a tool fires; they're not the firing."""
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = FixAtLineTool.name

        assert _is_fix_at_line_action(event) is False

    def test_arbitrary_object_returns_false(self):
        """Defensive: a stray object passed into the callback must not crash."""
        assert _is_fix_at_line_action(object()) is False

    def test_none_returns_false(self):
        """Defensive: a None event must not crash."""
        assert _is_fix_at_line_action(None) is False

    def test_action_event_without_tool_name_attribute_returns_false(self):
        """``getattr`` default handles a malformed ActionEvent without crashing."""

        class BareAction(ActionEvent):  # pragma: no cover — schema-only
            pass

        # Skip if ActionEvent can't be subclassed without args; fall back to
        # MagicMock with the attribute absent.
        event = MagicMock(spec=ActionEvent)
        del event.tool_name  # force the attribute to be missing

        assert _is_fix_at_line_action(event) is False


# ============================================================================
# _make_event_callback() — closure factory used by _call_llm_list
# ============================================================================
#
# The factory returns the conversation callback that (a) renders progress
# (delegating to _handle_agent_event) and (b) flips the
# ``fix_at_line_called`` flag when the tool actually fires.  Both behaviours
# must hold simultaneously — neither one alone is sufficient.


class TestMakeEventCallback:
    """``_make_event_callback`` builds the conversation event callback."""

    def test_returns_a_callable(self, capture_console):
        console, _ = capture_console
        cb = _make_event_callback(console, Path("/f.py"), [], [False])

        assert callable(cb)

    def test_fix_at_line_action_event_flips_flag(self, capture_console):
        console, _ = capture_console
        flag = [False]
        event = MagicMock(spec=ActionEvent)
        event.tool_name = FixAtLineTool.name
        event.thought = None

        cb = _make_event_callback(console, Path("/f.py"), [], flag)
        cb(event)

        assert flag[0] is True

    def test_file_editor_action_does_not_flip_flag(self, capture_console):
        console, _ = capture_console
        flag = [False]
        event = MagicMock(spec=ActionEvent)
        event.tool_name = "file_editor"
        event.thought = None

        cb = _make_event_callback(console, Path("/f.py"), [], flag)
        cb(event)

        assert flag[0] is False

    def test_terminal_action_does_not_flip_flag(self, capture_console):
        console, _ = capture_console
        flag = [False]
        event = MagicMock(spec=ActionEvent)
        event.tool_name = "terminal"
        event.thought = None

        cb = _make_event_callback(console, Path("/f.py"), [], flag)
        cb(event)

        assert flag[0] is False

    def test_message_event_does_not_flip_flag_even_when_narrating(self, capture_console):
        """Narration arrives as a MessageEvent — must not flip the flag."""
        console, _ = capture_console
        flag = [False]
        event = MagicMock(spec=MessageEvent)
        event.source = "agent"
        event.to_llm_message.return_value = SimpleNamespace(
            content="I'm now calling fix_at_line(start_line=42, ...)"
        )

        cb = _make_event_callback(console, Path("/f.py"), [], flag)
        cb(event)

        assert flag[0] is False

    def test_observation_event_does_not_flip_flag(self, capture_console):
        console, _ = capture_console
        flag = [False]
        event = MagicMock(spec=ObservationEvent)
        event.tool_name = FixAtLineTool.name
        event.observation = SimpleNamespace(
            text="ok", start_line=1, end_line=1, is_error=False, path="/f.py",
        )

        cb = _make_event_callback(console, Path("/f.py"), [], flag)
        cb(event)

        assert flag[0] is False

    def test_multiple_fix_at_line_actions_keep_flag_true(self, capture_console):
        """The flag is sticky: once True, subsequent events don't unset it."""
        console, _ = capture_console
        flag = [False]
        cb = _make_event_callback(console, Path("/f.py"), [], flag)

        first = MagicMock(spec=ActionEvent)
        first.tool_name = FixAtLineTool.name
        first.thought = None
        cb(first)

        second = MagicMock(spec=ActionEvent)
        second.tool_name = "file_editor"
        second.thought = None
        cb(second)

        assert flag[0] is True

    def test_callback_still_renders_to_console(self, capture_console):
        """The factory must preserve _handle_agent_event's rendering side."""
        console, buf = capture_console
        cb = _make_event_callback(console, Path("/f.py"), [], [False])

        event = MagicMock(spec=ActionEvent)
        event.tool_name = FixAtLineTool.name
        event.thought = None
        cb(event)

        assert "Calling" in buf.getvalue()
        assert "fix_at_line" in buf.getvalue()

    def test_callback_still_appends_agent_messages(self, capture_console):
        """The factory must preserve the message-accumulator side effect."""
        console, _ = capture_console
        messages: List[str] = []
        cb = _make_event_callback(console, Path("/f.py"), messages, [False])

        event = MagicMock(spec=MessageEvent)
        event.source = "agent"
        event.to_llm_message.return_value = SimpleNamespace(
            content="Fix applied on line 7."
        )
        cb(event)

        assert messages == ["Fix applied on line 7."]

    def test_callback_does_not_crash_on_arbitrary_object(self, capture_console):
        """Defensive: a non-Event object must not break the agent loop."""
        console, _ = capture_console
        flag = [False]
        cb = _make_event_callback(console, Path("/f.py"), [], flag)

        # Should not raise
        cb(object())  # type: ignore[arg-type]

        assert flag[0] is False

    def test_each_factory_call_returns_a_fresh_closure(self, capture_console):
        """Two callbacks must not share the flag; they own their own state."""
        console, _ = capture_console
        flag_a = [False]
        flag_b = [False]
        cb_a = _make_event_callback(console, Path("/f.py"), [], flag_a)
        cb_b = _make_event_callback(console, Path("/f.py"), [], flag_b)

        event = MagicMock(spec=ActionEvent)
        event.tool_name = FixAtLineTool.name
        event.thought = None
        cb_a(event)

        assert flag_a[0] is True
        assert flag_b[0] is False
