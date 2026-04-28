"""Integration tests for narration fail-fast inside ``LLMFixer._call_llm_list``.

These tests cover the runtime guard added to detect when the agent narrates a
``fix_at_line`` invocation as text instead of invoking the tool. The guard
relies on:

* :func:`devdox_ai_sonar.llm_fixer._is_fix_at_line_action` — the predicate.
* :func:`devdox_ai_sonar.llm_fixer._make_event_callback` — the callback that
  flips the ``fix_at_line_called`` flag when a real ActionEvent fires.
* The post-``conversation.run()`` block in ``_call_llm_list`` that returns
  ``None`` when the flag is still ``False``.

Lower-level unit coverage of the predicate and the callback factory lives in
``tests/test_llm_fixer_wiring.py``.  This module covers the integration: that
``_call_llm_list`` actually returns ``None`` (not a ``SonarFixResponse`` with
unchanged content) when the agent doesn't invoke the tool.

Strategy: the OpenHands ``Conversation`` and ``Agent`` are heavyweight; we
substitute them via ``unittest.mock.patch`` so we can capture the callback
list passed to ``Conversation(...)`` and drive the event stream
deterministically inside the patched ``run()``.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from openhands.sdk.event import ActionEvent, MessageEvent

from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.models.file_structures import FixContext
from devdox_ai_sonar.models.sonar import SonarIssue
from devdox_ai_sonar.openhands_tools.fix_at_line import FixAtLineTool


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_llm_client():
    """Mock the OpenHands LLM client so LLMFixer construction doesn't try
    to authenticate against a real provider."""
    with patch("devdox_ai_sonar.llm_fixer.LLM") as mock:
        yield mock


@pytest.fixture
def fixer(mock_llm_client) -> LLMFixer:
    """An LLMFixer with provider=gemini and a mocked client.

    Provider choice is arbitrary — every provider goes through the same
    ``LLM`` patch above. ``model`` is set so the early ``if not self.model``
    guard does not return ``None`` before our code path runs.
    """
    fx = LLMFixer(provider="gemini", api_key="test-key")
    fx.model = "gemini/gemini-1.5-flash"
    return fx


@pytest.fixture
def fix_target_file(tmp_path) -> Path:
    """A real file the fixer can ``read_text()``. Content is deliberately
    trivial — the agent's job is mocked, not exercised."""
    f = tmp_path / "target.py"
    f.write_text("x = 1\n")
    return f


@pytest.fixture
def basic_context(fix_target_file) -> FixContext:
    return FixContext(
        file_path=fix_target_file,
        file_path_tmp=fix_target_file,
        line_range={"first_line": 1, "last_line": 1},
        code_content="x = 1",
        language="python",
        import_section={
            "start_line": 0,
            "end_line": 0,
            "content": "",
            "has_imports": False,
        },
        class_name=None,
        functions=[],
        context_dict={
            "start_line": 1,
            "end_line": 1,
            "context": "x = 1",
            "new_context": [
                {"context": "x = 1", "start_line": 1, "end_line": 1}
            ],
            "problem_line_content": {1: "x = 1\n"},
        },
    )


@pytest.fixture
def basic_issue() -> SonarIssue:
    """A vanilla SonarIssue. Rule key is deliberately not one of the
    special-cased rules (S3776, S1172) so ``_create_fix_prompt_list`` takes
    the default path."""
    return SonarIssue(
        key="t-1",
        rule="python:S1234",
        severity="MAJOR",
        component="t.py",
        project="t",
        message="t",
        type="CODE_SMELL",
        status="OPEN",
        first_line=1,
        last_line=1,
    )


# ============================================================================
# HELPERS
# ============================================================================


def _action_event_for(tool_name: str) -> MagicMock:
    event = MagicMock(spec=ActionEvent)
    event.tool_name = tool_name
    event.thought = None
    return event


def _message_event(text: str, source: str = "agent") -> MagicMock:
    event = MagicMock(spec=MessageEvent)
    event.source = source
    event.to_llm_message.return_value = SimpleNamespace(content=text)
    return event


def _patch_agent_loop(
    run_drives_callback: Callable[[Callable[[Any], None]], None],
) -> Tuple[Any, Any]:
    """Build context-managers patching ``Conversation`` and ``Agent``.

    The patched ``Conversation`` captures the callback list and arranges for
    ``run()`` to invoke ``run_drives_callback(cb)`` — i.e. the test author
    decides what events arrive between ``send_message`` and the post-run
    fail-fast check.
    """

    def _conversation_factory(*args: Any, **kwargs: Any) -> MagicMock:
        callbacks: List[Callable[[Any], None]] = list(kwargs.get("callbacks", []))
        cb = callbacks[0] if callbacks else (lambda _e: None)

        mock_conv = MagicMock()
        mock_conv.run.side_effect = lambda: run_drives_callback(cb)
        return mock_conv

    return (
        patch("devdox_ai_sonar.llm_fixer.Conversation", side_effect=_conversation_factory),
        patch("devdox_ai_sonar.llm_fixer.Agent"),
    )


# ============================================================================
# Narration → fail-fast (return None)
# ============================================================================


class TestNarrationFailFast:
    """``_call_llm_list`` must return ``None`` when the agent finishes
    without invoking ``fix_at_line``."""

    def test_returns_none_when_run_emits_no_events(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path
    ):
        """Empty conversation: ``run()`` returns immediately, no events fire.
        The fail-fast guard must catch this."""

        def run(_cb):
            return None  # the agent did literally nothing

        conv_patch, agent_patch = _patch_agent_loop(run)
        with conv_patch, agent_patch:
            result = fixer._call_llm_list(
                [basic_issue], basic_context, ".py", {},
                tmp_path, fix_target_file, "",
            )

        assert result is None

    def test_returns_none_when_only_message_events_fire(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path
    ):
        """The bug we're guarding against: agent narrates the call as text.

        From the runtime's perspective, narration arrives as a
        ``MessageEvent`` with no corresponding ``ActionEvent``. Even if the
        chat text contains the literal word "fix_at_line", the file is
        unchanged and we must report failure."""

        def run(cb):
            cb(_message_event(
                "I'm calling fix_at_line(start_line=1, end_line=1, "
                "old_block='x = 1', new_block='x = 2')"
            ))

        conv_patch, agent_patch = _patch_agent_loop(run)
        with conv_patch, agent_patch:
            result = fixer._call_llm_list(
                [basic_issue], basic_context, ".py", {},
                tmp_path, fix_target_file, "",
            )

        assert result is None

    def test_returns_none_when_only_other_tools_fire(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path
    ):
        """Subtler narration: the agent did *some* tool work — viewed the
        file, ran the terminal — but never invoked ``fix_at_line`` itself.
        Still failure."""

        def run(cb):
            cb(_action_event_for("file_editor"))
            cb(_action_event_for("terminal"))

        conv_patch, agent_patch = _patch_agent_loop(run)
        with conv_patch, agent_patch:
            result = fixer._call_llm_list(
                [basic_issue], basic_context, ".py", {},
                tmp_path, fix_target_file, "",
            )

        assert result is None

    def test_returns_none_when_observation_event_for_fix_at_line_fires_without_action(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path
    ):
        """Defensive: an ``ObservationEvent`` referencing fix_at_line is NOT
        the same as an ``ActionEvent`` — the observation is what comes back
        *after* the tool fires. Without the originating ActionEvent, the
        flag must remain unset."""

        def run(cb):
            from openhands.sdk.event import ObservationEvent

            obs_event = MagicMock(spec=ObservationEvent)
            obs_event.tool_name = FixAtLineTool.name
            obs_event.observation = SimpleNamespace(
                text="ok", start_line=1, end_line=1, is_error=False, path="/x.py",
            )
            cb(obs_event)

        conv_patch, agent_patch = _patch_agent_loop(run)
        with conv_patch, agent_patch:
            result = fixer._call_llm_list(
                [basic_issue], basic_context, ".py", {},
                tmp_path, fix_target_file, "",
            )

        assert result is None

    def test_logs_warning_identifying_the_file_on_narration(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path, caplog
    ):
        """The fail-fast must produce a visible warning so silent failure
        becomes loud failure."""

        def run(_cb):
            return None

        conv_patch, agent_patch = _patch_agent_loop(run)
        with caplog.at_level(logging.WARNING, logger="devdox_ai_sonar.llm_fixer"):
            with conv_patch, agent_patch:
                fixer._call_llm_list(
                    [basic_issue], basic_context, ".py", {},
                    tmp_path, fix_target_file, "",
                )

        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        ]
        joined = " | ".join(warnings)

        assert any("narration" in w.lower() or "fix_at_line" in w for w in warnings), (
            f"Expected a narration/fix_at_line warning, got: {joined!r}"
        )
        assert fix_target_file.name in joined


# ============================================================================
# Happy path: ActionEvent for fix_at_line fires → continue past fail-fast
# ============================================================================


class TestHappyPath:
    """When the agent really does invoke ``fix_at_line``, ``_call_llm_list``
    must NOT short-circuit on the new guard."""

    def test_returns_response_when_fix_at_line_action_fires(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path
    ):
        """An ActionEvent for fix_at_line during the run flips the flag,
        the post-run guard does not short-circuit, the file is read back,
        and a SonarFixResponse is returned."""

        def run(cb):
            cb(_action_event_for(FixAtLineTool.name))
            # Simulate the agent having modified the file
            fix_target_file.write_text("x = 2\n")

        conv_patch, agent_patch = _patch_agent_loop(run)
        with conv_patch, agent_patch:
            result = fixer._call_llm_list(
                [basic_issue], basic_context, ".py", {},
                tmp_path, fix_target_file, "",
            )

        assert result is not None
        assert result.applied_by_agent is True

    def test_returns_response_when_action_fires_among_other_events(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path
    ):
        """The fix_at_line ActionEvent doesn't have to be alone — a realistic
        run mixes view actions, the fix_at_line action, terminal verification,
        and final agent text."""

        def run(cb):
            cb(_action_event_for("file_editor"))
            cb(_action_event_for(FixAtLineTool.name))
            cb(_action_event_for("terminal"))
            cb(_message_event("Replaced line 1; SYNTAX_OK."))
            fix_target_file.write_text("x = 2\n")

        conv_patch, agent_patch = _patch_agent_loop(run)
        with conv_patch, agent_patch:
            result = fixer._call_llm_list(
                [basic_issue], basic_context, ".py", {},
                tmp_path, fix_target_file, "",
            )

        assert result is not None
        # The agent's last message should propagate as the EXPLANATION.
        assert "SYNTAX_OK" in result.EXPLANATION

    def test_does_not_log_narration_warning_on_happy_path(
        self, fixer, basic_context, basic_issue, fix_target_file, tmp_path, caplog
    ):
        """No narration warning when fix_at_line fired."""

        def run(cb):
            cb(_action_event_for(FixAtLineTool.name))
            fix_target_file.write_text("x = 2\n")

        conv_patch, agent_patch = _patch_agent_loop(run)
        with caplog.at_level(logging.WARNING, logger="devdox_ai_sonar.llm_fixer"):
            with conv_patch, agent_patch:
                fixer._call_llm_list(
                    [basic_issue], basic_context, ".py", {},
                    tmp_path, fix_target_file, "",
                )

        narration_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "narration" in r.getMessage().lower()
        ]
        assert narration_warnings == []
