"""Unit tests for the ``fix_at_line`` tool executor.

Strategy (per ``how to test properly``): inject nothing, mock nothing.  The
executor is a plain callable whose entire contract is observable from its
inputs (the action) and its outputs (the observation plus the bytes on disk),
so every test drives it with a real ``tmp_path`` file and asserts on both.
"""

from pathlib import Path

import pytest

from devdox_ai_sonar.openhands_tools.fix_at_line.definition import FixAtLineAction
from devdox_ai_sonar.openhands_tools.fix_at_line.impl import FixAtLineExecutor


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def executor() -> FixAtLineExecutor:
    return FixAtLineExecutor()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A 5-line file used by most of the happy-path tests."""
    path = tmp_path / "sample.py"
    path.write_text(
        "import os\n"
        "\n"
        "def hello():\n"
        "    return 'world'\n"
        "\n"
    )
    return path


def _action(path: Path, start: int, end: int, old: str, new: str) -> FixAtLineAction:
    """Convenience wrapper to build an action with an absolute path."""
    return FixAtLineAction(
        path=str(path.resolve()),
        start_line=start,
        end_line=end,
        old_block=old,
        new_block=new,
    )


# ============================================================================
# HAPPY PATHS — SINGLE- AND MULTI-LINE REPLACEMENTS
# ============================================================================


class TestFixAtLineExecutorSuccess:
    """Successful edits: file content, observation fields, and size deltas."""

    def test_single_line_replace(self, executor, sample_file):
        action = _action(
            sample_file, 3, 3,
            old="def hello():",
            new="def greet():",
        )

        obs = executor(action)

        assert obs.is_error is False
        assert obs.start_line == 3
        assert obs.end_line == 3
        assert sample_file.read_text().splitlines()[2] == "def greet():"

    def test_multi_line_replace_same_length(self, executor, sample_file):
        action = _action(
            sample_file, 3, 4,
            old="def hello():\n    return 'world'",
            new="def greet():\n    return 'hi'",
        )

        obs = executor(action)

        assert obs.is_error is False
        new_lines = sample_file.read_text().splitlines()
        assert new_lines[2] == "def greet():"
        assert new_lines[3] == "    return 'hi'"

    def test_replacement_grows_the_file(self, executor, sample_file):
        original_line_count = len(sample_file.read_text().splitlines())
        action = _action(
            sample_file, 3, 4,
            old="def hello():\n    return 'world'",
            new=(
                "def greet(name):\n"
                "    greeting = 'hello, '\n"
                "    message = greeting + name\n"
                "    return message"
            ),
        )

        obs = executor(action)

        assert obs.is_error is False
        assert len(sample_file.read_text().splitlines()) == original_line_count + 2

    def test_replacement_shrinks_the_file(self, executor, sample_file):
        original_line_count = len(sample_file.read_text().splitlines())
        action = _action(
            sample_file, 3, 4,
            old="def hello():\n    return 'world'",
            new="hello = lambda: 'world'",
        )

        obs = executor(action)

        assert obs.is_error is False
        assert len(sample_file.read_text().splitlines()) == original_line_count - 1

    def test_empty_new_block_deletes_the_range(self, executor, sample_file):
        original_line_count = len(sample_file.read_text().splitlines())
        action = _action(
            sample_file, 3, 4,
            old="def hello():\n    return 'world'",
            new="",
        )

        obs = executor(action)

        assert obs.is_error is False
        assert len(sample_file.read_text().splitlines()) == original_line_count - 2

    def test_indentation_is_preserved_verbatim(self, executor, tmp_path):
        path = tmp_path / "indent.py"
        path.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 1\n"
        )
        action = _action(
            path, 3, 3,
            old="        return 1",
            new="        return 2  # updated",
        )

        obs = executor(action)

        assert obs.is_error is False
        assert "        return 2  # updated\n" in path.read_text()

    def test_unicode_content_is_preserved(self, executor, tmp_path):
        path = tmp_path / "unicode.py"
        path.write_text("GREETING = 'héllo 🌍'\n")
        action = _action(
            path, 1, 1,
            old="GREETING = 'héllo 🌍'",
            new="GREETING = 'hôlà 🚀'",
        )

        obs = executor(action)

        assert obs.is_error is False
        assert path.read_text() == "GREETING = 'hôlà 🚀'\n"

    def test_file_without_trailing_newline_is_handled(self, executor, tmp_path):
        """A file whose last line has no trailing \\n must still be editable."""
        path = tmp_path / "no_trail.py"
        path.write_text("a = 1")  # no trailing newline
        action = _action(
            path, 1, 1,
            old="a = 1",
            new="a = 2",
        )

        obs = executor(action)

        assert obs.is_error is False
        assert path.read_text() == "a = 2\n"

    def test_old_block_with_trailing_newline_is_accepted(self, executor, sample_file):
        """A trailing \\n on old_block must not break the safety check."""
        action = _action(
            sample_file, 3, 3,
            old="def hello():\n",
            new="def greet():",
        )

        obs = executor(action)

        assert obs.is_error is False

    def test_old_block_without_trailing_newline_is_accepted(self, executor, sample_file):
        """Absence of trailing \\n on old_block must also work."""
        action = _action(
            sample_file, 3, 3,
            old="def hello():",
            new="def greet():",
        )

        obs = executor(action)

        assert obs.is_error is False


# ============================================================================
# THE SIGNATURE TEST — DUPLICATED LINES
# ============================================================================


class TestFixAtLineExecutorDuplicateLines:
    """The tests that demonstrate we've fixed the original bug.

    A file contains the same line three times.  The stock OpenHands
    ``str_replace`` command would refuse to edit because ``old_str`` matches
    in multiple places.  ``fix_at_line`` targets the ONE occurrence named
    by ``start_line`` / ``end_line`` and leaves the others untouched.
    """

    @pytest.fixture
    def triplicate_file(self, tmp_path):
        path = tmp_path / "triplicate.py"
        path.write_text(
            'response = get(url, headers={"Content-Type": "application/json"})\n'
            'response = get(url, headers={"Content-Type": "application/json"})\n'
            'response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        return path

    def test_edits_only_the_requested_line(self, executor, triplicate_file):
        """Replace line 2 only; lines 1 and 3 remain byte-for-byte identical."""
        action = _action(
            triplicate_file, 2, 2,
            old='response = get(url, headers={"Content-Type": "application/json"})',
            new='response = get(url, headers={"Content-Type": APP_JSON})',
        )

        obs = executor(action)

        assert obs.is_error is False
        lines = triplicate_file.read_text().splitlines()
        assert lines[0] == 'response = get(url, headers={"Content-Type": "application/json"})'
        assert lines[1] == 'response = get(url, headers={"Content-Type": APP_JSON})'
        assert lines[2] == 'response = get(url, headers={"Content-Type": "application/json"})'

    def test_can_edit_any_of_the_duplicates_in_sequence(self, executor, triplicate_file):
        """Three separate edits, one per duplicate, all succeed."""
        executor(_action(
            triplicate_file, 3, 3,
            old='response = get(url, headers={"Content-Type": "application/json"})',
            new='response = C',
        ))
        executor(_action(
            triplicate_file, 2, 2,
            old='response = get(url, headers={"Content-Type": "application/json"})',
            new='response = B',
        ))
        executor(_action(
            triplicate_file, 1, 1,
            old='response = get(url, headers={"Content-Type": "application/json"})',
            new='response = A',
        ))

        assert triplicate_file.read_text() == (
            "response = A\n"
            "response = B\n"
            "response = C\n"
        )


# ============================================================================
# SUBSTRING REPLACEMENT — `old_block` is a piece of the line, not the whole line
# ============================================================================


class TestFixAtLineExecutorSubstring:
    """When ``old_block`` is a unique substring of the anchored line(s),
    do an in-place find-and-replace instead of rejecting the edit.

    The line range is still the anchor -- the search space is bounded
    to those lines, so the whole-file ambiguity problem fix_at_line was
    built for does not come back.
    """

    def test_substring_unique_within_line_is_replaced(
        self, executor, tmp_path
    ):
        path = tmp_path / "sub.py"
        path.write_text('    value = greet("hello")\n')

        obs = executor(_action(
            path, 1, 1,
            old='"hello"',
            new="GREETING",
        ))

        assert obs.is_error is False
        assert path.read_text() == '    value = greet(GREETING)\n'

    def test_substring_ambiguous_within_line_is_rejected(
        self, executor, tmp_path
    ):
        """Two copies of the same fragment on one line -> we don't
        know which to replace. Error; file untouched."""
        path = tmp_path / "ambig.py"
        original = '    value = "hello" + "hello"\n'
        path.write_text(original)

        obs = executor(_action(
            path, 1, 1,
            old='"hello"',
            new='HELLO',
        ))

        assert obs.is_error is True
        assert path.read_text() == original
        # The error tells the agent exactly why it's ambiguous and
        # what to do about it.
        assert "2" in obs.text or "multiple" in obs.text.lower()

    def test_substring_replacement_preserves_trailing_newline(
        self, executor, tmp_path
    ):
        """A substring edit must leave the file's trailing newline byte
        untouched -- regressions here corrupt diffs on every fix."""
        path = tmp_path / "nl.py"
        path.write_text('    x = greet("hello")\n')
        original_ends_with_newline = path.read_bytes().endswith(b"\n")
        assert original_ends_with_newline  # fixture sanity

        obs = executor(_action(
            path, 1, 1,
            old='"hello"',
            new='GREETING',
        ))

        assert obs.is_error is False
        assert path.read_bytes().endswith(b"\n")

    def test_empty_old_block_is_rejected(self, executor, sample_file):
        """An empty string is technically a substring of anything --
        guard the substring branch so the agent can't trigger runaway
        replacement with an empty payload."""
        original = sample_file.read_text()

        obs = executor(_action(
            sample_file, 3, 3,
            old="",
            new="anything",
        ))

        assert obs.is_error is True
        assert sample_file.read_text() == original


# ============================================================================
# ERROR CASES — INVALID INPUTS & INFEASIBLE EDITS
# ============================================================================


class TestFixAtLineExecutorErrors:
    """Errors surface as ``is_error=True`` observations and never mutate disk."""

    def test_relative_path_is_rejected(self, executor):
        action = FixAtLineAction(
            path="relative/path.py",
            start_line=1, end_line=1,
            old_block="x", new_block="y",
        )

        obs = executor(action)

        assert obs.is_error is True
        assert "absolute" in obs.text.lower()

    def test_missing_file_returns_error(self, executor, tmp_path):
        missing = tmp_path / "does_not_exist.py"
        action = _action(missing, 1, 1, old="foo", new="bar")

        obs = executor(action)

        assert obs.is_error is True
        assert "cannot read" in obs.text.lower()

    def test_end_line_beyond_file_length_is_rejected(self, executor, sample_file):
        action = _action(
            sample_file, 1, 100,
            old="whatever", new="whatever",
        )

        obs = executor(action)

        assert obs.is_error is True
        assert "out of range" in obs.text.lower()

    def test_old_block_mismatch_returns_error(self, executor, sample_file):
        original = sample_file.read_text()
        action = _action(
            sample_file, 3, 3,
            old="def goodbye():",
            new="def greet():",
        )

        obs = executor(action)

        assert obs.is_error is True
        assert sample_file.read_text() == original

    def test_old_block_mismatch_does_not_mutate_file_bytes(self, executor, sample_file):
        """Extra paranoia: compare raw bytes too, not just text."""
        original_bytes = sample_file.read_bytes()

        executor(_action(
            sample_file, 3, 3,
            old="wrong content",
            new="anything",
        ))

        assert sample_file.read_bytes() == original_bytes

    def test_no_op_edit_is_flagged_as_error(self, executor, sample_file):
        """If ``new_block`` reproduces the current content, report it."""
        action = _action(
            sample_file, 3, 3,
            old="def hello():",
            new="def hello():",
        )

        obs = executor(action)

        assert obs.is_error is True
        assert "no change" in obs.text.lower()


# ============================================================================
# ERROR DIAGNOSTICS — ENOUGH CONTEXT FOR THE AGENT TO SELF-CORRECT
# ============================================================================


class TestFixAtLineExecutorErrorDiagnostics:
    """Errors returned to the agent must be actionable, not opaque."""

    def test_mismatch_error_includes_expected_and_actual(self, executor, sample_file):
        action = _action(
            sample_file, 3, 3,
            old="def goodbye():",
            new="def greet():",
        )

        obs = executor(action)

        assert "def goodbye():" in obs.text  # what the agent THOUGHT was there
        assert "def hello():" in obs.text    # what's actually on line 3

    def test_mismatch_observation_populates_actual_old_block(self, executor, sample_file):
        action = _action(
            sample_file, 3, 3,
            old="wrong content",
            new="new content",
        )

        obs = executor(action)

        assert obs.old_block == "def hello():"

    def test_out_of_range_error_names_the_actual_length(self, executor, sample_file):
        action = _action(sample_file, 1, 9999, old="x", new="y")

        obs = executor(action)

        # sample_file has 5 lines (4 non-blank + 1 blank at the end).
        assert "5" in obs.text

    def test_success_observation_reports_line_count_delta(self, executor, sample_file):
        """Message calls out how many lines went in vs. came out."""
        action = _action(
            sample_file, 3, 4,
            old="def hello():\n    return 'world'",
            new="x = 1",
        )

        obs = executor(action)

        assert obs.is_error is False
        assert "2 line" in obs.text
        assert "1 line" in obs.text
