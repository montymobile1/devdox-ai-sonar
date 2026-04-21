"""Executor for the ``fix_at_line`` tool.

Self-contained: does not depend on any pre-OpenHands fix-application helpers.
Behaviour is purely "read file, validate range, splice lines, write file",
mirroring the semantics of ``file_editor str_replace`` but anchored by line
number instead of content match.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor

from devdox_ai_sonar.openhands_tools.fix_at_line.definition import (
    FixAtLineAction,
    FixAtLineObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation


class FixAtLineExecutor(ToolExecutor):
    """Splice an inclusive 1-indexed line range in a file."""

    def __call__(
        self,
        action: FixAtLineAction,
        conversation: "BaseConversation | None" = None,
    ) -> FixAtLineObservation:
        target = Path(action.path)

        if not target.is_absolute():
            return FixAtLineObservation.from_text(
                text=f"`path` must be absolute, got: {action.path!r}",
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
            )

        try:
            original_text = target.read_text(encoding="utf-8")
        except OSError as exc:
            return FixAtLineObservation.from_text(
                text=f"Cannot read file '{action.path}': {exc}",
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
            )

        lines = original_text.splitlines(keepends=True)
        total_lines = len(lines)

        if action.end_line > total_lines:
            return FixAtLineObservation.from_text(
                text=(
                    f"end_line {action.end_line} is out of range — file has "
                    f"{total_lines} lines."
                ),
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
            )

        start_idx = action.start_line - 1
        end_idx_exclusive = action.end_line  # Python slice upper bound.
        current_block_text = "".join(lines[start_idx:end_idx_exclusive])

        # Safety check — we compare after stripping a single trailing newline
        # on each side so callers don't have to worry about whether they
        # included the final \n in `old_block`.
        if _strip_trailing_newline(action.old_block) != _strip_trailing_newline(
            current_block_text
        ):
            return FixAtLineObservation.from_text(
                text=(
                    f"`old_block` does not match the current content of lines "
                    f"{action.start_line}-{action.end_line}.  Re-read the file "
                    f"with `file_editor view` and retry.\n"
                    f"--- old_block (provided) ---\n"
                    f"{_strip_trailing_newline(action.old_block)}\n"
                    f"--- actual lines {action.start_line}-{action.end_line} ---\n"
                    f"{_strip_trailing_newline(current_block_text)}"
                ),
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
                old_block=_strip_trailing_newline(current_block_text),
            )

        new_block_lines = _normalize_block_to_lines(action.new_block)
        new_file_lines = (
            lines[:start_idx] + new_block_lines + lines[end_idx_exclusive:]
        )
        new_text = "".join(new_file_lines)

        if new_text == original_text:
            return FixAtLineObservation.from_text(
                text=(
                    f"No change applied — `new_block` is identical to the "
                    f"current content of lines {action.start_line}-"
                    f"{action.end_line}."
                ),
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
            )

        try:
            target.write_text(new_text, encoding="utf-8")
        except OSError as exc:
            return FixAtLineObservation.from_text(
                text=f"Cannot write file '{action.path}': {exc}",
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
            )

        original_line_count = action.end_line - action.start_line + 1
        return FixAtLineObservation.from_text(
            text=(
                f"Replaced lines {action.start_line}-{action.end_line} in "
                f"{action.path} ({original_line_count} line(s) → "
                f"{len(new_block_lines)} line(s))."
            ),
            is_error=False,
            path=action.path,
            start_line=action.start_line,
            end_line=action.end_line,
            old_block=_strip_trailing_newline(current_block_text),
            new_block=_strip_trailing_newline(action.new_block),
        )


def _strip_trailing_newline(text: str) -> str:
    """Remove at most one trailing ``\\n``; leave other whitespace alone."""
    if text.endswith("\n"):
        return text[:-1]
    return text


def _normalize_block_to_lines(new_block: str) -> list[str]:
    """Split ``new_block`` into file-shaped lines that end in ``\\n``.

    Returns an empty list for the empty string so that an empty ``new_block``
    deletes the range cleanly.
    """
    if new_block == "":
        return []
    if not new_block.endswith("\n"):
        new_block = new_block + "\n"
    return new_block.splitlines(keepends=True)
