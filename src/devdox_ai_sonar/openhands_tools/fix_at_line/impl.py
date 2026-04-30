"""Executor for the ``fix_at_line`` tool.

Self-contained: does not depend on any pre-OpenHands fix-application helpers.
Behaviour is purely "read file, validate range, splice lines, write file",
mirroring the semantics of ``file_editor str_replace`` but anchored by line
number instead of content match.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor

from devdox_ai_sonar.openhands_tools.fix_at_line.definition import (
    FixAtLineAction,
    FixAtLineObservation,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation


class FixAtLineExecutor(ToolExecutor):
    """Splice an inclusive 1-indexed line range in a file."""

    def __call__(
        self,
        action: FixAtLineAction,
        conversation: "BaseConversation | None" = None,
    ) -> FixAtLineObservation:
        logger.info(
            "[fix_at_line-diagnostic] FixAtLineExecutor INVOKED: "
            "path=%s start_line=%s end_line=%s old_block_len=%s new_block_len=%s",
            action.path,
            action.start_line,
            action.end_line,
            len(action.old_block or ""),
            len(action.new_block or ""),
        )
        observation = self._run(action)
        logger.info(
            "[fix_at_line-diagnostic] FixAtLineExecutor RESULT: "
            "path=%s is_error=%s message=%s",
            action.path,
            getattr(observation, "is_error", None),
            (getattr(observation, "text", "") or "")[:200],
        )
        return observation

    def _run(
        self,
        action: FixAtLineAction,
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

        # Append-at-EOF: explicit "insert after end of file" semantic.
        # Agents commonly compute L from the last line shown by ``file_editor
        # view``, which can include a trailing-newline indicator (line N+1)
        # that the executor does not count. When the request is exactly
        # start==end==total+1 with empty old_block, treat it as a deliberate
        # append rather than rejecting it as out-of-range.
        if (
            action.start_line == total_lines + 1
            and action.end_line == total_lines + 1
            and action.old_block == ""
        ):
            new_text = original_text
            if new_text and not new_text.endswith("\n"):
                new_text += "\n"
            appended = action.new_block.rstrip("\n") + "\n" if action.new_block else ""
            new_text += appended

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

            new_block_line_count = len(_normalize_block_to_lines(action.new_block))
            syntax_status = (
                _run_py_compile(target) if target.suffix == ".py" else ""
            )
            message = (
                f"Appended {new_block_line_count} line(s) to {action.path} "
                f"(file was {total_lines} line(s))."
            )
            if syntax_status:
                message += f"\nSyntax: {syntax_status}"
            return FixAtLineObservation.from_text(
                text=message,
                is_error=False,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
                old_block="",
                new_block=_strip_trailing_newline(action.new_block),
            )

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

        old_stripped = _strip_trailing_newline(action.old_block)
        new_stripped = _strip_trailing_newline(action.new_block)
        current_stripped = _strip_trailing_newline(current_block_text)

        # An empty ``old_block`` is technically a substring of
        # everything -- reject it explicitly so the substring branch
        # below can't be tricked into runaway replacement.
        if old_stripped == "":
            return FixAtLineObservation.from_text(
                text=(
                    "`old_block` cannot be empty; supply either the exact "
                    f"current content of lines {action.start_line}-"
                    f"{action.end_line} or a unique substring of it."
                ),
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
            )

        # Three acceptable shapes for the safety check:
        #  (a) ``old_block`` equals the full line(s) -> replace the block.
        #  (b) ``old_block`` is a unique substring of the line(s) -> in-place
        #      find-and-replace inside the anchored range. The line range
        #      still bounds the search, so duplicated content elsewhere in
        #      the file is irrelevant.
        #  (c) ``old_block`` appears multiple times inside the range or
        #      isn't there at all -> error.
        if old_stripped == current_stripped:
            # (a) full-block replacement.
            new_block_lines = _normalize_block_to_lines(action.new_block)
            new_file_lines = (
                lines[:start_idx]
                + new_block_lines
                + lines[end_idx_exclusive:]
            )
            new_text = "".join(new_file_lines)
            new_block_line_count = len(new_block_lines)
            returned_old_block = current_stripped
            returned_new_block = new_stripped
        elif old_stripped in current_stripped:
            occurrences = current_stripped.count(old_stripped)
            if occurrences > 1:
                return FixAtLineObservation.from_text(
                    text=(
                        f"`old_block` appears {occurrences} times within "
                        f"lines {action.start_line}-{action.end_line}; "
                        f"cannot decide which occurrence to replace. Widen "
                        f"`old_block` to include enough surrounding context "
                        f"to be unique, or narrow the line range."
                    ),
                    is_error=True,
                    path=action.path,
                    start_line=action.start_line,
                    end_line=action.end_line,
                    old_block=current_stripped,
                )
            # (b) substring match, unique within the range.
            new_block_text = current_stripped.replace(
                old_stripped, new_stripped, 1
            )
            if current_block_text.endswith("\n"):
                new_block_text += "\n"
            prefix = "".join(lines[:start_idx])
            suffix = "".join(lines[end_idx_exclusive:])
            new_text = prefix + new_block_text + suffix
            new_block_line_count = len(
                _normalize_block_to_lines(new_block_text)
            )
            returned_old_block = current_stripped
            returned_new_block = _strip_trailing_newline(new_block_text)
        else:
            # (c) no match at all.
            return FixAtLineObservation.from_text(
                text=(
                    f"`old_block` does not match the current content of lines "
                    f"{action.start_line}-{action.end_line}.  Re-read the file "
                    f"with `file_editor view` and retry.\n"
                    f"--- old_block (provided) ---\n"
                    f"{old_stripped}\n"
                    f"--- actual lines {action.start_line}-{action.end_line} ---\n"
                    f"{current_stripped}"
                ),
                is_error=True,
                path=action.path,
                start_line=action.start_line,
                end_line=action.end_line,
                old_block=current_stripped,
            )

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
        syntax_status = (
            _run_py_compile(target) if target.suffix == ".py" else ""
        )
        message = (
            f"Replaced lines {action.start_line}-{action.end_line} in "
            f"{action.path} ({original_line_count} line(s) → "
            f"{new_block_line_count} line(s))."
        )
        if syntax_status:
            message += f"\nSyntax: {syntax_status}"
        return FixAtLineObservation.from_text(
            text=message,
            is_error=False,
            path=action.path,
            start_line=action.start_line,
            end_line=action.end_line,
            old_block=returned_old_block,
            new_block=returned_new_block,
        )


def _run_py_compile(target: Path) -> str:
    """Run ``py_compile`` and (best-effort) ``pyflakes`` on ``target``.

    Returns "OK" only when both pass. ``py_compile`` catches structural
    syntax errors. ``pyflakes`` (when available) additionally catches
    undefined-name references — the failure mode of an incomplete refactor
    where the simplified function calls helpers that haven't been appended
    yet. ``py_compile`` would report SYNTAX_OK on that file, but a real
    runtime call would raise ``NameError``; the pyflakes pass closes that
    gap.

    "FAILED — <reason>" on either syntax or undefined-name failure.
    Returns a non-empty status even on subprocess errors so the agent
    always sees feedback. Pyflakes is optional: if it's not installed,
    the undefined-name check is silently skipped.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(target)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"FAILED — py_compile error: {exc}"
    if result.returncode != 0:
        err_lines = [
            line for line in (result.stderr or "").splitlines() if line.strip()
        ]
        return f"FAILED — {err_lines[-1] if err_lines else 'unknown error'}"

    undefined = _run_pyflakes_undefined_names(target)
    if undefined:
        return f"FAILED — {undefined}"
    return "OK"


def _run_pyflakes_undefined_names(target: Path) -> str:
    """Best-effort undefined-name check via ``python -m pyflakes``.

    Returns the first ``undefined name`` finding (one short line) or "" if
    pyflakes is unavailable, times out, or finds no undefined-name issues.
    Other pyflakes warnings (unused imports, redefinitions, …) are ignored
    intentionally — we only want to catch the case where a refactor left
    a name unresolved.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", str(target)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    output_lines = (
        (result.stdout or "") + (result.stderr or "")
    ).splitlines()
    for line in output_lines:
        if "undefined name" in line.lower():
            return line.strip()
    return ""


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
