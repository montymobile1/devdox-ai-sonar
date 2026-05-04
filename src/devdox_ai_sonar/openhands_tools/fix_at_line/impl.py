"""Executor for the ``fix_at_line`` tool.

Self-contained: does not depend on any pre-OpenHands fix-application helpers.
Behaviour is purely "read file, validate range, splice lines, write file",
mirroring the semantics of ``file_editor str_replace`` but anchored by line
number instead of content match.
"""

import ast
import logging
import re
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

        # Snapshot pre-edit syntax for Python files. Used after the write
        # to detect the FAILED -> FAILED cascade, where the agent keeps
        # piling edits onto an already-broken file. Cheap (~ms) and only
        # runs once per call.
        pre_edit_syntax = (
            _run_py_compile(target) if target.suffix == ".py" else "OK"
        )

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
            # Drop trailing whitespace from the existing content before
            # composing the appended block. Lets us decide separator
            # newlines deterministically from a known-clean baseline.
            base_text = original_text.rstrip("\n")
            if action.new_block:
                appended_block = action.new_block.rstrip("\n") + "\n"
            else:
                appended_block = ""
            if (
                appended_block
                and target.suffix == ".py"
                and _starts_with_top_level_python_def(action.new_block)
            ):
                # PEP 8 wants two blank lines between top-level defs/classes.
                # The trailing-newline-stripped base + "\n\n\n" yields:
                # last-content-line\n  +  \n  +  \n  =  two blank lines, then
                # the appended block.
                separator = "\n\n\n"
            elif appended_block:
                separator = "\n"
            else:
                # Empty new_block — preserve a trailing newline so subsequent
                # appends keep behaving sensibly.
                separator = "\n" if base_text else ""
            new_text = base_text + separator + appended_block

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
            cascade_revert = _revert_if_syntax_cascade(
                target,
                pre_edit_syntax,
                syntax_status,
                original_text,
                action,
            )
            if cascade_revert is not None:
                return cascade_revert
            new_total_lines = len(new_text.splitlines())
            message = (
                f"Appended {new_block_line_count} line(s) to {action.path} "
                f"(file is now {new_total_lines} line(s); "
                f"next EOF append uses start_line=end_line={new_total_lines + 1})."
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

        # Empty ``old_block`` on a valid in-range slot is the
        # "trust my line range" signal: the agent has just viewed the
        # file and is replacing the whole [start_line..end_line] window
        # without asking the executor to byte-verify the existing
        # content. Necessary because some LLMs (e.g., Llama 3.3) cannot
        # reliably reproduce long multi-line strings as tool arguments
        # — they emit literal "..." stubs or mangle whitespace, which
        # makes the byte-match path always fail. Skipping the byte
        # check here is safe because (a) the line range itself is the
        # anchor, (b) the post-edit syntax check + cascade revert below
        # catches structurally-broken results.
        if old_stripped == "":
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
        # Three acceptable shapes for the safety check when ``old_block``
        # is non-empty:
        #  (a) ``old_block`` equals the full line(s) -> replace the block.
        #  (b) ``old_block`` is a unique substring of the line(s) -> in-place
        #      find-and-replace inside the anchored range. The line range
        #      still bounds the search, so duplicated content elsewhere in
        #      the file is irrelevant.
        #  (c) ``old_block`` appears multiple times inside the range or
        #      isn't there at all -> error.
        elif old_stripped == current_stripped:
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
            if target.suffix == ".py" and _new_block_adds_python_structure(
                old_stripped, new_stripped
            ):
                return FixAtLineObservation.from_text(
                    text=(
                        "`new_block` introduces def/class/@decorator "
                        "definitions absent from `old_block`. Substring "
                        "replacement would splice those into the middle of "
                        "existing statements and corrupt the file. Either:\n"
                        "  (a) set `old_block` to the empty string to "
                        f"replace the full line range "
                        f"{action.start_line}-{action.end_line} without "
                        "byte-verifying the existing content (safe when "
                        "you have just `view`ed the file), or\n"
                        "  (b) widen `old_block` to the exact full "
                        "content of those lines for verified full-block "
                        "replacement, or\n"
                        "  (c) use the EOF-append pattern "
                        "(start_line=end_line=total_lines+1, "
                        "old_block='') to append new helpers after the "
                        "end of the file."
                    ),
                    is_error=True,
                    path=action.path,
                    start_line=action.start_line,
                    end_line=action.end_line,
                    old_block=current_stripped,
                )
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
        cascade_revert = _revert_if_syntax_cascade(
            target,
            pre_edit_syntax,
            syntax_status,
            original_text,
            action,
        )
        if cascade_revert is not None:
            return cascade_revert
        new_total_lines = len(new_text.splitlines())
        message = (
            f"Replaced lines {action.start_line}-{action.end_line} in "
            f"{action.path} ({original_line_count} line(s) → "
            f"{new_block_line_count} line(s); file is now "
            f"{new_total_lines} line(s); next EOF append uses "
            f"start_line=end_line={new_total_lines + 1})."
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

    duplicate = _check_no_duplicate_class_methods(target)
    if duplicate:
        return f"FAILED — {duplicate}"

    undefined = _run_pyflakes_undefined_names(target)
    if undefined:
        return f"FAILED — {undefined}"
    return "OK"


def _check_no_duplicate_class_methods(target: Path) -> str:
    """Return a non-empty failure description if any class in
    ``target`` defines the same method name twice.

    Catches the failure mode where an agent refactor appends new
    method definitions whose names collide with existing methods
    elsewhere in the same class body — typically because the agent
    only saw a partial view of the file (``file_editor view``
    truncates large files) and invented helper names without
    grounding them in the class's actual API. The duplicate
    silently shadows the original; the agent's call sites land on
    the new (often broken) body.

    ``py_compile`` cannot catch this because the file is
    syntactically valid. ``pyflakes`` cannot catch it because the
    references are method-level (``self.X``), not module-level.

    Returns "" when no duplicates are found.
    """
    try:
        source = target.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # py_compile already covers this case; don't double-report.
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        seen: dict = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name in seen:
                    return (
                        f"class '{node.name}' defines method "
                        f"'{item.name}' twice (lines "
                        f"{seen[item.name]} and {item.lineno}); the "
                        f"later definition shadows the earlier one. "
                        f"Remove one of the two definitions."
                    )
                seen[item.name] = item.lineno
    return ""


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


_PYTHON_STRUCTURE_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|@\w)", re.MULTILINE
)

_TOP_LEVEL_PYTHON_DEF_RE = re.compile(
    r"^(?:async\s+)?(?:def|class|@\w)"
)


def _starts_with_top_level_python_def(block: str) -> bool:
    """True when ``block`` opens with a column-0 ``def`` / ``class`` / ``@dec``.

    Used by the EOF-append path to decide whether to insert PEP 8's
    canonical two blank lines between the existing content and the
    appended block. Indented defs (methods inside classes) need only
    one blank line and are handled by the default single-newline
    separator already in place.
    """
    if not block:
        return False
    first_line = block.lstrip("\n").split("\n", 1)[0]
    return bool(_TOP_LEVEL_PYTHON_DEF_RE.match(first_line))


def _revert_if_syntax_cascade(
    target: Path,
    pre_edit_syntax: str,
    post_edit_syntax: str,
    original_text: str,
    action: FixAtLineAction,
) -> FixAtLineObservation | None:
    """Revert this edit if the file remains in ``Syntax: FAILED`` state.

    When a previous fix_at_line call left the file with broken Python
    syntax AND the current call did not repair it, the agent is in a
    cascade of bad edits where each new write piles more wreckage onto
    a file that doesn't parse. Revert to the pre-edit state and return
    an error observation directing the agent to repair syntax first.

    Returns ``None`` when no revert is needed (non-Python target, file
    was clean before, or the edit fixed the syntax).
    """
    if target.suffix != ".py":
        return None
    if not pre_edit_syntax.startswith("FAILED"):
        return None
    if not post_edit_syntax.startswith("FAILED"):
        return None
    target.write_text(original_text, encoding="utf-8")
    return FixAtLineObservation.from_text(
        text=(
            "File was already in Syntax: FAILED state from a previous edit, "
            "and this edit did not repair the syntax. Reverting to prevent "
            "cascading corruption.\n"
            f"Pre-edit error:  {pre_edit_syntax}\n"
            f"Post-edit error: {post_edit_syntax}\n"
            "Re-view the file with `file_editor view`, identify the "
            "syntactically-broken region, and submit one fix_at_line call "
            "that restores valid Python syntax. Only then continue with "
            "further edits."
        ),
        is_error=True,
        path=action.path,
        start_line=action.start_line,
        end_line=action.end_line,
    )


def _new_block_adds_python_structure(old_block: str, new_block: str) -> bool:
    """True when ``new_block`` introduces def/class/decorator absent from ``old_block``.

    Substring replacement is meant for tiny in-line edits (rename, value
    swap). When an agent passes a short ``old_block`` like ``return "x"``
    and a ``new_block`` containing one or more ``def`` lines, the
    substring match silently inserts those defs in the middle of an
    existing statement — producing structurally-broken Python.

    This check counts top-level ``def`` / ``class`` / ``@decorator``
    occurrences in each block and returns True if ``new_block`` adds at
    least one that ``old_block`` doesn't already have, signalling that
    the agent should use full-block replacement or the EOF-append
    pattern instead.
    """
    new_count = len(_PYTHON_STRUCTURE_RE.findall(new_block))
    old_count = len(_PYTHON_STRUCTURE_RE.findall(old_block))
    return new_count > old_count


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
