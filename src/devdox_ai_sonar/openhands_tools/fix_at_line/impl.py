"""Executor for the ``fix_at_line`` tool.

Self-contained: does not depend on any pre-OpenHands fix-application helpers.
Behaviour is purely "read file, validate range, splice lines, write file",
mirroring the semantics of ``file_editor str_replace`` but anchored by line
number instead of content match.
"""

import ast
import difflib
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

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

    def __init__(
        self,
        allowed_roots: Optional[Sequence[Path]] = None,
    ) -> None:
        # Untrusted-path guard against SonarCloud python:S2083
        # (path-injection). ``action.path`` is set by the LLM agent,
        # which means it is downstream of attacker-influenced text
        # (Sonar issue messages, scanned code/comments). Without a
        # guard, a sufficiently confused or prompt-injected agent
        # could direct the executor to overwrite ``/etc/passwd``,
        # ``~/.ssh/authorized_keys``, or anywhere else the process
        # has write access.
        #
        # Policy: every read/write target must resolve under one of
        # the configured roots. ``None`` means "use the defaults
        # below" — broad enough to cover normal use (cwd is the
        # project root; ``/tmp`` covers devdox_sonar's cloned-repo
        # workspace and pytest's ``tmp_path``), narrow enough to
        # exclude ``/etc``, ``/root``, ``/home/*/.ssh``, and the
        # system Python tree.
        self._configured_roots = allowed_roots

    def _allowed_roots(self) -> List[Path]:
        if self._configured_roots is not None:
            return [Path(r).resolve() for r in self._configured_roots]
        return [Path("/tmp").resolve(), Path.cwd().resolve()]

    def _is_path_allowed(self, target: Path) -> bool:
        try:
            resolved = target.resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        for root in self._allowed_roots():
            if resolved == root or resolved.is_relative_to(root):
                return True
        return False

    def __call__(
        self,
        action: FixAtLineAction,
        conversation: "BaseConversation | None" = None,
    ) -> FixAtLineObservation:
        logger.debug(
            "[fix_at_line-diagnostic] FixAtLineExecutor INVOKED: "
            "path=%s start_line=%s end_line=%s old_block_len=%s new_block_len=%s",
            action.path,
            action.start_line,
            action.end_line,
            len(action.old_block or ""),
            len(action.new_block or ""),
        )
        observation = self._run(action)
        logger.debug(
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

        # SonarCloud S2083 guard. Refuse paths that resolve outside
        # the configured allowed roots — see ``__init__`` for the
        # threat model and policy. Done once here so the read at
        # ``read_text`` and the writes inside ``_eof_append`` /
        # below all run on a sanitised path.
        if not self._is_path_allowed(target):
            return FixAtLineObservation.from_text(
                text=(
                    f"`path` resolves outside the executor's allowed "
                    f"roots and is refused as a path-injection guard: "
                    f"{action.path!r}"
                ),
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
        # that the executor does not count. The on-the-nose case is
        # start==end==total+1 with empty old_block. When the agent's L_view
        # is overcounted (view added a trailing-blank affordance the file
        # itself doesn't have), the request comes in as
        # start==end > total+1 — same intent, off-by-N from a benign
        # display quirk. Treat both shapes as a deliberate append; the
        # block lands at true EOF either way.
        if (
            action.start_line == action.end_line
            and action.start_line >= total_lines + 1
            and action.old_block == ""
        ):
            return self._eof_append(
                action, target, original_text, pre_edit_syntax
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

        result = self._compute_inrange_replacement(
            action,
            target,
            lines,
            start_idx,
            end_idx_exclusive,
            current_block_text,
        )
        if isinstance(result, FixAtLineObservation):
            return result
        new_text, new_block_line_count, returned_old_block, returned_new_block = result

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

    def _eof_append(
        self,
        action: FixAtLineAction,
        target: Path,
        original_text: str,
        pre_edit_syntax: str,
    ) -> FixAtLineObservation:
        """Append ``action.new_block`` after the end of ``target``.

        Extracted from ``_run`` so the orchestrator's cognitive
        complexity stays under Sonar's S3776 threshold. Behaviour
        identical: handles separator selection (PEP 8 two-blanks for
        top-level Python defs, single blank otherwise, empty new_block
        as a no-op trailing-newline preserver), runs the post-edit
        syntax check + cascade revert, and returns the success or
        revert observation.
        """
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

    def _compute_inrange_replacement(
        self,
        action: FixAtLineAction,
        target: Path,
        lines: List[str],
        start_idx: int,
        end_idx_exclusive: int,
        current_block_text: str,
    ) -> Union[FixAtLineObservation, Tuple[str, int, str, str]]:
        """Decide the new file text for an in-range edit.

        Returns either a ``FixAtLineObservation`` (early-return error,
        e.g. would-corrupt-file / ambiguous-substring / no-match) or a
        tuple ``(new_text, new_block_line_count, returned_old_block,
        returned_new_block)`` on success. Extracted from ``_run`` so
        the orchestrator's cognitive complexity stays under Sonar's
        S3776 threshold; behaviour identical.
        """
        old_stripped = _strip_trailing_newline(action.old_block)
        new_stripped = _strip_trailing_newline(action.new_block)
        current_stripped = _strip_trailing_newline(current_block_text)

        # Two paths converge on a full-window replace:
        #
        #  * ``old_stripped == ""`` — the "trust my line range" signal.
        #    The agent has just viewed the file and is replacing the
        #    whole [start_line..end_line] window without asking the
        #    executor to byte-verify the existing content. Necessary
        #    because some LLMs (e.g., Llama 3.3) cannot reliably
        #    reproduce long multi-line strings as tool arguments —
        #    they emit literal "..." stubs or mangle whitespace, which
        #    makes the byte-match path always fail. Skipping the byte
        #    check is safe because (a) the line range itself is the
        #    anchor, (b) the post-edit syntax check + cascade revert
        #    catches structurally-broken results.
        #
        #  * ``old_stripped == current_stripped`` — explicit byte-
        #    verified full-block match.
        if old_stripped == "" or old_stripped == current_stripped:
            new_block_lines = _normalize_block_to_lines(action.new_block)
            new_file_lines = (
                lines[:start_idx]
                + new_block_lines
                + lines[end_idx_exclusive:]
            )
            return (
                "".join(new_file_lines),
                len(new_block_lines),
                current_stripped,
                new_stripped,
            )

        if old_stripped in current_stripped:
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
            # Substring match, unique within the range.
            new_block_text = current_stripped.replace(
                old_stripped, new_stripped, 1
            )
            if current_block_text.endswith("\n"):
                new_block_text += "\n"
            prefix = "".join(lines[:start_idx])
            suffix = "".join(lines[end_idx_exclusive:])
            return (
                prefix + new_block_text + suffix,
                len(_normalize_block_to_lines(new_block_text)),
                current_stripped,
                _strip_trailing_newline(new_block_text),
            )

        # No match at all.
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

    orphan = _check_no_orphan_self_methods(target)
    if orphan:
        return f"FAILED — {orphan}"

    misplaced_decorated = _check_no_misplaced_decorated_helpers(target)
    if misplaced_decorated:
        return f"FAILED — {misplaced_decorated}"

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


def _collect_orphan_self_methods(
    node: ast.AST, in_class: bool, orphans: List[Tuple[str, int]]
) -> None:
    """Walk ``node`` collecting (name, lineno) of orphan self-methods.

    Lifted out of ``_check_no_orphan_self_methods`` so the public
    checker stays under Sonar's S3776 cognitive complexity threshold.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            _collect_orphan_self_methods(
                child, in_class=True, orphans=orphans
            )
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not in_class:
                args = child.args.args
                if args and args[0].arg == "self":
                    orphans.append((child.name, child.lineno))
            # Recurse with in_class=False: nested defs are not class
            # members even if the outer function is.
            _collect_orphan_self_methods(
                child, in_class=False, orphans=orphans
            )
        else:
            _collect_orphan_self_methods(
                child, in_class=in_class, orphans=orphans
            )


def _check_no_orphan_self_methods(target: Path) -> str:
    """Return a non-empty failure description if any function whose
    first parameter is ``self`` is defined outside a class body.

    Catches the failure mode where an agent extracts class methods
    (the simplified caller still does ``self._helper(...)``) but
    appends the helper definitions outside the class — typically via
    the EOF-append pattern, which lands them at module level. When the
    appended block carries class-level indentation, Python parses it
    as a nested function inside whatever top-level function it lands
    after, which is even more invisible: ``py_compile`` accepts the
    file because nested defs are syntactically valid, ``pyflakes``
    doesn't flag method-level references, and the duplicate-method
    check sees no duplicates. At runtime the call site
    ``AttributeError``s on the missing class method.

    The check: walk the AST tracking whether each node's immediate
    parent is a ``ClassDef``. Any ``FunctionDef`` / ``AsyncFunctionDef``
    with ``self`` as its first parameter must have a ``ClassDef``
    parent — anything else is an orphan.

    Returns "" when no orphans are found.
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

    orphans: list = []
    _collect_orphan_self_methods(tree, in_class=False, orphans=orphans)

    if not orphans:
        return ""

    items = ", ".join(
        f"'{name}' (line {lineno})" for name, lineno in orphans
    )
    return (
        f"function(s) {items} have `self` as first parameter but "
        f"are defined outside a class body — likely orphaned helpers "
        f"extracted from a class method. Move them inside the class "
        f"so callers' `self.<name>(...)` references resolve. For "
        f"S3776-style refactors of an instance method, replace the "
        f"original function's line range with [simplified function "
        f"+ blank line + helpers] in ONE fix_at_line call at the "
        f"class-member indent — do NOT use the EOF-append pattern."
    )


def _decorator_name(dec: ast.AST) -> str:
    """Best-effort extraction of a decorator's surface name.

    Handles ``@name``, ``@module.name``, and ``@name(...)`` shapes.
    Returns "" for anything else (e.g. computed decorators), since the
    caller compares against a small allowlist of literal names.
    """
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    return ""


def _collect_misplaced_decorated_helpers(
    node: ast.AST,
    in_function: bool,
    misplaced: List[Tuple[str, int, str]],
) -> None:
    """Walk ``node`` collecting nested defs decorated @staticmethod/@classmethod.

    Lifted out of ``_check_no_misplaced_decorated_helpers`` so the
    public checker stays under Sonar's S3776 cognitive complexity
    threshold.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            # Class body resets the "inside a function" tracking:
            # methods inside the class are class members, not nested
            # defs of the enclosing function.
            _collect_misplaced_decorated_helpers(
                child, in_function=False, misplaced=misplaced
            )
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if in_function:
                for dec in child.decorator_list:
                    name = _decorator_name(dec)
                    if name in ("staticmethod", "classmethod"):
                        misplaced.append((child.name, child.lineno, name))
                        break
            _collect_misplaced_decorated_helpers(
                child, in_function=True, misplaced=misplaced
            )
        else:
            _collect_misplaced_decorated_helpers(
                child, in_function=in_function, misplaced=misplaced
            )


def _check_no_misplaced_decorated_helpers(target: Path) -> str:
    """Return a non-empty failure description if any ``FunctionDef``
    decorated with ``@staticmethod`` or ``@classmethod`` is defined
    inside another function body (rather than a class body).

    Catches the failure mode where an EOF-append for an extracted
    ``@staticmethod`` helper lands with leading whitespace and Python
    parses it as a nested def inside whatever module-level function was
    open at the append point — typically right after that function's
    ``return`` statement, so the def is also unreachable. The file
    parses, ``py_compile`` accepts it, ``pyflakes`` doesn't flag it
    (the decorator is just a no-op on a nested def), and the orphan-
    self check ignores it (no ``self`` parameter). At runtime the call
    site ``Cls._helper(...)`` ``AttributeError``s on the missing class
    member.

    The check: walk the AST tracking whether each node is inside a
    ``FunctionDef`` body. Any ``FunctionDef`` / ``AsyncFunctionDef``
    in that scope whose decorators include ``staticmethod`` /
    ``classmethod`` is misplaced — those decorators only carry meaning
    on class methods.

    Returns "" when no misplaced decorated helpers are found.
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

    misplaced: list = []
    _collect_misplaced_decorated_helpers(
        tree, in_function=False, misplaced=misplaced
    )

    if not misplaced:
        return ""

    items = ", ".join(
        f"'{name}' (line {lineno}, @{dec})"
        for name, lineno, dec in misplaced
    )
    return (
        f"function(s) {items} are decorated with @staticmethod or "
        f"@classmethod but defined inside another function body — that "
        f"decoration is meaningless outside a class body and the helper "
        f"will not be reachable as a class member. Likely an EOF-append "
        f"landed with leading whitespace and got parsed as a nested def "
        f"of the preceding module-level function (often right after its "
        f"`return`, so it's also dead code). Re-emit the helper at "
        f"column 0 (top-level) without the decorator, OR put it inside "
        f"the relevant class via the in-class one-call pattern."
    )


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
    """Revert this edit if it left the file in ``Syntax: FAILED`` state.

    Originally only fired on FAILED→FAILED to avoid disturbing an agent
    midway through a multi-step recovery. In practice a single OK→FAILED
    edit is also a bug we want to undo immediately: the agent stacked an
    edit with stale line numbers and shredded a real line. Leaving the
    file broken hopes the agent's *next* edit repairs it; reverting puts
    the file back to a known-good state and forces a re-view, which is
    almost always what the agent should have done.

    Returns ``None`` when no revert is needed (non-Python target, or the
    edit kept syntax valid).
    """
    if target.suffix != ".py":
        return None
    if not post_edit_syntax.startswith("FAILED"):
        return None
    # Capture the about-to-be-discarded text so we can show what the
    # agent actually wrote vs. what we kept. Both go through the logger
    # (warning for the bare event, debug for the diff) so a future test
    # run that reproduces the OK→FAILED chaos has a forensic trail
    # without requiring DEVDOX_DEBUG to be on at the time.
    try:
        rejected_text = target.read_text(encoding="utf-8")
    except OSError:
        rejected_text = ""
    target.write_text(original_text, encoding="utf-8")
    transition = (
        "FAILED→FAILED"
        if pre_edit_syntax.startswith("FAILED")
        else "OK→FAILED"
    )
    logger.warning(
        "[fix_at_line-diagnostic] cascade-revert triggered: "
        "transition=%s | path=%s | range=%s-%s | post_edit_error=%s",
        transition,
        action.path,
        action.start_line,
        action.end_line,
        post_edit_syntax,
    )
    if logger.isEnabledFor(logging.DEBUG) and rejected_text:
        diff = "\n".join(
            difflib.unified_diff(
                original_text.splitlines(),
                rejected_text.splitlines(),
                fromfile="pre-edit (kept)",
                tofile="post-edit (rejected)",
                lineterm="",
                n=3,
            )
        )
        if diff:
            logger.debug(
                "[fix_at_line-diagnostic] cascade-revert diff:\n%s", diff
            )
    pre_note = (
        f"Pre-edit syntax: {pre_edit_syntax}\n"
        if pre_edit_syntax.startswith("FAILED")
        else ""
    )
    return FixAtLineObservation.from_text(
        text=(
            "Your last edit left the file in Syntax: FAILED state. "
            "Reverting it so the file is back to its pre-edit content. "
            "Line numbers shift after every successful write — re-view "
            "the file with `file_editor view` to get fresh numbers, "
            "then retry.\n"
            f"{pre_note}"
            f"Post-edit error: {post_edit_syntax}"
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
