from pathlib import Path
import shutil
import tempfile
import re
from git import Repo
from typing import List, Optional, Tuple
import logging
from devdox_ai_sonar.models.file_structures import (
    LineRange,
    FixApplication,
    ImportState,
)
from devdox_ai_sonar.models.sonar import (
    FixSuggestion,
    CodeBlock,
    ChangeType,
    ChangeAction,
    LineChange,
    SearchReplace,
)

logger = logging.getLogger(__name__)


def remove_tmp_files(relative_path: str) -> bool:
    """
    Remove a file or directory safely.

    Args:
        relative_path: Path to file or directory to remove

    Returns:
        True if successfully removed

    Raises:
        ValueError: If path is invalid or contains dangerous components
        FileNotFoundError: If path doesn't exist
    """
    try:
        # Convert to Path object and validate
        path = Path(relative_path)

        # Validate path components
        if not path.parts:
            raise ValueError("Empty path provided")

        # Check for problematic path components
        if any(part in ("..", ".", "", " ") for part in path.parts):
            raise ValueError(f"Path contains invalid components: {path}")

        # Resolve path to absolute
        resolved_path = path.resolve()

        # Check if path exists
        if not resolved_path.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved_path}")

        # Remove based on type
        if resolved_path.is_file():
            resolved_path.unlink()  # ✅ Remove file
            print("File removed successfully")
        elif resolved_path.is_dir():
            shutil.rmtree(resolved_path)  # ✅ Remove directory tree
            print("Remove directory tree")
        else:
            raise ValueError(f"Path is neither file nor directory: {resolved_path}")

        return True

    except FileNotFoundError:
        raise
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path '{relative_path}': {e}") from e


def generate_tmp_path() -> str:
    tmp_dir = tempfile.mkdtemp(
        prefix="devdox_",
        suffix="_test",
        dir=None,  # Uses system temp dir (respects TMPDIR env var)
    )
    return tmp_dir


def download_latest_version(
    repo_url: str, repo_path: str, branch: str
) -> Optional[Repo]:
    try:
        repo = Repo.clone_from(repo_url, repo_path, branch=branch)
        return repo
    except Exception as e:
        print(f"Error loading files from {repo_url}: {e}")
        return None


def read_file_lines(file_path: Path) -> List[str]:
    """Read file and return list of lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_file_lines(file_path: Path, lines: List[str]) -> None:
    """Write lines to file."""
    content = "".join(lines)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def calculate_base_indentation(code: str) -> int:
    """
    Calculate the base indentation level from the first non-empty line.

    Args:
        code: Code string

    Returns:
        Number of leading spaces in the first non-empty line
    """
    for line in code.split("\n"):
        if line.strip():
            return len(line) - len(line.lstrip())
    return 0


def calculate_base_indentation_based_on_line(lines: List[str], line_number: int) -> str:
    """
    Calculate the base indentation for a specific line.

    Args:
        lines: All lines in the file
        line_number: Line number (1-indexed)

    Returns:
        Base indentation string (spaces or tabs)
    """
    if line_number < 1 or line_number > len(lines):
        return ""

    line_idx = line_number
    # Get the indentation of the target line
    target_line = lines[line_idx]
    if not target_line.strip():
        # If target line is empty, look at surrounding lines
        for offset in [1, -1, 2, -2]:
            check_idx = line_idx + offset
            if 0 <= check_idx < len(lines) and lines[check_idx].strip():
                target_line = lines[check_idx]
                break

    if target_line.strip():
        stripped = target_line.lstrip()
        indent_length = len(target_line) - len(stripped)
        return target_line[:indent_length]

    return ""


def apply_sibling_helper(
    lines: List[str],
    line_range: LineRange,
    helper_code: str,
) -> List[str]:
    """Apply fix with sibling helper code at the same indentation level."""
    # For SIBLING helpers, we need METHOD definition indent, not body indent
    # Extract the original method to find its definition indent
    original_method_lines = lines[line_range.start : line_range.end + 1]

    original_method_code = "\n".join(original_method_lines)

    # method_def_indent
    _ = get_method_definition_indent(original_method_code)

    # Apply method definition indent to helper (not body indent!)
    indented_helper = helper_code.replace("\\n", "\n")

    try:
        lines[line_range.end] = indented_helper + "\n"
    except IndexError:
        lines[len(lines) - 1] = indented_helper
    return lines


def apply_global_bottom_helper(lines: List[str], helper_code: str) -> List[str]:
    # Append helper at bottom
    lines.extend(["\n", helper_code, "\n"])
    return lines


def find_import_insertion_point(lines: List[str]) -> int:
    """
    Find the best position to insert import statements.
    Returns the line index where imports should be inserted.
    """
    state: ImportState = {
        "last_import_line": -1,
        "last_docstring_line": -1,
        "last_shebang_encoding_line": -1,
        "in_docstring": False,
        "docstring_quote": None,
    }
    for i, line in enumerate(lines):
        state, stop = process_import_line(i, line, lines, state)

        if stop:
            break

    if state["last_import_line"] >= 0:
        return state["last_import_line"] + 1
    elif state["last_docstring_line"] >= 0:
        return state["last_docstring_line"] + 1
    elif state["last_shebang_encoding_line"] >= 0:
        return state["last_shebang_encoding_line"] + 1
    else:
        return 0


def process_import_line(
    i: int, line: str, lines: List[str], state: ImportState
) -> Tuple[ImportState, bool]:
    """Process a single line for import detection."""

    stripped = line.strip()
    indent = len(line) - len(line.lstrip())
    # Ignore indented lines (inside function/class)
    if indent > 0:
        return state, False

    # Shebang / encoding
    validate_shebang_or_encoding, state = is_shebang_or_encoding(i, stripped, state)
    if validate_shebang_or_encoding:
        return state, False

    # Docstring
    validate_handle_docstring, state = handle_docstring(i, stripped, state)
    if validate_handle_docstring:
        return state, False

    # Top-level import
    if stripped.startswith(("import ", "from ")):
        state["last_import_line"] = i
        return state, False

    # Line continuation for import (top-level only)
    if state["last_import_line"] >= 0:
        prev_line = lines[state["last_import_line"]]
        if prev_line.rstrip().endswith("\\"):
            state["last_import_line"] = i
            return state, False

    # Any other code (non-empty, non-comment)
    if stripped and not stripped.startswith("#"):
        return state, True  # stop scanning

    return state, False


def handle_docstring(
    i: int, stripped: str, state: ImportState
) -> Tuple[bool, ImportState]:
    if not state.get("in_docstring"):
        if stripped.startswith('"""') or stripped.startswith("'''"):
            docstring_quote = stripped[:3]
            state["docstring_quote"] = stripped[:3]
            state["in_docstring"] = True

            if stripped.count(docstring_quote) >= 2:
                state["in_docstring"] = False
                state["last_docstring_line"] = i
            return True, state
    else:
        docstring_quote = state.get("docstring_quote") or '"""'

        if docstring_quote in stripped:
            state["in_docstring"] = False
            state["last_docstring_line"] = i
            return True, state
        # still inside multi‑line docstring
        return True, state
    return False, state


def is_shebang_or_encoding(
    i: int, stripped: str, state: ImportState
) -> Tuple[bool, ImportState]:
    if (
        i < 3
        and stripped.startswith("#")
        and ("coding" in stripped or "encoding" in stripped)
    ):
        state["last_shebang_encoding_line"] = i
        return True, state
    if i == 0 and stripped.startswith("#!"):
        state["last_shebang_encoding_line"] = i
        return True, state
    return False, state


def normalize_indentation(lines: List[str]) -> List[str]:
    """
    Remove common leading whitespace from all lines.

    Args:
        lines: Lines of code

    Returns:
        Lines with all leading whitespace stripped from each line
    """
    if not lines:
        return lines

    # Find minimum indentation of non-empty lines
    min_indent = 10**9
    for line in lines:
        if line.strip():  # Non-empty line
            stripped = line.lstrip()
            indent_length = len(line) - len(stripped)
            min_indent = min(min_indent, indent_length)

    if min_indent == float("inf") or min_indent == 0:
        return lines

    # Remove common leading whitespace
    normalized_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            normalized_lines.append(line[min_indent:])
        else:  # Empty line
            normalized_lines.append(line)

    return normalized_lines


def apply_indentation_to_fix(fixed_code: str, base_indent: str) -> str:
    """
    Apply base indentation to fixed code while preserving relative indentation.

    This function assumes the fixed_code has CORRECT RELATIVE indentation,
    and we just need to shift the entire block to start at base_indent.
    """

    no_whitespace = "".join(fixed_code.split())
    if not no_whitespace:
        return fixed_code

    lines = fixed_code.split("\n")

    if not lines:
        return fixed_code

    # Find the minimum indentation in the fixed code (excluding empty lines)
    non_empty_lines = [line for line in lines if "".join(line.split())]
    if not non_empty_lines:
        return fixed_code

    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    # Remove the minimum indentation from all lines, then add base_indent
    indented_lines = []
    for line in lines:
        no_whitespace_line = "".join(line)
        if no_whitespace_line:  # Non-empty line
            # Remove min_indent, then add base_indent
            dedented = line[min_indent:] if len(line) > min_indent else line.lstrip()
            base_indent_line = len(line) - len(line.lstrip())
            if base_indent_line < len(base_indent):
                indented_lines.append(base_indent + dedented)
            else:
                indented_lines.append(dedented)
        else:  # Empty line
            indented_lines.append(line)

    return "\n".join(indented_lines)


def apply_complex_fix(
    lines: List[str], fix: FixSuggestion, line_range: LineRange
) -> List[str]:
    """Apply a complex fix with potential helper code."""
    modified_blocks = fix.fixed_code_blocks

    for block in modified_blocks:
        lines, end_line = apply_single_code_block(lines, block)
        if end_line == 0:
            if block.end_line > block.start_line and block.end_line > line_range.end:
                end_line = block.end_line
        line_range = LineRange(block.start_line, end_line)

    # Step 2: Apply helper code if present
    if fix.helper_code:
        lines = apply_helper_code(lines, line_range, fix)

    if fix.import_block_code:
        lines = apply_import_block(
            lines, fix.import_block_code, fix.end_import_block_code or 0
        )

    #

    return lines


def apply_helper_code(
    lines: List[str], line_range: LineRange, fix: FixSuggestion
) -> List[str]:
    """
    Apply helper code based on placement strategy.

    Args:
        lines: File lines
        fix: FixSuggestion with new_helper_code and placement

    Returns:
        Modified lines
    """
    helper_code = (fix.helper_code or "").replace("\\n", "\n")
    placement = fix.placement_helper
    if placement == "GLOBAL_TOP":
        return apply_global_top_helper(
            lines=lines,
            helper_code=helper_code,
            end_import=fix.end_import_block_code or 0,
        )

    elif placement == "GLOBAL_BOTTOM":
        return apply_global_bottom_helper(lines, helper_code)

    elif placement == "SIBLING":
        return apply_sibling_helper(
            lines=lines, line_range=line_range, helper_code=helper_code
        )

    else:
        # Unknown placement - default to GLOBAL_TOP
        return apply_global_top_helper(
            lines=lines, helper_code=helper_code, end_import=0
        )


def normalize_code(code: str) -> str:
    """
    Normalize code by:
    1. Converting escaped newlines to actual newlines
    2. Converting escaped tabs to spaces
    3. Fixing broken docstrings (single quotes → triple quotes)
    """
    if not code:
        return code

    # Step 1: Handle escaped characters
    code = code.replace("\\n", "\n").replace("\\t", "    ")

    # Step 2: Fix broken docstrings
    lines = code.split("\n")
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this is a standalone quote (potential broken docstring)
        if stripped in ('"', "'"):
            quote_char = stripped
            indent = line[: len(line) - len(line.lstrip())]

            # Look for closing quote
            closing_found = False
            docstring_content: List[str] = []

            for j in range(i + 1, len(lines)):
                if lines[j].strip() == quote_char:
                    # Found closing quote
                    closing_found = True

                    # Add triple-quoted docstring
                    fixed_lines.append(f"{indent}{quote_char * 3}")
                    fixed_lines.extend(docstring_content)
                    fixed_lines.append(f"{indent}{quote_char * 3}")

                    i = j + 1
                    break
                else:
                    docstring_content.append(lines[j])

            if not closing_found:
                # No closing quote found, keep original line
                fixed_lines.append(line)
                i += 1
        else:
            fixed_lines.append(line)
            i += 1

    return "\n".join(fixed_lines)


def _apply_single_pattern(text: str, pattern: SearchReplace) -> str:
    """Apply a single SearchReplace pattern to a text string."""
    if pattern.is_regex:
        count = pattern.count if pattern.count else 0
        return re.sub(pattern.search, pattern.replace, text, count=count)

    if pattern.count is not None:
        return text.replace(pattern.search, pattern.replace, pattern.count)

    return text.replace(pattern.search, pattern.replace)


def _apply_all_patterns(text: str, patterns: List[SearchReplace]) -> str:
    """Apply all SearchReplace patterns to a text string sequentially."""
    for pattern in patterns:
        text = _apply_single_pattern(text, pattern)
    return text


def _apply_line_by_line_replacements(
    lines: List[str],
    replacements: List[SearchReplace],
    start_idx: int,
    end_idx: int,
) -> bool:
    """
    Try to apply replacements line by line.

    Returns True if at least one line was modified.
    """
    applied = False
    for i in range(start_idx, min(end_idx, len(lines))):
        original_line = lines[i]
        modified_line = _apply_all_patterns(original_line, replacements)

        if modified_line != original_line:
            lines[i] = modified_line
            applied = True

    return applied


def _apply_multiline_replacements(
    lines: List[str],
    replacements: List[SearchReplace],
    start_idx: int,
    end_idx: int,
) -> None:
    """
    Apply replacements across a joined block of lines (multiline strategy).

    Mutates `lines` in place if the block text changed.
    """
    block_text = "".join(lines[start_idx:end_idx])
    modified_text = _apply_all_patterns(block_text, replacements)

    if modified_text == block_text:
        return

    new_lines = modified_text.splitlines(keepends=True)

    # Preserve trailing newline from the original last line
    if (
        new_lines
        and not new_lines[-1].endswith("\n")
        and lines[end_idx - 1].endswith("\n")
    ):
        new_lines[-1] += "\n"

    lines[start_idx:end_idx] = new_lines


def apply_search_replace_change(lines: List[str], block: CodeBlock) -> List[str]:
    """
    Apply SEARCH_REPLACE change - find and replace patterns.

    Args:
        lines: File lines (with newlines)
        block: CodeBlock with replacements list

    Returns:
        Modified lines
    """
    if not block.replacements:
        return lines

    start_idx = block.start_line - 1
    end_idx = block.end_line

    # Strategy 1: Try line-by-line replacement first (for single-line patterns)
    if not _apply_line_by_line_replacements(
        lines, block.replacements, start_idx, end_idx
    ):
        # Strategy 2: Fall back to multiline replacement
        _apply_multiline_replacements(lines, block.replacements, start_idx, end_idx)

    return lines


def apply_full_code_change(lines: List[str], block: CodeBlock) -> Tuple[List[str], int]:
    """
    Apply FULL_CODE change - replace entire block with new code.

    Args:
        lines: File lines (with newlines)
        block: CodeBlock with context containing full replacement code

    Returns:
        Modified lines
    """
    if not block.context:
        return lines, 0

    # Convert to 0-indexed

    start_idx = max(0, block.start_line - 1)
    end_idx = max(0, block.end_line - 1)

    # Validate indices
    if start_idx < 0 or start_idx >= len(lines):
        print(f"Warning: Invalid start_line {block.start_line}")
        return lines, block.end_line

    # Calculate base indentation from original code
    # Kept for future auditing purposes, return value intentionally unused
    _ = calculate_base_indentation_based_on_line(lines, start_idx)

    # Normalize and indent the fixed code
    fixed_code = normalize_code(block.context)

    new_lines = fixed_code.split("\n")

    new_lines = [line + "\n" for line in new_lines]

    # Replace the lines
    if start_idx > end_idx:
        lines[start_idx:end_idx] = new_lines
    else:
        lines[start_idx : end_idx + 1] = new_lines

    print(
        f"[FULL_CODE] Replaced lines {block.start_line}-{block.end_line} "
        f"({end_idx - start_idx} lines) with {len(new_lines)} lines"
    )

    if len(new_lines) > 1:
        end_idx = len(new_lines) + block.start_line - 1

        print(
            f"Warning: Full code block has {len(new_lines)} lines, expected {end_idx - start_idx + 1}"
        )

    return lines, end_idx


def _has_own_indentation(text: str) -> bool:
    """Check if text has its own leading indentation (spaces or tabs)."""
    return text.startswith(" ") or text.startswith("\t")


def _replace_line_preserving_indent(
    lines: List[str], line_idx: int, change: LineChange
) -> None:
    """Replace a line, preserving original indentation when new content has none."""
    if change.new is None:
        return
    if _has_own_indentation(change.new):
        lines[line_idx] = change.new.rstrip() + "\n"
    else:
        original_indent = calculate_base_indentation(lines[line_idx])
        lines[line_idx] = " " * original_indent + change.new.strip() + "\n"
    print(f"[DIFF] Replaced line {change.line}: {change.old} -> {change.new}")


def _try_replace_at_corrected_line(
    lines: List[str],
    change: LineChange,
    start_line: int,
    end_line: int,
) -> None:
    """Find the correct line by content and apply the replacement there."""
    if change.old is None:
        return
    corrected_line = find_line_by_content(
        lines[start_line:end_line], change.old, start_line
    )
    if not corrected_line:
        return

    logger.info(f"Corrected line {change.line} -> {corrected_line}")
    change.line = corrected_line
    line_idx = change.line - 1

    if change.old.strip() == lines[line_idx].strip():
        _replace_line_preserving_indent(lines, line_idx, change)


def _apply_replace_action(
    lines: List[str],
    change: LineChange,
    start_line: int,
    end_line: int,
) -> None:
    """Apply a single REPLACE change to lines."""
    if change.new is None or change.old is None:
        return

    line_idx = change.line - 1

    if change.old.strip() == lines[line_idx].strip():
        _replace_line_preserving_indent(lines, line_idx, change)
    else:
        _try_replace_at_corrected_line(lines, change, start_line, end_line)


def _apply_insert_action(lines: List[str], change: LineChange) -> None:
    """Apply a single INSERT change to lines."""
    if change.new is None:
        return
    line_idx = change.line - 1
    indent = calculate_base_indentation_based_on_line(lines, line_idx)
    new_line = indent + change.new.strip() + "\n"
    lines.insert(line_idx, new_line)


def _apply_delete_action(lines: List[str], change: LineChange) -> None:
    """Apply a single DELETE change to lines."""
    line_idx = change.line - 1
    if 0 <= line_idx < len(lines):
        lines.pop(line_idx)


def apply_diff_change(lines: List[str], block: CodeBlock) -> List[str]:
    """
    Apply DIFF change - modify specific lines.

    Args:
        lines: File lines (with newlines)
        block: CodeBlock with changes list

    Returns:
        Modified lines
    """
    if not block.changes:
        return lines

    # Sort changes by line number (descending) to avoid index shifting
    sorted_changes = sorted(block.changes, key=lambda c: c.line, reverse=True)

    for change in sorted_changes:
        line_idx = change.line - 1

        if line_idx < 0 or line_idx >= len(lines):
            print(f"Warning: Invalid line number {change.line}")
            continue

        if change.action == ChangeAction.REPLACE:
            _apply_replace_action(lines, change, block.start_line, block.end_line)
        elif change.action == ChangeAction.INSERT:
            _apply_insert_action(lines, change)
        elif change.action == ChangeAction.DELETE:
            _apply_delete_action(lines, change)

    return lines


def find_line_by_content(lines: list[str], content: str, start_line: int) -> int | None:
    """Find the actual line number by matching content."""
    content_stripped = content.strip()

    for i, line in enumerate(lines):
        if content_stripped in line or line.strip() == content_stripped:
            return start_line + i + 1

    return None


def apply_single_code_block(
    lines: List[str], block: CodeBlock
) -> Tuple[List[str], int]:
    if block.change_type == ChangeType.FULL_CODE:
        return apply_full_code_change(lines, block)

    elif block.change_type == ChangeType.DIFF:
        return apply_diff_change(lines, block), 0
    elif block.change_type == ChangeType.SEARCH_REPLACE:
        return apply_search_replace_change(lines, block), 0
    else:
        print(f"Warning: Unknown change_type '{block.change_type}'")
        return lines, 0


def find_code_start(lines: List[str]) -> int:
    """
    Find where actual code starts (after shebang, encoding, docstrings).
    """
    idx = 0

    # Skip shebang
    if lines and lines[0].startswith("#!"):
        idx = 1

    # Skip encoding declaration
    if idx < len(lines) and "coding" in lines[idx]:
        idx += 1

    # Skip module docstring
    if idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = '"""' if stripped.startswith('"""') else "'''"

            # Check if docstring starts and ends on same line
            if stripped.endswith(quote) and len(stripped) > len(quote):
                return idx + 1

            # Multi-line docstring: skip opening line and find closing quote
            idx += 1
            while idx < len(lines):
                if quote in lines[idx]:
                    idx += 1
                    break
                idx += 1

    return idx


def apply_import_block(
    lines: List[str], import_block: str, end_block_import: int
) -> List[str]:
    """
    Apply/update import block at the top of the file.

    Strategy:
    1. Find the end of existing imports
    2. Replace/insert new import block
    """
    # Normalize import block
    import_lines = import_block.replace("\\n", "\n").split("\n")
    import_lines = [
        line + "\n" if not line.endswith("\n") else line
        for line in import_lines
        if line.strip()
    ]

    # Find end of existing imports
    import_end_idx = end_block_import
    if import_end_idx == 0:
        # No existing imports - insert at top (after shebang/encoding if present)
        insert_idx = find_code_start(lines)
        lines = lines[:insert_idx] + import_lines + ["\n"] + lines[insert_idx:]
    else:
        # Check if there's a blank line at end_block_import
        if end_block_import < len(lines) and not lines[end_block_import].strip():
            # Insert before blank line
            lines = lines[:end_block_import] + import_lines + lines[end_block_import:]
        else:
            # No blank line, just insert at end_block_import

            lines = (
                lines[:end_block_import]
                + import_lines
                + ["\n"]
                + lines[end_block_import:]
            )

    return lines


# def find_global_top_insertion_point(lines: List[str]) -> int:
#     """
#     Find the position for non-import global code (classes, functions, constants).
#     This should go after imports but before other code.
#     """
#     import_end = find_import_insertion_point(lines)
#
#     # Look for the first non-import, non-comment, non-empty line after imports
#     for i in range(import_end, len(lines)):
#         stripped = lines[i].strip()
#         if stripped and not stripped.startswith("#"):
#             return i
#
#     # If no code found, append at the end
#     return len(lines)
#


def apply_global_top_helper(
    lines: List[str],
    helper_code: str,
    end_import: int,
) -> List[str]:
    """Apply fix with global top helper code."""

    # Insert helper code
    helper_with_newline = (
        helper_code if helper_code.endswith("\n") else helper_code + "\n"
    )

    lines.insert(end_import, helper_with_newline + "\n")
    return lines


def apply_single_fix(
    lines: List[str], fix: FixSuggestion
) -> Tuple[FixApplication, List[str]]:
    """Apply a single fix to the lines array."""

    line_range = LineRange.from_fix(fix)
    if not line_range:
        return FixApplication(fix, False, "Missing line numbers"), lines
    if not line_range.is_valid(len(lines)):
        return FixApplication(fix, False, "Invalid line range"), lines

    # Handle complex fix with helper code
    lines = apply_complex_fix(lines, fix, line_range)
    return FixApplication(fix, True), lines


def get_method_definition_indent(code_chunk: str) -> str:
    """
    Get the indentation level where the method DEFINITION starts.

    For example:
        async def _fetch_user_profiles(self):  # <- This line's indent (0 spaces for class methods)
            pass  # <- Body indent (4 spaces)
    """
    lines = code_chunk.split("\n")

    for line in lines:
        # Match 'def' or 'async def'
        match = re.match(r"^(\s*)(?:async\s+)?def\s+", line)
        if match:
            return match.group(1)  # Return leading whitespace before 'def'

    return ""  # Fallback: no indentation
