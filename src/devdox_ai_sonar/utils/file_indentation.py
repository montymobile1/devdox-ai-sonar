from pathlib import Path
from typing import List, Tuple, Dict
from devdox_ai_sonar.models.file_structures import (
    LineRange,
    FixApplication,
    ImportState,
)
from devdox_ai_sonar.models.sonar import FixSuggestion
from devdox_ai_sonar.logging_config import setup_logging, get_logger

setup_logging(level="DEBUG")
logger = get_logger(__name__)


def read_file_lines(file_path: Path) -> List[str]:
    """Read file and return list of lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_file_lines(file_path: Path, lines: List[str]) -> None:
    """Write lines to file."""
    content = "".join(lines)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def is_simple_replacement(fix: FixSuggestion) -> bool:
    """Check if this is a simple single-line replacement."""
    has_no_newlines = "\n" not in fix.fixed_code
    has_sonar_line = fix.sonar_line_number != 0
    has_no_helper = fix.helper_code == ""

    return has_no_newlines and has_sonar_line and has_no_helper


def _apply_simple_replacement(lines: List[str], fix: FixSuggestion) -> None:
    """Apply a simple single-line replacement."""
    target_line = fix.sonar_line_number - 1
    base_indent = calculate_base_indentation(lines[target_line])
    indent_spaces = " " * base_indent

    indented_code = apply_indentation_to_fix(fix.fixed_code, indent_spaces)
    lines[target_line] = indented_code + "\n"


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

    # Convert to 0-indexed
    line_idx = line_number - 1

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
        return target_line[: len(target_line) - len(stripped)]

    return ""


def replace_lines_simple(
    lines: List[str], line_range: LineRange, new_code: str
) -> List[str]:
    """Replace line range with new code (no helper)."""
    lines[line_range.start : line_range.end + 1] = [new_code, "\n", "\n"]
    return lines


def apply_sibling_helper(
    lines: List[str],
    line_range: LineRange,
    indented_code: str,
    helper_code: str,
    base_indent: str,
) -> List[str]:
    """Apply fix with sibling helper code."""
    indented_helper = apply_indentation_to_fix(helper_code, base_indent)
    replacement = [indented_code, "\n", indented_helper, "\n"]
    lines[line_range.start : line_range.end + 1] = replacement
    return lines


def apply_global_bottom_helper(
    lines: List[str], line_range: LineRange, indented_code: str, helper_code: str
) -> List[str]:
    """Apply fix with global bottom helper code."""
    # Replace the target lines
    lines[line_range.start : line_range.end + 1] = [indented_code, "\n"]
    # Append helper at bottom
    lines.extend(["\n", helper_code, "\n"])
    return lines


def is_import_block(code: str) -> bool:
    """Check if code block contains imports."""
    helper_lines = code.split("\n")
    return any(
        line.strip().startswith(("import ", "from "))
        for line in helper_lines
        if line.strip()
    )


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
        state, stop = process_import_line(i, line, state)

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


def process_import_line(i: int, line: str, state: ImportState) -> tuple:
    stripped = line.strip()
    # Shebang / encoding lines
    validate_shebang_or_encoding, state = is_shebang_or_encoding(i, stripped, state)
    if validate_shebang_or_encoding:
        return state, False
    # Docstring handling
    validate_handle_docstring, state = handle_docstring(i, stripped, state)

    if validate_handle_docstring:
        return state, False
    # Import statements
    if stripped.startswith(("import ", "from ")):
        state["last_import_line"] = i
        return state, False
    # Actual code (non‑comment, non‑empty)
    if stripped and not stripped.startswith("#"):
        return state, True
    return state, False


def handle_docstring(
    i: int, stripped: str, state: ImportState
) -> Tuple[bool, ImportState]:
    if not state.get("in_docstring"):
        if stripped.startswith('"""') or stripped.startswith("'''"):
            state["docstring_quote"] = stripped[:3]
            state["in_docstring"] = True

            if stripped.count(state["docstring_quote"]) >= 2:
                state["in_docstring"] = False
                state["last_docstring_line"] = i
            return True, state
    else:
        if state["docstring_quote"] in stripped:
            state["in_docstring"] = False
            state["last_docstring_line"] = i
            return True, state
        # still inside multi‑line docstring
        return True, state
    return False, state


def is_shebang_or_encoding(
    i: int, stripped: str, state: ImportState
) -> Tuple[bool, Dict[str, str]]:
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
        Lines with normalized indentation
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

    Args:
        fixed_code: The fixed code to indent
        base_indent: Base indentation to apply

    Returns:
        Properly indented fixed code
    """
    if not fixed_code.strip():
        return fixed_code

    lines = fixed_code.split("\n")
    if not lines:
        return fixed_code

    # Remove common leading whitespace (normalize)
    lines = normalize_indentation(lines)

    # Apply base indentation to all non-empty lines
    indented_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            indented_lines.append(base_indent + line)
        else:  # Empty line
            indented_lines.append(line)

    return "\n".join(indented_lines)


def apply_complex_fix(
    lines: List[str], fix: FixSuggestion, line_range: LineRange
) -> None:
    """Apply a complex fix with potential helper code."""
    base_indent = calculate_base_indentation_based_on_line(lines, line_range.start)
    indented_code = apply_indentation_to_fix(fix.fixed_code, base_indent)

    # Normalize helper code
    helper_code = fix.helper_code.replace("\\n", "\n")

    if not helper_code:
        lines = replace_lines_simple(lines, line_range, indented_code)
    elif fix.placement_helper == "SIBLING":
        lines = apply_sibling_helper(
            lines, line_range, indented_code, helper_code, base_indent
        )
    elif fix.placement_helper == "GLOBAL_BOTTOM":
        lines = apply_global_bottom_helper(
            lines, line_range, indented_code, helper_code
        )
    elif fix.placement_helper == "GLOBAL_TOP":
        apply_global_top_helper(lines, line_range, indented_code, helper_code)
    else:
        lines = replace_lines_simple(lines, line_range, indented_code)
    return lines


def find_global_top_insertion_point(lines: List[str]) -> int:
    """
    Find the position for non-import global code (classes, functions, constants).
    This should go after imports but before other code.
    """
    import_end = find_import_insertion_point(lines)

    # Look for the first non-import, non-comment, non-empty line after imports
    for i in range(import_end, len(lines)):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            return i

    # If no code found, append at the end
    return len(lines)


def apply_global_top_helper(
    lines: List[str], line_range: LineRange, indented_code: str, helper_code: str
) -> None:
    """Apply fix with global top helper code."""
    # First replace the target lines
    lines[line_range.start : line_range.end + 1] = [indented_code, "\n"]

    # Determine where to insert helper
    if is_import_block(helper_code):
        insert_pos = find_import_insertion_point(lines)
        logger.debug(f"Inserting import block at line {insert_pos + 1}")
    else:
        insert_pos = find_global_top_insertion_point(lines)
        logger.debug(f"Inserting global code at line {insert_pos + 1}")

    # Insert helper code
    helper_with_newline = (
        helper_code if helper_code.endswith("\n") else helper_code + "\n"
    )
    lines.insert(insert_pos, helper_with_newline + "\n")
    return lines


def apply_single_fix(lines: List[str], fix: FixSuggestion) -> FixApplication:
    """Apply a single fix to the lines array."""
    line_range = LineRange.from_fix(fix)

    if not line_range:
        return FixApplication(fix, False, "Missing line numbers")

    if not line_range.is_valid(len(lines)):
        return FixApplication(fix, False, "Invalid line range")

    # Handle special single-line replacement case
    if is_simple_replacement(fix):
        _apply_simple_replacement(lines, fix)
        return FixApplication(fix, True)

    # Handle complex fix with helper code
    apply_complex_fix(lines, fix, line_range)
    return FixApplication(fix, True)
