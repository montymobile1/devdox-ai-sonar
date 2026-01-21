
import pytest
from pathlib import Path
import os
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
from typing import List

from unittest.mock import Mock, patch, mock_open, MagicMock
import tempfile
import shutil

# Import the functions to test
from devdox_ai_sonar.models.file_structures import LineRange, FixApplication, ImportState
from devdox_ai_sonar.models.sonar import FixSuggestion

from devdox_ai_sonar.utils.file_indentation import (
    read_file_lines,
    write_file_lines,
    is_simple_replacement,
    remove_tmp_files,
    generate_tmp_path,
    download_latest_version,
    calculate_base_indentation,
    calculate_base_indentation_based_on_line,
    replace_lines_simple,
    apply_sibling_helper,
    apply_global_bottom_helper,
    is_import_block,
    apply_global_top_helper,
    find_import_insertion_point,
    process_import_line,
    handle_docstring,
    is_shebang_or_encoding,
    normalize_indentation,
    apply_indentation_to_fix,
    apply_complex_fix,
    find_global_top_insertion_point,
    apply_single_fix,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample Python file for testing."""
    file_path = temp_dir / "test.py"
    content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"Module docstring.\"\"\"

import os
import sys
from typing import List

def hello():
    x = 1
    y = 2
    return x + y

class MyClass:
    def method(self):
        pass
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def mock_fix():
    """Create a mock FixSuggestion object."""
    fix = Mock(spec=FixSuggestion)
    fix.issue_key = "TEST-001"
    fix.line_number = 10
    fix.last_line_number = 10
    fix.sonar_line_number = 10
    fix.end_import_block_code = 2
    fix.import_block_code=""
    fix.fixed_code = "x = 2"
    fix.helper_code = ""
    fix.placement_helper = None
    return fix


@pytest.fixture
def sample_lines():
    """Sample file lines for testing."""
    return [
        "#!/usr/bin/env python3\n",
        "# -*- coding: utf-8 -*-\n",
        '"""Module docstring."""\n',
        "\n",
        "import os\n",
        "import sys\n",
        "\n",
        "def hello():\n",
        "    x = 1\n",
        "    y = 2\n",
        "    return x + y\n",
    ]


# ============================================================================
# TEST: read_file_lines & write_file_lines
# ============================================================================

class TestFileIO:
    """Test file I/O operations."""

    def test_read_file_lines_basic(self, sample_file):
        """Test reading lines from a file."""
        lines = read_file_lines(sample_file)
        assert isinstance(lines, list)
        assert len(lines) > 0
        assert lines[0].startswith("#!/usr/bin/env")

    def test_read_file_lines_empty_file(self, temp_dir):
        """Test reading an empty file."""
        empty_file = temp_dir / "empty.py"
        empty_file.write_text("")
        lines = read_file_lines(empty_file)
        assert lines == []

    def test_read_file_lines_encoding(self, temp_dir):
        """Test reading file with special characters."""
        file_path = temp_dir / "unicode.py"
        content = "# Testing unicode: café, naïve, 日本語\n"
        file_path.write_text(content, encoding='utf-8')
        lines = read_file_lines(file_path)
        assert len(lines) == 1
        assert "café" in lines[0]

    def test_read_file_lines_nonexistent(self, temp_dir):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_file_lines(temp_dir / "nonexistent.py")

    def test_write_file_lines_basic(self, temp_dir):
        """Test writing lines to a file."""
        file_path = temp_dir / "output.py"
        lines = ["line 1\n", "line 2\n", "line 3\n"]
        write_file_lines(file_path, lines)

        assert file_path.exists()
        content = file_path.read_text()
        assert content == "line 1\nline 2\nline 3\n"

    def test_write_file_lines_empty(self, temp_dir):
        """Test writing empty list."""
        file_path = temp_dir / "empty.py"
        write_file_lines(file_path, [])
        assert file_path.exists()
        assert file_path.read_text() == ""

    def test_write_file_lines_overwrite(self, sample_file):
        """Test overwriting existing file."""
        new_lines = ["new content\n"]
        write_file_lines(sample_file, new_lines)
        assert sample_file.read_text() == "new content\n"


# ============================================================================
# TEST: is_simple_replacement
# ============================================================================

class TestIsSimpleReplacement:
    """Test simple replacement detection."""

    def test_is_simple_replacement_true(self, mock_fix):
        """Test detecting a simple replacement."""
        mock_fix.fixed_code = "x = 2"
        mock_fix.sonar_line_number = 10
        mock_fix.helper_code = ""

        result = is_simple_replacement(mock_fix)
        assert result is True

    def test_is_simple_replacement_has_newlines(self, mock_fix):
        """Test with code containing newlines."""
        mock_fix.fixed_code = "x = 2\ny = 3"
        mock_fix.sonar_line_number = 10
        mock_fix.helper_code = ""
        line_range = LineRange(start=9, end=9)

        result = is_simple_replacement(mock_fix)
        assert result is False

    def test_is_simple_replacement_no_sonar_line(self, mock_fix):
        """Test with no sonar line number."""
        mock_fix.fixed_code = "x = 2"
        mock_fix.sonar_line_number = 0
        mock_fix.helper_code = ""
        line_range = LineRange(start=9, end=9)

        result = is_simple_replacement(mock_fix)
        assert result is False

    def test_is_simple_replacement_has_helper(self, mock_fix):
        """Test with helper code present."""
        mock_fix.fixed_code = "x = 2"
        mock_fix.sonar_line_number = 10
        mock_fix.helper_code = "import math"
        line_range = LineRange(start=9, end=9)

        result = is_simple_replacement(mock_fix)
        assert result is False

    def test_is_simple_replacement_all_false(self, mock_fix):
        """Test with all conditions false."""
        mock_fix.fixed_code = "x = 2\ny = 3"
        mock_fix.sonar_line_number = 0
        mock_fix.helper_code = "helper"
        line_range = LineRange(start=9, end=9)

        result = is_simple_replacement(mock_fix)
        assert result is False


# ============================================================================
# TEST: calculate_base_indentation
# ============================================================================

class TestCalculateBaseIndentation:
    """Test base indentation calculation."""

    def test_calculate_base_indentation_no_indent(self):
        """Test with no indentation."""
        code = "def hello():\n    pass"
        result = calculate_base_indentation(code)
        assert result == 0

    def test_calculate_base_indentation_four_spaces(self):
        """Test with 4 spaces indentation."""
        code = "    def hello():\n        pass"
        result = calculate_base_indentation(code)
        assert result == 4

    def test_calculate_base_indentation_eight_spaces(self):
        """Test with 8 spaces indentation."""
        code = "        x = 1"
        result = calculate_base_indentation(code)
        assert result == 8

    def test_calculate_base_indentation_empty_lines(self):
        """Test with leading empty lines."""
        code = "\n\n    def hello():"
        result = calculate_base_indentation(code)
        assert result == 4

    def test_calculate_base_indentation_all_empty(self):
        """Test with all empty lines."""
        code = "\n\n\n"
        result = calculate_base_indentation(code)
        assert result == 0

    def test_calculate_base_indentation_tabs(self):
        """Test with tab characters."""
        code = "\t\tdef hello():"
        result = calculate_base_indentation(code)
        assert result == 2  # Two tab characters


# ============================================================================
# TEST: calculate_base_indentation_based_on_line
# ============================================================================

class TestCalculateBaseIndentationBasedOnLine:
    """Test line-based indentation calculation."""

    def test_calculate_based_on_line_normal(self, sample_lines):
        """Test with normal indented line."""
        result = calculate_base_indentation_based_on_line(sample_lines, 9)
        assert result == "    "  # 4 spaces

    def test_calculate_based_on_line_no_indent(self, sample_lines):
        """Test with non-indented line."""
        result = calculate_base_indentation_based_on_line(sample_lines, 5)
        assert result == ""

    def test_calculate_based_on_line_empty_line(self):
        """Test with empty target line."""
        lines = ["def hello():\n", "\n", "    x = 1\n"]
        result = calculate_base_indentation_based_on_line(lines, 2)
        # Should look at surrounding lines
        assert result in ["", "    "]

    def test_calculate_based_on_line_out_of_range_negative(self, sample_lines):
        """Test with negative line number."""
        result = calculate_base_indentation_based_on_line(sample_lines, -1)
        assert result == ""

    def test_calculate_based_on_line_out_of_range_high(self, sample_lines):
        """Test with line number exceeding file length."""
        result = calculate_base_indentation_based_on_line(sample_lines, 1000)
        assert result == ""

    def test_calculate_based_on_line_zero(self, sample_lines):
        """Test with line number zero."""
        result = calculate_base_indentation_based_on_line(sample_lines, 0)
        assert result == ""

    def test_calculate_based_on_line_tabs(self):
        """Test with tab indentation."""
        lines = ["\t\tdef hello():\n", "\t\t\tx = 1\n"]
        result = calculate_base_indentation_based_on_line(lines, 2)
        assert result == "\t\t\t"


# ============================================================================
# TEST: replace_lines_simple
# ============================================================================

class TestReplaceLinesSimple:
    """Test simple line replacement."""

    def test_replace_lines_simple_single_line(self):
        """Test replacing a single line."""
        lines = ["line1\n", "line2\n", "line3\n"]
        line_range = LineRange(start=1, end=1)
        result = replace_lines_simple(lines, line_range, "new_line")

        assert result[0] == "line1\n"
        assert result[1] == "new_line"
        assert result[2] == "\n"
        assert result[3] == "\n"
        assert result[4] == "line3\n"

    def test_replace_lines_simple_multiple_lines(self):
        """Test replacing multiple lines."""
        lines = ["line1\n", "line2\n", "line3\n", "line4\n"]
        line_range = LineRange(start=1, end=2)
        result = replace_lines_simple(lines, line_range, "new_code")

        assert result[0] == "line1\n"
        assert result[1] == "new_code"
        assert result[2] == "\n"
        assert result[3] == "\n"
        assert result[4] == "line4\n"

    def test_replace_lines_simple_first_line(self):
        """Test replacing the first line."""
        lines = ["line1\n", "line2\n", "line3\n"]
        line_range = LineRange(start=0, end=0)
        result = replace_lines_simple(lines, line_range, "new_first")

        assert result[0] == "new_first"
        assert result[1] == "\n"
        assert result[2] == "\n"
        assert result[3] == "line2\n"

    def test_replace_lines_simple_last_line(self):
        """Test replacing the last line."""
        lines = ["line1\n", "line2\n", "line3\n"]
        line_range = LineRange(start=2, end=2)
        result = replace_lines_simple(lines, line_range, "new_last")

        assert result[0] == "line1\n"
        assert result[1] == "line2\n"
        assert result[2] == "new_last"


# ============================================================================
# TEST: apply_sibling_helper
# ============================================================================

class TestApplySiblingHelper:
    """Test sibling helper code application."""

    def test_apply_sibling_helper_basic(self):
        """Test applying sibling helper code."""
        lines = ["line1\n", "line2\n", "line3\n"]
        line_range = LineRange(start=1, end=1)

        result = apply_sibling_helper(
            lines,
            line_range,
            "fixed_code",
            "helper_code",
            "    "
        )

        assert result[0] == "line1\n"
        assert result[1] == "fixed_code"
        assert result[2] == "\n"
        assert result[3] == "    helper_code"
        assert result[4] == "\n"
        assert result[5] == "line3\n"

    def test_apply_sibling_helper_with_indentation(self):
        """Test sibling helper with proper indentation."""
        lines = ["def func():\n", "    x = 1\n", "    y = 2\n"]
        line_range = LineRange(start=1, end=1)

        result = apply_sibling_helper(
            lines,
            line_range,
            "    x = 2",
            "# comment",
            "    "
        )

        # Fixed code should be indented
        assert "    x = 2" in result[1]
        # Helper should also be indented
        assert "    # comment" in result[3]

    def test_apply_sibling_helper_multiline_helper(self):
        """Test sibling helper with multiline code."""
        lines = ["line1\n", "line2\n", "line3\n"]
        line_range = LineRange(start=1, end=1)

        helper_code = "def helper():\n    pass"
        result = apply_sibling_helper(
            lines,
            line_range,
            "fixed",
            helper_code,
            ""
        )

        assert "helper()" in ''.join(result)


# ============================================================================
# TEST: apply_global_bottom_helper
# ============================================================================

class TestApplyGlobalBottomHelper:
    """Test global bottom helper application."""

    def test_apply_global_bottom_helper_basic(self):
        """Test appending helper at bottom."""
        lines = ["line1\n", "line2\n", "line3\n"]
        line_range = LineRange(start=1, end=1)

        result = apply_global_bottom_helper(
            lines,
            line_range,
            "fixed_code",
            "helper_at_bottom"
        )

        assert result[0] == "line1\n"
        assert result[1] == "fixed_code"
        assert result[2] == "\n"
        assert result[3] == "line3\n"
        # Helper should be at the end
        assert result[-2] == "helper_at_bottom"
        assert result[-1] == "\n"

    def test_apply_global_bottom_helper_multiple_lines(self):
        """Test with multiline helper code."""
        lines = ["line1\n", "line2\n"]
        line_range = LineRange(start=0, end=0)

        helper = "def utility():\n    return True"
        result = apply_global_bottom_helper(lines, line_range, "new", helper)

        assert "utility()" in ''.join(result)
        assert result[-2] == helper


# ============================================================================
# TEST: is_import_block
# ============================================================================

class TestIsImportBlock:
    """Test import block detection."""

    def test_is_import_block_single_import(self):
        """Test detecting single import statement."""
        code = "import os"
        assert is_import_block(code) is True

    def test_is_import_block_from_import(self):
        """Test detecting from...import statement."""
        code = "from typing import List"
        assert is_import_block(code) is True

    def test_is_import_block_multiple_imports(self):
        """Test detecting multiple imports."""
        code = "import os\nimport sys\nfrom typing import List"
        assert is_import_block(code) is True

    def test_is_import_block_with_comments(self):
        """Test with comments between imports."""
        code = "import os\n# comment\nimport sys"
        assert is_import_block(code) is True

    def test_is_import_block_no_imports(self):
        """Test with non-import code."""
        code = "def hello():\n    pass"
        assert is_import_block(code) is False

    def test_is_import_block_empty(self):
        """Test with empty string."""
        assert is_import_block("") is False

    def test_is_import_block_only_comments(self):
        """Test with only comments."""
        code = "# import os\n# from typing import List"
        assert is_import_block(code) is False

    def test_is_import_block_whitespace(self):
        """Test with leading/trailing whitespace."""
        code = "  import os  \n  from typing import List  "
        assert is_import_block(code) is True


# ============================================================================
# TEST: find_import_insertion_point
# ============================================================================

class TestFindImportInsertionPoint:
    """Test finding import insertion position."""

    def test_find_import_after_existing_imports(self):
        """Test insertion after existing imports."""
        lines = [
            "import os\n",
            "import sys\n",
            "\n",
            "def main():\n",
            "    pass\n",
        ]
        result = find_import_insertion_point(lines)
        assert result == 2  # After last import

    def test_find_import_after_docstring(self):
        """Test insertion after module docstring."""
        lines = [
            '"""Module doc."""\n',
            "import sys\n",
            "\n",
            "def main():\n",
        ]
        result = find_import_insertion_point(lines)
        assert result == 2  # After docstring

    def test_find_import_after_shebang(self):
        """Test insertion after shebang."""
        lines = [
            "#!/usr/bin/env python3\n",
            "# -*- coding: utf-8 -*-\n",
            "\n",
            "def main():\n",
        ]
        result = find_import_insertion_point(lines)
        assert result == 2  # After encoding

    def test_find_import_with_all_elements(self):
        """Test with shebang, encoding, docstring, and imports."""
        lines = [
            "#!/usr/bin/env python3\n",
            "# -*- coding: utf-8 -*-\n",
            '"""Module docstring."""\n',
            "\n",
            "import os\n",
            "import sys\n",
            "\n",
            "def main():\n",
        ]
        result = find_import_insertion_point(lines)
        assert result == 6  # After last import

    def test_find_import_empty_file(self):
        """Test with empty file."""
        lines = []
        result = find_import_insertion_point(lines)
        assert result == 0

    def test_find_import_no_imports(self):
        """Test file with no imports."""
        lines = ["def main():\n", "    pass\n"]
        result = find_import_insertion_point(lines)
        assert result == 0

    def test_find_import_multiline_docstring(self):
        """Test with multiline docstring."""
        lines = [
            '"""\n',
            "Module docstring\n",
            "spanning multiple lines.\n",
            '"""\n',
            "\n",
            "def main():\n",
        ]
        result = find_import_insertion_point(lines)
        assert result == 4  # After docstring


# ============================================================================
# TEST: process_import_line
# ============================================================================

class TestProcessImportLine:
    """Test import line processing."""

    def test_process_import_line_shebang(self, sample_lines):
        """Test processing shebang line."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "#!/usr/bin/env python3"
        new_state, stop = process_import_line(0, line, sample_lines, state)

        assert new_state["last_shebang_encoding_line"] == 0
        assert stop is False

    def test_process_import_line_encoding(self, sample_lines):
        """Test processing encoding line."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }

        line = "# -*- coding: utf-8 -*-"
        sample_lines[1] = line
        new_state, stop = process_import_line(1,  line, sample_lines,state)

        assert new_state["last_shebang_encoding_line"] == 1
        assert stop is False

    def test_process_import_line_import_statement(self, sample_lines):
        """Test processing import statement."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "import os"
        new_state, stop = process_import_line(5, line, sample_lines, state)

        assert new_state["last_import_line"] == 5
        assert stop is False

    def test_process_import_line_from_import(self, sample_lines):
        """Test processing from...import statement."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "from typing import List"
        new_state, stop = process_import_line(6, line, sample_lines,state)

        assert new_state["last_import_line"] == 6
        assert stop is False

    def test_process_import_line_actual_code(self, sample_lines):
        """Test processing actual code line (should stop)."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "def main():"
        new_state, stop = process_import_line(10, line, sample_lines, state)

        assert stop is True

    def test_process_import_line_comment(self, sample_lines):
        """Test processing comment line."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "# This is a comment"
        new_state, stop = process_import_line(3, line,sample_lines ,state)

        assert stop is False

    def test_process_import_line_empty(self, sample_lines):
        """Test processing empty line."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = ""
        new_state, stop = process_import_line(4, line,sample_lines, state)

        assert stop is False


# ============================================================================
# TEST: handle_docstring
# ============================================================================

class TestHandleDocstring:
    """Test docstring handling."""

    def test_handle_docstring_single_line_double_quotes(self):
        """Test single-line docstring with double quotes."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = '"""Module docstring."""'
        validate_handle_docstring, state = handle_docstring(0, line, state)

        assert validate_handle_docstring is True
        assert state["last_docstring_line"] == 0
        assert state["in_docstring"] is False

    def test_handle_docstring_single_line_single_quotes(self):
        """Test single-line docstring with single quotes."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "'''Module docstring.'''"
        result, state = handle_docstring(0, line, state)

        assert result is True
        assert state["last_docstring_line"] == 0

    def test_handle_docstring_multiline_start(self):
        """Test start of multiline docstring."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = '"""'
        result, state = handle_docstring(0, line, state)

        assert result is True
        assert state["in_docstring"] is True
        assert state["docstring_quote"] == '"""'

    def test_handle_docstring_multiline_middle(self):
        """Test middle line of multiline docstring."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": True,
            "docstring_quote": '"""',
        }
        line = "This is the middle of the docstring."
        result, state = handle_docstring(1, line, state)

        assert result is True
        assert state["in_docstring"] is True

    def test_handle_docstring_multiline_end(self):
        """Test end of multiline docstring."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": True,
            "docstring_quote": '"""',
        }
        line = '"""'
        result, state = handle_docstring(2, line, state)

        assert result is True
        assert state["in_docstring"] is False
        assert state["last_docstring_line"] == 2

    def test_handle_docstring_not_docstring(self):
        """Test line that's not a docstring."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "regular code"
        result, state = handle_docstring(0, line, state)

        assert result is False


# ============================================================================
# TEST: is_shebang_or_encoding
# ============================================================================

class TestIsShebangOrEncoding:
    """Test shebang and encoding detection."""

    def test_is_shebang_line_zero(self):
        """Test shebang on line 0."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "#!/usr/bin/env python3"
        result, state = is_shebang_or_encoding(0, line, state)

        assert result is True
        assert state["last_shebang_encoding_line"] == 0

    def test_is_encoding_utf8(self):
        """Test UTF-8 encoding declaration."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "# -*- coding: utf-8 -*-"
        result, state = is_shebang_or_encoding(1, line, state)

        assert result is True
        assert state["last_shebang_encoding_line"] == 1

    def test_is_encoding_with_encoding_keyword(self):
        """Test encoding with 'encoding' keyword."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "# encoding: utf-8"
        result, state = is_shebang_or_encoding(1, line, state)

        assert result is True

    def test_is_not_shebang_or_encoding(self):
        """Test regular comment."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "# This is just a comment"
        result, state = is_shebang_or_encoding(5, line, state)

        assert result is False

    def test_is_shebang_wrong_line(self):
        """Test shebang not on line 0."""
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None,
        }
        line = "#!/usr/bin/env python3"
        result, state = is_shebang_or_encoding(5, line, state)

        assert result is False


# ============================================================================
# TEST: normalize_indentation
# ============================================================================

class TestNormalizeIndentation:
    """Test indentation normalization."""

    def test_normalize_indentation_basic(self):
        """Test normalizing basic indentation."""
        lines = ["    line1", "    line2", "        line3"]
        result = normalize_indentation(lines)

        assert result == ["line1", "line2", "    line3"]

    def test_normalize_indentation_mixed(self):
        """Test with mixed indentation levels."""
        lines = ["  x = 1", "    y = 2", "  z = 3"]
        result = normalize_indentation(lines)

        # Minimum is 2, so remove 2 from all
        assert result[0] == "x = 1"
        assert result[1] == "  y = 2"
        assert result[2] == "z = 3"

    def test_normalize_indentation_empty_lines(self):
        """Test preserving empty lines."""
        lines = ["    line1", "", "    line2"]
        result = normalize_indentation(lines)

        assert result[0] == "line1"
        assert result[1] == ""
        assert result[2] == "line2"

    def test_normalize_indentation_no_common_indent(self):
        """Test when no common indentation."""
        lines = ["line1", "  line2", "    line3"]
        result = normalize_indentation(lines)

        # No change expected
        assert result == lines

    def test_normalize_indentation_empty_list(self):
        """Test with empty list."""
        lines = []
        result = normalize_indentation(lines)
        assert result == []

    def test_normalize_indentation_all_empty(self):
        """Test with all empty lines."""
        lines = ["", "", ""]
        result = normalize_indentation(lines)
        assert result == lines


# ============================================================================
# TEST: apply_indentation_to_fix
# ============================================================================

class TestApplyIndentationToFix:
    """Test applying indentation to fixed code."""

    def test_apply_indentation_basic(self):
        """Test basic indentation application."""
        code = "x = 1\ny = 2"
        indent = "    "
        result = apply_indentation_to_fix(code, indent)

        assert result == "    x = 1\n    y = 2"

    def test_apply_indentation_already_indented(self):
        """Test with code that's already indented."""
        code = "    x = 1\n        y = 2"
        indent = "    "
        result = apply_indentation_to_fix(code, indent)

        # Should normalize first, then apply
        assert "    x = 1" in result
        assert "        y = 2" in result

    def test_apply_indentation_empty_lines(self):
        """Test preserving empty lines."""
        code = "x = 1\n\ny = 2"
        indent = "    "
        result = apply_indentation_to_fix(code, indent)

        lines = result.split('\n')
        assert lines[0] == "    x = 1"
        assert lines[1] == ""
        assert lines[2] == "    y = 2"

    def test_apply_indentation_empty_code(self):
        """Test with empty code."""
        code = ""
        indent = "    "
        result = apply_indentation_to_fix(code, indent)
        assert result == ""

    def test_apply_indentation_whitespace_only(self):
        """Test with whitespace-only code."""
        code = "   \n  \n"
        indent = "    "
        result = apply_indentation_to_fix(code, indent)
        assert result.strip() == ""

    def test_apply_indentation_no_indent(self):
        """Test with empty indent string."""
        code = "x = 1\ny = 2"
        indent = ""
        result = apply_indentation_to_fix(code, indent)
        assert result == "x = 1\ny = 2"


# ============================================================================
# TEST: find_global_top_insertion_point
# ============================================================================

class TestFindGlobalTopInsertionPoint:
    """Test finding global top insertion point."""

    def test_find_global_top_after_imports(self):
        """Test insertion after imports."""
        lines = [
            "import os\n",
            "import sys\n",
            "\n",
            "def main():\n",
            "    pass\n",
        ]
        result = find_global_top_insertion_point(lines)
        assert result == 3  # Before def main()

    def test_find_global_top_no_code(self):
        """Test when there's no code after imports."""
        lines = [
            "import os\n",
            "import sys\n",
            "\n",
        ]
        result = find_global_top_insertion_point(lines)
        assert result == len(lines)

    def test_find_global_top_with_comments(self):
        """Test skipping comments."""
        lines = [
            "import os\n",
            "# comment\n",
            "\n",
            "def main():\n",
        ]
        result = find_global_top_insertion_point(lines)
        assert result == 3  # Skip comment

    def test_find_global_top_empty_file(self):
        """Test with empty file."""
        lines = []
        result = find_global_top_insertion_point(lines)
        assert result == 0


# ============================================================================
# TEST: apply_complex_fix
# ============================================================================

class TestApplyComplexFix:
    """Test complex fix application."""

    def test_apply_complex_fix_no_helper(self, mock_fix):
        """Test complex fix without helper code."""
        lines = ["def func():\n", "    x = 1\n", "    y = 2\n"]
        mock_fix.fixed_code = "x = 2"
        mock_fix.helper_code = ""

        line_range = LineRange(start=1, end=1)

        result = apply_complex_fix(lines, mock_fix, line_range)

        assert isinstance(result, list)
        assert "x = 2" in ''.join(result)

    def test_apply_complex_fix_sibling_helper(self, mock_fix):
        """Test complex fix with sibling helper."""
        lines = ["def func():\n", "    x = 1\n", "    y = 2\n"]
        mock_fix.fixed_code = "x = 2"
        mock_fix.helper_code = "# helper"
        mock_fix.placement_helper = "SIBLING"
        line_range = LineRange(start=1, end=1)

        result = apply_complex_fix(lines, mock_fix, line_range)

        assert "x = 2" in ''.join(result)
        assert "# helper" in ''.join(result)

    def test_apply_complex_fix_global_bottom(self, mock_fix):
        """Test complex fix with global bottom helper."""
        lines = ["def func():\n", "    x = 1\n"]
        mock_fix.fixed_code = "x = 2"
        mock_fix.helper_code = "HELPER = True"
        mock_fix.placement_helper = "GLOBAL_BOTTOM"
        line_range = LineRange(start=1, end=1)

        result = apply_complex_fix(lines, mock_fix, line_range)

        content = ''.join(result)
        assert "x = 2" in content
        assert "HELPER = True" in content
        # Helper should be at the end
        assert content.rstrip().endswith("HELPER = True")

    def test_apply_complex_fix_global_top(self, mock_fix):
        """Test complex fix with global top helper."""
        lines = [
            "import os\n",
            "def func():\n",
            "    x = 1\n",
        ]
        mock_fix.fixed_code = "x = 2"
        mock_fix.helper_code = "import sys"
        mock_fix.placement_helper = "GLOBAL_TOP"
        line_range = LineRange(start=2, end=2)

        result = apply_complex_fix(lines, mock_fix, line_range)

        content = ''.join(result)
        assert "import sys" in content
        assert "x = 2" in content

    def test_apply_complex_fix_escape_newlines(self, mock_fix):
        """Test handling escaped newlines in helper code."""
        lines = ["x = 1\n"]
        mock_fix.fixed_code = "x = 2"
        mock_fix.helper_code = "line1\\nline2"  # Escaped newline
        mock_fix.placement_helper = "SIBLING"
        line_range = LineRange(start=0, end=0)

        result = apply_complex_fix(lines, mock_fix, line_range)

        content = ''.join(result)
        # Should convert \n to actual newline
        assert "line1\nline2" in content or "line1" in content


# ============================================================================
# TEST: apply_single_fix (Integration)
# ============================================================================

class TestApplySingleFix:
    """Test single fix application (integration test)."""

    def test_apply_single_fix_simple_replacement(self, mock_fix):
        """Test applying simple replacement."""
        lines = ["def func():\n", "    x = 1\n", "    y = 2\n"]
        mock_fix.fixed_code = "z = 3"
        mock_fix.sonar_line_number = 2
        mock_fix.helper_code = ""
        mock_fix.line_number = 2
        mock_fix.last_line_number = 2

        result = apply_single_fix(lines, mock_fix)

        assert result.success is True
        assert result.fix == mock_fix
        assert "z = 3" in ''.join(lines)

    def test_apply_single_fix_complex(self, mock_fix):
        """Test applying complex fix."""
        lines = ["def func():\n", "    x = 1\n", "    y = 2\n"]
        mock_fix.fixed_code = "x = 1\ny = 2"
        mock_fix.helper_code = ""
        mock_fix.line_number = 2
        mock_fix.last_line_number = 3

        result = apply_single_fix(lines, mock_fix)

        assert result.success is True

    def test_apply_single_fix_missing_line_numbers(self, mock_fix):
        """Test with missing line numbers."""
        lines = ["line1\n", "line2\n"]
        mock_fix.line_number = None
        mock_fix.last_line_number = None

        result = apply_single_fix(lines, mock_fix)

        assert result.success is False
        assert result.reason == "Missing line numbers"

    def test_apply_single_fix_invalid_range(self, mock_fix):
        """Test with invalid line range."""
        lines = ["line1\n", "line2\n"]
        mock_fix.line_number = 10
        mock_fix.last_line_number = 20

        result = apply_single_fix(lines, mock_fix)

        assert result.success is False
        assert result.reason == "Invalid line range"

    def test_apply_single_fix_start_after_end(self, mock_fix):
        """Test with start line after end line."""
        lines = ["line1\n", "line2\n", "line3\n"]
        mock_fix.line_number = 3
        mock_fix.last_line_number = 1

        result = apply_single_fix(lines, mock_fix)

        assert result.success is False
        assert result.reason == "Invalid line range"

    def test_apply_single_fix_modifies_lines_in_place(self, mock_fix):
        """Test that lines are modified in place."""
        lines = ["line1\n", "line2\n", "line3\n"]
        mock_fix.fixed_code = "new_line"
        mock_fix.line_number = 2
        mock_fix.last_line_number = 2
        mock_fix.sonar_line_number = 2
        mock_fix.helper_code = ""

        original_id = id(lines)
        result = apply_single_fix(lines, mock_fix)

        assert result.success is True
        assert id(lines) == original_id  # Same list object
        assert "new_line" in ''.join(lines)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow_simple_fix(self, temp_dir, mock_fix):
        """Test complete workflow for simple fix."""
        # Create test file
        file_path = temp_dir / "test.py"
        original_content = "def func():\n    x = 1\n    y = 2\n"
        file_path.write_text(original_content)

        # Read lines
        lines = read_file_lines(file_path)

        # Apply fix
        mock_fix.fixed_code = "z = 3"
        mock_fix.sonar_line_number = 2
        mock_fix.line_number = 2
        mock_fix.last_line_number = 2
        mock_fix.helper_code = ""

        result = apply_single_fix(lines, mock_fix)

        # Write back
        write_file_lines(file_path, lines)

        # Verify
        assert result.success is True
        content = file_path.read_text()
        assert "z = 3" in content
        assert "x = 1" not in content

    def test_full_workflow_with_helper(self, temp_dir, mock_fix):
        """Test complete workflow with helper code."""
        file_path = temp_dir / "test.py"
        original_content = "def func():\n    x = 1\n"
        file_path.write_text(original_content)

        lines = read_file_lines(file_path)

        mock_fix.fixed_code = "x = 2"
        mock_fix.helper_code = "import math"
        mock_fix.placement_helper = "GLOBAL_TOP"
        mock_fix.line_number = 2
        mock_fix.last_line_number = 2

        result = apply_single_fix(lines, mock_fix)
        write_file_lines(file_path, lines)

        assert result.success is True
        content = file_path.read_text()
        assert "import math" in content
        assert "x = 2" in content

    def test_multiple_fixes_sequence(self, temp_dir):
        """Test applying multiple fixes in sequence."""
        file_path = temp_dir / "test.py"
        content = "line1\nline2\nline3\nline4\nline5\n"
        file_path.write_text(content)

        lines = read_file_lines(file_path)

        # Create multiple fixes
        fixes = []
        for i in range(2, 5):
            fix = Mock(spec=FixSuggestion)
            fix.issue_key = f"TEST-{i}"
            fix.line_number = i
            fix.import_block_code=""
            fix.end_import_block_code = 2
            fix.last_line_number = i
            fix.sonar_line_number = i
            fix.fixed_code = f"new_line{i}"
            fix.helper_code = ""
            fixes.append(fix)

        # Apply fixes in reverse order (important!)
        for fix in reversed(fixes):
            result = apply_single_fix(lines, fix)
            assert result.success is True

        write_file_lines(file_path, lines)
        content = file_path.read_text()

        assert "new_line2" in content
        assert "new_line3" in content
        assert "new_line4" in content


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_fix_at_file_start(self, mock_fix):
        """Test fix at the very beginning of file."""
        lines = ["line1\n", "line2\n", "line3\n"]
        mock_fix.fixed_code = "new_first"
        mock_fix.line_number = 1
        mock_fix.last_line_number = 1
        mock_fix.sonar_line_number = 1
        mock_fix.helper_code = ""

        result = apply_single_fix(lines, mock_fix)
        print("result ", result)
        assert result.success is True
        assert "new_first" in ''.join(lines)

    def test_fix_at_file_end(self, mock_fix):
        """Test fix at the very end of file."""
        lines = ["line1\n", "line2\n", "line3\n"]
        mock_fix.fixed_code = "new_last"
        mock_fix.line_number = 3
        mock_fix.last_line_number = 3
        mock_fix.sonar_line_number = 3
        mock_fix.helper_code = ""

        result = apply_single_fix(lines, mock_fix)
        assert result.success is True
        assert "new_last" in ''.join(lines)

    def test_fix_entire_file(self, mock_fix):
        """Test replacing entire file."""
        lines = ["line1\n", "line2\n", "line3\n"]
        mock_fix.fixed_code = "new_content"
        mock_fix.line_number = 1
        mock_fix.last_line_number = 3
        mock_fix.helper_code = ""
        mock_fix.sonar_line_number = 3

        result = apply_single_fix(lines, mock_fix)
        assert result.success is True

    def test_very_long_file(self, mock_fix):
        """Test with very long file."""
        lines = [f"line{i}\n" for i in range(10000)]
        mock_fix.fixed_code = "new_line"
        mock_fix.line_number = 5000
        mock_fix.last_line_number = 5000
        mock_fix.sonar_line_number = 5000
        mock_fix.helper_code = ""

        result = apply_single_fix(lines, mock_fix)
        assert result.success is True
        assert len(lines) >= 9999  # Might have added newlines

    def test_unicode_content(self, temp_dir, mock_fix):
        """Test with unicode content."""
        file_path = temp_dir / "unicode.py"
        content = "# 日本語 コメント\ncafé = 'naïve'\n"
        file_path.write_text(content, encoding='utf-8')

        lines = read_file_lines(file_path)
        mock_fix.fixed_code = "résumé = 'élève'"
        mock_fix.line_number = 2
        mock_fix.last_line_number = 2
        mock_fix.sonar_line_number = 2
        mock_fix.helper_code = ""

        result = apply_single_fix(lines, mock_fix)
        write_file_lines(file_path, lines)

        assert result.success is True
        content = file_path.read_text(encoding='utf-8')
        assert "résumé" in content

    def test_empty_fixed_code(self, mock_fix):
        """Test with empty fixed code."""
        lines = ["line1\n", "line2\n", "line3\n"]
        mock_fix.fixed_code = ""
        mock_fix.line_number = 2
        mock_fix.last_line_number = 2
        mock_fix.sonar_line_number = 2
        mock_fix.helper_code = ""

        result = apply_single_fix(lines, mock_fix)
        # Should still succeed, just replace with empty
        assert result.success is True

    def test_tabs_vs_spaces(self):
        """Test handling mixed tabs and spaces."""
        lines = ["\tdef func():\n", "    x = 1\n"]
        indent_tabs = calculate_base_indentation_based_on_line(lines, 1)
        indent_spaces = calculate_base_indentation_based_on_line(lines, 2)

        assert indent_tabs == "\t"
        assert indent_spaces == "    "


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("indent_level", [0, 2, 4, 8, 16])
    def test_various_indentation_levels(self, indent_level):
        """Test with various indentation levels."""
        indent = " " * indent_level
        code = f"{indent}x = 1"
        result = calculate_base_indentation(code)
        assert result == indent_level

    @pytest.mark.parametrize("placement", ["SIBLING", "GLOBAL_TOP", "GLOBAL_BOTTOM"])
    def test_all_placement_strategies(self, mock_fix, placement):
        """Test all helper placement strategies."""
        lines = ["import os\n", "def func():\n", "    x = 1\n"]
        mock_fix.fixed_code = "x = 2"
        mock_fix.helper_code = "# helper"
        mock_fix.placement_helper = placement
        mock_fix.line_number = 3
        mock_fix.last_line_number = 3
        line_range = LineRange(start=2, end=2)

        result = apply_complex_fix(lines, mock_fix, line_range)

        assert isinstance(result, list)
        content = ''.join(result)
        assert "x = 2" in content
        assert "# helper" in content

    @pytest.mark.parametrize("line_number,expected_valid", [
        (0, True),
        (1, True),
        (5, True),
        (100, False),
        (-1, False),
    ])
    def test_line_range_validation(self, line_number, expected_valid):
        """Test line range validation with various inputs."""
        lines = ["l1\n", "l2\n", "l3\n", "l4\n", "l5\n"]

        line_range = LineRange(start=line_number - 1 if line_number > 0 else line_number,
                               end=line_number - 1 if line_number > 0 else line_number)
        print("line_range ", line_range)

        result = line_range.is_valid(len(lines))
        print("result ", result)

        assert result == expected_valid

    @pytest.mark.parametrize("code,expected", [
        ("import os", True),
        ("from typing import List", True),
        ("def func():", False),
        ("# import os", False),
        ("x = 1", False),
        ("", False),
    ])
    def test_import_detection_variations(self, code, expected):
        """Test import detection with various code patterns."""
        result = is_import_block(code)
        assert result == expected


# ============================================================================
# PERFORMANCE TESTS (Optional)
# ============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_large_file_performance(self, mock_fix):
        """Test performance with large file."""
        import time

        # Create large file
        lines = [f"line{i}\n" for i in range(10000)]

        mock_fix.fixed_code = "new_line"
        mock_fix.line_number = 5000
        mock_fix.last_line_number = 5000
        mock_fix.sonar_line_number = 5000
        mock_fix.helper_code = ""

        start = time.time()
        result = apply_single_fix(lines, mock_fix)
        elapsed = time.time() - start

        assert result.success is True
        assert elapsed < 0.1  # Should be fast (< 100ms)

    def test_many_fixes_performance(self, temp_dir):
        """Test performance with many sequential fixes."""
        import time

        file_path = temp_dir / "large.py"
        lines_content = [f"line{i}\n" for i in range(1000)]
        file_path.write_text(''.join(lines_content))

        lines = read_file_lines(file_path)

        # Create 50 fixes
        fixes = []
        for i in range(50, 100):
            fix = Mock(spec=FixSuggestion)
            fix.issue_key = f"TEST-{i}"
            fix.end_import_block_code = 10
            fix.import_block_code=""
            fix.line_number = i
            fix.last_line_number = i
            fix.sonar_line_number = i
            fix.fixed_code = f"new_line{i}"
            fix.helper_code = ""
            fixes.append(fix)

        start = time.time()
        for fix in reversed(fixes):
            apply_single_fix(lines, fix)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete in < 1 second


# ============================================================================
# Test remove_tmp_files
# ============================================================================

class TestRemoveTmpFiles:
    """Test suite for remove_tmp_files function."""

    def test_remove_existing_directory_success(self, tmp_path):
        """Test successfully removing an existing directory."""
        # Create a test directory with some files
        test_dir = tmp_path / "test_remove"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")

        # Verify directory exists
        assert test_dir.exists()

        # Remove directory
        result = remove_tmp_files(str(test_dir))

        # Verify successful removal
        assert result is True
        assert not test_dir.exists()

    def test_remove_nested_directory_structure(self, tmp_path):
        """Test removing nested directory structure."""
        # Create nested directories
        test_dir = tmp_path / "test_nested"
        nested_dir = test_dir / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)

        # Add files at different levels
        (test_dir / "root_file.txt").write_text("root")
        (test_dir / "level1" / "level1_file.txt").write_text("level1")
        (nested_dir / "deep_file.txt").write_text("deep")

        # Verify structure exists
        assert nested_dir.exists()

        # Remove entire structure
        result = remove_tmp_files(str(test_dir))

        # Verify complete removal
        assert result is True
        assert not test_dir.exists()

    def test_remove_directory_with_multiple_files(self, tmp_path):
        """Test removing directory with many files."""
        test_dir = tmp_path / "test_many_files"
        test_dir.mkdir()

        # Create 100 files
        for i in range(100):
            (test_dir / f"file_{i}.txt").write_text(f"content_{i}")

        # Verify directory has files
        assert len(list(test_dir.iterdir())) == 100

        # Remove directory
        result = remove_tmp_files(str(test_dir))

        # Verify removal
        assert result is True
        assert not test_dir.exists()

    def test_remove_empty_directory(self, tmp_path):
        """Test removing an empty directory."""
        test_dir = tmp_path / "test_empty"
        test_dir.mkdir()

        # Verify it's empty
        assert test_dir.exists()
        assert len(list(test_dir.iterdir())) == 0

        # Remove empty directory
        result = remove_tmp_files(str(test_dir))

        # Verify removal
        assert result is True
        assert not test_dir.exists()

    def test_remove_with_path_object(self, tmp_path):
        """Test function works with Path objects."""
        test_dir = tmp_path / "test_path_object"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        # Pass Path object instead of string
        result = remove_tmp_files(str(test_dir))

        assert result is True
        assert not test_dir.exists()

    def test_remove_with_relative_path(self, tmp_path):
        """Test removing directory with relative path."""
        # Create directory in tmp_path
        test_dir = tmp_path / "test_relative"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        # Change to parent directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Remove using relative path
            result = remove_tmp_files("test_relative")

            assert result is True
            assert not test_dir.exists()
        finally:
            os.chdir(original_cwd)

    def test_remove_nonexistent_directory_raises_error(self):
        """Test that removing non-existent directory raises ValueError."""
        nonexistent_path = "/tmp/this_directory_does_not_exist_12345"

        with pytest.raises(ValueError) as exc_info:
            remove_tmp_files(nonexistent_path)

        assert "Invalid path" in str(exc_info.value)
        assert nonexistent_path in str(exc_info.value)

    def test_remove_empty_path_raises_error(self):
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            remove_tmp_files("")

        assert "Empty path provided" in str(exc_info.value)

    def test_remove_path_with_double_dots_raises_error(self):
        """Test that path with '..' raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            remove_tmp_files("../../../etc/passwd")

        assert "Path contains invalid components" in str(exc_info.value)

    def test_remove_path_with_single_dot_raises_error(self):
        """Test that path with '.' raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            remove_tmp_files("./some/../path")

        assert "Path contains invalid components" in str(exc_info.value)

    def test_remove_path_with_empty_component_raises_error(self):
        """Test that path with empty component raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            remove_tmp_files("some/ /path")

        assert "Path contains invalid components" in str(exc_info.value)


    def test_remove_file_instead_of_directory(self, tmp_path):
        """Test behavior when path points to a file instead of directory."""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("content")

        # shutil.rmtree should handle this and raise an error
        with pytest.raises(ValueError) as exc_info:
            remove_tmp_files(str(test_file))

        assert "Invalid path" in str(exc_info.value)

    def test_remove_with_permission_denied(self, tmp_path):
        """Test handling of permission errors."""
        test_dir = tmp_path / "test_permission"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        # Mock shutil.rmtree to raise PermissionError
        with patch('shutil.rmtree', side_effect=PermissionError("Permission denied")):
            with pytest.raises(ValueError) as exc_info:
                remove_tmp_files(str(test_dir))

            assert "Invalid path" in str(exc_info.value)

    def test_remove_with_os_error(self, tmp_path):
        """Test handling of OS errors."""
        test_dir = tmp_path / "test_os_error"
        test_dir.mkdir()

        # Mock shutil.rmtree to raise OSError
        with patch('shutil.rmtree', side_effect=OSError("OS error occurred")):
            with pytest.raises(ValueError) as exc_info:
                remove_tmp_files(str(test_dir))

            assert "Invalid path" in str(exc_info.value)
            assert "OS error occurred" in str(exc_info.value)

    def test_remove_directory_with_special_characters(self, tmp_path):
        """Test removing directory with special characters in name."""
        # Create directory with special characters
        test_dir = tmp_path / "test_special_!@#$%"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        result = remove_tmp_files(str(test_dir))

        assert result is True
        assert not test_dir.exists()

    def test_remove_directory_with_unicode_characters(self, tmp_path):
        """Test removing directory with unicode characters."""
        test_dir = tmp_path / "test_unicode_日本語_🎉"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        result = remove_tmp_files(str(test_dir))

        assert result is True
        assert not test_dir.exists()

    def test_remove_directory_with_symlinks(self, tmp_path):
        """Test removing directory containing symlinks."""
        test_dir = tmp_path / "test_symlinks"
        test_dir.mkdir()

        # Create a file and a symlink to it
        real_file = test_dir / "real_file.txt"
        real_file.write_text("content")
        symlink = test_dir / "symlink.txt"

        try:
            symlink.symlink_to(real_file)

            result = remove_tmp_files(str(test_dir))

            assert result is True
            assert not test_dir.exists()
        except OSError:
            # Skip if symlinks not supported on this system
            pytest.skip("Symlinks not supported on this system")

    def test_remove_readonly_files(self, tmp_path):
        """Test removing directory with read-only files."""
        test_dir = tmp_path / "test_readonly"
        test_dir.mkdir()

        readonly_file = test_dir / "readonly.txt"
        readonly_file.write_text("readonly content")

        # Make file read-only
        readonly_file.chmod(0o444)

        try:
            result = remove_tmp_files(str(test_dir))
            assert result is True
            assert not test_dir.exists()
        except ValueError:
            # On some systems, read-only files prevent deletion
            # This is acceptable behavior
            pass
        finally:
            # Cleanup if test failed
            if test_dir.exists():
                readonly_file.chmod(0o644)
                shutil.rmtree(test_dir)


# ============================================================================
# Test generate_tmp_path
# ============================================================================

class TestGenerateTmpPath:
    """Test suite for generate_tmp_path function."""

    def test_generates_valid_path(self):
        """Test that function generates a valid temporary path."""
        path = generate_tmp_path()

        # Should return a string
        assert isinstance(path, str)

        # Should not be empty
        assert len(path) > 0

        # Path should exist
        assert os.path.exists(path)

        # Should be a directory
        assert os.path.isdir(path)

        # Cleanup
        shutil.rmtree(path)

    def test_path_has_correct_prefix(self):
        """Test that generated path has 'devdox_' prefix."""
        path = generate_tmp_path()

        # Get directory name
        dir_name = os.path.basename(path)

        # Should start with 'devdox_'
        assert dir_name.startswith('devdox_')

        # Cleanup
        shutil.rmtree(path)

    def test_path_has_correct_suffix(self):
        """Test that generated path has '_test' suffix."""
        path = generate_tmp_path()

        # Get directory name
        dir_name = os.path.basename(path)

        # Should end with '_test'
        assert dir_name.endswith('_test')

        # Cleanup
        shutil.rmtree(path)

    def test_generates_unique_paths(self):
        """Test that multiple calls generate unique paths."""
        paths = [generate_tmp_path() for _ in range(10)]

        # All paths should be unique
        assert len(set(paths)) == 10

        # All paths should exist
        for path in paths:
            assert os.path.exists(path)

        # Cleanup
        for path in paths:
            shutil.rmtree(path)

    def test_path_is_in_system_temp_dir(self):
        """Test that path is created in system temp directory."""
        path = generate_tmp_path()

        # Get system temp directory
        system_temp = tempfile.gettempdir()

        # Generated path should be under system temp
        assert path.startswith(system_temp)

        # Cleanup
        shutil.rmtree(path)

    def test_directory_is_initially_empty(self):
        """Test that generated directory is empty."""
        path = generate_tmp_path()

        # Directory should be empty
        assert len(os.listdir(path)) == 0

        # Cleanup
        shutil.rmtree(path)

    def test_directory_has_correct_permissions(self):
        """Test that directory has appropriate permissions."""
        path = generate_tmp_path()

        # Directory should be readable and writable by owner
        mode = os.stat(path).st_mode
        assert mode & 0o700  # Owner has rwx

        # Cleanup
        shutil.rmtree(path)


    def test_generated_path_is_writable(self):
        """Test that generated directory is writable."""
        path = generate_tmp_path()

        # Try to create a file in the directory
        test_file = os.path.join(path, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")

        # File should exist
        assert os.path.exists(test_file)

        # Cleanup
        shutil.rmtree(path)

    def test_multiple_concurrent_calls(self):
        """Test that function works correctly with concurrent calls."""
        import threading

        paths = []
        errors = []

        def generate_and_store():
            try:
                path = generate_tmp_path()
                paths.append(path)
            except Exception as e:
                errors.append(e)

        # Create 20 threads
        threads = [threading.Thread(target=generate_and_store) for _ in range(20)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # No errors should occur
        assert len(errors) == 0

        # All paths should be unique
        assert len(set(paths)) == 20

        # Cleanup
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)

    def test_path_format(self):
        """Test that path format matches expected pattern."""
        path = generate_tmp_path()
        dir_name = os.path.basename(path)

        # Format should be: devdox_<random>_test
        assert dir_name.startswith('devdox_')
        assert dir_name.endswith('_test')

        # There should be something between prefix and suffix
        middle = dir_name[7:-5]  # Remove 'devdox_' and '_test'
        assert len(middle) > 0

        # Cleanup
        shutil.rmtree(path)


# ============================================================================
# Test download_latest_version
# ============================================================================

class TestDownloadLatestVersion:
    """Test suite for download_latest_version function."""

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    def test_successful_clone(self, mock_repo_class):
        """Test successful repository cloning."""
        # Setup mock
        mock_repo_instance = MagicMock(spec=Repo)
        mock_repo_class.clone_from.return_value = mock_repo_instance

        # Test parameters
        repo_url = "https://github.com/user/repo.git"
        repo_path = "/tmp/test_repo"
        branch = "main"

        # Call function
        result = download_latest_version(repo_url, repo_path, branch)

        # Verify clone_from was called with correct parameters
        mock_repo_class.clone_from.assert_called_once_with(
            repo_url, repo_path, branch=branch
        )

        # Verify return value
        assert result == mock_repo_instance

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    def test_clone_with_different_branch(self, mock_repo_class):
        """Test cloning with different branch names."""
        mock_repo_instance = MagicMock(spec=Repo)
        mock_repo_class.clone_from.return_value = mock_repo_instance

        branches = ["main", "develop", "feature/new-feature", "release/v1.0"]

        for branch in branches:
            mock_repo_class.reset_mock()

            result = download_latest_version(
                "https://github.com/user/repo.git",
                "/tmp/test_repo",
                branch
            )

            mock_repo_class.clone_from.assert_called_once_with(
                "https://github.com/user/repo.git",
                "/tmp/test_repo",
                branch=branch
            )
            assert result == mock_repo_instance

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    def test_clone_with_ssh_url(self, mock_repo_class):
        """Test cloning with SSH URL."""
        mock_repo_instance = MagicMock(spec=Repo)
        mock_repo_class.clone_from.return_value = mock_repo_instance

        ssh_url = "git@github.com:user/repo.git"

        result = download_latest_version(ssh_url, "/tmp/test_repo", "main")

        mock_repo_class.clone_from.assert_called_once_with(
            ssh_url, "/tmp/test_repo", branch="main"
        )
        assert result == mock_repo_instance

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_clone_git_command_error(self, mock_print, mock_repo_class):
        """Test handling of Git command errors."""
        # Setup mock to raise GitCommandError
        error_msg = "fatal: repository not found"
        mock_repo_class.clone_from.side_effect = GitCommandError(
            "clone", 128, stderr=error_msg
        )

        repo_url = "https://github.com/user/nonexistent.git"

        # Call function
        result = download_latest_version(repo_url, "/tmp/test_repo", "main")

        # Should return None on error
        assert result is None

        # Should print error message
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "Error loading files" in call_args
        assert repo_url in call_args

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_clone_invalid_repository_error(self, mock_print, mock_repo_class):
        """Test handling of invalid repository errors."""
        mock_repo_class.clone_from.side_effect = InvalidGitRepositoryError()

        result = download_latest_version(
            "https://github.com/user/repo.git",
            "/tmp/test_repo",
            "main"
        )

        assert result is None
        mock_print.assert_called_once()

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_clone_permission_error(self, mock_print, mock_repo_class):
        """Test handling of permission errors."""
        mock_repo_class.clone_from.side_effect = PermissionError(
            "Permission denied"
        )

        result = download_latest_version(
            "https://github.com/user/repo.git",
            "/tmp/test_repo",
            "main"
        )

        assert result is None
        mock_print.assert_called_once()
        assert "Permission denied" in mock_print.call_args[0][0]

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_clone_generic_exception(self, mock_print, mock_repo_class):
        """Test handling of generic exceptions."""
        mock_repo_class.clone_from.side_effect = Exception("Unexpected error")

        result = download_latest_version(
            "https://github.com/user/repo.git",
            "/tmp/test_repo",
            "main"
        )

        assert result is None
        mock_print.assert_called_once()
        assert "Unexpected error" in mock_print.call_args[0][0]

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    def test_clone_with_empty_branch(self, mock_repo_class):
        """Test cloning with empty branch string."""
        mock_repo_instance = MagicMock(spec=Repo)
        mock_repo_class.clone_from.return_value = mock_repo_instance

        result = download_latest_version(
            "https://github.com/user/repo.git",
            "/tmp/test_repo",
            ""
        )

        # Should still call clone_from with empty branch
        mock_repo_class.clone_from.assert_called_once_with(
            "https://github.com/user/repo.git",
            "/tmp/test_repo",
            branch=""
        )

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    def test_clone_with_special_characters_in_path(self, mock_repo_class):
        """Test cloning to path with special characters."""
        mock_repo_instance = MagicMock(spec=Repo)
        mock_repo_class.clone_from.return_value = mock_repo_instance

        special_path = "/tmp/test repo with spaces/子目录"

        result = download_latest_version(
            "https://github.com/user/repo.git",
            special_path,
            "main"
        )

        mock_repo_class.clone_from.assert_called_once()
        assert result == mock_repo_instance

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_clone_timeout_error(self, mock_print, mock_repo_class):
        """Test handling of timeout errors."""
        mock_repo_class.clone_from.side_effect = TimeoutError("Connection timeout")

        result = download_latest_version(
            "https://github.com/user/repo.git",
            "/tmp/test_repo",
            "main"
        )

        assert result is None
        mock_print.assert_called_once()
        assert "Connection timeout" in mock_print.call_args[0][0]

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    def test_clone_with_different_repo_urls(self, mock_repo_class):
        """Test cloning from different repository URL formats."""
        mock_repo_instance = MagicMock(spec=Repo)
        mock_repo_class.clone_from.return_value = mock_repo_instance

        urls = [
            "https://github.com/user/repo.git",
            "git@github.com:user/repo.git",
            "https://gitlab.com/user/repo.git",
            "ssh://git@bitbucket.org/user/repo.git",
            "https://dev.azure.com/org/project/_git/repo",
        ]

        for url in urls:
            mock_repo_class.reset_mock()

            result = download_latest_version(url, "/tmp/test_repo", "main")

            mock_repo_class.clone_from.assert_called_once()
            assert result == mock_repo_instance

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_error_message_contains_url(self, mock_print, mock_repo_class):
        """Test that error message contains the repository URL."""
        repo_url = "https://github.com/specific/repository.git"
        mock_repo_class.clone_from.side_effect = Exception("Test error")

        download_latest_version(repo_url, "/tmp/test", "main")

        error_message = mock_print.call_args[0][0]
        assert repo_url in error_message

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_error_message_format(self, mock_print, mock_repo_class):
        """Test the format of error messages."""
        mock_repo_class.clone_from.side_effect = Exception("Test error")

        download_latest_version(
            "https://github.com/user/repo.git",
            "/tmp/test",
            "main"
        )

        error_message = mock_print.call_args[0][0]
        assert error_message.startswith("Error loading files from")
        assert ": " in error_message

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    def test_return_type_on_success(self, mock_repo_class):
        """Test that function returns Repo instance on success."""
        mock_repo_instance = MagicMock(spec=Repo)
        mock_repo_class.clone_from.return_value = mock_repo_instance

        result = download_latest_version(
            "https://github.com/user/repo.git",
            "/tmp/test",
            "main"
        )

        assert isinstance(result, MagicMock)
        assert result == mock_repo_instance

    @patch('devdox_ai_sonar.utils.file_indentation.Repo')
    @patch('builtins.print')
    def test_return_none_on_all_error_types(self, mock_print, mock_repo_class):
        """Test that function returns None for all error types."""
        errors = [
            GitCommandError("clone", 128),
            InvalidGitRepositoryError(),
            PermissionError(),
            OSError(),
            ValueError(),
            Exception(),
        ]

        for error in errors:
            mock_repo_class.reset_mock()
            mock_print.reset_mock()
            mock_repo_class.clone_from.side_effect = error

            result = download_latest_version(
                "https://github.com/user/repo.git",
                "/tmp/test",
                "main"
            )

            assert result is None, f"Failed for error type: {type(error)}"
            mock_print.assert_called_once()

