"""Live end-to-end tests for apply_fixes_safe (from llm_fixer.py).

No mocks — real filesystem, real apply_single_fix, real Python interpreter.

Line map for REALISTIC_FILE (used by stress tests):
     1: import os                          16: def load        31: def process_all
     2: import sys                         17:   docstring     32:   docstring
     3: (blank)                            18:   with open     33:   results = []
     4: (blank)                            19:     for line    34:   for rec
     5: MAX_RETRIES = 3                    20:       append    35:     append
     6: (blank)                            21:   return        36:   return
     7: (blank)                            22: (blank)         37: (blank)
     8: class DataProcessor:               23: def transform   38: def save
     9:   docstring                        24:   docstring     39:   docstring
    10: (blank)                            25:   if cache      40:   processed =
    11: def __init__                       26:     return      41:   with open
    12:   self.source_path                 27:   result =      42:     for item
    13:   self.records                     28:   cache[]=      43:       f.write
    14:   self._cache                      29:   return        44: (blank)
    15: (blank)                            30: (blank)         45: (blank)
                                           46: def helper_function
                                           47:   docstring
                                           48:   return value * 2
"""

import subprocess
import sys as _sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from devdox_ai_sonar.llm_fixer import apply_fixes_safe
from devdox_ai_sonar.models.sonar import (
    BlockType,
    ChangeType,
    CodeBlock,
    FixSuggestion,
)


def _validate_python(file_path: Path) -> Tuple[bool, Optional[str]]:
    try:
        subprocess.run(
            [_sys.executable, "-m", "py_compile", str(file_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or e.stdout or "Unknown error").strip()


def _make_fix(
    key: str,
    original: str,
    fixed: str,
    line: int,
    end_line: int = 0,
    helper_code: str = "",
    placement_helper: str = "",
    import_block_code: str = "",
    end_import_block_code: Optional[int] = None,
) -> FixSuggestion:
    if end_line == 0:
        end_line = line
    cb = CodeBlock(
        block_name="t",
        start_line=line,
        end_line=end_line,
        has_changes=True,
        change_type=ChangeType.FULL_CODE,
        block_type=BlockType.MODULE,
        context=fixed,
    )
    return FixSuggestion(
        issue_key=key,
        file_path="test.py",
        original_code=original,
        fixed_code=fixed,
        explanation="fix",
        confidence=0.9,
        sonar_line_number=line,
        line_number=line,
        last_line_number=end_line,
        llm_model="m",
        fixed_code_blocks=[cb],
        helper_code=helper_code,
        placement_helper=placement_helper,
        import_block_code=import_block_code,
        end_import_block_code=end_import_block_code,
    )


def _run(target: Path, fixes: List[FixSuggestion]):
    return apply_fixes_safe(
        target, fixes,
        target.read_text().splitlines(keepends=True),
        _validate_python,
    )


# ---------------------------------------------------------------------------
# Basic live tests
# ---------------------------------------------------------------------------

class TestLiveApplyFixesSafe:

    def test_all_valid_fixes_applied(self, tmp_path):
        target = tmp_path / "target.py"
        target.write_text("x = 1\ny = 2\nz = 3\n")

        fix1 = _make_fix("k1", "x = 1", "x = 10", 1)
        fix2 = _make_fix("k2", "y = 2", "y = 20", 2)
        fix3 = _make_fix("k3", "z = 3", "z = 30", 3)

        success, results = _run(target, [fix1, fix2, fix3])

        assert success is True
        assert len(results) == 3
        assert target.read_text() == "x = 10\ny = 20\nz = 30\n"

    def test_invalid_fix_rejected_others_survive(self, tmp_path):
        target = tmp_path / "target.py"
        target.write_text("x = 1\ny = 2\nz = 3\n")

        fix1 = _make_fix("k1", "x = 1", "x = 10", 1)
        fix2 = _make_fix("k2", "y = 2", "y = !!!", 2)
        fix3 = _make_fix("k3", "z = 3", "z = 30", 3)

        success, results = _run(target, [fix1, fix2, fix3])

        assert success is False
        content = target.read_text()
        assert "!!!" not in content
        assert content == "x = 10\ny = 2\nz = 30\n"

    def test_all_invalid_fixes_leave_file_untouched(self, tmp_path):
        target = tmp_path / "target.py"
        original = "x = 1\ny = 2\n"
        target.write_text(original)

        fix1 = _make_fix("k1", "x = 1", "x = !!!", 1)
        fix2 = _make_fix("k2", "y = 2", "y = @@@", 2)

        success, results = _run(target, [fix1, fix2])

        assert success is False
        assert target.read_text() == original

    def test_rejected_fix_does_not_poison_subsequent_fix(self, tmp_path):
        target = tmp_path / "target.py"
        target.write_text("a = 1\nb = 2\nc = 3\n")

        fix1 = _make_fix("k1", "a = 1", "a = 10", 1)
        fix2 = _make_fix("k2", "b = 2", "b = !!!!", 2)
        fix3 = _make_fix("k3", "c = 3", "c = 30", 3)

        success, results = _run(target, [fix1, fix2, fix3])

        content = target.read_text()
        assert "a = 10" in content
        assert "!!!!" not in content
        assert "c = 30" in content
        assert "b = 2" in content

    def test_no_temp_file_left_behind(self, tmp_path):
        target = tmp_path / "target.py"
        target.write_text("x = 1\n")
        _run(target, [_make_fix("k1", "x = 1", "x = 10", 1)])

        remaining = [f for f in tmp_path.iterdir() if f.name != "__pycache__"]
        assert len(remaining) == 1 and remaining[0].name == "target.py"

    def test_empty_fix_list_does_not_touch_file(self, tmp_path):
        target = tmp_path / "target.py"
        original = "x = 1\n"
        target.write_text(original)
        mtime_before = target.stat().st_mtime

        success, results = apply_fixes_safe(
            target, [], [original], _validate_python,
        )

        assert success is True
        assert results == []
        assert target.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# Stress tests — realistic multi-line, import, helper, indentation scenarios
# ---------------------------------------------------------------------------

REALISTIC_FILE = '''\
import os
import sys


MAX_RETRIES = 3


class DataProcessor:
    """Processes incoming data records."""

    def __init__(self, source_path):
        self.source_path = source_path
        self.records = []
        self._cache = {}

    def load(self):
        """Load records from source file."""
        with open(self.source_path) as f:
            for line in f:
                self.records.append(line.strip())
        return self.records

    def transform(self, record):
        """Transform a single record."""
        if record in self._cache:
            return self._cache[record]
        result = record.upper()
        self._cache[record] = result
        return result

    def process_all(self):
        """Process all loaded records."""
        results = []
        for rec in self.records:
            results.append(self.transform(rec))
        return results

    def save(self, output_path):
        """Save processed records to output."""
        processed = self.process_all()
        with open(output_path, "w") as f:
            for item in processed:
                f.write(item + "\\n")


def helper_function(value):
    """A standalone helper."""
    return value * 2
'''


class TestLiveApplyFixesSafeStress:

    def test_multiline_method_replacement(self, tmp_path):
        """Replace a multi-line method body inside a class."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        new_transform = (
            '    def transform(self, record):\n'
            '        """Transform a single record with validation."""\n'
            '        if not isinstance(record, str):\n'
            '            raise TypeError("record must be a string")\n'
            '        if record in self._cache:\n'
            '            return self._cache[record]\n'
            '        result = record.strip().upper()\n'
            '        self._cache[record] = result\n'
            '        return result'
        )
        # transform is lines 23-29
        fix = _make_fix("s1", "def transform", new_transform, line=23, end_line=29)

        success, results = _run(target, [fix])

        assert success is True
        content = target.read_text()
        assert 'raise TypeError("record must be a string")' in content
        assert "record.strip().upper()" in content
        assert "class DataProcessor:" in content
        assert "def load(self):" in content
        assert "def process_all(self):" in content

    def test_multiline_replacement_with_broken_syntax_rejected(self, tmp_path):
        """Multi-line replacement that breaks syntax should be rejected."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)
        original = target.read_text()

        broken_method = (
            '    def transform(self, record):\n'
            '        if record in self._cache\n'  # missing colon
            '            return self._cache[record]\n'
            '        return record.upper()'
        )
        fix = _make_fix("s1", "def transform", broken_method, line=23, end_line=29)

        success, results = _run(target, [fix])

        assert success is False
        assert target.read_text() == original

    def test_two_methods_fixed_one_rejected(self, tmp_path):
        """Fix load() and process_all(), reject transform().

        Replacements preserve line count to avoid stale-line-number issues
        when fixes are applied sequentially with absolute line numbers.
        """
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        # load is 6 lines (16-21), replacement also 6 lines
        new_load = (
            '    def load(self):\n'
            '        """Load and deduplicate records."""\n'
            '        seen = set()\n'
            '        with open(self.source_path) as f:\n'
            '            self.records = [l.strip() for l in f if l.strip() not in seen]\n'
            '        return self.records'
        )
        # transform is 7 lines (23-29), broken replacement
        broken_transform = (
            '    def transform(self, record):\n'
            '        return record.upper(\n'  # unclosed paren
        )
        # process_all is 6 lines (31-36), replacement also 6 lines
        new_process_all = (
            '    def process_all(self):\n'
            '        """Process using list comprehension."""\n'
            '        return [self.transform(rec) for rec in self.records]\n'
            '\n'
            '\n'
            '\n'
        )

        fix1 = _make_fix("s1", "def load", new_load, line=16, end_line=21)
        fix2 = _make_fix("s2", "def transform", broken_transform, line=23, end_line=29)
        fix3 = _make_fix("s3", "def process_all", new_process_all, line=31, end_line=36)

        success, results = _run(target, [fix1, fix2, fix3])

        assert success is False
        content = target.read_text()
        assert "seen = set()" in content, "fix1 (load) should be present"
        assert "return record.upper(" not in content, "fix2 (broken) must not leak"
        assert "list comprehension" in content, "fix3 (process_all) should be present"
        assert "self._cache[record] = result" in content, "original transform intact"

    def test_fix_with_import_block(self, tmp_path):
        """Fix that adds a new import line."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        new_load = (
            '    def load(self):\n'
            '        """Load records using pathlib."""\n'
            '        content = Path(self.source_path).read_text()\n'
            '        self.records = [l.strip() for l in content.splitlines()]\n'
            '        return self.records'
        )
        fix = _make_fix(
            "s1", "def load", new_load,
            line=16, end_line=21,
            import_block_code="from pathlib import Path",
            end_import_block_code=2,
        )

        success, results = _run(target, [fix])

        assert success is True
        content = target.read_text()
        assert "from pathlib import Path" in content
        assert "Path(self.source_path).read_text()" in content
        assert "import os" in content
        assert "import sys" in content

    def test_rejected_fix_does_not_add_import(self, tmp_path):
        """If code change fails, import must also not appear."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)
        original = target.read_text()

        broken_load = (
            '    def load(self):\n'
            '        content = Path(self.source_path).read_text(\n'  # unclosed
        )
        fix = _make_fix(
            "s1", "def load", broken_load,
            line=16, end_line=21,
            import_block_code="from pathlib import Path",
            end_import_block_code=2,
        )

        success, results = _run(target, [fix])

        assert success is False
        assert target.read_text() == original

    def test_fix_with_helper_code_global_bottom(self, tmp_path):
        """Fix that adds a helper function at the bottom."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        new_transform = (
            '    def transform(self, record):\n'
            '        """Transform using external sanitizer."""\n'
            '        sanitized = sanitize_record(record)\n'
            '        if sanitized in self._cache:\n'
            '            return self._cache[sanitized]\n'
            '        result = sanitized.upper()\n'
            '        self._cache[sanitized] = result\n'
            '        return result'
        )
        fix = _make_fix(
            "s1", "def transform", new_transform,
            line=23, end_line=29,
            helper_code='def sanitize_record(value):\n    return value.strip().replace("\\t", " ")',
            placement_helper="GLOBAL_BOTTOM",
        )

        success, results = _run(target, [fix])

        assert success is True
        content = target.read_text()
        assert "sanitize_record(record)" in content
        assert "def sanitize_record(value):" in content

    def test_rejected_fix_does_not_add_helper(self, tmp_path):
        """If code change breaks syntax, helper must not appear either."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)
        original = target.read_text()

        broken = (
            '    def transform(self, record):\n'
            '        sanitized = sanitize_record(record\n'  # unclosed
        )
        fix = _make_fix(
            "s1", "def transform", broken,
            line=23, end_line=29,
            helper_code='def sanitize_record(value):\n    return value.strip()',
            placement_helper="GLOBAL_BOTTOM",
        )

        success, results = _run(target, [fix])

        assert success is False
        assert target.read_text() == original

    def test_class_method_replacement_preserves_structure(self, tmp_path):
        """Replace save() and verify the whole file structure survives."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        # save is 6 lines (38-43), replacement also 6 lines
        new_save = (
            '    def save(self, output_path):\n'
            '        """Save processed records with count."""\n'
            '        processed = self.process_all()\n'
            '        with open(output_path, "w") as f:\n'
            '            f.writelines(processed)\n'
            '        return len(processed)'
        )
        fix = _make_fix("s1", "def save", new_save, line=38, end_line=43)

        success, results = _run(target, [fix])

        assert success is True
        content = target.read_text()
        assert "f.writelines(processed)" in content
        assert "return len(processed)" in content
        assert "class DataProcessor:" in content
        assert "def __init__" in content
        assert "def load" in content
        assert "def transform" in content
        assert "def process_all" in content
        assert "def helper_function" in content
        assert "MAX_RETRIES = 3" in content

    def test_fixes_across_class_and_module_level(self, tmp_path):
        """Fix a class method AND a module-level function."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        new_init = (
            '    def __init__(self, source_path, max_records=None):\n'
            '        self.source_path = source_path\n'
            '        self.max_records = max_records\n'
            '        self.records = []\n'
            '        self._cache = {}'
        )
        new_helper = (
            'def helper_function(value, multiplier=2):\n'
            '    """A standalone helper with configurable multiplier."""\n'
            '    return value * multiplier'
        )

        # __init__=11-14, helper_function=46-48
        fix1 = _make_fix("s1", "def __init__", new_init, line=11, end_line=14)
        fix2 = _make_fix("s2", "def helper_function", new_helper, line=46, end_line=48)

        success, results = _run(target, [fix1, fix2])

        assert success is True
        content = target.read_text()
        assert "max_records=None" in content
        assert "multiplier=2" in content

    def test_five_fixes_scattered_some_invalid(self, tmp_path):
        """Five fixes, some valid some invalid. Only valid ones survive.

        All replacements preserve the original line count so sequential
        application with absolute line numbers stays correct.
        """
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        # line 5: MAX_RETRIES (1 line → 1 line)
        fix1 = _make_fix("s1", "MAX_RETRIES = 3", "MAX_RETRIES = 5", line=5)
        # lines 11-14: __init__ (4 lines → 4 lines)
        fix2 = _make_fix(
            "s2", "def __init__",
            '    def __init__(self, source_path, timeout=30):\n'
            '        self.source_path = source_path\n'
            '        self.timeout = timeout\n'
            '        self._cache = {}',
            line=11, end_line=14,
        )
        # lines 16-21: load — BROKEN (6 lines → 2 lines, unclosed bracket)
        fix3 = _make_fix(
            "s3", "def load",
            '    def load(self):\n'
            '        self.records = [\n',
            line=16, end_line=21,
        )
        # lines 31-36: process_all (6 lines → 6 lines)
        fix4 = _make_fix(
            "s4", "def process_all",
            '    def process_all(self):\n'
            '        """Process using list comprehension."""\n'
            '        return [self.transform(r) for r in self.records]\n'
            '\n'
            '\n'
            '\n',
            line=31, end_line=36,
        )
        # lines 46-48: helper_function (3 lines → 3 lines)
        fix5 = _make_fix(
            "s5", "def helper_function",
            'def helper_function(value, power=2):\n'
            '    """A standalone helper with power."""\n'
            '    return value ** power',
            line=46, end_line=48,
        )

        success, results = _run(target, [fix1, fix2, fix3, fix4, fix5])

        assert success is False
        content = target.read_text()
        assert "MAX_RETRIES = 5" in content
        assert "timeout=30" in content
        assert "self.records = [\n" not in content, "broken fix3 must not leak"
        assert "list comprehension" in content
        assert "value ** power" in content

    def test_first_fix_invalid_rest_still_work(self, tmp_path):
        """First fix fails — subsequent fixes still get a clean base."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        broken_first = _make_fix("s1", "MAX_RETRIES = 3", "MAX_RETRIES = !!!", line=5)
        valid_second = _make_fix(
            "s2", "def helper_function",
            'def helper_function(value):\n'
            '    """Tripled helper."""\n'
            '    return value * 3',
            line=46, end_line=48,
        )

        success, results = _run(target, [broken_first, valid_second])

        assert success is False
        content = target.read_text()
        assert "MAX_RETRIES = !!!" not in content
        assert "MAX_RETRIES = 3" in content
        assert "return value * 3" in content

    def test_last_fix_invalid_earlier_ones_survive(self, tmp_path):
        """Last fix fails — all earlier valid fixes still appear."""
        target = tmp_path / "target.py"
        target.write_text(REALISTIC_FILE)

        valid = _make_fix("s1", "MAX_RETRIES = 3", "MAX_RETRIES = 10", line=5)
        broken_last = _make_fix(
            "s2", "def helper_function",
            'def helper_function(value):\n'
            '    return value *\n',  # dangling operator
            line=46, end_line=48,
        )

        success, results = _run(target, [valid, broken_last])

        assert success is False
        content = target.read_text()
        assert "MAX_RETRIES = 10" in content
        assert "return value *\n" not in content
        assert "return value * 2" in content
