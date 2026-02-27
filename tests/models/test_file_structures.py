

import pytest
from typing import Optional
from devdox_ai_sonar.models.file_structures import (
    FixApplication,
    LineRange,
    ImportState
)
from devdox_ai_sonar.models.sonar import (FixSuggestion,
    ChangeType,
    BlockType,
    CodeBlock)


# ============================================================================
# TEST CLASS: FixApplication
# ============================================================================

@pytest.fixture
def sample_code_block():
    return CodeBlock(block_name="test",
                     start_line="1",
                     end_line="10",
                     has_changes=True,
                     change_type=ChangeType.FULL_CODE,
                     block_type=BlockType.MODULE,
                     context="new_code"
                     )

class TestFixApplication:
    """Test FixApplication dataclass"""

    @pytest.fixture
    def sample_fix(self, sample_code_block):
        """Create sample fix suggestion"""
        return FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="old code",
            fixed_code="new code",
            explanation="Fixed it",
            confidence=0.95,
            sonar_line_number=10,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]
        )

    def test_fix_application_creation_success(self, sample_fix):
        """Test creating FixApplication with success"""
        app = FixApplication(
            fix=sample_fix,
            success=True,
            reason="Applied successfully"
        )

        assert app.fix == sample_fix
        assert app.success is True
        assert app.reason == "Applied successfully"

    def test_fix_application_creation_failure(self, sample_fix):
        """Test creating FixApplication with failure"""
        app = FixApplication(
            fix=sample_fix,
            success=False,
            reason="Syntax error in fixed code"
        )

        assert app.fix == sample_fix
        assert app.success is False
        assert app.reason == "Syntax error in fixed code"

    def test_fix_application_default_reason(self, sample_fix):
        """Test default empty reason"""
        app = FixApplication(
            fix=sample_fix,
            success=True
        )

        assert app.reason == ""

    def test_fix_application_is_dataclass(self, sample_fix):
        """Test that FixApplication is a dataclass"""
        app = FixApplication(fix=sample_fix, success=True)

        assert hasattr(app, '__dataclass_fields__')

    def test_fix_application_equality(self, sample_fix):
        """Test dataclass equality"""
        app1 = FixApplication(fix=sample_fix, success=True, reason="test")
        app2 = FixApplication(fix=sample_fix, success=True, reason="test")

        assert app1 == app2

    def test_fix_application_inequality(self, sample_fix):
        """Test dataclass inequality"""
        app1 = FixApplication(fix=sample_fix, success=True, reason="test1")
        app2 = FixApplication(fix=sample_fix, success=True, reason="test2")

        assert app1 != app2

    def test_fix_application_repr(self, sample_fix):
        """Test string representation"""
        app = FixApplication(fix=sample_fix, success=True, reason="test")

        repr_str = repr(app)
        assert "FixApplication" in repr_str
        assert "success=True" in repr_str


# ============================================================================
# TEST CLASS: LineRange
# ============================================================================

class TestLineRange:
    """Test LineRange dataclass and methods"""

    def test_line_range_creation(self):
        """Test creating LineRange"""
        line_range = LineRange(start=10, end=20)

        assert line_range.start == 10
        assert line_range.end == 20

    def test_line_range_single_line(self):
        """Test LineRange for single line"""
        line_range = LineRange(start=5, end=5)

        assert line_range.start == 5
        assert line_range.end == 5

    def test_line_range_zero_indexed(self):
        """Test LineRange is zero-indexed"""
        line_range = LineRange(start=0, end=10)

        assert line_range.start == 0

    def test_from_fix_with_valid_line_numbers(self, sample_code_block):
        """Test creating LineRange from fix with valid line numbers"""
        fix = FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="code",
            fixed_code="fixed",
            explanation="Done",
            helper_code="",
            confidence=0.9,
            sonar_line_number=10,
            line_number=10,  # 1-indexed
            last_line_number=15,  # 1-indexed,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]

        )

        line_range = LineRange.from_fix(fix)

        assert line_range is not None
        assert line_range.start == 9  # Converted to 0-indexed
        assert line_range.end == 14  # Converted to 0-indexed

    def test_from_fix_with_missing_line_number(self,sample_code_block):
        """Test from_fix returns None when line_number is missing"""
        fix = FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="code",
            fixed_code="fixed",
            explanation="Done",
            confidence=0.9,
            sonar_line_number=10,
            line_number=None,
            last_line_number=15,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]

        )

        line_range = LineRange.from_fix(fix)

        assert line_range is None

    def test_from_fix_with_missing_last_line_number(self, sample_code_block):
        """Test from_fix returns None when last_line_number is missing"""
        fix = FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="code",
            fixed_code="fixed",
            explanation="Done",
            confidence=0.9,
            sonar_line_number=10,
            line_number=10,
            last_line_number=None,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]
        )

        line_range = LineRange.from_fix(fix)

        assert line_range is None

    def test_from_fix_with_both_missing(self, sample_code_block):
        """Test from_fix returns None when both line numbers missing"""
        fix = FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="code",
            fixed_code="fixed",
            explanation="Done",
            confidence=0.9,
            sonar_line_number=10,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]
        )

        line_range = LineRange.from_fix(fix)

        assert line_range is None

    def test_is_valid_with_valid_range(self):
        """Test is_valid returns True for valid range"""
        line_range = LineRange(start=5, end=10)

        assert line_range.is_valid(total_lines=20) is True

    def test_is_valid_with_single_line(self):
        """Test is_valid for single line range"""
        line_range = LineRange(start=5, end=5)

        assert line_range.is_valid(total_lines=20) is True

    def test_is_valid_at_file_start(self):
        """Test is_valid at start of file (line 0)"""
        line_range = LineRange(start=0, end=5)

        assert line_range.is_valid(total_lines=20) is True

    def test_is_valid_at_file_end(self):
        """Test is_valid at end of file"""
        line_range = LineRange(start=15, end=19)

        assert line_range.is_valid(total_lines=20) is True

    def test_is_valid_negative_start(self):
        """Test is_valid returns False for negative start"""
        line_range = LineRange(start=-1, end=10)

        assert line_range.is_valid(total_lines=20) is False

    def test_is_valid_start_after_end(self):
        """Test is_valid returns False when start > end"""
        line_range = LineRange(start=15, end=10)

        assert line_range.is_valid(total_lines=20) is False

    def test_is_valid_end_beyond_file(self):
        """Test is_valid returns False when end >= total_lines"""
        line_range = LineRange(start=10, end=21)

        assert line_range.is_valid(total_lines=20) is False

    def test_is_valid_end_equals_total_lines(self):
        """Test is_valid returns True when end equals total_lines"""
        line_range = LineRange(start=10, end=20)


        assert line_range.is_valid(total_lines=20)

    def test_is_valid_edge_case_last_valid_line(self):
        """Test is_valid for last valid line in file"""
        line_range = LineRange(start=19, end=19)

        assert line_range.is_valid(total_lines=20) is True

    def test_is_valid_empty_file(self):
        """Test is_valid with empty file"""
        line_range = LineRange(start=0, end=0)

        # Can't have valid range in empty file
        assert line_range.is_valid(total_lines=0) is False

    def test_is_valid_single_line_file(self):
        """Test is_valid with single line file"""
        line_range = LineRange(start=0, end=0)

        assert line_range.is_valid(total_lines=1) is True

    def test_line_range_dataclass_fields(self):
        """Test LineRange has expected fields"""
        line_range = LineRange(start=1, end=2)

        assert hasattr(line_range, 'start')
        assert hasattr(line_range, 'end')
        assert hasattr(line_range, '__dataclass_fields__')

    def test_line_range_equality(self):
        """Test LineRange equality"""
        range1 = LineRange(start=5, end=10)
        range2 = LineRange(start=5, end=10)

        assert range1 == range2

    def test_line_range_inequality(self):
        """Test LineRange inequality"""
        range1 = LineRange(start=5, end=10)
        range2 = LineRange(start=5, end=11)

        assert range1 != range2


# ============================================================================
# TEST CLASS: ImportState
# ============================================================================

class TestImportState:
    """Test ImportState TypedDict"""

    def test_import_state_creation(self):
        """Test creating ImportState"""
        state: ImportState = {
            "last_import_line": 10,
            "last_docstring_line": 5,
            "last_shebang_encoding_line": 0,
            "in_docstring": False,
            "docstring_quote": None
        }

        assert state["last_import_line"] == 10
        assert state["last_docstring_line"] == 5
        assert state["last_shebang_encoding_line"] == 0
        assert state["in_docstring"] is False
        assert state["docstring_quote"] is None

    def test_import_state_with_docstring(self):
        """Test ImportState with active docstring"""
        state: ImportState = {
            "last_import_line": 0,
            "last_docstring_line": 3,
            "last_shebang_encoding_line": 1,
            "in_docstring": True,
            "docstring_quote": '"""'
        }

        assert state["in_docstring"] is True
        assert state["docstring_quote"] == '"""'

    def test_import_state_single_quote_docstring(self):
        """Test ImportState with single-quote docstring"""
        state: ImportState = {
            "last_import_line": 0,
            "last_docstring_line": 0,
            "last_shebang_encoding_line": 0,
            "in_docstring": True,
            "docstring_quote": "'''"
        }

        assert state["docstring_quote"] == "'''"

    def test_import_state_all_fields_present(self):
        """Test ImportState has all required fields"""
        state: ImportState = {
            "last_import_line": 1,
            "last_docstring_line": 2,
            "last_shebang_encoding_line": 3,
            "in_docstring": False,
            "docstring_quote": None
        }

        # Verify all keys exist
        assert "last_import_line" in state
        assert "last_docstring_line" in state
        assert "last_shebang_encoding_line" in state
        assert "in_docstring" in state
        assert "docstring_quote" in state

    def test_import_state_is_dict(self):
        """Test ImportState is a dictionary"""
        state: ImportState = {
            "last_import_line": 0,
            "last_docstring_line": 0,
            "last_shebang_encoding_line": 0,
            "in_docstring": False,
            "docstring_quote": None
        }

        assert isinstance(state, dict)

    def test_import_state_mutable(self):
        """Test ImportState can be modified"""
        state: ImportState = {
            "last_import_line": 0,
            "last_docstring_line": 0,
            "last_shebang_encoding_line": 0,
            "in_docstring": False,
            "docstring_quote": None
        }

        state["last_import_line"] = 15
        state["in_docstring"] = True

        assert state["last_import_line"] == 15
        assert state["in_docstring"] is True

    def test_import_state_optional_docstring_quote(self):
        """Test docstring_quote can be None"""
        state: ImportState = {
            "last_import_line": 5,
            "last_docstring_line": 3,
            "last_shebang_encoding_line": 0,
            "in_docstring": False,
            "docstring_quote": None
        }

        assert state["docstring_quote"] is None

    def test_import_state_typical_usage(self):
        """Test typical usage scenario"""
        # Initial state
        state: ImportState = {
            "last_import_line": -1,
            "last_docstring_line": -1,
            "last_shebang_encoding_line": -1,
            "in_docstring": False,
            "docstring_quote": None
        }

        # Update as we parse file
        state["last_shebang_encoding_line"] = 0
        state["last_docstring_line"] = 3
        state["last_import_line"] = 10

        assert state["last_import_line"] > state["last_docstring_line"]
        assert state["last_docstring_line"] > state["last_shebang_encoding_line"]


# ============================================================================
# TEST CLASS: Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration between components"""

    def test_line_range_from_fix_and_validate(self, sample_code_block):
        """Test creating LineRange from fix and validating"""
        fix = FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="code",
            fixed_code="fixed",
            explanation="Done",
            confidence=0.9,
            sonar_line_number=10,
            line_number=10,
            last_line_number=15,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]
        )

        line_range = LineRange.from_fix(fix)
        assert line_range is not None

        # Validate against a file with 100 lines
        assert line_range.is_valid(total_lines=100) is True

    def test_fix_application_with_line_range(self, sample_code_block):
        """Test using FixApplication with LineRange"""
        fix = FixSuggestion(
            issue_key="issue-1",
            file_path="test.py",
            original_code="code",
            fixed_code="fixed",
            explanation="Done",
            confidence=0.9,
            sonar_line_number=10,
            line_number=10,
            last_line_number=15,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]
        )

        line_range = LineRange.from_fix(fix)
        assert line_range is not None

        # Create application result
        if line_range.is_valid(total_lines=100):
            app = FixApplication(
                fix=fix,
                success=True,
                reason="Valid line range"
            )
        else:
            app = FixApplication(
                fix=fix,
                success=False,
                reason="Invalid line range"
            )

        assert app.success is True

    def test_workflow_fix_to_application(self, sample_code_block):
        """Test complete workflow from fix to application"""
        # Create fix
        fix = FixSuggestion(
            issue_key="test-issue",
            file_path="src/main.py",
            original_code="old_code",
            fixed_code="new_code",
            explanation="Improved",
            confidence=0.95,
            sonar_line_number=25,
            line_number=25,
            last_line_number=30,
            llm_model="openapi",
            fixed_code_blocks=[sample_code_block]
        )

        # Extract line range
        line_range = LineRange.from_fix(fix)
        assert line_range is not None
        assert line_range.start == 24
        assert line_range.end == 29

        # Validate against file
        file_lines = 50
        is_valid = line_range.is_valid(total_lines=file_lines)
        assert is_valid is True

        # Create application result
        app = FixApplication(
            fix=fix,
            success=is_valid,
            reason="Successfully applied" if is_valid else "Invalid range"
        )

        assert app.success is True
        assert "Successfully applied" in app.reason


