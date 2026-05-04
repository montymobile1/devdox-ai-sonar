from typing import Optional, TypedDict, List, Dict, Any
from pathlib import Path
from enum import Enum
from devdox_ai_sonar.models.sonar import FixSuggestion
from dataclasses import dataclass


@dataclass
class FixContext:
    """Value object containing all context needed for fix generation."""

    file_path: Path
    file_path_tmp: Path
    line_range: Dict[str, Any]
    code_content: str
    language: str
    import_section: Dict[str, Any]
    class_name: Optional[str]
    functions: List[Dict[str, Any]]
    context_dict: Dict[str, Any]


class ConversionRisk(Enum):
    SAFE = "safe"
    NEEDS_CHANGES = "needs_changes"
    BREAKING = "breaking"
    IMPOSSIBLE = "impossible"


@dataclass
class ConversionAnalysis:
    function_name: str
    current_type: str  # 'sync' or 'async'
    target_type: str
    risk_level: ConversionRisk
    blocking_issues: List[str]
    required_changes: List[str]
    caller_impact: List[Dict]
    internal_calls: List[Dict]
    suggestions: List[str]


@dataclass
class FixApplication:
    """Result of applying a single fix."""

    fix: FixSuggestion
    success: bool
    reason: str = ""


@dataclass
class LineRange:
    """Represents a zero-indexed line range in a file."""

    start: int
    end: int

    @classmethod
    def from_fix(cls, fix: FixSuggestion) -> Optional["LineRange"]:
        """Create LineRange from fix, return None if invalid."""
        if not fix.line_number or not fix.last_line_number:
            return None
        return cls(start=fix.line_number - 1, end=fix.last_line_number - 1)

    def is_valid(self, total_lines: int) -> bool:
        """Check if range is valid for given file size."""
        return (
            0 <= self.start <= self.end
            and (self.start == self.end or self.end > 0)
            and self.end <= total_lines
            and total_lines > 0
        )


class ImportState(TypedDict):
    """State for tracking import insertion point."""

    last_import_line: int
    last_docstring_line: int
    last_shebang_encoding_line: int
    in_docstring: bool
    docstring_quote: Optional[str]


# ---------------------------------------------------------------------------
# Slot arithmetic helpers
#
# The codebase represents an editable region as an inclusive 1-indexed
# line range ``[start_line .. end_line]``. The slot width is therefore
# ``end_line - start_line + 1``, and the end_line for a slot of width N
# starting at start_line is ``start_line + N - 1``.
#
# Both rules used to be re-implemented inline in two places — a rule
# handler computing ``end_line = start_line + N`` (off by one) and the
# apply-loop computing ``expected = end_idx - start_idx + 2`` (also off
# by one in the same direction). The two errors cancelled each other
# until one was fixed, at which point the warning surfaced. Centralising
# the arithmetic here makes the off-by-one structurally impossible: any
# new caller computing slot widths inline gets caught by code review.
# ---------------------------------------------------------------------------


def slot_width(start_line: int, end_line: int) -> int:
    """Return the number of lines covered by an inclusive slot.

    ``[start_line, end_line]`` is an inclusive 1-indexed range, so the
    slot covers ``end_line - start_line + 1`` lines. A slot where
    ``start_line == end_line`` has width 1.
    """
    return end_line - start_line + 1


def slot_end_for(start_line: int, n_lines: int) -> int:
    """Return the inclusive end_line for a slot of width ``n_lines``.

    A slot starting at ``start_line`` and spanning ``n_lines`` ends at
    ``start_line + n_lines - 1`` (inclusive). For example, a 3-line
    block starting at line 13 ends at line 15.
    """
    return start_line + n_lines - 1
