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
