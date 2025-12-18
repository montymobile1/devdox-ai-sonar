from typing import Optional, TypedDict
from devdox_ai_sonar.models.sonar import FixSuggestion
from dataclasses import dataclass


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
            and self.end < total_lines
        )


class ImportState(TypedDict):
    """State for tracking import insertion point."""

    last_import_line: int
    last_docstring_line: int
    last_shebang_encoding_line: int
    in_docstring: bool
    docstring_quote: Optional[str]
