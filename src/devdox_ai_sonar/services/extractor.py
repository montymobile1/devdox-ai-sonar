from pathlib import Path
import difflib
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    ValidationResult,
)

from devdox_ai_sonar.utils.async_file_io import AsyncFileReader
logger = logging.getLogger(__name__)



class IssueExtractor:
    """Validates issue groups and extracts file information."""

    def __init__(
            self,
            file_reader: AsyncFileReader
    ):
        self.file_reader = file_reader


    async def validate_issue_group(
            self,
            issues: List[Union[SonarIssue, SonarSecurityIssue]],
            tmp_path: Path,
            project_path: Path
    ) -> ValidationResult:
        """
        Validate that issues are from the same file and extract information.

        Args:
            issues: List of issues to validate
            tmp_path: Path to temporary files
            project_path: Project root path

        Returns:
            ValidationResult with file paths, line range, and language
        """
        if not issues:
            return ValidationResult(is_valid=False, error="No issues provided")

        try:
            # Step 1: Extract and validate file paths
            file_path_tmp, line_range_tmp = _validate_and_extract_issue_info(
                issues, tmp_path
            )

            file_path, _ = _validate_and_extract_issue_info(
                issues, project_path
            )

            # Step 2: Get content range from tmp file
            line_range_result: Optional[Dict[str, Any]] =  await self.get_content_range(
                file_path_tmp, line_range_tmp, file_path
            )

            if line_range_result is None:
                return ValidationResult(
                    is_valid=False, error="Could not determine line range"
                )

            # Check for errors in line range
            if line_range_result.get("error", ""):
                return ValidationResult(
                    is_valid=False, error=line_range_result["error"]
                )

            return ValidationResult(
                is_valid=True,
                file_path=file_path,
                file_path_tmp=file_path_tmp,
                line_range=line_range_result,
            )

        except FileNotFoundError as e:
            return ValidationResult(is_valid=False, error=f"File not found: {e}")
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return ValidationResult(is_valid=False, error=f"Unexpected error: {e}")

  async def get_content_range(
            self,
            file_path_tmp: Path,
            line_range_tmp: Dict[str, Any],
            file_path: Path
    ) -> Optional[Dict[str, Any]]:

        if not file_path_tmp.exists():
            raise FileNotFoundError(f"Temporary file not found: {file_path_tmp}")
        if not file_path.exists():
            raise FileNotFoundError(f"Actual file not found: {file_path}")

            # Extract line range info
        first_line_tmp = line_range_tmp.get('first_line')
        last_line_tmp = line_range_tmp.get('last_line')
        problem_lines_tmp = line_range_tmp.get('problem_lines', [])

        if first_line_tmp is None or last_line_tmp is None or len(problem_lines_tmp) == 0:
            raise ValueError("line_range_tmp must contain 'first_line' and 'last_line'")

        min_problem_line = min(problem_lines_tmp)
        if min_problem_line <= first_line_tmp:
            first_line_tmp = min_problem_line

        # Read files
        tmp_lines = await self.file_reader.read_lines(file_path_tmp)
        actual_lines = await self.file_reader.read_lines(file_path)

        # Extract target content from temp file (1-indexed to 0-indexed)
        start_idx = first_line_tmp - 1
        end_idx = last_line_tmp  # last_line is inclusive, so we don't subtract 1 for slice

        if start_idx < 0 or end_idx > len(tmp_lines):
            raise ValueError(f"Line range {first_line_tmp}-{last_line_tmp} out of bounds for temp file")

        target_content = tmp_lines[start_idx:end_idx]

        if not target_content:
            return None

        actual_content = actual_lines[start_idx:end_idx]
        if actual_content == target_content:
            return {
                'first_line': first_line_tmp,
                'last_line': last_line_tmp,
                'problem_lines': problem_lines_tmp,
                'confidence': 1,
                'match_type': 'exact',
                'error': ""
            }

        exact_match = _find_exact_match(target_content, actual_lines)
        if exact_match:
            return _create_result(
                exact_match,
                problem_lines_tmp,
                first_line_tmp,
                confidence=1.0,
                match_type='exact'
            )

        # Strategy 3: Fuzzy matching for similar content
        fuzzy_match = _find_fuzzy_match(target_content, actual_lines, threshold=0.65)
        if fuzzy_match:
            return _create_result(
                fuzzy_match,
                problem_lines_tmp,
                first_line_tmp,
                confidence=fuzzy_match['confidence'],
                match_type='fuzzy'
            )

        # Strategy 4: If single line, try to find all occurrences and use heuristics

        if len(target_content) == 1:

            all_matches = _find_all_single_line_matches(
                target_content[0],
                actual_lines,
                first_line_tmp
            )

            if all_matches:
                best_match = all_matches[0]  # Closest to original line number

                return _create_result(
                    {'start': best_match, 'end': best_match + 1},
                    problem_lines_tmp,
                    first_line_tmp,
                    confidence=0.7,
                    match_type='fuzzy'
                )

        # Content not found
        return {
            'first_line': None,
            'last_line': None,
            'problem_lines': [],
            'confidence': 0.0,
            'match_type': 'not_found',
            'error': f"Content from lines {first_line_tmp}-{last_line_tmp} not found in actual file {file_path}"
        }


def _validate_and_extract_issue_info(
    issues: List[Union[SonarIssue, SonarSecurityIssue]], project_path: Path
) -> Tuple[Path, Dict[str, Any]]:
    """
    Validate that all issues are from the same file and extract line range.

    Args:
        issues: List of issues to validate
        project_path: Project root path

    Returns:
        Tuple of (file_path, line_range_dict)
        where line_range_dict contains:
        - first_line: Earliest line number
        - last_line: Latest line number
        - problem_lines: List of specific problem line numbers

    Raises:
        ValueError: If issues are from different files
        FileNotFoundError: If file doesn't exist
    """
    first_issue = issues[0]
    if not first_issue.file:
        raise ValueError(f"Issue {first_issue.key} has no file path")
    file_path = project_path / first_issue.file

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Initialize range with first issue
    first_line = first_issue.first_line or 0
    last_line = first_issue.last_line or 0
    problem_line_numbers: List[int] = []

    # Validate all issues are from same file and calculate range
    for issue in issues:
        if issue.file != first_issue.file:
            logger.warning(
                f"Issue {issue.key} has different file than first issue "
                f"{first_issue.key}, skipping fix generation"
            )
            raise ValueError("Issues from different files cannot be fixed together")

        # Expand range to encompass all issues
        first_line = min(first_line, issue.first_line or 0)
        last_line = max(last_line, issue.last_line or 0)
        problem_line_numbers.extend(issue.problem_lines or [])

    return file_path, {
        "first_line": first_line,
        "last_line": last_line,
        "problem_lines": problem_line_numbers,
    }



def _find_fuzzy_match(
    target_content: List[str], actual_lines: List[str], threshold: float = 0.85
) -> Optional[Dict[str, Any]]:
    """
    Find fuzzy match allowing for minor modifications.
    Returns match with confidence score.
    """
    target_len = len(target_content)
    best_match: Optional[Dict[str, Any]] = None
    best_ratio = 0.0

    for i in range(len(actual_lines) - target_len + 1):
        candidate = actual_lines[i : i + target_len]

        # Calculate similarity ratio
        ratio = difflib.SequenceMatcher(None, target_content, candidate).ratio()

        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = {"start": i, "end": i + target_len}

    if best_match:
        best_match["confidence"] = best_ratio
        return best_match

    return None



def _find_all_single_line_matches(
    target_line: str, actual_lines: List[str], original_line_num: int
) -> List[int]:
    """
    Find all occurrences of a single line.
    Returns sorted by proximity to original line number.
    """
    matches = []
    target_stripped = target_line.strip()

    for i, line in enumerate(actual_lines):
        if line.strip() == target_stripped:
            matches.append(i)

    if not matches:
        return []

    # Sort by distance from original line number (0-indexed vs 1-indexed)
    original_idx = original_line_num - 1
    matches.sort(key=lambda x: abs(x - original_idx))

    return matches


def _find_exact_match(
    target_content: List[str], actual_lines: List[str]
) -> Optional[Dict[str, int]]:
    """Find exact match of target content in actual file."""
    target_len = len(target_content)
    for i in range(len(actual_lines) - target_len + 1):
        if actual_lines[i : i + target_len] == target_content:
            return {"start": i, "end": i + target_len}

    return None


def _create_result(
    match: Dict[str, int],
    problem_lines_tmp: List[int],
    first_line_tmp: int,
    confidence: float,
    match_type: str,
) -> Dict[str, Any]:
    """
    Create standardized result dictionary with adjusted problem lines.
    """
    start_line = match["start"] + 1  # Convert to 1-indexed
    end_line = match["end"]  # Already correct for 1-indexed inclusive range

    # Adjust problem lines based on offset
    offset = start_line - first_line_tmp
    adjusted_problem_lines = [line + offset for line in problem_lines_tmp]

    # Ensure problem lines are within the found range
    adjusted_problem_lines = [
        line for line in adjusted_problem_lines if start_line <= line <= end_line
    ]

    return {
        "first_line": start_line,
        "last_line": end_line,
        "problem_lines": adjusted_problem_lines,
        "confidence": confidence,
        "match_type": match_type,
    }
