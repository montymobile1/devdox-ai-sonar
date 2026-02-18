from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def is_file_processable(
    file_path: str,
    allowed_suffixes: Optional[List[str]] = None,
    excluded_suffixes: Optional[List[str]] = None,
    allowed_prefixes: Optional[List[str]] = None,
    excluded_prefixes: Optional[List[str]] = None,
) -> bool:
    """Check whether a file should be processed based on suffix and name prefix.

    Evaluates four filter lists against the file's extension and filename.
    Works identically for absolute and relative paths, since only the filename
    component is checked for prefix rules.

    Filter precedence (evaluated in order, first failure rejects):
      1. allowed_suffixes — suffix must be in list (None/empty = all allowed)
      2. excluded_suffixes — suffix must NOT be in list (None/empty = none excluded)
      3. allowed_prefixes — filename must start with one (None/empty = all allowed)
      4. excluded_prefixes — filename must NOT start with any (None/empty = none excluded)

    Conflict resolution: if the same value appears in both an allowed list
    and its excluded counterpart (e.g. ".py" in both allowed_suffixes and
    excluded_suffixes, or "test_" in both allowed_prefixes and
    excluded_prefixes), the excluded list wins.

    Args:
        file_path: Relative or absolute file path (e.g. "src/foo/bar.py").
        allowed_suffixes: Extensions that are allowed (e.g. [".py"]).
        excluded_suffixes: Extensions to reject (e.g. [".pyc"]).
        allowed_prefixes: Filename prefixes that are required (e.g. []).
        excluded_prefixes: Filename prefixes to reject (e.g. ["test_"]).

    Returns:
        True if the file passes all checks, False otherwise.
    """
    path = Path(file_path)
    suffix = path.suffix
    filename = path.name

    if allowed_suffixes and suffix not in allowed_suffixes:
        return False

    if excluded_suffixes and suffix in excluded_suffixes:
        return False

    if allowed_prefixes and not any(filename.startswith(p) for p in allowed_prefixes):
        return False

    if excluded_prefixes and any(filename.startswith(p) for p in excluded_prefixes):
        return False

    return True
