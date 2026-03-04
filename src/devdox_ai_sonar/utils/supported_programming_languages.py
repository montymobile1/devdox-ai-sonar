"""
Language metadata and file-filtering utilities for SonarCloud integration.

This module provides:

1. LanguageConfig -- Immutable identity metadata for a programming language.
   Contains only facts about the language itself: what SonarCloud calls it
   and what file extensions it uses. This never changes per-project.

2. Module-level language constants (``PYTHON``, ``IPYTHON``) and two lookup
   functions (``lang_from_rule_key``, ``lang_from_file_ext``) that map
   any SonarCloud identifier (rule key, repository name, file extension)
   back to a LanguageConfig.

3. ``is_file_processable`` -- a standalone function that decides whether
   a given file path should be processed based on suffix/prefix rules.

Typical usage::

    from devdox_ai_sonar.utils.supported_programming_languages import (
        PYTHON, lang_from_rule_key, is_file_processable,
    )

    # Resolve a language from a SonarCloud rule key
    lang = lang_from_rule_key("pythonsecurity:S5445")
    # lang.name == "python"

    # Check whether a file should be processed
    if is_file_processable("src/utils/helpers.py",
                           allowed_suffixes=PYTHON.file_extensions,
                           excluded_prefixes={"test_"}):
        ...  # process the file
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Set
from typing import Dict, FrozenSet, Optional


@dataclass(frozen=True)
class LanguageConfig:
    """Immutable metadata describing a programming language's identity.

    This dataclass holds only facts that are intrinsic to the language itself.
    It does NOT hold project-specific filtering rules
    (use ``is_file_processable`` for that).

    Attributes:
        name:
            The internal identifier for this language. This is NOT a
            SonarCloud concept — it is a name we control and define.

            Example: "python", "java", "javascript", "typescript"

        sonar_language_key:
            The **single** language key that SonarCloud uses in its API
            parameters.  Each language has exactly one key — SonarCloud
            does not accept alternatives or aliases.

            This is the value passed to the ``languages`` query parameter
            in ``GET /api/issues/search``.  It is distinct from the
            repository names (``sonar_repositories``), from our internal
            ``name``, and from ``file_extensions``.

            Why this is NOT derived from ``file_extensions``:
              - A language can have many extensions (Python: .py, .pyw;
                C++: .cpp, .h, .cc, .cxx, .hpp), but SonarCloud expects
                exactly one key per language.
              - The key does not always match any extension (e.g. C++ uses
                "cpp" but also owns ".h" files — "h" is NOT a valid key).

            To find the correct key for a new language, check the
            ``"language"`` field in ``all_sonarcloud_rules.json`` or call
            ``GET /api/languages/list`` on your SonarCloud instance.

            Example: "py" for Python, "java" for Java, "js" for JavaScript

        sonar_repositories:
            The complete set of SonarCloud repository names that belong to
            this language.

            This set exists because SonarCloud spreads a single language's
            rules across multiple repositories:
              - Python rules come from "python" AND "pythonsecurity".
              - Java rules come from "java", "javasecurity", and the legacy
                names "squid" and "javasquid" from older SonarQube versions.
              - C# rules come from "csharpsquid" (NOT "csharp").

            When a rule key like "pythonsecurity:S3649" arrives, the lookup
            splits on ":", extracts "pythonsecurity", and searches every
            language's sonar_repositories to find the match.

        file_extensions:
            All file extensions (with leading dot) associated with this
            language.

            Example: Python uses {".py", ".pyw"}, JavaScript uses
            {".js", ".jsx", ".mjs", ".cjs"}.
    """

    name: str
    sonar_language_key: str
    sonar_repositories: FrozenSet[str]
    file_extensions: FrozenSet[str]


# ---------------------------------------------------------------------------
# Supported language definitions
# ---------------------------------------------------------------------------
# Add new languages here as support is implemented.
#
# To find the correct sonar repository names for a new language:
#   - Check your SonarCloud instance's rules page and filter by
#     language to see which repositories appear.
#   - Or call ``GET api/rules/repositories`` on your SonarCloud
#     instance to get the full list.
#   - Do NOT guess — repository names are inconsistent across
#     languages (e.g. C# is "csharpsquid", not "csharp").

PYTHON = LanguageConfig(
    name="python",
    sonar_language_key="py",
    sonar_repositories=frozenset({"python", "pythonsecurity"}),
    file_extensions=frozenset({".py", ".pyw"}),
)

IPYTHON = LanguageConfig(
    name="ipython",
    sonar_language_key="ipynb",
    sonar_repositories=frozenset({"ipython"}),
    file_extensions=frozenset({".ipynb"}),
)

_SUPPORTED = [PYTHON, IPYTHON]

# Reverse indexes built once at import time.
_REPO_INDEX: Dict[str, LanguageConfig] = {
    repo: lang for lang in _SUPPORTED for repo in lang.sonar_repositories
}
_EXT_INDEX: Dict[str, LanguageConfig] = {
    ext: lang for lang in _SUPPORTED for ext in lang.file_extensions
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def lang_from_rule_key(rule_key: str) -> Optional[LanguageConfig]:
    """Resolve a LanguageConfig from a SonarCloud rule key or bare repository name.

    Handles both formats:
      - Full rule key: "python:S5445" -> extracts "python" -> looks up
      - Bare repo name: "pythonsecurity" -> looks up directly

    Args:
        rule_key: A SonarCloud rule key (e.g. "javasecurity:S3649")
            or bare repository name (e.g. "javasecurity").

    Returns:
        The matching LanguageConfig, or None if unrecognized.
    """
    repo = rule_key.split(":")[0] if ":" in rule_key else rule_key
    return _REPO_INDEX.get(repo)


def lang_from_file_ext(file_path: str) -> Optional[LanguageConfig]:
    """Resolve a LanguageConfig from a file path by extracting its extension.

    Args:
        file_path: Any file path (e.g. "src/utils/helpers.py").

    Returns:
        The matching LanguageConfig, or None if the extension is
        unrecognized.
    """
    ext = Path(file_path).suffix
    return _EXT_INDEX.get(ext)


# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------


def is_file_processable(
    file_path: str,
    *,
    allowed_suffixes: Optional[Set[str]] = None,
    excluded_suffixes: Optional[Set[str]] = None,
    allowed_prefixes: Optional[Set[str]] = None,
    excluded_prefixes: Optional[Set[str]] = None,
) -> bool:
    """Check whether a file should be processed based on suffix/prefix rules.

    Filter precedence:
      1. allowed_suffixes — suffix must be in set (None/empty = all allowed)
      2. excluded_suffixes — suffix must NOT be in set (None/empty = none excluded)
      3. allowed_prefixes — filename must start with one (None/empty = all allowed)
      4. excluded_prefixes — filename must NOT start with any (None/empty = none excluded)

    Conflict rule: if the same value appears in both an allowed set and its
    excluded counterpart, the excluded set wins.

    Args:
        file_path: Relative or absolute path (e.g. "src/foo/bar.py").
        allowed_suffixes:
            File extensions to include. None/empty means all extensions pass.
            Example: {".py"} means only .py files are processed.
        excluded_suffixes:
            File extensions to reject even if they pass allowed_suffixes.
            Example: {".pyc"} rejects compiled Python bytecode files.
        allowed_prefixes:
            Filename prefixes that are required. None/empty means all pass.
            Checked against the filename only (not the full path).
        excluded_prefixes:
            Filename prefixes to reject. Checked against the filename only.
            Example: {"test_"} rejects files like test_utils.py.

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
