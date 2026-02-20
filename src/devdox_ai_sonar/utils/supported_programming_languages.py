"""
Language Registry — Central source of truth for programming language metadata.

This module provides three distinct layers:

1. LanguageConfig -- Immutable identity metadata for a programming language.
   Contains only facts about the language itself: what SonarCloud calls it
   and what file extensions it uses. This never changes per-project.

2. FileFilter -- Project-specific rules for which files to process.
   Controls suffix/prefix inclusion and exclusion. Different projects can
   define different filters even for the same language.

3. LanguageRegistry -- Lookup service that maps any SonarCloud identifier
   (rule key, repository name, file extension) back to a LanguageConfig.
   Ships with built-in defaults for supported languages but accepts overrides
   for testing.

Typical usage::

    # Production — uses built-in defaults
    registry = LanguageRegistry()

    # Resolve a language from a SonarCloud rule key
    lang = registry.from_sonar_rule_key("pythonsecurity:S5445")
    # lang.name == "python"

    # Build a project-specific file filter seeded from one or more languages
    file_filter = FileFilter.for_languages(
        [lang],
        excluded_prefixes={"test_"},
    )
    if file_filter.is_processable("src/utils/helpers.py"):
        ...  # process the file

    # No-filter default — pass nothing to allow all files
    permissive = FileFilter.for_languages()
    # permissive.is_processable("anything.rs") == True

    # Testing — pass custom languages
    test_registry = LanguageRegistry(languages={
        "python": LanguageConfig(
            name="python",
            sonar_language_key="py",
            sonar_repositories=frozenset({"python", "pythonsecurity"}),
            file_extensions=frozenset({".py"}),
        ),
    })
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set


@dataclass(frozen=True)
class LanguageConfig:
    """Immutable metadata describing a programming language's identity.

    This dataclass holds only facts that are intrinsic to the language itself.
    It does NOT hold project-specific filtering rules (use FileFilter for that).

    Attributes:
        name:
            The internal identifier for this language. This is NOT a
            SonarCloud concept — it is a name we control and define.

            This value MUST match the dictionary key used when registering
            the language in LanguageRegistry, and MUST match the corresponding
            constant on LanguageRegistry (e.g. LanguageRegistry.PYTHON).

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

            When a rule key like "pythonsecurity:S3649" arrives, the registry
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


@dataclass
class FileFilter:
    """Project-specific rules for deciding which files to process.

    This is intentionally decoupled from LanguageConfig because filtering
    rules vary by project, not by language. For example:

      - Project A processes only ".py" and skips "test_" prefixed files.
      - Project B processes both ".py" and ".html" with no prefix exclusions.
      - Project C wants ".py" but not ".pyw".

    Filter precedence:
      1. allowed_suffixes — suffix must be in set (empty = all allowed)
      2. excluded_suffixes — suffix must NOT be in set (empty = none excluded)
      3. allowed_prefixes — filename must start with one (empty = all allowed)
      4. excluded_prefixes — filename must NOT start with any (empty = none excluded)

    Conflict rule: if the same value appears in both an allowed set and its
    excluded counterpart, the excluded set wins.

    Attributes:
        allowed_suffixes:
            File extensions to include. Empty means all extensions pass.
            Example: {".py"} means only .py files are processed.

        excluded_suffixes:
            File extensions to reject even if they pass allowed_suffixes.
            Example: {".pyc"} rejects compiled Python bytecode files.

        allowed_prefixes:
            Filename prefixes that are required. Empty means all pass.
            Checked against the filename only (not the full path).

        excluded_prefixes:
            Filename prefixes to reject. Checked against the filename only.
            Example: {"test_"} rejects files like test_utils.py.
    """

    allowed_suffixes: Set[str]
    excluded_suffixes: Set[str]
    allowed_prefixes: Set[str]
    excluded_prefixes: Set[str]

    @classmethod
    def for_languages(
        cls,
        languages: Optional[List[LanguageConfig]] = None,
        *,
        allowed_suffixes: Optional[Set[str]] = None,
        excluded_suffixes: Optional[Set[str]] = None,
        allowed_prefixes: Optional[Set[str]] = None,
        excluded_prefixes: Optional[Set[str]] = None,
    ) -> "FileFilter":
        """Create a filter, optionally seeded from one or more languages.

        When ``languages`` is provided and ``allowed_suffixes`` is None, all
        file extensions from every language in the list are merged into
        ``allowed_suffixes``.  When ``languages`` is None (or empty) and
        ``allowed_suffixes`` is also None, the resulting filter allows all
        file extensions (equivalent to the old ``allow_all()``).

        If ``allowed_suffixes`` is explicitly passed, it takes precedence
        over any language-derived extensions.

        Examples::

            # Seed from a single language — allows .py and .pyw
            filt = FileFilter.for_languages([python_config])

            # Seed from multiple languages — merges all extensions
            filt = FileFilter.for_languages([python_config, java_config])

            # Override language extensions — only .py, not .pyw
            filt = FileFilter.for_languages(
                [python_config],
                allowed_suffixes={".py"},
            )

            # No languages, no suffixes — allow everything
            filt = FileFilter.for_languages()

        Args:
            languages: Optional list of LanguageConfigs whose extensions
                seed the allowed_suffixes. None or empty = no extension
                restriction.
            allowed_suffixes: Override which extensions to allow. If None,
                defaults to the union of all languages' file_extensions
                (or empty if no languages given).
            excluded_suffixes: Extensions to reject. Defaults to empty.
            allowed_prefixes: Required prefixes. Defaults to empty (all pass).
            excluded_prefixes: Prefixes to reject. Defaults to empty.

        Returns:
            A new FileFilter instance.
        """
        if allowed_suffixes is not None:
            resolved_suffixes = allowed_suffixes
        elif languages:
            resolved_suffixes = set()
            for lang in languages:
                resolved_suffixes.update(lang.file_extensions)
        else:
            resolved_suffixes = set()

        return cls(
            allowed_suffixes=resolved_suffixes,
            excluded_suffixes=excluded_suffixes or set(),
            allowed_prefixes=allowed_prefixes or set(),
            excluded_prefixes=excluded_prefixes or set(),
        )

    def is_processable(self, file_path: str) -> bool:
        """Check whether a file should be processed based on all filter rules.

        Args:
            file_path: Relative or absolute path (e.g. "src/foo/bar.py").

        Returns:
            True if the file passes all checks, False otherwise.
        """
        path = Path(file_path)
        suffix = path.suffix
        filename = path.name

        if self.allowed_suffixes and suffix not in self.allowed_suffixes:
            return False

        if self.excluded_suffixes and suffix in self.excluded_suffixes:
            return False

        if self.allowed_prefixes and not any(
            filename.startswith(p) for p in self.allowed_prefixes
        ):
            return False

        if self.excluded_prefixes and any(
            filename.startswith(p) for p in self.excluded_prefixes
        ):
            return False

        return True


class LanguageRegistry:
    """Lookup service that resolves SonarCloud identifiers to LanguageConfig.

    Ships with built-in defaults for languages that we need to be supported. Pass a
    custom ``languages`` dict to override (useful for testing).

    On construction, builds two reverse indexes for a faster non-messy lookup:
      - _repo_index: maps every sonar repository name -> language name
      - _ext_index:  maps every file extension -> language name

    Raises ValueError at construction time if:
      - Two languages claim the same repository name or file extension.
      - A dict key does not match its LanguageConfig.name.

    Usage::

        registry = LanguageRegistry()
        lang = registry.from_sonar_rule_key("pythonsecurity:S5445")
        lang = registry.get(LanguageRegistry.PYTHON)
    """

    # ---- Language name constants ------------------------------------------
    # Use these instead of bare strings to reference languages elsewhere in
    # the codebase. Prevents typos and enables IDE autocomplete.
    PYTHON = "python"

    def __init__(
        self,
        languages: Optional[Dict[str, LanguageConfig]] = None,
    ):
        """Initialize the registry and build reverse indexes.

        Args:
            languages: Optional override for the language definitions.
                Keys must match each LanguageConfig.name. If None, uses
                the built-in defaults from _defaults().
        """
        self._languages = languages if languages is not None else self._defaults()

        # Reverse index: sonar repository name -> language name
        self._repo_index: Dict[str, str] = {}
        # Reverse index: file extension -> language name
        self._ext_index: Dict[str, str] = {}

        for key, lang in self._languages.items():
            if key != lang.name:
                raise ValueError(
                    f"Dict key '{key}' does not match LanguageConfig.name '{lang.name}'"
                )

            for repo in lang.sonar_repositories:
                if repo in self._repo_index:
                    raise ValueError(
                        f"Duplicate sonar repository '{repo}' claimed by "
                        f"both '{lang.name}' and '{self._repo_index[repo]}'"
                    )
                self._repo_index[repo] = lang.name

            for ext in lang.file_extensions:
                if ext in self._ext_index:
                    raise ValueError(
                        f"Duplicate file extension '{ext}' claimed by "
                        f"both '{lang.name}' and '{self._ext_index[ext]}'"
                    )
                self._ext_index[ext] = lang.name

    @staticmethod
    def _defaults() -> Dict[str, LanguageConfig]:
        """Built-in language definitions for all supported languages.

        Add new languages here as support is implemented. Each entry's
        dict key MUST match the LanguageConfig.name field.

        To find the correct sonar repository names for a new language:
          - Check your SonarCloud instance's rules page and filter by
            language to see which repositories appear.
          - Or call ``GET api/rules/repositories`` on your SonarCloud
            instance to get the full list.
          - Do NOT guess — repository names are inconsistent across
            languages (e.g. C# is "csharpsquid", not "csharp").
        """
        return {
            LanguageRegistry.PYTHON: LanguageConfig(
                name=LanguageRegistry.PYTHON,
                sonar_language_key="py",
                sonar_repositories=frozenset({"python", "pythonsecurity"}),
                file_extensions=frozenset({".py", ".pyw"}),
            ),
        }

    # ---- Lookup methods ---------------------------------------------------
    # Use these when you know what kind of identifier you have.

    def from_sonar_rule_key(self, rule_key: str) -> Optional[LanguageConfig]:
        """Resolve from a SonarCloud rule key or bare repository name.

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
        name = self._repo_index.get(repo)
        return self._languages.get(name) if name else None

    def from_file_extension(self, file_path: str) -> Optional[LanguageConfig]:
        """Resolve from a file path by extracting its extension.

        Args:
            file_path: Any file path (e.g. "src/utils/helpers.py").

        Returns:
            The matching LanguageConfig, or None if the extension is
            unrecognized.
        """
        ext = Path(file_path).suffix
        name = self._ext_index.get(ext)
        return self._languages.get(name) if name else None

    def get(self, name: str) -> Optional[LanguageConfig]:
        """Retrieve a language by its name directly.

        Use the constants for safety::

            lang = registry.get(LanguageRegistry.PYTHON)

        Args:
            name: The language name (e.g. "python").

        Returns:
            The matching LanguageConfig, or None.
        """
        return self._languages.get(name)
