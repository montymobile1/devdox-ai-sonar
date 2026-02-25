# Changelog

All notable changes to **devdox-ai-sonar** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.3] - Unreleased

### 📦 Added

- **Automated String Literal Deduplication — `python:S1192`**
  ([#46](https://github.com/montymobile1/devdox-ai-sonar/pull/46)) —
  New `StringLiteralDuplicateHandler` that automatically resolves SonarCloud rule
  `python:S1192` ("String literals should not be duplicated"). This handler is
  **purely AST-based** and requires zero LLM calls, making it fast and deterministic.
  It parses the source file into a Python AST, walks every `ast.Constant` node to
  locate all occurrences of the flagged string literal, extracts the duplicated
  literal from the SonarCloud issue message using a regex against the standard format,
  generates sequential constant names (`STRING_LITERAL_1`, `STRING_LITERAL_2`, …),
  and uses AST column offsets to preserve the original quoting style (single vs.
  double quotes). All replacements and the constant definition insertion are bundled
  into a single `SonarFixResponse` to avoid line-shift problems.

- **Language-Based Issue and Hotspot Filtering**
  ([#44](https://github.com/montymobile1/devdox-ai-sonar/pull/44)) —
  Server-side language filtering for regular issues via SonarCloud's `languages` API
  parameter. Previously, the tool fetched all issues regardless of programming
  language and discarded non-Python ones client-side, wasting API calls and making
  the `max_issues` cap misleading. Introduces three new abstractions:
  `LanguageConfig` (immutable data class capturing SonarCloud's inconsistent naming
  per language — API key, repository prefixes, file extensions, and internal name),
  `FileFilter` (project-specific file processing policy replacing the deleted
  `file_filter.py`), and `LanguageRegistry` (O(1) reverse-index lookup service
  mapping any SonarCloud identifier back to its language). For security hotspots
  (where the API has no `languages` parameter), implements a 3-pronged resolution
  strategy: (1) extract `ruleKey` from search response, (2) extract file extension
  from component path, (3) fall back to `GET /api/hotspots/show` per hotspot.

- **Robust Temporary Directory Lifecycle Management**
  ([#36](https://github.com/montymobile1/devdox-ai-sonar/pull/36)) —
  New `TmpCloneManager` async context manager guaranteeing temporary directory
  cleanup in `__aexit__` regardless of exit path (return, exception, `click.Abort`).
  Fixes at least 5 failure scenarios where `/tmp/devdox_*_test/` directories leaked
  and accumulated indefinitely. Includes `sweep_orphaned_tmp_dirs()` startup sweep
  for directories leaked by previous runs, and a gated `cleanup_tmp_py_file`
  mechanism preventing premature deletion during SonarCloud analysis.

- **Automated Unused Parameter Removal — `python:S1172`**
  ([#40](https://github.com/montymobile1/devdox-ai-sonar/pull/40)) —
  New handler that automatically removes unused function parameters from definitions
  and updates all corresponding call sites throughout the codebase.

- **Automated Non-Snake-Case Function Renaming — `python:S1542`**
  ([#40](https://github.com/montymobile1/devdox-ai-sonar/pull/40)) —
  New handler that renames functions not following the `snake_case` naming convention
  and automatically updates all call sites to use the new name.

### 🔧 Changed

- **`PlacementType` Enum Enforcement**
  ([#46](https://github.com/montymobile1/devdox-ai-sonar/pull/46)) —
  Replaced all hardcoded `PLACEMENT="SIBLING"` and `PLACEMENT="GLOBAL_TOP"` string
  literals across existing handlers with proper `PlacementType` enum values. No
  behavior change since `PlacementType` inherits from `str`, but the code is now
  explicit and type-safe.

- **GitHub Actions CI/CD Pipeline Improvements**
  ([#47](https://github.com/montymobile1/devdox-ai-sonar/pull/47)) —
  Updated and improved the GitHub Actions workflow for testing and SonarCloud
  analysis, building on the initial pipeline introduced in 0.0.2.

---

## [0.0.2] - 2026-02-23

### 📦 Added

- **GitHub Actions CI/CD Pipeline**
  ([#43](https://github.com/montymobile1/devdox-ai-sonar/pull/43)) —
  Added a GitHub Actions workflow for automated testing and SonarCloud static
  analysis. The pipeline runs on push and pull request events, executing the full
  test suite and uploading coverage results to SonarCloud for quality gate
  evaluation.

---

## [0.0.1] - 2026-02-20

### 📦 Added

- **SonarCloud Integration and Core Analysis Engine**
  ([#1](https://github.com/montymobile1/devdox-ai-sonar/pull/1)) —
  Full integration with the SonarCloud REST API capable of fetching issues, security
  hotspots, rule metadata, and project configuration. The engine parses SonarCloud
  responses into typed Python models, resolves component paths to local file paths,
  and orchestrates the end-to-end flow from issue discovery through fix generation
  to file modification.

- **LLM-Powered Fix Generation with Multi-Provider Support**
  ([#1](https://github.com/montymobile1/devdox-ai-sonar/pull/1),
  [#16](https://github.com/montymobile1/devdox-ai-sonar/pull/16),
  [#31](https://github.com/montymobile1/devdox-ai-sonar/pull/31)) —
  LLM integration layer generating code fixes using structured `SEARCH_REPLACE`
  blocks with helper code and placement directives. Prompts managed through Jinja2
  templates. Supports **Together AI** (initial provider) and **OpenRouter** (added
  via [#31](https://github.com/montymobile1/devdox-ai-sonar/pull/31), enabling
  access to a wider range of models through a single API).

- **Fix Validation Pipeline with AI Fallback**
  ([#2](https://github.com/montymobile1/devdox-ai-sonar/pull/2)) —
  Validation layer verifying LLM-generated fixes before applying them to source
  files. Uses `autopep8` for indentation correction and falls back to a secondary
  AI validation pass when structural issues are detected.

- **Security Hotspot Analysis and Fix Generation**
  ([#3](https://github.com/montymobile1/devdox-ai-sonar/pull/3)) —
  Extends the analysis engine to cover SonarCloud security hotspots in addition to
  regular code issues. Fetches hotspot details, generates targeted fixes using the
  LLM pipeline, and produces the same structured output as regular issue fixes.

- **Rule-Based Issue Filtering and Grouping**
  ([#10](https://github.com/montymobile1/devdox-ai-sonar/pull/10)) —
  Support for filtering fetched issues by specific SonarCloud rules and grouping
  them for batch processing, allowing users to focus on specific issue categories
  and process related issues together for more coherent fixes.

- **Configurable Rule Exclusion**
  ([#14](https://github.com/montymobile1/devdox-ai-sonar/pull/14),
  [#20](https://github.com/montymobile1/devdox-ai-sonar/pull/20)) —
  Users can specify SonarCloud rules to exclude from analysis, preventing fix
  generation for rules that are intentionally suppressed or handled differently
  in a given project. Configuration is persistent across runs.

- **PR/MR-Based and Branch-Based Analysis Modes**
  ([#13](https://github.com/montymobile1/devdox-ai-sonar/pull/13),
  [#21](https://github.com/montymobile1/devdox-ai-sonar/pull/21)) —
  Two analysis modes beyond full-project scanning: **Branch mode** (analyze only
  issues introduced on a specific branch) and **Pull Request mode** (analyze issues
  scoped to a specific PR/MR number using SonarCloud's pull request decoration data).
  Includes a fix for correctly resolving issues when using pull request numbers.

- **Automated Rule Handler: `python:S7503` (Async-to-Sync Conversion)**
  ([#23](https://github.com/montymobile1/devdox-ai-sonar/pull/23),
  [#24](https://github.com/montymobile1/devdox-ai-sonar/pull/24),
  [#25](https://github.com/montymobile1/devdox-ai-sonar/pull/25)) —
  First specialized rule handler — `AsyncToSyncHandler` — that fixes SonarCloud
  rule `python:S7503` without relying on the generic LLM pipeline. Removes the
  `async` keyword from functions that never use `await` and strips any `await`
  expressions on calls to those functions throughout the file.

- **Line Number Mapping from Temporary Files to Source**
  ([#17](https://github.com/montymobile1/devdox-ai-sonar/pull/17)) —
  Maps line numbers reported by SonarCloud against temporary analysis files back
  to the corresponding lines in the actual source files, ensuring generated fixes
  target the correct locations in the user's working copy.

- **Markdown Export for Fix Explanations**
  ([#11](https://github.com/montymobile1/devdox-ai-sonar/pull/11),
  [#12](https://github.com/montymobile1/devdox-ai-sonar/pull/12)) —
  Generated fixes are accompanied by human-readable Markdown explanations describing
  what was changed and why. Exportable as standalone files for documentation, code
  review comments, or compliance records.

- **File Processing Gatekeeper with Suffix/Prefix Filtering**
  ([#38](https://github.com/montymobile1/devdox-ai-sonar/pull/38)) —
  Gatekeeper layer filtering which files are eligible for processing based on
  configurable suffix and prefix rules, preventing the tool from modifying generated
  files, vendored dependencies, or other excluded paths.

- **Interactive CLI with Rich UI**
  ([#6](https://github.com/montymobile1/devdox-ai-sonar/pull/6),
  [#7](https://github.com/montymobile1/devdox-ai-sonar/pull/7),
  [#8](https://github.com/montymobile1/devdox-ai-sonar/pull/8)) —
  Command-line interface built with Click and Rich, featuring styled tables for issue
  summaries, progress indicators for long-running operations, and interactive prompts
  for fix selection.

- **Comprehensive Test Suite** —
  Unit and integration tests covering core components, CLI workflows, service layers,
  and utility functions. Tests use `pytest` with fixtures for SonarCloud API mocking,
  temporary file management, and LLM response simulation.

- **Project README**
  ([#39](https://github.com/montymobile1/devdox-ai-sonar/pull/39),
  [#15](https://github.com/montymobile1/devdox-ai-sonar/pull/15)) —
  Comprehensive README documenting installation, configuration, usage examples,
  supported rules, and overall architecture.

### 🔧 Changed

- **Asynchronous File I/O Across the Codebase**
  ([#27](https://github.com/montymobile1/devdox-ai-sonar/pull/27),
  [#29](https://github.com/montymobile1/devdox-ai-sonar/pull/29),
  [#30](https://github.com/montymobile1/devdox-ai-sonar/pull/30)) —
  Migrated all synchronous file I/O operations to async equivalents using `aiofiles`,
  preventing event loop blocking during file reads and writes.

- **Snake\_case Parameter Convention**
  ([#37](https://github.com/montymobile1/devdox-ai-sonar/pull/37)) —
  Refactored all public function parameters from mixed naming conventions to
  consistent `snake_case` following PEP 8. All call sites updated.

- **Explicit Emptiness Signaling**
  ([#28](https://github.com/montymobile1/devdox-ai-sonar/pull/28)) —
  Replaced ambiguous empty string values with an explicit sentinel to signify
  "no data," eliminating bugs where an intentionally empty string was
  indistinguishable from a missing value.

### 🐛 Fixed

- **Context Dictionary Access in User Prompt Template**
  ([#35](https://github.com/montymobile1/devdox-ai-sonar/pull/35)) —
  Fixed incorrect dictionary key access in the Jinja2 user prompt template that
  caused `KeyError` exceptions when rendering prompts for certain issue types.

- **Empty/Whitespace EXPLANATION Guard**
  ([#32](https://github.com/montymobile1/devdox-ai-sonar/pull/32)) —
  Added validation to guard against empty or whitespace-only `EXPLANATION` fields
  in fix preview and Markdown output.

- **Bug Fixes and Stability Improvements**
  ([#18](https://github.com/montymobile1/devdox-ai-sonar/pull/18),
  [#19](https://github.com/montymobile1/devdox-ai-sonar/pull/19),
  [#42](https://github.com/montymobile1/devdox-ai-sonar/pull/42)) —
  Multiple rounds of bug fixes addressing edge cases in fix application,
  indentation handling, and type checking compliance.

[0.0.3]: https://github.com/montymobile1/devdox-ai-sonar/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/montymobile1/devdox-ai-sonar/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/montymobile1/devdox-ai-sonar/releases/tag/0.0.1
