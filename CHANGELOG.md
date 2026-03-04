# Changelog

All notable changes to **devdox-ai-sonar** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---
## [0.0.5] - 2026-03-03

### CI

- Switch workflow trigger from automatic pull_request to manual workflow_dispatch only, removing automatic runs on every PR (#XX) ([#54](https://github.com/montymobile1/devdox-ai-sonar/pull/54))
- Add manual inputs pr_number and source_branch to allow targeted fixes on any PR from the Actions UI  ([#54](https://github.com/montymobile1/devdox-ai-sonar/pull/54))
- Remove job-level if: github.event_name == 'pull_request' condition, no longer needed with single manual trigger([#54](https://github.com/montymobile1/devdox-ai-sonar/pull/54))


## [0.0.4] - 2026-02-27

### CI

- Switch workflow trigger to manual dispatch only, removing automatic pull request runs ([#52](https://github.com/montymobile1/devdox-ai-sonar/pull/52))
- Commit and push sonar fixes directly to a new branch instead of uploading artifacts ([#52](https://github.com/montymobile1/devdox-ai-sonar/pull/52))
- Fix project path to use `/github/workspace` for correct Docker container resolution ([#52](https://github.com/montymobile1/devdox-ai-sonar/pull/52))
- Declare missing inputs in `action.yml` (`pull-number`, `llm-models`, `max-fixes`, `exclude-rules`) ([#52](https://github.com/montymobile1/devdox-ai-sonar/pull/52))

## [0.0.3] -2026-02-27

### Feat

- Automated string literal deduplication for `python:S1192`, AST-based with zero LLM calls ([#46](https://github.com/montymobile1/devdox-ai-sonar/pull/46))
- Server-side language filtering for issues and hotspots ([#44](https://github.com/montymobile1/devdox-ai-sonar/pull/44))
- Robust temporary directory lifecycle management with orphan cleanup ([#36](https://github.com/montymobile1/devdox-ai-sonar/pull/36))
- Automated unused parameter removal for `python:S1172` ([#40](https://github.com/montymobile1/devdox-ai-sonar/pull/40))
- Automated non-snake-case function renaming for `python:S1542` ([#40](https://github.com/montymobile1/devdox-ai-sonar/pull/40))
- Automated parameter naming convention enforcement for `python:S117` ([#40](https://github.com/montymobile1/devdox-ai-sonar/pull/40))

### Refactor

- Enforce enum-based placement types instead of hardcoded strings ([#46](https://github.com/montymobile1/devdox-ai-sonar/pull/46))
- Reduced cognitive complexity in fix validator by splitting into focused methods ([#50](https://github.com/montymobile1/devdox-ai-sonar/pull/50))

### Docs

- Added co-author and replaced Mermaid diagrams with PyPI-compatible text alternatives ([#49](https://github.com/montymobile1/devdox-ai-sonar/pull/49))

### CI

- Updated GitHub Actions workflow for testing and SonarCloud analysis ([#47](https://github.com/montymobile1/devdox-ai-sonar/pull/47))

---

## [0.0.2] - 2026-02-23

### CI

- Added GitHub Actions pipeline for automated testing and SonarCloud analysis ([#43](https://github.com/montymobile1/devdox-ai-sonar/pull/43))

---

## [0.0.1] - 2026-02-20

### Feat

- SonarCloud REST API integration for issues, hotspots, rules, and project config ([#1](https://github.com/montymobile1/devdox-ai-sonar/pull/1))
- LLM-powered fix generation with multi-provider support — OpenAI, Google Gemini, Together AI, and OpenRouter ([#1](https://github.com/montymobile1/devdox-ai-sonar/pull/1), [#6](https://github.com/montymobile1/devdox-ai-sonar/pull/6), [#16](https://github.com/montymobile1/devdox-ai-sonar/pull/16), [#31](https://github.com/montymobile1/devdox-ai-sonar/pull/31))
- Fix validation pipeline with auto-formatting and AI fallback ([#2](https://github.com/montymobile1/devdox-ai-sonar/pull/2))
- Security hotspot analysis and fix generation ([#3](https://github.com/montymobile1/devdox-ai-sonar/pull/3))
- Rule-based issue filtering and grouping ([#10](https://github.com/montymobile1/devdox-ai-sonar/pull/10))
- Configurable rule exclusion ([#14](https://github.com/montymobile1/devdox-ai-sonar/pull/14), [#20](https://github.com/montymobile1/devdox-ai-sonar/pull/20))
- PR/MR-based and branch-based analysis modes ([#13](https://github.com/montymobile1/devdox-ai-sonar/pull/13), [#21](https://github.com/montymobile1/devdox-ai-sonar/pull/21))
- Automated `python:S7503` async-to-sync conversion handler ([#23](https://github.com/montymobile1/devdox-ai-sonar/pull/23), [#24](https://github.com/montymobile1/devdox-ai-sonar/pull/24), [#25](https://github.com/montymobile1/devdox-ai-sonar/pull/25))
- LLM-based `python:S3776` cognitive complexity reduction handler ([#23](https://github.com/montymobile1/devdox-ai-sonar/pull/23))
- Line number mapping from temporary files to source ([#17](https://github.com/montymobile1/devdox-ai-sonar/pull/17))
- Markdown export for fix explanations ([#11](https://github.com/montymobile1/devdox-ai-sonar/pull/11), [#12](https://github.com/montymobile1/devdox-ai-sonar/pull/12))
- Persistent provider configuration with API key and model management ([#16](https://github.com/montymobile1/devdox-ai-sonar/pull/16))
- File processing gatekeeper with suffix/prefix filtering ([#38](https://github.com/montymobile1/devdox-ai-sonar/pull/38))
- Interactive CLI with styled tables, progress indicators, and prompts ([#6](https://github.com/montymobile1/devdox-ai-sonar/pull/6), [#7](https://github.com/montymobile1/devdox-ai-sonar/pull/7), [#8](https://github.com/montymobile1/devdox-ai-sonar/pull/8))
- Comprehensive test suite with pytest fixtures and mocking
- Project README ([#39](https://github.com/montymobile1/devdox-ai-sonar/pull/39), [#15](https://github.com/montymobile1/devdox-ai-sonar/pull/15))

### Refactor

- Migrated all file I/O to async ([#27](https://github.com/montymobile1/devdox-ai-sonar/pull/27), [#29](https://github.com/montymobile1/devdox-ai-sonar/pull/29), [#30](https://github.com/montymobile1/devdox-ai-sonar/pull/30))
- Standardized all parameters to snake_case ([#37](https://github.com/montymobile1/devdox-ai-sonar/pull/37))
- Replaced ambiguous empty strings with explicit empty sentinel ([#28](https://github.com/montymobile1/devdox-ai-sonar/pull/28))

### Fix

- Context dictionary key access in user prompt template ([#35](https://github.com/montymobile1/devdox-ai-sonar/pull/35))
- Empty/whitespace EXPLANATION guard in fix preview ([#32](https://github.com/montymobile1/devdox-ai-sonar/pull/32))
- Various bug fixes and stability improvements ([#18](https://github.com/montymobile1/devdox-ai-sonar/pull/18), [#19](https://github.com/montymobile1/devdox-ai-sonar/pull/19), [#42](https://github.com/montymobile1/devdox-ai-sonar/pull/42))

---

[0.0.3]: https://github.com/montymobile1/devdox-ai-sonar/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/montymobile1/devdox-ai-sonar/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/montymobile1/devdox-ai-sonar/releases/tag/0.0.1
