# DevDox AI Sonar

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/devdox-sonar)](https://pypi.org/project/devdox-sonar/)
[![Build](https://github.com/montymobile1/devdox-ai-sonar/actions/workflows/build.yml/badge.svg)](https://github.com/montymobile1/devdox-ai-sonar/actions/workflows/build.yml)

DevDox AI Sonar is a command-line tool that reads the analysis reports SonarCloud has already produced for your project — every bug, code smell, and security vulnerability it found — and sends each issue to a Large Language Model along with the relevant source code and context. The LLM generates a structured fix with code blocks, line numbers, and a confidence score. You review it, apply it if it looks good, and a markdown changelog documents everything.

The CLI is built on [Click](https://github.com/pallets/click) for command handling, [Questionary](https://github.com/tmbo/questionary) for interactive prompts, and [Rich](https://github.com/Textualize/rich) for terminal formatting. Issue data and fix suggestions are modeled with [Pydantic](https://github.com/pydantic/pydantic) for validation. LLM prompts are assembled from [Jinja2](https://github.com/pallets/jinja) templates, making them easy to inspect and modify. All file I/O during fix application is async via [aiofiles](https://github.com/Tinche/aiofiles).

**PyPI:** [pypi.org/project/devdox-sonar](https://pypi.org/project/devdox-sonar/)
**Source:** [github.com/montymobile1/devdox-ai-sonar](https://github.com/montymobile1/devdox-ai-sonar)

---

## Table of Contents

- [What is DevDox AI Sonar?](#what-is-devdox-ai-sonar)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [First-Time Setup](#first-time-setup)
  - [Subsequent Runs](#subsequent-runs)
- [Configuration](#configuration)
  - [Configuration Files](#configuration-files)
  - [LLM Providers](#llm-providers)
- [Using the CLI](#using-the-cli)
  - [Interactive Mode](#interactive-mode)
  - [Direct Mode](#direct-mode)
  - [CLI Options](#cli-options)
  - [The fix_issues Pipeline](#the-fix_issues-pipeline)
  - [Understanding the Output](#understanding-the-output)
  - [Workflow Recipes](#workflow-recipes)
- [Advanced Topics](#advanced-topics)
  - [Rule Exclusions](#rule-exclusions)
  - [Supported Languages](#supported-languages)
  - [Troubleshooting](#troubleshooting)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## What is DevDox AI Sonar?

**SonarCloud** is a service that scans your code on every push and produces an analysis report: bugs that will crash at runtime, security holes an attacker could exploit, and code smells that make your codebase harder to maintain. It tells you *what* is wrong. It does not tell you how to fix it.

For a project with hundreds of open issues, fixing them manually means reading each rule, understanding the context, writing a fix, and testing it. Most teams never get to it. The issues pile up.

**DevDox AI Sonar picks up where SonarCloud leaves off.** It requires that SonarCloud has already scanned your project and produced an analysis report. The tool then authenticates with SonarCloud, reads that report, fetches the flagged issues, and for each one, extracts the relevant source code with surrounding context and sends it to an LLM. The LLM returns a structured fix — actual code blocks with line numbers, import changes, helper functions, an explanation, and a confidence score. You review the fix, decide whether to apply it, and a markdown changelog records every change for audit.

You install it from PyPI and use it as a terminal command:

```bash
pip install devdox_sonar
devdox_sonar
```

<details>
<summary><strong>Glossary</strong></summary>

| Term | Meaning |
|------|---------|
| **SonarCloud** | Cloud service that scans your code and produces analysis reports. Free for open-source projects. |
| **Analysis Report** | The output SonarCloud generates after scanning your code. Contains all detected issues. DevDox AI Sonar reads this report. |
| **Issue** | A single problem SonarCloud found. Has a type, severity, rule, file, and line number. |
| **Rule** | The coding standard an issue violates. Example: `python:S1066` = "mergeable if statements should be combined." Each rule has a unique ID in the format `language:SXXXX`. |
| **Bug** | A logic error that will produce wrong results or crash. |
| **Code Smell** | Not a bug, but makes code harder to maintain — excessive complexity, duplication, dead code. |
| **Vulnerability** | A security issue — SQL injection, XSS, hardcoded credentials. |
| **Security Hotspot** | Code that *might* be a security issue and needs manual review. |
| **Severity** | How bad it is: **Blocker** > **Critical** > **Major** > **Minor** > **Info**. |
| **LLM Provider** | The AI service generating fixes. You bring your own API key. Supported: OpenAI, Google Gemini, TogetherAI, OpenRouter. |
| **Confidence Score** | 0.0 to 1.0 rating from the LLM indicating how certain it is about the fix. |
| **Dry Run** | Runs the full pipeline but skips all file writes. Safe to run anytime. |

</details>

---

## How It Works

SonarCloud must have already scanned your project before DevDox AI Sonar can do anything. The tool reads SonarCloud's existing analysis report — it does not perform its own code analysis.

```
SonarCloud          Fetch Issues       Clone Repo        Extract Code       Build Prompt
(already scanned) ──► from report ────► to /tmp ────────► + Context ────────► (Jinja2)
                                                                                 │
                                                                                 ▼
                  ┌── Apply + Changelog ◄── YES ── Preview ◄── Validate ◄── Call LLM
                  │                          │
                  │                         NO
                  │                          │
                  │                          ▼
                  │                        Skip
                  ▼
```

1. **Fetch** — Authenticates with SonarCloud and reads the analysis report via the Issues API. Filters by the types and severities you configured. Regular issues (bugs, code smells) are grouped **by rule** so all issues of the same kind are batched together. Security issues are grouped **by file**.

2. **Clone and Extract** — Your repository is cloned to a temporary directory (your working tree is never touched). For each issue, the tool reads the flagged file, locates the exact lines from the report, and extracts code with surrounding context (`context_lines=10` by default). If the code has changed since SonarCloud last scanned, fuzzy matching is used to find the right location.

3. **Build Prompt** — The extracted code, rule description, severity, and issue metadata are assembled into a prompt using Jinja2 templates (located in `prompts/python/`).

4. **Call LLM** — The prompt is sent to your configured provider. The LLM returns structured JSON containing code blocks with line numbers, an explanation, and a confidence score.

5. **Validate** — When applying fixes (not preview-only), a **validator agent** — a second LLM call — reviews the fix for logic errors, security issues, and syntax problems. If it finds issues, it can correct them.

6. **Preview and Apply** — The terminal shows the file path, confidence score, and explanation. If `apply = 1`, the fix is written to disk. If `create_backup = 1`, the entire project directory is copied to `<project>_backup_YYYYMMDD_HHMMSS` in the parent directory first. A markdown changelog documents every change.

---

## Prerequisites

**Python 3.12 or higher**

```bash
python --version
```

**Git** — The tool clones your repo to a temp directory for code extraction.

```bash
git --version
```

**A SonarCloud project that has already been scanned.** DevDox AI Sonar does not scan your code — SonarCloud does. You need an existing SonarCloud project with at least one completed analysis. If you have not set up SonarCloud yet, it is free for open-source: [sonarcloud.io](https://sonarcloud.io/).

You will need these from your SonarCloud account:
- **API Token** — generate at [sonarcloud.io/account/security](https://sonarcloud.io/account/security)
- **Organization Key** — visible in your dashboard URL: `sonarcloud.io/organizations/<org-key>/projects`
- **Project Key** — visible on your project page: `sonarcloud.io/project/overview?id=<project-key>`

**An API key for at least one LLM provider:** [OpenAI](#openai), [Google Gemini](#google-gemini), [TogetherAI](#togetherai), or [OpenRouter](#openrouter).

---

## Installation

```bash
pip install devdox_sonar
```

Verify:

```bash
devdox_sonar --version
# devdox_sonar, version 0.0.1-beta
```

For contributors installing from source:

```bash
git clone https://github.com/montymobile1/devdox-ai-sonar.git
cd devdox-ai-sonar
pip install -e ".[dev]"
```

---

## Getting Started

### First-Time Setup

Run the tool with no arguments to start the setup wizard:

```bash
devdox_sonar
```

The wizard walks you through three steps.

**Step 1 — SonarCloud Connection**

| It asks for | Where to find it | Example |
|---|---|---|
| Token | [sonarcloud.io/account/security](https://sonarcloud.io/account/security) | `squ_abc123def456...` |
| Organization Key | Your SonarCloud dashboard URL | `my-company` |
| Project Key | Your project's SonarCloud page | `my-company_my-app` |
| Project Path | Absolute path to the code on your machine | `/home/user/projects/my-app` |
| Git URL | Repository clone URL | `https://github.com/my-org/my-app.git` |

Saved to `~/devdox/auth.json`.

**Step 2 — LLM Provider**

1. Pick a provider (OpenAI, Gemini, TogetherAI, or OpenRouter)
2. Paste your API key — it is validated against the provider's API immediately
3. Choose a model from the provider's available list
4. Optionally set it as the default provider

You can add multiple providers. Saved to `~/devdox/config.toml`.

**Step 3 — Analysis Parameters**

The wizard prompts you for each of the following. You can press Enter to skip any field and accept the current value.

| Parameter | Prompt | Possible values |
|---|---|---|
| **Max Fixes** | `Maximum fixes to generate (0-20)` | Any integer from 0 to 20 |
| **Issue Types** | `Issue types (comma-separated, or press Enter to skip)` | `BUG`, `VULNERABILITY`, `CODE_SMELL` — pass one or more, comma-separated |
| **Severities** | `Issue severities (comma-separated, or press Enter to skip)` | `BLOCKER`, `CRITICAL`, `MAJOR`, `MINOR`, `INFO` — pass one or more, comma-separated |
| **Apply** | `Apply fixes of SonarQube (press Enter to skip)` | `yes` (apply fixes to files) or `no` (preview only) |
| **Create Backup** | `Create backup before apply fixes (press Enter to skip)` | `yes` (copy project dir before modifying) or `no` |
| **Exclude Rules** | `Rules to be excluded (comma-separated, or press Enter to skip)` | Comma-separated rule IDs, e.g. `python:S7503,python:S3776`. See [Rule Exclusions](#rule-exclusions) for format and recommendations. |

After setup completes, you land in the interactive menu.

### Subsequent Runs

On subsequent runs, `devdox_sonar` detects your existing configuration and skips the setup wizard entirely. It loads your saved credentials and parameters from `~/devdox/auth.json` and `~/devdox/config.toml`, and goes straight to the interactive menu.

To reconfigure, either use the menu options (Add Provider, Update Provider, Change Parameters Configuration) or edit the config files directly.

---

## Configuration

### Configuration Files

All configuration lives in `~/devdox/`:

```
~/devdox/
├── auth.json      # SonarCloud credentials (JSON)
└── config.toml    # LLM providers + analysis parameters (TOML)
```

**auth.json**

```json
{
  "SONAR_TOKEN": "squ_your_token",
  "SONAR_ORG": "your-org-key",
  "SONAR_PROJ": "your-project-key",
  "PROJECT_PATH": "/home/user/projects/my-app",
  "GIT_URL": "https://github.com/your-org/my-app.git"
}
```

**config.toml**

```toml
[llm]
default_provider = "openai"
default_model = "gpt-4o"

[[llm.providers]]
name = "openai"
api_key = "sk-your-key"
base_url = "https://api.openai.com/v1"
models = ["gpt-4o", "gpt-4-turbo"]

[configuration]
max_fixes = 5
types = "BUG,CODE_SMELL"
severities = "CRITICAL,MAJOR"
apply = 0
create_backup = 0
exclude_rules = ""
```

**Updating configuration**

Use the interactive menu (run `devdox_sonar` with no arguments) and select:
- **Add Provider** — add another LLM provider
- **Update Provider** — change an existing provider's API key or model
- **Change Parameters Configuration** — adjust types, severities, max fixes, apply, backup, and excluded rules

Or edit `~/devdox/auth.json` and `~/devdox/config.toml` directly.

---

### LLM Providers

You need at least one. The setup wizard lets you configure any of these:

#### OpenAI
- **Get a key:** [platform.openai.com](https://platform.openai.com) → API Keys → Create new secret key
- **Recommended models:** `gpt-4o`, `gpt-4-turbo`

#### Google Gemini
- **Get a key:** [ai.google.dev](https://ai.google.dev) → Google AI Studio → Get API Key
- **Recommended models:** `gemini-2.5-flash`, `gemini-pro`
- Has a **free tier** — useful for trying the tool without spending anything.

#### TogetherAI
- **Get a key:** [together.ai](https://www.together.ai) → Dashboard → Settings → API Keys
- **Recommended models:** `mixtral-8x7b`, `meta-llama/Llama-3-70b`
- Runs open-source models at lower cost.

#### OpenRouter
- **Get a key:** [openrouter.ai](https://openrouter.ai) → Dashboard → Keys → Create Key
- **Example models:** `anthropic/claude-sonnet-4`, `openai/gpt-4o`, `google/gemini-2.5-flash`
- One API key gives access to 400+ models. Model names use `provider/model-name` format.

---

## Using the CLI

DevDox AI Sonar provides two ways to run commands: an **interactive mode** with a menu, and a **direct mode** that skips the menu and runs a specific command immediately.

### Interactive Mode

```bash
devdox_sonar
```

On first run, the setup wizard runs (see [First-Time Setup](#first-time-setup)). On subsequent runs, your saved configuration is loaded and you go straight to the menu:

```
═══════════════════════════════════════════════
   DevDox AI Sonar - Interactive Mode
═══════════════════════════════════════════════

? What would you like to do?
  ➕ Add Provider - Add provider or sonar configuration
  ✏️  Update Provider - Update provider or sonar configuration
  🔧 Fix Issues - Generate and apply LLM-powered fixes
  🔒 Fix Security Issues - Specialized security vulnerability fixes
  📊 Analyze Project - Display SonarCloud analysis
  🔍 Inspect Project - Analyze local directory structure
  ⚙️  Change Parameters Configuration
  ❌ Exit
```

Use arrow keys to navigate, Enter to select. Type `/` during any prompt to switch to a different command. Press Ctrl+C to exit.

### Direct Mode

If you already know which command you want, skip the interactive menu entirely using the `-c` flag:

```bash
devdox_sonar -c <command> [OPTIONS]
```

This runs the command immediately using your saved configuration from `~/devdox/`. You can override specific parameters with CLI options (see below).

Four commands are available in direct mode:

| Command | What it does |
|---|---|
| `fix_issues` | Reads SonarCloud's analysis report, fetches bugs and code smells, generates LLM fixes, and lets you preview or apply them. Issues are grouped **by rule**. |
| `fix_security_issues` | Same pipeline, but for security vulnerabilities only. Issues are grouped **by file**. Generates a separate `CHANGES_SECURITY_*.md` changelog. |
| `analyze` | Displays project metrics from SonarCloud (lines of code, coverage, bugs, vulnerabilities) and an issues table. No fixes are generated. |
| `inspect` | Analyzes your local directory: file counts by language, git status, SonarCloud configuration presence. Does not contact SonarCloud. |

The remaining menu options (Add Provider, Update Provider, Change Parameters Configuration) are only available through the interactive menu because they require interactive prompts.

**Examples:**

```bash
# Fix up to 5 issues, preview only (no files modified)
devdox_sonar -c fix_issues --max-fixes 5

# Fix issues and apply them to files
devdox_sonar -c fix_issues --apply 1 --max-fixes 3

# Run the full pipeline but skip all file writes
devdox_sonar -c fix_issues --dry-run

# Only critical bugs
devdox_sonar -c fix_issues --types BUG --severity CRITICAL,BLOCKER

# Show project metrics from SonarCloud
devdox_sonar -c analyze
```

### CLI Options

These options can be passed with any direct mode command to override your saved configuration:

```
  -v, --verbose             Enable debug logging
  --types TEXT              Comma-separated issue types: BUG, VULNERABILITY, CODE_SMELL
  --severity TEXT           Comma-separated severities: BLOCKER, CRITICAL, MAJOR, MINOR, INFO
  --max-fixes INTEGER       Number of issues to process (0-20)
  --apply INTEGER           0 = preview only, 1 = apply fixes to files
  --dry-run                 Run the full pipeline but skip all file writes
```

### The fix_issues Pipeline

```
  Load Config (auth.json + config.toml)
           │
           ▼
  Clone Repo (git clone to /tmp)
           │
           ▼
  Fetch Issues from SonarCloud report
  Filter by type / severity
           │
           ▼
  Group by Rule
           │
           ▼
  ┌────────────────────────────────────────┐
  │  For Each Rule Group                   │
  │                                        │
  │  Extract Code (locate lines + context) │
  │           │                            │
  │           ▼                            │
  │  Select Handler                        │
  │           │                            │
  │           ▼                            │
  │  Generate Fix (LLM or AST-based)       │
  │           │                            │
  │           ▼                            │
  │  Preview (file, confidence, explain)   │
  │        /        \                      │
  │   apply=1     apply=0                  │
  │      │           │                     │
  │      ▼           ▼                     │
  │   Apply +      Skip                    │
  │   Validate       │                     │
  │      │           │                     │
  │      └─────┬─────┘                     │
  │            ▼                           │
  │     Continue to next issue? ─► YES ─┐  │
  │            │                        │  │
  │           NO                  (loop)│  │
  │            │                        │  │
  └────────────┼────────────────────────┘  │
               ▼                           │
  Write Changelog (CHANGES_REGULAR_*.md)   │
```

**Specialized rule handlers** — Most rules go through the LLM via `DefaultRuleHandler`. Two rules have dedicated handlers:

| Rule | Handler | Approach |
|---|---|---|
| `python:S7503` | `AsyncToSyncHandler` | Pure AST analysis. Detects async functions without `await`, removes the `async` keyword, and updates all call sites. No LLM call. |
| `python:S3776` | `CognitiveComplexityHandler` | Specialized refactoring prompt template optimized for breaking down complex functions. |

---

### Understanding the Output

**Terminal preview** — For each fix, the tool prints:

```
Fix Preview:
File: src/my_app/services/parser.py
Confidence: 0.92
Issues fixed: 1

Changes:
╭─ Explanation of changed ─────────────────────────────────────╮
│ Issue: python:S1066 - Collapsible if statements should be    │
│ merged → Fix: Combined nested if statements into a single    │
│ condition → Validation: Logic preserved, no side effects     │
╰──────────────────────────────────────────────────────────────╯
```

When `apply = 1`, a results summary follows:

```
Results:
Attempted: 1
Successful: 1
Failed: 0
Success Rate: 100.0%
```

Between issues, you are prompted:

```
? Continue to next issue? (Y/n)
```

**Markdown changelog** — For every issue processed, a detailed entry is written to a changelog file in your project root. The terminal shows the summary; the changelog contains the full code diffs.

Regular issues generate `CHANGES_REGULAR_YYYYMMDDHHMMSS.md`. Security issues generate `CHANGES_SECURITY_YYYYMMDDHHMMSS.md`.

Each changelog entry includes:

```markdown
## Issue: `python:S1066`

**Severity:** MAJOR
**File:** `src/my_app/services/parser.py`

### Problem
Merge this if statement with the enclosing one.

### Explanation
Issue: python:S1066 - Collapsible if statements should be merged
→ Fix: Combined nested if statements into a single condition
→ Validation: Logic preserved, no side effects

### Impact Assessment
- **Risk Level:** Medium
- **Breaking Change:** No
- **Logic Preserved:** Yes
- **Testing Required:** Recommended

### Suggested Fix

#### `parse_config` (Lines 42-48)

**Original:**
```python
if config is not None:
    if config.get("enabled"):
        run_parser(config)
```

**Fixed:**
```python
if config is not None and config.get("enabled"):
    run_parser(config)
```

### Review Checklist
- [ ] Code change preserves original logic
- [ ] No new bugs introduced
- [ ] Syntax validated
- [ ] Tests pass
- [ ] Ready for commit
```

**Confidence scores**

| Score | Guidance |
|---|---|
| 0.8 - 1.0 | Generally safe to apply. Review the changelog entry to confirm. |
| 0.6 - 0.8 | Read the changelog diff carefully. Test after applying. |
| Below 0.6 | Review manually or skip. |

**Backups** — When `create_backup = 1` and `apply = 1`, the tool copies your entire project directory to `<parent-dir>/<project-name>_backup_YYYYMMDD_HHMMSS/` before modifying any files.

---

### Workflow Recipes

```bash
# Preview fixes without modifying any files
devdox_sonar -c fix_issues --dry-run

# Fix up to 5 critical bugs, preview only
devdox_sonar -c fix_issues --types BUG --severity CRITICAL,BLOCKER --max-fixes 5

# Apply fixes to files
devdox_sonar -c fix_issues --apply 1 --max-fixes 10

# Security audit: review then apply
devdox_sonar -c analyze
devdox_sonar -c fix_security_issues --dry-run
devdox_sonar -c fix_security_issues --apply 1

# Code smell cleanup
devdox_sonar -c fix_issues --types CODE_SMELL --severity CRITICAL,MAJOR --max-fixes 10 --apply 1

# Debug logging
devdox_sonar --verbose -c fix_issues --max-fixes 3
```

---

# Advanced Topics

## Rule Exclusions

Some SonarCloud rules may not apply to your project. You can exclude specific rules so the tool skips them entirely.

**Format:** Comma-separated rule IDs in the format `language:SXXXX`. Example: `python:S7503,python:S3776,python:S107`.

**How to set them:**

Via the interactive menu: run `devdox_sonar`, select **Change Parameters Configuration**, and enter the rule IDs at the `Rules to be excluded` prompt.

Via `~/devdox/config.toml`:

```toml
[configuration]
exclude_rules = "python:S7503,python:S7493,python:S107"
```

**Commonly excluded rules:**

| Rule ID | What it flags | Why teams exclude it |
|---|---|---|
| `python:S7503` | Async functions that do not use `await` | Sometimes `async` is needed for interface compatibility even without `await` |
| `python:S7493` | Synchronous file I/O inside async functions | Intentional for small config files or startup code |
| `python:S107` | Functions with too many parameters | Common in FastAPI dependency injection |
| `python:S5852` | Regular expressions vulnerable to ReDoS | Safe patterns sometimes flagged incorrectly |
| `python:S3776` | Functions with high cognitive complexity | May prefer manual refactoring over LLM-generated rewrites |

**Finding rule IDs:** You can see rule IDs in the SonarCloud dashboard next to each issue, or in the changelog entries this tool generates.

---

## Supported Languages

**Issue fetching** works for all languages SonarCloud supports. The tool reads whatever SonarCloud has in its analysis report.

**Automated fixing** currently processes **Python files only** (`.py` extension, excluding files with a `test_` prefix). The prompt templates in `prompts/python/` are optimized for Python. The architecture (Jinja2 templates, language-agnostic models) is designed to support additional languages in the future.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | SonarCloud token is invalid or expired | Generate a new token at [sonarcloud.io/account/security](https://sonarcloud.io/account/security) and update `~/devdox/auth.json` |
| `Invalid API key` | LLM provider rejected the key | Verify the key is correct and billing is enabled. Use **Update Provider** in the interactive menu. |
| `Configuration not found` | No config files exist yet | Run `devdox_sonar` to start the setup wizard |
| `File not found` | `PROJECT_PATH` does not match the repo structure | Ensure `PROJECT_PATH` in `~/devdox/auth.json` points to the repository root |
| No issues returned | SonarCloud has not scanned the project yet, or all issues are resolved | Verify your project has a completed analysis at [sonarcloud.io](https://sonarcloud.io/) |

<details>
<summary><strong>FAQ</strong></summary>

**Can I use this with self-hosted SonarQube?**
The underlying `SonarCloudAnalyzer` class accepts a `base_url` parameter. If you are building custom tooling, you could point it at your SonarQube instance. The CLI currently targets SonarCloud.

**What if the LLM generates a bad fix?**
Every fix includes a confidence score. The validator agent catches many issues. You can enable backups before applying. `--dry-run` lets you run the full pipeline without writing any files.

**Does it modify my working directory?**
The tool clones your repo to `/tmp` for code extraction. Applied fixes are written to the path specified in `PROJECT_PATH`. The clone step does not affect your local uncommitted changes.

**Does SonarCloud need to have scanned my project first?**
Yes. DevDox AI Sonar reads SonarCloud's existing analysis report. It does not perform code analysis itself. If SonarCloud has not scanned your project, there are no issues to fix.

**How much does it cost?**
DevDox AI Sonar is free and open-source (Apache 2.0). You pay only for LLM API calls to your chosen provider. Google Gemini offers a free tier.

</details>

---

## License

This project is licensed under the [Apache License 2.0](LICENSE). You are free to use, modify, and distribute this software, including in commercial and proprietary projects, provided you include the original license and notice. The license also provides an express grant of patent rights from contributors.

## Support

[github.com/montymobile1/devdox-ai-sonar/issues](https://github.com/montymobile1/devdox-ai-sonar/issues)

## Authors

Created and maintained by **Hayat Bourji** (hayat.bourgi@montyholding.com) and **Mohammad Jaafar** (mohamadali.jaafar@montymobile.com) at [Monty Mobile](https://github.com/montymobile1).

## Acknowledgments

Built with [Click](https://github.com/pallets/click), [Rich](https://github.com/Textualize/rich), [Questionary](https://github.com/tmbo/questionary), [Pydantic](https://github.com/pydantic/pydantic), [Jinja2](https://github.com/pallets/jinja), and [aiofiles](https://github.com/Tinche/aiofiles). Powered by [OpenAI](https://openai.com), [Google Gemini](https://ai.google.dev), [TogetherAI](https://together.ai), and [OpenRouter](https://openrouter.ai). Integrates with [SonarCloud](https://sonarcloud.io/).
