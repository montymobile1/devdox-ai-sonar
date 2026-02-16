# DevDox AI SonarCloud

A powerful CLI tool and Python library that analyzes SonarCloud issues and generates AI-powered fix suggestions using Large Language Models (LLMs).

## Features

- 🔍 **Fetch SonarCloud Issues**: Retrieve issues from SonarCloud projects with filtering capabilities
- 📊 **Project Analytics**: Get comprehensive project metrics and quality insights
- 🤖 **AI-Powered Fixes**: Generate intelligent fix suggestions using OpenAI, Anthropic, or Google Gemini LLMs
- 🛠️ **Automated Application**: Apply fixes directly to your codebase with backup support
- 📁 **Project Inspection**: Analyze local project structure and configuration
- 🎯 **Interactive CLI**: Claude Code-style interactive interface with rich output formatting
- 🔄 **Command Switching**: Switch between commands mid-workflow by typing `/`
- 🔐 **Security Focus**: Specialized security vulnerability fixing mode
- 📦 **Library Support**: Use as a Python library in your own projects
- ⚙️ **Flexible Configuration**: Interactive configuration management for providers and SonarCloud
- 🚫 **Rule Exclusions**: Exclude specific SonarQube rules from analysis

## Installation
```bash
pip install devdox_sonar
```

### Development Installation
```bash
git clone https://github.com/montymobile1/devdox-ai-sonar.git
cd devdox-ai-sonar
pip install -e ".[dev]"
```

## Requirements

- Python 3.9+
- SonarCloud authentication token
- LLM API key (OpenAI, Anthropic, or Google Gemini)

## Quick Start

### 1. Interactive Configuration

The easiest way to get started is through the interactive configuration:
```bash
devdox_sonar
```

This will guide you through:
1. SonarCloud configuration (token, organization, project)
2. LLM provider setup (OpenAI, Anthropic, or Gemini)
3. Analysis parameters (issue types, severities, max fixes)

### 2. Direct Command Execution

Run specific commands directly:
```bash
# Fix issues
devdox_sonar -c fix_issues

# Fix security issues specifically
devdox_sonar -c fix_security_issues

# Analyze project
devdox_sonar -c analyze

# Inspect local project
devdox_sonar -c inspect
```

### 3. Command Line with Options
```bash
devdox_sonar -c fix_issues \
  --types BUG,CODE_SMELL \
  --severity CRITICAL,BLOCKER \
  --max-fixes 10 \
  --apply 1
```

## Interactive Mode

The interactive mode provides a Claude Code-style experience:
```
═══════════════════════════════════════════════
   DevDox AI Sonar - Interactive Mode          
═══════════════════════════════════════════════

? What would you like to do?
  ➕ Add Provider - Add provider or sonar configuration
  ✏️ Update Provider - Update provider or sonar configuration
  🔧 Fix Issues - Generate and apply LLM-powered fixes
  🔒 Fix Security Issues - Specialized security vulnerability fixes
  📊 Analyze Project - Display SonarCloud analysis
  🔍 Inspect Project - Analyze local directory structure
  ⚙️ Change Parameters Configuration
  ❌ Exit
```

### Command Switching

At any prompt, type `/` to return to the main menu and switch to a different command.

## CLI Commands

### Main Entry Point
```bash
devdox_sonar [OPTIONS]

Options:
  -v, --verbose              Enable verbose output
  -c, --command TEXT         Run specific command directly
  --types TEXT              Comma-separated issue types
  --severity TEXT           Comma-separated severities
  --max-fixes INTEGER       Maximum number of fixes (0-100)
  --apply INTEGER           Apply fixes (1=apply, 0=preview)
  --dry-run                 Show changes without applying
  --version                 Show version
  --help                    Show help message
```

### Configuration Commands

#### `add_provider`

Add a new LLM provider configuration:
```bash
devdox_sonar -c add_provider
```

**Supported Providers:**
- OpenAI (GPT-4, GPT-4 Turbo, GPT-3.5)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- Google Gemini (Gemini Pro, Gemini Ultra)

#### `update_provider`

Update an existing provider configuration:
```bash
devdox_sonar -c update_provider
```

#### `change_parameters`

Modify analysis parameters:
```bash
devdox_sonar -c change_parameters
```

**Configurable Parameters:**
- Branch or Pull Request selection
- Maximum number of fixes
- Issue types (BUG, VULNERABILITY, CODE_SMELL, SECURITY_HOTSPOT)
- Severities (BLOCKER, CRITICAL, MAJOR, MINOR, INFO)
- Apply fixes automatically
- Create backups before applying
- Exclude specific rules

### Analysis Commands

#### `fix_issues`

Generate and apply LLM-powered fixes for regular issues:
```bash
devdox_sonar -c fix_issues [OPTIONS]

Options:
  --types TEXT              Filter by issue types
  --severity TEXT           Filter by severities
  --max-fixes INTEGER       Maximum fixes to generate
  --apply INTEGER           Apply fixes (1=yes, 0=no)
  --dry-run                 Preview changes only
```

**Example:**
```bash
devdox_sonar -c fix_issues \
  --types "BUG,CODE_SMELL" \
  --severity "CRITICAL,MAJOR" \
  --max-fixes 5 \
  --apply 1
```

#### `fix_security_issues`

Specialized command for fixing security vulnerabilities:
```bash
devdox_sonar -c fix_security_issues [OPTIONS]
```

**Features:**
- Focuses exclusively on security issues
- Groups issues by file for comprehensive fixes
- Enhanced validation for security-sensitive changes
- Detailed markdown documentation of changes

#### `analyze`

Display SonarCloud analysis results:
```bash
devdox_sonar -c analyze
```

**Shows:**
- Project metrics (lines of code, coverage, bugs, vulnerabilities)
- Issue breakdown by severity and type
- Detailed issue list with file locations

#### `inspect`

Analyze local project directory structure:
```bash
devdox_sonar -c inspect
```

**Analyzes:**
- File counts by language
- Source directory detection
- SonarCloud configuration presence
- Git repository status

## Configuration

### Configuration Files

DevDox AI Sonar uses two configuration files:

1. **Auth Configuration** (`~/.devdox_sonar_auth.json`):
```json
   {
     "token": "your_sonarcloud_token",
     "organization": "your_org",
     "project": "your_project",
     "project_path": "/path/to/project"
   }
```

2. **LLM Configuration** (`~/.devdox_sonar_config.yaml`):
```yaml
   llm:
     default_provider: "openai",
     default_model: "gpt-4o"
     providers:
       openai:
         api_key: "your_openai_key",
         name: "openai"
         default_model: "gpt-4o",
         model: "gpt-4o",
         models:
            - "gpt-4o"
            - "mini-gpt-4o"

   configuration:
     max_fixes: 10
     types: "BUG,CODE_SMELL"
     severities: "CRITICAL,MAJOR"
     apply: 0
     create_backup: 1
     exclude_rules: "python:S3776,python:S7493"
```

### Environment Variables

Alternatively, use environment variables:
```bash
export SONAR_TOKEN="your_sonarcloud_token"
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

### SonarCloud Setup

1. Get your token: https://sonarcloud.io/account/security
2. Find organization/project keys in SonarCloud dashboard
3. Ensure your project has recent analysis results

### LLM Provider Setup

#### OpenAI
- Sign up: https://platform.openai.com
- Recommended models: `gpt-4o`, `gpt-4-turbo`

#### Anthropic
- Sign up: https://console.anthropic.com
- Recommended models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`

#### Google Gemini
- Sign up: https://makersuite.google.com
- Recommended models: `gemini-pro`, `gemini-ultra`

## Rule Exclusions

Exclude specific SonarQube rules from analysis:

### Via Interactive Configuration
```bash
devdox_sonar -c change_parameters
# Select "Rules to be excluded"
# Enter: python:S7503,python:S7493,python:S107
```

### Via Configuration File

Add to `~/.config.yaml`:
```yaml
configura[config.toml](../../../devdox/config.toml)tion:
  exclude_rules: "python:S7503,python:S7493,python:S107"
```

### Common Rules to Exclude

| Rule ID | Description | Reason to Exclude |
|---------|-------------|-------------------|
| `python:S7503` | Async functions should use async features |
| `python:S7493` | Async functions should not contain synchronous file operations |
| `python:S107` | Too Many Parameters | FastAPI dependency injection |
| `python:S5852` | ReDoS | Safe patterns flagged incorrectly |

## Issue Types and Processing

### Regular Issues (By Rule)

Regular issues are grouped by rule and processed individually:

- **BUG**: Logic errors, null pointer issues
- **CODE_SMELL**: Style issues, maintainability problems
- **SECURITY_HOTSPOT**: Potential security concerns

### Security Issues (By File)

Security issues are grouped by file for comprehensive fixes:

- **VULNERABILITY**: SQL injection, XSS, authentication issues
- Processed once per file with all security issues
- Enhanced validation and documentation

## Supported Languages

- Python (.py)
- JavaScript/TypeScript (.js, .jsx, .ts, .tsx)
- Java (.java)
- Kotlin (.kt)
- Scala (.scala)
- Go (.go)
- Rust (.rs)
- C/C++ (.c, .cpp)
- C# (.cs)
- PHP (.php)
- Ruby (.rb)
- Swift (.swift)

## Best Practices

### Before Using Fixes

1. **Backup Your Code**: Automatic backup creation is enabled by default
2. **Review Generated Fixes**: Always review before applying to production code
3. **Test Thoroughly**: Run your test suite after applying fixes
4. **Version Control**: Commit changes incrementally
5. **Use Dry Run**: Preview changes with `--dry-run` flag

### Fix Confidence Levels

- **High (>0.8)**: Generally safe to apply
- **Medium (0.6-0.8)**: Review recommended
- **Low (<0.6)**: Manual review required

### Security Fixes

- Always manually review security-related fixes
- Test security fixes thoroughly
- Consider security implications of automated changes
- Review generated markdown documentation

### Performance Tips

- Use `--max-fixes` to limit processing for large projects
- Filter by severity to prioritize critical issues
- Exclude false-positive rules to reduce noise
- Use branch/PR filtering to focus on specific changes

## Change Documentation

All fixes generate markdown documentation:

- Regular issues: `CHANGES_REGULAR_YYYYMMDDHHMMSS.md`
- Security issues: `CHANGES_SECURITY_YYYYMMDDHHMMSS.md`

Documentation includes:
- Original code
- Fixed code
- Explanation of changes
- Rule information
- Confidence scores

## Troubleshooting

### Common Issues

**Authentication Error**
```
Error: 401 Unauthorized
```
Solution: Verify SonarCloud token and permissions

**LLM API Errors**
```
Error: Invalid API key
```
Solution: Check API key validity and billing status

**File Not Found**
```
File not found: /path/to/file.py
```
Solution: Verify project path matches SonarCloud structure

**Configuration Not Found**
```
❌ Configuration not found
```
Solution: Run `devdox_sonar` without options to initialize configuration

### Debug Mode

Enable verbose output:
```bash
devdox_sonar --verbose -c fix_issues
```


## Development

### Setup
```bash
git clone https://github.com/montymobile1/devdox-ai-sonar.git
cd devdox-ai-sonar
pip install -e ".[dev]"
```



## Command Reference

### Quick Command List
```bash
# Interactive mode
devdox_sonar

# Configuration
devdox_sonar -c add_provider
devdox_sonar -c update_provider
devdox_sonar -c change_parameters

# Analysis and fixing
devdox_sonar -c fix_issues
devdox_sonar -c fix_security_issues
devdox_sonar -c analyze
devdox_sonar -c inspect

# With options
devdox_sonar -c fix_issues --apply 1 --max-fixes 10
devdox_sonar -c fix_issues --dry-run
devdox_sonar --verbose -c analyze
```

## Workflow Examples

### First-Time Setup
```bash
# Step 1: Initialize configuration
devdox_sonar

# Step 2: Configure SonarCloud
# Follow interactive prompts for token, org, project

# Step 3: Add LLM provider
# Select and configure your preferred provider

# Step 4: Set parameters
# Configure issue types, severities, etc.

# Step 5: Run analysis
# Select "Fix Issues" from menu
```

### Regular Workflow
```bash
# Preview fixes without applying
devdox_sonar -c fix_issues --dry-run

# Apply fixes with backup
devdox_sonar -c fix_issues --apply 1

# Focus on critical bugs only
devdox_sonar -c fix_issues \
  --types BUG \
  --severity CRITICAL,BLOCKER \
  --max-fixes 5 \
  --apply 1
```

### Security Audit Workflow
```bash
# Analyze security issues
devdox_sonar -c analyze

# Fix security vulnerabilities
devdox_sonar -c fix_security_issues --dry-run

# Review generated CHANGES_SECURITY_*.md

# Apply after review
devdox_sonar -c fix_security_issues --apply 1
```

## License

MIT License - see [LICENSE](LICENSE) file for details.



## Support

- **Issues**: https://github.com/montymobile1/devdox-ai-sonar/issues