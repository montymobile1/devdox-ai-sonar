# DevDox AI SonarCloud
A powerful CLI tool and Python library that analyzes SonarCloud issues and generates AI-powered fix suggestions using Large Language Models (LLMs).

## Features

- 🔍 **Fetch SonarCloud Issues**: Retrieve issues from SonarCloud projects with filtering capabilities
- 📊 **Project Analytics**: Get comprehensive project metrics and quality insights
- 🤖 **AI-Powered Fixes**: Generate intelligent fix suggestions using OpenAI or Anthropic LLMs
- 🛠️ **Automated Application**: Apply fixes directly to your codebase with backup support
- 📁 **Project Inspection**: Analyze local project structure and configuration
- 🎯 **CLI Interface**: Easy-to-use command-line interface with rich output formatting
- 📦 **Library Support**: Use as a Python library in your own projects

## Installation

```bash
pip install devdox_ai_sonar
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
- LLM API key (OpenAI or Anthropic)
- `sonar-tools` CLI (automatically installed)

## Quick Start

### 1. Set Environment Variables

```bash
export SONAR_TOKEN="your_sonarcloud_token"
export OPENAI_API_KEY="your_openai_api_key"
# OR
export GEMINI_API_KEY="your_anthropic_api_key"
```

### 2. Analyze a SonarCloud Project

```bash
devdox_ai_sonar analyze \
  --token $SONAR_TOKEN \
  --organization "your-org" \
  --project "your-project" \
  --branch "main"
```

### 3. Generate and Apply Fixes

```bash
devdox_ai_sonar fix \
  --token $SONAR_TOKEN \
  --organization "your-org" \
  --project "your-project" \
  --project-path "/path/to/your/project" \
  --apply
```

## CLI Commands

### `analyze`

Fetch and display SonarCloud issues for a project.

```bash
devdox_ai_sonar analyze [OPTIONS]

Options:
  -t, --token TEXT              SonarCloud authentication token [required]
  -o, --organization TEXT       SonarCloud organization key [required]
  -p, --project TEXT            SonarCloud project key [required]
  -b, --branch TEXT             Branch to analyze (default: main)
  --severity [BLOCKER|CRITICAL|MAJOR|MINOR|INFO]
                               Filter by severity (multiple values allowed)
  --type [BUG|VULNERABILITY|CODE_SMELL|SECURITY_HOTSPOT]
                               Filter by issue type (multiple values allowed)
  --output PATH                Output file for results (JSON format)
  --limit INTEGER              Limit number of issues to display
```

**Example:**
```bash
devdox_ai_sonar analyze \
  --token $SONAR_TOKEN \
  --organization "devdox" \
  --project "devdox-ai-context" \
  --severity CRITICAL MAJOR \
  --type BUG CODE_SMELL \
  --limit 20 \
  --output results.json
```

### `fix`

Generate LLM-powered fixes and optionally apply them to your codebase.

```bash
devdox_ai_sonar fix [OPTIONS]

Options:
  -t, --token TEXT              SonarCloud authentication token [required]
  -o, --organization TEXT       SonarCloud organization key [required]
  -p, --project TEXT            SonarCloud project key [required]
  --project-path PATH           Path to local project directory [required]
  -b, --branch TEXT             Branch to analyze (default: main)
  --provider [openai|anthropic] LLM provider (default: openai)
  --model TEXT                  LLM model name
  --api-key TEXT                LLM API key
  --max-fixes INTEGER           Maximum fixes to generate (default: 10)
  --apply                       Apply fixes to the codebase
  --dry-run                     Show what would be changed without applying
  --backup / --no-backup        Create backup before applying (default: true)
```

**Example:**
```bash
devdox_ai_sonar fix \
  --token $SONAR_TOKEN \
  --organization "devdox" \
  --project "devdox-ai-context" \
  --project-path "/Users/hayat/PycharmProjects/devdox-ai-context" \
  --provider "openai" \
  --model "gpt-4o" \
  --max-fixes 5 \
  --apply \
  --backup
```

### `inspect`

Analyze local project directory structure.

```bash
devdox_ai_sonar inspect PROJECT_PATH
```

**Example:**
```bash
devdox_ai_sonar inspect /path/to/your/project
```

## Configuration

### Environment Variables

- `SONAR_TOKEN`: SonarCloud authentication token
- `OPENAI_API_KEY`: OpenAI API key (for OpenAI provider)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Anthropic provider)

### SonarCloud Setup

1. Get your SonarCloud token from: https://sonarcloud.io/account/security
2. Find your organization and project keys in SonarCloud dashboard
3. Ensure your project is analyzed and has issues

### LLM Provider Setup

#### OpenAI
- Sign up at https://platform.openai.com
- Create API key in your account settings
- Recommended models: `gpt-4o`, `gpt-4-turbo`

#### Anthropic
- Sign up at https://console.anthropic.com
- Create API key in your account settings  
- Recommended models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`

## Supported Languages

The tool supports fix generation for:

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

## Issue Types Supported for Fixes

- **Bugs**: Logic errors, null pointer issues, etc.
- **Code Smells**: Style issues, maintainability problems
- **Security Hotspots**: Some security-related issues (with caution)

**Note**: Vulnerabilities are typically not auto-fixed as they require careful human review.

## Best Practices

### Before Using Fixes

1. **Backup Your Code**: Always create backups before applying fixes (enabled by default)
2. **Review Fixes**: Examine generated fixes before applying, especially for critical code
3. **Test Thoroughly**: Run your test suite after applying fixes
4. **Version Control**: Commit changes incrementally to track fix results

### Fix Quality

- **High Confidence** (>0.8): Generally safe to apply
- **Medium Confidence** (0.6-0.8): Review before applying
- **Low Confidence** (<0.6): Manual review required

### Performance Tips

- Use `--max-fixes` to limit processing time for large projects
- Filter by severity to focus on critical issues first
- Use `--dry-run` to preview changes before applying

## Development

### Setup

```bash
git clone https://github.com/montymobile1/devdox-ai-sonar.git
cd devdox_ai_sonar
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/

# Test
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure code quality checks pass
6. Submit a pull request

## Troubleshooting

### Common Issues

**SonarCloud Authentication Error**
```
Error: 401 Unauthorized
```
- Check your SonarCloud token is valid
- Verify token has access to the organization/project

**LLM API Errors**
```
Error: Invalid API key
```
- Verify your OpenAI/Anthropic API key
- Check rate limits and billing status

**File Not Found**
```
File not found: /path/to/file.py
```
- Ensure project path matches SonarCloud project structure
- Check file permissions

### Debug Mode

Enable verbose output for detailed error information:

```bash
devdox_ai_sonar --verbose [command]
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.