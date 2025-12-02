#!/bin/bash
# DevDox AI Sonar - Test Commands
# This file provides convenient shortcuts for running tests
# Usage: source test_commands.sh

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to tests directory
cd_tests() {
    cd "$(dirname "${BASH_SOURCE[0]}")" || exit
}

# Verify setup
test_verify() {
    echo -e "${BLUE}Verifying test setup...${NC}"
    cd_tests
    python verify_setup.py
}

# Run all tests
test_all() {
    echo -e "${BLUE}Running all tests...${NC}"
    cd_tests
    python run_tests.py "$@"
}

# Run with coverage
test_coverage() {
    echo -e "${BLUE}Running tests with coverage...${NC}"
    cd_tests
    python run_tests.py --coverage "$@"
}

# Run specific suite
test_suite() {
    if [ -z "$1" ]; then
        echo -e "${YELLOW}Usage: test_suite <suite_name>${NC}"
        echo "Available suites: config, analyzer, fixer, validator, cli, integration"
        return 1
    fi
    
    echo -e "${BLUE}Running $1 test suite...${NC}"
    cd_tests
    python run_tests.py --suite "$1" "${@:2}"
}

# Run config tests
test_config() {
    echo -e "${BLUE}Running configuration tests...${NC}"
    cd_tests
    python run_tests.py --suite config "$@"
}

# Run analyzer tests
test_analyzer() {
    echo -e "${BLUE}Running analyzer tests...${NC}"
    cd_tests
    python run_tests.py --suite analyzer "$@"
}

# Run fixer tests
test_fixer() {
    echo -e "${BLUE}Running fixer tests...${NC}"
    cd_tests
    python run_tests.py --suite fixer "$@"
}

# Run CLI tests
test_cli() {
    echo -e "${BLUE}Running CLI tests...${NC}"
    cd_tests
    python run_tests.py --suite cli "$@"
}

# Run integration tests
test_integration() {
    echo -e "${BLUE}Running integration tests...${NC}"
    cd_tests
    python run_tests.py --suite integration "$@"
}

# Quick test (fast, no slow tests)
test_quick() {
    echo -e "${BLUE}Running quick tests (skipping slow tests)...${NC}"
    cd_tests
    pytest -m "not slow" -v "$@"
}

# Run failed tests only
test_failed() {
    echo -e "${BLUE}Re-running failed tests...${NC}"
    cd_tests
    pytest --lf -v "$@"
}

# Run with debugger
test_debug() {
    echo -e "${BLUE}Running tests with debugger...${NC}"
    cd_tests
    pytest --pdb "$@"
}

# Show test statistics
test_stats() {
    echo -e "${BLUE}Test Suite Statistics${NC}"
    echo "===================="
    cd_tests
    
    echo -e "\n${GREEN}Test Files:${NC}"
    find . -name "test_*.py" -type f | wc -l
    
    echo -e "\n${GREEN}Test Functions:${NC}"
    grep -r "def test_" test_*.py | wc -l
    
    echo -e "\n${GREEN}Test Classes:${NC}"
    grep -r "class Test" test_*.py | wc -l
    
    echo -e "\n${GREEN}Lines of Test Code:${NC}"
    wc -l test_*.py conftest.py | tail -1
}

# Clean test artifacts
test_clean() {
    echo -e "${BLUE}Cleaning test artifacts...${NC}"
    cd_tests
    
    rm -rf __pycache__ .pytest_cache htmlcov .coverage
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete
    find . -type f -name "test_run.log" -delete
    
    echo -e "${GREEN}✓ Cleaned${NC}"
}

# Show help
test_help() {
    echo -e "${BLUE}DevDox AI Sonar - Test Commands${NC}"
    echo "================================="
    echo ""
    echo "Setup & Verification:"
    echo "  test_verify         - Verify test setup"
    echo "  test_help           - Show this help"
    echo ""
    echo "Running Tests:"
    echo "  test_all            - Run all tests"
    echo "  test_coverage       - Run with coverage report"
    echo "  test_quick          - Run fast tests only"
    echo "  test_failed         - Re-run failed tests"
    echo ""
    echo "Test Suites:"
    echo "  test_suite <name>   - Run specific suite"
    echo "  test_config         - Run configuration tests"
    echo "  test_analyzer       - Run SonarCloud analyzer tests"
    echo "  test_fixer          - Run LLM fixer tests"
    echo "  test_cli            - Run CLI tests"
    echo "  test_integration    - Run integration tests"
    echo ""
    echo "Utilities:"
    echo "  test_debug          - Run with debugger (pdb)"
    echo "  test_stats          - Show test statistics"
    echo "  test_clean          - Clean test artifacts"
    echo ""
    echo "Examples:"
    echo "  test_all -v                    # Verbose output"
    echo "  test_coverage                  # Generate coverage report"
    echo "  test_suite fixer              # Run fixer tests only"
    echo "  test_config -k settings       # Run config tests matching 'settings'"
}

# Export functions
export -f cd_tests
export -f test_verify
export -f test_all
export -f test_coverage
export -f test_suite
export -f test_config
export -f test_analyzer
export -f test_fixer
export -f test_cli
export -f test_integration
export -f test_quick
export -f test_failed
export -f test_debug
export -f test_stats
export -f test_clean
export -f test_help

# Show available commands
echo -e "${GREEN}DevDox AI Sonar test commands loaded!${NC}"
echo "Type 'test_help' for available commands"
