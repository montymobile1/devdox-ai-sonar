"""Test runner script for devdox-ai-sonar test suite."""

import sys
import pytest
from pathlib import Path


def main():
    """Run the test suite with appropriate configuration."""

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = Path(__file__).parent

    # Default pytest arguments
    pytest_args = [
        str(tests_dir),
        "-v",
        "--tb=short",
        "--strict-markers",
    ]

    # Parse custom arguments
    if len(sys.argv) > 1:
        custom_args = sys.argv[1:]

        # Handle special flags
        if "--coverage" in custom_args:
            custom_args.remove("--coverage")
            pytest_args.extend([
                f"--cov={project_root}/src/devdox_ai_sonar",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=70",
            ])

        if "--suite" in custom_args:
            suite_idx = custom_args.index("--suite")
            suite_name = custom_args[suite_idx + 1]
            custom_args = custom_args[:suite_idx] + custom_args[suite_idx + 2:]

            # Map suite names to test files
            suite_map = {
                "config": "test_config.py",
                "models": "test_models.py",
                "analyzer": "test_sonar_analyzer.py",
                "fixer": "test_llm_fixer.py",
                "validator": "test_fix_validator.py",
                "cli": "test_cli.py",
                "integration": "test_integration.py",
            }

            if suite_name in suite_map:
                pytest_args = [str(tests_dir / suite_map[suite_name])] + pytest_args[1:]
            else:
                print(f"Unknown test suite: {suite_name}")
                print(f"Available suites: {', '.join(suite_map.keys())}")
                return 1

        # Add remaining custom arguments
        pytest_args.extend(custom_args)

    # Run pytest
    print("=" * 80)
    print("Running DevDox AI Sonar Test Suite")
    print("=" * 80)
    print(f"Test directory: {tests_dir}")
    print(f"Arguments: {' '.join(pytest_args)}")
    print("=" * 80)
    print()

    exit_code = pytest.main(pytest_args)

    # Print summary
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
