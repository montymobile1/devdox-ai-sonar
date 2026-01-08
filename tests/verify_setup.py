#!/usr/bin/env python3
"""
Verification script for DevDox AI Sonar test suite.
Run this to verify the test setup is correct.
"""

import sys
from pathlib import Path


def verify_test_setup():
    """Verify that the test suite is set up correctly."""

    print("=" * 80)
    print("DevDox AI Sonar - Test Suite Verification")
    print("=" * 80)
    print()

    issues = []
    successes = []

    # Check if we're in the right directory
    current_dir = Path.cwd()
    if current_dir.name != "tests":
        print("⚠️  WARNING: Not in tests directory")
        print(f"   Current directory: {current_dir}")
        print(f"   Please cd to: {current_dir.parent / 'tests'}")
        print()
    else:
        successes.append("✅ In tests directory")

    # Check for required files
    required_files = [
        "conftest.py",
        "pytest.ini",
        "test_config.py",
        "test_llm_fixer.py",
        "test_sonar_analyzer.py",
        "test_cli.py",
        "run_tests.py",
        "README.md",
    ]

    print("Checking for required files...")
    print("-" * 80)

    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            successes.append(f"✅ {filename} ({size:,} bytes)")
        else:
            issues.append(f"❌ Missing: {filename}")

    print()

    # Try to import pytest
    print("Checking dependencies...")
    print("-" * 80)

    try:
        import pytest

        successes.append(f"✅ pytest {pytest.__version__} installed")
    except ImportError:
        issues.append("❌ pytest not installed (pip install pytest)")

    try:

        successes.append("✅ pytest-cov installed")
    except ImportError:
        issues.append("❌ pytest-cov not installed (pip install pytest-cov)")

    print()

    # Check Python version
    print("Checking Python version...")
    print("-" * 80)

    version = sys.version_info
    if version >= (3, 9):
        successes.append(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    else:
        issues.append(f"❌ Python {version.major}.{version.minor} (need 3.9+)")

    print()

    # Print results
    print("=" * 80)
    print("Verification Results")
    print("=" * 80)
    print()

    if successes:
        print("Successes:")
        for success in successes:
            print(f"  {success}")
        print()

    if issues:
        print("Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        print()
        print("Please resolve the issues above before running tests.")
        return False
    else:
        print("🎉 All checks passed! Test suite is ready to use.")
        print()
        print("Quick start commands:")
        print("  python run_tests.py                    # Run all tests")
        print("  python run_tests.py --coverage         # Run with coverage")
        print("  python run_tests.py --suite config     # Run config tests")
        print("  pytest test_config.py -v               # Run specific file")
        print()
        return True


if __name__ == "__main__":
    success = verify_test_setup()
    sys.exit(0 if success else 1)
