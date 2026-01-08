#!/usr/bin/env python3
"""Quick test script to verify the indentation fix."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from devdox_ai_sonar.utils.file_indentation import (
    normalize_indentation,
    apply_indentation_to_fix,
)


def test_inconsistent_indentation():
    """Test the main bug: LLM returns inconsistently indented code."""
    print("Testing inconsistent indentation fix...")

    # Simulate LLM returning code with inconsistent indentation
    lines = [
        "self.__user_wallet_service = UserWalletService()",
        "        self.__promotion_service = PromotionService()",
        "        self.__bundle_service = BundleService()",
    ]

    print(f"Input lines:")
    for i, line in enumerate(lines):
        print(f"  {i}: '{line}' (length: {len(line)}, stripped: '{line.strip()}')")

    result = normalize_indentation(lines)

    print(f"\nNormalized lines:")
    for i, line in enumerate(result):
        print(f"  {i}: '{line}' (length: {len(line)})")

    # Verify all lines have no leading spaces
    assert result[0] == "self.__user_wallet_service = UserWalletService()"
    assert result[1] == "self.__promotion_service = PromotionService()"
    assert result[2] == "self.__bundle_service = BundleService()"

    print("✓ normalize_indentation test PASSED")


def test_apply_indentation():
    """Test that apply_indentation_to_fix works correctly."""
    print("\nTesting apply_indentation_to_fix...")

    # LLM returns code with inconsistent indentation
    fixed_code = """self.__user_wallet_service = UserWalletService()
        self.__promotion_service = PromotionService()
        self.__bundle_service = BundleService()"""

    base_indent = "        "  # 8 spaces

    result = apply_indentation_to_fix(fixed_code, base_indent)

    print(f"Input code:\n{repr(fixed_code)}")
    print(f"\nBase indent: '{base_indent}' (length: {len(base_indent)})")
    print(f"\nResult:\n{repr(result)}")

    # All lines should have exactly 8 spaces of indentation
    result_lines = result.split("\n")
    print(f"\nResult lines:")
    for i, line in enumerate(result_lines):
        spaces = len(line) - len(line.lstrip())
        print(f"  {i}: '{line}' (leading spaces: {spaces})")

    # Verify each line has exactly 8 leading spaces
    for i, line in enumerate(result_lines):
        assert line.startswith("        "), f"Line {i} should start with 8 spaces, got: '{line}'"
        assert not line.startswith("         "), f"Line {i} should NOT start with 9+ spaces, got: '{line}'"

    print("✓ apply_indentation_to_fix test PASSED")


if __name__ == "__main__":
    try:
        test_inconsistent_indentation()
        test_apply_indentation()
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
