"""Test cases for indentation normalization fix."""

import pytest
from devdox_ai_sonar.utils.file_indentation import (
    normalize_indentation,
    apply_indentation_to_fix,
)


def test_normalize_indentation_with_inconsistent_spacing():
    """Test that normalize_indentation handles inconsistently indented code from LLM."""
    # Simulate LLM returning code with inconsistent indentation
    # First line has no spaces, subsequent lines have 8 spaces
    lines = [
        "self.__user_wallet_service = UserWalletService()",
        "        self.__promotion_service = PromotionService()",
        "        self.__bundle_service = BundleService()",
        "        self.__dcb_service = dcb_service_instance()",
    ]

    result = normalize_indentation(lines)

    # All lines should now have NO leading spaces
    assert result[0] == "self.__user_wallet_service = UserWalletService()"
    assert result[1] == "self.__promotion_service = PromotionService()"
    assert result[2] == "self.__bundle_service = BundleService()"
    assert result[3] == "self.__dcb_service = dcb_service_instance()"


def test_normalize_indentation_with_empty_lines():
    """Test that normalize_indentation preserves empty lines."""
    lines = [
        "def foo():",
        "    return 42",
        "",
        "    print('hello')",
    ]

    result = normalize_indentation(lines)

    assert result[0] == "def foo():"
    assert result[1] == "return 42"
    assert result[2] == ""
    assert result[3] == "print('hello')"


def test_apply_indentation_to_fix_with_inconsistent_code():
    """Test that apply_indentation_to_fix correctly handles inconsistently indented code."""
    # LLM returns code with inconsistent indentation
    fixed_code = """self.__user_wallet_service = UserWalletService()
        self.__promotion_service = PromotionService()
        self.__bundle_service = BundleService()"""

    base_indent = "        "  # 8 spaces

    result = apply_indentation_to_fix(fixed_code, base_indent)

    # All lines should have exactly 8 spaces of indentation
    expected_lines = [
        "        self.__user_wallet_service = UserWalletService()",
        "        self.__promotion_service = PromotionService()",
        "        self.__bundle_service = BundleService()",
    ]
    expected = "\n".join(expected_lines)

    assert result == expected


def test_apply_indentation_preserves_relative_indentation():
    """Test that apply_indentation preserves relative indentation within code blocks."""
    fixed_code = """def foo():
    if True:
        return 42
    return 0"""

    base_indent = "    "  # 4 spaces

    result = apply_indentation_to_fix(fixed_code, base_indent)

    # Each line gets base indent, relative indentation is preserved
    expected_lines = [
        "    def foo():",
        "    if True:",
        "    return 42",
        "    return 0",
    ]
    expected = "\n".join(expected_lines)

    assert result == expected


def test_normalize_indentation_all_same_indent():
    """Test normalize_indentation when all lines have the same indentation."""
    lines = [
        "    line1",
        "    line2",
        "    line3",
    ]

    result = normalize_indentation(lines)

    assert result[0] == "line1"
    assert result[1] == "line2"
    assert result[2] == "line3"


def test_normalize_indentation_mixed_tabs_and_spaces():
    """Test normalize_indentation with mixed tabs and spaces."""
    lines = [
        "\t\tline1",
        "    line2",
        "        line3",
    ]

    result = normalize_indentation(lines)

    # All leading whitespace should be stripped
    assert result[0] == "line1"
    assert result[1] == "line2"
    assert result[2] == "line3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
