import re
import pytest

from devdox_ai_sonar.services.rule_handler import AWAIT_REMOVAL_PATTERN


def build_pattern(func_name: str) -> str:
    """Build the full regex pattern for a given function name."""
    return AWAIT_REMOVAL_PATTERN.format(func_name=re.escape(func_name))


def apply_removal(line: str, func_name: str, count: int = 1) -> str:
    """Apply the await removal regex, mirroring how apply_search_replace_change uses it."""
    return re.sub(build_pattern(func_name), "", line, count=count)


# ============================================================================
# BASIC HAPPY PATHS
# ============================================================================


class TestAwaitRemovalBasic:
    """Basic cases where await should be removed."""

    def test_simple_call(self):
        assert apply_removal("await foo()", "foo") == "foo()"

    def test_call_with_assignment(self):
        assert apply_removal("result = await foo()", "foo") == "result = foo()"

    def test_call_with_positional_args(self):
        assert apply_removal("result = await foo(x, y)", "foo") == "result = foo(x, y)"

    def test_call_with_keyword_args(self):
        assert apply_removal("await foo(x=1, y=2)", "foo") == "foo(x=1, y=2)"

    def test_call_no_args(self):
        assert apply_removal("await foo()", "foo") == "foo()"

    def test_call_with_trailing_comment(self):
        assert apply_removal("await foo()  # do stuff", "foo") == "foo()  # do stuff"

    def test_return_await(self):
        assert apply_removal("return await foo()", "foo") == "return foo()"

    def test_yield_await(self):
        assert apply_removal("yield await foo()", "foo") == "yield foo()"

    def test_if_condition(self):
        assert apply_removal("if await foo():", "foo") == "if foo():"

    def test_while_condition(self):
        assert apply_removal("while await foo():", "foo") == "while foo():"


# ============================================================================
# WHITESPACE VARIATIONS
# ============================================================================


class TestAwaitRemovalWhitespace:
    """Whitespace between await and function name."""

    def test_multiple_spaces(self):
        assert apply_removal("await   foo()", "foo") == "foo()"

    def test_tab_between(self):
        assert apply_removal("await\tfoo()", "foo") == "foo()"

    def test_mixed_whitespace(self):
        assert apply_removal("await \t foo()", "foo") == "foo()"

    def test_leading_indentation_spaces(self):
        assert apply_removal("    await foo()", "foo") == "    foo()"

    def test_leading_indentation_tab(self):
        assert apply_removal("\tawait foo()", "foo") == "\tfoo()"

    def test_deep_indentation(self):
        assert apply_removal("        result = await foo()", "foo") == "        result = foo()"

    def test_space_before_paren(self):
        assert apply_removal("await foo ()", "foo") == "foo ()"

    def test_multiple_spaces_before_paren(self):
        assert apply_removal("await foo  ()", "foo") == "foo  ()"


# ============================================================================
# DOTTED / CHAINED ACCESS
# ============================================================================


class TestAwaitRemovalDottedAccess:
    """Await removal with module/object prefixes."""

    def test_single_module_prefix(self):
        assert apply_removal("await module.foo()", "foo") == "module.foo()"

    def test_double_module_prefix(self):
        assert apply_removal("await a.b.foo()", "foo") == "a.b.foo()"

    def test_triple_module_prefix(self):
        assert apply_removal("await a.b.c.foo()", "foo") == "a.b.c.foo()"

    def test_self_prefix(self):
        assert apply_removal("await self.foo()", "foo") == "self.foo()"

    def test_cls_prefix(self):
        assert apply_removal("await cls.foo()", "foo") == "cls.foo()"

    def test_dotted_with_assignment(self):
        assert apply_removal("result = await self.foo(x)", "foo") == "result = self.foo(x)"

    def test_dotted_with_underscore_module(self):
        assert apply_removal("await _private.foo()", "foo") == "_private.foo()"


# ============================================================================
# MUST NOT MATCH — DIFFERENT FUNCTION NAMES
# ============================================================================


class TestAwaitRemovalNoMatch:
    """Cases where the regex must NOT match."""

    def test_different_function_name(self):
        assert apply_removal("await bar()", "foo") == "await bar()"

    def test_function_name_is_substring_prefix(self):
        """'foo' should not match 'foobar'."""
        assert apply_removal("await foobar()", "foo") == "await foobar()"

    def test_function_name_is_substring_suffix(self):
        """'foo' should not match 'barfoo'."""
        assert apply_removal("await barfoo()", "foo") == "await barfoo()"

    def test_function_name_with_leading_underscore(self):
        """'foo' should not match '_foo'."""
        assert apply_removal("await _foo()", "foo") == "await _foo()"

    def test_function_name_with_trailing_underscore(self):
        """'foo' should not match 'foo_'."""
        assert apply_removal("await foo_()", "foo") == "await foo_()"

    def test_function_name_different_case(self):
        """'foo' should not match 'Foo' (case sensitive)."""
        assert apply_removal("await Foo()", "foo") == "await Foo()"

    def test_function_name_all_caps(self):
        """'foo' should not match 'FOO'."""
        assert apply_removal("await FOO()", "foo") == "await FOO()"

    def test_no_parentheses(self):
        """'await foo' without parens is not a call — should not match."""
        assert apply_removal("await foo", "foo") == "await foo"

    def test_function_name_used_as_argument(self):
        """'foo' as an argument name should not trigger removal."""
        assert apply_removal("await bar(foo)", "foo") == "await bar(foo)"

    def test_dotted_but_wrong_method(self):
        """'foo' should not match 'module.bar()'."""
        assert apply_removal("await module.bar()", "foo") == "await module.bar()"


# ============================================================================
# AWAIT AS PART OF ANOTHER WORD
# ============================================================================


class TestAwaitRemovalWordBoundary:
    """Ensure \\bawait\\b word boundary works."""

    def test_not_await_prefix(self):
        """'not_await' should not be matched."""
        assert apply_removal("not_await foo()", "foo") == "not_await foo()"

    def test_awaiting(self):
        """'awaiting' should not be matched."""
        assert apply_removal("awaiting foo()", "foo") == "awaiting foo()"

    def test_my_await(self):
        """'my_await' should not be matched."""
        assert apply_removal("my_await foo()", "foo") == "my_await foo()"


# ============================================================================
# MULTIPLE AWAITS ON ONE LINE
# ============================================================================


class TestAwaitRemovalMultipleAwaits:
    """Lines with more than one await expression."""

    def test_target_is_inner_await(self):
        """Only the await before 'foo' should be removed."""
        assert apply_removal("await bar(await foo())", "foo") == "await bar(foo())"

    def test_target_is_outer_await(self):
        """Only the await before 'bar' should be removed."""
        assert apply_removal("await bar(await foo())", "bar") == "bar(await foo())"

    def test_two_awaits_same_function_count_1(self):
        """With count=1, only the first matching await is removed."""
        result = apply_removal("x = await foo() + await foo()", "foo", count=1)
        assert result == "x = foo() + await foo()"

    def test_two_awaits_same_function_count_0(self):
        """With count=0 (replace all), both are removed."""
        result = apply_removal("x = await foo() + await foo()", "foo", count=0)
        assert result == "x = foo() + foo()"

    def test_await_in_both_sides_of_or(self):
        """Only await before target is removed."""
        assert apply_removal("x = await foo() or await bar()", "foo") == "x = foo() or await bar()"

    def test_await_in_both_sides_of_and(self):
        assert apply_removal("x = await bar() and await foo()", "foo") == "x = await bar() and foo()"

    def test_keyword_arg_same_as_func_name(self):
        """'foo=' should not be confused with 'foo('."""
        assert apply_removal("await bar(foo=await foo())", "foo") == "await bar(foo=foo())"


# ============================================================================
# PARENTHESIZED AWAIT
# ============================================================================


class TestAwaitRemovalParenthesized:
    """Await inside parentheses for chaining."""

    def test_paren_await_dot_access(self):
        assert apply_removal("x = (await foo()).bar", "foo") == "x = (foo()).bar"

    def test_paren_await_subscript(self):
        assert apply_removal("x = (await foo())[0]", "foo") == "x = (foo())[0]"

    def test_paren_await_dotted_prefix(self):
        assert apply_removal("x = (await module.foo()).bar", "foo") == "x = (module.foo()).bar"

    def test_double_paren(self):
        assert apply_removal("x = ((await foo()))", "foo") == "x = ((foo()))"


# ============================================================================
# NESTED IN EXPRESSIONS
# ============================================================================


class TestAwaitRemovalNestedExpressions:
    """Await inside various expression contexts."""

    def test_inside_list_literal(self):
        assert apply_removal("x = [await foo()]", "foo") == "x = [foo()]"

    def test_inside_dict_key(self):
        assert apply_removal("x = {await foo(): 1}", "foo") == "x = {foo(): 1}"

    def test_inside_dict_value(self):
        assert apply_removal("x = {1: await foo()}", "foo") == "x = {1: foo()}"

    def test_inside_set_literal(self):
        assert apply_removal("x = {await foo()}", "foo") == "x = {foo()}"

    def test_inside_tuple(self):
        assert apply_removal("x = (await foo(),)", "foo") == "x = (foo(),)"

    def test_inside_f_string_expression(self):
        assert apply_removal("x = f'{await foo()}'", "foo") == "x = f'{foo()}'"

    def test_as_function_argument(self):
        assert apply_removal("print(await foo())", "foo") == "print(foo())"

    def test_ternary_expression(self):
        assert apply_removal("x = await foo() if True else None", "foo") == "x = foo() if True else None"

    def test_comparison(self):
        assert apply_removal("if await foo() == 1:", "foo") == "if foo() == 1:"

    def test_negation(self):
        assert apply_removal("if not await foo():", "foo") == "if not foo():"


# ============================================================================
# MULTILINE CONTEXT (LINE-BY-LINE APPLICATION)
# ============================================================================


class TestAwaitRemovalMultiline:
    """The regex applies per-line. Test that the await line matches correctly
    even when the call spans multiple lines."""

    def test_await_and_func_on_same_line(self):
        """When await and foo( are on the same line, it matches."""
        assert apply_removal("result = await foo(", "foo") == "result = foo("

    def test_await_alone_on_line(self):
        """If 'await' is alone (no function name follows), no match."""
        assert apply_removal("result = await", "foo") == "result = await"

    def test_continuation_line(self):
        """Continuation line with just args — no await to remove."""
        assert apply_removal("    x, y)", "foo") == "    x, y)"


# ============================================================================
# SPECIAL FUNCTION NAMES
# ============================================================================


class TestAwaitRemovalSpecialNames:
    """Function names with underscores, numbers, etc."""

    def test_underscore_name(self):
        assert apply_removal("await _private_func()", "_private_func") == "_private_func()"

    def test_dunder_name(self):
        assert apply_removal("await __init__()", "__init__") == "__init__()"

    def test_name_with_numbers(self):
        assert apply_removal("await func123()", "func123") == "func123()"

    def test_single_char_name(self):
        assert apply_removal("await f()", "f") == "f()"

    def test_long_name(self):
        name = "very_long_function_name_that_goes_on"
        assert apply_removal(f"await {name}()", name) == f"{name}()"

    def test_name_starting_with_underscore_and_number(self):
        assert apply_removal("await _f2()", "_f2") == "_f2()"


# ============================================================================
# PATTERN CONSTANT INTEGRITY
# ============================================================================


class TestAwaitRemovalPatternConstant:
    """Verify the pattern constant itself."""

    def test_pattern_is_a_string(self):
        assert isinstance(AWAIT_REMOVAL_PATTERN, str)

    def test_pattern_has_func_name_placeholder(self):
        assert "{func_name}" in AWAIT_REMOVAL_PATTERN

    def test_pattern_compiles_after_formatting(self):
        """Formatted pattern should be valid regex."""
        formatted = build_pattern("foo")
        compiled = re.compile(formatted)
        assert compiled is not None

    def test_pattern_with_regex_special_chars_in_name(self):
        """re.escape should handle hypothetical special chars safely.
        Python identifiers can't have these, but re.escape shouldn't break."""
        formatted = build_pattern("foo.bar")
        compiled = re.compile(formatted)
        assert compiled is not None


# ============================================================================
# KNOWN LIMITATIONS (documented behavior)
# ============================================================================


class TestAwaitRemovalKnownLimitations:
    """These document known regex limitations. The tests assert the ACTUAL
    behavior, not the ideal behavior. This ensures we're aware of what
    the regex does in edge cases."""

    def test_await_inside_string_literal_matches(self):
        """Regex cannot distinguish code from string contents.
        This is acceptable because the analyzer won't flag string lines as call sites."""
        line = 'x = "await foo()"'
        result = apply_removal(line, "foo")
        # The regex WILL match inside the string — this is the known limitation
        assert result == 'x = "foo()"'

    def test_await_inside_comment_matches(self):
        """Regex cannot distinguish code from comments.
        Acceptable because the analyzer won't flag comment lines as call sites."""
        line = "# await foo()"
        result = apply_removal(line, "foo")
        assert result == "# foo()"

    def test_await_no_space_no_match(self):
        """'await(foo())' with no space — not valid Python await syntax,
        and \\s+ requires at least one space."""
        assert apply_removal("await(foo())", "foo") == "await(foo())"
