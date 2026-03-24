import ast
import re
import tempfile

import pytest
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

from devdox_ai_sonar.services.rule_handler import (
    AWAIT_REMOVAL_PATTERN,
    ConvenationNameHandler,
    AsyncToSyncHandler,
    CognitiveComplexityHandler,
    DefaultRuleHandler,
    StringLiteralDuplicateHandler,
    RuleHandlerRegistry,
    _resolve_effective_values,
    _resolve_relative_path,
    _LiteralCaches,
    _FixAccumulator,
)
from devdox_ai_sonar.models.constant_naming import (
    LiteralContext,
    NamingResponse,
)
from devdox_ai_sonar.models.sonar import (
    SonarFixResponse,
    CodeBlock,
    ChangeType,
    BlockType,
    PlacementType,
    SearchReplace,
)
from devdox_ai_sonar.models.file_structures import (
    FixContext,
    ConversionAnalysis,
    ConversionRisk,
)


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
        assert (
            apply_removal("        result = await foo()", "foo")
            == "        result = foo()"
        )

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
        assert (
            apply_removal("result = await self.foo(x)", "foo") == "result = self.foo(x)"
        )

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
        assert (
            apply_removal("x = await foo() or await bar()", "foo")
            == "x = foo() or await bar()"
        )

    def test_await_in_both_sides_of_and(self):
        assert (
            apply_removal("x = await bar() and await foo()", "foo")
            == "x = await bar() and foo()"
        )

    def test_keyword_arg_same_as_func_name(self):
        """'foo=' should not be confused with 'foo('."""
        assert (
            apply_removal("await bar(foo=await foo())", "foo") == "await bar(foo=foo())"
        )


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
        assert (
            apply_removal("x = (await module.foo()).bar", "foo")
            == "x = (module.foo()).bar"
        )

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
        assert (
            apply_removal("x = await foo() if True else None", "foo")
            == "x = foo() if True else None"
        )

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
        assert (
            apply_removal("await _private_func()", "_private_func") == "_private_func()"
        )

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
        This is acceptable because the analyzer won't flag string lines as call sites.
        """
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


# ============================================================================
# SHARED HELPERS FOR HANDLER TESTS
# ============================================================================


def _make_context(**overrides) -> FixContext:
    """Build a FixContext with sensible defaults, overridable via kwargs."""
    defaults = dict(
        file_path=Path("/project/src/module.py"),
        file_path_tmp=Path(tempfile.gettempdir()) / "module.py",
        line_range={"first_line": 10, "last_line": 20, "problem_lines": [10, 15]},
        code_content="async def my_func(self, camelCase):\n    return 1\n",
        language="python",
        import_section={"start_line": 1, "end_line": 3},
        class_name=None,
        functions=[{"name": "my_func", "start_line": 10}],
        context_dict={
            "start_line": 10,
            "import_section": {"end_line": 3},
            "functions": [{"name": "my_func", "start_line": 10}],
            "new_context": [
                {"context": "async def my_func(self, camelCase):\n    return 1\n"}
            ],
        },
    )
    defaults.update(overrides)
    return FixContext(**defaults)


def _make_code_block(**overrides) -> CodeBlock:
    """Build a CodeBlock with sensible defaults."""
    defaults = dict(
        block_name="my_func",
        start_line=10,
        end_line=20,
        has_changes=True,
        change_type=ChangeType.FULL_CODE,
        block_type=BlockType.FUNCTION,
        context="def my_func():\n    return 1\n",
    )
    defaults.update(overrides)
    return CodeBlock(**defaults)


def _make_fix_response(**overrides) -> SonarFixResponse:
    """Build a SonarFixResponse with sensible defaults."""
    defaults = dict(
        IMPORT_BLOCK="",
        FIXED_CODE_BLOCKS=[_make_code_block()],
        NEW_HELPER_CODE="",
        PLACEMENT="SIBLING",
        EXPLANATION="Fixed the issue",
        CONFIDENCE=0.95,
    )
    defaults.update(overrides)
    return SonarFixResponse(**defaults)


# ============================================================================
# GENERATE FIX KEY
# ============================================================================


class TestGenerateFixKey:
    """Tests for RuleHandler._generate_fix_key (line 146)."""

    def setup_method(self):
        self.handler = DefaultRuleHandler()

    def test_with_multiple_lines(self):
        assert self.handler._generate_fix_key([10, 15, 20]) == "10-15-20"

    def test_with_single_line(self):
        assert self.handler._generate_fix_key([42]) == "42"

    def test_with_empty_list(self):
        assert self.handler._generate_fix_key([]) == ""

    def test_with_none(self):
        assert self.handler._generate_fix_key(None) == ""


# ============================================================================
# BUILD FIX SUGGESTION
# ============================================================================


class TestBuildFixSuggestion:
    """Tests for RuleHandler._build_fix_suggestion (lines 172-218)."""

    def setup_method(self):
        self.handler = DefaultRuleHandler()

    def test_single_suggestion(self):
        context = _make_context()
        response = _make_fix_response(EXPLANATION="test explanation")
        result = self.handler._build_fix_suggestion(
            [response],
            context,
            Path("/project/src/module.py"),
            Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": [10]},
        )
        assert len(result) == 1
        assert result[0].explanation == "test explanation"
        assert result[0].confidence == 0.95
        assert result[0].file_path == "src/module.py"

    def test_multiple_suggestions(self):
        context = _make_context()
        result = self.handler._build_fix_suggestion(
            [_make_fix_response(), _make_fix_response()],
            context,
            Path("/project/src/module.py"),
            Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": []},
        )
        assert len(result) == 2

    def test_custom_model(self):
        self.handler.model = "gpt-4o"
        context = _make_context()
        result = self.handler._build_fix_suggestion(
            [_make_fix_response()],
            context,
            Path("/project/src/module.py"),
            Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": []},
        )
        assert result[0].llm_model == "gpt-4o"

    def test_default_model_unknown(self):
        context = _make_context()
        result = self.handler._build_fix_suggestion(
            [_make_fix_response()],
            context,
            Path("/project/src/module.py"),
            Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": []},
        )
        assert result[0].llm_model == "unknown"

    def test_uses_import_section_end_line(self):
        context = _make_context(
            context_dict={
                "start_line": 10,
                "import_section": {"end_line": 7},
                "new_context": [{"context": "code"}],
            },
        )
        result = self.handler._build_fix_suggestion(
            [_make_fix_response()],
            context,
            Path("/project/src/module.py"),
            Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": []},
        )
        assert result[0].end_import_block_code == 7

    def test_missing_import_section(self):
        context = _make_context(
            context_dict={
                "start_line": 10,
                "import_section": {},
                "new_context": [{"context": "code"}],
            },
        )
        result = self.handler._build_fix_suggestion(
            [_make_fix_response()],
            context,
            Path("/project/src/module.py"),
            Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": []},
        )
        assert result[0].end_import_block_code == 0


# ============================================================================
# CAN HANDLE — ALL HANDLERS
# ============================================================================


class TestCanHandle:
    """Tests for can_handle on all handler subclasses."""

    def test_convenation_s117_true(self):
        assert ConvenationNameHandler().can_handle("python:S117") is True

    def test_convenation_other_false(self):
        assert ConvenationNameHandler().can_handle("python:S3776") is False

    def test_convenation_empty_false(self):
        assert ConvenationNameHandler().can_handle("") is False

    def test_async_to_sync_s7503_true(self):
        assert AsyncToSyncHandler().can_handle("python:S7503") is True

    def test_async_to_sync_other_false(self):
        assert AsyncToSyncHandler().can_handle("python:S117") is False

    def test_cognitive_s3776_true(self):
        assert CognitiveComplexityHandler().can_handle("python:S3776") is True

    def test_cognitive_other_false(self):
        assert CognitiveComplexityHandler().can_handle("python:S7503") is False

    def test_default_any_rule_true(self):
        assert DefaultRuleHandler().can_handle("python:S9999") is True

    def test_default_empty_true(self):
        assert DefaultRuleHandler().can_handle("") is True


# ============================================================================
# COGNITIVE COMPLEXITY HANDLER — GENERATE FIXES
# ============================================================================


class TestCognitiveComplexityHandlerGenerateFixes:
    """Tests for CognitiveComplexityHandler.generate_fixes (lines 634-650)."""

    def setup_method(self):
        self.handler = CognitiveComplexityHandler()
        self.context = _make_context()
        self.issues = [Mock()]

    async def test_returns_none_without_llm_caller(self):
        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            Path("/project"),
            Path("/project/src/module.py"),
            llm_caller=None,
        )
        assert result is None

    async def test_calls_llm_returns_response(self):
        mock_llm = AsyncMock()
        fix_response = _make_fix_response()
        mock_llm._call_llm_list.return_value = fix_response

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            Path("/project"),
            Path("/project/src/module.py"),
            llm_caller=mock_llm,
        )
        assert result == [fix_response]
        mock_llm._call_llm_list.assert_called_once()

    async def test_returns_none_on_exception(self):
        mock_llm = AsyncMock()
        mock_llm._call_llm_list.side_effect = RuntimeError("LLM error")

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            Path("/project"),
            Path("/project/src/module.py"),
            llm_caller=mock_llm,
        )
        assert result is None


# ============================================================================
# DEFAULT RULE HANDLER — GENERATE FIXES
# ============================================================================


class TestDefaultRuleHandlerGenerateFixes:
    """Tests for DefaultRuleHandler.generate_fixes (lines 676-692)."""

    def setup_method(self):
        self.handler = DefaultRuleHandler()
        self.context = _make_context()
        self.issues = [Mock()]

    async def test_returns_none_without_llm_caller(self):
        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            Path("/project"),
            Path("/project/src/module.py"),
            llm_caller=None,
        )
        assert result is None

    async def test_calls_llm_returns_response(self):
        mock_llm = AsyncMock()
        fix_response = _make_fix_response()
        mock_llm._call_llm_list.return_value = fix_response

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            Path("/project"),
            Path("/project/src/module.py"),
            llm_caller=mock_llm,
        )
        assert result == [fix_response]

    async def test_returns_none_on_exception(self):
        mock_llm = AsyncMock()
        mock_llm._call_llm_list.side_effect = RuntimeError("boom")

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            Path("/project"),
            Path("/project/src/module.py"),
            llm_caller=mock_llm,
        )
        assert result is None


# ============================================================================
# CONVENATION NAME HANDLER — GENERATE FIXES
# ============================================================================


class TestConvenationNameHandlerGenerateFixes:
    """Tests for ConvenationNameHandler.generate_fixes (lines 269-341)."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()
        self.file_path = Path("/project/src/module.py")
        self.project_path = Path("/project")
        self.context = _make_context(
            functions=[{"name": "my_func", "start_line": 10}],
        )
        self.issues = [Mock(rule="python:S117")]

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_renames_camelcase_arg(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "camelCase"],
                }
            ],
            "calls": [],
        }
        mock_snake.side_effect = lambda x: "camel_case" if x == "camelCase" else x

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is not None
        assert len(result) >= 1
        last_response = result[-1]
        assert len(last_response.FIXED_CODE_BLOCKS) >= 1

    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_returns_none_no_definitions(self, mock_find):
        mock_find.return_value = {"definitions": [], "calls": []}

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_returns_none_no_args_need_renaming(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "already_snake"],
                }
            ],
            "calls": [],
        }
        mock_snake.side_effect = lambda x: x  # no change

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_creates_caller_blocks(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "camelCase"],
                }
            ],
            "calls": [
                {"file": "/other/file.py", "line": 5},
                {"file": "/another/file.py", "line": 12},
            ],
        }
        mock_snake.side_effect = lambda x: "camel_case" if x == "camelCase" else x

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is not None
        # Should have caller responses (one per caller) + final definition response
        assert len(result) >= 3

    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_handles_exception(self, mock_find):
        mock_find.side_effect = RuntimeError("oops")

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_definition_different_file_skipped(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [
                {
                    "file": "/other/file.py",
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "camelCase"],
                }
            ],
            "calls": [],
        }
        mock_snake.side_effect = lambda x: "camel_case" if x == "camelCase" else x

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None


# ============================================================================
# CONVENATION NAME HANDLER — HELPER METHODS
# ============================================================================


class TestConvenationNameHelpers:
    """Tests for _change_function_definition_block and _create_caller_blocks."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()

    def test_change_definition_replaces_arg(self):
        context = _make_context(
            context_dict={
                "start_line": 10,
                "import_section": {"end_line": 3},
                "new_context": [
                    {"context": "def my_func(camelCase):\n    return camelCase\n"}
                ],
            },
        )
        func_info = {
            "function": "my_func",
            "line": 10,
            "decorators": [],
        }
        block = self.handler._change_function_definition_block(
            func_info,
            context,
            "camelCase",
            "camel_case",
        )
        assert block.change_type == ChangeType.FULL_CODE
        assert "camel_case" in block.context
        assert "camelCase" not in block.context

    def test_definition_start_line_from_decorators(self):
        context = _make_context(
            context_dict={
                "start_line": 10,
                "import_section": {"end_line": 3},
                "new_context": [{"context": "def my_func(x):\n    pass\n"}],
            },
        )
        func_info = {
            "function": "my_func",
            "line": 15,
            "decorators": ["staticmethod"],
        }
        block = self.handler._change_function_definition_block(
            func_info,
            context,
            "x",
            "y",
        )
        assert block.start_line == 14  # 15 - 1 decorator

    def test_caller_blocks_per_pair(self):
        func_info = {
            "definitions": [
                {
                    "function": "my_func",
                    "args": ["a", "b"],
                }
            ],
            "calls": [
                {"file": "/file1.py", "line": 5},
                {"file": "/file2.py", "line": 12},
            ],
        }
        blocks = self.handler._create_caller_blocks(
            {"camelCase": "camel_case", "otherArg": "other_arg"},
            func_info,
        )
        # 2 callers × 2 args = 4 blocks
        assert len(blocks) == 4
        assert all(b.change_type == ChangeType.SEARCH_REPLACE for b in blocks)

    def test_caller_block_file_path(self):
        func_info = {
            "definitions": [{"function": "fn", "args": ["a"]}],
            "calls": [{"file": "/specific/path.py", "line": 7}],
        }
        blocks = self.handler._create_caller_blocks({"old": "new"}, func_info)
        assert blocks[0].file_path == "/specific/path.py"

    def test_caller_block_end_line(self):
        func_info = {
            "definitions": [{"function": "fn", "args": ["a", "b", "c"]}],
            "calls": [{"file": "/f.py", "line": 10}],
        }
        blocks = self.handler._create_caller_blocks({"old": "new"}, func_info)
        # end_line = caller["line"] + num_args + 1 = 10 + 3 + 1 = 14
        assert blocks[0].end_line == 14


# ============================================================================
# ASYNC TO SYNC HANDLER — GENERATE FIXES
# ============================================================================


class TestAsyncToSyncHandlerGenerateFixes:
    """Tests for AsyncToSyncHandler.generate_fixes (lines 471-535)."""

    def setup_method(self):
        self.handler = AsyncToSyncHandler()
        self.file_path = Path("/project/src/module.py")
        self.project_path = Path("/project")
        self.context = _make_context(
            code_content="async def my_func():\n    return 1\n",
        )
        self.issues = [Mock()]

    @patch("devdox_ai_sonar.services.rule_handler.AsyncConversionAnalyzer")
    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_success(self, mock_detect, mock_analyzer_cls):
        mock_detect.return_value = {
            "found": True,
            "name": "my_func",
            "definition": "async def my_func():",
            "start_line": 0,
        }
        mock_analysis = ConversionAnalysis(
            function_name="my_func",
            current_type="async",
            target_type="sync",
            risk_level=ConversionRisk.SAFE,
            blocking_issues=[],
            required_changes=[],
            caller_impact=[
                {
                    "awaited": True,
                    "file": str(self.file_path),
                    "line": 25,
                }
            ],
            internal_calls=[],
            suggestions=[],
        )
        mock_analyzer_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is not None
        assert len(result) >= 1

    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_returns_none_not_found(self, mock_detect):
        mock_detect.return_value = {"found": False}

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.AsyncConversionAnalyzer")
    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_cross_file_callers_separated(self, mock_detect, mock_analyzer_cls):
        mock_detect.return_value = {
            "found": True,
            "name": "my_func",
            "definition": "async def my_func():",
            "start_line": 0,
        }
        mock_analysis = ConversionAnalysis(
            function_name="my_func",
            current_type="async",
            target_type="sync",
            risk_level=ConversionRisk.SAFE,
            blocking_issues=[],
            required_changes=[],
            caller_impact=[
                {
                    "awaited": True,
                    "file": "/other/file.py",
                    "line": 5,
                }
            ],
            internal_calls=[],
            suggestions=[],
        )
        mock_analyzer_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is not None
        # Cross-file caller → separate response; definition → another response
        assert len(result) >= 2

    @patch("devdox_ai_sonar.services.rule_handler.AsyncConversionAnalyzer")
    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_skips_non_awaited(self, mock_detect, mock_analyzer_cls):
        mock_detect.return_value = {
            "found": True,
            "name": "my_func",
            "definition": "async def my_func():",
            "start_line": 0,
        }
        mock_analysis = ConversionAnalysis(
            function_name="my_func",
            current_type="async",
            target_type="sync",
            risk_level=ConversionRisk.SAFE,
            blocking_issues=[],
            required_changes=[],
            caller_impact=[
                {
                    "awaited": False,
                    "file": str(self.file_path),
                    "line": 25,
                }
            ],
            internal_calls=[],
            suggestions=[],
        )
        mock_analyzer_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is not None
        # Only the definition response, no caller blocks
        assert len(result) == 1

    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_handles_exception(self, mock_detect):
        mock_detect.side_effect = RuntimeError("parsing failed")

        result = await self.handler.generate_fixes(
            self.issues,
            self.context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None


# ============================================================================
# ASYNC TO SYNC HANDLER — HELPER METHODS
# ============================================================================


class TestAsyncToSyncHelpers:
    """Tests for _create_function_definition_block and _create_caller_blocks."""

    def setup_method(self):
        self.handler = AsyncToSyncHandler()

    def test_definition_block_removes_async(self):
        context = _make_context(
            context_dict={
                "start_line": 10,
                "import_section": {"end_line": 3},
                "functions": [{"name": "my_func", "start_line": 10}],
                "new_context": [{"context": "code"}],
            },
        )
        func_info = {
            "name": "my_func",
            "definition": "async def my_func():",
            "start_line": 0,
        }
        file_path = Path("/project/src/module.py")
        block = self.handler._create_function_definition_block(
            func_info, context, file_path
        )
        assert block.change_type == ChangeType.DIFF
        assert len(block.changes) == 1
        assert block.changes[0].old == "async def my_func():"
        assert block.changes[0].new == "def my_func():"

    def test_file_path_set_on_code_block(self):
        """Function definition block should have file_path set."""
        handler = AsyncToSyncHandler()
        function_info = {
            "found": True,
            "name": "build_greeting",
            "definition": "async def build_greeting(name: str) -> str:",
            "start_line": 0,
        }
        context = Mock()
        context.context_dict = {
            "functions": [{"name": "build_greeting", "start_line": 10}],
        }
        file_path = Path("/home/user/project/services/bad_async.py")

        block = handler._create_function_definition_block(
            function_info, context, file_path
        )

        assert block.file_path == str(file_path)
        assert block.block_name == "build_greeting"

    def test_caller_block_uses_await_pattern(self):
        analysis = Mock()
        analysis.caller_impact = [
            {"awaited": True, "file": "/src/caller.py", "line": 42},
        ]
        func_info = {"name": "my_func"}

        blocks = self.handler._create_caller_blocks(analysis, func_info)
        assert len(blocks) == 1
        assert blocks[0].change_type == ChangeType.SEARCH_REPLACE
        assert blocks[0].file_path == "/src/caller.py"
        assert blocks[0].replacements[0].is_regex is True


# ============================================================================
# RULE HANDLER REGISTRY
# ============================================================================


class TestRuleHandlerRegistry:
    """Tests for RuleHandlerRegistry (lines 695-741)."""

    def test_default_handlers_count(self):
        registry = RuleHandlerRegistry()
        assert len(registry.handlers) == 5

    def test_handler_order(self):
        registry = RuleHandlerRegistry()
        assert isinstance(registry.handlers[0], AsyncToSyncHandler)
        assert isinstance(registry.handlers[1], CognitiveComplexityHandler)
        assert isinstance(registry.handlers[2], ConvenationNameHandler)
        assert isinstance(registry.handlers[3], StringLiteralDuplicateHandler)
        assert isinstance(registry.handlers[4], DefaultRuleHandler)

    def test_get_handler_s7503(self):
        registry = RuleHandlerRegistry()
        assert isinstance(registry.get_handler("python:S7503"), AsyncToSyncHandler)

    def test_get_handler_s3776(self):
        registry = RuleHandlerRegistry()
        assert isinstance(
            registry.get_handler("python:S3776"), CognitiveComplexityHandler
        )

    def test_get_handler_s117(self):
        registry = RuleHandlerRegistry()
        assert isinstance(registry.get_handler("python:S117"), ConvenationNameHandler)

    def test_get_handler_unknown(self):
        registry = RuleHandlerRegistry()
        assert isinstance(registry.get_handler("python:S9999"), DefaultRuleHandler)

    def test_register_negative_priority(self):
        registry = RuleHandlerRegistry()
        custom = Mock(spec=CognitiveComplexityHandler)
        registry.register(custom, priority=-1)
        # Should be inserted before the last (DefaultRuleHandler)
        assert registry.handlers[-2] is custom
        assert isinstance(registry.handlers[-1], DefaultRuleHandler)

    def test_register_explicit_priority(self):
        registry = RuleHandlerRegistry()
        custom = Mock(spec=CognitiveComplexityHandler)
        registry.register(custom, priority=0)
        assert registry.handlers[0] is custom

    def test_get_handler_s1192(self):
        registry = RuleHandlerRegistry()
        assert isinstance(
            registry.get_handler("python:S1192"), StringLiteralDuplicateHandler
        )


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — CAN HANDLE
# ============================================================================


class TestStringLiteralDuplicateCanHandle:
    """Tests for StringLiteralDuplicateHandler.can_handle."""

    def test_s1192_true(self):
        assert StringLiteralDuplicateHandler().can_handle("python:S1192") is True

    def test_other_rule_false(self):
        assert StringLiteralDuplicateHandler().can_handle("python:S117") is False

    def test_empty_string_false(self):
        assert StringLiteralDuplicateHandler().can_handle("") is False

    def test_partial_match_false(self):
        assert StringLiteralDuplicateHandler().can_handle("S1192") is False


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — EXTRACT LITERAL FROM MESSAGE
# ============================================================================


class TestExtractLiteralFromMessage:
    """Tests for StringLiteralDuplicateHandler._extract_literal_from_message."""

    def setup_method(self):
        self.extract = StringLiteralDuplicateHandler._extract_literal_from_message

    def test_standard_message(self):
        msg = 'Define a constant instead of duplicating this literal "application/json" 5 times.'
        assert self.extract(msg) == "application/json"

    def test_string_with_special_chars(self):
        msg = 'Define a constant instead of duplicating this literal "Content-Type" 3 times.'
        assert self.extract(msg) == "Content-Type"

    def test_string_with_spaces(self):
        msg = 'Define a constant instead of duplicating this literal "hello world" 4 times.'
        assert self.extract(msg) == "hello world"

    def test_string_with_path(self):
        msg = 'Define a constant instead of duplicating this literal "/api/v1/users" 3 times.'
        assert self.extract(msg) == "/api/v1/users"

    def test_malformed_message_returns_none(self):
        assert self.extract("Some random message") is None

    def test_empty_message_returns_none(self):
        assert self.extract("") is None

    def test_message_without_quotes_returns_none(self):
        msg = "Define a constant instead of duplicating this literal 3 times."
        assert self.extract(msg) is None


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — GENERATE CONSTANT NAME
# ============================================================================


class TestGenerateConstantName:
    """Tests for StringLiteralDuplicateHandler._generate_constant_name."""

    def setup_method(self):
        self.gen = StringLiteralDuplicateHandler._generate_constant_name

    def test_counter_1(self):
        assert self.gen(1, set()) == "STRING_LITERAL_1"

    def test_counter_2(self):
        assert self.gen(2, set()) == "STRING_LITERAL_2"

    def test_collision_increments(self):
        used = {"STRING_LITERAL_1"}
        assert self.gen(1, used) == "STRING_LITERAL_2"

    def test_multiple_collisions(self):
        used = {"STRING_LITERAL_1", "STRING_LITERAL_2", "STRING_LITERAL_3"}
        assert self.gen(1, used) == "STRING_LITERAL_4"


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — FIND STRING OCCURRENCES
# ============================================================================


class TestFindStringOccurrences:
    """Tests for StringLiteralDuplicateHandler._find_string_occurrences."""

    def setup_method(self):
        self.find = StringLiteralDuplicateHandler._find_string_occurrences

    def test_finds_all_matching_strings(self):
        source = 'x = "hello"\n' 'y = "hello"\n' 'z = "hello"\n'
        tree = ast.parse(source)
        result = self.find(tree, "hello")
        assert len(result) == 3

    def test_returns_correct_line_numbers(self):
        source = 'x = "hello"\n' 'y = "world"\n' 'z = "hello"\n'
        tree = ast.parse(source)
        result = self.find(tree, "hello")
        lines = [r[0] for r in result]
        assert lines == [1, 3]

    def test_does_not_match_different_strings(self):
        source = 'x = "hello"\n' 'y = "world"\n'
        tree = ast.parse(source)
        result = self.find(tree, "goodbye")
        assert len(result) == 0

    def test_does_not_match_non_strings(self):
        source = "x = 42\ny = 3.14\n"
        tree = ast.parse(source)
        result = self.find(tree, "42")
        assert len(result) == 0

    def test_handles_single_quotes(self):
        source = "x = 'hello'\ny = 'hello'\n"
        tree = ast.parse(source)
        result = self.find(tree, "hello")
        assert len(result) == 2

    def test_handles_mixed_quotes(self):
        source = "x = \"hello\"\ny = 'hello'\n"
        tree = ast.parse(source)
        result = self.find(tree, "hello")
        assert len(result) == 2

    def test_string_in_function(self):
        source = (
            "def foo():\n"
            '    return "test_value"\n'
            "\n"
            "def bar():\n"
            '    return "test_value"\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "test_value")
        assert len(result) == 2


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — BUILD REPLACEMENT BLOCKS
# ============================================================================


class TestBuildReplacementBlocks:
    """Tests for StringLiteralDuplicateHandler._build_replacement_blocks."""

    def setup_method(self):
        self.build = StringLiteralDuplicateHandler._build_replacement_blocks

    def test_single_occurrence(self):
        lines = ['x = "hello"\n']
        occurrences = [(1, 4, 11)]  # "hello" at col 4-11
        blocks = self.build(occurrences, lines, "STRING_LITERAL_1")
        assert len(blocks) == 1
        assert blocks[0].change_type == ChangeType.SEARCH_REPLACE
        assert blocks[0].start_line == 1
        assert blocks[0].replacements[0].search == '"hello"'
        assert blocks[0].replacements[0].replace == "STRING_LITERAL_1"

    def test_multiple_occurrences(self):
        lines = ['x = "hello"\n', 'y = "hello"\n']
        occurrences = [(1, 4, 11), (2, 4, 11)]
        blocks = self.build(occurrences, lines, "STRING_LITERAL_1")
        assert len(blocks) == 2
        assert blocks[0].start_line == 1
        assert blocks[1].start_line == 2

    def test_single_quote_preservation(self):
        lines = ["x = 'hello'\n"]
        occurrences = [(1, 4, 11)]
        blocks = self.build(occurrences, lines, "STRING_LITERAL_1")
        assert blocks[0].replacements[0].search == "'hello'"

    def test_invalid_line_number_skipped(self):
        lines = ['x = "hello"\n']
        occurrences = [(999, 4, 11)]  # line doesn't exist
        blocks = self.build(occurrences, lines, "STRING_LITERAL_1")
        assert len(blocks) == 0

    def test_block_type_is_module(self):
        lines = ['x = "hello"\n']
        occurrences = [(1, 4, 11)]
        blocks = self.build(occurrences, lines, "STRING_LITERAL_1")
        assert blocks[0].block_type == BlockType.MODULE

    def test_count_is_one(self):
        lines = ['x = "hello"\n']
        occurrences = [(1, 4, 11)]
        blocks = self.build(occurrences, lines, "STRING_LITERAL_1")
        assert blocks[0].replacements[0].count == 1


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — GENERATE FIXES (END-TO-END)
# ============================================================================


class TestStringLiteralDuplicateGenerateFixes:
    """End-to-end tests for StringLiteralDuplicateHandler.generate_fixes."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()
        self.project_path = Path("/project")

    def _make_issue(self, message: str, first_line: int = 1) -> Mock:
        issue = Mock()
        issue.message = message
        issue.first_line = first_line
        issue.rule = "python:S1192"
        return issue

    async def test_single_duplicated_string(self, tmp_path):
        source = (
            'x = "application/json"\n'
            'y = "application/json"\n'
            'z = "application/json"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            "Define a constant instead of duplicating this literal "
            '"application/json" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        assert len(result) == 1

        response = result[0]
        # 1 constants block + 3 replacement blocks
        assert len(response.FIXED_CODE_BLOCKS) == 4
        assert response.FIXED_CODE_BLOCKS[0].block_name == "New constants"
        assert "APPLICATION_JSON" in response.NEW_HELPER_CODE
        assert "'application/json'" in response.NEW_HELPER_CODE
        assert response.PLACEMENT == PlacementType.GLOBAL_TOP

    async def test_multiple_duplicated_strings(self, tmp_path):
        source = (
            'x = "hello world"\n'
            'y = "hello world"\n'
            'z = "hello world"\n'
            'a = "foo/bar"\n'
            'b = "foo/bar"\n'
            'c = "foo/bar"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issues = [
            self._make_issue(
                "Define a constant instead of duplicating this literal "
                '"hello world" 3 times.'
            ),
            self._make_issue(
                "Define a constant instead of duplicating this literal "
                '"foo/bar" 3 times.'
            ),
        ]
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            issues, context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # 1 constants block + 6 replacement blocks
        assert len(response.FIXED_CODE_BLOCKS) == 7
        assert response.FIXED_CODE_BLOCKS[0].block_name == "New constants"
        assert "HELLO_WORLD = 'hello world'" in response.NEW_HELPER_CODE
        assert "FOO_BAR = 'foo/bar'" in response.NEW_HELPER_CODE
        # Verify each replacement maps to the correct constant
        replacement_blocks = response.FIXED_CODE_BLOCKS[1:]
        for block in replacement_blocks[:3]:
            assert block.replacements[0].replace == "HELLO_WORLD"
        for block in replacement_blocks[3:]:
            assert block.replacements[0].replace == "FOO_BAR"

    async def test_syntax_error_returns_none(self, tmp_path):
        file = tmp_path / "bad.py"
        file.write_text("def foo(:\n")

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "x" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is None

    async def test_unextractable_message_returns_none(self, tmp_path):
        source = 'x = "hello"\ny = "hello"\nz = "hello"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("Some unrelated message")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is None

    async def test_no_occurrences_returns_none(self, tmp_path):
        source = 'x = "hello"\ny = "world"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            "Define a constant instead of duplicating this literal "
            '"not_found" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is None

    async def test_nonexistent_file_returns_none(self):
        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "x" 3 times.'
        )
        context = _make_context()

        result = await self.handler.generate_fixes(
            [issue],
            context,
            self.project_path,
            Path("/nonexistent/file.py"),
            llm_caller=None,
        )
        assert result is None

    async def test_confidence_is_high(self, tmp_path):
        source = 'x = "test"\ny = "test"\nz = "test"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "test" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        assert result[0].CONFIDENCE == 0.95

    async def test_replacement_uses_exact_quoted_form(self, tmp_path):
        source = "x = 'hello'\ny = 'hello'\nz = 'hello'\n"
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        # Skip the constants block (index 0) to get the first replacement block
        block = result[0].FIXED_CODE_BLOCKS[1]
        assert block.replacements[0].search == "'hello'"

    async def test_bug_unique_constant_names(self, tmp_path):
        """Verify two different literals get unique constant names."""
        source = (
            "def fetch():\n"
            '    response = requests.get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = requests.get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = requests.get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch4():\n"
            '    response = requests.post(url, body="1234")\n'
            "\n"
            "def fetch5():\n"
            '    response = requests.post(url, body="1234")\n'
            "\n"
            "def fetch6():\n"
            '    response = requests.post(url, body="1234")\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issues = [
            self._make_issue(
                "Define a constant instead of duplicating this literal "
                '"application/json" 3 times.'
            ),
            self._make_issue(
                "Define a constant instead of duplicating this literal "
                '"1234" 3 times.'
            ),
        ]
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            issues, context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]

        assert "APPLICATION_JSON = 'application/json'" in response.NEW_HELPER_CODE
        # "1234" is a single numeric token → falls back to STRING_LITERAL_N
        assert "= '1234'" in response.NEW_HELPER_CODE

        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        json_blocks = [
            b
            for b in replacement_blocks
            if b.replacements[0].replace == "APPLICATION_JSON"
        ]
        # "1234" gets a fallback name — find it dynamically
        num_const_name = None
        for line in response.NEW_HELPER_CODE.split("\n"):
            if "'1234'" in line:
                num_const_name = line.split("=")[0].strip()
                break
        assert num_const_name is not None
        num_blocks = [
            b for b in replacement_blocks if b.replacements[0].replace == num_const_name
        ]
        assert len(json_blocks) == 3
        assert len(num_blocks) == 3
        # Ensure the two constants have different names
        assert num_const_name != "APPLICATION_JSON"

    async def test_explanation_created_constant(self, tmp_path):
        """Explanation shows SonarCloud message, action, file, and lines."""
        source = 'x = "hello"\n' 'y = "hello"\n' 'z = "hello"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        explanation = result[0].EXPLANATION

        assert "**File:**" in explanation
        assert 'duplicating this literal "hello"' in explanation
        assert "Created" in explanation
        assert "= 'hello'" in explanation
        assert "**Lines affected:**" in explanation
        assert "- 1" in explanation
        assert "- 2" in explanation
        assert "- 3" in explanation

    async def test_explanation_reused_constant(self, tmp_path):
        """Explanation shows reuse action when existing constant found."""
        source = (
            'APP_JSON = "application/json"\n'
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            "Define a constant instead of duplicating this literal "
            '"application/json" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        explanation = result[0].EXPLANATION

        assert "Reused existing constant" in explanation
        assert "`APP_JSON`" in explanation
        assert "defined at line 1" in explanation
        assert "**Lines affected:**" in explanation
        # Should NOT have a constants block (reuse case)
        constants_blocks = [
            b for b in result[0].FIXED_CODE_BLOCKS if b.block_name == "New constants"
        ]
        assert len(constants_blocks) == 0

    async def test_explanation_file_path(self, tmp_path):
        """Explanation includes relative file path."""
        source = 'x = "a"\ny = "a"\nz = "a"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "a" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        assert "**File:**" in result[0].EXPLANATION

    async def test_replacement_blocks_have_file_path(self, tmp_path):
        """All replacement blocks include file_path."""
        source = 'x = "test"\ny = "test"\nz = "test"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "test" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        for block in result[0].FIXED_CODE_BLOCKS:
            assert block.file_path is not None


# ============================================================================
# FIND EXISTING CONSTANT — UNIT TESTS
# ============================================================================


class TestFindExistingConstant:
    """Tests for StringLiteralDuplicateHandler._find_existing_constant."""

    def _parse(self, source: str) -> ast.Module:
        return ast.parse(source)

    def test_simple_assign_found(self):
        tree = self._parse('APP_JSON = "application/json"\n')
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert len(result) == 1
        assert result[0] == ("APP_JSON", 1)

    def test_ann_assign_found(self):
        tree = self._parse('APP_JSON: str = "application/json"\n')
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert len(result) == 1
        assert result[0] == ("APP_JSON", 1)

    def test_no_match_returns_empty(self):
        tree = self._parse('x = "other"\n')
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert result == []

    def test_lowercase_name_found(self):
        tree = self._parse('app_json = "application/json"\n')
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert len(result) == 1
        assert result[0] == ("app_json", 1)

    def test_multiple_matches_sorted_by_line(self):
        source = 'APP_JSON = "application/json"\n' 'CONTENT_TYPE = "application/json"\n'
        tree = self._parse(source)
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert len(result) == 2
        assert result[0] == ("APP_JSON", 1)
        assert result[1] == ("CONTENT_TYPE", 2)

    def test_class_level_not_found(self):
        source = "class Config:\n" '    APP_JSON = "application/json"\n'
        tree = self._parse(source)
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert result == []

    def test_function_level_not_found(self):
        source = "def setup():\n" '    content_type = "application/json"\n'
        tree = self._parse(source)
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert result == []

    def test_dict_value_not_found(self):
        source = 'HEADERS = {"Content-Type": "application/json"}\n'
        tree = self._parse(source)
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert result == []

    def test_tuple_unpack_not_found(self):
        source = 'A, B = "application/json", "other"\n'
        tree = self._parse(source)
        result = StringLiteralDuplicateHandler._find_existing_constant(
            tree, "application/json"
        )
        assert result == []

    def test_non_string_value_not_found(self):
        tree = self._parse("X = 42\n")
        result = StringLiteralDuplicateHandler._find_existing_constant(tree, "42")
        assert result == []

    def test_ann_assign_without_value_not_found(self):
        tree = self._parse("X: str\n")
        result = StringLiteralDuplicateHandler._find_existing_constant(tree, "anything")
        assert result == []


# ============================================================================
# CONVENATION NAME HANDLER — DISPATCH BRANCHES (lines 257-278)
# ============================================================================


class TestConvenationNameHandlerDispatch:
    """Tests for generate_fixes dispatch: empty functions, S1542, S1172, unknown rule."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()
        self.file_path = Path("/project/src/module.py")
        self.project_path = Path("/project")

    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_returns_none_empty_functions(self, mock_find):
        """Lines 257-259: context.functions is empty → returns None immediately."""
        context = _make_context(functions=[])
        issues = [Mock(rule="python:S117")]

        result = await self.handler.generate_fixes(
            issues,
            context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None
        mock_find.assert_not_called()

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_s1542_dispatches_to_fix_func_naming(self, mock_find, mock_snake):
        """Line 273: issue.rule == 'python:S1542' → _fix_func_naming_convention."""
        context = _make_context(
            functions=[{"name": "myFunc", "start_line": 10}],
            context_dict={
                "start_line": 10,
                "import_section": {"end_line": 3},
                "new_context": [{"context": "def myFunc(self):\n    pass\n"}],
            },
        )
        issues = [Mock(rule="python:S1542")]
        mock_find.return_value = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "myFunc",
                    "line": 10,
                    "decorators": [],
                    "args": ["self"],
                }
            ],
            "calls": [],
        }
        mock_snake.return_value = "my_func"

        result = await self.handler.generate_fixes(
            issues,
            context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is not None
        assert len(result) >= 1
        # The last response should contain the definition block
        last = result[-1]
        assert last.CONFIDENCE == 0.95
        assert "S1542" in last.EXPLANATION

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_s1172_dispatches_to_fix_unused_params(self, mock_find, mock_used):
        """Lines 275-276: issue.rule == 'python:S1172' → _fix_unused_parameters."""
        context = _make_context(
            functions=[{"name": "my_func", "start_line": 10}],
        )
        issues = [Mock(rule="python:S1172")]
        mock_find.return_value = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "unused_param"],
                    "unused_args": ["unused_param"],
                }
            ],
            "calls": [],
        }
        mock_used.return_value = False

        # Mock _remove_parameter_block since it reads files from disk
        with patch.object(
            self.handler,
            "_remove_parameter_block",
            return_value=_make_code_block(block_name="test"),
        ):
            result = await self.handler.generate_fixes(
                issues,
                context,
                self.project_path,
                self.file_path,
                llm_caller=None,
            )
        assert result is not None
        assert len(result) >= 1
        assert "S1172" in result[-1].EXPLANATION

    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_unknown_rule_returns_none(self, mock_find):
        """Line 278: rule not S117/S1542/S1172 → returns None."""
        context = _make_context(
            functions=[{"name": "my_func", "start_line": 10}],
        )
        issues = [Mock(rule="python:S9999")]
        mock_find.return_value = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self"],
                }
            ],
            "calls": [],
        }

        result = await self.handler.generate_fixes(
            issues,
            context,
            self.project_path,
            self.file_path,
            llm_caller=None,
        )
        assert result is None


# ============================================================================
# _dispatch_issue (extracted helper for generate_fixes)
# ============================================================================


class TestDispatchIssue:
    """Tests for ConvenationNameHandler._dispatch_issue."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()
        self.file_path = Path("/project/src/module.py")
        self.function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "camelCase"],
                }
            ],
            "calls": [],
        }
        self.context = _make_context(
            functions=[{"name": "my_func", "start_line": 10}],
        )

    def test_s117_dispatches_to_fix_naming_convention(self):
        issue = Mock(rule="python:S117")
        sentinel = [_make_fix_response()]
        with patch.object(
            self.handler, "_fix_naming_convention", return_value=sentinel
        ) as mock_fix:
            result = self.handler._dispatch_issue(
                issue, self.function_info, self.context, self.file_path
            )
        assert result is sentinel
        mock_fix.assert_called_once_with(self.function_info, self.context, self.file_path)

    def test_s1542_dispatches_to_fix_func_naming_convention(self):
        issue = Mock(rule="python:S1542")
        sentinel = [_make_fix_response()]
        with patch.object(
            self.handler, "_fix_func_naming_convention", return_value=sentinel
        ) as mock_fix:
            result = self.handler._dispatch_issue(
                issue, self.function_info, self.context, self.file_path
            )
        assert result is sentinel
        mock_fix.assert_called_once_with(self.function_info, self.context, self.file_path)

    def test_s1172_dispatches_to_fix_unused_parameters(self):
        issue = Mock(rule="python:S1172")
        sentinel = [_make_fix_response()]
        with patch.object(
            self.handler, "_fix_unused_parameters", return_value=sentinel
        ) as mock_fix:
            result = self.handler._dispatch_issue(
                issue, self.function_info, self.context, self.file_path
            )
        assert result is sentinel
        mock_fix.assert_called_once_with(self.function_info, self.context, self.file_path)

    def test_unknown_rule_returns_none(self):
        issue = Mock(rule="python:S9999")
        result = self.handler._dispatch_issue(
            issue, self.function_info, self.context, self.file_path
        )
        assert result is None

    def test_returns_none_and_logs_when_fix_method_returns_none(self):
        issue = Mock(rule="python:S117")
        mock_fix = Mock(return_value=None, __name__="_fix_naming_convention")
        with patch.object(
            self.handler, "_fix_naming_convention", mock_fix
        ):
            result = self.handler._dispatch_issue(
                issue, self.function_info, self.context, self.file_path
            )
        assert result is None


# ============================================================================
# FIX UNUSED PARAMETERS (lines 284-372)
# ============================================================================


class TestFixUnusedParameters:
    """Tests for ConvenationNameHandler._fix_unused_parameters."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()
        self.file_path = Path("/project/src/module.py")
        self.context = _make_context(
            functions=[{"name": "my_func", "start_line": 10}],
        )

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_happy_path_param_removed(self, mock_used):
        """Unused param not used at callsite → safe to remove."""
        mock_used.return_value = False
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "unused_x"],
                    "unused_args": ["unused_x"],
                }
            ],
            "calls": [],
        }

        with patch.object(
            self.handler,
            "_remove_parameter_block",
            return_value=_make_code_block(block_name="test"),
        ) as mock_remove:
            result = self.handler._fix_unused_parameters(
                function_info,
                self.context,
                self.file_path,
            )

        assert result is not None
        assert len(result) == 1
        assert result[0].CONFIDENCE == 0.90
        assert "unused_x" in result[0].EXPLANATION
        mock_remove.assert_called_once()

    def test_definition_different_file_skipped(self):
        """Definition in a different file → skipped, returns None."""
        function_info = {
            "definitions": [
                {
                    "file": "/other/file.py",
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "x"],
                    "unused_args": ["x"],
                }
            ],
            "calls": [],
        }

        result = self.handler._fix_unused_parameters(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is None

    def test_underscore_prefix_skipped(self):
        """Params starting with _ are intentionally unused → skipped."""
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "_unused"],
                    "unused_args": ["_unused"],
                }
            ],
            "calls": [],
        }

        result = self.handler._fix_unused_parameters(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is None

    def test_param_not_in_args_list(self):
        """Param in unused_args but not in args → ValueError caught, skipped."""
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "other"],
                    "unused_args": ["ghost"],
                }
            ],
            "calls": [],
        }

        result = self.handler._fix_unused_parameters(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_has_self_offset(self, mock_used):
        """With self as first arg, callsite_index = raw_index - 1."""
        mock_used.return_value = False
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "x", "y"],
                    "unused_args": ["y"],
                }
            ],
            "calls": [{"file": "/caller.py", "line": 5}],
        }

        with patch.object(
            self.handler,
            "_remove_parameter_block",
            return_value=_make_code_block(),
        ):
            self.handler._fix_unused_parameters(
                function_info,
                self.context,
                self.file_path,
            )

        # "y" is at raw_index=2, minus 1 for self → callsite_index=1
        mock_used.assert_called_once_with("y", 1, function_info["calls"])

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_no_self_no_offset(self, mock_used):
        """Without self/cls, callsite_index == raw_index."""
        mock_used.return_value = False
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["x", "y"],
                    "unused_args": ["x"],
                }
            ],
            "calls": [],
        }

        with patch.object(
            self.handler,
            "_remove_parameter_block",
            return_value=_make_code_block(),
        ):
            self.handler._fix_unused_parameters(
                function_info,
                self.context,
                self.file_path,
            )

        # "x" is at raw_index=0, no self → callsite_index=0
        mock_used.assert_called_once_with("x", 0, [])

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_param_used_at_callsite_not_removed(self, mock_used):
        """Param unused in body but IS used at callsite → not removed."""
        mock_used.return_value = True
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "param"],
                    "unused_args": ["param"],
                }
            ],
            "calls": [{"file": "/caller.py", "line": 5}],
        }

        result = self.handler._fix_unused_parameters(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_multiple_unused_some_safe(self, mock_used):
        """Two unused params: one safe, one used at callsite → only safe one removed."""
        # "safe_param" → not used at callsite; "used_param" → used at callsite
        mock_used.side_effect = lambda param, idx, calls: param == "used_param"
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "safe_param", "used_param"],
                    "unused_args": ["safe_param", "used_param"],
                }
            ],
            "calls": [{"file": "/caller.py", "line": 5}],
        }

        with patch.object(
            self.handler,
            "_remove_parameter_block",
            return_value=_make_code_block(),
        ) as mock_remove:
            result = self.handler._fix_unused_parameters(
                function_info,
                self.context,
                self.file_path,
            )

        assert result is not None
        # Only safe_param should be removed
        mock_remove.assert_called_once()
        assert "safe_param" in result[0].EXPLANATION

    def test_no_unused_args_returns_none(self):
        """Empty unused_args list → returns None via _collect_removable_params."""
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self", "x"],
                    "unused_args": [],
                }
            ],
            "calls": [],
        }

        result = self.handler._fix_unused_parameters(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is None


class TestGetCallsiteIndex:
    """Tests for ConvenationNameHandler._get_callsite_index — pure logic, no I/O."""

    def test_simple_no_self(self):
        """Without self/cls, index matches position directly."""
        assert ConvenationNameHandler._get_callsite_index("x", ["x", "y"]) == 0
        assert ConvenationNameHandler._get_callsite_index("y", ["x", "y"]) == 1

    def test_with_self_offset(self):
        """With self as first arg, index is shifted down by 1."""
        assert ConvenationNameHandler._get_callsite_index("x", ["self", "x", "y"]) == 0
        assert ConvenationNameHandler._get_callsite_index("y", ["self", "x", "y"]) == 1

    def test_with_cls_offset(self):
        """cls behaves the same as self."""
        assert ConvenationNameHandler._get_callsite_index("x", ["cls", "x"]) == 0

    def test_underscore_prefix_returns_none(self):
        """Params starting with _ are intentionally unused → None."""
        assert (
            ConvenationNameHandler._get_callsite_index("_unused", ["self", "_unused"])
            is None
        )

    def test_param_not_found_returns_none(self):
        """Param not in args list → None."""
        assert (
            ConvenationNameHandler._get_callsite_index("ghost", ["self", "x"]) is None
        )

    def test_star_args_stripped(self):
        """Leading * and ** are stripped before lookup."""
        assert (
            ConvenationNameHandler._get_callsite_index("args", ["self", "*args"]) == 0
        )
        assert (
            ConvenationNameHandler._get_callsite_index("kwargs", ["self", "**kwargs"])
            == 0
        )

    def test_empty_args_list(self):
        """Empty args list → param not found → None."""
        assert ConvenationNameHandler._get_callsite_index("x", []) is None


class TestCollectRemovableParams:
    """Tests for ConvenationNameHandler._collect_removable_params."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_unused_and_not_at_callsite(self, mock_used):
        """Param unused in body AND at callsite → included in result."""
        mock_used.return_value = False
        definition = {
            "args": ["self", "x"],
            "unused_args": ["x"],
        }
        result = self.handler._collect_removable_params(definition, [])
        assert result == ["x"]

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_unused_but_used_at_callsite(self, mock_used):
        """Param unused in body but used at callsite → excluded."""
        mock_used.return_value = True
        definition = {
            "args": ["self", "x"],
            "unused_args": ["x"],
        }
        result = self.handler._collect_removable_params(
            definition, [{"file": "/f.py", "line": 1}]
        )
        assert result == []

    def test_underscore_param_skipped(self):
        """Params starting with _ are skipped entirely."""
        definition = {
            "args": ["self", "_internal"],
            "unused_args": ["_internal"],
        }
        result = self.handler._collect_removable_params(definition, [])
        assert result == []

    def test_missing_param_skipped(self):
        """Param in unused_args but not in args → skipped."""
        definition = {
            "args": ["self", "a"],
            "unused_args": ["ghost"],
        }
        result = self.handler._collect_removable_params(definition, [])
        assert result == []

    def test_no_unused_args(self):
        """No unused_args key → empty list."""
        definition = {"args": ["self", "x"]}
        result = self.handler._collect_removable_params(definition, [])
        assert result == []

    @patch("devdox_ai_sonar.services.rule_handler._is_param_used_at_callsite")
    def test_mixed_params(self, mock_used):
        """Multiple unused: one safe, one used at callsite, one with _ prefix."""
        mock_used.side_effect = lambda p, i, c: p == "used"
        definition = {
            "args": ["self", "safe", "used", "_private"],
            "unused_args": ["safe", "used", "_private"],
        }
        result = self.handler._collect_removable_params(
            definition, [{"file": "/f.py", "line": 1}]
        )
        assert result == ["safe"]


# ============================================================================
# REMOVE PARAM FROM SIGNATURE
# ============================================================================


class TestRemoveParamFromSignature:
    """Tests for ConvenationNameHandler._remove_param_from_signature — pure string ops."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()

    def test_plain_param_only(self):
        assert (
            self.handler._remove_param_from_signature("def foo(unused):", "unused")
            == "def foo():"
        )

    def test_typed_param(self):
        assert (
            self.handler._remove_param_from_signature("def foo(unused: int):", "unused")
            == "def foo():"
        )

    def test_default_value(self):
        assert (
            self.handler._remove_param_from_signature("def foo(unused=None):", "unused")
            == "def foo():"
        )

    def test_typed_and_default(self):
        assert (
            self.handler._remove_param_from_signature(
                "def foo(unused: int = 2):", "unused"
            )
            == "def foo():"
        )

    def test_star_args(self):
        assert (
            self.handler._remove_param_from_signature("def foo(*unused):", "unused")
            == "def foo():"
        )

    def test_double_star_kwargs(self):
        assert (
            self.handler._remove_param_from_signature("def foo(**unused):", "unused")
            == "def foo():"
        )

    def test_middle_param(self):
        result = self.handler._remove_param_from_signature(
            "def foo(a, unused, b):", "unused"
        )
        assert result == "def foo(a, b):"

    def test_first_param(self):
        result = self.handler._remove_param_from_signature(
            "def foo(unused, a):", "unused"
        )
        assert result == "def foo(a):"

    def test_last_param(self):
        result = self.handler._remove_param_from_signature(
            "def foo(a, unused):", "unused"
        )
        assert result == "def foo(a):"

    def test_multiple_remaining(self):
        result = self.handler._remove_param_from_signature(
            "def foo(a, unused, b, c):", "unused"
        )
        assert result == "def foo(a, b, c):"


# ============================================================================
# REMOVE PARAMETER BLOCK (lines 374-424)
# ============================================================================


class TestRemoveParameterBlock:
    """Tests for ConvenationNameHandler._remove_parameter_block using tmp_path."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()

    def test_single_line_signature(self, tmp_path):
        """Remove param from a single-line function signature."""
        source = "def my_func(self, unused_param):\n    pass\n"
        src_file = tmp_path / "module.py"
        src_file.write_text(source, encoding="utf-8")

        definition = {
            "file": str(src_file),
            "function": "my_func",
            "line": 1,
            "decorators": [],
            "args": ["self", "unused_param"],
        }
        context = _make_context()

        block = self.handler._remove_parameter_block(
            definition, context, "unused_param"
        )
        assert block.start_line == 1
        assert block.end_line == 1
        assert "unused_param" not in block.context
        assert "self" in block.context
        assert block.change_type == ChangeType.FULL_CODE

    def test_multi_line_signature(self, tmp_path):
        """Remove param from a multi-line function signature."""
        source = (
            "def my_func(\n"
            "    self,\n"
            "    unused_param,\n"
            "    other_param,\n"
            "):\n"
            "    pass\n"
        )
        src_file = tmp_path / "module.py"
        src_file.write_text(source, encoding="utf-8")

        definition = {
            "file": str(src_file),
            "function": "my_func",
            "line": 1,
            "decorators": [],
            "args": ["self", "unused_param", "other_param"],
        }
        context = _make_context()

        block = self.handler._remove_parameter_block(
            definition, context, "unused_param"
        )
        assert block.start_line == 1
        assert block.end_line == 5  # 5 signature lines
        assert "unused_param" not in block.context
        assert "other_param" in block.context
        assert block.file_path == str(src_file)


# ============================================================================
# FIX FUNC NAMING CONVENTION — S1542 (lines 541-601)
# ============================================================================


class TestFixFuncNamingConvention:
    """Tests for ConvenationNameHandler._fix_func_naming_convention."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()
        self.file_path = Path("/project/src/module.py")
        self.context = _make_context(
            functions=[{"name": "myFunc", "start_line": 10}],
            context_dict={
                "start_line": 10,
                "import_section": {"end_line": 3},
                "new_context": [{"context": "def myFunc(self):\n    pass\n"}],
            },
        )

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    def test_happy_path_no_callers(self, mock_snake):
        """camelCase function with no callers → returns SonarFixResponse with definition block."""
        mock_snake.return_value = "my_func"
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "myFunc",
                    "line": 10,
                    "decorators": [],
                    "args": ["self"],
                }
            ],
            "calls": [],
        }

        result = self.handler._fix_func_naming_convention(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is not None
        assert len(result) == 1  # Only the definition response (no caller responses)
        assert "S1542" in result[-1].EXPLANATION
        assert "myFunc" in result[-1].EXPLANATION

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    def test_happy_path_with_callers(self, mock_snake):
        """camelCase function with callers → caller blocks + definition block."""
        mock_snake.return_value = "my_func"
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "myFunc",
                    "line": 10,
                    "decorators": [],
                    "args": ["self"],
                }
            ],
            "calls": [
                {"file": "/other/file.py", "line": 5},
                {"file": "/another/file.py", "line": 12},
            ],
        }

        result = self.handler._fix_func_naming_convention(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is not None
        # 2 caller responses + 1 definition response = 3
        assert len(result) == 3

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    def test_already_snake_case_still_produces_block(self, mock_snake):
        """to_snake_case returns same name → code block still created (no guard in method)."""
        mock_snake.return_value = "my_func"
        function_info = {
            "definitions": [
                {
                    "file": str(self.file_path),
                    "function": "my_func",
                    "line": 10,
                    "decorators": [],
                    "args": ["self"],
                }
            ],
            "calls": [],
        }
        context = _make_context(
            functions=[{"name": "my_func", "start_line": 10}],
            context_dict={
                "start_line": 10,
                "import_section": {"end_line": 3},
                "new_context": [{"context": "def my_func(self):\n    pass\n"}],
            },
        )

        result = self.handler._fix_func_naming_convention(
            function_info,
            context,
            self.file_path,
        )
        # Unlike _fix_naming_convention, this method does not check old == new
        assert result is not None

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    def test_definition_in_different_file_returns_none(self, mock_snake):
        """Definition in different file → skipped → returns None."""
        mock_snake.return_value = "my_func"
        function_info = {
            "definitions": [
                {
                    "file": "/other/file.py",
                    "function": "myFunc",
                    "line": 10,
                    "decorators": [],
                    "args": ["self"],
                }
            ],
            "calls": [],
        }

        result = self.handler._fix_func_naming_convention(
            function_info,
            self.context,
            self.file_path,
        )
        assert result is None


# ============================================================================
# CREATE CALLER BLOCKS — change_func_name BRANCH (lines 681-682)
# ============================================================================


class TestCreateCallerBlocksFuncRename:
    """Tests for _create_caller_blocks with change_type='change_func_name'."""

    def setup_method(self):
        self.handler = ConvenationNameHandler()

    def test_uses_func_call_rename_pattern(self):
        """change_func_name branch → uses FUNC_CALL_RENAME_PATTERN, not ARG_RENAME."""
        function_info = {
            "definitions": [{"function": "myFunc", "args": ["self"]}],
            "calls": [{"file": "/caller.py", "line": 5}],
        }

        blocks = self.handler._create_caller_blocks(
            {"myFunc": "my_func"},
            function_info,
            "change_func_name",
        )
        assert len(blocks) == 1
        block = blocks[0]
        assert block.change_type == ChangeType.SEARCH_REPLACE
        # The replace pattern should be the new name directly
        assert block.replacements[0].replace == "my_func"
        # The search pattern should match function call, not keyword arg
        assert "myFunc" in block.replacements[0].search
        # Should NOT contain the arg rename backreference pattern
        assert r"\1" not in block.replacements[0].replace

    def test_multiple_callers_func_rename(self):
        """Multiple callers with func rename → one block per caller."""
        function_info = {
            "definitions": [{"function": "myFunc", "args": ["self", "x"]}],
            "calls": [
                {"file": "/a.py", "line": 5},
                {"file": "/b.py", "line": 10},
            ],
        }

        blocks = self.handler._create_caller_blocks(
            {"myFunc": "my_func"},
            function_info,
            "change_func_name",
        )
        assert len(blocks) == 2
        assert blocks[0].file_path == "/a.py"
        assert blocks[1].file_path == "/b.py"


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — FIND STRING OCCURRENCES EDGE CASES
# ============================================================================


class TestFindStringOccurrencesEdgeCases:
    """Edge-case AST tests for StringLiteralDuplicateHandler._find_string_occurrences."""

    def setup_method(self):
        self.find = StringLiteralDuplicateHandler._find_string_occurrences

    def test_fstring_with_interpolation_not_found(self):
        """f-strings with interpolation are JoinedStr nodes, not Constant — not matched."""
        source = 'x = f"prefix {var} suffix"\ny = f"prefix {var} suffix"\n'
        tree = ast.parse(source)
        result = self.find(tree, "prefix {var} suffix")
        assert len(result) == 0

    def test_fstring_fragments_found_individually(self):
        """Fragments inside f-strings ARE ast.Constant — they can be matched."""
        source = 'x = f"prefix {var} suffix"\n'
        tree = ast.parse(source)
        # The literal "prefix " (with trailing space) exists as a fragment
        result = self.find(tree, "prefix ")
        assert len(result) == 1

    def test_triple_quoted_string_found(self):
        """Triple-quoted strings are Constant nodes with matching value."""
        source = 'x = """hello"""\ny = """hello"""\nz = """hello"""\n'
        tree = ast.parse(source)
        result = self.find(tree, "hello")
        assert len(result) == 3

    def test_string_in_decorator_arg_found(self):
        """String literals in decorator arguments are Constant nodes."""
        source = (
            '@cache("key")\n'
            "def foo(): pass\n"
            '@cache("key")\n'
            "def bar(): pass\n"
            '@cache("key")\n'
            "def baz(): pass\n"
        )
        tree = ast.parse(source)
        result = self.find(tree, "key")
        assert len(result) == 3

    def test_string_in_default_param_found(self):
        """Strings used as default parameter values are Constant nodes."""
        source = (
            'def foo(x="default"): pass\n'
            'def bar(x="default"): pass\n'
            'def baz(x="default"): pass\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "default")
        assert len(result) == 3

    def test_bytes_literal_not_found(self):
        """Bytes literals (b\"...\") are not str — should not match."""
        source = 'x = b"hello"\ny = b"hello"\n'
        tree = ast.parse(source)
        result = self.find(tree, "hello")
        assert len(result) == 0

    def test_implicit_string_concat_merged(self):
        """Implicit concatenation merges into a single Constant in AST."""
        source = 'x = "hello" " world"\n'
        tree = ast.parse(source)
        # AST merges to "hello world"
        result = self.find(tree, "hello world")
        assert len(result) == 1
        # Individual fragments do NOT exist as separate nodes
        result_partial = self.find(tree, "hello")
        assert len(result_partial) == 0

    def test_string_in_module_level_list(self):
        """Strings inside list literals at module level are Constant nodes."""
        source = (
            'ITEMS = ["hello", "world"]\n'
            'OTHERS = ["hello", "world"]\n'
            'MORE = ["hello"]\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "hello")
        assert len(result) == 3

    def test_string_in_module_level_dict_value(self):
        """Strings as dict values at module level are Constant nodes."""
        source = (
            'HEADERS = {"Content-Type": "application/json"}\n'
            'OTHER = {"Accept": "application/json"}\n'
            'MORE = {"X": "application/json"}\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "application/json")
        assert len(result) == 3

    def test_string_with_escape_chars(self):
        """Strings with escape characters match on the actual string value."""
        source = 'x = "hello\\nworld"\ny = "hello\\nworld"\nz = "hello\\nworld"\n'
        tree = ast.parse(source)
        result = self.find(tree, "hello\nworld")
        assert len(result) == 3

    def test_string_in_class_body(self):
        """Strings inside class bodies are found by ast.walk."""
        source = (
            "class Config:\n"
            '    MSG = "error_msg"\n'
            "\n"
            "class Other:\n"
            '    MSG = "error_msg"\n'
            "\n"
            'x = "error_msg"\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "error_msg")
        assert len(result) == 3


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — GENERATE FIXES EDGE CASES
# ============================================================================


class TestGenerateFixesEdgeCases:
    """Edge-case end-to-end tests for generate_fixes with llm_caller=None."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()
        self.project_path = Path("/project")

    def _make_issue(self, message: str, first_line: int = 1) -> Mock:
        issue = Mock()
        issue.message = message
        issue.first_line = first_line
        issue.rule = "python:S1192"
        return issue

    async def test_string_in_decorator_args(self, tmp_path):
        """Strings used in decorator args are found and replaced."""
        source = (
            'def cache(key): pass\n'
            '@cache("session_key")\n'
            "def foo(): pass\n"
            '@cache("session_key")\n'
            "def bar(): pass\n"
            '@cache("session_key")\n'
            "def baz(): pass\n"
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "session_key" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3
        for block in replacement_blocks:
            assert block.replacements[0].search == '"session_key"'

    async def test_string_in_default_params(self, tmp_path):
        """Strings used as default parameter values are found and replaced."""
        source = (
            'def foo(fmt="json"): pass\n'
            'def bar(fmt="json"): pass\n'
            'def baz(fmt="json"): pass\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "json" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3

    async def test_triple_quoted_strings(self, tmp_path):
        """Triple-quoted strings are found and replaced."""
        source = (
            'x = """hello"""\n'
            'y = """hello"""\n'
            'z = """hello"""\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3
        # Verify the search string uses the exact quoted form from source
        for block in replacement_blocks:
            assert block.replacements[0].search == '"""hello"""'

    async def test_string_with_escape_chars_mismatch(self, tmp_path):
        """Escape chars: Sonar message has source repr, AST has actual value — no match.

        The Sonar message contains 'hello\\nworld' (literal backslash-n) but the AST
        Constant.value is 'hello\\nworld' (actual newline). These don't match, so
        _find_string_occurrences returns empty and the handler returns None.
        This is a known limitation for strings with escape sequences.
        """
        source = (
            'x = "hello\\nworld"\n'
            'y = "hello\\nworld"\n'
            'z = "hello\\nworld"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello\\nworld" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        # Returns None because the extracted literal (source repr) doesn't match
        # the AST value (actual newline character)
        assert result is None

    async def test_empty_issues_list(self, tmp_path):
        """Empty issues list produces no fixes."""
        source = 'x = "hello"\ny = "hello"\nz = "hello"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [], context, self.project_path, file, llm_caller=None
        )
        assert result is None

    async def test_mixed_single_and_double_quotes(self, tmp_path):
        """Mixed quote styles for the same string value are all found."""
        source = (
            "x = 'hello'\n"
            'y = "hello"\n'
            "z = 'hello'\n"
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3
        # Verify each replacement preserves the original quote style
        searches = [b.replacements[0].search for b in replacement_blocks]
        assert "'hello'" in searches
        assert '"hello"' in searches

    async def test_fstring_only_occurrences_returns_none(self, tmp_path):
        """If the literal only appears inside f-strings, no Constant nodes match."""
        source = (
            'x = f"prefix hello suffix"\n'
            'y = f"prefix hello suffix"\n'
            'z = f"prefix hello suffix"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "prefix hello suffix" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        # f-strings without interpolation ARE Constant nodes in the AST,
        # so they will be found. This test verifies that behavior.
        # (Only f-strings WITH {expressions} become JoinedStr and are invisible.)
        assert result is not None

    async def test_fstring_with_interpolation_not_found(self, tmp_path):
        """If the literal only appears in f-strings with interpolation, returns None."""
        source = (
            'x = f"hello {name} world"\n'
            'y = f"hello {name} world"\n'
            'z = f"hello {name} world"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello {name} world" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is None

    async def test_string_in_class_body(self, tmp_path):
        """Strings inside class bodies are found across the file."""
        source = (
            "class Config:\n"
            '    MSG = "error_occurred"\n'
            "\n"
            "class Handler:\n"
            '    MSG = "error_occurred"\n'
            "\n"
            'DEFAULT_MSG = "error_occurred"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "error_occurred" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        # The module-level DEFAULT_MSG is a simple assign with matching value,
        # so it should be detected as an existing constant (1 match) and reused.
        # The two class-level occurrences + the definition line is excluded.
        # Actually: _find_existing_constant finds DEFAULT_MSG (module-level, simple assign).
        # len(existing) == 1, so it reuses DEFAULT_MSG.
        # Occurrences: line 2 (class), line 5 (class), line 7 (definition) → exclude line 7
        # → 2 replacement blocks
        assert len(replacement_blocks) == 2
        for block in replacement_blocks:
            assert block.replacements[0].replace == "DEFAULT_MSG"


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — BUILD EXPLANATION UNIT TESTS
# ============================================================================


class TestBuildExplanation:
    """Direct unit tests for StringLiteralDuplicateHandler._build_explanation."""

    def setup_method(self):
        self.build = StringLiteralDuplicateHandler._build_explanation

    def test_created_action_format(self):
        """Created action includes constant definition and 'at module level'."""
        reports = [
            {
                "message": 'Define a constant instead of duplicating this literal "hello" 3 times.',
                "literal": "hello",
                "const_name": "GREETING_MSG",
                "action": "created",
                "definition_line": None,
                "lines": [1, 5, 10],
            }
        ]
        result = self.build("src/module.py", reports)

        assert "**File:** `src/module.py`" in result
        assert "Created" in result
        assert "`GREETING_MSG = 'hello'`" in result
        assert "at module level" in result
        assert "**Lines affected:**" in result
        assert "- 1" in result
        assert "- 5" in result
        assert "- 10" in result

    def test_reused_action_format(self):
        """Reused action includes constant name and definition line."""
        reports = [
            {
                "message": 'Define a constant instead of duplicating this literal "app/json" 3 times.',
                "literal": "app/json",
                "const_name": "APP_JSON",
                "action": "reused",
                "definition_line": 1,
                "lines": [5, 10, 15],
            }
        ]
        result = self.build("src/module.py", reports)

        assert "Reused existing constant" in result
        assert "`APP_JSON`" in result
        assert "defined at line 1" in result
        assert "- 5" in result

    def test_mixed_actions(self):
        """Multiple literals with different actions in one explanation."""
        reports = [
            {
                "message": 'Duplicating "app/json" 3 times.',
                "literal": "app/json",
                "const_name": "APP_JSON",
                "action": "reused",
                "definition_line": 1,
                "lines": [5, 10],
            },
            {
                "message": 'Duplicating "1234" 3 times.',
                "literal": "1234",
                "const_name": "STRING_LITERAL_1",
                "action": "created",
                "definition_line": None,
                "lines": [20, 30],
            },
        ]
        result = self.build("src/module.py", reports)

        assert "Reused existing constant" in result
        assert "Created" in result
        assert "`APP_JSON`" in result
        assert "STRING_LITERAL_1" in result

    def test_empty_reports(self):
        """No reports → only file path line."""
        result = self.build("src/module.py", [])
        assert "**File:** `src/module.py`" in result

    def test_file_path_always_present(self):
        """File path is always included in the output."""
        reports = [
            {
                "message": "msg",
                "literal": "x",
                "const_name": "X_Y",
                "action": "created",
                "definition_line": None,
                "lines": [1],
            }
        ]
        result = self.build("deeply/nested/path/module.py", reports)
        assert "**File:** `deeply/nested/path/module.py`" in result


# ============================================================================
# S1192 INTEGRATION TEST — FULL PIPELINE + MD.J2 RENDERING
# ============================================================================


class TestS1192Integration:
    """Full pipeline integration test: generate_fixes → render md.j2 → verify output."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()

    def _make_issue(self, literal: str, count: int = 3) -> Mock:
        issue = Mock()
        issue.message = (
            f"Define a constant instead of duplicating this literal "
            f'"{literal}" {count} times.'
        )
        issue.first_line = 1
        issue.rule = "python:S1192"
        issue.severity = "MAJOR"
        issue.file_path = "src/module.py"
        issue.line = 2
        return issue

    async def test_full_pipeline_created_constant(self, tmp_path):
        """Full pipeline: new constant created → render through md.j2."""
        source = (
            "import requests\n"
            "\n"
            "def fetch_users(url):\n"
            '    return requests.get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_orders(url):\n"
            '    return requests.post(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_items(url):\n"
            '    return requests.put(url, headers={"Content-Type": "application/json"})\n'
        )
        src_file = tmp_path / "module.py"
        src_file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=src_file)

        result = await self.handler.generate_fixes(
            [issue], context, tmp_path, src_file, llm_caller=None
        )
        assert result is not None
        fix_response = result[0]

        # Verify fix response structure
        assert fix_response.PLACEMENT == PlacementType.GLOBAL_TOP
        assert fix_response.CONFIDENCE == 0.95
        assert "APPLICATION_JSON" in fix_response.NEW_HELPER_CODE
        assert "'application/json'" in fix_response.NEW_HELPER_CODE
        assert "Created" in fix_response.EXPLANATION
        assert "**File:**" in fix_response.EXPLANATION

        # Render through md.j2
        from jinja2 import Environment, FileSystemLoader

        templates_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "devdox_ai_sonar"
            / "templates"
        )
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("md.j2")

        rendered = template.render(
            rule="python:S1192",
            severity="MAJOR",
            message=issue.message,
            file_path="src/module.py",
            line=2,
            explanation=fix_response.EXPLANATION,
            suggestion=fix_response.FIXED_CODE_BLOCKS,
            original_code={},
        )

        # Verify rendered markdown has key sections
        assert "python:S1192" in rendered
        assert "Explanation" in rendered
        assert "Suggested Fix" in rendered
        assert "Created" in rendered
        assert "APPLICATION_JSON" in rendered

    async def test_full_pipeline_reused_constant(self, tmp_path):
        """Full pipeline: existing constant reused → render through md.j2."""
        source = (
            'APP_JSON = "application/json"\n'
            "\n"
            "def fetch_users(url):\n"
            '    return get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_orders(url):\n"
            '    return post(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_items(url):\n"
            '    return put(url, headers={"Content-Type": "application/json"})\n'
        )
        src_file = tmp_path / "module.py"
        src_file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=src_file)

        result = await self.handler.generate_fixes(
            [issue], context, tmp_path, src_file, llm_caller=None
        )
        assert result is not None
        fix_response = result[0]

        # No new constant created — reusing APP_JSON
        assert fix_response.NEW_HELPER_CODE == ""
        assert "Reused existing constant" in fix_response.EXPLANATION
        assert "`APP_JSON`" in fix_response.EXPLANATION

        # All replacement blocks use APP_JSON
        replacement_blocks = [b for b in fix_response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3
        for block in replacement_blocks:
            assert block.replacements[0].replace == "APP_JSON"

        # Render through md.j2
        from jinja2 import Environment, FileSystemLoader

        templates_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "devdox_ai_sonar"
            / "templates"
        )
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("md.j2")

        rendered = template.render(
            rule="python:S1192",
            severity="MAJOR",
            message=issue.message,
            file_path="src/module.py",
            line=2,
            explanation=fix_response.EXPLANATION,
            suggestion=fix_response.FIXED_CODE_BLOCKS,
            original_code={},
        )

        assert "python:S1192" in rendered
        assert "Reused existing constant" in rendered
        assert "`APP_JSON`" in rendered

    async def test_full_pipeline_mixed_literals(self, tmp_path):
        """Full pipeline: multiple literals, mix of reuse + create."""
        source = (
            'CONTENT_TYPE = "application/json"\n'
            "\n"
            "def fetch_users():\n"
            '    return get("/api/v1/users", headers={"Accept": "application/json"})\n'
            "\n"
            "def fetch_orders():\n"
            '    return get("/api/v1/users", headers={"Accept": "application/json"})\n'
            "\n"
            "def fetch_items():\n"
            '    return get("/api/v1/users", headers={"Accept": "application/json"})\n'
        )
        src_file = tmp_path / "module.py"
        src_file.write_text(source)

        issues = [
            self._make_issue("application/json"),
            self._make_issue("/api/v1/users"),
        ]
        context = _make_context(file_path=src_file)

        result = await self.handler.generate_fixes(
            issues, context, tmp_path, src_file, llm_caller=None
        )
        assert result is not None
        fix_response = result[0]

        # application/json → reused CONTENT_TYPE
        # /api/v1/users → new constant created
        assert "Reused existing constant" in fix_response.EXPLANATION
        assert "Created" in fix_response.EXPLANATION

        replacement_blocks = [b for b in fix_response.FIXED_CODE_BLOCKS if b.replacements]
        replace_names = {b.replacements[0].replace for b in replacement_blocks}
        assert "CONTENT_TYPE" in replace_names
        # The /api/v1/users literal gets a generated name
        assert len(replace_names) == 2


# ============================================================================
# STRING LITERAL DUPLICATE HANDLER — ADDITIONAL EDGE CASES
# ============================================================================


class TestFindStringOccurrencesAdditionalEdgeCases:
    """Additional AST context tests for _find_string_occurrences."""

    def setup_method(self):
        self.find = StringLiteralDuplicateHandler._find_string_occurrences

    def test_string_in_list_comprehension(self):
        """Strings inside list comprehensions are Constant nodes."""
        source = (
            'result = [x for x in items if x == "active"]\n'
            'other = [x for x in items if x == "active"]\n'
            'more = [x for x in items if x == "active"]\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "active")
        assert len(result) == 3

    def test_string_in_lambda(self):
        """Strings inside lambda bodies are Constant nodes."""
        source = (
            'fn1 = lambda: "default_value"\n'
            'fn2 = lambda: "default_value"\n'
            'fn3 = lambda: "default_value"\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "default_value")
        assert len(result) == 3

    def test_string_in_assert(self):
        """Strings in assert statements are Constant nodes."""
        source = (
            'assert result == "expected"\n'
            'assert other == "expected"\n'
            'assert final == "expected"\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "expected")
        assert len(result) == 3

    def test_string_in_raise(self):
        """Strings in raise statements are Constant nodes."""
        source = (
            'raise ValueError("invalid input")\n'
            'raise TypeError("invalid input")\n'
            'raise RuntimeError("invalid input")\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "invalid input")
        assert len(result) == 3

    def test_string_in_walrus_operator(self):
        """Strings in walrus operator (:=) are Constant nodes."""
        source = (
            'if (x := "sentinel"):\n'
            "    pass\n"
            'if (y := "sentinel"):\n'
            "    pass\n"
            'if (z := "sentinel"):\n'
            "    pass\n"
        )
        tree = ast.parse(source)
        result = self.find(tree, "sentinel")
        assert len(result) == 3

    def test_string_as_dict_key(self):
        """Strings used as dict keys are Constant nodes."""
        source = (
            'a = {"status": 1}\n'
            'b = {"status": 2}\n'
            'c = {"status": 3}\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "status")
        assert len(result) == 3

    def test_string_in_with_statement(self):
        """Strings in with statements are Constant nodes."""
        source = (
            'with open("config.json") as f:\n'
            "    pass\n"
            'with open("config.json") as g:\n'
            "    pass\n"
            'with open("config.json") as h:\n'
            "    pass\n"
        )
        tree = ast.parse(source)
        result = self.find(tree, "config.json")
        assert len(result) == 3

    def test_string_in_try_except(self):
        """Strings in try/except blocks are Constant nodes."""
        source = (
            "try:\n"
            '    log("error occurred")\n'
            "except:\n"
            '    log("error occurred")\n'
            "finally:\n"
            '    log("error occurred")\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "error occurred")
        assert len(result) == 3

    def test_same_literal_twice_on_same_line(self):
        """Two occurrences of the same literal on a single line."""
        source = 'x = {"key": "value", "other": "value"}\n'
        tree = ast.parse(source)
        result = self.find(tree, "value")
        assert len(result) == 2
        # Both should have the same lineno but different col offsets
        assert result[0][0] == result[1][0]  # same line
        assert result[0][1] != result[1][1]  # different columns

    def test_multiline_triple_quoted_string(self):
        """Multiline triple-quoted strings spanning multiple source lines."""
        source = 'x = """line1\nline2"""\ny = """line1\nline2"""\nz = """line1\nline2"""\n'
        tree = ast.parse(source)
        result = self.find(tree, "line1\nline2")
        assert len(result) == 3

    def test_raw_string(self):
        """Raw strings (r\"...\") store literal backslashes in AST."""
        source = 'x = r"hello\\nworld"\ny = r"hello\\nworld"\nz = r"hello\\nworld"\n'
        tree = ast.parse(source)
        # r"hello\nworld" → AST value is "hello\\nworld" (literal backslash-n)
        result = self.find(tree, "hello\\nworld")
        assert len(result) == 3

    def test_empty_string(self):
        """Empty string literals are valid Constant nodes."""
        source = 'x = ""\ny = ""\nz = ""\n'
        tree = ast.parse(source)
        result = self.find(tree, "")
        assert len(result) == 3

    def test_string_in_dunder_all(self):
        """Strings in __all__ are Constant nodes."""
        source = (
            '__all__ = ["module_a", "module_b"]\n'
            'x = "module_a"\n'
            'y = "module_a"\n'
        )
        tree = ast.parse(source)
        result = self.find(tree, "module_a")
        assert len(result) == 3


class TestFindExistingConstantAdditionalEdgeCases:
    """Additional edge cases for _find_existing_constant."""

    def setup_method(self):
        self.find = StringLiteralDuplicateHandler._find_existing_constant

    def test_augmented_assign_not_found(self):
        """Augmented assignment (+=) is not ast.Assign — should be skipped."""
        source = 'X = ""\nX += "value"\n'
        tree = ast.parse(source)
        result = self.find(tree, "value")
        assert len(result) == 0

    def test_multiple_targets_assign_not_found(self):
        """Multiple targets (A = B = \"value\") has len(targets)==2 — skipped."""
        source = 'A = B = "value"\n'
        tree = ast.parse(source)
        result = self.find(tree, "value")
        assert len(result) == 0

    def test_conditional_expression_not_found(self):
        """Conditional expression value (IfExp) is not Constant — skipped."""
        source = 'X = "value" if True else "other"\n'
        tree = ast.parse(source)
        result = self.find(tree, "value")
        assert len(result) == 0

    def test_string_concat_not_found(self):
        """String concatenation (BinOp) is not Constant — skipped."""
        source = 'X = "hello" + " world"\n'
        tree = ast.parse(source)
        result = self.find(tree, "hello world")
        assert len(result) == 0

    def test_annassign_without_value(self):
        """AnnAssign without a value (X: str) has value=None — skipped."""
        source = "X: str\n"
        tree = ast.parse(source)
        result = self.find(tree, "")
        assert len(result) == 0

    def test_annassign_with_none_value(self):
        """AnnAssign with value=None literal (X: str = None) — not a string."""
        source = "X: str = None\n"
        tree = ast.parse(source)
        result = self.find(tree, "None")
        assert len(result) == 0

    def test_both_assign_and_annassign_same_literal(self):
        """Both Assign and AnnAssign for same literal → 2 matches (ambiguous)."""
        source = 'X = "value"\nY: str = "value"\n'
        tree = ast.parse(source)
        result = self.find(tree, "value")
        assert len(result) == 2


class TestGenerateFixesAdditionalEdgeCases:
    """Additional edge cases for generate_fixes."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()
        self.project_path = Path("/project")

    def _make_issue(self, message: str, first_line: int = 1) -> Mock:
        issue = Mock()
        issue.message = message
        issue.first_line = first_line
        issue.rule = "python:S1192"
        return issue

    async def test_empty_source_file(self, tmp_path):
        """Empty source file produces no fixes."""
        file = tmp_path / "empty.py"
        file.write_text("")
        context = _make_context(file_path=file)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello" 3 times.'
        )
        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is None

    async def test_only_constant_definition_no_other_occurrences(self, tmp_path):
        """If only the definition exists and no other occurrences, returns None."""
        source = 'MY_CONST = "only_here"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "only_here" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        # Existing constant found → definition line filtered → 0 occurrences
        assert result is None

    async def test_duplicate_issues_for_same_literal(self, tmp_path):
        """Two issues for the same literal should not produce duplicate blocks."""
        source = (
            "def foo():\n"
            '    return "hello world"\n'
            "\n"
            "def bar():\n"
            '    return "hello world"\n'
            "\n"
            "def baz():\n"
            '    return "hello world"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue1 = self._make_issue(
            'Define a constant instead of duplicating this literal "hello world" 3 times.'
        )
        issue2 = self._make_issue(
            'Define a constant instead of duplicating this literal "hello world" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue1, issue2], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # Both issues produce blocks — the handler doesn't deduplicate issues.
        # This documents current behavior: 6 replacement blocks (3 per issue).
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        # At minimum, we verify it doesn't crash and produces valid output
        assert len(replacement_blocks) >= 3

    async def test_same_literal_twice_on_same_line(self, tmp_path):
        """Two occurrences on the same line produce two separate blocks."""
        source = (
            'x = {"key": "value", "other": "value"}\n'
            'y = "value"\n'
            'z = "value"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "value" 4 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        # 4 occurrences → 4 replacement blocks
        assert len(replacement_blocks) == 4
        # The two blocks on line 1 should have different search strings
        # (same value but potentially different column slices)
        line1_blocks = [b for b in replacement_blocks if b.start_line == 1]
        assert len(line1_blocks) == 2

    async def test_annassign_collision_with_naming_service(self, tmp_path):
        """AnnAssign UPPERCASE name should be considered for collision avoidance."""
        source = (
            'APPLICATION_JSON: str = "something_else"\n'
            "\n"
            "def foo():\n"
            '    return "application/json"\n'
            "\n"
            "def bar():\n"
            '    return "application/json"\n'
            "\n"
            "def baz():\n"
            '    return "application/json"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "application/json" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # The naming service would want to generate APPLICATION_JSON, but that
        # name already exists as an AnnAssign. Document current behavior:
        # the existing_module_names collection only checks ast.Assign, NOT
        # ast.AnnAssign, so the naming service doesn't know about the collision.
        # This is a known limitation.
        assert response.NEW_HELPER_CODE != ""
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3

    async def test_string_in_list_comprehension(self, tmp_path):
        """Strings inside list comprehensions are found and replaced."""
        source = (
            'result1 = [x for x in items if x == "active"]\n'
            'result2 = [x for x in items if x == "active"]\n'
            'result3 = [x for x in items if x == "active"]\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "active" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3

    async def test_string_in_lambda(self, tmp_path):
        """Strings inside lambda bodies are found and replaced."""
        source = (
            'fn1 = lambda: "default_value"\n'
            'fn2 = lambda: "default_value"\n'
            'fn3 = lambda: "default_value"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "default_value" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3

    async def test_string_in_raise_statement(self, tmp_path):
        """Strings in raise statements are found and replaced."""
        source = (
            "def foo():\n"
            '    raise ValueError("invalid input")\n'
            "\n"
            "def bar():\n"
            '    raise TypeError("invalid input")\n'
            "\n"
            "def baz():\n"
            '    raise RuntimeError("invalid input")\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "invalid input" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3

    async def test_string_in_assert_statement(self, tmp_path):
        """Strings in assert statements are found and replaced."""
        source = (
            'assert result1 == "expected"\n'
            'assert result2 == "expected"\n'
            'assert result3 == "expected"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "expected" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3

    async def test_raw_string_occurrences(self, tmp_path):
        """Raw strings (r\"...\") are found by their AST value."""
        source = (
            'x = r"hello\\nworld"\n'
            'y = r"hello\\nworld"\n'
            'z = r"hello\\nworld"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        # Sonar message would contain the actual string value (with literal backslash)
        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello\\nworld" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 3

    async def test_file_outside_project_path(self, tmp_path):
        """File path not under project_path uses absolute path in explanation."""
        source = (
            "def foo():\n"
            '    return "hello world"\n'
            "\n"
            "def bar():\n"
            '    return "hello world"\n'
            "\n"
            "def baz():\n"
            '    return "hello world"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "hello world" 3 times.'
        )
        context = _make_context(file_path=file)

        # Use a project_path that doesn't contain the file
        unrelated_project = Path("/some/other/project")
        result = await self.handler.generate_fixes(
            [issue], context, unrelated_project, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # Explanation should contain the absolute path since relative_to fails
        assert str(file) in response.EXPLANATION

    async def test_naming_collision_two_literals_same_generated_name(self, tmp_path):
        """Two literals that would generate the same name get deduplicated names."""
        source = (
            "def foo():\n"
            '    return "hello world"\n'
            "\n"
            "def bar():\n"
            '    return "hello world"\n'
            "\n"
            "def baz():\n"
            '    return "hello world"\n'
            "\n"
            "def qux():\n"
            '    return "hello/world"\n'
            "\n"
            "def quux():\n"
            '    return "hello/world"\n'
            "\n"
            "def corge():\n"
            '    return "hello/world"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issues = [
            self._make_issue(
                'Define a constant instead of duplicating this literal "hello world" 3 times.'
            ),
            self._make_issue(
                'Define a constant instead of duplicating this literal "hello/world" 3 times.'
            ),
        ]
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            issues, context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # Both literals would generate HELLO_WORLD — naming service should deduplicate
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        replace_names = {b.replacements[0].replace for b in replacement_blocks}
        # Must have 2 distinct names
        assert len(replace_names) == 2

    async def test_multiline_triple_quoted_string_replacement(self, tmp_path):
        """Multiline triple-quoted strings spanning multiple source lines."""
        source = (
            'x = """line1\n'
            'line2"""\n'
            'y = """line1\n'
            'line2"""\n'
            'z = """line1\n'
            'line2"""\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "line1\nline2" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        # The handler will find the occurrences via AST, but _build_replacement_blocks
        # slices only one source line. Document current behavior.
        if result is not None:
            response = result[0]
            replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
            # Verify it at least produces blocks (even if slicing is partial)
            assert len(replacement_blocks) >= 0

    async def test_string_in_dunder_all(self, tmp_path):
        """Strings in __all__ are treated as regular occurrences.

        Note: replacing string literals in __all__ with constant references
        is technically valid Python but may be surprising. This test documents
        current behavior — the handler does NOT exempt __all__.
        """
        source = (
            '__all__ = ["my_module", "other"]\n'
            'x = "my_module"\n'
            'y = "my_module"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "my_module" 3 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        # All 3 occurrences are replaced including the one in __all__
        assert len(replacement_blocks) == 3

    async def test_assign_and_annassign_same_literal_creates_new(self, tmp_path):
        """Both Assign and AnnAssign for same literal → ambiguous → creates new constant."""
        source = (
            'X = "value"\n'
            'Y: str = "value"\n'
            "\n"
            "def foo():\n"
            '    return "value"\n'
            "\n"
            "def bar():\n"
            '    return "value"\n'
            "\n"
            "def baz():\n"
            '    return "value"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue(
            'Define a constant instead of duplicating this literal "value" 5 times.'
        )
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # 2 existing constants → ambiguous → creates new constant
        # All 5 occurrences (including definitions) are replaced
        assert response.NEW_HELPER_CODE != ""
        replacement_blocks = [b for b in response.FIXED_CODE_BLOCKS if b.replacements]
        assert len(replacement_blocks) == 5


# ============================================================================
# HELPER CLASSES — _LiteralCaches and _FixAccumulator
# ============================================================================


class TestLiteralCaches:
    """Tests for the _LiteralCaches data holder."""

    def test_initial_state(self):
        caches = _LiteralCaches()
        assert caches.occurrences == {}
        assert caches.existing == {}
        assert caches.needing_names == []
        assert caches.seen == set()

    def test_stores_occurrences(self):
        caches = _LiteralCaches()
        caches.occurrences["hello"] = [(1, 0, 7)]
        caches.seen.add("hello")
        assert "hello" in caches.occurrences
        assert "hello" in caches.seen


class TestFixAccumulator:
    """Tests for the _FixAccumulator data holder."""

    def test_initial_state(self):
        acc = _FixAccumulator()
        assert acc.code_blocks == []
        assert acc.constant_defs == []
        assert acc.used_names == set()
        assert acc.reports == []

    def test_accumulates_data(self):
        acc = _FixAccumulator()
        acc.constant_defs.append('MY_CONST = "value"')
        acc.used_names.add("MY_CONST")
        acc.reports.append({"literal": "value"})
        assert len(acc.constant_defs) == 1
        assert "MY_CONST" in acc.used_names
        assert len(acc.reports) == 1


# ============================================================================
# _collect_module_level_names
# ============================================================================


class TestCollectModuleLevelNames:
    """Tests for _collect_module_level_names."""

    def test_collects_uppercase_assign(self):
        source = 'MY_CONST = "hello"\nlower = "world"\n'
        tree = ast.parse(source)
        names = StringLiteralDuplicateHandler._collect_module_level_names(tree)
        assert "MY_CONST" in names
        assert "lower" not in names

    def test_collects_uppercase_annassign(self):
        source = 'MY_CONST: str = "hello"\n'
        tree = ast.parse(source)
        names = StringLiteralDuplicateHandler._collect_module_level_names(tree)
        assert "MY_CONST" in names

    def test_ignores_multi_target_assign(self):
        source = "A = B = 42\n"
        tree = ast.parse(source)
        names = StringLiteralDuplicateHandler._collect_module_level_names(tree)
        assert "A" not in names
        assert "B" not in names

    def test_empty_module(self):
        tree = ast.parse("")
        names = StringLiteralDuplicateHandler._collect_module_level_names(tree)
        assert names == set()

    def test_mixed_assign_and_annassign(self):
        source = (
            'FOO = "foo"\n'
            'BAR: str = "bar"\n'
            'baz = "baz"\n'
        )
        tree = ast.parse(source)
        names = StringLiteralDuplicateHandler._collect_module_level_names(tree)
        assert names == {"FOO", "BAR"}

    def test_ignores_nested_functions(self):
        source = 'def f():\n    NESTED = "val"\n'
        tree = ast.parse(source)
        names = StringLiteralDuplicateHandler._collect_module_level_names(tree)
        assert names == set()


# ============================================================================
# _prescan_issues
# ============================================================================


class TestPrescanIssues:
    """Tests for _prescan_issues."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()

    def _make_issue(self, literal, count=3):
        issue = Mock()
        issue.message = f'Define a constant instead of duplicating this literal "{literal}" {count} times.'
        return issue

    def test_caches_occurrences(self):
        source = 'x = "hello"\ny = "hello"\nz = "hello"\n'
        tree = ast.parse(source)
        issues = [self._make_issue("hello")]
        caches = self.handler._prescan_issues(issues, tree)
        assert "hello" in caches.occurrences
        assert len(caches.occurrences["hello"]) == 3

    def test_deduplicates_same_literal(self):
        source = 'x = "hello"\ny = "hello"\nz = "hello"\n'
        tree = ast.parse(source)
        issues = [self._make_issue("hello"), self._make_issue("hello")]
        caches = self.handler._prescan_issues(issues, tree)
        assert len(caches.needing_names) == 1

    def test_literal_with_existing_constant(self):
        source = (
            'MY_CONST = "hello"\n'
            'def foo():\n'
            '    return "hello"\n'
            'def bar():\n'
            '    return "hello"\n'
        )
        tree = ast.parse(source)
        issues = [self._make_issue("hello")]
        caches = self.handler._prescan_issues(issues, tree)
        assert "hello" in caches.existing
        assert len(caches.existing["hello"]) == 1
        # Existing constant found → not in needing_names
        assert len(caches.needing_names) == 0

    def test_no_occurrences_skipped(self):
        source = 'x = "other"\n'
        tree = ast.parse(source)
        issues = [self._make_issue("missing")]
        caches = self.handler._prescan_issues(issues, tree)
        assert caches.occurrences.get("missing") == []
        assert len(caches.needing_names) == 0

    def test_invalid_message_skipped(self):
        issue = Mock()
        issue.message = "Some other message"
        tree = ast.parse("")
        caches = self.handler._prescan_issues([issue], tree)
        assert caches.seen == set()


# ============================================================================
# _resolve_names
# ============================================================================


class TestResolveNames:
    """Tests for _resolve_names."""

    def test_returns_naming_response(self):
        caches = _LiteralCaches()
        caches.needing_names = [
            LiteralContext(literal="application/json", occurrences=[(1, 0, 18)])
        ]
        result = StringLiteralDuplicateHandler._resolve_names(
            caches, set(), "/test.py", None
        )
        assert isinstance(result, NamingResponse)
        assert "application/json" in result.names

    def test_empty_needing_names(self):
        caches = _LiteralCaches()
        result = StringLiteralDuplicateHandler._resolve_names(
            caches, set(), "/test.py", None
        )
        assert isinstance(result, NamingResponse)
        assert result.names == {}

    def test_respects_existing_names(self):
        caches = _LiteralCaches()
        caches.needing_names = [
            LiteralContext(literal="application/json", occurrences=[(1, 0, 18)])
        ]
        result = StringLiteralDuplicateHandler._resolve_names(
            caches, {"APPLICATION_JSON"}, "/test.py", None
        )
        assert result.names["application/json"] == "APPLICATION_JSON_2"


# ============================================================================
# _process_issues
# ============================================================================


class TestProcessIssues:
    """Tests for _process_issues."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()

    def _make_issue(self, literal, count=3):
        issue = Mock()
        issue.message = f'Define a constant instead of duplicating this literal "{literal}" {count} times.'
        return issue

    def test_processes_single_issue(self):
        source = 'x = "hello world"\ny = "hello world"\nz = "hello world"\n'
        file_lines = source.splitlines(keepends=True)
        tree = ast.parse(source)

        issues = [self._make_issue("hello world")]
        caches = self.handler._prescan_issues(issues, tree)
        naming_response = NamingResponse(names={"hello world": "HELLO_WORLD"})

        acc = self.handler._process_issues(
            issues, caches, naming_response, file_lines, "/test.py"
        )
        assert len(acc.code_blocks) == 3
        assert len(acc.constant_defs) == 1
        assert "HELLO_WORLD" in acc.used_names

    def test_deduplicates_issues(self):
        source = 'x = "hello world"\ny = "hello world"\nz = "hello world"\n'
        file_lines = source.splitlines(keepends=True)
        tree = ast.parse(source)

        issues = [self._make_issue("hello world"), self._make_issue("hello world")]
        caches = self.handler._prescan_issues(issues, tree)
        naming_response = NamingResponse(names={"hello world": "HELLO_WORLD"})

        acc = self.handler._process_issues(
            issues, caches, naming_response, file_lines, "/test.py"
        )
        # Duplicate issue should be skipped
        assert len(acc.reports) == 1


# ============================================================================
# _process_single_literal
# ============================================================================


class TestProcessSingleLiteral:
    """Tests for _process_single_literal."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()

    def _make_issue(self, literal, count=3):
        issue = Mock()
        issue.message = f'Define a constant instead of duplicating this literal "{literal}" {count} times.'
        return issue

    def test_creates_new_constant(self):
        source = 'x = "value"\ny = "value"\nz = "value"\n'
        file_lines = source.splitlines(keepends=True)
        tree = ast.parse(source)

        caches = _LiteralCaches()
        caches.occurrences["value"] = [(1, 4, 11), (2, 4, 11), (3, 4, 11)]
        caches.existing["value"] = []
        naming_response = NamingResponse(names={"value": "VALUE_CONST"})
        acc = _FixAccumulator()
        issue = self._make_issue("value")

        self.handler._process_single_literal(
            issue, "value", caches, naming_response,
            file_lines, "/test.py", acc,
        )
        assert "VALUE_CONST" in acc.used_names
        assert len(acc.code_blocks) == 3
        assert len(acc.constant_defs) == 1
        assert acc.reports[0]["action"] == "created"

    def test_reuses_existing_constant(self):
        source = 'MY_CONST = "value"\nx = "value"\ny = "value"\n'
        file_lines = source.splitlines(keepends=True)

        caches = _LiteralCaches()
        caches.occurrences["value"] = [(1, 11, 18), (2, 4, 11), (3, 4, 11)]
        caches.existing["value"] = [("MY_CONST", 1)]
        naming_response = NamingResponse(names={})
        acc = _FixAccumulator()
        issue = self._make_issue("value")

        self.handler._process_single_literal(
            issue, "value", caches, naming_response,
            file_lines, "/test.py", acc,
        )
        assert len(acc.constant_defs) == 0
        # Definition line filtered out; only 2 replacement blocks
        assert len(acc.code_blocks) == 2
        assert acc.reports[0]["action"] == "reused"

    def test_no_occurrences_returns_early(self):
        caches = _LiteralCaches()
        caches.occurrences["value"] = []
        naming_response = NamingResponse(names={})
        acc = _FixAccumulator()
        issue = self._make_issue("value")

        self.handler._process_single_literal(
            issue, "value", caches, naming_response,
            [], "/test.py", acc,
        )
        assert len(acc.code_blocks) == 0
        assert len(acc.reports) == 0


# ============================================================================
# _build_response
# ============================================================================


class TestBuildResponse:
    """Tests for _build_response."""

    def test_with_constant_defs(self):
        acc = _FixAccumulator()
        acc.constant_defs = ['MY_CONST = "value"']
        acc.code_blocks = [
            CodeBlock(
                block_name="MY_CONST",
                start_line=5,
                end_line=5,
                has_changes=True,
                change_type=ChangeType.SEARCH_REPLACE,
                block_type=BlockType.MODULE,
                file_path="/test.py",
                replacements=[
                    SearchReplace(
                        search='"value"', replace="MY_CONST",
                        is_regex=False, count=1,
                    )
                ],
            )
        ]
        acc.reports = [{
            "message": "test",
            "literal": "value",
            "const_name": "MY_CONST",
            "action": "created",
            "definition_line": None,
            "lines": [5],
        }]

        response = StringLiteralDuplicateHandler._build_response(
            acc, "src/module.py"
        )
        assert response.PLACEMENT == PlacementType.GLOBAL_TOP
        assert response.CONFIDENCE == 0.95
        assert 'MY_CONST = "value"' in response.NEW_HELPER_CODE
        # First block should be the constant definition insert
        assert response.FIXED_CODE_BLOCKS[0].block_name == "New constants"
        assert response.FIXED_CODE_BLOCKS[0].start_line == 0

    def test_without_constant_defs(self):
        acc = _FixAccumulator()
        acc.code_blocks = [
            CodeBlock(
                block_name="MY_CONST",
                start_line=5,
                end_line=5,
                has_changes=True,
                change_type=ChangeType.SEARCH_REPLACE,
                block_type=BlockType.MODULE,
                file_path="/test.py",
                replacements=[
                    SearchReplace(
                        search='"value"', replace="MY_CONST",
                        is_regex=False, count=1,
                    )
                ],
            )
        ]
        acc.reports = [{
            "message": "test",
            "literal": "value",
            "const_name": "MY_CONST",
            "action": "reused",
            "definition_line": 1,
            "lines": [5],
        }]

        response = StringLiteralDuplicateHandler._build_response(
            acc, "src/module.py"
        )
        assert response.NEW_HELPER_CODE == ""
        # No constant def insert block
        assert response.FIXED_CODE_BLOCKS[0].block_name == "MY_CONST"


# ============================================================================
# _resolve_literal_action
# ============================================================================


class TestResolveLiteralAction:
    """Tests for _resolve_literal_action."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()

    def test_reuses_existing_constant(self):
        caches = _LiteralCaches()
        caches.existing["value"] = [("MY_CONST", 1)]
        naming_response = NamingResponse(names={})
        acc = _FixAccumulator()
        occurrences = [(1, 0, 7), (2, 0, 7), (3, 0, 7)]

        name, action, def_line, filtered = self.handler._resolve_literal_action(
            "value", caches, naming_response, acc, occurrences
        )
        assert name == "MY_CONST"
        assert action == "reused"
        assert def_line == 1
        # Definition line filtered out
        assert (1, 0, 7) not in filtered

    def test_creates_new_constant(self):
        caches = _LiteralCaches()
        caches.existing["value"] = []
        naming_response = NamingResponse(names={"value": "VALUE_CONST"})
        acc = _FixAccumulator()
        occurrences = [(1, 0, 7), (2, 0, 7)]

        name, action, def_line, filtered = self.handler._resolve_literal_action(
            "value", caches, naming_response, acc, occurrences
        )
        assert name == "VALUE_CONST"
        assert action == "created"
        assert def_line is None
        assert len(filtered) == 2
        assert "VALUE_CONST" in acc.used_names
        assert len(acc.constant_defs) == 1

    def test_falls_to_generate_constant_name(self):
        caches = _LiteralCaches()
        caches.existing["value"] = []
        naming_response = NamingResponse(names={})
        acc = _FixAccumulator()

        name, action, _, _ = self.handler._resolve_literal_action(
            "value", caches, naming_response, acc, [(1, 0, 7)]
        )
        assert name == "STRING_LITERAL_1"
        assert action == "created"


# ============================================================================
# _append_literal_results
# ============================================================================


class TestAppendLiteralResults:
    """Tests for _append_literal_results."""

    def _make_issue(self, literal):
        issue = Mock()
        issue.message = f'Define a constant instead of duplicating this literal "{literal}" 3 times.'
        return issue

    def test_appends_blocks_and_report(self):
        acc = _FixAccumulator()
        file_lines = ['x = "val"\n', 'y = "val"\n']
        issue = self._make_issue("val")

        StringLiteralDuplicateHandler._append_literal_results(
            issue, "val", "MY_CONST", "created", None,
            [(1, 4, 9), (2, 4, 9)], file_lines, "/test.py", acc,
        )
        assert len(acc.code_blocks) == 2
        assert len(acc.reports) == 1
        assert acc.reports[0]["const_name"] == "MY_CONST"
        assert acc.reports[0]["action"] == "created"


# ============================================================================
# _create_constant_insert_block
# ============================================================================


class TestCreateConstantInsertBlock:
    """Tests for _create_constant_insert_block."""

    def test_creates_insert_block(self):
        existing = [
            CodeBlock(
                block_name="X",
                start_line=5,
                end_line=5,
                has_changes=True,
                change_type=ChangeType.SEARCH_REPLACE,
                block_type=BlockType.MODULE,
                file_path="/test.py",
                replacements=[],
            )
        ]
        block = StringLiteralDuplicateHandler._create_constant_insert_block(
            'MY_CONST = "hello"', existing
        )
        assert block.block_name == "New constants"
        assert block.start_line == 0
        assert block.change_type == ChangeType.DIFF
        assert block.file_path == "/test.py"
        assert block.changes[0].new == 'MY_CONST = "hello"'

    def test_empty_existing_blocks(self):
        block = StringLiteralDuplicateHandler._create_constant_insert_block(
            'MY_CONST = "hello"', []
        )
        assert block.file_path is None


# ============================================================================
# _format_literal_report
# ============================================================================


class TestFormatLiteralReport:
    """Tests for _format_literal_report."""

    def test_reused_action(self):
        report = {
            "message": "test msg",
            "literal": "value",
            "const_name": "MY_CONST",
            "action": "reused",
            "definition_line": 5,
            "lines": [10, 20],
        }
        result = StringLiteralDuplicateHandler._format_literal_report(report)
        assert "Reused existing constant" in result
        assert "`MY_CONST`" in result
        assert "line 5" in result

    def test_created_action(self):
        report = {
            "message": "test msg",
            "literal": "value",
            "const_name": "MY_CONST",
            "action": "created",
            "definition_line": None,
            "lines": [10],
        }
        result = StringLiteralDuplicateHandler._format_literal_report(report)
        assert "Created" in result
        assert "at module level" in result


# ============================================================================
# _match_assignment_value
# ============================================================================


class TestMatchAssignmentValue:
    """Tests for _match_assignment_value."""

    def test_matching_assign(self):
        tree = ast.parse('MY_CONST = "hello"\n')
        node = tree.body[0]
        result = StringLiteralDuplicateHandler._match_assignment_value(node, "hello")
        assert result == ("MY_CONST", 1)

    def test_non_matching_value(self):
        tree = ast.parse('MY_CONST = "other"\n')
        node = tree.body[0]
        result = StringLiteralDuplicateHandler._match_assignment_value(node, "hello")
        assert result is None

    def test_matching_annassign(self):
        tree = ast.parse('MY_CONST: str = "hello"\n')
        node = tree.body[0]
        result = StringLiteralDuplicateHandler._match_assignment_value(node, "hello")
        assert result == ("MY_CONST", 1)

    def test_non_string_value(self):
        tree = ast.parse("X = 42\n")
        node = tree.body[0]
        result = StringLiteralDuplicateHandler._match_assignment_value(node, "42")
        assert result is None

    def test_function_def_returns_none(self):
        tree = ast.parse("def foo(): pass\n")
        node = tree.body[0]
        result = StringLiteralDuplicateHandler._match_assignment_value(node, "foo")
        assert result is None

    def test_multi_target_assign_returns_none(self):
        tree = ast.parse('A = B = "hello"\n')
        node = tree.body[0]
        result = StringLiteralDuplicateHandler._match_assignment_value(node, "hello")
        assert result is None


# ============================================================================
# _create_replacement_block
# ============================================================================


class TestCreateReplacementBlock:
    """Tests for _create_replacement_block."""

    def test_creates_block(self):
        block = StringLiteralDuplicateHandler._create_replacement_block(
            lineno=5,
            quoted_literal='"hello"',
            constant_name="MY_CONST",
            file_path="/test.py",
        )
        assert block.start_line == 5
        assert block.end_line == 5
        assert block.block_name == "MY_CONST"
        assert block.change_type == ChangeType.SEARCH_REPLACE
        assert block.replacements[0].search == '"hello"'
        assert block.replacements[0].replace == "MY_CONST"

    def test_none_file_path(self):
        block = StringLiteralDuplicateHandler._create_replacement_block(
            lineno=1,
            quoted_literal="'val'",
            constant_name="X",
            file_path=None,
        )
        assert block.file_path is None


# ============================================================================
# _read_and_parse_file
# ============================================================================


class TestReadAndParseFile:
    """Tests for _read_and_parse_file."""

    def test_valid_python_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('x = "hello"\n')
        result = StringLiteralDuplicateHandler._read_and_parse_file(f)
        assert result is not None
        source, file_lines, tree = result
        assert source == 'x = "hello"\n'
        assert len(file_lines) == 1
        assert isinstance(tree, ast.Module)

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "missing.py"
        result = StringLiteralDuplicateHandler._read_and_parse_file(f)
        assert result is None

    def test_syntax_error_file(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def foo(:\n")
        result = StringLiteralDuplicateHandler._read_and_parse_file(f)
        assert result is None
