import re
import pytest
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

from devdox_ai_sonar.services.rule_handler import (
    AWAIT_REMOVAL_PATTERN,
    ConvenationNameHandler,
    AsyncToSyncHandler,
    CognitiveComplexityHandler,
    DefaultRuleHandler,
    RuleHandlerRegistry,
    _resolve_effective_values,
    _resolve_relative_path,
)
from devdox_ai_sonar.models.sonar import (
    SonarFixResponse,
    CodeBlock,
    ChangeType,
    BlockType,
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


# ============================================================================
# SHARED HELPERS FOR HANDLER TESTS
# ============================================================================


def _make_context(**overrides) -> FixContext:
    """Build a FixContext with sensible defaults, overridable via kwargs."""
    defaults = dict(
        file_path=Path("/project/src/module.py"),
        file_path_tmp=Path("/tmp/module.py"),
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
            [response], context,
            Path("/project/src/module.py"), Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": [10]},
        )
        assert len(result) == 1
        assert result[0].explanation == "test explanation"
        assert result[0].confidence == 0.95
        assert result[0].file_path == "src/module.py"

    def test_multiple_suggestions(self):
        context = _make_context()
        result = self.handler._build_fix_suggestion(
            [_make_fix_response(), _make_fix_response()], context,
            Path("/project/src/module.py"), Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": []},
        )
        assert len(result) == 2

    def test_custom_model(self):
        self.handler.model = "gpt-4o"
        context = _make_context()
        result = self.handler._build_fix_suggestion(
            [_make_fix_response()], context,
            Path("/project/src/module.py"), Path("/project"),
            {"first_line": 10, "last_line": 20, "problem_lines": []},
        )
        assert result[0].llm_model == "gpt-4o"

    def test_default_model_unknown(self):
        context = _make_context()
        result = self.handler._build_fix_suggestion(
            [_make_fix_response()], context,
            Path("/project/src/module.py"), Path("/project"),
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
            [_make_fix_response()], context,
            Path("/project/src/module.py"), Path("/project"),
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
            [_make_fix_response()], context,
            Path("/project/src/module.py"), Path("/project"),
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
            self.issues, self.context, Path("/project"),
            Path("/project/src/module.py"), llm_caller=None,
        )
        assert result is None

    async def test_calls_llm_returns_response(self):
        mock_llm = Mock()
        fix_response = _make_fix_response()
        mock_llm._call_llm_list.return_value = fix_response

        result = await self.handler.generate_fixes(
            self.issues, self.context, Path("/project"),
            Path("/project/src/module.py"), llm_caller=mock_llm,
        )
        assert result == [fix_response]
        mock_llm._call_llm_list.assert_called_once()

    async def test_returns_none_on_exception(self):
        mock_llm = Mock()
        mock_llm._call_llm_list.side_effect = RuntimeError("LLM error")

        result = await self.handler.generate_fixes(
            self.issues, self.context, Path("/project"),
            Path("/project/src/module.py"), llm_caller=mock_llm,
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
            self.issues, self.context, Path("/project"),
            Path("/project/src/module.py"), llm_caller=None,
        )
        assert result is None

    async def test_calls_llm_returns_response(self):
        mock_llm = Mock()
        fix_response = _make_fix_response()
        mock_llm._call_llm_list.return_value = fix_response

        result = await self.handler.generate_fixes(
            self.issues, self.context, Path("/project"),
            Path("/project/src/module.py"), llm_caller=mock_llm,
        )
        assert result == [fix_response]

    async def test_returns_none_on_exception(self):
        mock_llm = Mock()
        mock_llm._call_llm_list.side_effect = RuntimeError("boom")

        result = await self.handler.generate_fixes(
            self.issues, self.context, Path("/project"),
            Path("/project/src/module.py"), llm_caller=mock_llm,
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
        self.issues = [Mock()]

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_renames_camelcase_arg(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [{
                "file": str(self.file_path),
                "function": "my_func",
                "line": 10,
                "decorators": [],
                "args": ["self", "camelCase"],
            }],
            "calls": [],
        }
        mock_snake.side_effect = lambda x: "camel_case" if x == "camelCase" else x

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is not None
        assert len(result) >= 1
        last_response = result[-1]
        assert len(last_response.FIXED_CODE_BLOCKS) >= 1

    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_returns_none_no_definitions(self, mock_find):
        mock_find.return_value = {"definitions": [], "calls": []}

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_returns_none_no_args_need_renaming(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [{
                "file": str(self.file_path),
                "function": "my_func",
                "line": 10,
                "decorators": [],
                "args": ["self", "already_snake"],
            }],
            "calls": [],
        }
        mock_snake.side_effect = lambda x: x  # no change

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_creates_caller_blocks(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [{
                "file": str(self.file_path),
                "function": "my_func",
                "line": 10,
                "decorators": [],
                "args": ["self", "camelCase"],
            }],
            "calls": [
                {"file": "/other/file.py", "line": 5},
                {"file": "/another/file.py", "line": 12},
            ],
        }
        mock_snake.side_effect = lambda x: "camel_case" if x == "camelCase" else x

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is not None
        # Should have caller responses (one per caller) + final definition response
        assert len(result) >= 3

    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_handles_exception(self, mock_find):
        mock_find.side_effect = RuntimeError("oops")

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.to_snake_case")
    @patch("devdox_ai_sonar.services.rule_handler.find_function_implementations")
    async def test_definition_different_file_skipped(self, mock_find, mock_snake):
        mock_find.return_value = {
            "definitions": [{
                "file": "/other/file.py",
                "function": "my_func",
                "line": 10,
                "decorators": [],
                "args": ["self", "camelCase"],
            }],
            "calls": [],
        }
        mock_snake.side_effect = lambda x: "camel_case" if x == "camelCase" else x

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
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
            "function": "my_func", "line": 10, "decorators": [],
        }
        block = self.handler._change_function_definition_block(
            func_info, context, "camelCase", "camel_case",
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
            "function": "my_func", "line": 15, "decorators": ["staticmethod"],
        }
        block = self.handler._change_function_definition_block(
            func_info, context, "x", "y",
        )
        assert block.start_line == 14  # 15 - 1 decorator

    def test_caller_blocks_per_pair(self):
        func_info = {
            "definitions": [{
                "function": "my_func",
                "args": ["a", "b"],
            }],
            "calls": [
                {"file": "/file1.py", "line": 5},
                {"file": "/file2.py", "line": 12},
            ],
        }
        blocks = self.handler._create_caller_blocks(
            {"camelCase": "camel_case", "otherArg": "other_arg"}, func_info,
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
            "found": True, "name": "my_func",
            "definition": "async def my_func():",
            "start_line": 0,
        }
        mock_analysis = ConversionAnalysis(
            function_name="my_func", current_type="async", target_type="sync",
            risk_level=ConversionRisk.SAFE, blocking_issues=[], required_changes=[],
            caller_impact=[{
                "awaited": True, "file": str(self.file_path), "line": 25,
            }],
            internal_calls=[], suggestions=[],
        )
        mock_analyzer_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is not None
        assert len(result) >= 1

    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_returns_none_not_found(self, mock_detect):
        mock_detect.return_value = {"found": False}

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is None

    @patch("devdox_ai_sonar.services.rule_handler.AsyncConversionAnalyzer")
    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_cross_file_callers_separated(self, mock_detect, mock_analyzer_cls):
        mock_detect.return_value = {
            "found": True, "name": "my_func",
            "definition": "async def my_func():",
            "start_line": 0,
        }
        mock_analysis = ConversionAnalysis(
            function_name="my_func", current_type="async", target_type="sync",
            risk_level=ConversionRisk.SAFE, blocking_issues=[], required_changes=[],
            caller_impact=[{
                "awaited": True, "file": "/other/file.py", "line": 5,
            }],
            internal_calls=[], suggestions=[],
        )
        mock_analyzer_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is not None
        # Cross-file caller → separate response; definition → another response
        assert len(result) >= 2

    @patch("devdox_ai_sonar.services.rule_handler.AsyncConversionAnalyzer")
    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_skips_non_awaited(self, mock_detect, mock_analyzer_cls):
        mock_detect.return_value = {
            "found": True, "name": "my_func",
            "definition": "async def my_func():",
            "start_line": 0,
        }
        mock_analysis = ConversionAnalysis(
            function_name="my_func", current_type="async", target_type="sync",
            risk_level=ConversionRisk.SAFE, blocking_issues=[], required_changes=[],
            caller_impact=[{
                "awaited": False, "file": str(self.file_path), "line": 25,
            }],
            internal_calls=[], suggestions=[],
        )
        mock_analyzer_cls.return_value.analyze = AsyncMock(return_value=mock_analysis)

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
        )
        assert result is not None
        # Only the definition response, no caller blocks
        assert len(result) == 1

    @patch("devdox_ai_sonar.services.rule_handler.detect_original_function_type")
    async def test_handles_exception(self, mock_detect):
        mock_detect.side_effect = RuntimeError("parsing failed")

        result = await self.handler.generate_fixes(
            self.issues, self.context, self.project_path,
            self.file_path, llm_caller=None,
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
        block = self.handler._create_function_definition_block(func_info, context)
        assert block.change_type == ChangeType.DIFF
        assert len(block.changes) == 1
        assert block.changes[0].old == "async def my_func():"
        assert block.changes[0].new == "def my_func():"

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
        assert len(registry.handlers) == 4

    def test_handler_order(self):
        registry = RuleHandlerRegistry()
        assert isinstance(registry.handlers[0], AsyncToSyncHandler)
        assert isinstance(registry.handlers[1], CognitiveComplexityHandler)
        assert isinstance(registry.handlers[2], ConvenationNameHandler)
        assert isinstance(registry.handlers[3], DefaultRuleHandler)

    def test_get_handler_s7503(self):
        registry = RuleHandlerRegistry()
        assert isinstance(registry.get_handler("python:S7503"), AsyncToSyncHandler)

    def test_get_handler_s3776(self):
        registry = RuleHandlerRegistry()
        assert isinstance(registry.get_handler("python:S3776"), CognitiveComplexityHandler)

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
