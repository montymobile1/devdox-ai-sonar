"""Tests for agent_supervisor extracted implementation functions."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from devdox_ai_sonar.agent_supervisor import (
    MAX_RETRIES,
    _build_fixes_from_response,
    _build_validation_result,
    _check_syntax_impl,
    _check_testing_impl,
    _check_validation_result,
    _noop_reporter,
    _read_file_for_validation,
    _run_handler_impl,
    _run_syntax_check,
    _run_test_suite,
    _run_validation_loop,
    _validate_fix_impl,
    _validate_issue_impl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(rule="python:S3776", file_path="src/app.py"):
    issue = MagicMock()
    issue.rule = rule
    issue.file_path = file_path
    issue.file = file_path
    return issue


def _make_state(**overrides):
    defaults = {
        "issues": [_make_issue()],
        "project_path": Path("/proj"),
        "tmp_path": Path("/tmp"),
        "modified_content": "",
        "file_md": "",
        "validation": None,
        "fixes": [],
        "accepted_fixes": [],
        "error": "",
        "retry_count": 0,
        "error_feedback": "",
        "syntax_err": None,
        "_report": _noop_reporter,
    }
    defaults.update(overrides)
    return defaults


def _make_validation(is_valid=True, file_path=None, line_range=None):
    v = MagicMock()
    v.is_valid = is_valid
    v.error = "validation error"
    v.file_path = file_path or Path("/proj/src/app.py")
    v.line_range = line_range or {"first_line": 1, "last_line": 50}
    return v


def _make_fix(original_code="code", helper_code=""):
    fix = MagicMock()
    fix.original_code = original_code
    fix.helper_code = helper_code
    return fix


# ===========================================================================
# TestCheckValidationResult
# ===========================================================================


class TestCheckValidationResult:
    def test_invalid_returns_error(self):
        v = _make_validation(is_valid=False)
        result = _check_validation_result(v, _noop_reporter)
        assert "error" in result
        assert result["error"] == "validation error"

    def test_valid_but_no_file_path(self):
        v = _make_validation(is_valid=True)
        v.file_path = None
        result = _check_validation_result(v, _noop_reporter)
        assert "error" in result

    def test_valid_but_no_line_range(self):
        v = _make_validation(is_valid=True, line_range=None)
        v.line_range = None
        result = _check_validation_result(v, _noop_reporter)
        assert "error" in result

    def test_valid_returns_validation(self):
        v = _make_validation()
        result = _check_validation_result(v, _noop_reporter)
        assert "validation" in result
        assert result["validation"] is v


# ===========================================================================
# TestValidateIssueImpl
# ===========================================================================


class TestValidateIssueImpl:
    @pytest.mark.asyncio
    async def test_delegates_to_extractor(self):
        fixer = MagicMock()
        validation = _make_validation()
        mock_extractor = MagicMock()
        mock_extractor.validate_issue_group = AsyncMock(
            return_value=validation,
        )
        state = _make_state()
        with patch(
            "devdox_ai_sonar.agent_supervisor.IssueExtractor",
            return_value=mock_extractor,
        ):
            result = await _validate_issue_impl(state, fixer)
        assert result["validation"] is validation

    @pytest.mark.asyncio
    async def test_returns_error_on_invalid(self):
        fixer = MagicMock()
        validation = _make_validation(is_valid=False)
        mock_extractor = MagicMock()
        mock_extractor.validate_issue_group = AsyncMock(
            return_value=validation,
        )
        state = _make_state()
        with patch(
            "devdox_ai_sonar.agent_supervisor.IssueExtractor",
            return_value=mock_extractor,
        ):
            result = await _validate_issue_impl(state, fixer)
        assert "error" in result


# ===========================================================================
# TestRunHandlerImpl
# ===========================================================================


class TestRunHandlerImpl:
    @pytest.mark.asyncio
    async def test_returns_fixes(self):
        fixer = MagicMock()
        context = MagicMock()
        context.context_dict = {}
        fixer._prepare_fix_context = AsyncMock(return_value=context)
        fixer._build_fix_suggestion = MagicMock(return_value=["fix1"])
        fixer.write_explaination = MagicMock()

        handler = MagicMock()
        handler.MOIDY_LINE_RANGE = False
        handler.generate_fixes = AsyncMock(return_value=["resp"])

        registry = MagicMock()
        registry.get_handler.return_value = handler

        state = _make_state(validation=_make_validation())
        result = await _run_handler_impl(
            state, fixer, registry, "LLM fix",
        )
        assert "fixes" in result

    @pytest.mark.asyncio
    async def test_returns_error_no_context(self):
        fixer = MagicMock()
        fixer._prepare_fix_context = AsyncMock(return_value=None)
        registry = MagicMock()
        registry.get_handler.return_value = MagicMock()
        state = _make_state(validation=_make_validation())
        result = await _run_handler_impl(
            state, fixer, registry, "LLM fix",
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_error_no_fixes(self):
        fixer = MagicMock()
        context = MagicMock()
        context.context_dict = {}
        fixer._prepare_fix_context = AsyncMock(return_value=context)
        handler = MagicMock()
        handler.generate_fixes = AsyncMock(return_value=[])
        registry = MagicMock()
        registry.get_handler.return_value = handler
        state = _make_state(validation=_make_validation())
        result = await _run_handler_impl(
            state, fixer, registry, "LLM fix",
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_injects_error_feedback(self):
        fixer = MagicMock()
        context = MagicMock()
        context.context_dict = {}
        fixer._prepare_fix_context = AsyncMock(return_value=context)
        fixer._build_fix_suggestion = MagicMock(return_value=["fix1"])
        fixer.write_explaination = MagicMock()
        handler = MagicMock()
        handler.MOIDY_LINE_RANGE = False
        handler.generate_fixes = AsyncMock(return_value=["resp"])
        registry = MagicMock()
        registry.get_handler.return_value = handler

        state = _make_state(
            validation=_make_validation(),
            error_feedback="previous rejection",
        )
        await _run_handler_impl(state, fixer, registry, "LLM fix")
        assert context.context_dict["error_feedback"] == "previous rejection"


# ===========================================================================
# TestBuildFixesFromResponse
# ===========================================================================


class TestBuildFixesFromResponse:
    def test_builds_fix_list(self):
        fixer = MagicMock()
        fixer._build_fix_suggestion.return_value = ["fix1", "fix2"]
        fixer.write_explaination = MagicMock()
        handler = MagicMock()
        handler.MOIDY_LINE_RANGE = False
        context = MagicMock()
        state = _make_state(validation=_make_validation())
        result = _build_fixes_from_response(
            state, fixer, handler, ["resp"], context,
            _noop_reporter, "LLM fix",
        )
        assert result["fixes"] == ["fix1", "fix2"]

    def test_writes_docs_when_file_md(self):
        fixer = MagicMock()
        fixer._build_fix_suggestion.return_value = []
        handler = MagicMock()
        handler.MOIDY_LINE_RANGE = False
        context = MagicMock()
        state = _make_state(
            validation=_make_validation(),
            file_md="docs/fix.md",
        )
        _build_fixes_from_response(
            state, fixer, handler, ["resp"], context,
            _noop_reporter, "LLM fix",
        )
        fixer.write_explaination.assert_called_once()

    def test_skips_docs_when_no_file_md(self):
        fixer = MagicMock()
        fixer._build_fix_suggestion.return_value = []
        handler = MagicMock()
        handler.MOIDY_LINE_RANGE = False
        context = MagicMock()
        state = _make_state(validation=_make_validation(), file_md="")
        _build_fixes_from_response(
            state, fixer, handler, ["resp"], context,
            _noop_reporter, "LLM fix",
        )
        fixer.write_explaination.assert_not_called()


# ===========================================================================
# TestCheckSyntaxImpl
# ===========================================================================


class TestCheckSyntaxImpl:
    def test_no_fixes_returns_none(self):
        fixer = MagicMock()
        state = _make_state(fixes=[])
        result = _check_syntax_impl(state, fixer)
        assert result == {"syntax_err": None}

    def test_delegates_to_run_syntax_check(self):
        fixer = MagicMock()
        fix = _make_fix()
        state = _make_state(fixes=[fix])
        with patch(
            "devdox_ai_sonar.agent_supervisor._run_syntax_check",
            return_value={"syntax_err": None},
        ) as mock_run:
            result = _check_syntax_impl(state, fixer)
        mock_run.assert_called_once()
        assert result == {"syntax_err": None}


# ===========================================================================
# TestRunSyntaxCheck
# ===========================================================================


class TestRunSyntaxCheck:
    def test_syntax_ok(self, tmp_path):
        fixer = MagicMock()
        fixer.check_python_interpreter.return_value = (True, "")
        fix = _make_fix(original_code="x = 1")
        result = _run_syntax_check(fix, tmp_path, fixer, _noop_reporter)
        assert result == {"syntax_err": None}

    def test_syntax_error(self, tmp_path):
        fixer = MagicMock()
        fixer.check_python_interpreter.return_value = (False, "SyntaxError")
        fix = _make_fix(original_code="x = 1")
        result = _run_syntax_check(fix, tmp_path, fixer, _noop_reporter)
        assert result["syntax_err"] == "SyntaxError"

    def test_exception_returns_error(self, tmp_path):
        fixer = MagicMock()
        fixer.check_python_interpreter.side_effect = RuntimeError("boom")
        fix = _make_fix(original_code="x = 1")
        result = _run_syntax_check(fix, tmp_path, fixer, _noop_reporter)
        assert "boom" in result["syntax_err"]

    def test_includes_helper_code(self, tmp_path):
        fixer = MagicMock()
        fixer.check_python_interpreter.return_value = (True, "")
        fix = _make_fix(original_code="x = 1", helper_code="def h(): pass")
        _run_syntax_check(fix, tmp_path, fixer, _noop_reporter)
        written = (tmp_path / ".supervisor_tmp.py").exists()
        # File is cleaned up in finally, so just verify the call
        fixer.check_python_interpreter.assert_called_once()


# ===========================================================================
# TestCheckTestingImpl
# ===========================================================================


class TestCheckTestingImpl:
    def test_no_fixes_skips(self):
        fixer = MagicMock()
        state = _make_state(fixes=[])
        result = _check_testing_impl(state, fixer)
        assert result == {"syntax_err": None}

    def test_tests_dir_exists(self, tmp_path):
        (tmp_path / "tests").mkdir()
        fixer = MagicMock()
        fix = _make_fix()
        state = _make_state(fixes=[fix], project_path=tmp_path)
        with patch(
            "devdox_ai_sonar.agent_supervisor._run_test_suite",
            return_value={"syntax_err": None},
        ) as mock_run:
            result = _check_testing_impl(state, fixer)
        assert result == {"syntax_err": None}

    def test_falls_back_to_test_dir(self, tmp_path):
        (tmp_path / "test").mkdir()
        fixer = MagicMock()
        fix = _make_fix()
        state = _make_state(fixes=[fix], project_path=tmp_path)
        with patch(
            "devdox_ai_sonar.agent_supervisor._run_test_suite",
            return_value={"syntax_err": None},
        ) as mock_run:
            _check_testing_impl(state, fixer)
        call_path = mock_run.call_args[0][0]
        assert str(call_path).endswith("test")


# ===========================================================================
# TestRunTestSuite
# ===========================================================================


class TestRunTestSuite:
    def test_tests_pass(self, tmp_path):
        fixer = MagicMock()
        fixer.run_testing_cases.return_value = (True, "")
        result = _run_test_suite(
            tmp_path, fixer, _noop_reporter, tmp_path,
        )
        assert result == {"syntax_err": None}

    def test_tests_fail(self, tmp_path):
        fixer = MagicMock()
        fixer.run_testing_cases.return_value = (False, "AssertionError")
        result = _run_test_suite(
            tmp_path, fixer, _noop_reporter, tmp_path,
        )
        assert result["syntax_err"] == "AssertionError"

    def test_exception(self, tmp_path):
        fixer = MagicMock()
        fixer.run_testing_cases.side_effect = RuntimeError("crash")
        result = _run_test_suite(
            tmp_path, fixer, _noop_reporter, tmp_path,
        )
        assert "crash" in result["syntax_err"]


# ===========================================================================
# TestReadFileForValidation
# ===========================================================================


class TestReadFileForValidation:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello")
        validation = MagicMock()
        validation.file_path = f
        assert _read_file_for_validation(validation) == "hello"

    def test_returns_empty_on_missing_file(self):
        validation = MagicMock()
        validation.file_path = Path("/nonexistent/file.py")
        assert _read_file_for_validation(validation) == ""

    def test_returns_empty_on_none_validation(self):
        assert _read_file_for_validation(None) == ""

    def test_returns_empty_on_none_file_path(self):
        validation = MagicMock()
        validation.file_path = None
        assert _read_file_for_validation(validation) == ""


# ===========================================================================
# TestRunValidationLoop
# ===========================================================================


class TestRunValidationLoop:
    def test_all_accepted(self):
        fix = _make_fix()
        issue = _make_issue()
        vresult = MagicMock()
        vresult.should_apply = True
        vresult.final_fix = fix
        validator = MagicMock()
        validator.validate_fix.return_value = vresult

        accepted, explanation = _run_validation_loop(
            [fix], [issue], validator, "content", "",
        )
        assert len(accepted) == 1
        assert explanation == ""

    def test_all_rejected(self):
        fix = _make_fix()
        issue = _make_issue()
        vresult = MagicMock()
        vresult.should_apply = False
        vresult.explanation = "bad fix"
        validator = MagicMock()
        validator.validate_fix.return_value = vresult

        accepted, explanation = _run_validation_loop(
            [fix], [issue], validator, "content", "",
        )
        assert len(accepted) == 0
        assert explanation == "bad fix"

    def test_mixed_results(self):
        fix1, fix2 = _make_fix(), _make_fix()
        issue = _make_issue()
        r1 = MagicMock(should_apply=True, final_fix=fix1)
        r2 = MagicMock(should_apply=False, explanation="rejected")
        validator = MagicMock()
        validator.validate_fix.side_effect = [r1, r2]

        accepted, explanation = _run_validation_loop(
            [fix1, fix2], [issue], validator, "content", "",
        )
        assert len(accepted) == 1
        assert explanation == "rejected"


# ===========================================================================
# TestBuildValidationResult
# ===========================================================================


class TestBuildValidationResult:
    def test_accepted_returns_fixes(self):
        fix = _make_fix()
        state = _make_state()
        result = _build_validation_result(
            [fix], "", [fix], state, _noop_reporter,
        )
        assert result["accepted_fixes"] == [fix]
        assert result["error_feedback"] == ""

    def test_rejected_with_retries_left(self):
        state = _make_state(retry_count=0)
        result = _build_validation_result(
            [], "bad", [_make_fix()], state, _noop_reporter,
        )
        assert result["accepted_fixes"] == []
        assert "bad" in result["error_feedback"]

    def test_rejected_retries_exhausted(self):
        state = _make_state(retry_count=MAX_RETRIES)
        result = _build_validation_result(
            [], "bad", [_make_fix()], state, _noop_reporter,
        )
        assert result["accepted_fixes"] == []
        assert "bad" in result["error_feedback"]


# ===========================================================================
# TestValidateFixImpl
# ===========================================================================


class TestValidateFixImpl:
    def test_no_validator_accepts_all(self):
        fix = _make_fix()
        state = _make_state(fixes=[fix])
        result = _validate_fix_impl(state, validator=None)
        assert result["accepted_fixes"] == [fix]

    def test_no_fixes_returns_empty(self):
        validator = MagicMock()
        state = _make_state(fixes=[])
        result = _validate_fix_impl(state, validator)
        assert result["accepted_fixes"] == []

    def test_delegates_to_validation_loop(self):
        fix = _make_fix()
        vresult = MagicMock(should_apply=True, final_fix=fix)
        validator = MagicMock()
        validator.validate_fix.return_value = vresult
        validation = _make_validation()
        validation.file_path = MagicMock()
        validation.file_path.read_text.return_value = "code"
        state = _make_state(
            fixes=[fix], validation=validation,
        )
        result = _validate_fix_impl(state, validator)
        assert len(result["accepted_fixes"]) == 1
