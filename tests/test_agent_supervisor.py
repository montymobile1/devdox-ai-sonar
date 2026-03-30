"""Tests for agent_supervisor extracted implementation functions."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from devdox_ai_sonar.agent_supervisor import (
    MAX_RETRIES,
    _AUTOMATED_RULES,
    _accept_node,
    _error_node,
    _make_node,
    _make_sync_node,
    _noop_reporter,
    _retry_node,
    build_graph,
    build_supervisor,
    build_tools,
    route_after_syntax,
    route_after_test,
    route_after_validation,
    route_after_validation_fix,
    route_by_rule,
    AgentSupervisor,
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
        "test_err": None,
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
# _noop_reporter
# ===========================================================================


class TestNoopReporter:
    def test_accepts_any_string(self):
        # Must not raise
        _noop_reporter("anything")
        _noop_reporter("")
        _noop_reporter("[bold red]error[/bold red]")


# ===========================================================================
# route_after_validation
# ===========================================================================


class TestRouteAfterValidation:
    def test_returns_error_when_error_set(self):
        state = _make_state(error="something went wrong")
        assert route_after_validation(state) == "error"

    def test_returns_route_rule_when_no_error(self):
        state = _make_state(error="")
        assert route_after_validation(state) == "route_rule"

    def test_returns_route_rule_when_error_key_missing(self):
        state = {"issues": [_make_issue()]}
        assert route_after_validation(state) == "route_rule"






# ===========================================================================
# TestRouteAfterValidation
# ===========================================================================


class TestRouteAfterValidation:
    def test_returns_error_when_error_set(self):
        state = _make_state(error="something went wrong")
        assert route_after_validation(state) == "error"

    def test_returns_route_rule_when_no_error(self):
        state = _make_state(error="")
        assert route_after_validation(state) == "route_rule"

    def test_returns_route_rule_when_error_key_missing(self):
        state = {"issues": [_make_issue()]}
        assert route_after_validation(state) == "route_rule"


# ===========================================================================
# TestRouteByRule
# ===========================================================================


class TestRouteByRule:
    def test_automated_rule(self):
        issue = _make_issue(rule="python:S1192")
        state = _make_state(issues=[issue])
        assert route_by_rule(state) == "automated"

    def test_llm_rule(self):
        issue = _make_issue(rule="python:S3776")
        state = _make_state(issues=[issue])
        assert route_by_rule(state) == "llm"


# ===========================================================================
# TestRouteAfterSyntax
# ===========================================================================


class TestRouteAfterSyntax:
    def test_syntax_ok_routes_to_testing(self):
        state = _make_state(syntax_err=None)
        assert route_after_syntax(state) == "testing"

    def test_syntax_err_routes_to_validate(self):
        state = _make_state(syntax_err="SyntaxError line 5")
        assert route_after_syntax(state) == "validate"

    def test_syntax_ok_automated_rule_routes_to_accept(self):
        state = _make_state(syntax_err=None, issues=[_make_issue(rule="python:S1192")])
        assert route_after_syntax(state) == "accept"


# ===========================================================================
# TestRouteAfterTest
# ===========================================================================


class TestRouteAfterTest:
    def test_tests_passed_routes_to_accept(self):
        state = _make_state(test_err=None)
        assert route_after_test(state) == "accept"

    def test_tests_failed_routes_to_validate(self):
        state = _make_state(test_err="test failure output")
        assert route_after_test(state) == "validate"


# ===========================================================================
# TestRouteAfterValidationFix
# ===========================================================================


class TestRouteAfterValidationFix:
    def test_accepted_fixes_routes_to_accept(self):
        fix = _make_fix()
        state = _make_state(accepted_fixes=[fix])
        assert route_after_validation_fix(state) == "accept"

    def test_rejected_llm_rule_retries_left(self):
        issue = _make_issue(rule="python:S3776")
        state = _make_state(
            issues=[issue], accepted_fixes=[], retry_count=0,
        )
        assert route_after_validation_fix(state) == "retry"

    def test_rejected_llm_rule_retries_exhausted(self):
        issue = _make_issue(rule="python:S3776")
        state = _make_state(
            issues=[issue], accepted_fixes=[], retry_count=MAX_RETRIES,
        )
        assert route_after_validation_fix(state) == "error"

    def test_rejected_automated_rule_routes_to_error(self):
        issue = _make_issue(rule="python:S1192")
        state = _make_state(
            issues=[issue], accepted_fixes=[], retry_count=0,
        )
        assert route_after_validation_fix(state) == "error"


# ===========================================================================
# TestNodeWrappers
# ===========================================================================


class TestMakeNode:
    @pytest.mark.asyncio
    async def test_make_node_merges_result(self):
        mock_tool = MagicMock()
        mock_tool.ainvoke = AsyncMock(return_value={"fixes": ["fix1"]})
        node = _make_node(mock_tool)
        state = _make_state()
        result = await node(state)
        assert result["fixes"] == ["fix1"]
        mock_tool.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_node_preserves_existing_state(self):
        mock_tool = MagicMock()
        mock_tool.ainvoke = AsyncMock(return_value={"fixes": ["fix1"]})
        node = _make_node(mock_tool)
        state = _make_state(error="kept")
        result = await node(state)
        assert result["error"] == "kept"
        assert result["fixes"] == ["fix1"]


class TestMakeSyncNode:
    def test_make_sync_node_merges_result(self):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = {"syntax_err": None}
        node = _make_sync_node(mock_tool)
        state = _make_state()
        result = node(state)
        assert result["syntax_err"] is None
        mock_tool.invoke.assert_called_once()


# ===========================================================================
# TestTerminalNodes
# ===========================================================================


class TestAcceptNode:
    def test_copies_fixes_to_accepted_when_empty(self):
        fix = _make_fix()
        state = _make_state(fixes=[fix], accepted_fixes=[])
        result = _accept_node(state)
        assert result["accepted_fixes"] == [fix]


    def test_copies_fixes_to_accepted(self):
        fix = _make_fix()
        state = _make_state(fixes=[fix], accepted_fixes=[])
        result = _accept_node(state)
        assert result["accepted_fixes"] == [fix]




class TestErrorNode:
    def test_clears_accepted_fixes(self):
        state = _make_state(error="fatal", accepted_fixes=[_make_fix()])
        result = _error_node(state)
        assert result["accepted_fixes"] == []

    def test_reports_error(self):
        reported = []
        state = _make_state(error="boom", _report=reported.append)
        _error_node(state)
        assert any("boom" in msg for msg in reported)


class TestRetryNode:
    def test_increments_retry_count(self):
        state = _make_state(retry_count=0)
        result = _retry_node(state)
        assert result["retry_count"] == 1

    def test_increments_from_existing_count(self):
        state = _make_state(retry_count=1)
        result = _retry_node(state)
        assert result["retry_count"] == 2


# ===========================================================================
# TestBuildTools
# ===========================================================================


class TestBuildTools:
    def test_returns_all_tool_keys(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)
        expected_keys = {
            "validate_issue_tool",
            "run_automated_tool",
            "run_llm_tool",
            "check_syntax_tool",
            "check_testing_tool",
            "validate_fix_tool",
        }
        assert set(tools.keys()) == expected_keys

    def test_returns_all_required_tool_keys(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)
        expected_keys = {
            "validate_issue_tool",
            "run_automated_tool",
            "run_llm_tool",
            "check_syntax_tool",
            "check_testing_tool",
            "validate_fix_tool",
        }
        assert set(tools.keys()) == expected_keys


    @pytest.mark.asyncio
    async def test_validate_issue_tool_delegates(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)
        validation = _make_validation()
        mock_extractor = MagicMock()
        mock_extractor.validate_issue_group = AsyncMock(return_value=validation)
        state = _make_state()
        with patch(
            "devdox_ai_sonar.agent_supervisor.IssueExtractor",
            return_value=mock_extractor,
        ):
            result = await tools["validate_issue_tool"].ainvoke({"state": state})
        assert "validation" in result

    @pytest.mark.asyncio
    async def test_validate_issue_tool_returns_error_on_invalid(self):
        fixer = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)

        validation = _make_validation(is_valid=False)
        mock_extractor = MagicMock()
        mock_extractor.validate_issue_group = AsyncMock(return_value=validation)
        state = _make_state()

        with patch(
                "devdox_ai_sonar.agent_supervisor.IssueExtractor",
                return_value=mock_extractor,
        ):
            result = await tools["validate_issue_tool"].ainvoke({"state": state})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_validate_issue_tool_returns_error_when_file_path_none(self):
        fixer = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)

        validation = _make_validation(is_valid=True)
        validation.file_path = None
        mock_extractor = MagicMock()
        mock_extractor.validate_issue_group = AsyncMock(return_value=validation)
        state = _make_state()

        with patch(
                "devdox_ai_sonar.agent_supervisor.IssueExtractor",
                return_value=mock_extractor,
        ):
            result = await tools["validate_issue_tool"].ainvoke({"state": state})

        assert "error" in result

    @pytest.mark.asyncio
    async def test_validate_issue_tool_returns_error_when_line_range_none(self):
        fixer = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)

        validation = _make_validation(is_valid=True, line_range=None)
        validation.line_range = None
        mock_extractor = MagicMock()
        mock_extractor.validate_issue_group = AsyncMock(return_value=validation)
        state = _make_state()

        with patch(
                "devdox_ai_sonar.agent_supervisor.IssueExtractor",
                return_value=mock_extractor,
        ):
            result = await tools["validate_issue_tool"].ainvoke({"state": state})

        assert "error" in result

    def test_check_syntax_tool_delegates(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)
        state = _make_state(fixes=[])
        result = tools["check_syntax_tool"].invoke({"state": state})
        assert result == {"syntax_err": None}

    def test_check_syntax_tool_no_fixes_returns_none(self):
        fixer = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)
        state = _make_state(fixes=[])
        result = tools["check_syntax_tool"].invoke({"state": state})
        assert result == {"syntax_err": None}

    def test_check_syntax_tool_passes_on_valid_code(self, tmp_path):
        fixer = MagicMock()
        fixer.check_python_interpreter.return_value = (True, "")
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)

        fix = _make_fix(original_code="x = 1")
        fix.file_path = "app.py"

        # Write a real file so open() inside the tool succeeds
        (tmp_path / "app.py").write_text("x = 1\n")

        state = _make_state(fixes=[fix], project_path=tmp_path)

        with patch("devdox_ai_sonar.agent_supervisor.apply_single_fix", return_value=(True, ["x = 1\n"])):
            result = tools["check_syntax_tool"].invoke({"state": state})

        assert result == {"syntax_err": None}

    def test_check_syntax_tool_returns_error_on_bad_syntax(self, tmp_path):
        fixer = MagicMock()
        fixer.check_python_interpreter.return_value = (False, "SyntaxError: invalid syntax")
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)

        fix = _make_fix(original_code="def :")
        fix.file_path = "app.py"
        (tmp_path / "app.py").write_text("def :\n")

        state = _make_state(fixes=[fix], project_path=tmp_path)

        with patch("devdox_ai_sonar.agent_supervisor.apply_single_fix", return_value=(True, ["def :\n"])):
            result = tools["check_syntax_tool"].invoke({"state": state})

        assert "SyntaxError" in result["syntax_err"]

        # ------------------------------------------------------------------
        # check_testing_tool
        # ------------------------------------------------------------------

        def test_check_testing_tool_no_fixes_skips(self):
            fixer = MagicMock()
            registry = MagicMock()
            tools = build_tools(fixer, None, registry)
            state = _make_state(fixes=[], skip_tests=False)
            result = tools["check_testing_tool"].invoke({"state": state})
            assert result == {"test_err": None}

        def test_check_testing_tool_skip_tests_flag(self):
            fixer = MagicMock()
            registry = MagicMock()
            tools = build_tools(fixer, None, registry)
            state = _make_state(fixes=[_make_fix()], skip_tests=True)
            result = tools["check_testing_tool"].invoke({"state": state})
            assert result == {"test_err": None}

        def test_check_testing_tool_passes_when_tests_pass(self, tmp_path):
            fixer = MagicMock()
            fixer.run_testing_cases.return_value = (True, "")
            registry = MagicMock()
            tools = build_tools(fixer, None, registry)

            (tmp_path / "tests").mkdir()
            fix = _make_fix()
            state = _make_state(fixes=[fix], project_path=tmp_path, skip_tests=False)
            result = tools["check_testing_tool"].invoke({"state": state})
            assert result == {"test_err": None}

        def test_check_testing_tool_fails_when_tests_fail(self, tmp_path):
            fixer = MagicMock()
            fixer.run_testing_cases.return_value = (False, "AssertionError: 1 != 2")
            registry = MagicMock()
            tools = build_tools(fixer, None, registry)

            (tmp_path / "tests").mkdir()
            fix = _make_fix()
            state = _make_state(fixes=[fix], project_path=tmp_path, skip_tests=False)
            result = tools["check_testing_tool"].invoke({"state": state})
            assert "AssertionError" in result["test_err"]

        def test_check_testing_tool_falls_back_to_test_dir(self, tmp_path):
            fixer = MagicMock()
            fixer.run_testing_cases.return_value = (True, "")
            registry = MagicMock()
            tools = build_tools(fixer, None, registry)

            (tmp_path / "test").mkdir()  # "test" not "tests"
            fix = _make_fix()
            state = _make_state(fixes=[fix], project_path=tmp_path, skip_tests=False)
            tools["check_testing_tool"].invoke({"state": state})
            call_path = fixer.run_testing_cases.call_args[0][0]
            assert call_path.endswith("test")

    # ------------------------------------------------------------------
    # validate_fix_tool — no validator
    # ------------------------------------------------------------------

    def test_validate_fix_tool_no_validator_accepts_all(self):
        fixer = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)  # validator=None

        fix = _make_fix()
        state = _make_state(fixes=[fix])
        result = tools["validate_fix_tool"].invoke({"state": state})
        assert result["accepted_fixes"] == [fix]

    def test_validate_fix_tool_no_fixes_returns_empty(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)

        state = _make_state(fixes=[], validation=_make_validation())
        result = tools["validate_fix_tool"].invoke({"state": state})
        assert result["accepted_fixes"] == []

    def test_validate_fix_tool_accepts_approved_fixes(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)

        fix = _make_fix()

        validation = _make_validation()
        state = _make_state(fixes=[fix], validation=validation)
        result = tools["validate_fix_tool"].invoke({"state": state})
        assert len(result["accepted_fixes"]) == 1
        assert result["error_feedback"] == ""


    def test_check_testing_tool_delegates(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)
        state = _make_state(fixes=[])
        result = tools["check_testing_tool"].invoke({"state": state})
        assert result == {"test_err": None}

    def test_forwards_test_err_to_validator(self):
        fixer = MagicMock()
        fix = _make_fix()
        issue = _make_issue()
        vresult = MagicMock(should_apply=True, final_fix=fix)
        validator = MagicMock()
        validator.validate_fix.return_value = vresult
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)

        validation = _make_validation()
        validation.file_path = MagicMock()
        validation.file_path.read_text.return_value = "code"

        test_error = "FAILED tests/test_app.py::test_login - AssertionError"
        state = _make_state(
            fixes=[fix], issues=[issue], validation=validation,
            test_err=test_error,
        )
        tools["validate_fix_tool"].invoke({"state": state})
        validator.validate_fix.assert_called_once_with(
            fix=fix, issue=issue,
            file_content="code", new_error_msg="",
            test_err=test_error,
        )

    def test_validate_fix_tool_no_validator(self):
        fixer = MagicMock()
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)
        fix = _make_fix()
        state = _make_state(fixes=[fix])
        result = tools["validate_fix_tool"].invoke({"state": state})
        assert result["accepted_fixes"] == [fix]

    @pytest.mark.asyncio
    async def test_run_llm_tool_with_retry(self):
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

        tools = build_tools(fixer, None, registry)
        state = _make_state(
            validation=_make_validation(), retry_count=1,
        )
        result = await tools["run_llm_tool"].ainvoke({"state": state})
        assert "fixes" in result

    @pytest.mark.asyncio
    async def test_run_automated_tool_delegates(self):
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

        tools = build_tools(fixer, None, registry)
        state = _make_state(validation=_make_validation())
        result = await tools["run_automated_tool"].ainvoke({"state": state})
        assert "fixes" in result

    @pytest.mark.asyncio
    async def test_run_automated_tool_no_context(self):
        fixer = MagicMock()
        fixer._prepare_fix_context = AsyncMock(return_value=None)
        registry = MagicMock()
        registry.get_handler.return_value = MagicMock()
        tools = build_tools(fixer, None, registry)
        state = _make_state(validation=_make_validation())
        result = await tools["run_automated_tool"].ainvoke({"state": state})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_automated_tool_no_fixes(self):
        fixer = MagicMock()
        context = MagicMock()
        context.context_dict = {}
        fixer._prepare_fix_context = AsyncMock(return_value=context)
        handler = MagicMock()
        handler.generate_fixes = AsyncMock(return_value=[])
        registry = MagicMock()
        registry.get_handler.return_value = handler
        tools = build_tools(fixer, None, registry)
        state = _make_state(validation=_make_validation())
        result = await tools["run_automated_tool"].ainvoke({"state": state})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_llm_tool_no_context(self):
        fixer = MagicMock()
        fixer._prepare_fix_context = AsyncMock(return_value=None)
        registry = MagicMock()
        registry.get_handler.return_value = MagicMock()
        tools = build_tools(fixer, None, registry)
        state = _make_state(validation=_make_validation())
        result = await tools["run_llm_tool"].ainvoke({"state": state})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_llm_tool_no_fixes(self):
        fixer = MagicMock()
        context = MagicMock()
        context.context_dict = {}
        fixer._prepare_fix_context = AsyncMock(return_value=context)
        handler = MagicMock()
        handler.generate_fixes = AsyncMock(return_value=[])
        registry = MagicMock()
        registry.get_handler.return_value = handler
        tools = build_tools(fixer, None, registry)
        state = _make_state(validation=_make_validation())
        result = await tools["run_llm_tool"].ainvoke({"state": state})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_llm_tool_injects_error_feedback(self):
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
        tools = build_tools(fixer, None, registry)
        state = _make_state(
            validation=_make_validation(),
            error_feedback="previous rejection",
        )
        await tools["run_llm_tool"].ainvoke({"state": state})
        assert context.context_dict["error_feedback"] == "previous rejection"

    def test_check_syntax_tool_exception(self, tmp_path):
        fixer = MagicMock()
        fixer.check_python_interpreter.side_effect = RuntimeError("boom")
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)
        fix = _make_fix(original_code="x = 1")
        fix.file_path = "app.py"
        (tmp_path / "app.py").write_text("x = 1\n")
        state = _make_state(fixes=[fix], project_path=tmp_path)
        with patch("devdox_ai_sonar.agent_supervisor.apply_single_fix", return_value=(True, ["x = 1\n"])):
            result = tools["check_syntax_tool"].invoke({"state": state})
        assert "boom" in result["syntax_err"]

    def test_check_testing_tool_tests_pass(self, tmp_path):
        fixer = MagicMock()
        fixer.run_testing_cases.return_value = (True, "")
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)
        (tmp_path / "tests").mkdir()
        fix = _make_fix()
        state = _make_state(fixes=[fix], project_path=tmp_path, skip_tests=False)
        result = tools["check_testing_tool"].invoke({"state": state})
        assert result == {"test_err": None}

    def test_check_testing_tool_tests_fail(self, tmp_path):
        fixer = MagicMock()
        fixer.run_testing_cases.return_value = (False, "AssertionError")
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)
        (tmp_path / "tests").mkdir()
        fix = _make_fix()
        state = _make_state(fixes=[fix], project_path=tmp_path, skip_tests=False)
        result = tools["check_testing_tool"].invoke({"state": state})
        assert result["test_err"] == "AssertionError"

    def test_check_testing_tool_exception(self, tmp_path):
        fixer = MagicMock()
        fixer.run_testing_cases.side_effect = RuntimeError("crash")
        registry = MagicMock()
        tools = build_tools(fixer, None, registry)
        (tmp_path / "tests").mkdir()
        fix = _make_fix()
        state = _make_state(fixes=[fix], project_path=tmp_path, skip_tests=False)
        result = tools["check_testing_tool"].invoke({"state": state})
        assert "crash" in result["test_err"]

    def test_validate_fix_tool_all_rejected_retries_left(self):
        fixer = MagicMock()
        fix = _make_fix()
        vresult = MagicMock(should_apply=False, explanation="bad fix",
                            confidence=0.5)
        validator = MagicMock()
        validator.validate_fix.return_value = vresult
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)
        validation = _make_validation()
        validation.file_path = MagicMock()
        validation.file_path.read_text.return_value = "code"
        state = _make_state(
            fixes=[fix], validation=validation, retry_count=0,
        )
        result = tools["validate_fix_tool"].invoke({"state": state})
        assert result["accepted_fixes"] == []
        assert "bad fix" in result["error_feedback"]

    def test_validate_fix_tool_all_rejected_retries_exhausted(self):
        fixer = MagicMock()
        fix = _make_fix()
        vresult = MagicMock(should_apply=False, explanation="still bad",
                            confidence=0.3)
        validator = MagicMock()
        validator.validate_fix.return_value = vresult
        registry = MagicMock()
        tools = build_tools(fixer, validator, registry)
        validation = _make_validation()
        validation.file_path = MagicMock()
        validation.file_path.read_text.return_value = "code"
        state = _make_state(
            fixes=[fix], validation=validation, retry_count=MAX_RETRIES,
        )
        result = tools["validate_fix_tool"].invoke({"state": state})
        assert result["accepted_fixes"] == []
        assert "still bad" in result["error_feedback"]


# ===========================================================================
# TestBuildGraph
# ===========================================================================


class TestBuildGraph:
    def test_returns_compiled_graph(self):
        fixer = MagicMock()
        validator = MagicMock()
        registry = MagicMock()
        graph = build_graph(fixer, validator, registry)
        # A compiled graph should have an ainvoke method
        assert hasattr(graph, "ainvoke")


# ===========================================================================
# TestAgentSupervisor
# ===========================================================================


class TestAgentSupervisor:
    def test_init(self):
        fixer = MagicMock()
        validator = MagicMock()
        sup = AgentSupervisor(fixer=fixer, validator=validator)
        assert sup.fixer is fixer
        assert sup.validator is validator
        assert sup._registry is not None
        assert sup._graph is not None

    @pytest.mark.asyncio
    async def test_run_no_accepted_fixes(self):
        fixer = MagicMock()
        validator = MagicMock()
        sup = AgentSupervisor(fixer=fixer, validator=validator)
        # Mock _graph.ainvoke to return state with empty accepted_fixes
        sup._graph = MagicMock()
        sup._graph.ainvoke = AsyncMock(
            return_value={"accepted_fixes": []},
        )
        issue = _make_issue(rule="python:S3776", file_path="src/app.py")
        result = await sup.run(
            issues=[issue],
            project_path=Path("/proj"),
            tmp_path=Path("/tmp"),
        )
        assert result.total_fixes_attempted == 1
        assert result.successful_fixes == []

    @pytest.mark.asyncio
    async def test_run_with_accepted_fixes(self):
        fixer = MagicMock()
        fix = _make_fix()
        fix_result = MagicMock()
        fixer.apply_fixes = AsyncMock(return_value=fix_result)
        validator = MagicMock()
        sup = AgentSupervisor(fixer=fixer, validator=validator)
        sup._graph = MagicMock()
        sup._graph.ainvoke = AsyncMock(
            return_value={"accepted_fixes": [fix]},
        )
        issue = _make_issue(rule="python:S3776", file_path="src/app.py")
        result = await sup.run(
            issues=[issue],
            project_path=Path("/proj"),
            tmp_path=Path("/tmp"),
        )
        fixer.apply_fixes.assert_called_once()
        assert result is fix_result

    @pytest.mark.asyncio
    async def test_run_for_file_group_exception(self):
        fixer = MagicMock()
        validator = MagicMock()
        sup = AgentSupervisor(fixer=fixer, validator=validator)
        sup._graph = MagicMock()
        sup._graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph exploded"))
        issue = _make_issue(rule="python:S3776", file_path="src/app.py")
        accepted = await sup._run_for_file_group(
            issues=[issue],
            project_path=Path("/proj"),
            tmp_path=Path("/tmp"),
            modified_content="",
            file_md="",
        )
        assert accepted == []

    @pytest.mark.asyncio
    async def test_run_passes_status_reporter(self):
        fixer = MagicMock()
        validator = MagicMock()
        sup = AgentSupervisor(fixer=fixer, validator=validator)
        sup._graph = MagicMock()
        sup._graph.ainvoke = AsyncMock(
            return_value={"accepted_fixes": []},
        )
        reporter = MagicMock()
        issue = _make_issue(rule="python:S3776", file_path="src/app.py")
        await sup.run(
            issues=[issue],
            project_path=Path("/proj"),
            tmp_path=Path("/tmp"),
            status_reporter=reporter,
        )
        # Verify ainvoke was called with state containing the reporter
        call_args = sup._graph.ainvoke.call_args
        initial_state = call_args[0][0]
        assert initial_state["_report"] is reporter

    @pytest.mark.asyncio
    async def test_run_passes_skip_tests_true(self):
        fixer = MagicMock()
        validator = MagicMock()
        sup = AgentSupervisor(fixer=fixer, validator=validator)
        sup._graph = MagicMock()
        sup._graph.ainvoke = AsyncMock(
            return_value={"accepted_fixes": []},
        )
        issue = _make_issue(rule="python:S3776", file_path="src/app.py")
        await sup.run(
            issues=[issue],
            project_path=Path("/proj"),
            tmp_path=Path("/tmp"),
            skip_tests=True,
        )
        call_args = sup._graph.ainvoke.call_args
        initial_state = call_args[0][0]
        assert initial_state["skip_tests"] is True

    @pytest.mark.asyncio
    async def test_run_passes_skip_tests_false(self):
        fixer = MagicMock()
        validator = MagicMock()
        sup = AgentSupervisor(fixer=fixer, validator=validator)
        sup._graph = MagicMock()
        sup._graph.ainvoke = AsyncMock(
            return_value={"accepted_fixes": []},
        )
        issue = _make_issue(rule="python:S3776", file_path="src/app.py")
        await sup.run(
            issues=[issue],
            project_path=Path("/proj"),
            tmp_path=Path("/tmp"),
            skip_tests=False,
        )
        call_args = sup._graph.ainvoke.call_args
        initial_state = call_args[0][0]
        assert initial_state["skip_tests"] is False


# ===========================================================================
# TestGroupByFile
# ===========================================================================


class TestGroupByFile:
    def test_groups_by_file_path(self):
        issue1 = _make_issue(rule="python:S3776", file_path="src/app.py")
        issue2 = _make_issue(rule="python:S1481", file_path="src/app.py")
        issue3 = _make_issue(rule="python:S3776", file_path="src/util.py")
        groups = AgentSupervisor._group_by_file(
            [issue1, issue2, issue3], Path("/proj"),
        )
        assert len(groups) == 2
        assert len(groups["/proj/src/app.py"]) == 2
        assert len(groups["/proj/src/util.py"]) == 1

    def test_skips_non_python_files(self):
        issue = _make_issue(rule="python:S3776", file_path="src/app.js")
        groups = AgentSupervisor._group_by_file([issue], Path("/proj"))
        assert len(groups) == 0

    def test_skips_issues_without_file_path(self):
        issue = MagicMock()
        issue.rule = "python:S3776"
        # file_path property returns None
        del issue.file_path
        groups = AgentSupervisor._group_by_file([issue], Path("/proj"))
        assert len(groups) == 0


# ===========================================================================
# TestBuildSupervisor
# ===========================================================================


class TestBuildSupervisor:
    @patch("devdox_ai_sonar.agent_supervisor.FixValidator")
    @patch("devdox_ai_sonar.agent_supervisor.LLMFixer")
    def test_builds_with_validator(self, mock_fixer_cls, mock_validator_cls):
        mock_fixer_cls.return_value = MagicMock()
        mock_validator_cls.return_value = MagicMock()
        with patch(
            "devdox_ai_sonar.agent_supervisor.build_graph",
            return_value=MagicMock(),
        ):
            sup = build_supervisor(
                provider="openai",
                model="gpt-4",
                api_key="sk-test",
                use_validator=True,
            )
        assert isinstance(sup, AgentSupervisor)
        mock_validator_cls.assert_called_once()

    @patch("devdox_ai_sonar.agent_supervisor.LLMFixer")
    def test_builds_without_validator(self, mock_fixer_cls):
        mock_fixer_cls.return_value = MagicMock()
        with patch(
            "devdox_ai_sonar.agent_supervisor.build_graph",
            return_value=MagicMock(),
        ):
            sup = build_supervisor(
                provider="openai",
                model="gpt-4",
                api_key="sk-test",
                use_validator=False,
            )
        assert isinstance(sup, AgentSupervisor)
        assert sup.validator is None

    @patch("devdox_ai_sonar.agent_supervisor.FixValidator")
    @patch("devdox_ai_sonar.agent_supervisor.LLMFixer")
    def test_validator_uses_fallback_params(self, mock_fixer_cls, mock_validator_cls):
        mock_fixer_cls.return_value = MagicMock()
        mock_validator_cls.return_value = MagicMock()
        with patch(
            "devdox_ai_sonar.agent_supervisor.build_graph",
            return_value=MagicMock(),
        ):
            build_supervisor(
                provider="openai",
                model="gpt-4",
                api_key="sk-test",
                use_validator=True,
                validator_provider=None,
                validator_model=None,
                validator_api_key=None,
                min_confidence=0.8,
            )
        # Should fall back to main provider/model/api_key
        mock_validator_cls.assert_called_once_with(
            provider="openai",
            model="gpt-4",
            api_key="sk-test",
            min_confidence_threshold=0.8,
        )
