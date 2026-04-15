"""
Integration tests for the agent_supervisor LangGraph pipeline.

These tests build the real compiled graph via build_graph() and invoke it
end-to-end with mocked fixer/validator/registry dependencies. They exercise
graph wiring and routing decisions that pure unit tests cannot catch.

Regression coverage for A1 (DEV-107): handler failures in run_automated /
run_llm must route to the error node, not silently fall through to
check_syntax and report a successful run with zero fixes.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from devdox_ai_sonar.agent_supervisor import (
    _noop_reporter,
    build_graph,
)


def _make_issue(rule, file_path="src/app.py"):
    issue = MagicMock()
    issue.rule = rule
    issue.file_path = file_path
    issue.file = file_path
    return issue


def _make_validation():
    v = MagicMock()
    v.is_valid = True
    v.error = ""
    v.file_path = Path("/proj/src/app.py")
    v.line_range = {"first_line": 1, "last_line": 50}
    return v


def _make_state(rule):
    return {
        "issues": [_make_issue(rule)],
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


class TestGraphHandlerFailureRouting:
    """
    A1 regression: when run_automated / run_llm sets state['error'], the graph
    must route to the error node instead of continuing to check_syntax.
    """

    @pytest.mark.asyncio
    async def test_llm_context_prep_failure_reaches_error_node(self):
        fixer = MagicMock()
        fixer.file_reader = MagicMock()
        fixer._prepare_fix_context = AsyncMock(return_value=None)
        validator = MagicMock()
        registry = MagicMock()
        registry.get_handler.return_value = MagicMock()

        with patch(
            "devdox_ai_sonar.agent_supervisor.IssueExtractor"
        ) as extractor_cls, patch(
            "devdox_ai_sonar.agent_supervisor._check_syntax_impl"
        ) as check_syntax_spy:
            extractor_cls.return_value.validate_issue_group = AsyncMock(
                return_value=_make_validation()
            )

            graph = build_graph(fixer, validator, registry)
            result = await graph.ainvoke(_make_state(rule="python:S3776"))

            check_syntax_spy.assert_not_called()

        assert result["error"]
        assert "Could not prepare fix context" in result["error"]
        assert result["accepted_fixes"] == []

    @pytest.mark.asyncio
    async def test_automated_context_prep_failure_reaches_error_node(self):
        fixer = MagicMock()
        fixer.file_reader = MagicMock()
        fixer._prepare_fix_context = AsyncMock(return_value=None)
        validator = MagicMock()
        registry = MagicMock()
        registry.get_handler.return_value = MagicMock()

        with patch(
            "devdox_ai_sonar.agent_supervisor.IssueExtractor"
        ) as extractor_cls, patch(
            "devdox_ai_sonar.agent_supervisor._check_syntax_impl"
        ) as check_syntax_spy:
            extractor_cls.return_value.validate_issue_group = AsyncMock(
                return_value=_make_validation()
            )

            graph = build_graph(fixer, validator, registry)
            result = await graph.ainvoke(_make_state(rule="python:S1192"))

            check_syntax_spy.assert_not_called()

        assert result["error"]
        assert result["accepted_fixes"] == []

    @pytest.mark.asyncio
    async def test_llm_handler_no_fixes_returned_reaches_error_node(self):
        fixer = MagicMock()
        fixer.file_reader = MagicMock()
        context = MagicMock()
        context.context_dict = {}
        fixer._prepare_fix_context = AsyncMock(return_value=context)
        fixer.write_explaination = MagicMock()
        validator = MagicMock()

        handler = MagicMock()
        handler.generate_fixes = AsyncMock(return_value=[])
        registry = MagicMock()
        registry.get_handler.return_value = handler

        with patch(
            "devdox_ai_sonar.agent_supervisor.IssueExtractor"
        ) as extractor_cls, patch(
            "devdox_ai_sonar.agent_supervisor._check_syntax_impl"
        ) as check_syntax_spy:
            extractor_cls.return_value.validate_issue_group = AsyncMock(
                return_value=_make_validation()
            )

            graph = build_graph(fixer, validator, registry)
            result = await graph.ainvoke(_make_state(rule="python:S3776"))

            check_syntax_spy.assert_not_called()

        assert result["error"]
        assert "returned no fixes" in result["error"]
        assert result["accepted_fixes"] == []
