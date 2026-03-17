"""
agent_supervisor.py — LangGraph-based orchestration for SonarCloud fix generation.

Graph structure (one run per file group):

    START
      │
      ▼
  [validate_issue_tool]          ← Tool: validate issue group, extract line range
      │
      ▼
  route_after_validation()       ← Conditional edge
      ├─ "error"   ──────────────► END   (invalid issues / file not found)
      └─ "route_rule" ───────────►
                                  │
                              route_by_rule()    ← Conditional edge
                                  ├─ "automated" ─► [run_automated_tool]
                                  └─ "llm"       ─► [run_llm_tool]
                                        │                  │
                                        └─────────┬────────┘
                                                  ▼
                                        [check_syntax_tool]
                                                  │
                                                  ▼
                                        route_after_syntax()  ← Conditional edge
                                            ├─ "validate" ──► [validate_fix_tool]
                                            └─ "testing"  ──► [check_testing_tool]
                                                    │                  │
                                            route_after_validation_fix() ← Conditional edge
                                                │   ├─ "accept" ──► END
                                                │   ├─ "retry"  ──► [run_llm_tool]
                                                │   └─ "error"  ──► END
                                                │
                                            route_after_test()  ← Conditional edge
                                                ├─ "accept"   ──► END   (tests passed)
                                                └─ "validate" ──► [validate_fix_tool]

State keys:
    issues          — input issues for this file group
    project_path    — project root
    tmp_path        — tmp path for extractor
    modified_content
    file_md
    validation      — ValidationResult from extractor (set by validate_issue_tool)
    fixes           — List[FixSuggestion] (set by run_automated_tool / run_llm_tool)
    syntax_err      — str | None (set by check_syntax_tool / check_testing_tool)
    accepted_fixes  — List[FixSuggestion] (final output)
    error           — str (set on any fatal error)
    retry_count     — int (incremented on each retry)
    error_feedback  — str (REJECTED explanation passed back to llm tool on retry)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from devdox_ai_sonar.fix_validator import FixValidator
from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.models.sonar import FixResult, FixSuggestion, SonarIssue, SonarSecurityIssue
from devdox_ai_sonar.services.extractor import IssueExtractor
from devdox_ai_sonar.services.rule_handler import RuleHandlerRegistry

logger = logging.getLogger(__name__)

IssueUnion = Union[SonarIssue, SonarSecurityIssue]

# Rules handled by deterministic AST transforms — no LLM needed.
_AUTOMATED_RULES = {
    "python:S7503",
    "python:S117",
    "python:S1172",
    "python:S1542",
    "python:S1192",
}

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class SonarState(TypedDict, total=False):
    # Inputs
    issues: List[IssueUnion]
    project_path: Path
    tmp_path: Path
    modified_content: str
    file_md: str
    # Set during graph execution
    validation: Any          # ValidationResult from IssueExtractor
    fixes: List[FixSuggestion]
    syntax_err: Optional[str]
    accepted_fixes: List[FixSuggestion]
    error: str
    retry_count: int
    error_feedback: str


# ---------------------------------------------------------------------------
# Tool factory — tools are closures over fixer / validator / registry
# ---------------------------------------------------------------------------


def build_tools(
    fixer: LLMFixer,
    validator: Optional[FixValidator],
    registry: RuleHandlerRegistry,
) -> Dict[str, Any]:
    """
    Build all graph tools. Tools read/write SonarState via their closure.
    Each tool receives the current state dict and returns a state patch.
    """

    # ------------------------------------------------------------------ #
    # Tool 1: validate_issue_tool
    # Purpose: validate issue group, extract file path + line range
    # Conditional edge after: route_after_validation
    # ------------------------------------------------------------------ #

    @tool
    async def validate_issue_tool(state: dict) -> dict:  # type: ignore[misc]
        """
        Validate that all issues belong to the same Python file and extract
        the line range to fix. Sets state['validation'] on success,
        state['error'] on failure.
        """
        issues: List[IssueUnion] = state["issues"]
        project_path: Path = state["project_path"]
        tmp_path: Path = state["tmp_path"]

        extractor = IssueExtractor(fixer.file_reader)
        validation = await extractor.validate_issue_group(issues, tmp_path, project_path)

        if not validation.is_valid:
            logger.error("Issue validation failed: %s", validation.error)
            return {"error": validation.error}

        if validation.file_path is None or validation.line_range is None:
            err = "Validation succeeded but file_path or line_range is None"
            logger.error(err)
            return {"error": err}

        logger.info(
            "Validated: file=%s lines=%s-%s",
            validation.file_path,
            validation.line_range.get("first_line"),
            validation.line_range.get("last_line"),
        )
        return {"validation": validation}

    # ------------------------------------------------------------------ #
    # Tool 2a: run_automated_tool
    # Purpose: run deterministic AST handler (no LLM)
    # ------------------------------------------------------------------ #

    @tool
    async def run_automated_tool(state: dict) -> dict:  # type: ignore[misc]
        """
        Run the deterministic (AST-based) rule handler for rules that don't
        need an LLM. Sets state['fixes'] on success, state['error'] on failure.
        """
        issues: List[IssueUnion] = state["issues"]
        project_path: Path = state["project_path"]
        validation = state["validation"]
        modified_content: str = state.get("modified_content", "")
        file_md: str = state.get("file_md", "")

        handler = registry.get_handler(issues[0].rule)

        logger.info("Automated handler: %s for rule %s", handler.__class__.__name__, issues[0].rule)

        context = await fixer._prepare_fix_context(
            validation.file_path, validation.line_range, modified_content
        )
        if not context:

            return {"error": f"Could not prepare fix context for {issues[0].rule}"}

        fix_response_lst = await handler.generate_fixes(
            issues, context, project_path, validation.file_path, llm_caller=fixer
        )
        if not fix_response_lst:

            return {"error": f"{handler.__class__.__name__} returned no fixes"}

        if file_md:
            fixer.write_explaination(
                project_path / file_md, fix_response_lst, issues,
                context.context_dict, project_path=project_path,
            )

        modify_line_range = getattr(handler, "MOIDY_LINE_RANGE", False)
        fixes = fixer._build_fix_suggestion(
            fix_response_lst, context, validation.file_path,
            project_path, validation.line_range, modify_line_range,
        )
        logger.info("Automated handler produced %d fix(es)", len(fixes))

        return {"fixes": fixes}

    # ------------------------------------------------------------------ #
    # Tool 2b: run_llm_tool
    # Purpose: run LLM-backed handler, optionally with error_feedback on retry
    # ------------------------------------------------------------------ #

    @tool
    async def run_llm_tool(state: dict) -> dict:  # type: ignore[misc]
        """
        Run the LLM-backed rule handler. On retry, injects error_feedback into
        the fix context so the LLM can self-correct.
        Sets state['fixes'] on success, state['error'] on failure.
        """

        issues: List[IssueUnion] = state["issues"]
        project_path: Path = state["project_path"]
        validation = state["validation"]
        modified_content: str = state.get("modified_content", "")
        file_md: str = state.get("file_md", "")
        error_feedback: str = state.get("error_feedback", "")

        handler = registry.get_handler(issues[0].rule)
        logger.info(
            "LLM handler: %s for rule %s (retry=%d feedback=%s)",
            handler.__class__.__name__, issues[0].rule,
            state.get("retry_count", 0), bool(error_feedback),
        )

        context = await fixer._prepare_fix_context(
            validation.file_path, validation.line_range, modified_content
        )
        if not context:
            return {"error": f"Could not prepare fix context for {issues[0].rule}"}

        if error_feedback:
            context.context_dict["error_feedback"] = error_feedback

        fix_response_lst = await handler.generate_fixes(
            issues, context, project_path, validation.file_path, llm_caller=fixer
        )
        if not fix_response_lst:
            return {"error": f"{handler.__class__.__name__} returned no fixes"}

        if file_md:
            fixer.write_explaination(
                project_path / file_md, fix_response_lst, issues,
                context.context_dict, project_path=project_path,
            )

        modify_line_range = getattr(handler, "MOIDY_LINE_RANGE", False)
        fixes = fixer._build_fix_suggestion(
            fix_response_lst, context, validation.file_path,
            project_path, validation.line_range, modify_line_range,
        )
        logger.info("LLM handler produced %d fix(es)", len(fixes))
        return {"fixes": fixes}

    # ------------------------------------------------------------------ #
    # Tool 3: check_syntax_tool
    # Purpose: run Python interpreter on the fix to detect syntax errors
    # Conditional edge after: route_after_syntax
    # ------------------------------------------------------------------ #

    @tool
    def check_syntax_tool(state: dict) -> dict:  # type: ignore[misc]
        """
        Write the first fix to a temp file and run the Python interpreter.
        Sets state['syntax_err'] = None on success, or an error string on failure.
        """

        fixes: List[FixSuggestion] = state.get("fixes", [])
        project_path: Path = state["project_path"]

        if not fixes:

            logger.warning("check_syntax_tool: no fixes to check")
            return {"syntax_err": None}

        fix = fixes[0]
        code = fix.original_code or ""
        if fix.helper_code:
            code = fix.helper_code + "\n" + code

        tmp_file = project_path / ".supervisor_tmp.py"
        try:
            tmp_file.write_text(code, encoding="utf-8")
            ok, err = fixer.check_python_interpreter(str(tmp_file))

            if ok:
                logger.info("Syntax check passed")
                return {"syntax_err": None}
            logger.warning("Syntax check failed: %s", err)
            return {"syntax_err": err}
        except Exception as exc:
            logger.error("Syntax check exception: %s", exc)
            return {"syntax_err": str(exc)}
        finally:
            tmp_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Tool 4: check_testing_tool
    # Purpose: run the project test suite after syntax passes
    # Conditional edge after: route_after_test
    # ------------------------------------------------------------------ #

    @tool
    def check_testing_tool(state: dict) -> dict:  # type: ignore[misc]
        """
        Run the project test suite (pytest) to verify the fix does not break
        any existing tests.
        Sets state['syntax_err'] = None when all tests pass, or a pytest
        error string when tests fail — which will forward to validate_fix.
        """

        fixes: List[FixSuggestion] = state.get("fixes", [])
        project_path: Path = state["project_path"]

        if not fixes:
            logger.warning("check_testing_tool: no fixes to check — skipping tests")
            return {"syntax_err": None}

        # Prefer a 'tests' directory; fall back to 'test' for legacy layouts.
        testing_path = project_path / "tests"
        if not testing_path.exists():
            testing_path = project_path / "test"

        try:
            ok, err = fixer.run_testing_cases(str(testing_path))

            if ok:
                logger.info("Test suite passed")
                return {"syntax_err": None}
            logger.warning("Test suite failed: %s", err)
            return {"syntax_err": err}
        except Exception as exc:
            logger.error("Test suite exception: %s", exc)
            return {"syntax_err": str(exc)}
        finally:
            logger.debug("check_testing_tool finished for %s", project_path)

    # ------------------------------------------------------------------ #
    # Tool 5: validate_fix_tool
    # Purpose: run FixValidator on each fix when syntax error is present
    # Conditional edge after: route_after_validation_fix
    # ------------------------------------------------------------------ #

    @tool
    def validate_fix_tool(state: dict) -> dict:  # type: ignore[misc]
        """
        Run FixValidator on each fix. Uses syntax_err as additional context.
        Sets state['accepted_fixes'] with fixes that passed, and
        state['error_feedback'] with rejection reason for retry.
        """

        if validator is None:
            # No validator configured — accept all fixes
            return {
                "accepted_fixes": state.get("fixes", []),
                "error_feedback": "",
            }

        fixes: List[FixSuggestion] = state.get("fixes", [])
        issues: List[IssueUnion] = state["issues"]
        validation = state["validation"]
        syntax_err: str = state.get("syntax_err") or ""

        if not fixes:
            return {"accepted_fixes": [], "error_feedback": "No fixes to validate"}

        # Read file content for validator context
        file_content = ""
        if validation and validation.file_path:
            try:
                file_content = validation.file_path.read_text(encoding="utf-8")
            except OSError:
                pass

        accepted: List[FixSuggestion] = []
        last_explanation = ""

        for fix in fixes:
            result = validator.validate_fix(
                fix=fix,
                issue=issues[0],
                file_content=file_content,
                new_error_msg=syntax_err,
            )

            if result.should_apply:
                accepted.append(result.final_fix)
            else:

                last_explanation = result.explanation
                logger.warning(
                    "Validator rejected fix for %s: %s", issues[0].rule, result.explanation
                )


        if accepted:
            logger.info("Validator accepted %d/%d fix(es)", len(accepted), len(fixes))

            return {"accepted_fixes": accepted, "error_feedback": ""}

        logger.warning("All fixes rejected: %s", last_explanation)
        return {
            "accepted_fixes": [],
            "error_feedback": f"Fix rejected — {last_explanation}",
        }

    return {
        "validate_issue_tool": validate_issue_tool,
        "run_automated_tool": run_automated_tool,
        "run_llm_tool": run_llm_tool,
        "check_syntax_tool": check_syntax_tool,
        "check_testing_tool": check_testing_tool,
        "validate_fix_tool": validate_fix_tool,
    }


# ---------------------------------------------------------------------------
# Conditional edge routing functions
# ---------------------------------------------------------------------------


def route_after_validation(state: SonarState) -> str:
    """
    Edge after validate_issue_tool.
    - "error"      → fatal, stop
    - "route_rule" → proceed to rule routing
    """
    if state.get("error"):
        logger.info("Routing → error (validation failed)")
        return "error"
    logger.info("Routing → route_rule")
    return "route_rule"


def route_by_rule(state: SonarState) -> str:
    """
    Edge from route_rule decision node.
    Checks the first issue's rule against _AUTOMATED_RULES.
    - "automated" → deterministic AST handler
    - "llm"       → LLM-backed handler
    """
    rule = state["issues"][0].rule
    if rule in _AUTOMATED_RULES:
        logger.info("Routing → automated (rule=%s)", rule)
        return "automated"
    logger.info("Routing → llm (rule=%s)", rule)
    return "llm"


def route_after_syntax(state: SonarState) -> str:
    """
    Edge after check_syntax_tool.
    - "testing"  → syntax OK, run the test suite before accepting
    - "validate" → syntax error present, run FixValidator immediately
    """
    if state.get("syntax_err") is None:
        logger.info("Routing → testing (syntax OK)")
        return "testing"
    logger.info("Routing → validate (syntax_err=%s)", state["syntax_err"])
    return "validate"


def route_after_test(state: SonarState) -> str:
    """
    Edge after check_testing_tool.
    - "accept"   → all tests passed, fix is good
    - "validate" → tests failed, forward failure output to FixValidator
    """
    if state.get("syntax_err") is None:
        logger.info("Routing → accept (tests passed)")
        return "accept"
    logger.info("Routing → validate (test failure: %s)", state["syntax_err"])
    return "validate"


def route_after_validation_fix(state: SonarState) -> str:
    """
    Edge after validate_fix_tool.
    - "accept" → fixes accepted (APPROVED or MODIFIED)
    - "retry"  → fixes rejected AND rule is LLM-backed AND retries remain
    - "error"  → fixes rejected, automated rule or retries exhausted
    """
    if state.get("accepted_fixes"):
        logger.info("Routing → accept (validator approved)")
        return "accept"

    rule = state["issues"][0].rule
    retry_count = state.get("retry_count", 0)

    if rule not in _AUTOMATED_RULES and retry_count < MAX_RETRIES:
        logger.info(
            "Routing → retry (retry %d/%d for rule %s)",
            retry_count + 1, MAX_RETRIES, rule,
        )
        return "retry"

    logger.info("Routing → error (rejected, no retries left or automated rule)")
    return "error"


# ---------------------------------------------------------------------------
# Node wrappers — adapt tool call signature to graph node signature
# ---------------------------------------------------------------------------


def _make_node(tool_fn: Any):
    """Wrap a @tool function so it can be used as a graph node."""
    async def node(state: SonarState) -> SonarState:
        result = await tool_fn.ainvoke({"state": dict(state)})
        # tool_fn returns a dict patch — merge into state
        return {**state, **result}  # type: ignore[return-value]
    return node


def _make_sync_node(tool_fn: Any):
    """Wrap a synchronous @tool function as a graph node."""
    def node(state: SonarState) -> SonarState:
        result = tool_fn.invoke({"state": dict(state)})
        return {**state, **result}  # type: ignore[return-value]
    return node


def _accept_node(state: SonarState) -> SonarState:
    """Terminal node: copy fixes → accepted_fixes if not already set."""
    if not state.get("accepted_fixes"):
        return {**state, "accepted_fixes": state.get("fixes", [])}  # type: ignore[return-value]
    return state


def _error_node(state: SonarState) -> SonarState:
    """Terminal node: ensure accepted_fixes is empty on error."""

    return {**state, "accepted_fixes": []}  # type: ignore[return-value]


def _retry_node(state: SonarState) -> SonarState:
    """Increment retry_count before looping back to run_llm."""

    return {**state, "retry_count": state.get("retry_count", 0) + 1}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    fixer: LLMFixer,
    validator: Optional[FixValidator],
    registry: RuleHandlerRegistry,
) -> Any:
    """
    Build and compile the LangGraph for one file group.

    Nodes:
        validate_issue   — validate issues, extract line range
        route_rule       — pass-through decision node (conditional edge picks next)
        run_automated    — AST handler (no LLM)
        run_llm          — LLM handler (with optional error_feedback on retry)
        check_syntax     — Python interpreter check
        check_testing    — pytest test suite check
        validate_fix     — FixValidator (on syntax error OR test failure)
        accept           — terminal: fixes accepted
        error            — terminal: fixes rejected / fatal error
        retry            — increment counter then loop to run_llm

    Edges:
        START → validate_issue
        validate_issue → route_after_validation → {error, route_rule}
        route_rule → route_by_rule → {automated, llm}
        run_automated → check_syntax
        run_llm → check_syntax
        check_syntax → route_after_syntax → {testing, validate}
        check_testing → route_after_test → {accept, validate}
        validate_fix → route_after_validation_fix → {accept, retry, error}
        retry → run_llm
        accept → END
        error  → END
    """
    tools = build_tools(fixer, validator, registry)

    graph = StateGraph(SonarState)

    # ── Nodes ────────────────────────────────────────────────────────────
    graph.add_node("validate_issue", _make_node(tools["validate_issue_tool"]))
    graph.add_node("route_rule", lambda s: s)              # pass-through
    graph.add_node("run_automated", _make_node(tools["run_automated_tool"]))
    graph.add_node("run_llm", _make_node(tools["run_llm_tool"]))
    graph.add_node("check_syntax", _make_sync_node(tools["check_syntax_tool"]))
    graph.add_node("check_testing", _make_sync_node(tools["check_testing_tool"]))
    graph.add_node("validate_fix", _make_sync_node(tools["validate_fix_tool"]))
    graph.add_node("accept", _accept_node)
    graph.add_node("error", _error_node)
    graph.add_node("retry", _retry_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    graph.set_entry_point("validate_issue")

    graph.add_conditional_edges(
        "validate_issue",
        route_after_validation,
        {"error": "error", "route_rule": "route_rule"},
    )
    graph.add_conditional_edges(
        "route_rule",
        route_by_rule,
        {"automated": "run_automated", "llm": "run_llm"},
    )

    graph.add_edge("run_automated", "check_syntax")
    graph.add_edge("run_llm", "check_syntax")

    # Bug fix: syntax OK → run tests first; syntax error → skip tests, go straight to validator
    graph.add_conditional_edges(
        "check_syntax",
        route_after_syntax,
        {"testing": "check_testing", "validate": "validate_fix"},
    )

    # Bug fix: tests pass → accept; tests fail → send failure output to validator
    graph.add_conditional_edges(
        "check_testing",
        route_after_test,
        {"accept": "accept", "validate": "validate_fix"},
    )

    graph.add_conditional_edges(
        "validate_fix",
        route_after_validation_fix,
        {"accept": "accept", "retry": "retry", "error": "error"},
    )

    graph.add_edge("retry", "run_llm")
    graph.add_edge("accept", END)
    graph.add_edge("error", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentSupervisor
# ---------------------------------------------------------------------------


class AgentSupervisor:
    """
    Runs the LangGraph fix pipeline for each file group in a batch of issues.

    Usage:
        supervisor = build_supervisor(provider="openai", api_key="sk-...")
        result = await supervisor.run(issues, project_path, tmp_path)
    """

    def __init__(
        self,
        fixer: LLMFixer,
        validator: Optional[FixValidator] = None,
    ) -> None:
        self.fixer = fixer
        self.validator = validator
        self._registry = RuleHandlerRegistry()
        self._graph = build_graph(fixer, validator, self._registry)

    async def run(
        self,
        issues: List[IssueUnion],
        project_path: Path,
        tmp_path: Path,
        create_backup: bool = True,
        dry_run: bool = False,
        modified_content: str = "",
        file_md: str = "",
    ) -> FixResult:
        """Entry point — groups issues by file then runs the graph per group."""
        all_accepted: List[FixSuggestion] = []

        for file_issues in self._group_by_file(issues, project_path).values():
            accepted = await self._run_for_file_group(
                file_issues, project_path, tmp_path, modified_content, file_md
            )
            all_accepted.extend(accepted)

        if not all_accepted:
            return FixResult(project_path=project_path, total_fixes_attempted=len(issues))

        return await self.fixer.apply_fixes(
            all_accepted, project_path,
            create_backup=create_backup, dry_run=dry_run,
        )

    async def _run_for_file_group(
        self,
        issues: List[IssueUnion],
        project_path: Path,
        tmp_path: Path,
        modified_content: str,
        file_md: str,
    ) -> List[FixSuggestion]:
        initial_state: SonarState = {
            "issues": issues,
            "project_path": project_path,
            "tmp_path": tmp_path,
            "modified_content": modified_content,
            "file_md": file_md,
            "fixes": [],
            "accepted_fixes": [],
            "error": "",
            "retry_count": 0,
            "error_feedback": "",
            "syntax_err": None,
        }

        try:
            final_state = await self._graph.ainvoke(
                initial_state,
                config={"recursion_limit": 30},
            )
            return final_state.get("accepted_fixes", [])
        except Exception as exc:
            logger.error(
                "Graph execution failed for rule=%s: %s",
                issues[0].rule if issues else "?", exc, exc_info=True,
            )
            return []

    @staticmethod
    def _group_by_file(
        issues: List[IssueUnion], project_path: Path
    ) -> Dict[str, List[IssueUnion]]:
        groups: Dict[str, List[IssueUnion]] = {}
        for issue in issues:
            fp = getattr(issue, "file_path", None)
            if not fp:
                continue
            resolved = str(project_path / fp)
            if not resolved.endswith(".py"):
                continue
            groups.setdefault(resolved, []).append(issue)
        return groups


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_supervisor(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    context_lines: int = 10,
    use_validator: bool = True,
    validator_provider: Optional[str] = None,
    validator_model: Optional[str] = None,
    validator_api_key: Optional[str] = None,
    min_confidence: float = 0.7,
) -> AgentSupervisor:
    """
    Factory — builds AgentSupervisor with fixer and optional validator.

    use_validator=False → validate_fix_tool accepts all fixes immediately,
                          no LLM validation cost.
    use_validator=True  → FixValidator runs when a syntax error or test
                          failure is detected.
    """
    fixer = LLMFixer(
        provider=provider,
        model=model,
        api_key=api_key,
        context_lines=context_lines,
    )

    validator: Optional[FixValidator] = None
    if use_validator:
        validator = FixValidator(
            provider=validator_provider or provider,
            model=validator_model or model,
            api_key=validator_api_key or api_key,
            min_confidence_threshold=min_confidence,
        )

    return AgentSupervisor(fixer=fixer, validator=validator)
