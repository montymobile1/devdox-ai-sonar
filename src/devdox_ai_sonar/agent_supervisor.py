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
    syntax_err      — str | None (set by check_syntax_tool)
    test_err        — str | None (set by check_testing_tool)
    accepted_fixes  — List[FixSuggestion] (final output)
    error           — str (set on any fatal error)
    retry_count     — int (incremented on each retry)
    error_feedback  — str (REJECTED explanation passed back to llm tool on retry)
    _report         — Callable[[str], None] | None  (phase status reporter)
"""
import logging
import operator
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union, Annotated

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from devdox_ai_sonar.fix_validator import FixValidator
from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.utils.file_indentation import apply_single_fix
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

# Type alias for the phase-reporter callback.
# Receives a Rich-markup string, e.g. "[cyan]⚙ Checking syntax…[/cyan]"
StatusReporter = Callable[[str], None]


def keep_last(left: Any, right: Any) -> Any:
    """
    Always keep the most recent write.

    Use for scalar fields that:
      a) are set once in initial_state and should never be overridden, OR
      b) are deliberately updated by a node during the run.

    In both cases, the last explicit write is the correct one.
    Do NOT use keep_first — it silently keeps a wrong default (e.g. Path("."))
    if a node accidentally emits the field before the real value is set.
    """
    return right

def _noop_reporter(msg: str) -> None:  # noqa: D401
    """Default reporter — does nothing (keeps graph hermetic when no UI is attached)."""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class SonarState(TypedDict, total=False):
    # Inputs
    issues: Annotated[List[IssueUnion], keep_last]
    fixes: Annotated[List[FixSuggestion], keep_last]  # single declaration
    accepted_fixes: Annotated[List[FixSuggestion], keep_last]
    project_path: Annotated[Path, keep_last]
    tmp_path: Annotated[Path, keep_last]
    modified_content: Annotated[str, keep_last]
    file_md: Annotated[str, keep_last]
    # Set during graph execution
    validation: Annotated[Any, keep_last]
    syntax_err: Annotated[Optional[str], keep_last]
    test_err: Annotated[Optional[str], keep_last]
    skip_tests: Annotated[bool, keep_last]
    error: Annotated[str, keep_last]
    retry_count: Annotated[int, keep_last]
    error_feedback: Annotated[str, keep_last]
    # UI callback — not serialised, lives only in-process
    _report: Annotated[Optional[StatusReporter], keep_last]


# ---------------------------------------------------------------------------
# Tool helper functions (extracted for low cognitive complexity)
# ---------------------------------------------------------------------------


async def _validate_issue_impl(state: dict, fixer: LLMFixer) -> dict:
    """Validate issues and extract file/line info."""
    report: StatusReporter = state.get("_report") or _noop_reporter
    issues: List[IssueUnion] = state["issues"]

    rule = issues[0].rule if issues else "?"
    file_hint = getattr(issues[0], "file_path", getattr(issues[0], "file", "?")) if issues else "?"
    report(f"[bold cyan]  ▶ Phase 1/6 — Validating issue[/bold cyan]  rule=[yellow]{rule}[/yellow]  file=[dim]{file_hint}[/dim]")

    extractor = IssueExtractor(fixer.file_reader)
    validation = await extractor.validate_issue_group(
        issues, state["tmp_path"], state["project_path"],
    )
    return _check_validation_result(validation, report)


def _check_validation_result(validation: Any, report: StatusReporter) -> dict:
    """Check validation result and return state patch."""
    if not validation.is_valid:
        logger.error("Issue validation failed: %s", validation.error)
        report(f"[red]  ✗ Validation failed:[/red] {validation.error}")
        return {"error": validation.error}
    if validation.file_path is None or validation.line_range is None:
        err = "Validation succeeded but file_path or line_range is None"
        logger.error(err)
        report(f"[red]  ✗ {err}[/red]")
        return {"error": err}
    report(
        f"[green]  ✓ Issue validated[/green]  "
        f"lines {validation.line_range.get('first_line')}–{validation.line_range.get('last_line')}"
    )
    logger.info(
        "Validated: file=%s lines=%s-%s",
        validation.file_path,
        validation.line_range.get("first_line"),
        validation.line_range.get("last_line"),
    )
    return {"validation": validation}


async def _run_handler_impl(
    state: dict, fixer: LLMFixer, registry: RuleHandlerRegistry,
    handler_label: str,
) -> dict:
    """Shared logic for both automated and LLM handler tools."""
    report: StatusReporter = state.get("_report") or _noop_reporter
    issues: List[IssueUnion] = state["issues"]
    validation = state["validation"]
    handler = registry.get_handler(issues[0].rule)

    report(f"[bold cyan]  ▶ Phase 2/6 — {handler_label}[/bold cyan]  handler=[yellow]{handler.__class__.__name__}[/yellow]")
    logger.info("%s: %s for rule %s", handler_label, handler.__class__.__name__, issues[0].rule)

    context = await _prepare_context(fixer, validation, state, report)
    if not context:
        return {"error": f"Could not prepare fix context for {issues[0].rule}"}

    return await _generate_and_build_fixes(handler, fixer, state, context, report, handler_label)


async def _prepare_context(fixer: LLMFixer, validation: Any, state: dict,
                           report: StatusReporter) -> Any:
    """Prepare fix context and inject error feedback if present."""
    context = await fixer._prepare_fix_context(
        validation.file_path, validation.line_range, state.get("modified_content", ""),
    )
    if not context:
        report("[red]  ✗ Could not prepare fix context[/red]")
        return None
    error_feedback = state.get("error_feedback", "")
    if error_feedback:
        context.context_dict["error_feedback"] = error_feedback
        report("[dim]    ↳ Injecting previous rejection feedback into prompt[/dim]")
    return context


async def _generate_and_build_fixes(
    handler: Any, fixer: LLMFixer, state: dict,
    context: Any, report: StatusReporter, handler_label: str,
) -> dict:
    """Generate fixes from handler and build suggestions."""
    issues = state["issues"]
    project_path = state["project_path"]
    fix_response_lst = await handler.generate_fixes(
        issues, context, project_path, state["validation"].file_path, llm_caller=fixer,
    )
    if not fix_response_lst:
        report(f"[red]  ✗ {handler_label} returned no fixes[/red]")
        return {"error": f"{handler.__class__.__name__} returned no fixes"}

    _write_explanation_if_needed(fixer, state, fix_response_lst, issues, context)
    return _build_fix_suggestions(fixer, state, fix_response_lst, context, handler, report, handler_label)


def _write_explanation_if_needed(fixer: LLMFixer, state: dict, fix_response_lst: list,
                                 issues: list, context: Any) -> None:
    """Write explanation docs if file_md is set."""
    file_md: str = state.get("file_md", "")
    if file_md:
        fixer.write_explaination(
            state["project_path"] / file_md, fix_response_lst, issues,
            context.context_dict, project_path=state["project_path"],
        )


def _build_fix_suggestions(fixer: LLMFixer, state: dict, fix_response_lst: list,
                           context: Any, handler: Any, report: StatusReporter,
                           handler_label: str) -> dict:
    """Build FixSuggestion list from handler response."""
    modify_line_range = getattr(handler, "MOIDY_LINE_RANGE", False)
    fixes = fixer._build_fix_suggestion(
        fix_response_lst, context, state["validation"].file_path,
        state["project_path"], state["validation"].line_range, modify_line_range,
    )
    report(f"[green]  ✓ {handler_label} generated[/green]  ({len(fixes)} suggestion(s))")
    logger.info("%s produced %d fix(es)", handler_label, len(fixes))
    return {"fixes": fixes}


def _read_and_apply_fix(fix: Any, project_path: Path) -> list:
    """Read source file, apply fix, return modified lines."""
    file_path = project_path / fix.file_path
    with open(str(file_path), "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.splitlines(keepends=True)
    _, lines = apply_single_fix(lines, fix)
    return lines


def _check_syntax_for_fix(fix: Any, project_path: Path, fixer: LLMFixer,
                          report: StatusReporter) -> Optional[dict]:
    """Check syntax for a single fix. Returns error dict or None on success."""
    lines = _read_and_apply_fix(fix, project_path)
    report("[bold cyan]  ▶ Phase 3/6 — Checking syntax[/bold cyan]  (python -m py_compile)")
    return _run_py_compile(lines, project_path, fixer, report)


def _run_py_compile(lines: list, project_path: Path, fixer: LLMFixer,
                    report: StatusReporter) -> Optional[dict]:
    """Write lines to temp file and run py_compile."""
    tmp_file = project_path / ".supervisor_tmp.py"
    try:
        tmp_file.write_text("".join(lines), encoding="utf-8")
        ok, err = fixer.check_python_interpreter(str(tmp_file))
        if ok:
            logger.info("Syntax check passed")
            report("[green]  ✓ Syntax OK[/green]")
            return None
        logger.warning("Syntax check failed: %s", err)
        report(f"[yellow]  ⚠ Syntax error detected[/yellow] — forwarding to validator\n[dim]    {err}[/dim]")
        return {"syntax_err": err}
    except Exception as exc:
        logger.error("Syntax check exception: %s", exc)
        report(f"[red]  ✗ Syntax check raised exception:[/red] {exc}")
        return {"syntax_err": str(exc)}
    finally:
        tmp_file.unlink(missing_ok=True)


def _run_test_suite(testing_path: Path, fixer: LLMFixer,
                    report: StatusReporter, project_path: Path) -> dict:
    """Execute pytest and return test_err result."""
    try:
        ok, err = fixer.run_testing_cases(str(testing_path))
        if ok:
            logger.info("Test suite passed")
            report("[green]  ✓ All tests passed[/green]")
            return {"test_err": None}
        logger.warning("Test suite failed: %s", err)
        report("[yellow]  ⚠ Tests failed[/yellow] — forwarding to validator")
        return {"test_err": err}
    except Exception as exc:
        logger.error("Test suite exception: %s", exc)
        report(f"[red]  ✗ Test runner raised exception:[/red] {exc}")
        return {"test_err": str(exc)}
    finally:
        logger.debug("check_testing_tool finished for %s", project_path)


def _run_validation_loop(
    fixes: List[FixSuggestion], issues: List[IssueUnion],
    validator: FixValidator, file_content: str,
    syntax_err: str, test_err: str,
) -> tuple:
    """Validate each fix and return (accepted, last_explanation)."""
    accepted: List[FixSuggestion] = []
    last_explanation = ""
    for fix in fixes:
        result = validator.validate_fix(
            fix=fix, issue=issues[0],
            file_content=file_content, new_error_msg=syntax_err,
            test_err=test_err,
        )
        if result.should_apply:
            accepted.append(result.final_fix)
        else:
            last_explanation = result.explanation
            logger.warning("Validator rejected fix for %s: %s", issues[0].rule, result.explanation)
    return accepted, last_explanation


def _build_rejection_result(
    last_explanation: str, state: dict, report: StatusReporter,
) -> dict:
    """Build result dict when all fixes are rejected."""
    logger.warning("All fixes rejected: %s", last_explanation)
    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRIES:
        report(
            f"[yellow]  ⚠ All fixes rejected — will retry ({retry_count + 1}/{MAX_RETRIES})[/yellow]\n"
            f"[dim]    Reason: {last_explanation}[/dim]"
        )
    else:
        report(
            f"[red]  ✗ All fixes rejected — retries exhausted[/red]\n"
            f"[dim]    Reason: {last_explanation}[/dim]"
        )
    return {"accepted_fixes": [], "error_feedback": f"Fix rejected — {last_explanation}"}


def _read_file_for_validation(validation: Any) -> str:
    """Read source file content for validator context."""
    if validation and validation.file_path:
        try:
            return validation.file_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return ""


# ---------------------------------------------------------------------------
# Tool factory — thin @tool wrappers delegating to helper functions
# ---------------------------------------------------------------------------


def _check_syntax_impl(state: dict, fixer: LLMFixer) -> dict:
    """Check syntax of each fix via py_compile."""
    report: StatusReporter = state.get("_report") or _noop_reporter
    fixes: List[FixSuggestion] = state.get("fixes", [])
    if not fixes:
        logger.warning("check_syntax_tool: no fixes to check")
        report("[dim]    ↳ No fixes to check — skipping[/dim]")
        return {"syntax_err": None}
    for fix in fixes:
        err_result = _check_syntax_for_fix(fix, state["project_path"], fixer, report)
        if err_result is not None:
            return err_result
    return {"syntax_err": None}


def _check_testing_impl(state: dict, fixer: LLMFixer) -> dict:
    """Run pytest to verify fix does not break existing tests."""
    if state.get("skip_tests", False):
        return {"test_err": None}
    report: StatusReporter = state.get("_report") or _noop_reporter
    fixes: List[FixSuggestion] = state.get("fixes", [])
    project_path: Path = state["project_path"]
    report("[bold cyan]  ▶ Phase 4/5 — Running test suite[/bold cyan]  (pytest)")
    if not fixes:
        logger.warning("check_testing_tool: no fixes to check — skipping tests")
        report("[dim]    ↳ No fixes to check — skipping tests[/dim]")
        return {"test_err": None}
    testing_path = project_path / "tests"
    if not testing_path.exists():
        testing_path = project_path / "test"
    report(f"[dim]    ↳ Test directory: {testing_path}[/dim]")
    return _run_test_suite(testing_path, fixer, report, project_path)


def _validate_fix_impl(state: dict, validator: Optional[FixValidator]) -> dict:
    """Run FixValidator on each fix and collect accepted/rejected results."""
    report: StatusReporter = state.get("_report") or _noop_reporter
    report("[bold cyan]  ▶ Phase 5/5 — Validating fix quality[/bold cyan]  (FixValidator)")
    if validator is None:
        report("[dim]    ↳ No validator configured — accepting all fixes[/dim]")
        return {"accepted_fixes": state.get("fixes", []), "error_feedback": ""}
    fixes: List[FixSuggestion] = state.get("fixes", [])
    if not fixes:
        report("[yellow]  ⚠ No fixes to validate[/yellow]")
        return {"accepted_fixes": [], "error_feedback": "No fixes to validate"}
    file_content = _read_file_for_validation(state.get("validation"))
    accepted, last_explanation = _run_validation_loop(
        fixes, state["issues"], validator, file_content,
        state.get("syntax_err") or "", state.get("test_err") or "",
    )
    if accepted:
        logger.info("Validator accepted %d/%d fix(es)", len(accepted), len(fixes))
        report(f"[green]  ✓ Validator accepted {len(accepted)}/{len(fixes)} fix(es)[/green]")
        return {"accepted_fixes": accepted, "error_feedback": ""}
    return _build_rejection_result(last_explanation, state, report)


def build_tools(
    fixer: LLMFixer,
    validator: Optional[FixValidator],
    registry: RuleHandlerRegistry,
) -> Dict[str, Any]:
    """Build all graph tools as thin closures over fixer/validator/registry."""

    @tool
    async def validate_issue_tool(state: dict) -> dict:  # type: ignore[misc]
        """Validate that all issues belong to the same file and extract line range."""
        return await _validate_issue_impl(state, fixer)

    @tool
    async def run_automated_tool(state: dict) -> dict:  # type: ignore[misc]
        """Run deterministic AST-based rule handler."""
        return await _run_handler_impl(state, fixer, registry, "AST fix")

    @tool
    async def run_llm_tool(state: dict) -> dict:  # type: ignore[misc]
        """Run LLM-backed rule handler with optional retry feedback."""
        report: StatusReporter = state.get("_report") or _noop_reporter
        retry_count: int = state.get("retry_count", 0)
        if retry_count > 0:
            report(f"[bold yellow]  ↻ Retry {retry_count}/{MAX_RETRIES}[/bold yellow]")
        return await _run_handler_impl(state, fixer, registry, "LLM fix")

    @tool
    def check_syntax_tool(state: dict) -> dict:  # type: ignore[misc]
        """Check syntax of each fix via py_compile."""
        return _check_syntax_impl(state, fixer)

    @tool
    def check_testing_tool(state: dict) -> dict:  # type: ignore[misc]
        """Run pytest to verify fix does not break existing tests."""
        return _check_testing_impl(state, fixer)

    @tool
    def validate_fix_tool(state: dict) -> dict:  # type: ignore[misc]
        """Run FixValidator on each fix and collect accepted/rejected results."""
        return _validate_fix_impl(state, validator)

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
       - automated rule + syntax OK  → "accept"   (no LLM, no validator needed)
       - llm rule + syntax OK        → "testing"  (run tests before accepting)
       - syntax error (any source)   → "validate" (forward to FixValidator)
       """
    if state.get("syntax_err") is not None:
        logger.info("Routing → validate (syntax_err=%s)", state["syntax_err"])
        return "validate"

    rule = state["issues"][0].rule
    if rule in _AUTOMATED_RULES:
        logger.info("Routing → accept (automated rule, syntax OK)")
        return "accept"

    logger.info("Routing → testing (llm rule, syntax OK)")
    return "testing"




def route_after_test(state: SonarState) -> str:
    """
    Edge after check_testing_tool.
    - "accept"   → all tests passed, fix is good
    - "validate" → tests failed, forward failure output to FixValidator
    """
    if state.get("test_err") is None:
        logger.info("Routing → accept (tests passed)")
        return "accept"
    logger.info("Routing → validate (test failure: %s)", state["test_err"])
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
    report: StatusReporter = state.get("_report") or _noop_reporter
    report("[bold green]  ✔ Fix accepted — applying to file[/bold green]")
    if not state.get("accepted_fixes"):
        return {**state, "accepted_fixes": state.get("fixes", [])}  # type: ignore[return-value]
    return state


def _error_node(state: SonarState) -> SonarState:
    """Terminal node: ensure accepted_fixes is empty on error."""
    report: StatusReporter = state.get("_report") or _noop_reporter
    err = state.get("error", "unknown error")
    report(f"[bold red]  ✗ Pipeline ended with error:[/bold red] {err}")
    return {**state, "accepted_fixes": []}  # type: ignore[return-value]


def _retry_node(state: SonarState) -> SonarState:
    """Increment retry_count before looping back to run_llm."""
    new_count = state.get("retry_count", 0) + 1
    report: StatusReporter = state.get("_report") or _noop_reporter
    report(f"[bold yellow]  ↻ Scheduling retry {new_count}/{MAX_RETRIES}[/bold yellow]")
    return {**state, "retry_count": new_count}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _add_graph_nodes(graph: StateGraph, tools: Dict[str, Any]) -> None:
    """Register all nodes on the graph."""
    graph.add_node("validate_issue", _make_node(tools["validate_issue_tool"]))
    graph.add_node("route_rule", lambda s: s)
    graph.add_node("run_automated", _make_node(tools["run_automated_tool"]))
    graph.add_node("run_llm", _make_node(tools["run_llm_tool"]))
    graph.add_node("check_syntax", _make_sync_node(tools["check_syntax_tool"]))
    graph.add_node("check_testing", _make_sync_node(tools["check_testing_tool"]))
    graph.add_node("validate_fix", _make_sync_node(tools["validate_fix_tool"]))
    graph.add_node("accept", _accept_node)
    graph.add_node("error", _error_node)
    graph.add_node("retry", _retry_node)


def _add_graph_edges(graph: StateGraph) -> None:
    """Wire all edges and conditional edges on the graph."""
    graph.set_entry_point("validate_issue")
    graph.add_conditional_edges(
        "validate_issue", route_after_validation,
        {"error": "error", "route_rule": "route_rule"},
    )
    graph.add_conditional_edges(
        "route_rule", route_by_rule,
        {"automated": "run_automated", "llm": "run_llm"},
    )
    graph.add_edge("run_automated", "check_syntax")
    graph.add_edge("run_llm", "check_syntax")
    graph.add_conditional_edges(
        "check_syntax", route_after_syntax,
        {"accept": "accept", "testing": "check_testing", "validate": "validate_fix"},
    )
    graph.add_conditional_edges(
        "check_testing", route_after_test,
        {"accept": "accept", "validate": "validate_fix"},
    )
    graph.add_conditional_edges(
        "validate_fix", route_after_validation_fix,
        {"accept": "accept", "retry": "retry", "error": "error"},
    )
    graph.add_edge("retry", "run_llm")
    graph.add_edge("accept", END)
    graph.add_edge("error", END)


def build_graph(
    fixer: LLMFixer,
    validator: Optional[FixValidator],
    registry: RuleHandlerRegistry,
) -> Any:
    """Build and compile the LangGraph for one file group."""
    tools = build_tools(fixer, validator, registry)
    graph = StateGraph(SonarState)
    _add_graph_nodes(graph, tools)
    _add_graph_edges(graph)
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

    Pass `status_reporter` to receive live phase updates in the calling UI:
        supervisor.run(..., status_reporter=console.print)
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
        skip_tests: bool = True,
        status_reporter: Optional[StatusReporter] = None,
    ) -> FixResult:
        """Entry point — groups issues by file then runs the graph per group."""
        all_accepted: List[FixSuggestion] = []

        for file_issues in self._group_by_file(issues, project_path).values():
            accepted = await self._run_for_file_group(
                file_issues, project_path, tmp_path,
                modified_content, file_md,
                skip_tests=skip_tests,
                status_reporter=status_reporter or _noop_reporter,
            )
            all_accepted.extend(accepted)

        if not all_accepted:
            return FixResult(project_path=project_path, total_fixes_attempted=len(issues))

        return await self.fixer.apply_fixes(
            all_accepted, project_path,
            create_backup=create_backup, dry_run=dry_run,
        )

    @staticmethod
    def _build_initial_state(
        issues: List[IssueUnion], project_path: Path, tmp_path: Path,
        modified_content: str, file_md: str, skip_tests: bool,
        status_reporter: StatusReporter,
    ) -> "SonarState":
        """Build the initial state dict for a file group run."""
        return {
            "issues": issues, "project_path": project_path,
            "tmp_path": tmp_path, "modified_content": modified_content,
            "file_md": file_md, "fixes": [], "accepted_fixes": [],
            "error": "", "retry_count": 0, "error_feedback": "",
            "syntax_err": None, "test_err": None,
            "skip_tests": skip_tests, "_report": status_reporter,
        }

    async def _run_for_file_group(
        self,
        issues: List[IssueUnion],
        project_path: Path,
        tmp_path: Path,
        modified_content: str,
        file_md: str,
        skip_tests: bool = True,
        status_reporter: StatusReporter = _noop_reporter,
    ) -> List[FixSuggestion]:
        initial_state = self._build_initial_state(
            issues, project_path, tmp_path,
            modified_content, file_md, skip_tests, status_reporter,
        )
        try:
            final_state = await self._graph.ainvoke(
                initial_state, config={"recursion_limit": 30},
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


def _build_validator(provider: str, model: Optional[str], api_key: Optional[str],
                     validator_provider: Optional[str], validator_model: Optional[str],
                     validator_api_key: Optional[str], min_confidence: float) -> FixValidator:
    """Build a FixValidator with fallback to fixer params."""
    return FixValidator(
        provider=validator_provider or provider,
        model=validator_model or model,
        api_key=validator_api_key or api_key,
        min_confidence_threshold=min_confidence,
    )


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
    """Factory — builds AgentSupervisor with fixer and optional validator."""
    fixer = LLMFixer(provider=provider, model=model, api_key=api_key, context_lines=context_lines)
    validator: Optional[FixValidator] = None
    if use_validator:
        validator = _build_validator(
            provider, model, api_key, validator_provider, validator_model, validator_api_key, min_confidence,
        )
    return AgentSupervisor(fixer=fixer, validator=validator)
