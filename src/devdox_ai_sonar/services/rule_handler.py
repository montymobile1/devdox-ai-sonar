from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import re

from devdox_ai_sonar.models.file_structures import FixContext
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    FixSuggestion,
    SonarFixResponse,
    CodeBlock,
    ChangeType,
    BlockType,
    LineChange,
    ChangeAction,
    SearchReplace,
)
from devdox_ai_sonar.utils.function_finder import (
    detect_original_function_type,
    find_function_implementations,
    AsyncConversionAnalyzer,
    to_snake_case,
)

logger = logging.getLogger(__name__)


def _resolve_effective_values(
    code_blocks: List[CodeBlock],
    file_path: Path,
    context: FixContext,
    line_range: Dict[str, Any],
    modify_line_range: bool = True,
) -> tuple:
    """
    Determine effective file path and line numbers from code blocks.

    If code blocks specify a different file, use those values instead of the originals.
    """
    effective_file_path = file_path
    effective_start_line = context.context_dict.get("start_line")
    effective_sonar_line = line_range.get("first_line")
    effective_last_line = line_range.get("last_line", line_range.get("first_line"))

    if not (code_blocks and modify_line_range):
        return (
            effective_file_path,
            effective_start_line,
            effective_sonar_line,
            effective_last_line,
        )

    first_block = code_blocks[0]
    if not (hasattr(first_block, "file_path") and first_block.file_path):
        return (
            effective_file_path,
            effective_start_line,
            effective_sonar_line,
            effective_last_line,
        )

    raw_block_file_path = first_block.file_path
    block_file_path: Path = (
        Path(raw_block_file_path) if isinstance(raw_block_file_path, str) else file_path
    )

    if block_file_path != file_path:
        effective_file_path = block_file_path
        effective_start_line = first_block.start_line
        effective_sonar_line = first_block.start_line
        effective_last_line = first_block.end_line

    return (
        effective_file_path,
        effective_start_line,
        effective_sonar_line,
        effective_last_line,
    )


def _resolve_relative_path(effective_file_path: Path, project_path: Path) -> str:
    """Compute relative file path, falling back to absolute if outside project."""
    try:
        return str(effective_file_path.relative_to(project_path))
    except ValueError:
        return str(effective_file_path)


AWAIT_REMOVAL_PATTERN = r"\bawait\s+(?=(?:[\w]+\.)*{func_name}\s*\()"
ARG_RENAME_PATTERN = r"(\b{func_name}\s*\([\s\S]*?\b){old_arg}\s*="


class RuleHandler(ABC):
    """Abstract base class for rule-specific fix handlers."""

    # Subclasses must declare whether they modify the line range
    MOIDY_LINE_RANGE: bool = False

    # Subclasses should set this to the LLM model identifier, or leave as None
    model: Optional[str] = None

    @abstractmethod
    def can_handle(self, rule: str) -> bool:
        """
        Check if this handler can process the given rule.

        Args:
            rule: SonarCloud rule identifier (e.g., 'python:S7503')

        Returns:
            True if handler supports this rule
        """
        pass

    @abstractmethod
    async def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any,  # Forward reference to avoid circular import
    ) -> Optional[List[SonarFixResponse]]:
        """
        Generate fix suggestions for the given issues.

        Args:
            issues: List of SonarCloud issues to fix
            context: Fix context with file and code information
            project_path: Root path of the project
            llm_caller: LLM client for generating fixes

        Returns:
            List of fix SonarFixResponse
        """
        pass

    def _generate_fix_key(self, problem_lines: List[Any]) -> str:
        """
        Generate a unique key for the fix based on problem lines.

        Subclasses may override this if a different key strategy is needed.
        """
        return "-".join(str(line) for line in problem_lines) if problem_lines else ""

    def _build_fix_suggestion(
        self,
        fix_response_list: List[SonarFixResponse],
        context: FixContext,
        file_path: Path,
        project_path: Path,
        line_range: Dict[str, Any],
    ) -> List[FixSuggestion]:
        """
        Build FixSuggestion objects from a SonarFixResponse list.

        Handles cases where code blocks target different files than the original issue,
        overriding file_path and line numbers accordingly.

        Args:
            fix_response_list: List of responses from the handler with code blocks
            context: Fix context for the original issue
            file_path: Original file path (from the issue)
            project_path: Project root
            line_range: Line range dictionary for the original issue

        Returns:
            List of FixSuggestion objects ready for application
        """
        lst_suggestion = []

        for fix_response_single in fix_response_list:
            code_blocks = fix_response_single.FIXED_CODE_BLOCKS

            (
                effective_file_path,
                effective_start_line,
                effective_sonar_line,
                effective_last_line,
            ) = _resolve_effective_values(
                code_blocks,
                file_path,
                context,
                line_range,
                modify_line_range=self.MOIDY_LINE_RANGE,
            )

            relative_file_path = _resolve_relative_path(
                effective_file_path, project_path
            )

            lst_suggestion.append(
                FixSuggestion(
                    issue_key=self._generate_fix_key(
                        line_range.get("problem_lines", [])
                    ),
                    original_code=context.code_content,
                    fixed_code_blocks=code_blocks,
                    fixed_code="",
                    import_block_code=fix_response_single.IMPORT_BLOCK,
                    helper_code=fix_response_single.NEW_HELPER_CODE,
                    placement_helper=fix_response_single.PLACEMENT,
                    explanation=fix_response_single.EXPLANATION,
                    confidence=fix_response_single.CONFIDENCE,
                    llm_model=self.model or "unknown",
                    rule_description="",
                    file_path=relative_file_path,
                    line_number=effective_start_line,
                    sonar_line_number=effective_sonar_line,
                    end_import_block_code=context.context_dict.get(
                        "import_section", {}
                    ).get("end_line", 0),
                    last_line_number=effective_last_line,
                )
            )

        logger.debug(f"Built {len(lst_suggestion)} fix suggestions")
        return lst_suggestion


class ConvenationNameHandler(RuleHandler):
    """
    Handler for python:S117 - Local variable and function parameter names
    should comply with a naming convention (snake_case).

    This handler identifies function parameters that violate the naming convention
    and generates fixes to rename them to snake_case across both the function
    definition and all known call sites in the project.
    """

    RULE_ID = "python:S117"
    MOIDY_LINE_RANGE = True

    def can_handle(self, rule: str) -> bool:
        return rule == self.RULE_ID

    async def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any = None,
    ) -> Optional[List[SonarFixResponse]]:
        """
        Generate fixes for parameter naming convention violations (python:S117).

        Strategy:
        1. Locate all function definitions in the project matching the issue context.
        2. For each definition in the affected file, identify parameters that are
           not snake_case and compute their renamed equivalents.
        3. Build code blocks to update:
           a. The function definition — rename the offending parameter(s).
           b. All call sites — update keyword argument usage to match new names.
        4. Group code blocks by file: same-file changes are bundled into one
           SonarFixResponse; cross-file changes each get their own response.

        Args:
            issues:       List of Sonar issues for this rule (python:S117).
            context:      Fix context including the surrounding code and metadata.
            project_path: Root of the project, used to scan for call sites.
            file_path:    Path to the file containing the violating definition.
            llm_caller:   Optional LLM caller (unused for this rule).

        Returns:
            A list of SonarFixResponse objects covering the definition and all
            impacted call sites, or None if no actionable fixes can be generated.
        """
        try:
            code_blocks = []
            response_lst = []

            # Step 1: Find all function implementations
            function_info = find_function_implementations(
                project_path, context.functions[0]["name"]
            )

            if not function_info.get("definitions"):
                logger.warning("Could not detect function type for async conversion")
                return None

            # Step 2: Identify parameters to rename in the affected file
            args_to_be_changed = {}
            for definition in function_info["definitions"]:
                if Path(definition["file"]) == file_path:
                    for arg in definition["args"]:
                        new_arg = to_snake_case(arg)
                        if new_arg != arg:
                            args_to_be_changed[arg] = new_arg
                            code_blocks.append(
                                self._change_function_definition_block(
                                    definition, context, arg, new_arg
                                )
                            )

            if not code_blocks:
                return None

            # Step 3: Build caller blocks for cross-file call sites
            caller_blocks = self._create_caller_blocks(
                args_to_be_changed, function_info
            )
            for block in caller_blocks:
                response_lst.append(
                    SonarFixResponse(
                        IMPORT_BLOCK="",
                        FIXED_CODE_BLOCKS=[block],
                        NEW_HELPER_CODE="",
                        PLACEMENT="SIBLING",
                        EXPLANATION="",
                        CONFIDENCE=0.95,
                    )
                )

            explanation = (
                f"Renamed non-snake_case parameter(s) {list(args_to_be_changed.keys())} "
                f"to {list(args_to_be_changed.values())} in '{context.functions[0]['name']}'. "
                f"Updated {len(caller_blocks)} call site(s) to use the new keyword argument name(s). "
                f"This satisfies python:S117, which requires all local variables and function "
                f"parameters to follow the snake_case naming convention."
            )

            logger.info(
                f"Generated {len(code_blocks)} code blocks for async-to-sync conversion"
            )

            response_lst.append(
                SonarFixResponse(
                    IMPORT_BLOCK="",
                    FIXED_CODE_BLOCKS=code_blocks,
                    NEW_HELPER_CODE="",
                    PLACEMENT="SIBLING",
                    EXPLANATION=explanation,
                    CONFIDENCE=0.95,
                )
            )

            return response_lst

        except Exception as e:
            print(f"Error in ConventationName: {e}")
            logger.error(f"Error in AsyncToSyncHandler: {e}", exc_info=True)
            return None

    def _change_function_definition_block(
        self,
        function_info: Dict[str, Any],
        context: FixContext,
        arg_name: str,
        new_arg_name: str,
    ) -> CodeBlock:
        """
        Build a CodeBlock that renames a single parameter in the function definition.

        Replaces all occurrences of ``arg_name`` with ``new_arg_name`` in the
        original function definition context, and computes the correct start/end
        line numbers accounting for any decorators above the function.

        Args:
            function_info: Parsed metadata for the function (line number, decorators, etc.).
            context:       Fix context providing the raw source of the definition.
            arg_name:      The parameter name that violates the naming convention.
            new_arg_name:  The snake_case replacement for the parameter name.

        Returns:
            A CodeBlock targeting the function definition with the rename applied.
        """
        original_def = context.context_dict["new_context"][0]["context"]
        num_lines = len(original_def.strip().split("\n"))
        new_def = original_def.replace(arg_name, new_arg_name)

        actual_start_line = function_info["line"] - len(function_info["decorators"])
        end_line = actual_start_line + num_lines

        return CodeBlock(
            block_name=function_info["function"],
            start_line=actual_start_line,
            end_line=end_line,
            has_changes=True,
            change_type=ChangeType.FULL_CODE,
            block_type=BlockType.FUNCTION,
            context=new_def,
        )

    def _create_caller_blocks(
        self,
        args_to_be_changed: Dict[str, str],
        function_info: Dict[str, Any],
    ) -> List[CodeBlock]:
        """
        Build CodeBlocks to rename keyword arguments at all call sites.

        For each call site found in ``function_info['calls']``, creates a
        SEARCH_REPLACE CodeBlock that rewrites keyword argument names from their
        old form to the new snake_case form using a regex pattern.

        Args:
            args_to_be_changed: Mapping of ``{old_arg_name: new_arg_name}`` for
                                all parameters being renamed.
            function_info:      Parsed function metadata including the list of
                                call sites (file path and line number).

        Returns:
            List of CodeBlock objects, one per (call site × renamed argument) pair.
        """
        blocks = []
        func_name = function_info["definitions"][0]["function"]
        num_args = len(function_info["definitions"][0]["args"])

        for caller in function_info["calls"]:
            for old_arg_name, new_arg_name in args_to_be_changed.items():
                blocks.append(
                    CodeBlock(
                        block_name=func_name,
                        start_line=caller["line"],
                        end_line=caller["line"] + num_args + 1,
                        has_changes=True,
                        change_type=ChangeType.SEARCH_REPLACE,
                        block_type=BlockType.FUNCTION,
                        replacements=[
                            SearchReplace(
                                search=ARG_RENAME_PATTERN.format(
                                    func_name=re.escape(func_name),
                                    old_arg=re.escape(old_arg_name),
                                ),
                                replace=r"\1" + f"{new_arg_name}=",
                                is_regex=True,
                                count=None,
                            )
                        ],
                        file_path=caller["file"],
                    )
                )

        return blocks


class AsyncToSyncHandler(RuleHandler):
    """
    Handler for python:S7503 - unnecessary async functions.

    This rule identifies async functions that don't use await and should be sync.
    The handler analyzes the function and all its call sites to generate appropriate fixes.
    """

    RULE_ID = "python:S7503"
    MOIDY_LINE_RANGE = True

    def can_handle(self, rule: str) -> bool:
        return rule == self.RULE_ID

    async def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any = None,
    ) -> Optional[List[SonarFixResponse]]:
        """
        Generate fixes for async-to-sync conversion.

        Strategy:
        1. Parse the async function definition
        2. Analyze all call sites in the codebase
        3. Generate fixes for:
           a. The function definition (remove 'async')
           b. All call sites (remove 'await')

        Returns:
            SonarFixResponse with all code blocks for function and call sites
        """
        try:
            code_blocks = []
            response_lst = []
            source_lines = context.code_content

            # Step 1: Detect function type
            function_info = detect_original_function_type(source_lines.strip(), 1)

            if not function_info.get("found"):
                logger.warning("Could not detect function type for async conversion")
                return None

            # Step 2: Build code block for function definition
            code_blocks.append(
                self._create_function_definition_block(
                    function_info, context, file_path
                )
            )

            # Step 3: Analyze call sites
            analyzer = AsyncConversionAnalyzer(function_info["name"], project_path)
            analysis = await analyzer.analyze()

            # Step 4: Generate code blocks for all awaited call sites
            explanation = (
                f"Converting async function '{function_info['name']}' to sync:\n"
                f"- Removed 'async' keyword from function definition\n"
                f"- Removed 'await' from {{caller_count}} call site(s)"
            )

            caller_blocks = self._create_caller_blocks(analysis, function_info)
            for block in caller_blocks:
                if block.file_path == file_path:
                    code_blocks.append(block)
                else:
                    response_lst.append(
                        SonarFixResponse(
                            IMPORT_BLOCK="",
                            FIXED_CODE_BLOCKS=[block],
                            NEW_HELPER_CODE="",
                            PLACEMENT="SIBLING",
                            EXPLANATION="",
                            CONFIDENCE=0.95,
                        )
                    )

            logger.info(
                f"Generated {len(code_blocks)} code blocks for async-to-sync conversion"
            )

            response_lst.append(
                SonarFixResponse(
                    IMPORT_BLOCK="",
                    FIXED_CODE_BLOCKS=code_blocks,
                    NEW_HELPER_CODE="",
                    PLACEMENT="SIBLING",
                    EXPLANATION=explanation.format(caller_count=len(caller_blocks)),
                    CONFIDENCE=0.95,
                )
            )

            return response_lst

        except Exception as e:
            print(f"Error in AsyncToSyncHandler: {e}")
            logger.error(f"Error in AsyncToSyncHandler: {e}", exc_info=True)
            return None

    def _create_function_definition_block(
        self,
        function_info: Dict[str, Any],
        context: FixContext,
        file_path: Path,
    ) -> CodeBlock:
        """Create code block for the function definition (remove 'async' keyword)."""
        original_def = function_info["definition"]
        new_def = original_def.replace("async def", "def")

        start_line = 0
        for func in context.context_dict.get("functions", []):
            if func["name"] == function_info["name"]:
                start_line = func["start_line"]
                break

        actual_line = start_line + function_info["start_line"]

        return CodeBlock(
            block_name=function_info["name"],
            start_line=actual_line,
            end_line=actual_line + 1,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.FUNCTION,
            changes=[
                LineChange(
                    line=actual_line - 1,
                    action=ChangeAction.REPLACE,
                    old=original_def,
                    new=new_def,
                )
            ],
            file_path=str(file_path),
        )

    def _create_caller_blocks(
        self,
        analysis: Any,
        function_info: Dict[str, Any],
    ) -> List[CodeBlock]:
        """Create code blocks for all call sites that use 'await'."""
        blocks = []

        for caller in analysis.caller_impact:
            if not caller.get("awaited"):
                continue  # Only fix awaited calls

            blocks.append(
                CodeBlock(
                    block_name=function_info["name"],
                    start_line=caller["line"],
                    end_line=caller["line"],
                    has_changes=True,
                    change_type=ChangeType.SEARCH_REPLACE,
                    block_type=BlockType.FUNCTION,
                    replacements=[
                        SearchReplace(
                            search=AWAIT_REMOVAL_PATTERN.format(
                                func_name=re.escape(function_info["name"])
                            ),
                            replace="",
                            is_regex=True,
                            count=None,
                        )
                    ],
                    file_path=caller["file"],
                )
            )

        return blocks


class CognitiveComplexityHandler(RuleHandler):
    """
    Handler for python:S3776 - cognitive complexity reduction.

    This rule requires breaking down complex functions into simpler helper methods.
    """

    RULE_ID = "python:S3776"
    MOIDY_LINE_RANGE = False

    def can_handle(self, rule: str) -> bool:
        return rule == self.RULE_ID

    async def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any,
    ) -> Optional[List[SonarFixResponse]]:
        """
        Generate fixes for cognitive complexity issues.

        This delegates to the LLM with specialized refactoring prompts.
        """
        if not llm_caller:
            logger.error("LLM caller required for cognitive complexity fixes")
            return None

        try:
            fix_response = llm_caller._call_llm_list(
                issues,
                context,
                context.file_path.suffix,
                {},
                error_message="",
            )
            return [fix_response]

        except Exception as e:
            logger.error(f"Error in CognitiveComplexityHandler: {e}", exc_info=True)
            return None


class DefaultRuleHandler(RuleHandler):
    """
    Default handler for standard SonarCloud rules.

    Uses the LLM with general-purpose fixing prompts for rules that don't
    require specialized logic.
    """

    MOIDY_LINE_RANGE = False

    def can_handle(self, rule: str) -> bool:
        """Default handler accepts any rule."""
        return True

    async def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any,
    ) -> Optional[List[SonarFixResponse]]:
        """Generate fixes using standard LLM approach."""
        if not llm_caller:
            logger.error("LLM caller required for default rule handling")
            return None

        try:
            fix_response = llm_caller._call_llm_list(
                issues,
                context,
                context.file_path.suffix,
                {},
                error_message="",
            )
            return [fix_response]

        except Exception as e:
            logger.error(f"Error in DefaultRuleHandler: {e}", exc_info=True)
            return None


class RuleHandlerRegistry:
    """
    Registry for managing rule handlers.

    Implements a chain-of-responsibility pattern where handlers are tried
    in priority order until one accepts the rule.
    """

    def __init__(self) -> None:
        """Initialize registry with default handlers in priority order."""
        self.handlers: List[RuleHandler] = [
            AsyncToSyncHandler(),
            CognitiveComplexityHandler(),
            ConvenationNameHandler(),
            DefaultRuleHandler(),  # Catch-all at the end
        ]

    def register(self, handler: RuleHandler, priority: int = -1) -> None:
        """
        Register a custom handler.

        Args:
            handler: Handler instance to register
            priority: Position in handler chain (default: before catch-all)
        """
        if priority < 0:
            self.handlers.insert(len(self.handlers) - 1, handler)
        else:
            self.handlers.insert(priority, handler)

    def get_handler(self, rule: str) -> RuleHandler:
        """
        Get appropriate handler for a rule.

        Args:
            rule: SonarCloud rule identifier

        Returns:
            First handler that can handle the rule (always returns a handler)
        """
        for handler in self.handlers:
            if handler.can_handle(rule):
                logger.debug(f"Using {handler.__class__.__name__} for rule {rule}")
                return handler

        logger.warning(f"No handler found for rule {rule}, using default")
        return self.handlers[-1]
