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
    AsyncConversionAnalyzer,
)

logger = logging.getLogger(__name__)


def _resolve_effective_values(
    code_blocks,
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

    block_file_path = first_block.file_path
    if isinstance(block_file_path, str):
        block_file_path = Path(block_file_path)

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


class RuleHandler(ABC):
    """Abstract base class for rule-specific fix handlers."""

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
    def generate_fixes(
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


class AsyncToSyncHandler(RuleHandler):
    """
    Handler for python:S7503 - unnecessary async functions.

    This rule identifies async functions that don't use await and should be sync.
    The handler analyzes the function and all its call sites to generate appropriate fixes.
    """

    RULE_ID = "python:S7503"
    MOIDY_LINE_RANGE = True

    def can_handle(self, rule: str) -> bool:
        """Check if this is the async-to-sync conversion rule."""
        return rule == self.RULE_ID

    def generate_fixes(
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
            function_block = self._create_function_definition_block(
                function_info, context
            )
            code_blocks.append(function_block)

            # Step 3: Analyze call sites
            analyzer = AsyncConversionAnalyzer(function_info["name"], project_path)
            analysis = analyzer.analyze()

            # Step 4: Generate code blocks for all awaited call sites
            caller_blocks = self._create_caller_blocks(analysis, function_info)
            for blocks in caller_blocks:
                if blocks.file_path == file_path:
                    code_blocks.append(blocks)
                else:
                    response_lst.append(
                        SonarFixResponse(
                            IMPORT_BLOCK="",
                            FIXED_CODE_BLOCKS=[blocks],
                            NEW_HELPER_CODE="",
                            PLACEMENT="SIBLING",
                            EXPLANATION="",
                            CONFIDENCE=0.95,
                        )
                    )

                    # Step 5: Build unified response
                explanation = (
                    f"Converting async function '{function_info['name']}' to sync:\n"
                    f"- Removed 'async' keyword from function definition\n"
                    f"- Removed 'await' from {len(caller_blocks)} call site(s)"
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
            print(f"Error in AsyncToSyncHandler: {e}")
            logger.error(f"Error in AsyncToSyncHandler: {e}", exc_info=True)
            return None

    def _create_function_definition_block(
        self, function_info: Dict[str, Any], context: FixContext
    ) -> CodeBlock:
        """Create code block for the function definition (remove 'async' keyword)."""

        # Extract the original definition
        original_def = function_info["definition"]
        new_def = original_def.replace("async def", "def")

        # Find function start line in context
        function_names = context.context_dict.get("functions", [])
        start_line = 0

        for func in function_names:
            if func["name"] == function_info["name"]:
                start_line = func["start_line"]
                break

        # Calculate actual line number
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
        )

    def _create_caller_blocks(
        self, analysis: Any, function_info: Dict[str, Any]
    ) -> List[CodeBlock]:
        """Create code blocks for all call sites that use 'await'."""

        blocks = []

        for caller in analysis.caller_impact:
            if not caller.get("awaited"):
                continue  # Only fix awaited calls

            # Build code block for removing 'await'
            block = CodeBlock(
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

            blocks.append(block)

        return blocks

    def _build_fix_suggestion(
        self,
        fix_response_list: List[SonarFixResponse],
        context: FixContext,
        file_path: Path,
        project_path: Path,
        line_range: Dict[str, Any],
    ) -> List[FixSuggestion]:
        """
        Build FixSuggestion objects from SonarFixResponse list.

        Handles cases where code blocks target different files than the original issue,
        overriding file_path and line numbers accordingly.

        Args:
            fix_response_list: List of responses from handler with code blocks
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
            ) = _resolve_effective_values(code_blocks, file_path, context, line_range)

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


class CognitiveComplexityHandler(RuleHandler):
    """
    Handler for python:S3776 - cognitive complexity reduction.

    This rule requires breaking down complex functions into simpler helper methods.
    """

    RULE_ID = "python:S3776"
    MOIDY_LINE_RANGE = False

    def can_handle(self, rule: str) -> bool:
        """Check if this is a cognitive complexity rule."""
        return rule == self.RULE_ID

    def generate_fixes(
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

        Returns:
            SonarFixResponse with refactored code blocks
        """
        if not llm_caller:
            logger.error("LLM caller required for cognitive complexity fixes")
            return None

        try:
            # Use specialized refactoring template
            fix_response = llm_caller._call_llm_list(
                issues,
                context,
                context.file_path.suffix,
                {},  # rule_info will be populated by caller
                error_message="",
            )

            return [fix_response]

        except Exception as e:
            logger.error(f"Error in CognitiveComplexityHandler: {e}", exc_info=True)
            return None


class DefaultRuleHandler(RuleHandler):
    """
    Default handler for standard SonarCloud rules.

    This handler uses the LLM with general-purpose fixing prompts
    for rules that don't require specialized logic.
    """

    MOIDY_LINE_RANGE = False

    def can_handle(self, rule: str) -> bool:
        """Default handler accepts any rule."""
        return True

    def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any,
    ) -> Optional[List[SonarFixResponse]]:
        """
        Generate fixes using standard LLM approach.

        Args:
            issues: Issues to fix
            context: Fix context
            project_path: Project root
            llm_caller: LLM client instance

        Returns:
            SonarFixResponse object or None if generation fails
        """
        if not llm_caller:
            logger.error("LLM caller required for default rule handling")
            return None

        try:
            # Use standard fixing approach
            fix_response = llm_caller._call_llm_list(
                issues,
                context,
                context.file_path.suffix,
                {},  # rule_info populated by caller
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

    def __init__(self):
        """Initialize registry with default handlers in priority order."""
        self.handlers: List[RuleHandler] = [
            AsyncToSyncHandler(),
            CognitiveComplexityHandler(),
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
            # Insert before DefaultRuleHandler (last position)
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

        # Should never reach here due to DefaultRuleHandler
        logger.warning(f"No handler found for rule {rule}, using default")
        return self.handlers[-1]
