from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import ast
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
    PlacementType,
    LineChange,
    ChangeAction,
    SearchReplace,
)
from devdox_ai_sonar.utils.function_finder import (
    detect_original_function_type,
    find_function_implementations,
    AsyncConversionAnalyzer,
    _is_param_used_at_callsite,
    to_snake_case,
)
from devdox_ai_sonar.services.constant_namer import (
    ConstantNamingService,
    LLMFixerAdapter,
)
from devdox_ai_sonar.models.constant_naming import (
    LiteralContext,
    NamingRequest,
    NamingResponse,
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
FUNC_CALL_RENAME_PATTERN = r"(?<!\w)(?<!def ){func_name}(?![\w])(?=\s*\()"


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
        rule_info: Optional[Dict[str, Any]] = None,
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

        logger.debug("Built %d fix suggestion(s)", len(lst_suggestion))
        return lst_suggestion


class ConvenationNameHandler(RuleHandler):
    """
    Handler for python:S117 - Local variable and function parameter names
    should comply with a naming convention (snake_case).

    This handler identifies function parameters that violate the naming convention
    and generates fixes to rename them to snake_case across both the function
    definition and all known call sites in the project.
    """

    RULE_ID = ["python:S117", "python:S1172", "python:S1542"]
    MOIDY_LINE_RANGE = True

    def can_handle(self, rule: str) -> bool:
        return rule in self.RULE_ID

    async def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any = None,
        rule_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[SonarFixResponse]]:
        """
        Generate fixes for parameter naming/usage violations.

        - python:S117  → Rename non-snake_case parameters in definition + call sites.
        - python:S1542  → Rename non-snake_case function in definition + call sites.
        - python:S1172 → Remove unused parameters not referenced anywhere in the codebase.
        """
        try:
            if len(context.functions) == 0:
                logger.warning("Could not find functions ")
                return None

            function_info = find_function_implementations(
                project_path, context.functions[0]["name"]
            )

            if not function_info.get("definitions"):
                logger.warning(
                    "Could not find function definitions for %s",
                    context.functions[0]["name"],
                )
                return None

            for issue in issues:
                if issue.rule == "python:S117":
                    return self._fix_naming_convention(
                        function_info, context, file_path
                    )
                if issue.rule == "python:S1542":
                    return self._fix_func_naming_convention(
                        function_info, context, file_path
                    )
                elif issue.rule == "python:S1172":
                    return self._fix_unused_parameters(
                        function_info, context, file_path
                    )

            return None

        except Exception:
            logger.exception(
                "ConvenationNameHandler failed to generate fixes for %s", self.RULE_ID
            )
            return None

    def _fix_unused_parameters(
        self,
        function_info: Dict,
        context: FixContext,
        file_path: Path,
    ) -> Optional[List[SonarFixResponse]]:
        args_safe_to_remove: List[str] = []

        for definition in function_info["definitions"]:
            if Path(definition["file"]) != file_path:
                continue
            args_safe_to_remove.extend(
                self._collect_removable_params(definition, function_info["calls"])
            )

        if not args_safe_to_remove:
            logger.info("No safely removable unused parameters found for S1172")
            return None

        code_blocks: List[CodeBlock] = []
        for definition in function_info["definitions"]:
            if Path(definition["file"]) == file_path:
                for param in args_safe_to_remove:
                    code_blocks.append(
                        self._remove_parameter_block(definition, context, param)
                    )

        if not code_blocks:
            return None

        explanation = (
            f"Removed unused parameter(s) {args_safe_to_remove} "
            f"from '{context.functions[0]['name']}'. "
            f"These parameters were never referenced inside the function body "
            f"and never passed (positionally or by keyword) at any call site. "
            f"This satisfies python:S1172 (unused function parameters)."
        )

        logger.info(
            "Generated %d block(s) for S1172 unused param removal", len(code_blocks)
        )

        return [
            SonarFixResponse(
                IMPORT_BLOCK="",
                FIXED_CODE_BLOCKS=code_blocks,
                NEW_HELPER_CODE="",
                PLACEMENT="SIBLING",
                EXPLANATION=explanation,
                CONFIDENCE=0.90,
            )
        ]

    def _collect_removable_params(
        self,
        definition: Dict,
        calls: List[Dict],
    ) -> List[str]:
        """
        Identify which unused parameters in a single definition are safe to remove.

        Iterates the definition's unused_args, skips intentionally-unused names
        (prefixed with ``_``), resolves the call-site positional index, and
        checks every call site.  Only parameters that are unused at **all**
        call sites are returned.
        """
        safe: List[str] = []

        for param in definition.get("unused_args", []):
            callsite_index = self._get_callsite_index(param, definition.get("args", []))
            if callsite_index is None:
                continue

            if not _is_param_used_at_callsite(param, callsite_index, calls):
                safe.append(param)
                logger.info(
                    "Param '%s' (index %d) unused in body and at all call sites — safe to remove",
                    param,
                    callsite_index,
                )
            else:
                logger.info(
                    "Param '%s' is unused in body but IS used at call sites — skipping",
                    param,
                )

        return safe

    @staticmethod
    def _get_callsite_index(param: str, all_args: List[str]) -> Optional[int]:
        """
        Compute the 0-based positional index a caller would use for *param*.

        Returns ``None`` (skip this param) when:
        - The param name starts with ``_`` (intentionally unused).
        - The param is not found in the args list.

        When the first argument is ``self`` or ``cls`` it is not passed by
        callers, so the returned index is shifted down by one.
        """
        if param.startswith("_"):
            logger.debug("Skipping intentionally unused param: %s", param)
            return None

        clean_args = [a.lstrip("*") for a in all_args]
        has_self = clean_args[0] in ("self", "cls") if clean_args else False

        try:
            raw_index = clean_args.index(param)
        except ValueError:
            logger.warning("Param '%s' not found in args list %s", param, clean_args)
            return None

        return raw_index - 1 if has_self else raw_index

    def _remove_parameter_block(
        self,
        definition: Dict,
        context: FixContext,
        param: str,
    ) -> CodeBlock:
        """
        Build a CodeBlock that removes a single unused parameter from a function definition.

        Strategy:
            1. Parse the function signature from the definition's source line range.
            2. Strip the target parameter (and its default value if any) from the signature.
            3. Return a CodeBlock replacing the old signature with the cleaned one.

        Args:
            definition: Function definition dict from find_function_implementations.
            context:    FixContext with surrounding code metadata.
            param:      The parameter name to remove.

        Returns:
            A CodeBlock with the corrected function signature.
        """
        file_path = Path(definition["file"])
        source = file_path.read_text(encoding="utf-8")
        source_lines = source.splitlines()

        # Extract the full signature (may span multiple lines)
        start_line = definition["line"] - 1  # Convert to 0-indexed
        signature_lines = []

        for i in range(start_line, len(source_lines)):
            line = source_lines[i]
            signature_lines.append(line)
            if line.rstrip().endswith(":"):
                break

        original_signature = "\n".join(signature_lines)
        cleaned_signature = self._remove_param_from_signature(original_signature, param)

        return CodeBlock(
            block_name="test",
            file_path=str(file_path),
            block_type=BlockType.FUNCTION,
            has_changes=True,
            change_type=ChangeType.FULL_CODE,
            start_line=definition["line"],
            end_line=definition["line"] + len(signature_lines) - 1,
            context=cleaned_signature,
        )

    def _remove_param_from_signature(self, signature: str, param: str) -> str:
        """
        Remove a parameter and its default value from a function signature string.

        Handles all forms:
            - Plain:          param
            - Typed:          param: int
            - Defaulted:      param=None
            - Typed+Default:  param: int = 2
            - vararg:         *param
            - kwarg:          **param

        Args:
            signature: The raw function signature string (possibly multi-line).
            param:     The bare parameter name to remove (no * or **).

        Returns:
            The signature with the parameter removed and trailing commas cleaned up.
        """
        import re

        # Match any of: **param, *param, param — with optional type annotation and default
        # Covers: param, param=val, param: Type, param: Type = val
        pattern = re.compile(
            r"""
            \*{0,2}             # optional * or **
            \b"""
            + re.escape(param)
            + r"""\b
            (?:\s*:\s*[^,=)]+)? # optional type annotation:  : SomeType
            (?:\s*=\s*[^,)]+)?  # optional default value:     = some_value
            \s*,?\s*            # trailing comma + whitespace
            """,
            re.VERBOSE,
        )

        cleaned = pattern.sub("", signature)

        # Clean up any leading comma left behind when removing the first param
        # e.g. "(, remaining_param)" → "(remaining_param)"
        cleaned = re.sub(r"\(\s*,\s*", "(", cleaned)

        # Clean up double commas from middle removal
        # e.g. "(a, , b)" → "(a, b)"
        cleaned = re.sub(r",\s*,", ",", cleaned)

        # Clean up trailing comma before closing paren
        # e.g. "(a, b, )" → "(a, b)"
        cleaned = re.sub(r",\s*\)", ")", cleaned)

        return cleaned

    def _fix_naming_convention(
        self,
        function_info: Dict,
        context: FixContext,
        file_path: Path,
    ) -> Optional[List[SonarFixResponse]]:
        """
        S117: Rename non-snake_case parameters in the function definition
        and update all keyword argument call sites accordingly.
        """
        code_blocks = []
        response_lst = []
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

        caller_blocks = self._create_caller_blocks(
            args_to_be_changed, function_info, "change_arg_name"
        )
        for block in caller_blocks:
            response_lst.append(
                SonarFixResponse(
                    IMPORT_BLOCK="",
                    FIXED_CODE_BLOCKS=[block],
                    NEW_HELPER_CODE="",
                    PLACEMENT=PlacementType.SIBLING,
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
            "Generated %d definition block(s) for S117 rename", len(code_blocks)
        )

        response_lst.append(
            SonarFixResponse(
                IMPORT_BLOCK="",
                FIXED_CODE_BLOCKS=code_blocks,
                NEW_HELPER_CODE="",
                PLACEMENT=PlacementType.SIBLING,
                EXPLANATION=explanation,
                CONFIDENCE=0.95,
            )
        )

        return response_lst

    def _fix_func_naming_convention(
        self,
        function_info: Dict,
        context: FixContext,
        file_path: Path,
    ) -> Optional[List[SonarFixResponse]]:
        """
        S117: Rename non-snake_case parameters in the function definition
        and update all keyword argument call sites accordingly.
        """
        code_blocks = []
        response_lst = []
        funcs_to_be_changed = {}

        for definition in function_info["definitions"]:
            if Path(definition["file"]) == file_path:
                new_name = to_snake_case(definition["function"])
                funcs_to_be_changed[definition["function"]] = new_name
                code_blocks.append(
                    self._change_function_definition_block(
                        definition, context, definition["function"], new_name
                    )
                )
        if not code_blocks:
            return None

        caller_blocks = self._create_caller_blocks(
            funcs_to_be_changed, function_info, "change_func_name"
        )
        for block in caller_blocks:
            response_lst.append(
                SonarFixResponse(
                    IMPORT_BLOCK="",
                    FIXED_CODE_BLOCKS=[block],
                    NEW_HELPER_CODE="",
                    PLACEMENT=PlacementType.SIBLING,
                    EXPLANATION="",
                    CONFIDENCE=0.95,
                )
            )

        explanation = (
            f"Renamed non-snake_case function(s) {list(funcs_to_be_changed.keys())} "
            f"to {list(funcs_to_be_changed.values())}. "
            f"Updated {len(caller_blocks)} call site(s) to use the new function name(s). "
            f"This satisfies python:S1542, which requires function names "
            f"to follow snake_case naming convention."
        )

        logger.info(
            "Generated %d definition block(s) for S1542 rename", len(code_blocks)
        )

        response_lst.append(
            SonarFixResponse(
                IMPORT_BLOCK="",
                FIXED_CODE_BLOCKS=code_blocks,
                NEW_HELPER_CODE="",
                PLACEMENT=PlacementType.SIBLING,
                EXPLANATION=explanation,
                CONFIDENCE=0.95,
            )
        )

        return response_lst

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
        change_type: str = "change_arg_name",
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
                if change_type == "change_arg_name":
                    search_pattern = ARG_RENAME_PATTERN.format(
                        func_name=re.escape(func_name),
                        old_arg=re.escape(old_arg_name),
                    )
                    replace_pattern = r"\1" + f"{new_arg_name}="
                else:
                    search_pattern = FUNC_CALL_RENAME_PATTERN.format(
                        func_name=re.escape(old_arg_name)
                    )
                    replace_pattern = new_arg_name
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
                                search=search_pattern,
                                replace=replace_pattern,
                                is_regex=True,
                                count=None,
                            )
                        ],
                        file_path=caller["file"],
                    )
                )

        return blocks


_S1192_MESSAGE_PATTERN = re.compile(
    r'Define a constant instead of duplicating this literal "(.*?)" (\d+) times\.'
)


class _LiteralCaches:
    """Caches for pre-scanned literal AST lookups."""

    def __init__(self) -> None:
        self.occurrences: Dict[str, List[Tuple[int, int, int]]] = {}
        self.existing: Dict[str, List[Tuple[str, int]]] = {}
        self.needing_names: List[LiteralContext] = []
        self.seen: Set[str] = set()


class _FixAccumulator:
    """Accumulates code blocks, constant defs, and reports during processing."""

    def __init__(self) -> None:
        self.code_blocks: List[CodeBlock] = []
        self.constant_defs: List[str] = []
        self.used_names: Set[str] = set()
        self.reports: List[Dict[str, Any]] = []


class StringLiteralDuplicateHandler(RuleHandler):
    """
    Handler for python:S1192 - string literals should not be duplicated.

    This rule flags string literals that appear 3 or more times in a file.
    The handler extracts each duplicated literal into a module-level constant
    and replaces all occurrences with the constant name.
    """

    RULE_ID = "python:S1192"
    MOIDY_LINE_RANGE = False

    def can_handle(self, rule: str) -> bool:
        return rule == self.RULE_ID

    async def generate_fixes(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        context: FixContext,
        project_path: Path,
        file_path: Path,
        llm_caller: Any = None,
        rule_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[SonarFixResponse]]:
        """Generate fixes for duplicated string literal issues (python:S1192)."""
        parsed = self._read_and_parse_file(file_path)
        if parsed is None:
            return None
        _, file_lines, tree = parsed

        file_path_str = str(file_path)
        relative_path = _resolve_relative_path(file_path, project_path)

        existing_names = self._collect_module_level_names(tree)
        caches = self._prescan_issues(issues, tree)
        naming_response = self._resolve_names(
            caches, existing_names, file_path_str, llm_caller
        )

        result = self._process_issues(
            issues, caches, naming_response, file_lines, file_path_str
        )
        if not result.code_blocks:
            return None

        return [self._build_response(result, relative_path)]

    @staticmethod
    def _read_and_parse_file(
        file_path: Path,
    ) -> Optional[Tuple[str, List[str], ast.Module]]:
        """Read and parse a Python source file, returning source, lines, and AST."""
        try:
            source = file_path.read_text(encoding="utf-8")
            file_lines = source.splitlines(keepends=True)
            tree = ast.parse(source, filename=str(file_path))
            return source, file_lines, tree
        except Exception:
            logger.error("Failed to read/parse %s", file_path, exc_info=True)
            return None

    @staticmethod
    def _collect_module_level_names(tree: ast.Module) -> Set[str]:
        """Collect existing module-level UPPERCASE names to avoid collisions."""
        names: Set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.add(target.id)
        return names

    def _prescan_issues(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        tree: ast.Module,
    ) -> "_LiteralCaches":
        """Pre-scan issues to cache AST lookups per unique literal."""
        caches = _LiteralCaches()
        for issue in issues:
            literal = self._extract_literal_from_message(issue.message)
            if literal is None or literal in caches.seen:
                continue
            caches.seen.add(literal)

            occurrences = self._find_string_occurrences(tree, literal)
            caches.occurrences[literal] = occurrences
            if not occurrences:
                continue

            existing = self._find_existing_constant(tree, literal)
            caches.existing[literal] = existing
            if len(existing) != 1:
                caches.needing_names.append(
                    LiteralContext(literal=literal, occurrences=occurrences)
                )
        return caches

    @staticmethod
    def _resolve_names(
        caches: "_LiteralCaches",
        existing_names: Set[str],
        file_path_str: str,
        llm_caller: Any,
    ) -> NamingResponse:
        """Call the naming service once for all literals needing new names."""
        naming_service = ConstantNamingService(
            llm_caller=LLMFixerAdapter(llm_caller) if llm_caller else None,
        )
        return naming_service.name_literals(
            NamingRequest(
                file_path=file_path_str,
                literals=caches.needing_names,
                existing_names=existing_names,
            )
        )

    def _process_issues(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        caches: "_LiteralCaches",
        naming_response: NamingResponse,
        file_lines: List[str],
        file_path_str: str,
    ) -> "_FixAccumulator":
        """Process each issue into code blocks and literal reports."""
        acc = _FixAccumulator()
        processed: Set[str] = set()

        for issue in issues:
            literal = self._extract_literal_from_message(issue.message)
            if literal is None or literal in processed:
                continue
            processed.add(literal)

            self._process_single_literal(
                issue,
                literal,
                caches,
                naming_response,
                file_lines,
                file_path_str,
                acc,
            )
        return acc

    def _process_single_literal(
        self,
        issue: Union[SonarIssue, SonarSecurityIssue],
        literal: str,
        caches: "_LiteralCaches",
        naming_response: NamingResponse,
        file_lines: List[str],
        file_path_str: str,
        acc: "_FixAccumulator",
    ) -> None:
        """Resolve one literal into replacement blocks and a report entry."""
        occurrences = caches.occurrences.get(literal, [])
        if not occurrences:
            return

        const_name, action, definition_line, occurrences = self._resolve_literal_action(
            literal, caches, naming_response, acc, occurrences
        )
        if not occurrences:
            return

        self._append_literal_results(
            issue,
            literal,
            const_name,
            action,
            definition_line,
            occurrences,
            file_lines,
            file_path_str,
            acc,
        )

    def _resolve_literal_action(
        self,
        literal: str,
        caches: "_LiteralCaches",
        naming_response: NamingResponse,
        acc: "_FixAccumulator",
        occurrences: List[Tuple[int, int, int]],
    ) -> Tuple[str, str, Optional[int], List[Tuple[int, int, int]]]:
        """Determine constant name, action, and filtered occurrences."""
        existing = caches.existing.get(literal, [])
        if len(existing) == 1:
            const_name = existing[0][0]
            definition_line = existing[0][1]
            occurrences = [o for o in occurrences if o[0] != definition_line]
            return const_name, "reused", definition_line, occurrences

        const_name = naming_response.names.get(
            literal,
            self._generate_constant_name(len(acc.constant_defs) + 1, acc.used_names),
        )
        acc.used_names.add(const_name)
        acc.constant_defs.append(f"{const_name} = {repr(literal)}")
        return const_name, "created", None, occurrences

    @staticmethod
    def _append_literal_results(
        issue: Union[SonarIssue, SonarSecurityIssue],
        literal: str,
        const_name: str,
        action: str,
        definition_line: Optional[int],
        occurrences: List[Tuple[int, int, int]],
        file_lines: List[str],
        file_path_str: str,
        acc: "_FixAccumulator",
    ) -> None:
        """Build replacement blocks and append the report entry."""
        blocks = StringLiteralDuplicateHandler._build_replacement_blocks(
            occurrences, file_lines, const_name, file_path_str
        )
        acc.code_blocks.extend(blocks)
        acc.reports.append(
            {
                "message": issue.message,
                "literal": literal,
                "const_name": const_name,
                "action": action,
                "definition_line": definition_line,
                "lines": [occ[0] for occ in occurrences],
            }
        )

    @staticmethod
    def _build_response(acc: "_FixAccumulator", relative_path: str) -> SonarFixResponse:
        """Assemble the final SonarFixResponse from accumulated results."""
        helper_code = "\n".join(acc.constant_defs)
        all_blocks = list(acc.code_blocks)

        if acc.constant_defs:
            insert_block = StringLiteralDuplicateHandler._create_constant_insert_block(
                helper_code, all_blocks
            )
            all_blocks.insert(0, insert_block)

        explanation = StringLiteralDuplicateHandler._build_explanation(
            relative_path, acc.reports
        )
        return SonarFixResponse(
            IMPORT_BLOCK="",
            FIXED_CODE_BLOCKS=all_blocks,
            NEW_HELPER_CODE=helper_code,
            PLACEMENT=PlacementType.GLOBAL_TOP,
            EXPLANATION=explanation,
            CONFIDENCE=0.95,
        )

    @staticmethod
    def _create_constant_insert_block(
        helper_code: str, existing_blocks: List[CodeBlock]
    ) -> CodeBlock:
        """Create a CodeBlock that inserts constant definitions at module top."""
        return CodeBlock(
            block_name="New constants",
            start_line=0,
            end_line=0,
            has_changes=True,
            change_type=ChangeType.DIFF,
            block_type=BlockType.MODULE,
            file_path=existing_blocks[0].file_path if existing_blocks else None,
            changes=[
                LineChange(
                    line=0,
                    action=ChangeAction.INSERT,
                    new=helper_code,
                )
            ],
        )

    @staticmethod
    def _extract_literal_from_message(message: str) -> Optional[str]:
        """Extract the duplicated string value from a SonarCloud S1192 message."""
        match = _S1192_MESSAGE_PATTERN.search(message)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _build_explanation(
        relative_path: str, literal_reports: List[Dict[str, Any]]
    ) -> str:
        """Build a structured explanation from per-literal metadata."""
        parts: List[str] = [f"**File:** `{relative_path}`"]
        for report in literal_reports:
            parts.append(StringLiteralDuplicateHandler._format_literal_report(report))
        return "\n\n".join(parts)

    @staticmethod
    def _format_literal_report(report: Dict[str, Any]) -> str:
        """Format a single literal report for the explanation."""
        lines_list = "\n".join(f"        - {ln}" for ln in report["lines"])
        if report["action"] == "reused":
            action_text = (
                f"Reused existing constant "
                f"`{report['const_name']}` "
                f"(defined at line {report['definition_line']})"
            )
        else:
            action_text = (
                f"Created "
                f"`{report['const_name']} = {repr(report['literal'])}` "
                f"at module level"
            )
        return (
            f"> {report['message']}\n"
            f"    - **Action:** {action_text}\n"
            f"    - **Lines affected:**\n"
            f"{lines_list}"
        )

    @staticmethod
    def _find_existing_constant(tree: ast.Module, target: str) -> List[Tuple[str, int]]:
        """
        Find module-level simple assignments whose value is the target string.

        Walks only top-level statements (direct children of ast.Module) and
        checks both ast.Assign and ast.AnnAssign nodes for a single ast.Name
        target with an ast.Constant string value matching the target.

        Args:
            tree:   Parsed AST of the source file.
            target: The duplicated string literal to search for.

        Returns:
            List of (variable_name, line_number) tuples sorted by line number.
            Empty list if no matching assignments found.
        """
        matches: List[Tuple[str, int]] = []
        for node in tree.body:
            result = StringLiteralDuplicateHandler._match_assignment_value(node, target)
            if result is not None:
                matches.append(result)
        matches.sort(key=lambda m: m[1])
        return matches

    @staticmethod
    def _match_assignment_value(
        node: ast.stmt, target: str
    ) -> Optional[Tuple[str, int]]:
        """Check if a top-level AST node assigns the target string value."""
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                return None
            name_node = node.targets[0]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            name_node = node.target
            if node.value is None:
                return None
            value_node = node.value
        else:
            return None

        if not isinstance(name_node, ast.Name):
            return None
        if not isinstance(value_node, ast.Constant):
            return None
        if not isinstance(value_node.value, str):
            return None
        if value_node.value != target:
            return None
        return (name_node.id, node.lineno)

    @staticmethod
    def _find_string_occurrences(
        tree: ast.Module, target: str
    ) -> List[Tuple[int, int, int]]:
        """
        Walk the AST and return (lineno, col_offset, end_col_offset) for every
        ast.Constant node whose value equals the target string.
        """
        occurrences: List[Tuple[int, int, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            if node.value != target:
                continue
            if node.end_col_offset is None:
                continue
            occurrences.append((node.lineno, node.col_offset, node.end_col_offset))
        return occurrences

    @staticmethod
    def _generate_constant_name(counter: int, used_names: Set[str]) -> str:
        """Return a sequential constant name like STRING_LITERAL_1."""
        name = f"STRING_LITERAL_{counter}"
        while name in used_names:
            counter += 1
            name = f"STRING_LITERAL_{counter}"
        return name

    @staticmethod
    def _build_replacement_blocks(
        occurrences: List[Tuple[int, int, int]],
        file_lines: List[str],
        constant_name: str,
        file_path: Optional[str] = None,
    ) -> List[CodeBlock]:
        """
        Build one SEARCH_REPLACE CodeBlock per occurrence.

        Uses the exact quoted representation from the source line (via column
        offsets) so that both single-quoted and double-quoted strings are
        handled correctly.
        """
        blocks: List[CodeBlock] = []
        for lineno, col_start, col_end in occurrences:
            line_idx = lineno - 1
            if line_idx < 0 or line_idx >= len(file_lines):
                continue
            quoted_literal = file_lines[line_idx][col_start:col_end]
            block = StringLiteralDuplicateHandler._create_replacement_block(
                lineno, quoted_literal, constant_name, file_path
            )
            blocks.append(block)
        return blocks

    @staticmethod
    def _create_replacement_block(
        lineno: int,
        quoted_literal: str,
        constant_name: str,
        file_path: Optional[str],
    ) -> CodeBlock:
        """Create a single SEARCH_REPLACE CodeBlock for one occurrence."""
        return CodeBlock(
            block_name=constant_name,
            start_line=lineno,
            end_line=lineno,
            has_changes=True,
            change_type=ChangeType.SEARCH_REPLACE,
            block_type=BlockType.MODULE,
            file_path=file_path,
            replacements=[
                SearchReplace(
                    search=quoted_literal,
                    replace=constant_name,
                    is_regex=False,
                    count=1,
                )
            ],
        )


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
        rule_info: Optional[Dict[str, Any]] = None,
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
            logger.debug(
                "AsyncToSyncHandler: processing %d issue(s) in %s",
                len(issues),
                file_path,
            )

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
                            PLACEMENT=PlacementType.SIBLING,
                            EXPLANATION="",
                            CONFIDENCE=0.95,
                        )
                    )

            logger.info(
                "Generated %d code block(s) for async-to-sync conversion",
                len(code_blocks),
            )

            response_lst.append(
                SonarFixResponse(
                    IMPORT_BLOCK="",
                    FIXED_CODE_BLOCKS=code_blocks,
                    NEW_HELPER_CODE="",
                    PLACEMENT=PlacementType.SIBLING,
                    EXPLANATION=explanation.format(caller_count=len(caller_blocks)),
                    CONFIDENCE=0.95,
                )
            )

            return response_lst

        except Exception:
            logger.exception("AsyncToSyncHandler failed to generate fixes")
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
        rule_info: Optional[Dict[str, Any]] = {},
    ) -> Optional[List[SonarFixResponse]]:
        """
        Generate fixes for cognitive complexity issues.

        This delegates to the LLM with specialized refactoring prompts.
        """
        if not llm_caller:
            logger.error("LLM caller required for cognitive complexity fixes")
            return None

        if rule_info is None:
            rule_info = {}
        try:
            fix_response = llm_caller._call_llm_list(
                issues,
                context,
                context.file_path.suffix,
                rule_info,
                project_path,
                file_path,
                error_message="",
            )
            return [fix_response]

        except Exception:
            logger.exception("CognitiveComplexityHandler failed to generate fixes")
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
        rule_info: Optional[Dict[str, Any]] = {},
    ) -> Optional[List[SonarFixResponse]]:
        """Generate fixes using standard LLM approach."""
        if not llm_caller:
            logger.error("LLM caller required for default rule handling")
            return None

        if rule_info is None:
            rule_info = {}

        try:
            fix_response = llm_caller._call_llm_list(
                issues,
                context,
                context.file_path.suffix,
                rule_info,
                project_path,
                file_path,
                error_message="",
            )
            return [fix_response]

        except Exception:
            logger.exception("DefaultRuleHandler failed to generate fixes")
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
            StringLiteralDuplicateHandler(),
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
                logger.debug("Using %s for rule %s", handler.__class__.__name__, rule)
                return handler

        logger.warning("No handler found for rule %s, using default", rule)
        return self.handlers[-1]
