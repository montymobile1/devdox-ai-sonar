"""LLM-powered code fixer for SonarCloud issues."""

import os
import re
import shutil
import sys
import asyncio
from collections import defaultdict

import subprocess

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from pathlib import Path
import json
from pydantic import ValidationError
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence
from datetime import datetime
import logging

from .fix_validator import FixValidator, ValidationStatus
from devdox_ai_sonar.models.file_structures import FixApplication, FixContext
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    FixSuggestion,
    FixResult,
    SonarFixResponse,
    CodeBlock,
    ChangeType,
    ChangeAction,
    LineChange,
)
from devdox_ai_sonar.utils.file_indentation import (
    apply_single_fix,
    write_file_lines,
    normalize_indentation,
    cleanup_tmp_py_file,
)
from devdox_ai_sonar.services.rule_handler import (
    RuleHandlerRegistry,
    _resolve_effective_values,
    _resolve_relative_path,
)
from devdox_ai_sonar.services.extractor import IssueExtractor
from devdox_ai_sonar.utils.async_file_io import AsyncFileReader

logger = logging.getLogger(__name__)


try:
    from together import Together

    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from google import genai
    from google.genai import types

    HAS_GEMINI = True
except ImportError as e:
    logger.warning(f"Failed to import Gemini library: {e}")
    HAS_GEMINI = False

java_extension = ".java"
scala_extension = ".scala"


FUNCTION_ALREADY_CALLED = (
    "• As function can be already called so don't remove the parameters."
)
STATICMETHOD_DECORATOR = "@staticmethod"
PYTHON_CODE_BLOCK = "```python"
CODE_BLOCK_END = "```"


class LLMFixer:
    """LLM-powered code fixer for SonarCloud issues."""

    def __init__(
        self,
        provider: Optional[str] = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        context_lines: int = 10,
    ):
        """
        Initialize the LLM fixer.

        Args:
            provider: LLM provider ("openai" or "gemini")
            model: Model name (defaults to provider's default)
            api_key: API key (uses environment variables if not provided)
            context_lines: Number of lines to include around the issue for context
        """
        self.provider = str(provider).lower()
        self.context_lines = context_lines
        self.file_reader = AsyncFileReader()
        self.model = model
        self.prompt_dir = Path(__file__).parent / "prompts"
        self.template_dir = Path(__file__).parent / "templates"
        self.client: Any = None
        self._validate_and_configure_provider(provider, model, api_key)
        self._setup_jinja_env()

    def _setup_jinja_env(self) -> None:
        """Setup Jinja2 environment with custom filters"""
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=select_autoescape(["html", "xml"]),
        )
        self.jinja_env_templates = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=select_autoescape(["html", "xml"]),
        )

    async def read_file_async(self, file_path: str | Path) -> str:
        """Async version."""
        return await self.file_reader.read_text(file_path)

    def read_file(self, file_path: str | Path) -> str:
        """Sync version (backwards compatible)."""
        return asyncio.run(self.read_file_async(file_path))

    def _validate_and_configure_provider(
        self, provider: Optional[str], model: Optional[str], api_key: Optional[str]
    ) -> None:
        provider = str(provider).lower()
        if provider == "togetherai":
            self._configure_togetherai(model, api_key)
        elif provider == "openai":
            self._configure_openai(model, api_key)
        elif provider == "gemini":
            self._configure_gemini(model, api_key)
        elif provider == "openrouter":
            self._configure_openrouter(model, api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'openai', 'gemini', 'togetherai', or 'openrouter'"
            )

    def _configure_togetherai(
        self, model: Optional[str], api_key: Optional[str]
    ) -> None:
        if not HAS_TOGETHER:
            raise ImportError(
                "Together AI library not installed. Install with: pip install together"
            )
        self.model = model or "gpt-4o"
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Together API key not provided. Set TOGETHER_API_KEY environment variable."
            )

        self.client = Together(api_key=self.api_key)

    def _configure_openai(self, model: Optional[str], api_key: Optional[str]) -> None:
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        self.model = model or "gpt-4o"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(api_key=self.api_key)

    def _configure_gemini(self, model: Optional[str], api_key: Optional[str]) -> None:
        if not HAS_GEMINI:
            raise ImportError(
                "Gemini library not installed. Install with: pip install google-genai"
            )

        self.model = model or "claude-3-5-sonnet-20241022"
        self.api_key = api_key or os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_KEY environment variable."
            )

        self.client = genai.Client(api_key=self.api_key)

    def _configure_openrouter(
        self, model: Optional[str], api_key: Optional[str]
    ) -> None:
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        self.model = model or "anthropic/claude-sonnet-4"
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable."
            )

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://devdox.ai",
                "X-Title": "DevDox AI Sonar",
            },
        )

    @staticmethod
    def _convert_regex_to_diff(
        block: CodeBlock, file_cache: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Convert a SearchReplace block to a DIFF with actual source code.

        Reads the real source line from disk, applies the replacement, and
        returns a dict of updates for model_copy (change_type, changes,
        replacements).  Handles both regex and plain-string replacements.
        Returns an empty dict if conversion is not applicable or fails.
        """
        if not (
            block.change_type == ChangeType.SEARCH_REPLACE
            and block.replacements
            and block.file_path
        ):
            return {}

        try:
            abs_path = block.file_path
            if abs_path not in file_cache:
                file_cache[abs_path] = (
                    Path(abs_path).read_text(encoding="utf-8").splitlines()
                )
            lines = file_cache[abs_path]
            if not (0 < block.start_line <= len(lines)):
                return {}

            source_line = lines[block.start_line - 1]
            fixed_line = source_line
            for repl in block.replacements:
                if repl.is_regex:
                    fixed_line = re.sub(
                        repl.search, repl.replace, fixed_line, count=repl.count or 0
                    )
                else:
                    fixed_line = fixed_line.replace(
                        repl.search, repl.replace, repl.count or 1
                    )
            return {
                "change_type": ChangeType.DIFF,
                "replacements": None,
                "changes": [
                    LineChange(
                        line=block.start_line,
                        action=ChangeAction.REPLACE,
                        old=source_line.strip(),
                        new=fixed_line.strip(),
                    )
                ],
            }
        except (IndexError, OSError):
            return {}

    @staticmethod
    def _build_display_blocks(
        all_code_blocks: List[CodeBlock],
        project_path: Optional[Path] = None,
    ) -> List[CodeBlock]:
        """Create documentation-friendly display copies of code blocks.

        Converts absolute file paths to project-relative paths and replaces
        regex SearchReplace blocks with human-readable DIFF blocks showing
        actual before/after source code. Original blocks are not mutated.
        """
        display_blocks = []
        file_cache: Dict[str, List[str]] = {}

        for block in all_code_blocks:
            updates: Dict[str, Any] = {}

            if block.file_path and project_path:
                try:
                    updates["file_path"] = str(
                        Path(block.file_path).relative_to(project_path)
                    )
                except ValueError:
                    pass

            updates.update(LLMFixer._convert_regex_to_diff(block, file_cache))

            if updates:
                display_blocks.append(block.model_copy(update=updates))
            else:
                display_blocks.append(block)

        return display_blocks

    def write_explaination(
        self,
        file_md: Path,
        fix_response_lst: List[SonarFixResponse],
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        original_code: Dict[str, Any],
        project_path: Optional[Path] = None,
    ) -> None:
        file_md.parent.mkdir(parents=True, exist_ok=True)
        if not file_md.exists():
            file_md.touch()

        all_code_blocks = []
        for resp in fix_response_lst:
            all_code_blocks.extend(resp.FIXED_CODE_BLOCKS)

        display_blocks = self._build_display_blocks(all_code_blocks, project_path)
        explanation = fix_response_lst[-1].EXPLANATION
        template = self.jinja_env_templates.get_template("md.j2")

        with open(file_md, mode="a", encoding="utf-8") as f:
            for issue in issues:
                context = {
                    "rule": getattr(issue, "rule", "Unknown rule"),
                    "severity": getattr(issue, "severity", ""),
                    "message": getattr(issue, "message", "No message provided"),
                    "file_path": getattr(issue, "file_path", "Unknown file"),
                    "line": getattr(issue, "line", "N/A"),
                    "explanation": explanation,
                    "suggestion": display_blocks,
                    "original_code": original_code,
                }
                template_md = template.render(**context)
                f.write(template_md.strip() + "\n\n")

    async def generate_fix_by_file(
        self,
        issues: List[Union[SonarIssue, SonarSecurityIssue]],
        project_path: Path,
        tmp_path: Path,
        modified_content: str = "",
        file_md: str = "",
        check_tmp_path: bool = True,
        rule_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[FixSuggestion]]:
        """
        Generate fix suggestions for issues in a single file.

        REFACTORED: This method now has single responsibility - orchestration.
        All complex logic delegated to specialized helpers.

        Args:
            issues: List of SonarCloud issues to fix (must be from same file)
            project_path: Path to the project root
            tmp_path: Path to temporary files
            modified_content: Optional pre-extracted content
            file_md: Optional markdown file path for documentation

        Returns:
            List of FixSuggestion objects or None if fix cannot be generated
        """
        if not issues:
            logger.warning("No issues provided for fix generation")
            return None

        rule_registry = RuleHandlerRegistry()
        extractor = IssueExtractor(self.file_reader)

        try:
            # Step 1: Validate issues and extract file information
            validation = await extractor.validate_issue_group(
                issues, tmp_path, project_path, check_tmp_path
            )

            if not validation.is_valid:
                logger.error(f"Validation failed: {validation.error}")
                return None

            if validation.file_path is None or validation.line_range is None:
                logger.error("Validation passed but file_path or line_range is None")
                return None

            # Step 2: Prepare context for fix generation
            context = await self._prepare_fix_context(
                validation.file_path,
                validation.line_range,
                modified_content,
            )

            if not context:
                logger.warning("Failed to prepare fix context")
                return None

            # Step 3: Get appropriate handler and generate fix response
            handler = rule_registry.get_handler(issues[0].rule)

            fix_response_lst = await handler.generate_fixes(
                issues,
                context,
                project_path,
                validation.file_path,
                llm_caller=self,  # Pass self for LLM access
                rule_info=rule_info,
            )

            fix_response_lst = [r for r in (fix_response_lst or []) if r is not None]
            if not fix_response_lst:
                logger.warning(f"Handler returned no fix for rule {issues[0].rule}")
                return None

            # Step 4: Build FixSuggestion from the response
            modify_line_range = getattr(handler, "MOIDY_LINE_RANGE", False)
            fix_suggestion_lst = self._build_fix_suggestion(
                fix_response_lst,
                context,
                validation.file_path,
                project_path,
                validation.line_range,
                modify_line_range,
            )

            # Step 5: Write documentation if requested
            if file_md:
                self.write_explaination(
                    project_path / file_md,
                    fix_response_lst,
                    issues,
                    context.context_dict,
                    project_path=project_path,
                )

            logger.info(f"Generated fix for {len(issues)} issue(s)")
            return fix_suggestion_lst

        except FileNotFoundError as e:
            logger.warning(f"File not found: {e}")
            return None
        except Exception:
            logger.exception("Error generating fixes")
            return None

    def _map_fix_suggestion_to_fix_suggestion_dto(
        self,
        line_range: Dict[str, Any],
        context: FixContext,
        code_blocks: List[CodeBlock],
        fix_response_single: SonarFixResponse,
        llm_model: str,
        relative_file_path: str,
        effective_start_line: Optional[int],
        effective_sonar_line: Optional[int],
        effective_last_line: Optional[int],
    ) -> FixSuggestion:
        return FixSuggestion(
            issue_key=self._generate_fix_key(line_range.get("problem_lines", [])),
            original_code=context.code_content,
            fixed_code_blocks=code_blocks,
            fixed_code="",
            import_block_code=fix_response_single.IMPORT_BLOCK,
            helper_code=fix_response_single.NEW_HELPER_CODE,
            placement_helper=fix_response_single.PLACEMENT.value,
            explanation=fix_response_single.EXPLANATION or "",
            confidence=fix_response_single.CONFIDENCE,
            llm_model=llm_model,
            rule_description="",
            file_path=relative_file_path,
            line_number=effective_start_line,
            sonar_line_number=effective_sonar_line,
            end_import_block_code=context.context_dict.get("import_section", {}).get(
                "end_line", 0
            ),
            last_line_number=effective_last_line,
        )

    def _build_fix_suggestion(
        self,
        fix_response_list: List[SonarFixResponse],
        context: FixContext,
        file_path: Path,
        project_path: Path,
        line_range: Dict[str, Any],
        modify_line_range: bool = False,
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
            ) = _resolve_effective_values(
                code_blocks, file_path, context, line_range, modify_line_range
            )

            relative_file_path = _resolve_relative_path(
                effective_file_path, project_path
            )

            lst_suggestion.append(
                self._map_fix_suggestion_to_fix_suggestion_dto(
                    line_range=line_range,
                    context=context,
                    code_blocks=code_blocks,
                    fix_response_single=fix_response_single,
                    llm_model=self.model or "unknown",
                    relative_file_path=relative_file_path,
                    effective_start_line=effective_start_line,
                    effective_sonar_line=effective_sonar_line,
                    effective_last_line=effective_last_line,
                )
            )

        logger.debug(f"Built {len(lst_suggestion)} fix suggestions")
        return lst_suggestion

    def _generate_fix_key(self, problem_lines: List[int]) -> str:
        """
        Generate a meaningful key for the fix based on problem lines.

        Args:
            problem_lines: List of problem line numbers

        Returns:
            String key like "fix_L10" or "fix_L10-L15"
        """
        if not problem_lines:
            return "fix_unknown"

        if len(problem_lines) == 1:
            return f"fix_L{problem_lines[0]}"

        min_line = min(problem_lines)
        max_line = max(problem_lines)
        return f"fix_L{min_line}-L{max_line}"

    async def _prepare_fix_context(
        self,
        file_path: Path,
        line_range: Dict[str, Any],
        modified_content: str,
    ) -> Optional[FixContext]:
        """
        Prepare context for fix generation.

        Args:
            file_path: Path to the file
            line_range: Dictionary with line range information
            modified_content: Optional pre-extracted content


        Returns:
            FixContext object or None if preparation fails
        """
        try:
            # Read file content
            file_lines = await self.file_reader.read_lines(file_path)
            language = self._get_language_from_extension(file_path.suffix)

            # Extract import section
            import_section = self._extract_import_section(file_lines, language)

            # Extract class name
            first_problem_line_idx = line_range["first_line"] - 1
            class_name = self._extract_class_name_from_file(
                file_lines, first_problem_line_idx
            )
            problem_line_content = _extract_problem_lines(
                file_lines, line_range["problem_lines"]
            )
            # Build context dictionary
            if modified_content:
                context_dict = {
                    "context": modified_content,
                    "start_line": line_range["first_line"],
                    "end_line": line_range["last_line"],
                    "problem_line_content": _extract_problem_lines(
                        file_lines, line_range["problem_lines"]
                    ),
                    "class_name": class_name,
                    "import_section": import_section,
                    "language": language,
                }
            else:
                context_dict = self._extract_context(
                    file_lines,
                    line_range["first_line"],
                    line_range["last_line"],
                    line_range["problem_lines"],
                    self.context_lines,
                )
                context_dict["language"] = language
                if (
                    "class_name" not in context_dict
                    or context_dict["class_name"] is None
                ):
                    context_dict["class_name"] = class_name
                context_dict["import_section"] = import_section
            context_dict["problem_line_content"] = problem_line_content
            code_chunk = ""
            # 1. Context Setup
            full_content = context_dict.get("new_context", [])

            for content in full_content:
                code_chunk += content["context"] + "\n"

            return FixContext(
                file_path=file_path,
                file_path_tmp=file_path,  # Will be set by caller if different
                line_range=line_range,
                code_content=code_chunk,
                language=language,
                import_section=import_section,
                class_name=class_name,
                functions=context_dict.get("functions", []),
                context_dict=context_dict,
            )

        except Exception:
            logger.exception("Error preparing context")
            return None

    async def apply_fixes(
        self,
        fixes: List[FixSuggestion],
        project_path: Path,
        create_backup: bool = True,
        dry_run: bool = False,
    ) -> FixResult:
        """
        Apply multiple fixes to the project.

        Args:
            fixes: List of fix suggestions to apply
            project_path: Path to the project root
            create_backup: Whether to create a backup before applying fixes
            dry_run: If True, don't actually modify files

        Returns:
            FixResult with application results
        """
        result = FixResult(project_path=project_path, total_fixes_attempted=len(fixes))

        # Create backup if requested
        if create_backup and not dry_run:
            backup_path = self._create_backup(project_path)
            result.backup_created = True
            result.backup_path = backup_path

        # Group fixes by file for efficient processing
        fixes_by_file: Dict[str, List[FixSuggestion]] = {}

        for fix in fixes:
            # Extract file from issue key or find associated issue
            file_key = self._get_file_from_fix(fix, project_path)

            if file_key:
                if file_key not in fixes_by_file:
                    fixes_by_file[file_key] = []
                fixes_by_file[file_key].append(fix)

        # Apply fixes file by file
        for file_path_str, file_fixes in fixes_by_file.items():
            try:
                file_path = Path(file_path_str)
                success, _ = await self._apply_fixes_to_file(
                    file_path, file_fixes, dry_run
                )
                if success:
                    result.successful_fixes.extend(file_fixes)

                else:
                    result.failed_fixes.extend(
                        [
                            {
                                "fix": fix,
                                "error": f"Failed to apply fix to file {file_path}",
                            }
                            for fix in file_fixes
                        ]
                    )
                    logger.error(f"Failed to apply fixes to {file_path}")
            except Exception as e:
                result.failed_fixes.extend(
                    [{"fix": fix, "error": str(e)} for fix in file_fixes]
                )
                logger.error(
                    f"Error processing file {file_path_str}: {e}", exc_info=True
                )

        return result

    def _extract_context(
        self,
        lines: List[str],
        first_line_number: int,
        last_line_number: int,
        problem_lines: List[int],
        context_lines: int,
    ) -> Dict[str, Any]:
        """
        Extract context around a problematic line with intelligent function/method detection.

        If the issue is on a function/method definition line, extracts the complete function.
        Otherwise, provides normal context around the issue.

        Args:
            lines: All lines in the file
            first_line_number: First line number with the issue (1-indexed)
            last_line_number: Last line number with the issue (1-indexed)
            context_lines: Number of lines to include before/after

        Returns:
            Dictionary with context information including:
            - context: The extracted code context
            - problem_line: The specific problematic line
            - line_number: Line number of the issue
            - start_line: Starting line of context (1-indexed)
            - end_line: Ending line of context (1-indexed)
            - is_complete_function: Boolean indicating if complete function was extracted
            - function_name: Name of function if applicable
        """
        extractor = ContextExtractor(lines)
        # Convert to 0-indexed
        first_line_idx = first_line_number - 1
        last_line_idx = last_line_number - 1

        if first_line_idx >= len(lines) or last_line_idx >= len(lines):
            logger.error(
                "Line range %d-%d exceeds file length %d",
                first_line_number,
                last_line_number,
                len(lines),
            )
            return extractor._get_empty_context(first_line_number)

        functions_context = extractor._extract_all_functions_in_range(
            first_line_idx, last_line_idx, problem_lines
        )

        if functions_context:
            logger.debug(
                "Found %d function(s) in range %d-%d",
                len(functions_context.get("functions", [])),
                first_line_number,
                last_line_number,
            )

            return functions_context

            # Fallback: No functions found, use normal context
        logger.debug("No functions found in range, using normal context extraction")
        return extractor._extract_normal_context(
            first_line_idx, last_line_idx, context_lines
        )

    def _call_llm_list(
        self,
        issues: Union[List[SonarIssue], List[SonarSecurityIssue]],
        context: FixContext,
        file_extension: str,
        rule_info_dict: Dict[str, Dict[str, str]],
        error_message: str = "",
    ) -> Optional[SonarFixResponse]:
        """
        Call the LLM to generate a fix.

        Args:
            issue: SonarCloud issue
            context: Code context around the issue
            rule_info_dict: Rule info from sonar cloud
            file_extension: File extension to determine language

        Returns:
            Dictionary with fix information or None
        """
        if not self.model:
            logger.error("Model not configured")
            return None

        # Determine programming language
        language = self._get_language_from_extension(file_extension)

        logger.debug(
            "LLM call — provider=%s, model=%s, language=%s, issues=%d",
            self.provider,
            self.model,
            language,
            len(issues),
        )
        # Prepare prompt
        prompt, system_template = self._create_fix_prompt_list(
            issues, context, rule_info_dict, language, error_message
        )

        prompt_system = system_template.render()

        try:
            if self.provider == "openai":
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": prompt},
                    ],
                    text_format=SonarFixResponse,
                )
                return self._parse_openai_response(response)

            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SonarFixResponse,
                    ),
                )

                return response.parsed  # type: ignore[no-any-return]

            elif self.provider in ("togetherai", "openrouter"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=8000,
                    temperature=0.08,
                    response_format={"type": "json_object"},
                )
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                logger.debug(
                    "Token usage — input: %d, output: %d, total: %d",
                    input_tokens,
                    output_tokens,
                    total_tokens,
                )

                return self._parse_chat_completion_response(response)
            else:
                logger.error("Unknown provider: %s", self.provider)
                return None
        except Exception:
            logger.exception("Error calling %s LLM", self.provider)
            return None

    def _is_init_method(self, context: str) -> bool:
        """
        Detect if the code context contains an __init__ method or constructor.

        Args:
            context: Code context string

        Returns:
            True if this appears to be a constructor/init method
        """

        # Python __init__ method
        if re.search(r"def\s+__init__\s*\(", context):
            return True

        # Java/C# constructor patterns
        if re.search(r"public\s+\w+\s*\([^)]*\)\s*\{", context):
            return True

        # JavaScript constructor
        if re.search(r"constructor\s*\([^)]*\)\s*\{", context):
            return True

        # Check for typical initialization patterns
        init_patterns = [
            r"self\.\w+\s*=",  # Python self.attribute =
            r"this\.\w+\s*=",  # JavaScript/Java this.attribute =
            r"_\w+\s*=",  # Private member initialization
        ]

        init_count = sum(1 for pattern in init_patterns if re.search(pattern, context))

        # If we see mAny initialization patterns, likely a constructor
        return init_count >= 3

    def _extract_complexity_info(self, message: str) -> Dict[str, str]:
        """
        Extract complexity numbers from the issue message.

        Args:
            message: SonarQube issue message

        Returns:
            Dictionary with current and target complexity values
        """

        # Pattern: "Cognitive Complexity from X to the Y allowed"
        pattern = r"complexity from (\d+) to the (\d+) allowed"
        match = re.search(pattern, message, re.IGNORECASE)

        if match:
            return {"current": match.group(1), "target": match.group(2)}

        # SECURE: Explicitly exclude digits and match expected separators
        alt_pattern = r"complexity is (\d+)[^\d]*?maximum[^\d]*?(\d+)"
        alt_match = re.search(alt_pattern, message, re.IGNORECASE)

        if alt_match:
            return {"current": alt_match.group(1), "target": alt_match.group(2)}

        # Default fallback
        return {"current": "Unknown", "target": "15"}

    def _extend_strategies_for_issue(
        self,
        issue: Any,
    ) -> List[str]:
        msg_lower = getattr(issue, "message", "").lower()

        # Start with core rules (always included)

        strategies_list = [
            "🚨 CRITICAL - PRESERVE ORIGINAL FUNCTION:",
            ". Take care of syntax , logic and structure",
            "• Keep EXACT same function name from original code",
            "• Keep EXACT same function signature (name, parameters, types)",
            "• DO NOT rename the function",
            f"• DO NOT change {STATICMETHOD_DECORATOR} to instance method or vice versa",
            "• DO NOT invent new function names",
            " ",
            "🚨 CRITICAL - NO HALLUCINATION / NO INVENTION:",
            "• DO NOT create new attributes that don't exist in original code",
            "• DO NOT create new methods that aren't called in original code",
            "• DO NOT add new logic or validation that wasn't in original",
            "• If original code uses `promotion.is_active`, you MUST use `promotion.is_active`",
            "• DO NOT rename attributes (e.g., changing `is_active` to `status`)",
            "• DO NOT change enum values (e.g., if original has 'PROMOTION_NOT_ACTIVE', use exactly that)",
            "• DO NOT create new attributes that aren't in the original code",
            "• DO NOT guess at attribute names from other similar code"
            "• DO NOT invent new error messages or exception types",
            "• ONLY extract and reorganize EXISTING logic from the original code",
            ". Be sure that No duplicate queries - pass data as parameters"
            "• Avoid duplication of defined functions and methods"
            "",
            "Example of FORBIDDEN changes:",
            "❌ Original: `if not promotion.is_active:`",
            "❌ Changed to: `if promotion.status != PromotionStatusEnum.ACTIVE:`  # WRONG - 'status' doesn't exist!",
            "",
            "✅ Correct approach:",
            "✅ Keep: `if not promotion.is_active:`  # EXACT same check",
            "",
            "## CORE RULES:",
            "• Return ONLY valid JSON",
            "• Use ONLY types from original code or Built-in types ",
            "• Make MINIMAL changes necessary to satisfy the rule",
            "• Built-in types always OK: str, int, Optional, List, Dict",
            "• FIXED_SELECTION = complete function with fixes applied",
            "• NEW_HELPER_CODE = only new helper definitions (or empty string)",
            "",
            "🚨 REFACTORING vs REWRITING:",
            "✅ ALLOWED:",
            "- Extract logic into helpers with early returns",
            "- Remove redundant checks (helpers trust preconditions)",
            "- Rename local variables only",
            "- Simplify conditionals",
            " ",
            "❌ FORBIDDEN:",
            "- New validation logic",
            "- Changing attributes/data models",
            "- New repository calls",
            "- New enums/constants",
            "- Changing exceptions/messages",
            " ",
            "🚨 IMPORTS:",
            "- Scan helper code for ALL used symbols",
            "- Check against existing imports",
            "- Add missing symbols to IMPORT_BLOCK",
            "- Move function-internal imports to top",
            "- Type hints need direct imports (e.g., `date` not just `datetime`)",
            "- NO imports inside helpers",
            # "🚨 REFACTORING vs REWRITING:",
            # "",
            # "✅ REFACTORING (allowed):",
            # "• Extract existing logic into helper methods",
            # "• Reduce nesting with early returns",
            # "•  No redundant checks - helpers trust their preconditions",
            # "•  Preserve correct parameters - uses user_id, not invented promotion.user_id",
            # "•  Preserved control flow - early return logic intact"  ,
            # "• Rename variables for clarity (only local variables, not attributes)",
            # "• Simplify complex conditionals by breaking them down",
            # "",
            # "❌ REWRITING (forbidden):",
            # "• Adding new validation logic not in original",
            # "• Changing data model attributes (e.g., promotion.is_active → promotion.status)",
            # "• Creating new repository method calls not in original",
            # "• Inventing new enum values or constants",
            # "• Changing exception types or messages",
            # "",
            # "🚨 CRITICAL: IMPORT MANAGEMENT 🚨",
            # "",
            # "When creating helper methods:",
            # "• SCAN NEW_HELPER_CODE for ALL modules/symbols used",
            # "• CHECK each symbol against the Import Section shown above",
            # "• If symbol NOT in Import Section → ADD to IMPORT_BLOCK",
            # "• When moving imports from inside functions:",
            # " - If original uses `datetime.now().date()` → Import BOTH `datetime` and `date`",
            # " - If helper has type hint `-> date` → Import `date` explicitly",
            # " - Type annotations require direct imports even if parent class exists",
            # "• NEVER leave 'import' or 'from' statements inside helper functions",
            "",
            "Examples:",
            "❌ WRONG - Import inside helper:",
            f"{PYTHON_CODE_BLOCK}",
            "def __parse_date(self, date_str: str):",
            "    from datetime import datetime  # ❌ BAD",
            "    return datetime.strptime(date_str, '%Y-%m-%d')",
            "```",
            "",
            "✅ CORRECT - Import in IMPORT_BLOCK:",
            f"{PYTHON_CODE_BLOCK}",
            "# In IMPORT_BLOCK:",
            '"IMPORT_BLOCK": "from datetime import datetime \n"',
            "",
            "# In NEW_HELPER_CODE:",
            "def __parse_date(self, date_str: str):",
            "    return datetime.strptime(date_str, '%Y-%m-%d')  # ✅ GOOD",
            "```",
            "",
            "IMPORT_BLOCK Rules:",
            '• Empty string "" if all imports already in Import Section',
            "• Add ONLY missing imports (don't duplicate existing ones)",
            '• Format: "from module import symbol \n" or "import module \n"',
            "• Check Import Section (shown above) before adding",
            "",
        ]
        if "cognitive complexity" in msg_lower:
            comp_info = self._extract_complexity_info(getattr(issue, "message", ""))
            target = comp_info.get("target", "15")
            min_helpers = (max(3, int(target) // 6),)
            strategies_list.extend(
                [
                    f"TARGET: Reduce complexity to <{target}",
                    f"- Create at least {min_helpers} helpers in NEW_HELPER_CODE",
                    "- Update original function to call them all",
                ]
            )

        # Unused Code
        elif "unused" in getattr(issue, "rule", "").lower() or "unused" in msg_lower:
            strategies_list.append("• Remove ONLY the specific unused variable/import.")
            strategies_list.append(
                "• Do not break code that references adjacent lines."
            )

        # Literal Duplication
        elif "duplicating this literal" in msg_lower:
            match = re.search(
                r'duplicating this literal "([^"]+)"', getattr(issue, "message", "")
            )
            literal = match.group(1) if match else "the repeated value"
            strategies_list.extend(
                [
                    f"LITERAL DUPLICATION (extract '{literal}'):",
                    "• Create a constant with SCREAMING_SNAKE_CASE name",
                    "• Put constant definition in NEW_HELPER_CODE",
                    "• Use constant in FIXED_SELECTION",
                    "• PLACEMENT: GLOBAL_TOP",
                    "",
                ]
            )
        elif "be sure that every parameter is used" in msg_lower:
            strategies_list.append(FUNCTION_ALREADY_CALLED)
            strategies_list.append("• Print or Log the unused parameters values.")
            strategies_list.append(
                "• Check the syntax and be sure that is working code"
            )
        # Null Checks
        elif "null" in getattr(issue, "rule", "").lower() or "nullable" in msg_lower:
            strategies_list.extend(
                [
                    "NULL/NONE CHECK:",
                    "• Add defensive checks before accessing attributes",
                    "• Use: if value is not None: ...",
                    "",
                ]
            )

        strategies_list.extend(
            [
                "PLACEMENT GUIDE:",
                f"• SIBLING = Instance methods needing 'self' (no {STATICMETHOD_DECORATOR})",
                "• GLOBAL_TOP = Imports or constants",
                f"• GLOBAL_BOTTOM = Pure utilities (no 'self', no {STATICMETHOD_DECORATOR})",
                "",
                f"🚨 DO NOT USE {STATICMETHOD_DECORATOR} DECORATOR - it causes issues!",
            ]
        )

        return strategies_list

    def _create_fix_prompt_list(
        self,
        issues: Union[List[SonarIssue], List[SonarSecurityIssue]],
        context: FixContext,
        rule_info_list: Dict[str, Dict[str, Any]],
        language: str = "python",
        error_message: str = "",
    ) -> Tuple[str, Template]:
        """Create a concise, focused prompt for the LLM to generate a fix."""

        # 1. Context Setup
        code_chunk = context.code_content
        class_name = context.class_name
        strategies = []
        template = self.jinja_env.get_template("python/user_prompt.j2")
        system_template = self.jinja_env.get_template("python/system_fix_issues.j2")
        for issue in issues:
            rule_key = getattr(issue, "rule", "")

            if rule_key in ["python:S3776"]:
                template = self.jinja_env.get_template(
                    "python/refactoring/user_prompt.j2"
                )
                system_template = self.jinja_env.get_template(
                    "python/refactoring/system_fix_issues.j2"
                )

                context.import_section["has_imports"] = True

            if rule_key not in rule_info_list:
                continue
            if rule_key == "python:S1172":
                steps = (
                    rule_info_list.get(rule_key, {})
                    .get("how_to_fix", {})
                    .get("steps", [])
                )
                # Filter out unwanted step
                filtered_steps = [
                    step
                    for step in steps
                    if step
                    != "Remove unused elements or implement their intended purpose"
                ]
                issue.message = "Be sure that every parameter is used"

                # Add new strategies
                filtered_steps.append(FUNCTION_ALREADY_CALLED)
                filtered_steps.append("• Print or Log the unused parameters.")

                rule_info_list[rule_key]["how_to_fix"]["steps"] = filtered_steps
                rule_info_list[rule_key]["name"] = (
                    "Be sure that every parameter is used"
                )
                rule_info_list[rule_key]["root_cause"] = None

            strategies = self._extend_strategies_for_issue(
                issue=issue,
            )

        # Detect method type from ORIGINAL code
        original_is_static = STATICMETHOD_DECORATOR in code_chunk
        original_has_self = (
            "def " in code_chunk and "self" in code_chunk.split("def ")[1].split(")")[0]
        )

        if original_is_static:
            method_instruction_list = [
                f"🚨 ORIGINAL METHOD IS {STATICMETHOD_DECORATOR}:",
                f"- Keep {STATICMETHOD_DECORATOR} decorator in FIXED_SELECTION",
                f"- ALL helper methods MUST also be {STATICMETHOD_DECORATOR}",
                "- NO 'self' or 'cls' parameters in ANY helpers",
                f"- Call ALL helpers as: {class_name or 'ClassName'}._helper_name(args)",
                "- ALL helpers use PLACEMENT: SIBLING",
                "",
                f"✅ CONSISTENCY: Original {STATICMETHOD_DECORATOR} → ALL helpers {STATICMETHOD_DECORATOR}",
                "",
            ]

        elif original_has_self:
            method_instruction_list = [
                "🚨 ORIGINAL METHOD USES 'self' (INSTANCE METHOD):",
                "- Keep 'self' as first parameter in FIXED_SELECTION",
                "- ALL helper methods MUST have 'self' as first parameter",
                f"- NO {STATICMETHOD_DECORATOR} decorator on ANY helpers",
                "- Call ALL helpers as: self._helper_name(args)",
                "- ALL helpers use PLACEMENT: SIBLING",
                "",
                "✅ CONSISTENCY: Original has 'self' → ALL helpers have 'self'",
                "",
                f"❌ FORBIDDEN: Creating {STATICMETHOD_DECORATOR} helpers for instance method",
                "",
            ]

        else:
            method_instruction_list = [
                "🚨 METHOD TYPE UNCLEAR:",
                "- Preserve the original method signature exactly",
                "- Match helper style to original method style",
                "",
            ]

        method_instruction = "\n".join(method_instruction_list)

        # 3. Construct Prompt
        # We join strategies with newlines for a clean list
        strategy_text = "\n".join(strategies)

        # Prepare context for template
        context_dic = {
            "language": language,
            "issues": issues,
            "rule_info": rule_info_list,
            "context": context,
            "method_instruction": method_instruction,
            "error_message": error_message,
            "strategy_text": strategy_text,
        }
        # Render enhanced content
        prompt = template.render(**context_dic)

        return prompt.strip(), system_template

    def _parse_openai_response(self, response: Any) -> Optional[SonarFixResponse]:
        """Parse OpenAI API response."""
        try:
            return response.output_parsed  # type: ignore[no-any-return]

        except Exception:
            logger.exception("Error parsing OpenAI response")
            return None

    def _parse_gemini_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parse Gemini API response."""
        try:
            content = response.text
            return self._extract_fix_from_response(content)
        except Exception:
            logger.exception("Error parsing Gemini response")
            return None

    def _parse_chat_completion_response(
        self, response: Any
    ) -> Optional[SonarFixResponse]:
        """Parse an OpenAI-compatible chat completion response (used by TogetherAI and OpenRouter)."""
        try:
            content = response.choices[0].message.content
            return self.parse_llm_response(content)
        except Exception:
            logger.exception("Error parsing %s response", self.provider)
            return None

    # Aliases for backward compatibility and test clarity
    _parse_togetherai_response = _parse_chat_completion_response
    _parse_openrouter_response = _parse_chat_completion_response

    def _extract_using_regex_fallback(self, content: str) -> Optional[Dict[str, Any]]:
        """Fallback extraction using regex for malformed JSON."""

        try:
            results = self._apply_regex_patterns(content)
            return self._validate_results(results)
        except Exception as e:
            logger.error(f"Regex fallback failed: {e}")
            return None

    def _apply_regex_patterns(self, content: str) -> Dict[str, Any]:
        patterns = {
            "IMPORT_BLOCK": r'"IMPORT_BLOCK"\s*:\s*"((?:[^"\\]|\\.)*)"|\'IMPORT_BLOCK\'\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            "FIXED_SELECTION": r'"FIXED_SELECTION"\s*:\s*"((?:[^"\\]|\\.)*)"|\'FIXED_SELECTION\'\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            "NEW_HELPER_CODE": r'"NEW_HELPER_CODE"\s*:\s*"((?:[^"\\]|\\.)*)"|\'NEW_HELPER_CODE\'\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            "PLACEMENT": r'"PLACEMENT"\s*:\s*"([^"]*)"|\'PLACEMENT\'\s*:\s*\'([^\']*)\'',
            "EXPLANATION": r'"EXPLANATION"\s*:\s*"((?:[^"\\]|\\.)*)"|\'EXPLANATION\'\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            "CONFIDENCE": r'"CONFIDENCE"\s*:\s*([0-9]*\.?[0-9]+)',
        }

        results: Dict[str, Any] = {}

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            value = self._get_match_value(match)
            results[key] = self._process_match_value(key, value)
        return results

    def _get_match_value(self, match: Any) -> Any:
        if match:
            return match.group(1) or (
                match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            )
        return None

    def _process_match_value(self, key: str, value: Optional[str]) -> Any:
        if key == "CONFIDENCE":
            if value is not None:
                return float(value)
            return 0.5
        if key == "PLACEMENT":
            return "SIBLING"
        # default processing for other keys
        if value is None:
            return ""
        value = value.replace('"', '"').replace(  # escaped quotes
            "\\\\", "\\"
        )  # escaped backslashes
        value = value.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
        return value.strip()

    def _validate_results(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not results.get("FIXED_SELECTION"):
            logger.error(" No FIXED_SELECTION found")
            return None

        return {
            "import_block": results["IMPORT_BLOCK"],
            "fixed_code": results["FIXED_SELECTION"],
            "helper_code": results["NEW_HELPER_CODE"],
            "placement_helper": results["PLACEMENT"].upper(),
            "explanation": results["EXPLANATION"] or "Code fix applied",
            "confidence": max(0.0, min(1.0, results["CONFIDENCE"])),
        }

    def _extract_fields_from_parsed_json(
        self, fix_data: dict
    ) -> Optional[Dict[str, Any]]:
        """Extract and validate fields from parsed JSON."""

        # Extract with type conversion and defaults
        fixed_code = str(fix_data.get("FIXED_SELECTION", "")).strip()

        fixed_code = (
            fixed_code.replace('\\"', '"').replace("\\", "").replace("\\'", "'")
        )

        import_block = str(fix_data.get("IMPORT_BLOCK", "")).strip()

        helper_code = str(fix_data.get("NEW_HELPER_CODE", "")).strip()

        helper_code = (
            helper_code.replace("\\", "").replace('\\"', '"').replace("\\'", "'")
        )
        placement = str(fix_data.get("PLACEMENT", "SIBLING")).strip().upper()

        explanation = str(fix_data.get("EXPLANATION", "")).strip()
        confidence = fix_data.get("CONFIDENCE", 0.5)

        # Convert confidence to float
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # Validate placement
        if placement not in ["SIBLING", "GLOBAL_TOP", "GLOBAL_BOTTOM"]:
            placement = "SIBLING"

        # Provide default explanation
        if not explanation:
            explanation = "Code fix applied"

        # # Require fixed_code
        # if not fixed_code:
        #     logger.error("❌ FIXED_SELECTION is empty")
        #     return None

        return {
            "import_block": import_block,
            "fixed_code": fixed_code,
            "helper_code": helper_code,
            "placement_helper": placement,
            "explanation": explanation,
            "confidence": confidence,
        }

    def parse_llm_response(self, response_json: str) -> SonarFixResponse:
        """
        Parse and validate LLM response.

        Args:
            response_json: Raw JSON string from LLM

        Returns:
            Validated SonarFixResponse object

        Raises:
            ValidationError: If response doesn't match schema
        """

        try:
            # Parse JSON
            data = json.loads(response_json)

            # Validate with Pydantic
            return SonarFixResponse(**data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e} for response {response_json}")

        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}")

    def _extract_fix_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Robust extraction that handles various JSON response formats.
        """
        try:
            # Step 1: Try direct JSON parsing first (for well-formed responses)
            cleaned_content = content.strip()

            cleaned_content.replace("```json", "").replace("```", "")
            cleaned_content = cleaned_content.split("{", 1)[1]
            cleaned_content = "{" + cleaned_content

            # 2. Trim after last }
            end = cleaned_content.rfind("}")
            cleaned_content = (
                cleaned_content[: end + 1] if end != -1 else cleaned_content
            )

            if cleaned_content.startswith("{") and cleaned_content.endswith("}"):
                try:
                    fix_data = json.loads(cleaned_content)

                    return self._extract_fields_from_parsed_json(fix_data)
                except json.JSONDecodeError:
                    pass  # Continue to cleaning steps

            # Step 2: Last resort - regex extraction
            return self._extract_using_regex_fallback(content)

        except Exception:
            logger.exception("Failed to extract LLM response")
            return None

    def _get_language_from_extension(self, extension: str) -> str:
        """Get programming language from file extension."""
        extension = extension.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            java_extension: "java",
            ".kt": "kotlin",
            scala_extension: "scala",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
        }
        return language_map.get(extension, "text")

    def _create_backup(self, project_path: Path) -> Path:
        """Create a backup of the project."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = project_path.parent / f"{project_path.name}_backup_{timestamp}"
        shutil.copytree(
            project_path, backup_path, ignore=shutil.ignore_patterns(".git")
        )

        return backup_path

    def _try_stored_file_path(
        self, fix: FixSuggestion, project_path: Path
    ) -> Optional[str]:
        """Try to get file from stored path in fix."""
        if not fix.file_path:
            return None

        file_path = project_path / fix.file_path
        return str(file_path) if file_path.exists() else None

    def _try_extract_from_issue_key(
        self, fix: FixSuggestion, project_path: Path
    ) -> Optional[str]:
        """Extract file path from SonarCloud issue key pattern."""
        if ":" not in fix.issue_key:
            return None

        parts = fix.issue_key.split(":")
        for part in parts:
            if not self._looks_like_file_path(part):
                continue

            file_path = project_path / part
            if file_path.exists():
                return str(file_path)

        return None

    def _get_file_from_fix(
        self, fix: FixSuggestion, project_path: Path
    ) -> Optional[str]:
        """
        Get file path associated with a fix.

        Args:
            fix: FixSuggestion containing issue information
            project_path: Path to the project root

        Returns:
            Absolute file path as string, or None if not found
        """
        # First, try to use the stored file path from the fix
        file_path = self._try_stored_file_path(fix, project_path)
        if file_path:
            return file_path

        # Fallback: Try to extract file path from issue key if it follows SonarCloud pattern
        file_path = self._try_extract_from_issue_key(fix, project_path)
        if file_path:
            return file_path

        # Strategy 3: Search by content
        file_path = self._try_find_by_content(fix, project_path)
        if file_path:
            return file_path

        logger.warning(f"Could not determine file path for fix {fix.issue_key}")
        return None

    def apply_indentation_to_fix(self, fixed_code: str, base_indent: str) -> str:
        """
        Apply base indentation to fixed code while preserving relative indentation.

        Args:
            fixed_code: The fixed code to indent
            base_indent: Base indentation to apply

        Returns:
            Properly indented fixed code
        """
        if not fixed_code.strip():
            return fixed_code

        lines = fixed_code.split("\n")
        if not lines:
            return fixed_code

        # Remove common leading whitespace (normalize)
        lines = normalize_indentation(lines)

        # Apply base indentation to all non-empty lines
        indented_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                indented_lines.append(base_indent + line)
            else:  # Empty line
                indented_lines.append(line)

        return "\n".join(indented_lines)

    async def _apply_fixes_to_file(
        self, file_path: Path, fixes: List[FixSuggestion], dry_run: bool
    ) -> Tuple[bool, List[FixApplication]]:
        if dry_run:
            return True, []

        logger.debug("Applying %d fix(es) to %s", len(fixes), file_path)
        try:
            lines = await self.file_reader.read_lines(file_path)
            results = []
            for fix in fixes:
                result, lines = apply_single_fix(lines, fix)
                if not result.success:
                    logger.warning("Fix %s skipped: %s", fix.issue_key, result.reason)
                    continue
                file_path_tmp = file_path.with_suffix(f".tmp{file_path.suffix}")

                write_file_lines(file_path_tmp, lines)
                validate, msg = self.check_python_interpreter(file_path_tmp)
                cleanup_tmp_py_file(file_path_tmp)
                result.success = validate
                result.reason = msg or ""

                results.append(result)
                if result.success:
                    logger.debug("Fix %s applied and validated", fix.issue_key)
                    write_file_lines(file_path, lines)
                else:
                    logger.debug("Fix %s failed validation: %s", fix.issue_key, msg)

            succeeded = sum(1 for r in results if r.success)
            logger.debug(
                "Finished %s — %d/%d fixes succeeded",
                file_path,
                succeeded,
                len(results),
            )
            return all(r.success for r in results), results

        except Exception:
            logger.exception("Error applying fixes to %s", file_path)
            return False, []

    def check_python_interpreter(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Check Python syntax and formatting.

        Returns:
            (True, None) if everything is valid
            (False, error_message) otherwise
        """
        try:
            # 1. Syntax check
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(file_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            return True, None

        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout or "Unknown error"
            return False, error_output.strip()

    def _check_bracket_balance(self, content: str) -> bool:
        """
        Check if brackets, parentheses, and braces are balanced.

        This is a simple check that doesn't account for strings/comments
        but catches obvious structural corruption.
        """
        stack: List[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}

        for char in content:
            if char in pairs.keys():
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack[-1]] != char:
                    return False
                stack.pop()

        return len(stack) == 0

    def _check_no_duplicate_definitions(
        self, content: str, file_extension: str
    ) -> bool:
        """
        Check for duplicate function/class definitions.

        Duplicate definitions are a strong indicator that code was incorrectly
        inserted/replaced, as this would almost never happen intentionally.
        """
        if file_extension != ".py":
            return True  # Only check Python files for now

        # Find all function and class definitions
        func_pattern = r"^\s*(async\s+)?def\s+(\w+)\s*\("
        class_pattern = r"^\s*class\s+(\w+)\s*[:\(]"

        definitions: Dict[str, List[int]] = {}  # name -> line numbers where defined

        for line_num, line in enumerate(content.split("\n"), 1):
            func_match = re.match(func_pattern, line)
            if func_match:
                def_name = func_match.group(2)
                if def_name not in definitions:
                    definitions[def_name] = []
                definitions[def_name].append(line_num)

            class_match = re.match(class_pattern, line)
            if class_match:
                def_name = class_match.group(1)
                if def_name not in definitions:
                    definitions[def_name] = []
                definitions[def_name].append(line_num)

        # Check for duplicates
        for def_name, line_numbers in definitions.items():
            if len(line_numbers) > 1:
                logger.error(
                    f"Found duplicate definition '{def_name}' at lines: {line_numbers}"
                )
                return False

        return True

    def _create_backup_if_needed(
        self,
        result: FixResult,
        project_path: Path,
        create_backup: bool,
        dry_run: bool,
    ) -> None:
        """Conditionally create a project backup and update the result."""
        if create_backup and not dry_run:
            backup_path = self._create_backup(project_path)
            result.backup_created = True
            result.backup_path = backup_path

    def _group_fixes_by_file(
        self,
        fixes: List[FixSuggestion],
        issues: Sequence[Union[SonarIssue, SonarSecurityIssue]],
        project_path: Path,
    ) -> Dict[str, List[Tuple[FixSuggestion, Union[SonarIssue, SonarSecurityIssue]]]]:
        """Group fix/issue pairs by resolved file path."""
        fixes_by_file: Dict[
            str, List[Tuple[FixSuggestion, Union[SonarIssue, SonarSecurityIssue]]]
        ] = defaultdict(list)
        for fix, issue in zip(fixes, issues):
            file_key = self._get_file_from_fix(fix, project_path)
            if file_key:
                fixes_by_file[file_key].append((fix, issue))
        return dict(fixes_by_file)

    async def _process_single_file(
        self,
        file_path: Path,
        file_fix_pairs: List[
            Tuple[FixSuggestion, Union[SonarIssue, SonarSecurityIssue]]
        ],
        result: FixResult,
        validator: Optional[FixValidator],
        dry_run: bool,
    ) -> None:
        """Process all fixes for a single file: try direct application, fall back to validator."""
        file_fixes = [fix for fix, _ in file_fix_pairs]

        original_content = ""
        if file_path.exists():
            original_content = await self.read_file_async(str(file_path))

        success, new_fixes = await self._apply_fixes_to_file(
            file_path, file_fixes, dry_run
        )

        if success:
            result.successful_fixes.extend(file_fixes)
            return

        if validator:
            await self._handle_failed_fixes_with_validator(
                file_path,
                file_fix_pairs,
                new_fixes,
                original_content,
                result,
                validator,
                dry_run,
            )
        else:
            result.failed_fixes.extend(
                [
                    {
                        "fix": fix,
                        "error": "Direct application failed and no validator available",
                    }
                    for fix in file_fixes
                ]
            )
            logger.error(
                f"✗ Failed to apply fixes to {file_path} (no validator fallback)"
            )

    async def _handle_failed_fixes_with_validator(
        self,
        file_path: Path,
        file_fix_pairs: List[
            Tuple[FixSuggestion, Union[SonarIssue, SonarSecurityIssue]]
        ],
        new_fixes: List[Any],
        original_content: str,
        result: FixResult,
        validator: FixValidator,
        dry_run: bool,
    ) -> None:
        """Handle validator fallback when direct fix application fails."""
        reason_msg = ", ".join(
            fix_app.reason for fix_app in new_fixes if fix_app.reason is not None
        )

        logger.warning(
            f"Direct fix application failed for {file_path}. "
            f"Trying AI validator fallback to fix {reason_msg}..."
        )

        if not dry_run and file_path.exists():
            await self.file_reader.write_text(file_path, original_content)

        for fix, issue in file_fix_pairs:
            try:
                current_content = original_content
                file_path_tmp = file_path.with_suffix(f".tmp{file_path.suffix}")
                if file_path_tmp.exists():
                    current_content = await self.read_file_async(file_path_tmp)
                    cleanup_tmp_py_file(file_path_tmp)

                validation_result = validator.validate_fix(
                    fix,
                    issue,
                    current_content,
                    new_error_msg=reason_msg,
                )

                await self._process_validation_result(
                    validation_result,
                    fix,
                    file_path,
                    result,
                    dry_run,
                )

            except Exception as e:
                logger.error(
                    f"Error in validator fallback for fix {fix.issue_key}: {e}",
                    exc_info=True,
                )
                result.failed_fixes.append(
                    {"fix": fix, "error": f"Validator error: {str(e)}"}
                )

    async def _process_validation_result(
        self,
        validation_result: Any,
        fix: FixSuggestion,
        file_path: Path,
        result: FixResult,
        dry_run: bool,
    ) -> None:
        """Dispatch on validation status and update the result accordingly."""
        status = validation_result.status

        if status == ValidationStatus.APPROVED:
            result.successful_fixes.append(fix)
            return

        if status == ValidationStatus.REJECTED:
            logger.warning(
                f"✗ Fix {fix.issue_key} REJECTED by validator: "
                f"{validation_result.explanation}"
            )
            result.failed_fixes.append(
                {
                    "fix": fix,
                    "error": f"Rejected by validator: {validation_result.explanation}",
                }
            )
            return

        if status == ValidationStatus.NEEDS_REVIEW:
            logger.warning(
                f"? Fix {fix.issue_key} NEEDS_REVIEW: {validation_result.explanation}"
            )
            result.failed_fixes.append(
                {
                    "fix": fix,
                    "error": f"Needs manual review: {validation_result.explanation}",
                }
            )
            return

        # MODIFIED
        if not validation_result.final_fix:
            result.failed_fixes.append(
                {
                    "fix": fix,
                    "error": "Validator marked as MODIFIED but provided no improved fix",
                }
            )
            return

        improved_success, _ = await self._apply_fixes_to_file(
            file_path,
            [validation_result.final_fix],
            dry_run,
        )
        if improved_success:
            result.successful_fixes.append(validation_result.final_fix)
        else:
            result.failed_fixes.append(
                {
                    "fix": fix,
                    "error": "Validator improved fix but application still failed",
                }
            )

    async def apply_fixes_with_validation(
        self,
        fixes: List[FixSuggestion],
        issues: Sequence[Union[SonarIssue, SonarSecurityIssue]],
        project_path: Path,
        create_backup: bool = True,
        dry_run: bool = False,
        use_validator: bool = True,
        validator_provider: str = "openai",
        validator_model: Optional[str] = None,
        validator_api_key: Optional[str] = None,
        min_confidence: float = 0.7,
    ) -> FixResult:
        """
        Apply fixes with optional validation by a senior code reviewer agent.

        WORKFLOW:
        1. Group fixes by file
        2. Apply fixes to files directly first
        3. If _validate_modified_content fails, use AI validator as fallback
        4. AI validator can fix syntax errors or improve the applied fix

        Args:
            fixes: List of fix suggestions
            issues: List of corresponding SonarCloud issues
            project_path: Path to the project root
            create_backup: Whether to create a backup before applying
            dry_run: If True, don't actually modify files
            use_validator: If True, use AI validator as fallback when validation fails
            validator_provider: LLM provider for validation
            validator_model: LLM model for validation
            validator_api_key: API key for validator
            min_confidence: Minimum confidence threshold for approval

        Returns:
            FixResult with detailed application results
        """
        result = FixResult(project_path=project_path, total_fixes_attempted=len(fixes))

        self._create_backup_if_needed(result, project_path, create_backup, dry_run)

        validator = None
        if use_validator:
            validator = FixValidator(
                provider=validator_provider,
                model=validator_model,
                api_key=validator_api_key,
                min_confidence_threshold=min_confidence,
            )

        fixes_by_file = self._group_fixes_by_file(fixes, issues, project_path)

        for file_path_str, file_fix_pairs in fixes_by_file.items():
            try:
                file_path = Path(file_path_str)
                await self._process_single_file(
                    file_path,
                    file_fix_pairs,
                    result,
                    validator,
                    dry_run,
                )
            except Exception as e:
                result.failed_fixes.extend(
                    [{"fix": fix, "error": str(e)} for fix, _ in file_fix_pairs]
                )
                logger.error(
                    f"✗ Error processing file {file_path_str}: {e}", exc_info=True
                )

        return result

    def _try_find_by_content(
        self, fix: FixSuggestion, project_path: Path
    ) -> Optional[str]:
        """Search for file containing the original code snippet."""
        if not fix.original_code or len(fix.original_code.strip()) <= 10:
            return None

        matching_files = self._find_files_with_content(
            project_path, fix.original_code.strip()
        )

        return str(matching_files[0]) if matching_files else None

    def _find_files_with_content(self, project_path: Path, content: str) -> List[Path]:
        """
        Find files containing specific content.

        Args:
            project_path: Path to search in
            content: Content to search for

        Returns:
            List of file paths containing the content
        """
        matching_files = []

        # Define file extensions to search
        extensions = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            java_extension,
            ".kt",
            scala_extension,
            ".go",
            ".rs",
            ".cpp",
            ".c",
            ".cs",
            ".php",
            ".rb",
            ".swift",
        }

        try:
            # Search through source files
            for file_path in project_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix in extensions
                    and not any(
                        part.startswith(".") for part in file_path.parts
                    )  # Skip hidden dirs
                    and "node_modules" not in file_path.parts
                    and "venv" not in file_path.parts
                    and "__pycache__" not in file_path.parts
                ):
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            file_content = f.read()
                            if content in file_content:
                                matching_files.append(file_path)
                                # Limit to first few matches to avoid performance issues
                                if len(matching_files) >= 3:
                                    break
                    except (IOError, UnicodeDecodeError):
                        continue

        except Exception as e:
            logger.warning(f"Error searching for files with content: {e}")

        return matching_files

    def _looks_like_file_path(self, path_candidate: str) -> bool:
        """Check if string looks like a file path with supported extension."""
        if "/" not in path_candidate:
            return False

        # Define supported extensions
        supported_extensions = {
            ".py",
            ".js",
            java_extension,
            ".ts",
            ".jsx",
            ".tsx",
            ".kt",
            scala_extension,
        }

        return any(path_candidate.endswith(ext) for ext in supported_extensions)

    async def _prepare_context(
        self,
        file_path: Path,
        line_range: Dict[str, Any],
        modified_content: str,
        context_lines: int,
        language: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Read file and prepare context for LLM.

        Args:
            file_path: Path to the file
            line_range: Dictionary with first_line, last_line, problem_lines
            modified_content: Optional pre-extracted content
            context_lines: Number of context lines to include around issues

        Returns:
            Dictionary containing:
            - context_dict: Context information for LLM
            - file_lines: Full file content as list of lines
            - problem_line_content: List of actual problem line strings

            Returns None if file cannot be read
        """
        try:
            # Read file content
            file_lines = await self.file_reader.read_lines(file_path)

            # Extract problem line content
            # CRITICAL: SonarCloud uses 1-based line numbers, Python uses 0-based indexing
            problem_line_content = _extract_problem_lines(
                file_lines, line_range["problem_lines"]
            )

            first_problem_line_idx = line_range["first_line"] - 1
            class_name = self._extract_class_name_from_file(
                file_lines, first_problem_line_idx
            )

            # Log class name detection for debugging
            if class_name:
                logger.debug(
                    f"Detected class name: {class_name} at line {line_range['first_line']}"
                )
            else:
                logger.debug(
                    f"No class name detected for line {line_range['first_line']} (likely module-level function)"
                )

            import_section = self._extract_import_section(file_lines, language)

            import_section["has_imports"] = False
            # Build context dictionary
            if modified_content:
                # Use provided modified content
                context_dict = {
                    "context": modified_content,
                    "start_line": line_range["first_line"],
                    "end_line": line_range["last_line"],
                    "problem_line_content": problem_line_content,
                    "class_name": class_name,
                    "import_section": import_section,
                    "language": language,
                }
            else:
                # Extract context from file
                context_dict = self._extract_context(
                    file_lines,
                    line_range["first_line"],
                    line_range["last_line"],
                    line_range["problem_lines"],
                    context_lines,
                )
                context_dict["language"] = language
                if (
                    "class_name" not in context_dict
                    or context_dict["class_name"] is None
                ):
                    context_dict["class_name"] = class_name
                context_dict["import_section"] = import_section
            context_dict["problem_line_content"] = problem_line_content
            return {
                "context_dict": context_dict,
                "file_lines": file_lines,
                "problem_line_content": problem_line_content,
            }

        except UnicodeDecodeError as e:
            logger.error(f"File encoding error for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    def _extract_import_section(
        self, file_lines: List[str], language: str
    ) -> Dict[str, Any]:
        """
        Extract only the import section (line range and content) from a file.

        Args:
            file_lines: All lines in the file
            language: Programming language (python, javascript, java, etc.)

        Returns:
            Dictionary with:
            - start_line: First line number of imports (1-indexed)
            - end_line: Last line number of imports (1-indexed)
            - content: String containing all import statements
            - has_imports: Boolean indicating if imports were found
        """
        if language == "python":
            return self._extract_python_import_section(file_lines)
        else:
            logger.debug(f"Import extraction not implemented for language: {language}")
            return {"start_line": 0, "end_line": 0, "content": "", "has_imports": False}

    def _should_skip_docstring(self, stripped: str, start_line: Optional[int]) -> bool:
        """
        Check if line is a docstring that should be skipped.

        Args:
            stripped: Stripped line content
            start_line: Current start line (None if not found yet)

        Returns:
            True if line is a docstring to skip
        """
        return stripped.startswith(('"""', "'''"))

    def _handle_backslash_continuation_line(
        self,
        original_line: str,
        stripped: str,
        line_num: int,
        import_lines: List[str],
        state: Dict[str, bool],
    ) -> int:
        """
        Process a line that's part of a backslash continuation.

        Args:
            original_line: Original line with indentation
            stripped: Stripped line content
            line_num: Current line number (1-indexed)
            import_lines: List to append line to
            state: Dictionary with import state flags

        Returns:
            Updated end_line value
        """
        import_lines.append(original_line)

        # Check if this line also ends with backslash
        if not stripped.rstrip().endswith("\\"):
            state["in_backslash_continuation"] = False
            state["in_multiline_import"] = False

        return line_num

    def _handle_parentheses_import_line(
        self,
        original_line: str,
        stripped: str,
        line_num: int,
        import_lines: List[str],
        state: Dict[str, bool],
    ) -> int:
        """
        Process a line that's part of a parentheses-based multi-line import.

        Args:
            original_line: Original line with indentation
            stripped: Stripped line content
            line_num: Current line number (1-indexed)
            import_lines: List to append line to
            state: Dictionary with import state flags

        Returns:
            Updated end_line value
        """
        import_lines.append(original_line)

        if ")" in stripped:
            state["in_parentheses_import"] = False
            state["in_multiline_import"] = False

            # Check if there's a backslash after closing parenthesis
            if stripped.rstrip().endswith("\\"):
                state["in_backslash_continuation"] = True
                state["in_multiline_import"] = True

        return line_num

    def _extract_python_import_section(self, file_lines: List[str]) -> Dict[str, Any]:
        """
        Extract Python import section (handles multi-line imports with parentheses and backslash).

        Handles cases like:
        - from module import (
              symbol1,
              symbol2
          )
        - from module import symbol1, symbol2, \
              symbol3, symbol4
        - import module

        Returns line range and content of all import statements.
        """
        import_lines: List[str] = []
        start_line: Optional[int] = None
        end_line: Optional[int] = None

        state = {
            "in_multiline_import": False,
            "in_parentheses_import": False,
            "in_backslash_continuation": False,
        }

        for idx, line in enumerate(file_lines):
            line_num = idx + 1  # Convert to 1-indexed
            stripped = line.strip()
            original_line = line

            if self._should_skip_docstring(stripped, start_line):
                if start_line is not None:
                    break
                continue

            # Handle multi-line import with parentheses
            if state["in_parentheses_import"]:
                end_line = self._handle_parentheses_import_line(
                    original_line, stripped, line_num, import_lines, state
                )
                continue

            # Handle backslash continuation
            if state["in_backslash_continuation"]:
                end_line = self._handle_backslash_continuation_line(
                    original_line, stripped, line_num, import_lines, state
                )
                continue
            # Detect start of import statement
            if self._is_import_statement_start(stripped):
                if start_line is None:
                    start_line = line_num

                import_lines.append(original_line)
                end_line = line_num

                state = self._detect_multiline_import_pattern(stripped, state)
                continue

            # Check if we should stop at decorator
            if self._should_stop_at_decorator(stripped, start_line):
                break

            # Check if we should stop at code
            if self._should_stop_at_code(stripped, start_line, state):
                break

        return self._build_import_section_result(start_line, end_line, import_lines)

    def _should_stop_at_code(
        self, stripped: str, start_line: Optional[int], state: Dict[str, bool]
    ) -> bool:
        """
        Check if code line indicates end of import section.

        Args:
            stripped: Stripped line content
            start_line: Current start line (None if not found yet)
            state: Dictionary with import state flags

        Returns:
            True if we should stop at this code line
        """
        # If we're in a multi-line import, don't stop
        if state["in_multiline_import"]:
            return False

        # If we haven't found any imports yet, don't stop
        if start_line is None:
            return False

        # If line is empty, don't stop (allow blank lines between imports)
        if not stripped:
            return False

        # Check if it's a special case like __all__ or __version__
        if self._is_module_level_dunder(stripped):
            # Module-level dunder variables are OK after imports
            return False

        # If we get here, it's actual code - stop collecting imports
        return True

    def _is_module_level_dunder(self, stripped: str) -> bool:
        """
        Check if line is a module-level dunder variable.

        Args:
            stripped: Stripped line content

        Returns:
            True if line starts with __all__, __version__, etc.
        """
        return stripped.startswith(("__all__", "__version__", "__author__", "__"))

    def _should_stop_at_decorator(
        self, stripped: str, start_line: Optional[int]
    ) -> bool:
        """
        Check if decorator indicates end of import section.

        Args:
            stripped: Stripped line content
            start_line: Current start line (None if not found yet)

        Returns:
            True if we should stop at this decorator
        """
        if stripped.startswith("@"):
            # Decorators indicate we're past imports
            return start_line is not None
        return False

    def _detect_multiline_import_pattern(
        self, stripped: str, state: Dict[str, bool]
    ) -> Dict[str, bool]:
        """
        Detect and set flags for multi-line import patterns.

        Args:
            stripped: Stripped line content
            state: Dictionary with import state flags to update
        """
        if "(" in stripped and ")" not in stripped:
            # Parentheses-based multi-line import
            state["in_parentheses_import"] = True
            state["in_multiline_import"] = True
        elif stripped.rstrip().endswith("\\"):
            # Backslash continuation
            state["in_backslash_continuation"] = True
            state["in_multiline_import"] = True
        else:
            # Single line import, reset flags
            # Single line import, reset flags
            state["in_multiline_import"] = False
            state["in_parentheses_import"] = False
            state["in_backslash_continuation"] = False
        return state

    def _is_import_statement_start(self, stripped: str) -> bool:
        """
        Check if line starts an import statement.

        Args:
            stripped: Stripped line content

        Returns:
            True if line starts with 'from ' or 'import '
        """
        return stripped.startswith("from ") or stripped.startswith("import ")

    def _build_import_section_result(
        self,
        start_line: Optional[int],
        end_line: Optional[int],
        import_lines: List[str],
    ) -> Dict[str, Any]:
        """
        Build the final import section result dictionary.

        Args:
            start_line: First line of import section (1-indexed)
            end_line: Last line of import section (1-indexed)
            import_lines: List of import line contents

        Returns:
            Dictionary with start_line, end_line, content, has_imports
        """
        if start_line is None:
            return {"start_line": 0, "end_line": 0, "content": "", "has_imports": False}

        return {
            "start_line": start_line,
            "end_line": end_line or start_line,
            "content": "".join(import_lines),
            "has_imports": True,
        }

    def _extract_class_name_from_file(
        self, file_lines: List[str], target_line_idx: int
    ) -> Optional[str]:
        """
        Extract the class name that contains the target line.

        Args:
            file_lines: All lines in the file
            target_line_idx: Line index (0-based) to find the containing class

        Returns:
            Class name if found, None otherwise
        """
        # Search backwards from target line to find class definition
        for i in range(target_line_idx, -1, -1):
            line = file_lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            # Check for class definition
            match = re.search(r"class\s+(\w+)\s*[:(]", line)
            if match:
                class_name = match.group(1)

                # Verify target line is actually inside this class
                if self._is_line_inside_class(file_lines, target_line_idx, i):
                    return class_name

        return None

    def _is_line_inside_class(
        self, file_lines: List[str], target_idx: int, class_start_idx: int
    ) -> bool:
        """
        Check if target line is inside the class starting at class_start_idx.

        Args:
            file_lines: All lines in the file
            target_idx: Target line index (0-based)
            class_start_idx: Class definition line index (0-based)

        Returns:
            True if target is inside the class
        """
        if target_idx <= class_start_idx:
            return False

        class_line = file_lines[class_start_idx]
        class_indent = len(class_line) - len(class_line.lstrip())

        # Check lines between class definition and target
        for i in range(class_start_idx + 1, target_idx + 1):
            line = file_lines[i]

            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith("#"):
                continue

            current_indent = len(line) - len(line.lstrip())

            # If we find a line with same or less indentation as class
            # and it's not our target line, target is outside class
            if current_indent <= class_indent and i != target_idx:
                # Check if this is another class definition
                if re.search(r"class\s+\w+\s*[:(]", line):
                    return False

        # Target line should have greater indentation than class
        target_line = file_lines[target_idx]
        if not target_line.strip():
            return True

        target_indent = len(target_line) - len(target_line.lstrip())
        return target_indent > class_indent


class ContextExtractor:
    """Handles context extraction logic"""

    def __init__(self, lines: List[str]):
        self.lines = lines

    def _find_functions_containing_problems(
        self, problem_indices: List[int], range_start: int, range_end: int
    ) -> List[Dict[str, Any]]:
        """
        Find all functions that contain at least one problem line.

        This is the KEY optimization - we only process functions with actual issues.
        """
        functions_with_issues = []
        processed_function_starts = set()  # Avoid duplicates

        for problem_idx in problem_indices:
            # Skip if problem line is outside the specified range
            if problem_idx < range_start or problem_idx > range_end:
                logger.debug(f"Skipping problem line {problem_idx + 1} - outside range")
                continue

            # Find which function contains this problem line
            containing_func_idx = self._find_containing_function(problem_idx)

            if containing_func_idx is None:
                logger.warning(
                    f"Problem line {problem_idx + 1} is not inside any function"
                )
                continue

            # Skip if we already processed this function
            if containing_func_idx in processed_function_starts:
                continue

            # Get function info
            function_info = self._get_function_info(containing_func_idx)
            if not function_info:
                logger.error(
                    f"Failed to get function info for line {containing_func_idx + 1}"
                )
                continue

            # Track which problem lines are in this function
            function_problem_lines = [
                idx + 1  # Convert back to 1-based
                for idx in problem_indices
                if function_info["start_idx"] <= idx <= function_info["end_idx"]
            ]

            function_info["problem_lines_in_function"] = function_problem_lines
            functions_with_issues.append(function_info)
            processed_function_starts.add(containing_func_idx)

        # Sort by start line to maintain file order
        functions_with_issues.sort(key=lambda f: int(f["start_idx"]))

        logger.info(
            f"Found {len(functions_with_issues)} functions containing "
            f"{len(problem_indices)} problem lines"
        )

        return functions_with_issues

    def _build_optimized_context(
        self, functions: List[Dict[str, Any]], problem_lines: List[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Build the new context format with individual function entries.

        Returns both:
        - new_context: List of individual function contexts
        - context: Combined context (for backward compatibility)
        """
        if not functions:
            return None

        new_context = []
        combined_lines = []
        function_metadata = []

        for func in functions:
            # Build individual function context entry
            function_context = {
                "context": "".join(func["lines"]),
                "line_number": func["start_idx"] + 1,
                "start_line": func["start_idx"] + 1,
                "end_line": func["end_idx"] + 1,
                "is_complete_function": True,
                "function_name": func["name"],
                "problem_lines_in_function": func.get("problem_lines_in_function", []),
                "line_count": len(func["lines"]),
            }
            new_context.append(function_context)

            # Add to combined context
            combined_lines.extend(func["lines"])

            # Build metadata
            function_metadata.append(
                {
                    "name": func["name"],
                    "start_line": func["start_idx"] + 1,
                    "end_line": func["end_idx"] + 1,
                    "problem_lines": func.get("problem_lines_in_function", []),
                    "line_count": len(func["lines"]),
                }
            )

        overall_start = functions[0]["start_idx"]
        overall_end = functions[-1]["end_idx"]

        return {
            # NEW: Individual function contexts
            "new_context": new_context,
            # LEGACY: Combined context for backward compatibility
            "context": "".join(combined_lines),
            "start_line": overall_start + 1,
            "end_line": overall_end + 1,
            "is_multi_function": len(functions) > 1,
            "function_count": len(functions),
            "functions": function_metadata,
            "is_complete_function": True,
            # METADATA: Additional tracking
            "total_problem_lines": len(problem_lines),
            "functions_with_issues": len(functions),
            "coverage_percentage": (
                round(
                    (
                        len(functions)
                        / self._count_functions_in_range(overall_start, overall_end)
                    )
                    * 100,
                    2,
                )
                if self._count_functions_in_range(overall_start, overall_end) > 0
                else 100
            ),
        }

    def _count_functions_in_range(self, start_idx: int, end_idx: int) -> int:
        """Helper to count total functions in range for metrics."""
        count = 0
        for line_idx in range(start_idx, end_idx + 1):
            if line_idx >= len(self.lines):
                break
            if self._is_function_definition(self.lines[line_idx].rstrip()):
                count += 1
        return count

    def _extract_all_functions_in_range(
        self,
        first_idx: int,
        last_idx: int,
        problem_lines: List[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract ONLY functions that contain problem lines.

        Example:
            Range: 400-1700
            Problem lines: [402, 600, 1200, 1250, 1600]

            If functions are:
            - func_a: lines 380-450 (contains 402) ✓ INCLUDE
            - func_b: lines 500-650 (contains 600) ✓ INCLUDE
            - func_c: lines 700-800 (no issues) ✗ EXCLUDE
            - func_d: lines 1150-1300 (contains 1200, 1250) ✓ INCLUDE
            - func_e: lines 1550-1650 (contains 1600) ✓ INCLUDE

        Returns:
            {
                "new_context": [
                    {
                        "context": <func_a_code>,
                        "line_number": 380,
                        "start_line": 380,
                        "end_line": 450,
                        "is_complete_function": True,
                        "function_name": "func_a",
                        "problem_lines_in_function": [402]
                    },
                    # ... func_b, func_d, func_e
                ],
                "context": <combined_code_of_all_functions>,
                "start_line": 380,  # First function start
                "end_line": 1650,   # Last function end
                "is_multi_function": True,
                "function_count": 4,
                "functions": [<metadata>],
                "is_complete_function": True,
                "total_problem_lines": 5,
                "functions_with_issues": 4
            }
        """
        # Convert problem_lines to 0-based indices for internal consistency
        problem_indices = [line - 1 for line in problem_lines]
        # Step 1: Find all functions that contain at least one problem line
        functions_with_issues = self._find_functions_containing_problems(
            problem_indices, first_idx, last_idx
        )

        if not functions_with_issues:
            logger.warning(
                f"No functions found containing problem lines {problem_lines} "
                f"in range {first_idx + 1}-{last_idx + 1}"
            )
            return None

        # Step 2: Build the response structure
        return self._build_optimized_context(functions_with_issues, problem_lines)

    def _get_function_info(self, func_start_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get complete information about a function starting at given index.

        Returns:
            Dictionary with:
            - start_idx: Function start (0-based)
            - end_idx: Function end (0-based)
            - name: Function name
            - lines: List of code lines
            - indentation: Base indentation level
        """

        if func_start_idx >= len(self.lines):
            return None

        func_line = self.lines[func_start_idx].rstrip()
        if not self._is_function_definition(func_line):
            return None

        # Extract function name
        func_name = self._extract_function_name(func_line)

        decorator_start_idx = self._find_decorator_start(func_start_idx)

        # Find function end
        func_end_idx = self._find_function_end(func_start_idx)
        if func_end_idx is None:
            return None

        # Get base indentation
        base_indent = len(func_line) - len(func_line.lstrip())

        decorators = []
        if decorator_start_idx < func_start_idx:
            decorators = self.lines[decorator_start_idx:func_start_idx]

        return {
            "start_idx": decorator_start_idx,
            "end_idx": func_end_idx,
            "name": func_name,
            "lines": self.lines[decorator_start_idx : func_end_idx + 1],
            "indentation": base_indent,
            "decorators": decorators,
        }

    def _find_decorator_start(self, func_def_idx: int) -> int:
        """
        Find the starting index of decorators above a function definition.

        Args:
            func_def_idx: Index of the function definition line

        Returns:
            Index of first decorator, or func_def_idx if no decorators found
        """
        if func_def_idx == 0:
            return func_def_idx

        # Get function's indentation level
        func_line = self.lines[func_def_idx].rstrip()
        func_indent = len(func_line) - len(func_line.lstrip())

        # Walk backwards to find decorators
        current_idx = func_def_idx - 1
        decorator_start = func_def_idx

        while current_idx >= 0:
            line = self.lines[current_idx].rstrip()

            # Skip empty lines and comments immediately above function
            if not line or line.lstrip().startswith("#"):
                current_idx -= 1
                continue

            # Check if line is a decorator
            stripped = line.lstrip()
            line_indent = len(line) - len(stripped)

            # Decorator must:
            # 1. Start with @
            # 2. Have same indentation as function
            if stripped.startswith("@") and line_indent == func_indent:
                decorator_start = current_idx
                current_idx -= 1
            else:
                # Not a decorator, stop searching
                break

        return decorator_start

    def _is_line_inside_function(
        self, lines: List[str], line_idx: int, function_start_idx: int
    ) -> bool:
        """Check if a line is inside the function starting at function_start_idx."""
        if line_idx < function_start_idx:
            return False

        function_end_idx = self._find_function_end(function_start_idx)
        if function_end_idx is None:
            # If we can't find the end, use heuristic based on indentation
            return self._check_indentation_containment(
                lines, line_idx, function_start_idx
            )

        return line_idx <= function_end_idx

    def _find_brace_function_end(
        self, lines: List[str], start_idx: int
    ) -> Optional[int]:
        """
        Find end of function using brace matching (JavaScript, Java, C#, etc.).
        """
        if start_idx >= len(lines):
            return None

        brace_count = 0
        found_opening_brace = False

        # Start from the function definition line
        for i in range(start_idx, len(lines)):
            line = lines[i]

            # Count braces, ignoring those in strings and comments
            clean_line = self._remove_strings_and_comments(line)

            for char in clean_line:
                if char == "{":
                    brace_count += 1
                    found_opening_brace = True
                elif char == "}":
                    brace_count -= 1

                    # When we close all braces, we've found the end
                    if found_opening_brace and brace_count == 0:
                        return i

        return None

    def _remove_strings_and_comments(self, line: str) -> str:
        """
        Remove string literals and comments to avoid counting braces inside them.
        """

        # Remove single-line comments
        line = re.sub(r"//.*$", "", line)  # // comments
        line = re.sub(r"#.*$", "", line)  # # comments

        # Remove string literals (simplified)
        line = re.sub(r'"[^"]*"', '""', line)  # Double quotes
        line = re.sub(r"'[^']*'", "''", line)  # Single quotes
        line = re.sub(r"`[^`]*`", "``", line)  # Backticks

        return line

    def _check_indentation_containment(
        self, lines: List[str], line_idx: int, function_start_idx: int
    ) -> bool:
        """Fallback method to check if a line is inside a function using indentation."""
        if function_start_idx >= len(lines) or line_idx >= len(lines):
            return False

        func_line = lines[function_start_idx]
        func_indent = len(func_line) - len(func_line.lstrip())

        target_line = lines[line_idx]
        if not target_line.strip():  # Empty line
            return True

        target_indent = len(target_line) - len(target_line.lstrip())

        # If target line has greater indentation, it's likely inside the function
        return target_indent > func_indent

    def _find_python_function_end(
        self, lines: List[str], start_idx: int
    ) -> Optional[int]:
        if start_idx >= len(lines):
            return None
        function_line = lines[start_idx]
        function_indent = len(function_line) - len(function_line.lstrip())
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip() or line.strip().startswith("#"):
                continue
            current_indent = len(line) - len(line.lstrip())
            if self._has_reached_function_end(
                current_indent, function_indent, line, lines[i - 1]
            ):
                return i - 1
        return len(lines) - 1

    def _has_reached_function_end(
        self, current_indent: int, function_indent: int, line: str, prev_line: str
    ) -> bool:
        if (current_indent <= function_indent) and not (
            prev_line.rstrip().endswith(("\\", ",", "("))
            or line.strip().startswith((")", "def ", "async def"))
        ):
            return True
        if current_indent == function_indent and self._is_function_definition(line):
            return True
        return False

    def _is_function_definition(self, line: str) -> bool:
        """
        Check if a line contains a function or method definition.

        Supports Python, JavaScript/TypeScript, Java, and C# function patterns.
        """

        stripped_line = line.strip()

        # Python function/method patterns
        python_patterns = [
            r"^def\s+\w+\s*\(",  # def function_name(
            r"^async\s+def\s+\w+\s*\(",  # async def function_name(
            r"^\s*def\s+\w+\s*\(",  # indented def (method in class)
            r"^\s*async\s+def\s+\w+\s*\(",  # indented async def
            r"^\s*async\s+def\s+\w+\s*\(",  # indented async def
        ]

        # JavaScript/TypeScript function patterns
        js_patterns = [
            r"^function\s+\w+\s*\(",  # function functionName(
            r"^async\s+function\s+\w+\s*\(",  # async function functionName(
            r"^\w+\s*:\s*function\s*\(",  # methodName: function(
            r"^\w+\s*:\s*async\s+function\s*\(",  # methodName: async function(
            r"^\w+\s*=\s*function\s*\(",  # functionName = function(
            r"^\w+\s*=\s*async\s+function\s*\(",  # functionName = async function(
            r"^\w+\s*=\s*\([^)]*\)\s*=>\s*\{",  # functionName = (params) => {
            r"^\w+\s*=\s*async\s*\([^)]*\)\s*=>\s*\{",  # functionName = async (params) => {
            r"^\s*\w+\s*\([^)]*\)\s*\{",  # methodName() { (in class/object)
            r"^\s*async\s+\w+\s*\([^)]*\)\s*\{",  # async methodName() {
            r"^(?:const|let|var)\s+\w+\s*=\s*\([^)]*\)\s*=>\s*\{",
            r"^(?:const|let|var)\s+\w+\s*=\s*async\s*\([^)]*\)\s*=>\s*\{",
        ]

        # Java/C# method patterns
        java_csharp_patterns = [
            r"^(public|private|protected|static|internal|\s)+(.*\s+)?\w+\s*\([^)]*\)\s*\{",
            r"^\s*(public|private|protected|static|internal|\s)+(.*\s+)?\w+\s*\([^)]*\)\s*\{",
        ]

        all_patterns = python_patterns + js_patterns + java_csharp_patterns

        for pattern in all_patterns:
            if re.match(pattern, stripped_line, re.IGNORECASE):
                return True

        return False

    def _is_function_line(self, line_idx: int) -> bool:
        """Check if line is a function definition"""
        if line_idx >= len(self.lines):
            return False
        return self._is_function_definition(self.lines[line_idx])

    def _find_containing_function(self, target_line_idx: int) -> Optional[int]:
        """Find the function that contains the given line index."""
        # Search backwards from the target line to find a function definition
        lines = self.lines
        for i in range(target_line_idx, -1, -1):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Check if this line is a function definition
            if self._is_function_definition(lines[i]):
                # Verify that the target line is actually within this function
                if self._is_line_inside_function(lines, target_line_idx, i):
                    return i

        return None

    def _find_function_end(self, start_idx: int) -> Optional[int]:
        """
        Find the end line of a function/method based on language-specific rules.
        """
        lines = self.lines
        if start_idx >= len(lines):
            return None

        start_line = lines[start_idx]

        # Determine language and strategy based on the function definition
        if re.search(r"\bdef\b", start_line):
            # Python function - use indentation
            return self._find_python_function_end(lines, start_idx)
        elif "{" in start_line or (
            re.search(r"\bfunction\b", start_line)
            and re.search(r"[a-zA-Z_]\w*\s*\([\w,\s]*\)\s*\{", start_line)
        ):
            # JavaScript/Java/C# - use brace matching
            return self._find_brace_function_end(lines, start_idx)
        elif re.search(r"\b@click.\b", start_line):
            return self._find_python_function_end(lines, start_idx)
        else:
            # Try both strategies
            python_end = self._find_python_function_end(lines, start_idx)
            if python_end is not None:
                return python_end
            return self._find_brace_function_end(lines, start_idx)

    def _extract_function_name(self, function_line: str) -> str:
        """
        Extract function name from function definition line.
        """

        # Python function name extraction
        python_match = re.search(r"def\s+(\w+)", function_line)
        if python_match:
            return python_match.group(1)

        # JavaScript function name extraction
        js_match = re.search(r"function\s+(\w+)", function_line)
        if js_match:
            return js_match.group(1)

        # Method assignment patterns
        assignment_pattern = r"([a-zA-Z_][a-zA-Z0-9_]{0,99})\s{0,20}[:=]"
        assignment_match = re.search(assignment_pattern, function_line)
        if assignment_match:
            return assignment_match.group(1)

        # Java/C# method name extraction
        method_match = re.search(r"\b(\w+)\s*\([^)]*\)\s*\{?", function_line)
        if method_match:
            name = method_match.group(1)
            # Filter out keywords
            keywords = {
                "public",
                "private",
                "protected",
                "static",
                "async",
                "void",
                "int",
                "string",
                "bool",
                "return",
            }
            if name.lower() not in keywords:
                return name

        return ""

    def _is_actual_function_def(self, line: str) -> bool:
        """
        Check if a line contains an actual function definition (not a decorator).

        Returns:
            bool: True if the line is a function definition, False otherwise
        """
        stripped_line = line.strip()

        # Python function/method patterns only
        python_patterns = [
            r"^def\s+\w+\s*\(",  # def function_name(
            r"^async\s+def\s+\w+\s*\(",  # async def function_name(
        ]

        for pattern in python_patterns:
            if re.match(pattern, stripped_line):
                return True

        return False

    def _extract_complete_function(
        self, start_idx: int, problem_line_idx: int, last_line_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract complete function/method from start to end, including decorators.

        Args:
            lines: All lines in the file
            start_idx: Starting index (0-indexed) of function definition or where issue was detected

        Returns:
            Context dictionary with complete function or None if not found
        """
        lines = self.lines
        if start_idx >= len(lines):
            return None

        # Find the actual function definition line (in case start_idx is on a decorator)
        function_def_line_idx = start_idx

        # If we're starting on a decorator, find the actual function definition
        if self._is_decorator(lines[start_idx].strip()):
            for i in range(
                start_idx, min(len(lines), start_idx + 10)
            ):  # Look ahead max 10 lines
                if self._is_actual_function_def(lines[i]):
                    function_def_line_idx = i
                    break

        # Find the start including decorators
        function_start = self._find_function_start_with_decorators(
            lines, function_def_line_idx
        )

        # Find the end of the function
        function_end = self._find_function_end(function_def_line_idx)

        if function_end is None:
            # If we can't find the end, include reasonable context
            function_end = min(
                len(lines) - 1, function_def_line_idx + 50
            )  # Max 50 lines

        # Extract function lines (including decorators)
        function_lines = lines[function_start : last_line_idx + 1]
        if function_end > last_line_idx:
            function_lines = lines[function_start : function_end + 1]
        context_text = "".join(function_lines)

        # Get the actual function definition line
        function_def_line = lines[function_def_line_idx].rstrip()

        # Extract decorators
        decorators = []
        for i in range(function_start, function_def_line_idx):
            decorator_line = lines[i].strip()
            if decorator_line:  # Skip empty lines
                decorators.append(decorator_line)
        problem_line = lines[problem_line_idx].rstrip()

        return {
            "context": context_text,
            "function_definition_line": function_def_line,
            "decorators": decorators,
            "line_number": start_idx
            + 1,  # Convert back to 1-indexed (original issue line)
            "function_start_line": function_start + 1,  # Convert back to 1-indexed
            "end_line": function_end + 1,  # Convert back to 1-indexed
            "is_complete_function": True,
            "function_name": self._extract_function_name(function_def_line),
            "has_decorators": len(decorators) > 0,
            "decorator_count": len(decorators),
            "start_line": function_start + 1,
            "problem_lines": [problem_line],
        }

    def _find_function_start_with_decorators(
        self, lines: List[str], target_line_idx: int
    ) -> int:
        """
        Find the actual start of a function, including Any decorators.

        Args:
            lines: All lines in the file
            target_line_idx: Line index where the issue was detected (0-indexed)

        Returns:
            int: The line index where the function actually starts (including decorators)
        """
        # Start from the target line and work backwards to find decorators
        start_idx = target_line_idx

        # Look backwards for decorators
        for i in range(target_line_idx - 1, -1, -1):
            line = lines[i].strip()

            # Stop if we hit an empty line or non-decorator line
            if not line:
                break

            if self._is_decorator(line):
                start_idx = i
            else:
                # If it's not a decorator and not empty, we've found the boundary
                break

        return start_idx

    def _is_decorator(self, line: str) -> bool:
        """
        Check if a line is a Python decorator.

        Returns:
            bool: True if the line is a decorator, False otherwise
        """
        stripped_line = line.strip()

        # Python decorator patterns
        decorator_patterns = [
            r"^@\w+$",  # @decorator
            r"^@\w+\.[^(]*$",  # @module.decorator
            r"^@\w+\(",  # @decorator(args)
            r"^@\w+\.[^(]*\(",  # @module.decorator(args)
        ]

        for pattern in decorator_patterns:
            if re.match(pattern, stripped_line):
                return True

        return False

    def _try_function_extraction_strategies(
        self,
        first_idx: int,
        last_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Try various strategies to extract complete function context.

        Strategies (in order of priority):
        1. Issue is ON a function definition line
        2. Issue is INSIDE a function
        3. Multi-line issue that SPANS a function

        Returns:
            Context dict if successful, None if no function context found
        """

        # Strategy 1: Issue is inside a function
        context = self._try_extract_containing_function(first_idx, last_idx)
        if context:
            return context

        # Strategy 3: Multi-line issue - try to expand to complete function
        if first_idx != last_idx:
            context = self._try_expand_multiline_to_function(first_idx, last_idx)
            if context:
                return context

        return None

    def _try_expand_multiline_to_function(
        self,
        first_idx: int,
        last_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        For multi-line issues, try to expand the range to include complete function.

        Strategy:
        1. Find function containing the LAST line of the issue
        2. If found and it extends beyond issue range, use the complete function

        Args:
            extractor: Context extractor instance
            line_range: Range of problematic lines (multi-line)

        Returns:
            Expanded function context if applicable, None otherwise
        """
        # Find function containing the end of the issue
        function_start_idx = self._find_containing_function(last_idx)

        if function_start_idx is None:
            return None

        # Find where this function ends
        function_end_idx = self._find_function_end(function_start_idx)

        if function_end_idx is None:
            return None

        # Only use this if function extends beyond our current range
        if function_end_idx <= last_idx:
            return None

        # Extract the complete function
        return self._extract_complete_function(function_start_idx, first_idx, last_idx)

    def _try_extract_containing_function(
        self, first_idx: int, last_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Try to extract the complete function that contains the issue.

        Returns:
            Function context if issue is inside a function, None otherwise
        """
        function_start_idx = self._find_containing_function(first_idx)

        if function_start_idx is None:
            return None

        return self._extract_complete_function(function_start_idx, first_idx, last_idx)

    def _extract_normal_context(
        self, first_idx: int, last_idx: int, context_lines: int
    ) -> Dict[str, Any]:
        """Extract normal context window"""
        start_idx = max(0, first_idx - context_lines)
        end_idx = min(len(self.lines), last_idx + context_lines + 1)

        return {
            "new_context": [
                {
                    "context": "".join(self.lines[start_idx:end_idx]),
                    "line_number": first_idx + 1,
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                    "is_complete_function": False,
                }
            ],
            # "context": "".join(self.lines[start_idx:end_idx]),
            "line_number": first_idx + 1,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "is_complete_function": False,
        }

    def _get_empty_context(self, line_number: int) -> Dict[str, Any]:
        """
        Return empty context when line number is out of range.
        """
        return {
            "new_context": [
                {
                    "context": "",
                    "line_number": line_number,
                    "start_line": line_number,
                    "end_line": line_number,
                    "is_complete_function": False,
                }
            ],
            # "context": "",
            "line_number": line_number,
            "start_line": line_number,
            "end_line": line_number,
            "is_complete_function": False,
        }


def _build_fix_suggestion(
    fix_response: SonarFixResponse,
    context_info: Dict[str, Any],
    file_path: Path,
    project_path: Path,
    line_range: Dict[str, int],
    model_name: Optional[str],
) -> FixSuggestion:
    """
    Build a FixSuggestion object from LLM response and context.

    Args:
        fix_response: Response from LLM containing fix details
        context_info: Context information including context_dict
        file_path: Absolute path to the file
        project_path: Project root path
        line_range: Line range information
        model_name: Name of the LLM model used

    Returns:
        FixSuggestion object
    """
    context_dict = context_info["context_dict"]

    # Use actual end line from context if available, otherwise from line_range
    end_line = context_dict.get("end_line", line_range["last_line"])

    problem_lines_raw: Any = line_range.get("problem_lines", [])

    # Normalize to List[int]
    if isinstance(problem_lines_raw, list):
        problem_lines: List[int] = problem_lines_raw
    elif isinstance(problem_lines_raw, int):
        problem_lines = [problem_lines_raw]
    else:
        problem_lines = []

    return FixSuggestion(
        issue_key=_generate_fix_key(problem_lines),
        original_code=context_dict["context"],
        fixed_code_blocks=fix_response.FIXED_CODE_BLOCKS,
        fixed_code="",
        import_block_code=fix_response.IMPORT_BLOCK,
        helper_code=fix_response.NEW_HELPER_CODE,
        placement_helper=fix_response.PLACEMENT.value,
        explanation=fix_response.EXPLANATION or "",
        confidence=fix_response.CONFIDENCE,
        llm_model=model_name or "unknown",
        rule_description="",
        file_path=str(file_path.relative_to(project_path)),
        line_number=context_dict.get("start_line"),
        sonar_line_number=line_range["first_line"],
        end_import_block_code=context_dict.get("import_section", {}).get("end_line", 0),
        last_line_number=end_line,
    )


def _generate_fix_key(problem_lines: List[int]) -> str:
    """
    Generate a meaningful key for the fix based on problem lines.

    Args:
        problem_lines: List of problem line numbers

    Returns:
        String key like "fix_L10" or "fix_L10-L15"
    """
    if not problem_lines:
        return "fix_unknown"

    if len(problem_lines) == 1:
        return f"fix_L{problem_lines[0]}"

    min_line = min(problem_lines)
    max_line = max(problem_lines)
    return f"fix_L{min_line}-L{max_line}"


def _extract_problem_lines(
    file_lines: List[str], problem_line_numbers: List[int]
) -> Dict[int, str]:
    """
    Extract actual line content for problem lines.

    Args:
        file_lines: All lines in the file
        problem_line_numbers: List of 1-based line numbers to extract


    Note:
        Handles off-by-one conversion from SonarCloud's 1-based line numbers
        to Python's 0-based list indexing.
    """
    problem_lines = {}

    for line_number in problem_line_numbers:
        # Convert 1-based line number to 0-based index
        line_index = line_number - 1

        # Bounds check
        if 0 <= line_index < len(file_lines):
            problem_lines[line_number] = file_lines[line_index]

        else:
            logger.warning(
                f"Line number {line_number} out of bounds "
                f"(file has {len(file_lines)} lines)"
            )

    return problem_lines
