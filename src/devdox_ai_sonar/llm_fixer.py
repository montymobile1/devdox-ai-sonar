"""LLM-powered code fixer for SonarCloud issues."""

import os
import re
import sys
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
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
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
    FixGraphState,
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

# ---------------------------------------------------------------------------
# All LLM calls go through LangChain — no raw SDK imports here.
# build_llm() lazily imports the correct langchain package per provider.
# LLMFixer (fixer) and FixValidator (validator) are fully independent:
#   fixer     = LLMFixer(provider="togetherai", model="llama-3-70b", ...)
#   validator = FixValidator(provider="openai",   model="gpt-4o",     ...)
# ---------------------------------------------------------------------------

java_extension = ".java"
scala_extension = ".scala"


FUNCTION_ALREADY_CALLED = (
    "• As function can be already called so don't remove the parameters."
)
STATICMETHOD_DECORATOR = "@staticmethod"
PYTHON_CODE_BLOCK = "```python"
CODE_BLOCK_END = "```"

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: Dict[str, str] = {
    "openai":     "gpt-4o",
    "togetherai": "meta-llama/Llama-3-70b-chat-hf",
    "openrouter": "anthropic/claude-sonnet-4",
    "gemini":     "gemini-1.5-flash",
}

_PROVIDER_ENV_KEYS: Dict[str, str] = {
    "openai":     "OPENAI_API_KEY",
    "togetherai": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini":     "GEMINI_KEY",
}


def build_llm(
    provider: str,
    model: str,
    api_key: str,
    max_tokens: int = 8000,
    temperature: float = 0.08,
) -> BaseChatModel:
    """
    Build a LangChain chat model for the given provider.

    All four providers use their dedicated LangChain package:
      - openai     → langchain_openai.ChatOpenAI
      - togetherai → langchain_together.ChatTogether
      - openrouter → langchain_openai.ChatOpenAI  (OpenAI-compatible endpoint)
      - gemini     → langchain_google_genai.ChatGoogleGenerativeAI

    No raw SDK clients (openai.OpenAI, Together, genai.Client) are used.
    """
    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    elif provider == "togetherai":
        from langchain_together import ChatTogether
        return ChatTogether(
            model=model,
            together_api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=max_tokens,
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "https://devdox.ai",
                "X-Title": "DevDox AI Sonar",
            },
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Use 'openai', 'togetherai', 'openrouter', or 'gemini'."
        )


def call_llm_with_graph(
    provider: str,
    model: str,
    api_key: str,
    prompt_system: str,
    prompt: str,
    max_tokens: int = 8000,
    temperature: float = 0.08,
    max_attempts: int = 3,
) -> Optional[Any]:
    """
    Run the LangGraph fix pipeline for any provider.

    Returns:
        SonarFixResponse on success, None on failure.
    """
    initial_state: FixGraphState = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "prompt_system": prompt_system,
        "original_prompt": prompt,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_attempts": max_attempts,
        "fix_result": None,
        "validation_error": None,
        "attempt": 0,
    }

    final_state = _FIX_GRAPH.invoke(initial_state)

    if final_state["fix_result"] is None:
        logger.error(
            "LLM pipeline failed after %d attempts [provider=%s]. Last error: %s",
            max_attempts, provider, final_state.get("validation_error"),
        )
        return None

    return final_state["fix_result"]


def fix_node(state: FixGraphState) -> FixGraphState:
    """Call the LLM with structured output via the appropriate provider."""
    from devdox_ai_sonar.models.sonar import SonarFixResponse  # local import to avoid circular

    try:
        llm = build_llm(
            provider=state["provider"],
            model=state["model"],
            api_key=state["api_key"],
            max_tokens=state["max_tokens"],
            temperature=state["temperature"],
        )

        structured_llm = llm.with_structured_output(SonarFixResponse)

        chain = ChatPromptTemplate.from_messages([
            ("system", "{prompt_system}"),
            ("human", "{prompt}"),
        ]) | structured_llm

        result: SonarFixResponse = chain.invoke({
            "prompt_system": state["prompt_system"],
            "prompt": state["prompt"],
        })

        return {**state, "fix_result": result, "validation_error": None}

    except Exception as e:
        logger.error("[fix_node] %s call failed: %s", state["provider"], e, exc_info=True)
        return {**state, "fix_result": None, "validation_error": str(e)}


def validate_node(state: FixGraphState) -> FixGraphState:
    """Validate that the structured output is usable."""
    result = state.get("fix_result")

    if result is None:
        return {**state, "validation_error": "No result produced by fix node"}

    errors = []

    if not result.FIXED_CODE_BLOCKS:
        errors.append("FIXED_CODE_BLOCKS is empty")

    if not getattr(result, "EXPLANATION", "").strip():
        errors.append("EXPLANATION is empty")

    confidence = getattr(result, "CONFIDENCE", 0.0)
    if confidence < 0.3:
        errors.append(f"CONFIDENCE too low: {confidence}")

    if errors:
        error_msg = f"Validation failed: {'; '.join(errors)}"
        logger.warning("[validate_node] %s", error_msg)
        return {**state, "validation_error": error_msg, "fix_result": None}

    return {**state, "validation_error": None}


def retry_node(state: FixGraphState) -> FixGraphState:
    """Increment attempt counter and enrich the prompt with the previous error."""
    attempt = state["attempt"] + 1
    enriched_prompt = (
        f"{state['original_prompt']}\n\n"
        f"[Attempt {attempt} correction: previous response failed validation — "
        f"{state['validation_error']}. "
        f"Ensure FIXED_CODE_BLOCKS is populated, EXPLANATION is clear, "
        f"and CONFIDENCE >= 0.3. Return valid JSON only.]"
    )
    logger.info("[retry_node] Retrying — attempt %d", attempt)
    return {
        **state,
        "attempt": attempt,
        "prompt": enriched_prompt,
        "validation_error": None,
    }


def should_retry(state: FixGraphState) -> str:
    if state.get("validation_error") is None:
        return "done"
    if state["attempt"] >= state["max_attempts"]:
        logger.warning(
            "[should_retry] Max attempts (%d) reached — giving up",
            state["max_attempts"],
        )
        return "give_up"
    return "retry"


def build_fix_graph():
    graph = StateGraph(FixGraphState)
    graph.add_node("fix", fix_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.set_entry_point("fix")
    graph.add_edge("fix", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {"done": END, "give_up": END, "retry": "retry"},
    )
    graph.add_edge("retry", "fix")
    return graph.compile()


_FIX_GRAPH = build_fix_graph()


class LLMFixer:
    """LLM-powered code fixer for SonarCloud issues.

    Uses LangChain for all provider calls — no raw SDK clients.
    The fixer and the validator (FixValidator) are independent instances
    and can each use a different provider/model:

        fixer     = LLMFixer(provider="togetherai", model="llama-3-70b")
        validator = FixValidator(provider="openai",   model="gpt-4o")
    """

    def __init__(
        self,
        provider: Optional[str] = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        context_lines: int = 10,
    ):
        self.provider = str(provider).lower()
        self.context_lines = context_lines
        self.file_reader = AsyncFileReader()
        self.prompt_dir = Path(__file__).parent / "prompts"
        self.template_dir = Path(__file__).parent / "templates"
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

    # ------------------------------------------------------------------
    # Provider configuration — LangChain only, no raw SDK clients
    # ------------------------------------------------------------------

    def _validate_and_configure_provider(
        self, provider: Optional[str], model: Optional[str], api_key: Optional[str]
    ) -> None:
        p = str(provider).lower()
        if p not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unsupported provider: {p}. "
                "Use 'openai', 'gemini', 'togetherai', or 'openrouter'."
            )
        self.provider = p
        self.model: str = model or _PROVIDER_DEFAULTS[p]
        resolved_key = api_key or os.getenv(_PROVIDER_ENV_KEYS[p], "")
        if not resolved_key:
            raise ValueError(
                f"API key not provided for provider '{p}'. "
                f"Set the {_PROVIDER_ENV_KEYS[p]} environment variable."
            )
        self.api_key: str = resolved_key

        # Build the LangChain client once and cache it on the instance.
        # This is used by _call_llm_list and LLMFixerAdapter (constant_namer).
        self._llm: BaseChatModel = build_llm(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )

    # Keep the individual _configure_* names so any external callers that
    # reference them directly still work (they now just delegate).
    def _configure_openai(self, model: Optional[str], api_key: Optional[str]) -> None:
        self._validate_and_configure_provider("openai", model, api_key)

    def _configure_togetherai(self, model: Optional[str], api_key: Optional[str]) -> None:
        self._validate_and_configure_provider("togetherai", model, api_key)

    def _configure_gemini(self, model: Optional[str], api_key: Optional[str]) -> None:
        self._validate_and_configure_provider("gemini", model, api_key)

    def _configure_openrouter(self, model: Optional[str], api_key: Optional[str]) -> None:
        self._validate_and_configure_provider("openrouter", model, api_key)

    @staticmethod
    def _convert_regex_to_diff(
        block: CodeBlock, file_cache: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Convert a SearchReplace block to a DIFF with actual source code."""
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
        """Create documentation-friendly display copies of code blocks."""
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
    ) -> Optional[List[FixSuggestion]]:
        """Generate fix suggestions for issues in a single file."""
        if not issues:
            logger.warning("No issues provided for fix generation")
            return None

        rule_registry = RuleHandlerRegistry()
        extractor = IssueExtractor(self.file_reader)

        try:
            validation = await extractor.validate_issue_group(
                issues, tmp_path, project_path, check_tmp_path
            )

            if not validation.is_valid:
                logger.error("Validation failed: %s", validation.error)
                return None

            if validation.file_path is None or validation.line_range is None:
                logger.error("Validation passed but file_path or line_range is None")
                return None

            context = await self._prepare_fix_context(
                validation.file_path,
                validation.line_range,
                modified_content,
            )

            if not context:
                logger.warning("Failed to prepare fix context")
                return None

            handler = rule_registry.get_handler(issues[0].rule)

            fix_response_lst = await handler.generate_fixes(
                issues,
                context,
                project_path,
                validation.file_path,
                llm_caller=self,
            )

            if not fix_response_lst or len(fix_response_lst) == 0:
                logger.warning("Handler returned no fix for rule %s", issues[0].rule)
                return None

            modify_line_range = getattr(handler, "MOIDY_LINE_RANGE", False)
            fix_suggestion_lst = self._build_fix_suggestion(
                fix_response_lst,
                context,
                validation.file_path,
                project_path,
                validation.line_range,
                modify_line_range,
            )

            if file_md:
                self.write_explaination(
                    project_path / file_md,
                    fix_response_lst,
                    issues,
                    context.context_dict,
                    project_path=project_path,
                )

            logger.info("Generated fix for %d issue(s)", len(issues))
            return fix_suggestion_lst

        except FileNotFoundError as e:
            logger.warning("File not found: %s", e)
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

        logger.debug("Built %d fix suggestions", len(lst_suggestion))
        return lst_suggestion

    def _generate_fix_key(self, problem_lines: List[int]) -> str:
        if not problem_lines:
            return "fix_unknown"
        if len(problem_lines) == 1:
            return f"fix_L{problem_lines[0]}"
        return f"fix_L{min(problem_lines)}-L{max(problem_lines)}"

    async def _prepare_fix_context(
        self,
        file_path: Path,
        line_range: Dict[str, Any],
        modified_content: str,
    ) -> Optional[FixContext]:
        """Prepare context for fix generation."""
        try:
            file_lines = await self.file_reader.read_lines(file_path)
            language = self._get_language_from_extension(file_path.suffix)
            import_section = self._extract_import_section(file_lines, language)
            first_problem_line_idx = line_range["first_line"] - 1
            class_name = self._extract_class_name_from_file(
                file_lines, first_problem_line_idx
            )
            problem_line_content = _extract_problem_lines(
                file_lines, line_range["problem_lines"]
            )
            if modified_content:
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
                context_dict = self._extract_context(
                    file_lines,
                    line_range["first_line"],
                    line_range["last_line"],
                    line_range["problem_lines"],
                    self.context_lines,
                )
                context_dict["language"] = language
                if "class_name" not in context_dict or context_dict["class_name"] is None:
                    context_dict["class_name"] = class_name
                context_dict["import_section"] = import_section
            context_dict["problem_line_content"] = problem_line_content

            code_chunk = ""
            for content in context_dict.get("new_context", []):
                code_chunk += content["context"] + "\n"

            return FixContext(
                file_path=file_path,
                file_path_tmp=file_path,
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
        """Apply multiple fixes to the project."""
        result = FixResult(project_path=project_path, total_fixes_attempted=len(fixes))

        if create_backup and not dry_run:
            backup_path = self._create_backup(project_path)
            result.backup_created = True
            result.backup_path = backup_path

        fixes_by_file: Dict[str, List[FixSuggestion]] = {}
        for fix in fixes:
            file_key = self._get_file_from_fix(fix, project_path)
            if file_key:
                if file_key not in fixes_by_file:
                    fixes_by_file[file_key] = []
                fixes_by_file[file_key].append(fix)

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
                    logger.error("Failed to apply fixes to %s", file_path)
            except Exception as e:
                result.failed_fixes.extend(
                    [{"fix": fix, "error": str(e)} for fix in file_fixes]
                )
                logger.error("Error processing file %s: %s", file_path_str, e, exc_info=True)

        return result

    def _extract_context(
        self,
        lines: List[str],
        first_line_number: int,
        last_line_number: int,
        problem_lines: List[int],
        context_lines: int,
    ) -> Dict[str, Any]:
        """Extract context around a problematic line with intelligent function/method detection."""
        extractor = ContextExtractor(lines)
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

        All four providers go through call_llm_with_graph() which uses
        build_llm() → LangChain.  No raw SDK calls remain here.
        """
        if not self.model:
            logger.error("Model not configured")
            return None

        language = self._get_language_from_extension(file_extension)

        logger.debug(
            "LLM call — provider=%s, model=%s, language=%s, issues=%d",
            self.provider,
            self.model,
            language,
            len(issues),
        )

        prompt, system_template = self._create_fix_prompt_list(
            issues, context, rule_info_dict, language, error_message
        )
        prompt_system = system_template.render()

        return call_llm_with_graph(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            prompt_system=prompt_system,
            prompt=prompt,
            max_tokens=8000,
            temperature=0.08,
            max_attempts=3,
        )

    def _is_init_method(self, context: str) -> bool:
        if re.search(r"def\s+__init__\s*\(", context):
            return True
        if re.search(r"public\s+\w+\s*\([^)]*\)\s*\{", context):
            return True
        if re.search(r"constructor\s*\([^)]*\)\s*\{", context):
            return True
        init_patterns = [
            r"self\.\w+\s*=",
            r"this\.\w+\s*=",
            r"_\w+\s*=",
        ]
        return sum(1 for p in init_patterns if re.search(p, context)) >= 3

    def _extract_complexity_info(self, message: str) -> Dict[str, str]:
        pattern = r"complexity from (\d+) to the (\d+) allowed"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return {"current": match.group(1), "target": match.group(2)}
        alt_pattern = r"complexity is (\d+)[^\d]*?maximum[^\d]*?(\d+)"
        alt_match = re.search(alt_pattern, message, re.IGNORECASE)
        if alt_match:
            return {"current": alt_match.group(1), "target": alt_match.group(2)}
        return {"current": "Unknown", "target": "15"}

    def _extend_strategies_for_issue(self, issue: Any) -> List[str]:
        msg_lower = getattr(issue, "message", "").lower()

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
        elif "unused" in getattr(issue, "rule", "").lower() or "unused" in msg_lower:
            strategies_list.append("• Remove ONLY the specific unused variable/import.")
            strategies_list.append("• Do not break code that references adjacent lines.")
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
            strategies_list.append("• Check the syntax and be sure that is working code")
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
                filtered_steps = [
                    step
                    for step in steps
                    if step != "Remove unused elements or implement their intended purpose"
                ]
                issue.message = "Be sure that every parameter is used"
                filtered_steps.append(FUNCTION_ALREADY_CALLED)
                filtered_steps.append("• Print or Log the unused parameters.")
                rule_info_list[rule_key]["how_to_fix"]["steps"] = filtered_steps
                rule_info_list[rule_key]["name"] = "Be sure that every parameter is used"
                rule_info_list[rule_key]["root_cause"] = None

            strategies = self._extend_strategies_for_issue(issue=issue)

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
        strategy_text = "\n".join(strategies)

        context_dic = {
            "language": language,
            "issues": issues,
            "rule_info": rule_info_list,
            "context": context,
            "method_instruction": method_instruction,
            "error_message": error_message,
            "strategy_text": strategy_text,
        }
        prompt = template.render(**context_dic)
        return prompt.strip(), system_template

    def _parse_openai_response(self, response: Any) -> Optional[SonarFixResponse]:
        try:
            return response.output_parsed
        except Exception:
            logger.exception("Error parsing OpenAI response")
            return None

    def _parse_gemini_response(self, response: Any) -> Optional[Dict[str, Any]]:
        try:
            content = response.text
            return self._extract_fix_from_response(content)
        except Exception:
            logger.exception("Error parsing Gemini response")
            return None

    def _parse_chat_completion_response(self, response: Any) -> Optional[SonarFixResponse]:
        try:
            content = response.choices[0].message.content
            return self.parse_llm_response(content)
        except Exception:
            logger.exception("Error parsing %s response", self.provider)
            return None

    _parse_togetherai_response = _parse_chat_completion_response
    _parse_openrouter_response = _parse_chat_completion_response

    def _extract_using_regex_fallback(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            results = self._apply_regex_patterns(content)
            return self._validate_results(results)
        except Exception as e:
            logger.error("Regex fallback failed: %s", e)
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
            return float(value) if value is not None else 0.5
        if key == "PLACEMENT":
            return "SIBLING"
        if value is None:
            return ""
        value = value.replace('"', '"').replace("\\\\", "\\")
        value = value.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
        return value.strip()

    def _validate_results(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not results.get("FIXED_SELECTION"):
            logger.error("No FIXED_SELECTION found")
            return None
        return {
            "import_block": results["IMPORT_BLOCK"],
            "fixed_code": results["FIXED_SELECTION"],
            "helper_code": results["NEW_HELPER_CODE"],
            "placement_helper": results["PLACEMENT"].upper(),
            "explanation": results["EXPLANATION"] or "Code fix applied",
            "confidence": max(0.0, min(1.0, results["CONFIDENCE"])),
        }

    def _extract_fields_from_parsed_json(self, fix_data: dict) -> Optional[Dict[str, Any]]:
        fixed_code = str(fix_data.get("FIXED_SELECTION", "")).strip()
        fixed_code = fixed_code.replace('\\"', '"').replace("\\", "").replace("\\'", "'")
        import_block = str(fix_data.get("IMPORT_BLOCK", "")).strip()
        helper_code = str(fix_data.get("NEW_HELPER_CODE", "")).strip()
        helper_code = helper_code.replace("\\", "").replace('\\"', '"').replace("\\'", "'")
        placement = str(fix_data.get("PLACEMENT", "SIBLING")).strip().upper()
        explanation = str(fix_data.get("EXPLANATION", "")).strip()
        confidence = fix_data.get("CONFIDENCE", 0.5)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        if placement not in ["SIBLING", "GLOBAL_TOP", "GLOBAL_BOTTOM"]:
            placement = "SIBLING"
        if not explanation:
            explanation = "Code fix applied"
        return {
            "import_block": import_block,
            "fixed_code": fixed_code,
            "helper_code": helper_code,
            "placement_helper": placement,
            "explanation": explanation,
            "confidence": confidence,
        }

    def parse_llm_response(self, response_json: str) -> SonarFixResponse:
        try:
            data = json.loads(response_json)
            return SonarFixResponse(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e} for response {response_json}")
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}")

    def _extract_fix_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned_content = content.strip()
            cleaned_content.replace("```json", "").replace("```", "")
            cleaned_content = cleaned_content.split("{", 1)[1]
            cleaned_content = "{" + cleaned_content
            end = cleaned_content.rfind("}")
            cleaned_content = cleaned_content[: end + 1] if end != -1 else cleaned_content
            if cleaned_content.startswith("{") and cleaned_content.endswith("}"):
                try:
                    fix_data = json.loads(cleaned_content)
                    return self._extract_fields_from_parsed_json(fix_data)
                except json.JSONDecodeError:
                    pass
            return self._extract_using_regex_fallback(content)
        except Exception:
            logger.exception("Failed to extract LLM response")
            return None

    def _get_language_from_extension(self, extension: str) -> str:
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = project_path.parent / f"{project_path.name}_backup_{timestamp}"
        shutil.copytree(
            project_path, backup_path, ignore=shutil.ignore_patterns(".git")
        )

        return backup_path

    def _try_stored_file_path(self, fix: FixSuggestion, project_path: Path) -> Optional[str]:
        if not fix.file_path:
            return None
        file_path = project_path / fix.file_path
        return str(file_path) if file_path.exists() else None

    def _try_extract_from_issue_key(self, fix: FixSuggestion, project_path: Path) -> Optional[str]:
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

    def _get_file_from_fix(self, fix: FixSuggestion, project_path: Path) -> Optional[str]:
        file_path = self._try_stored_file_path(fix, project_path)
        if file_path:
            return file_path
        file_path = self._try_extract_from_issue_key(fix, project_path)
        if file_path:
            return file_path
        file_path = self._try_find_by_content(fix, project_path)
        if file_path:
            return file_path
        logger.warning("Could not determine file path for fix %s", fix.issue_key)
        return None

    def apply_indentation_to_fix(self, fixed_code: str, base_indent: str) -> str:
        if not fixed_code.strip():
            return fixed_code
        lines = fixed_code.split("\n")
        if not lines:
            return fixed_code
        lines = normalize_indentation(lines)
        indented_lines = []
        for line in lines:
            if line.strip():
                indented_lines.append(base_indent + line)
            else:
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

    # ------------------------------------------------------------------
    # Test execution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dotenv_file(env_file: Path) -> Dict[str, str]:
        """
        Parse a .env file into a dict without any third-party dependency.

        Handles:
          - KEY=value
          - KEY="quoted value"
          - KEY='single-quoted value'
          - export KEY=value
          - blank lines and # comments
        """
        result: Dict[str, str] = {}
        try:
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                # Strip optional leading "export "
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, _, raw_value = line.partition("=")
                key = key.strip()
                raw_value = raw_value.strip()
                # Strip surrounding quotes
                if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in ('"', "'"):
                    raw_value = raw_value[1:-1]
                if key:
                    result[key] = raw_value
        except OSError:
            pass
        return result

    def _build_subprocess_env(self, project_root: Path,test_path:Path, python_executable: str) -> Dict[str, str]:
        """
        Build the environment dict for subprocess.run so that pytest inherits:
          1. The current process environment (os.environ)
          2. Any KEY=VALUE pairs from the project's .env file
          3. Venv activation vars: VIRTUAL_ENV, prepended PATH, cleared PYTHONHOME

        This mirrors what `source venv/bin/activate` does in a shell,
        ensuring env-vars like `export DB_HOST=localhost` are visible to tests.
        """
        env = os.environ.copy()

        # Layer 2: .env file vars (overrides process env)
        env_file = project_root / ".env"
        test_env_file = test_path / ".env"
        if env_file.exists():
            dotenv_vars = self._load_dotenv_file(env_file)
            env.update(dotenv_vars)
            logger.debug("Loaded %d vars from %s", len(dotenv_vars), env_file)
        elif test_env_file.exists():
            dotenv_vars = self._load_dotenv_file(test_env_file)
            env.update(dotenv_vars)
            logger.debug("Loaded %d vars from %s", len(dotenv_vars), test_env_file)
        # Layer 3: venv activation (overrides PATH / interpreter resolution)
        venv_dir = Path(python_executable).parent.parent  # e.g. project/venv
        if (venv_dir / "bin" / "python").exists() or (venv_dir / "Scripts" / "python.exe").exists():
            env["VIRTUAL_ENV"] = str(venv_dir)
            # Prepend venv bin so subprocesses find the right pip/pytest
            bin_dir = str(venv_dir / "bin") if (venv_dir / "bin").exists() else str(venv_dir / "Scripts")
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
            # Avoid PYTHONHOME from a different env bleeding in
            env.pop("PYTHONHOME", None)

        return env

    def run_testing_cases(self, testing_path: str) -> Tuple[bool, Optional[str]]:
        """
        Run pytest for the given test path, using the project's virtual environment
        if one exists, otherwise falling back to the current Python interpreter.

        Env variable loading order (each layer overrides the previous):
          1. Current process environment (os.environ)
          2. Project .env file (if present) — picks up export KEY=value declarations
          3. Venv activation vars (VIRTUAL_ENV, PATH prepended with venv bin)
        """
        test_path = Path(testing_path)
        project_root = self._find_project_root(test_path)
        python_executable = self._resolve_python_executable(project_root)
        env = self._build_subprocess_env(project_root, test_path,python_executable)
        logger.debug("run_testing_cases: python=%s cwd=%s", python_executable, project_root)

        try:
            subprocess.run(
                [python_executable, "-m", "pytest", str(test_path), "-v"],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(project_root),  # run from project root so conftest imports resolve
                env=env,
            )
            return True, None

        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout or "Unknown error"

            # If it's a missing module, try auto-installing from requirements
            if "ModuleNotFoundError" in error_output or "ImportError" in error_output:
                installed = self._try_install_requirements(project_root, python_executable)
                if installed:
                    # Retry once after installing
                    return self.run_testing_cases(testing_path)

            return False, error_output.strip()

    def _find_project_root(self, test_path: Path) -> Path:
        """
        Walk up from test_path to find the project root
        (directory containing pyproject.toml, setup.py, or requirements.txt).
        """
        current = test_path if test_path.is_dir() else test_path.parent
        for parent in [current, *current.parents]:
            if any((parent / marker).exists() for marker in [
                "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"
            ]):
                return parent
        return current  # fallback to test directory

    def _resolve_python_executable(self, project_root: Path) -> str:
        """
        Prefer the venv's Python inside the project root.
        Checks common venv directory names: venv, .venv, env, .env
        Falls back to sys.executable (the interpreter running devdox-ai-sonar).
        """
        venv_candidates = ["venv", ".venv", "env", ".env"]
        for venv_name in venv_candidates:
            venv_python = project_root / venv_name / "bin" / "python"       # Linux/Mac
            venv_python_win = project_root / venv_name / "Scripts" / "python.exe"  # Windows
            if venv_python.exists():
                logger.debug("Using venv python: %s", venv_python)
                return str(venv_python)
            if venv_python_win.exists():
                logger.debug("Using venv python (win): %s", venv_python_win)
                return str(venv_python_win)

        # Fallback: use the same Python that's running devdox-ai-sonar
        logger.debug("No venv found in %s, falling back to sys.executable", project_root)
        return sys.executable

    def _try_install_requirements(self, project_root: Path, python_executable: str) -> bool:
        """
        Attempt to pip install from requirements.txt or pyproject.toml in the project root.
        Returns True if installation succeeded.
        """
        req_file = project_root / "requirements.txt"
        pyproject = project_root / "pyproject.toml"

        try:
            if req_file.exists():
                logger.info("Installing dependencies from %s", req_file)
                subprocess.run(
                    [python_executable, "-m", "pip", "install", "-r", str(req_file)],
                    check=True, capture_output=True, text=True,
                )
                return True

            if pyproject.exists():
                logger.info("Installing project dependencies from %s", pyproject)
                subprocess.run(
                    [python_executable, "-m", "pip", "install", "-e", str(project_root)],
                    check=True, capture_output=True, text=True,
                )
                return True

        except subprocess.CalledProcessError as e:
            logger.warning("Failed to auto-install dependencies: %s", e.stderr)

        return False

    def check_python_interpreter(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        try:
            subprocess.run(
                ["python", "-m", "py_compile", str(file_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return True, None
        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout or "Unknown error"
            return False, error_output.strip()

    def _check_bracket_balance(self, content: str) -> bool:
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

    def _check_no_duplicate_definitions(self, content: str, file_extension: str) -> bool:
        if file_extension != ".py":
            return True
        func_pattern = r"^\s*(async\s+)?def\s+(\w+)\s*\("
        class_pattern = r"^\s*class\s+(\w+)\s*[:\(]"
        definitions: Dict[str, List[int]] = {}
        for line_num, line in enumerate(content.split("\n"), 1):
            func_match = re.match(func_pattern, line)
            if func_match:
                def_name = func_match.group(2)
                definitions.setdefault(def_name, []).append(line_num)
            class_match = re.match(class_pattern, line)
            if class_match:
                def_name = class_match.group(1)
                definitions.setdefault(def_name, []).append(line_num)
        for def_name, line_numbers in definitions.items():
            if len(line_numbers) > 1:
                logger.error("Found duplicate definition '%s' at lines: %s", def_name, line_numbers)
                return False
        return True

    def _create_backup_if_needed(
        self, result: FixResult, project_path: Path, create_backup: bool, dry_run: bool
    ) -> None:
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
        file_fix_pairs: List[Tuple[FixSuggestion, Union[SonarIssue, SonarSecurityIssue]]],
        result: FixResult,
        validator: Optional[FixValidator],
        dry_run: bool,
    ) -> None:
        file_fixes = [fix for fix, _ in file_fix_pairs]
        original_content = ""
        if file_path.exists():
            original_content = await self.read_file_async(str(file_path))

        success, new_fixes = await self._apply_fixes_to_file(file_path, file_fixes, dry_run)

        if success:
            result.successful_fixes.extend(file_fixes)
            return

        if validator:
            await self._handle_failed_fixes_with_validator(
                file_path, file_fix_pairs, new_fixes, original_content, result, validator, dry_run
            )
        else:
            result.failed_fixes.extend(
                [{"fix": fix, "error": "Direct application failed and no validator available"}
                 for fix in file_fixes]
            )
            logger.error("✗ Failed to apply fixes to %s (no validator fallback)", file_path)

    async def _handle_failed_fixes_with_validator(
        self,
        file_path: Path,
        file_fix_pairs: List[Tuple[FixSuggestion, Union[SonarIssue, SonarSecurityIssue]]],
        new_fixes: List[Any],
        original_content: str,
        result: FixResult,
        validator: FixValidator,
        dry_run: bool,
    ) -> None:
        reason_msg = ", ".join(
            fix_app.reason for fix_app in new_fixes if fix_app.reason is not None
        )
        logger.warning(
            "Direct fix application failed for %s. Trying AI validator fallback to fix %s...",
            file_path, reason_msg,
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
                    fix, issue, current_content, new_error_msg=reason_msg,
                )
                await self._process_validation_result(
                    validation_result, fix, file_path, result, dry_run,
                )
            except Exception as e:
                logger.error("Error in validator fallback for fix %s: %s", fix.issue_key, e, exc_info=True)
                result.failed_fixes.append({"fix": fix, "error": f"Validator error: {str(e)}"})

    async def _process_validation_result(
        self,
        validation_result: Any,
        fix: FixSuggestion,
        file_path: Path,
        result: FixResult,
        dry_run: bool,
    ) -> None:
        status = validation_result.status

        if status == ValidationStatus.APPROVED:
            result.successful_fixes.append(fix)
            return

        if status == ValidationStatus.REJECTED:
            logger.warning("✗ Fix %s REJECTED: %s", fix.issue_key, validation_result.explanation)
            result.failed_fixes.append(
                {"fix": fix, "error": f"Rejected by validator: {validation_result.explanation}"}
            )
            return

        if status == ValidationStatus.NEEDS_REVIEW:
            logger.warning("? Fix %s NEEDS_REVIEW: %s", fix.issue_key, validation_result.explanation)
            result.failed_fixes.append(
                {"fix": fix, "error": f"Needs manual review: {validation_result.explanation}"}
            )
            return

        if not validation_result.final_fix:
            result.failed_fixes.append(
                {"fix": fix, "error": "Validator marked as MODIFIED but provided no improved fix"}
            )
            return

        improved_success, _ = await self._apply_fixes_to_file(
            file_path, [validation_result.final_fix], dry_run,
        )
        if improved_success:
            result.successful_fixes.append(validation_result.final_fix)
        else:
            result.failed_fixes.append(
                {"fix": fix, "error": "Validator improved fix but application still failed"}
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
        """Apply fixes with optional validation by a senior code reviewer agent."""
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
                    file_path, file_fix_pairs, result, validator, dry_run,
                )
            except Exception as e:
                result.failed_fixes.extend(
                    [{"fix": fix, "error": str(e)} for fix, _ in file_fix_pairs]
                )
                logger.error("✗ Error processing file %s: %s", file_path_str, e, exc_info=True)

        return result

    def _try_find_by_content(self, fix: FixSuggestion, project_path: Path) -> Optional[str]:
        if not fix.original_code or len(fix.original_code.strip()) <= 10:
            return None
        matching_files = self._find_files_with_content(project_path, fix.original_code.strip())
        return str(matching_files[0]) if matching_files else None

    def _find_files_with_content(self, project_path: Path, content: str) -> List[Path]:
        matching_files = []
        extensions = {
            ".py", ".js", ".jsx", ".ts", ".tsx",
            java_extension, ".kt", scala_extension,
            ".go", ".rs", ".cpp", ".c", ".cs", ".php", ".rb", ".swift",
        }
        try:
            for file_path in project_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix in extensions
                    and not any(part.startswith(".") for part in file_path.parts)
                    and "node_modules" not in file_path.parts
                    and "venv" not in file_path.parts
                    and "__pycache__" not in file_path.parts
                ):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            if content in f.read():
                                matching_files.append(file_path)
                                if len(matching_files) >= 3:
                                    break
                    except (IOError, UnicodeDecodeError):
                        continue
        except Exception as e:
            logger.warning("Error searching for files with content: %s", e)
        return matching_files

    def _looks_like_file_path(self, path_candidate: str) -> bool:
        if "/" not in path_candidate:
            return False
        supported_extensions = {
            ".py", ".js", java_extension, ".ts", ".jsx", ".tsx", ".kt", scala_extension,
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
        try:
            file_lines = await self.file_reader.read_lines(file_path)
            problem_line_content = _extract_problem_lines(file_lines, line_range["problem_lines"])
            first_problem_line_idx = line_range["first_line"] - 1
            class_name = self._extract_class_name_from_file(file_lines, first_problem_line_idx)
            if class_name:
                logger.debug("Detected class name: %s at line %d", class_name, line_range["first_line"])
            else:
                logger.debug("No class name detected for line %d", line_range["first_line"])

            import_section = self._extract_import_section(file_lines, language)
            import_section["has_imports"] = False

            if modified_content:
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
                context_dict = self._extract_context(
                    file_lines,
                    line_range["first_line"],
                    line_range["last_line"],
                    line_range["problem_lines"],
                    context_lines,
                )
                context_dict["language"] = language
                if "class_name" not in context_dict or context_dict["class_name"] is None:
                    context_dict["class_name"] = class_name
                context_dict["import_section"] = import_section
            context_dict["problem_line_content"] = problem_line_content
            return {
                "context_dict": context_dict,
                "file_lines": file_lines,
                "problem_line_content": problem_line_content,
            }
        except UnicodeDecodeError as e:
            logger.error("File encoding error for %s: %s", file_path, e)
            return None
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, e)
            return None

    def _extract_import_section(self, file_lines: List[str], language: str) -> Dict[str, Any]:
        if language == "python":
            return self._extract_python_import_section(file_lines)
        logger.debug("Import extraction not implemented for language: %s", language)
        return {"start_line": 0, "end_line": 0, "content": "", "has_imports": False}

    def _should_skip_docstring(self, stripped: str, start_line: Optional[int]) -> bool:
        return stripped.startswith(('"""', "'''"))

    def _handle_backslash_continuation_line(
        self, original_line: str, stripped: str, line_num: int,
        import_lines: List[str], state: Dict[str, bool],
    ) -> int:
        import_lines.append(original_line)
        if not stripped.rstrip().endswith("\\"):
            state["in_backslash_continuation"] = False
            state["in_multiline_import"] = False
        return line_num

    def _handle_parentheses_import_line(
        self, original_line: str, stripped: str, line_num: int,
        import_lines: List[str], state: Dict[str, bool],
    ) -> int:
        import_lines.append(original_line)
        if ")" in stripped:
            state["in_parentheses_import"] = False
            state["in_multiline_import"] = False
            if stripped.rstrip().endswith("\\"):
                state["in_backslash_continuation"] = True
                state["in_multiline_import"] = True
        return line_num

    def _extract_python_import_section(self, file_lines: List[str]) -> Dict[str, Any]:
        import_lines: List[str] = []
        start_line: Optional[int] = None
        end_line: Optional[int] = None
        state = {
            "in_multiline_import": False,
            "in_parentheses_import": False,
            "in_backslash_continuation": False,
        }
        for idx, line in enumerate(file_lines):
            line_num = idx + 1
            stripped = line.strip()
            original_line = line
            if self._should_skip_docstring(stripped, start_line):
                if start_line is not None:
                    break
                continue
            if state["in_parentheses_import"]:
                end_line = self._handle_parentheses_import_line(
                    original_line, stripped, line_num, import_lines, state
                )
                continue
            if state["in_backslash_continuation"]:
                end_line = self._handle_backslash_continuation_line(
                    original_line, stripped, line_num, import_lines, state
                )
                continue
            if self._is_import_statement_start(stripped):
                if start_line is None:
                    start_line = line_num
                import_lines.append(original_line)
                end_line = line_num
                state = self._detect_multiline_import_pattern(stripped, state)
                continue
            if self._should_stop_at_decorator(stripped, start_line):
                break
            if self._should_stop_at_code(stripped, start_line, state):
                break
        return self._build_import_section_result(start_line, end_line, import_lines)

    def _should_stop_at_code(self, stripped: str, start_line: Optional[int], state: Dict[str, bool]) -> bool:
        if state["in_multiline_import"]:
            return False
        if start_line is None:
            return False
        if not stripped:
            return False
        if self._is_module_level_dunder(stripped):
            return False
        return True

    def _is_module_level_dunder(self, stripped: str) -> bool:
        return stripped.startswith(("__all__", "__version__", "__author__", "__"))

    def _should_stop_at_decorator(self, stripped: str, start_line: Optional[int]) -> bool:
        if stripped.startswith("@"):
            return start_line is not None
        return False

    def _detect_multiline_import_pattern(self, stripped: str, state: Dict[str, bool]) -> Dict[str, bool]:
        if "(" in stripped and ")" not in stripped:
            state["in_parentheses_import"] = True
            state["in_multiline_import"] = True
        elif stripped.rstrip().endswith("\\"):
            state["in_backslash_continuation"] = True
            state["in_multiline_import"] = True
        else:
            state["in_multiline_import"] = False
            state["in_parentheses_import"] = False
            state["in_backslash_continuation"] = False
        return state

    def _is_import_statement_start(self, stripped: str) -> bool:
        return stripped.startswith("from ") or stripped.startswith("import ")

    def _build_import_section_result(
        self, start_line: Optional[int], end_line: Optional[int], import_lines: List[str],
    ) -> Dict[str, Any]:
        if start_line is None:
            return {"start_line": 0, "end_line": 0, "content": "", "has_imports": False}
        return {
            "start_line": start_line,
            "end_line": end_line or start_line,
            "content": "".join(import_lines),
            "has_imports": True,
        }

    def _extract_class_name_from_file(self, file_lines: List[str], target_line_idx: int) -> Optional[str]:
        for i in range(target_line_idx, -1, -1):
            line = file_lines[i].strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            match = re.search(r"class\s+(\w+)\s*[:(]", line)
            if match:
                class_name = match.group(1)
                if self._is_line_inside_class(file_lines, target_line_idx, i):
                    return class_name
        return None

    def _is_line_inside_class(self, file_lines: List[str], target_idx: int, class_start_idx: int) -> bool:
        if target_idx <= class_start_idx:
            return False
        class_line = file_lines[class_start_idx]
        class_indent = len(class_line) - len(class_line.lstrip())
        for i in range(class_start_idx + 1, target_idx + 1):
            line = file_lines[i]
            if not line.strip() or line.strip().startswith("#"):
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= class_indent and i != target_idx:
                if re.search(r"class\s+\w+\s*[:(]", line):
                    return False
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
        functions_with_issues = []
        processed_function_starts = set()

        for problem_idx in problem_indices:
            if problem_idx < range_start or problem_idx > range_end:
                logger.debug("Skipping problem line %d - outside range", problem_idx + 1)
                continue
            containing_func_idx = self._find_containing_function(problem_idx)
            if containing_func_idx is None:
                logger.warning("Problem line %d is not inside any function", problem_idx + 1)
                continue
            if containing_func_idx in processed_function_starts:
                continue
            function_info = self._get_function_info(containing_func_idx)
            if not function_info:
                logger.error("Failed to get function info for line %d", containing_func_idx + 1)
                continue
            function_problem_lines = [
                idx + 1
                for idx in problem_indices
                if function_info["start_idx"] <= idx <= function_info["end_idx"]
            ]
            function_info["problem_lines_in_function"] = function_problem_lines
            functions_with_issues.append(function_info)
            processed_function_starts.add(containing_func_idx)

        functions_with_issues.sort(key=lambda f: int(f["start_idx"]))
        logger.info(
            "Found %d functions containing %d problem lines",
            len(functions_with_issues), len(problem_indices),
        )
        return functions_with_issues

    def _build_optimized_context(
        self, functions: List[Dict[str, Any]], problem_lines: List[int]
    ) -> Optional[Dict[str, Any]]:
        if not functions:
            return None

        new_context = []
        combined_lines = []
        function_metadata = []

        for func in functions:
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
            combined_lines.extend(func["lines"])
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
            "new_context": new_context,
            "context": "".join(combined_lines),
            "start_line": overall_start + 1,
            "end_line": overall_end + 1,
            "is_multi_function": len(functions) > 1,
            "function_count": len(functions),
            "functions": function_metadata,
            "is_complete_function": True,
            "total_problem_lines": len(problem_lines),
            "functions_with_issues": len(functions),
            "coverage_percentage": (
                round(
                    (len(functions) / self._count_functions_in_range(overall_start, overall_end)) * 100,
                    2,
                )
                if self._count_functions_in_range(overall_start, overall_end) > 0
                else 100
            ),
        }

    def _count_functions_in_range(self, start_idx: int, end_idx: int) -> int:
        count = 0
        for line_idx in range(start_idx, end_idx + 1):
            if line_idx >= len(self.lines):
                break
            if self._is_function_definition(self.lines[line_idx].rstrip()):
                count += 1
        return count

    def _extract_all_functions_in_range(
        self, first_idx: int, last_idx: int, problem_lines: List[int],
    ) -> Optional[Dict[str, Any]]:
        problem_indices = [line - 1 for line in problem_lines]
        functions_with_issues = self._find_functions_containing_problems(
            problem_indices, first_idx, last_idx
        )
        if not functions_with_issues:
            logger.warning(
                "No functions found containing problem lines %s in range %d-%d",
                problem_lines, first_idx + 1, last_idx + 1,
            )
            return None
        return self._build_optimized_context(functions_with_issues, problem_lines)

    def _get_function_info(self, func_start_idx: int) -> Optional[Dict[str, Any]]:
        if func_start_idx >= len(self.lines):
            return None
        func_line = self.lines[func_start_idx].rstrip()
        if not self._is_function_definition(func_line):
            return None
        func_name = self._extract_function_name(func_line)
        decorator_start_idx = self._find_decorator_start(func_start_idx)
        func_end_idx = self._find_function_end(func_start_idx)
        if func_end_idx is None:
            return None
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
        if func_def_idx == 0:
            return func_def_idx
        func_line = self.lines[func_def_idx].rstrip()
        func_indent = len(func_line) - len(func_line.lstrip())
        current_idx = func_def_idx - 1
        decorator_start = func_def_idx
        while current_idx >= 0:
            line = self.lines[current_idx].rstrip()
            if not line or line.lstrip().startswith("#"):
                current_idx -= 1
                continue
            stripped = line.lstrip()
            line_indent = len(line) - len(stripped)
            if stripped.startswith("@") and line_indent == func_indent:
                decorator_start = current_idx
                current_idx -= 1
            else:
                break
        return decorator_start

    def _is_line_inside_function(self, lines: List[str], line_idx: int, function_start_idx: int) -> bool:
        if line_idx < function_start_idx:
            return False
        function_end_idx = self._find_function_end(function_start_idx)
        if function_end_idx is None:
            return self._check_indentation_containment(lines, line_idx, function_start_idx)
        return line_idx <= function_end_idx

    def _find_brace_function_end(self, lines: List[str], start_idx: int) -> Optional[int]:
        if start_idx >= len(lines):
            return None
        brace_count = 0
        found_opening_brace = False
        for i in range(start_idx, len(lines)):
            clean_line = self._remove_strings_and_comments(lines[i])
            for char in clean_line:
                if char == "{":
                    brace_count += 1
                    found_opening_brace = True
                elif char == "}":
                    brace_count -= 1
                    if found_opening_brace and brace_count == 0:
                        return i
        return None

    def _remove_strings_and_comments(self, line: str) -> str:
        line = re.sub(r"//.*$", "", line)
        line = re.sub(r"#.*$", "", line)
        line = re.sub(r'"[^"]*"', '""', line)
        line = re.sub(r"'[^']*'", "''", line)
        line = re.sub(r"`[^`]*`", "``", line)
        return line

    def _check_indentation_containment(self, lines: List[str], line_idx: int, function_start_idx: int) -> bool:
        if function_start_idx >= len(lines) or line_idx >= len(lines):
            return False
        func_line = lines[function_start_idx]
        func_indent = len(func_line) - len(func_line.lstrip())
        target_line = lines[line_idx]
        if not target_line.strip():
            return True
        target_indent = len(target_line) - len(target_line.lstrip())
        return target_indent > func_indent

    def _find_python_function_end(self, lines: List[str], start_idx: int) -> Optional[int]:
        if start_idx >= len(lines):
            return None
        function_line = lines[start_idx]
        function_indent = len(function_line) - len(function_line.lstrip())
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip() or line.strip().startswith("#"):
                continue
            current_indent = len(line) - len(line.lstrip())
            if self._has_reached_function_end(current_indent, function_indent, line, lines[i - 1]):
                return i - 1
        return len(lines) - 1

    def _has_reached_function_end(self, current_indent: int, function_indent: int, line: str, prev_line: str) -> bool:
        if (current_indent <= function_indent) and not (
            prev_line.rstrip().endswith(("\\", ",", "("))
            or line.strip().startswith((")", "def ", "async def"))
        ):
            return True
        if current_indent == function_indent and self._is_function_definition(line):
            return True
        return False

    def _is_function_definition(self, line: str) -> bool:
        stripped_line = line.strip()
        python_patterns = [
            r"^def\s+\w+\s*\(",
            r"^async\s+def\s+\w+\s*\(",
            r"^\s*def\s+\w+\s*\(",
            r"^\s*async\s+def\s+\w+\s*\(",
        ]
        js_patterns = [
            r"^function\s+\w+\s*\(",
            r"^async\s+function\s+\w+\s*\(",
            r"^\w+\s*:\s*function\s*\(",
            r"^\w+\s*:\s*async\s+function\s*\(",
            r"^\w+\s*=\s*function\s*\(",
            r"^\w+\s*=\s*async\s+function\s*\(",
            r"^\w+\s*=\s*\([^)]*\)\s*=>\s*\{",
            r"^\w+\s*=\s*async\s*\([^)]*\)\s*=>\s*\{",
            r"^\s*\w+\s*\([^)]*\)\s*\{",
            r"^\s*async\s+\w+\s*\([^)]*\)\s*\{",
            r"^(?:const|let|var)\s+\w+\s*=\s*\([^)]*\)\s*=>\s*\{",
            r"^(?:const|let|var)\s+\w+\s*=\s*async\s*\([^)]*\)\s*=>\s*\{",
        ]
        java_csharp_patterns = [
            r"^(public|private|protected|static|internal|\s)+(.*\s+)?\w+\s*\([^)]*\)\s*\{",
            r"^\s*(public|private|protected|static|internal|\s)+(.*\s+)?\w+\s*\([^)]*\)\s*\{",
        ]
        for pattern in python_patterns + js_patterns + java_csharp_patterns:
            if re.match(pattern, stripped_line, re.IGNORECASE):
                return True
        return False

    def _is_function_line(self, line_idx: int) -> bool:
        if line_idx >= len(self.lines):
            return False
        return self._is_function_definition(self.lines[line_idx])

    def _find_containing_function(self, target_line_idx: int) -> Optional[int]:
        lines = self.lines
        for i in range(target_line_idx, -1, -1):
            line = lines[i].strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if self._is_function_definition(lines[i]):
                if self._is_line_inside_function(lines, target_line_idx, i):
                    return i
        return None

    def _find_function_end(self, start_idx: int) -> Optional[int]:
        lines = self.lines
        if start_idx >= len(lines):
            return None
        start_line = lines[start_idx]
        if re.search(r"\bdef\b", start_line):
            return self._find_python_function_end(lines, start_idx)
        elif "{" in start_line or (
            re.search(r"\bfunction\b", start_line)
            and re.search(r"[a-zA-Z_]\w*\s*\([\w,\s]*\)\s*\{", start_line)
        ):
            return self._find_brace_function_end(lines, start_idx)
        elif re.search(r"\b@click.\b", start_line):
            return self._find_python_function_end(lines, start_idx)
        else:
            python_end = self._find_python_function_end(lines, start_idx)
            if python_end is not None:
                return python_end
            return self._find_brace_function_end(lines, start_idx)

    def _extract_function_name(self, function_line: str) -> str:
        python_match = re.search(r"def\s+(\w+)", function_line)
        if python_match:
            return python_match.group(1)
        js_match = re.search(r"function\s+(\w+)", function_line)
        if js_match:
            return js_match.group(1)
        assignment_match = re.search(r"([a-zA-Z_]\w{0,99})\s{0,20}[:=]", function_line)
        if assignment_match:
            return assignment_match.group(1)
        method_match = re.search(r"\b(\w+)\s*\([^)]*\)\s*\{?", function_line)
        if method_match:
            name = method_match.group(1)
            keywords = {"public", "private", "protected", "static", "async", "void", "int", "string", "bool", "return"}
            if name.lower() not in keywords:
                return name
        return ""

    def _is_actual_function_def(self, line: str) -> bool:
        stripped_line = line.strip()
        for pattern in [r"^def\s+\w+\s*\(", r"^async\s+def\s+\w+\s*\("]:
            if re.match(pattern, stripped_line):
                return True
        return False

    def _extract_complete_function(
        self, start_idx: int, problem_line_idx: int, last_line_idx: int
    ) -> Optional[Dict[str, Any]]:
        lines = self.lines
        if start_idx >= len(lines):
            return None
        function_def_line_idx = start_idx
        if self._is_decorator(lines[start_idx].strip()):
            for i in range(start_idx, min(len(lines), start_idx + 10)):
                if self._is_actual_function_def(lines[i]):
                    function_def_line_idx = i
                    break
        function_start = self._find_function_start_with_decorators(lines, function_def_line_idx)
        function_end = self._find_function_end(function_def_line_idx)
        if function_end is None:
            function_end = min(len(lines) - 1, function_def_line_idx + 50)
        function_lines = lines[function_start : last_line_idx + 1]
        if function_end > last_line_idx:
            function_lines = lines[function_start : function_end + 1]
        context_text = "".join(function_lines)
        function_def_line = lines[function_def_line_idx].rstrip()
        decorators = [lines[i].strip() for i in range(function_start, function_def_line_idx) if lines[i].strip()]
        problem_line = lines[problem_line_idx].rstrip()
        return {
            "context": context_text,
            "function_definition_line": function_def_line,
            "decorators": decorators,
            "line_number": start_idx + 1,
            "function_start_line": function_start + 1,
            "end_line": function_end + 1,
            "is_complete_function": True,
            "function_name": self._extract_function_name(function_def_line),
            "has_decorators": len(decorators) > 0,
            "decorator_count": len(decorators),
            "start_line": function_start + 1,
            "problem_lines": [problem_line],
        }

    def _find_function_start_with_decorators(self, lines: List[str], target_line_idx: int) -> int:
        start_idx = target_line_idx
        for i in range(target_line_idx - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                break
            if self._is_decorator(line):
                start_idx = i
            else:
                break
        return start_idx

    def _is_decorator(self, line: str) -> bool:
        stripped_line = line.strip()
        for pattern in [r"^@\w+$", r"^@\w+\.[^(]*$", r"^@\w+\(", r"^@\w+\.[^(]*\("]:
            if re.match(pattern, stripped_line):
                return True
        return False

    def _try_function_extraction_strategies(self, first_idx: int, last_idx: int) -> Optional[Dict[str, Any]]:
        context = self._try_extract_containing_function(first_idx, last_idx)
        if context:
            return context
        if first_idx != last_idx:
            context = self._try_expand_multiline_to_function(first_idx, last_idx)
            if context:
                return context
        return None

    def _try_expand_multiline_to_function(self, first_idx: int, last_idx: int) -> Optional[Dict[str, Any]]:
        function_start_idx = self._find_containing_function(last_idx)
        if function_start_idx is None:
            return None
        function_end_idx = self._find_function_end(function_start_idx)
        if function_end_idx is None:
            return None
        if function_end_idx <= last_idx:
            return None
        return self._extract_complete_function(function_start_idx, first_idx, last_idx)

    def _try_extract_containing_function(self, first_idx: int, last_idx: int) -> Optional[Dict[str, Any]]:
        function_start_idx = self._find_containing_function(first_idx)
        if function_start_idx is None:
            return None
        return self._extract_complete_function(function_start_idx, first_idx, last_idx)

    def _extract_normal_context(self, first_idx: int, last_idx: int, context_lines: int) -> Dict[str, Any]:
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
            "line_number": first_idx + 1,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "is_complete_function": False,
        }

    def _get_empty_context(self, line_number: int) -> Dict[str, Any]:
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
    context_dict = context_info["context_dict"]
    end_line = context_dict.get("end_line", line_range["last_line"])
    problem_lines_raw: Any = line_range.get("problem_lines", [])
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
    if not problem_lines:
        return "fix_unknown"
    if len(problem_lines) == 1:
        return f"fix_L{problem_lines[0]}"
    return f"fix_L{min(problem_lines)}-L{max(problem_lines)}"


def _extract_problem_lines(
    file_lines: List[str], problem_line_numbers: List[int]
) -> Dict[int, str]:
    problem_lines = {}
    for line_number in problem_line_numbers:
        line_index = line_number - 1
        if 0 <= line_index < len(file_lines):
            problem_lines[line_number] = file_lines[line_index]
        else:
            logger.warning(
                "Line number %d out of bounds (file has %d lines)",
                line_number, len(file_lines),
            )
    return problem_lines
