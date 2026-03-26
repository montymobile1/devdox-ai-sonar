"""Fix validation agent for reviewing and improving LLM-generated fixes."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, cast
from enum import Enum
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
import logging
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate
from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    FixSuggestion,
    CodeBlock,
    ChangeType,
    ChangeAction,
    SonarFixResponse,
)

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Status of fix validation."""

    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ValidationResult:
    """Result of fix validation."""

    def __init__(
        self,
        status: ValidationStatus,
        original_fix: FixSuggestion,
        modified_fix: Optional[FixSuggestion] = None,
        explanation: str = "",
        concerns: Optional[List[str]] = None,
        confidence: float = 0.0,
    ):
        self.status = status
        self.original_fix = original_fix
        self.modified_fix = modified_fix or original_fix
        self.explanation = explanation
        self.concerns = concerns or []
        self.confidence = confidence

    @property
    def final_fix(self) -> FixSuggestion:
        """Get the final fix to apply (modified or original)."""
        return (
            self.modified_fix
            if self.status == ValidationStatus.MODIFIED
            else self.original_fix
        )

    @property
    def should_apply(self) -> bool:
        """Check if the fix should be applied."""
        return self.status in [ValidationStatus.APPROVED, ValidationStatus.MODIFIED]


# ---------------------------------------------------------------------------
# Formatting helpers (unchanged)
# ---------------------------------------------------------------------------

def _format_block_header(idx: int, block: CodeBlock) -> str:
    return (
        f"\n{'=' * 60}\n"
        f"Block {idx}: {block.block_name} (Lines {block.start_line}-{block.end_line})\n"
        f"Type: {block.block_type.value} | Change Type: {block.change_type.value}\n"
        f"Has Changes: {block.has_changes}\n{'=' * 60}\n"
    )


def _format_full_code_content(block: CodeBlock) -> str:
    return f"```python\n{block.context}\n```\n"


def _format_diff_content(block: CodeBlock) -> str:
    parts = ["Changes:\n"]
    for change in block.changes or []:
        if change.action == ChangeAction.REPLACE:
            parts.append(f"  Line {change.line} (REPLACE):\n")
            parts.append(f"    - Old: {change.old}\n")
            parts.append(f"    + New: {change.new}\n")
        elif change.action == ChangeAction.INSERT:
            parts.append(f"  Line {change.line} (INSERT):\n")
            parts.append(f"    + New: {change.new}\n")
        elif change.action == ChangeAction.DELETE:
            parts.append(f"  Line {change.line} (DELETE):\n")
            parts.append(f"    - Old: {change.old}\n")
    parts.append("\n")
    return "".join(parts)


def _format_search_replace_content(block: CodeBlock) -> str:
    parts = ["Search/Replace Operations:\n"]
    for op_idx, repl in enumerate(block.replacements or [], 1):
        regex_marker = " (REGEX)" if repl.is_regex else ""
        count_info = f" (count: {repl.count})" if repl.count else " (all occurrences)"
        parts.append(f"\n  Operation {op_idx}{regex_marker}{count_info}:\n")
        parts.append(f"    Search:  {repr(repl.search)}\n")
        parts.append(f"    Replace: {repr(repl.replace)}\n")
    parts.append("\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# LangChain builder (mirrors build_llm in llm_fixer.py)
# ---------------------------------------------------------------------------

def _build_validator_llm(
    provider: str,
    model: str,
    api_key: str,
) -> Any:
    """
    Build a LangChain chat model for the validator.
    Uses with_structured_output(SonarFixResponse) for all providers.
    Each provider uses its own LangChain library — no raw SDK clients.
    """
    from langchain_core.language_models import BaseChatModel

    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            max_tokens=8000,
            temperature=0.1,
        )

    if provider == "togetherai":
        from langchain_together import ChatTogether
        return ChatTogether(
            model=model,
            together_api_key=api_key,
            max_tokens=8000,
            temperature=0.1,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=8000,
            temperature=0.1,
            default_headers={
                "HTTP-Referer": "https://devdox.ai",
                "X-Title": "DevDox AI Sonar",
            },
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            max_output_tokens=8000,
            temperature=0.1,
        )

    raise ValueError(
        f"Unsupported provider: {provider}. "
        "Use 'openai', 'togetherai', 'openrouter', or 'gemini'."
    )


# ---------------------------------------------------------------------------
# FixValidator
# ---------------------------------------------------------------------------

class FixValidator:
    """
    Senior code reviewer that validates and potentially improves LLM-generated fixes.

    Uses LangChain with_structured_output(SonarFixResponse) for all providers.
    No raw SDK clients — provider SDKs are accessed only through LangChain.

    LLMFixer and FixValidator are independent: each can use a different
    provider/model combination. For example:
        fixer     = LLMFixer(provider="togetherai", model="llama-3-70b")
        validator = FixValidator(provider="openai",   model="gpt-4o")
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        min_confidence_threshold: float = 0.7,
    ):
        self.provider = provider.lower()
        self.min_confidence_threshold = min_confidence_threshold
        self.model: str = self._resolve_model(provider, model)
        self.api_key: str = self._resolve_api_key(provider, api_key)

        self._llm = _build_validator_llm(self.provider, self.model, self.api_key)
        self._structured_llm = self._llm.with_structured_output(SonarFixResponse)

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(Path(__file__).parent / "prompts")),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=select_autoescape(["html", "xml"]),
        )

    # ------------------------------------------------------------------
    # Provider / model resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(provider: str, model: Optional[str]) -> str:
        defaults = {
            "openai":     "gpt-4o",
            "togetherai": "meta-llama/Llama-3-70b-chat-hf",
            "openrouter": "anthropic/claude-sonnet-4",
            "gemini":     "gemini-1.5-flash",
        }
        return model or defaults.get(provider.lower(), "gpt-4o")

    @staticmethod
    def _resolve_api_key(provider: str, api_key: Optional[str]) -> str:
        env_vars = {
            "openai":     "OPENAI_API_KEY",
            "togetherai": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "gemini":     "GEMINI_KEY",
        }
        resolved = api_key or os.getenv(env_vars.get(provider.lower(), ""), "")
        if not resolved:
            raise ValueError(
                f"API key not provided for provider '{provider}'. "
                f"Set the {env_vars.get(provider.lower())} environment variable."
            )
        return resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_fix(
        self,
        fix: FixSuggestion,
        issue: Union[SonarIssue, SonarSecurityIssue],
        file_content: str,
        context_lines: int = 20,
        new_error_msg: str = "",
        test_err: str = "",
    ) -> ValidationResult:
        """
        Validate a fix suggestion using a senior code reviewer persona.

        Returns ValidationResult with approval status and potential improvements.
        Uses LangChain with_structured_output — no raw SDK calls.
        """
        try:
            first_line = issue.first_line if issue.first_line is not None else 1
            last_line = issue.last_line if issue.last_line is not None else first_line

            context = self._extract_validation_context(
                file_content, first_line, last_line, context_lines
            )
            formatted_fix = self._format_code_blocks_for_validation(
                fix.fixed_code_blocks
            )
            prompt = self._create_validation_prompt(
                fix, issue, context, new_error_msg, formatted_fix,
                test_err=test_err,
            )

            validation_response = self._call_llm_validator(prompt)

            if not validation_response:
                logger.warning("Failed to validate fix for issue %s", issue.key)
                return ValidationResult(
                    status=ValidationStatus.NEEDS_REVIEW,
                    original_fix=fix,
                    explanation="Validation failed - manual review required",
                    confidence=0.0,
                )

            modified_fix = fix
            blocks = validation_response.FIXED_CODE_BLOCKS
            for each_block in blocks:
                if (
                    each_block.has_changes
                    and each_block.context is None
                    and each_block.change_type == ChangeType.FULL_CODE
                ):
                    matching_block = next(
                        (b for b in fix.fixed_code_blocks
                         if b.start_line == each_block.start_line),
                        None,
                    )
                    if matching_block:
                        each_block.context = matching_block.context
                    else:
                        logger.warning(
                            "No matching block for lines %s-%s",
                            each_block.start_line, each_block.end_line,
                        )

            modified_fix.fixed_code_blocks = blocks
            helper_code = validation_response.NEW_HELPER_CODE
            if isinstance(helper_code, str) and "".join(helper_code.split()):
                modified_fix.helper_code = helper_code
                modified_fix.placement_helper = validation_response.PLACEMENT.value

            return ValidationResult(
                status=ValidationStatus.MODIFIED,
                original_fix=fix,
                modified_fix=modified_fix,
                explanation=validation_response.EXPLANATION or "",
                confidence=validation_response.CONFIDENCE,
            )

        except Exception as exc:
            logger.error("Error validating fix for issue %s: %s", issue.key, exc, exc_info=True)
            return ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                original_fix=fix,
                explanation=f"Validation error: {exc}",
                confidence=0.0,
            )

    # ------------------------------------------------------------------
    # LLM call — LangChain only, no raw SDK
    # ------------------------------------------------------------------

    def _call_llm_validator(self, prompt: str) -> Optional[SonarFixResponse]:
        """
        Call the validator LLM using LangChain with_structured_output.

        All four providers (openai, togetherai, openrouter, gemini) use the same
        chain — provider differences are handled by _build_validator_llm().
        """
        system_prompt = (
            "You are a senior software engineer and security expert "
            "specializing in code review. Your reviews are thorough, "
            "critical, and focused on preventing bugs and security issues."
        )
        try:
            chain = ChatPromptTemplate.from_messages([
                ("system", "{system}"),
                ("human",  "{prompt}"),
            ]) | self._structured_llm

            result: SonarFixResponse = chain.invoke({
                "system": system_prompt,
                "prompt": prompt,
            })
            return result
        except Exception as exc:
            logger.error(
                "Validator LLM call failed [provider=%s model=%s]: %s",
                self.provider, self.model, exc, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Prompt / context helpers (unchanged logic)
    # ------------------------------------------------------------------

    def _format_code_blocks_for_validation(self, code_blocks: List[CodeBlock]) -> str:
        formatted_parts = []
        for idx, block in enumerate(code_blocks, 1):
            formatted_parts.append(_format_block_header(idx, block))
            if block.change_type == ChangeType.FULL_CODE and block.context:
                formatted_parts.append(_format_full_code_content(block))
            elif block.change_type == ChangeType.DIFF and block.changes:
                formatted_parts.append(_format_diff_content(block))
            elif block.change_type == ChangeType.SEARCH_REPLACE and block.replacements:
                formatted_parts.append(_format_search_replace_content(block))
            if idx < len(code_blocks):
                formatted_parts.append("\n" + "-" * 60 + "\n")
        return "".join(formatted_parts)

    def _extract_validation_context(
        self, file_content: str, first_line: int, last_line: int, context_lines: int
    ) -> Dict[str, Any]:
        lines = file_content.split("\n")
        first_idx = first_line - 1
        last_idx = last_line - 1
        start_idx = max(0, first_idx - context_lines)
        end_idx = min(len(lines), last_idx + context_lines + 1)
        return {
            "full_context":  "\n".join(lines[start_idx:end_idx]),
            "problem_lines": "\n".join(lines[first_idx : last_idx + 1]),
            "start_line":    start_idx + 1,
            "end_line":      end_idx,
            "issue_start":   first_line,
            "issue_end":     last_line,
        }

    def _create_validation_prompt(
        self,
        fix: FixSuggestion,
        issue: Union[SonarIssue, SonarSecurityIssue],
        context: Dict[str, Any],
        new_error: str,
        formatted_fix: str,
        test_err: str = "",
    ) -> str:
        template = self.jinja_env.get_template("python/validator.j2")
        prompt = template.render(
            fix=fix,
            issue=issue,
            severity=getattr(issue, "severity", "N/A"),
            issue_type=getattr(issue, "type", "N/A"),
            context=context,
            error_message=new_error,
            test_error=test_err,
            formatted_fix=formatted_fix,
        )
        return prompt.strip()
