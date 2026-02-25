"""Fix validation agent for reviewing and improving LLM-generated fixes."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, cast
from enum import Enum
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
import logging
from pydantic import ValidationError
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


class ValidationStatus(str, Enum):
    """Status of fix validation."""

    APPROVED = "APPROVED"  # Fix is good as-is
    MODIFIED = "MODIFIED"  # Fix was improved/corrected
    REJECTED = "REJECTED"  # Fix is unsafe or incorrect
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Requires manual review


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


def _format_block_header(idx: int, block: CodeBlock) -> str:
    """Format the header section for a code block."""
    header = (
        f"\n{'=' * 60}\n"
        f"Block {idx}: {block.block_name} (Lines {block.start_line}-{block.end_line})\n"
        f"Type: {block.block_type.value} | Change Type: {block.change_type.value}\n"
        f"Has Changes: {block.has_changes}\n{'=' * 60}\n"
    )
    return header


def _format_full_code_content(block: CodeBlock) -> str:
    """Format a FULL_CODE block as a fenced code block."""
    return f"```python\n{block.context}\n```\n"


def _format_diff_content(block: CodeBlock) -> str:
    """Format a DIFF block with line-by-line changes."""
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
    """Format a SEARCH_REPLACE block with replacement operations."""
    parts = ["Search/Replace Operations:\n"]
    for op_idx, repl in enumerate(block.replacements or [], 1):
        regex_marker = " (REGEX)" if repl.is_regex else ""
        count_info = f" (count: {repl.count})" if repl.count else " (all occurrences)"
        parts.append(f"\n  Operation {op_idx}{regex_marker}{count_info}:\n")
        parts.append(f"    Search:  {repr(repl.search)}\n")
        parts.append(f"    Replace: {repr(repl.replace)}\n")
    parts.append("\n")
    return "".join(parts)


class FixValidator:
    """
    Senior code reviewer agent that validates and potentially improves LLM-generated fixes.

    Acts as a second pair of eyes to catch:
    - Logic errors in fixes
    - Security issues
    - Edge cases not handled
    - Better alternative solutions
    - Breaking changes
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
        self.model: str = ""

        self.api_key: Optional[str] = None

        self.client: Any = None
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(Path(__file__).parent / "prompts")),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=select_autoescape(["html", "xml"]),
        )
        self._setup_provider(model, api_key)

    def _setup_provider(self, model: Optional[str], api_key: Optional[str]) -> None:
        if self.provider == "togetherai":
            self._setup_together_ai(model, api_key)
        elif self.provider == "openai":
            self._setup_open_ai(model, api_key)
        elif self.provider == "gemini":
            self._setup_gemini(model, api_key)
        elif self.provider == "openrouter":
            self._setup_openrouter(model, api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Use 'openai', 'gemini', 'togetherai', or 'openrouter'."
            )

    def _setup_together_ai(self, model: Optional[str], api_key: Optional[str]) -> None:
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

    def _setup_open_ai(self, model: Optional[str], api_key: Optional[str]) -> None:
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

    def _setup_gemini(self, model: Optional[str], api_key: Optional[str]) -> None:
        if not HAS_GEMINI:
            raise ImportError(
                "Gemini library not installed. Install with: pip install google-genai"
            )
        self.model = model or "gemini-1.5-flash"
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=self.api_key)

    def _setup_openrouter(self, model: Optional[str], api_key: Optional[str]) -> None:
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

    def validate_fix(
        self,
        fix: FixSuggestion,
        issue: Union[SonarIssue, SonarSecurityIssue],
        file_content: str,
        context_lines: int = 20,
        new_error_msg: str = "",
    ) -> ValidationResult:
        """
        Validate a fix suggestion using a senior code reviewer persona.

        Args:
            fix: The fix suggestion to validate
            issue: The original SonarCloud issue
            file_content: Complete content of the file being fixed
            context_lines: Number of lines of context to provide

        Returns:
            ValidationResult with approval status and potential improvements
        """

        try:
            # Handle None values for line numbers
            first_line = issue.first_line if issue.first_line is not None else 1
            last_line = issue.last_line if issue.last_line is not None else first_line

            # Extract broader context around the fix
            context = self._extract_validation_context(
                file_content, first_line, last_line, context_lines
            )

            formatted_fix = self._format_code_blocks_for_validation(
                fix.fixed_code_blocks
            )

            # Generate validation prompt
            prompt = self._create_validation_prompt(
                fix, issue, context, new_error_msg, formatted_fix
            )

            # Call LLM for validation
            validation_response = self._call_llm_validator(prompt)

            if not validation_response:
                logger.warning(f"Failed to validate fix for issue {issue.key}")
                return ValidationResult(
                    status=ValidationStatus.NEEDS_REVIEW,
                    original_fix=fix,
                    explanation="Validation failed - manual review required",
                    confidence=0.0,
                )
            validation_response = self._normalize_code_blocks(validation_response,fix)


            modified_fix = fix
            blocks = validation_response.FIXED_CODE_BLOCKS

            for each_block in blocks:
                if (
                    each_block.has_changes
                    and each_block.context is None
                    and each_block.change_type == ChangeType.FULL_CODE
                ):
                    matching_block = next(
                        (
                            block
                            for block in fix.fixed_code_blocks
                            if block.start_line == each_block.start_line
                        ),
                        None,
                    )

                    if matching_block:
                        each_block.context = matching_block.context

                    else:
                        # Fallback or error handling
                        print(
                            f"⚠️ Warning: No matching block found for lines {each_block.start_line}-{each_block.end_line}"
                        )

            modified_fix.fixed_code_blocks = blocks
            helper_code = validation_response.NEW_HELPER_CODE

            no_whitespace = "".join(helper_code.split())

            if isinstance(helper_code, str) and no_whitespace:
                modified_fix.helper_code = helper_code
                modified_fix.placement_helper = validation_response.PLACEMENT.value

            return ValidationResult(
                status=ValidationStatus.MODIFIED,
                original_fix=fix,
                modified_fix=modified_fix,
                explanation=validation_response.EXPLANATION or "",
                confidence=validation_response.CONFIDENCE,
            )

        except Exception as e:
            logger.error(
                f"Error validating fix for issue {issue.key}: {e}", exc_info=True
            )
            return ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                original_fix=fix,
                explanation=f"Validation error: {str(e)}",
                confidence=0.0,
            )

    def _format_code_blocks_for_validation(self, code_blocks: List[CodeBlock]) -> str:
        """
        Format code blocks into a readable string for validation.

        Args:
            code_blocks: List of CodeBlock objects

        Returns:
            Formatted string representation of all fixes
        """
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
        """
        Extract broader context for validation.

        Args:
            file_content: Complete file content
            first_line: First line of the issue
            last_line: Last line of the issue
            context_lines: Number of context lines

        Returns:
            Dictionary with context information
        """

        lines = file_content.split("\n")

        # Convert to 0-indexed
        first_idx = first_line - 1
        last_idx = last_line - 1

        # Calculate boundaries with broader context for validation
        start_idx = max(0, first_idx - context_lines)
        end_idx = min(len(lines), last_idx + context_lines + 1)

        context_text = "\n".join(lines[start_idx:end_idx])
        problem_lines = "\n".join(lines[first_idx : last_idx + 1])

        return {
            "full_context": context_text,
            "problem_lines": problem_lines,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "issue_start": first_line,
            "issue_end": last_line,
        }

    def _create_validation_prompt(
        self,
        fix: FixSuggestion,
        issue: Union[SonarIssue, SonarSecurityIssue],
        context: Dict[str, Any],
        new_error: str,
        formatted_fix: str,
    ) -> str:
        """Create a prompt for fix validation."""
        severity = getattr(issue, "severity", "N/A")
        issue_type = getattr(issue, "type", "N/A")
        context_dic = {
            "fix": fix,
            "issue": issue,
            "severity": severity,
            "issue_type": issue_type,
            "context": context,
            "error_message": new_error,
            "formatted_fix": formatted_fix,
        }
        template = self.jinja_env.get_template("python/validator.j2")
        # Render enhanced content
        prompt = template.render(**context_dic)
        return prompt.strip()

    def _normalize_code_blocks(self, response: SonarFixResponse, fix: FixSuggestion) -> SonarFixResponse:
        """Normalize code blocks in the response."""

        blocks = response.FIXED_CODE_BLOCKS
        if not isinstance(blocks, list):
            return response

        normalized = []
        for block in blocks:

            context = block.context
            changes = block.changes
            replacements = block.replacements



            has_context = bool(context)
            has_changes = bool(changes)
            has_replacements = bool(replacements)

            # Auto-detect correct change_type based on which field has data
            if has_context and not has_changes and not has_replacements:
                block.change_type = "FULL_CODE"
                block.changes = None
                block.replacements = None

            elif has_changes and not has_context and not has_replacements:
                block.change_type = "DIFF"
                block.context = ""
                block.replacements = None

            elif has_replacements and not has_context and not has_changes:
                block.change_type = "SEARCH_REPLACE"
                block.context = ""
                block.changes = None

            elif has_context and (has_changes or has_replacements):
                # LLM filled multiple fields — context wins (most complete data)
                block.change_type = "FULL_CODE"
                block.changes = None
                block.replacements = None

            elif not has_context and not has_changes and not has_replacements:
                # Nothing filled — default to FULL_CODE with empty context
                # Pydantic will catch this as a validation error
                block.change_type = "FULL_CODE"

            else:
                # Fallback: trust declared change_type if present, else FULL_CODE
                declared = block.get("change_type", "FULL_CODE")
                block.change_type = declared

            normalized.append(block)

            # --- Step 2: Sort by block_name, FULL_CODE first within each group ---
        change_type_order = {
            ChangeType.FULL_CODE: 0,
            ChangeType.DIFF: 1,
            ChangeType.SEARCH_REPLACE: 2,
        }
        normalized.sort(
            key=lambda b: (
                b.block_name,
                change_type_order.get(b.change_type, 99),
            )
        )

        # --- Step 3: Override start_line/end_line from FULL_CODE to siblings ---
        # Build a map: block_name -> (start_line, end_line) from FULL_CODE blocks
        full_code_line_map: dict[str, tuple[int, int]] = {}
        for block in normalized:
            if block.change_type == ChangeType.FULL_CODE and block.has_changes:
                full_code_line_map[block.block_name] = (block.start_line, block.end_line)

        # Apply authoritative line numbers to all blocks sharing the same name
        for block in normalized:
            if block.block_name in full_code_line_map:
                authoritative_start, authoritative_end = full_code_line_map[block.block_name]
                if (
                        block.start_line != authoritative_start
                        or block.end_line != authoritative_end
                ):
                    block.start_line = authoritative_start
                    block.end_line = authoritative_end
                    block.line_number = authoritative_start

        response.FIXED_CODE_BLOCKS = normalized
        return response

    def _call_llm_validator(self, prompt: str) -> Optional[SonarFixResponse]:
        """Call LLM for validation."""
        try:
            if not self.client:
                logger.error(f"{self.provider} client not properly initialized")
                return None

            if self.provider == "openai":
                return self._call_openai_validator(prompt)

            if self.provider == "gemini":
                return self._call_gemini_validator(prompt)

            if self.provider in ("togetherai", "openrouter"):
                return self._call_openai_compatible_validator(prompt)

            return None

        except Exception as e:
            logger.error(f"Error calling validator LLM: {e}", exc_info=True)
            return None

    def _call_gemini_validator(self, prompt: str) -> Optional[SonarFixResponse]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SonarFixResponse,
            ),
        )
        return cast(Optional[SonarFixResponse], response.parsed)

    def _call_openai_validator(self, prompt: str) -> Optional[SonarFixResponse]:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer and security expert "
                        "specializing in code review. Your reviews are thorough, "
                        "critical, and focused on preventing bugs and security issues."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text_format=SonarFixResponse,
        )

        return cast(Optional[SonarFixResponse], response.output_parsed)

    def _call_openai_compatible_validator(
        self, prompt: str
    ) -> Optional[SonarFixResponse]:
        """Call an OpenAI-compatible validator (used by TogetherAI and OpenRouter)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer and security expert "
                        "specializing in code review."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=8000,
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sonar_fix_response",
                    "schema": SonarFixResponse.model_json_schema(),
                    "strict": True,
                },
            },
        )
        response_json = response.choices[0].message.content

        try:
            data = json.loads(response_json)
            return SonarFixResponse(**data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}")

    # Aliases for backward compatibility and test clarity
    _call_togetherai_validator = _call_openai_compatible_validator
    _call_openrouter_validator = _call_openai_compatible_validator
