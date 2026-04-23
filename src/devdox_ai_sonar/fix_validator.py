"""Fix validation agent for reviewing and improving LLM-generated fixes.

The validator is a thin wrapper around ``openhands.sdk.LLM``: it renders
the validator Jinja prompt, sends a single JSON-mode completion request
to whichever provider the user's :class:`LLMProfile` points at, and
parses the response into a :class:`SonarFixResponse`. All routing --
provider, model, base_url, auth -- is handled by OpenHands/litellm; this
module never imports any provider SDK directly.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape
from openhands.sdk import LLM, Message, TextContent
from pydantic import ValidationError

from devdox_ai_sonar.llm.profile import LLMProfile
from devdox_ai_sonar.models.sonar import (
    ChangeAction,
    ChangeType,
    CodeBlock,
    FixSuggestion,
    SonarFixResponse,
    SonarIssue,
    SonarSecurityIssue,
)

logger = logging.getLogger(__name__)


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
        profile: LLMProfile,
        min_confidence_threshold: float = 0.7,
    ):
        """Initialise the validator from an :class:`LLMProfile`.

        Args:
            profile: The LLM configuration to use for validation requests.
                Typically passed down from the same profile the parent
                ``LLMFixer`` was constructed with, so fixer and validator
                share a provider.
            min_confidence_threshold: Validated fixes below this threshold
                are marked :attr:`ValidationStatus.NEEDS_REVIEW`.
        """
        self.profile = profile
        # Convenience aliases so existing callers (and downstream code
        # that reads ``validator.model`` / ``validator.api_key`` /
        # ``validator.provider``) keep working.
        self.model = profile.model
        self.api_key = profile.api_key
        self.provider = profile.family()
        self.min_confidence_threshold = min_confidence_threshold
        self.client: LLM = LLM(
            model=profile.model,
            api_key=profile.api_key or None,
            base_url=profile.base_url,
        )
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(Path(__file__).parent / "prompts")),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=select_autoescape(["html", "xml"]),
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

            formatted_fix, start_line, end_line = (
                self._format_code_blocks_for_validation(fix.fixed_code_blocks)
            )
            if formatted_fix != "":
                context["start_line"] = start_line
                context["end_line"] = end_line

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
            modified_fix = fix
            blocks = validation_response.FIXED_CODE_BLOCKS
            self._backfill_missing_context(blocks, fix.fixed_code_blocks)
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

    @staticmethod
    def _backfill_missing_context(
        blocks: List[CodeBlock], original_blocks: List[CodeBlock]
    ) -> None:
        """Copy context from original blocks into validated blocks that lost it.

        When the validator returns FULL_CODE blocks with has_changes=True but
        no context, we look up the original block by start_line and copy its
        context across.  Mutates *blocks* in place.
        """
        for each_block in blocks:
            if not (
                each_block.has_changes
                and each_block.context is None
                and each_block.change_type == ChangeType.FULL_CODE
            ):
                continue

            matching_block = next(
                (
                    block
                    for block in original_blocks
                    if block.start_line == each_block.start_line
                ),
                None,
            )

            if matching_block:
                each_block.context = matching_block.context
            else:
                print(
                    f"Warning: No matching block found for lines "
                    f"{each_block.start_line}-{each_block.end_line}"
                )

    def _format_code_blocks_for_validation(
        self, code_blocks: List[CodeBlock]
    ) -> Tuple[str, int, int]:
        """
        Format code blocks into a readable string for validation.

        Args:
            code_blocks: List of CodeBlock objects

        Returns:
            Formatted string representation of all fixes
        """
        formatted_parts = []
        if not code_blocks:
            return "", 0, 0

        start_line = code_blocks[0].start_line
        end_line = code_blocks[0].end_line
        for idx, block in enumerate(code_blocks, 1):
            if block.start_line < start_line:
                start_line = block.start_line

            if block.end_line > end_line:
                end_line = block.end_line
            formatted_parts.append(_format_block_header(idx, block))

            if block.change_type == ChangeType.FULL_CODE and block.context:
                formatted_parts.append(_format_full_code_content(block))
            elif block.change_type == ChangeType.DIFF and block.changes:
                formatted_parts.append(_format_diff_content(block))
            elif block.change_type == ChangeType.SEARCH_REPLACE and block.replacements:
                formatted_parts.append(_format_search_replace_content(block))

            if idx < len(code_blocks):
                formatted_parts.append("\n" + "-" * 60 + "\n")

        return "".join(formatted_parts), start_line, end_line

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

    _VALIDATOR_SYSTEM_PROMPT = (
        "You are a senior software engineer and security expert "
        "specializing in code review. Your reviews are thorough, "
        "critical, and focused on preventing bugs and security issues."
    )

    def _call_llm_validator(self, prompt: str) -> Optional[SonarFixResponse]:
        """Send the validator prompt through OpenHands and parse the response.

        Uses a JSON-mode completion so litellm asks the downstream provider
        to emit structured output. The response content is then parsed
        against :class:`SonarFixResponse`; malformed JSON and schema
        mismatches are logged and re-raised as ``ValueError`` so the
        caller can fall back to regex extraction or mark the fix as
        needing manual review.
        """
        if not self.client:
            logger.error("Validator client not initialised")
            return None

        try:
            response = self.client.completion(
                messages=[
                    Message(
                        role="system",
                        content=[TextContent(text=self._VALIDATOR_SYSTEM_PROMPT)],
                    ),
                    Message(role="user", content=[TextContent(text=prompt)]),
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("Validator LLM call failed for model=%s", self.model)
            return None

        content = _extract_completion_content(response)
        if content is None:
            logger.warning("Validator response had no parseable content")
            return None

        return _parse_validator_response(content, model=self.model)

    @staticmethod
    def _try_extract_json(text: str) -> Optional[str]:
        """Strip markdown fences and whitespace from LLM response."""
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return cleaned.strip()
        return cleaned


def _extract_completion_content(response: Any) -> Optional[str]:
    """Pull the text content out of an LLM completion response.

    Handles two shapes:

    - ``openhands.sdk.LLMResponse``: ``.message.content`` is a sequence
      of ``TextContent``/``ImageContent`` blocks; we concatenate the
      text from any ``TextContent`` entries.
    - Raw litellm ``ModelResponse``: ``.choices[0].message.content`` is
      a plain string. Supported so that tests that mock the raw
      response shape keep working without needing OpenHands' own types.
    """
    # Try OpenHands LLMResponse first: .message.content is a list of blocks.
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)

    # Fallback: litellm/OpenAI-compatible ModelResponse shape.
    try:
        choices = response.choices
        if not choices:
            return None
        litellm_message = choices[0].message
        litellm_content = getattr(litellm_message, "content", None)
        return litellm_content if isinstance(litellm_content, str) else None
    except (AttributeError, IndexError, TypeError):
        logger.exception("Could not extract content from validator response")
        return None


def _parse_validator_response(
    response_json: str, *, model: str
) -> Optional[SonarFixResponse]:
    """Parse a JSON-mode validator response into a :class:`SonarFixResponse`.

    Raises:
        ValueError: On malformed JSON or schema mismatch. Callers are
            expected to catch this and fall back to regex extraction or
            mark the fix as needing review.
    """
    try:
        data = json.loads(response_json)
        return SonarFixResponse(**data)
    except json.JSONDecodeError as exc:
        cleaned = FixValidator._try_extract_json(response_json)
        if cleaned and cleaned != response_json:
            try:
                data = json.loads(cleaned)
                return SonarFixResponse(**data)
            except (json.JSONDecodeError, ValidationError):
                pass

        response_len = len(response_json) if response_json else 0
        starts_with_fence = (
            response_json.startswith("```") if response_json else False
        )
        is_truncated = (
            not response_json.rstrip().endswith("}") if response_json else False
        )
        logger.warning(
            "Malformed JSON from validator (model=%s) at position %d of %d "
            "chars (truncated=%s, markdown_wrapped=%s): %s",
            model,
            exc.pos,
            response_len,
            is_truncated,
            starts_with_fence,
            exc.msg,
        )
        logger.debug("Raw validator response: %s", response_json)
        raise ValueError("LLM returned malformed JSON response") from exc
    except ValidationError as exc:
        logger.warning(
            "Schema validation failed for validator response (model=%s): "
            "%d error(s)",
            model,
            exc.error_count(),
        )
        logger.debug("Validation errors: %s", exc.errors())
        raise ValueError("LLM response did not match expected schema") from exc
