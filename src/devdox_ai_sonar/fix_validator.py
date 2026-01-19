"""Fix validation agent for reviewing and improving LLM-generated fixes."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import re
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
import logging

from devdox_ai_sonar.models.sonar import SonarIssue, SonarSecurityIssue, FixSuggestion

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
            loader=FileSystemLoader(str( Path(__file__).parent / "prompts")),
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
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Use 'openai' or 'gemini' or 'togetherai'."
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

    def validate_fix(
        self,
        fix: FixSuggestion,
        issue: Union[SonarIssue, SonarSecurityIssue],
        file_content: str,
        context_lines: int = 20,
        new_error_msg:str=""
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


            # Generate validation prompt
            prompt = self._create_validation_prompt(fix, issue, context, new_error_msg)

            # Call LLM for validation
            validation_response = self._call_llm_validator(prompt)
            if not validation_response:
                logger.warning(f"Failed to validate fix for issue {issue.key}")
                return ValidationResult(
                    status=ValidationStatus.NEEDS_REVIEW,
                    original_fix=fix,

                    validation_notes="Validation failed - manual review required",
                    confidence=0.0,
                )

            # Parse validation response
            result = self._parse_validation_response(validation_response, fix)

            return result

        except Exception as e:
            logger.error(
                f"Error validating fix for issue {issue.key}: {e}", exc_info=True
            )
            return ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                original_fix=fix,
                validation_notes=f"Validation error: {str(e)}",
                confidence=0.0,
            )

    def validate_fixes_batch(
        self,
        fixes: List[Tuple[FixSuggestion, SonarIssue, str]],
        stop_on_rejection: bool = False,
    ) -> List[ValidationResult]:
        """
        Validate multiple fixes in batch.

        Args:
            fixes: List of tuples (fix, issue, file_content)
            stop_on_rejection: Stop validation if a fix is rejected

        Returns:
            List of ValidationResult objects
        """
        results = []

        for fix, issue, file_content in fixes:
            result = self.validate_fix(fix, issue, file_content)
            results.append(result)

            if stop_on_rejection and result.status == ValidationStatus.REJECTED:
                logger.warning(
                    f"Stopping batch validation due to rejection of {issue.key}"
                )
                break

        return results

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
        new_error: str
    ) -> str:
        """Create a prompt for fix validation."""
        severity = getattr(issue, "severity", "N/A")
        issue_type = getattr(issue, "type", "N/A")
        context_dic = {
            "fix": fix,
            "issue": issue,
            "severity":severity,
            "issue_type":issue_type,
            "context": context,
            "new_error": new_error,

        }
        template = self.jinja_env.get_template("python/validator_today_2026_01_19.j2")
        # Render enhanced content
        prompt = template.render(**context_dic)

        return prompt.strip()



    def _call_llm_validator(self, prompt: str) -> Optional[str]:
        """Call LLM for validation."""
        try:
            if not self.client:
                logger.error(f"{self.provider} client not properly initialized")
                return None

            if self.provider == "openai":
                return self._call_openai_validator(prompt)

            if self.provider == "gemini":
                return self._call_gemini_validator(prompt)

            if self.provider == "togetherai":
                return self._call_togetherai_validator(prompt)

            return None

        except Exception as e:
            logger.error(f"Error calling validator LLM: {e}", exc_info=True)
            return None

    def _call_gemini_validator(self, prompt: str) -> Optional[str]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return str(response.text) if hasattr(response, "text") else None

    def _call_openai_validator(self, prompt: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
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
            temperature=0.1,
            max_tokens=2000,
        )

        return self._extract_openai_content(response)

    def _call_togetherai_validator(self, prompt: str) -> Optional[str]:
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
            temperature=0.1,
            # max_tokens=2000,
        )

        return self._extract_openai_content(response)

    def _extract_openai_content(self, response: Any) -> Optional[str]:
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content

            return str(content) if content else None
        return None

    def _extract_fields_from_parsed_json(
        self, fix_data: dict
    ) -> Optional[Dict[str, Any]]:
        """Extract and validate fields from parsed JSON."""

        # Extract with type conversion and defaults
        improved_fix = str(fix_data.get("IMPROVED_FIX", "")).strip()
        improved_helper_code = str(fix_data.get("NEW_HELPER_CODE", "")).strip()
        placement_helper = str(fix_data.get("PLACEMENT", "")).strip()
        improved_explanation = str(fix_data.get("IMPROVED_EXPLANATION", "")).strip()

        confidence = fix_data.get("CONFIDENCE", 0.5)

        # Convert confidence to float
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # Validate placement

        # Provide default explanation
        if not improved_explanation:
            improved_explanation = "Code fix applied"

        # Require fixed_code
        if not improved_fix:
            logger.error("❌ FIXED_SELECTION is empty")
            return None

        return {
            "improved_fix": improved_fix,
            "helper_code": improved_helper_code,
            "placement_helper":placement_helper,
            "improved_explanation": improved_explanation,
            "confidence": confidence,
        }

    def _parse_validation_response(
        self, response_text: str, original_fix: FixSuggestion
    ) -> ValidationResult:
        """Parse the validation response from LLM."""

        try:
            # Extract status
            cleaned_content = response_text.strip()

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

                    response =  self._extract_fields_from_parsed_json(fix_data)
                except json.JSONDecodeError:
                  response = {}


            modified_fix= original_fix
            modified_fix.fixed_code = response.get("improved_fix")
            helper_code = response.get("helper_code")
            no_whitespace = ''.join(helper_code.split())
            if isinstance(helper_code, str) and no_whitespace:

                modified_fix.helper_code= response.get("helper_code")
                modified_fix.placement_helper= response.get("placement_helper")

            return ValidationResult(
                status=ValidationStatus.MODIFIED,
                original_fix=original_fix,
                modified_fix=modified_fix,
                explanation= response.get("improved_explanation",""),
                confidence=response.get("confidence", 0.0)
            )

        except Exception as e:
            logger.error(f"Error parsing validation response: {e}", exc_info=True)
            return ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                original_fix=original_fix,
                modified_fix=original_fix,
                explanation=f"Failed to parse validation response: {str(e)}",
                confidence=0.0,
            )

    def validate_fixes_with_agent(
        self,
        fixes: List[FixSuggestion],
        issues: List[SonarIssue],
        project_path: Path,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        min_confidence: float = 0.7,
    ) -> List[ValidationResult]:
        """
        Convenience function to validate a list of fixes.

        Args:
            fixes: List of fix suggestions
            issues: List of corresponding SonarCloud issues
            project_path: Path to the project
            provider: LLM provider
            model: LLM model name
            api_key: API key
            min_confidence: Minimum confidence threshold

        Returns:
            List of validation results
        """
        validator = FixValidator(
            provider=provider,
            model=model,
            api_key=api_key,
            min_confidence_threshold=min_confidence,
        )

        results = []

        for fix, issue in zip(fixes, issues):
            # Read file content
            file_path = project_path / fix.file_path if fix.file_path else None

            if not file_path or not file_path.exists():
                logger.warning(f"File not found for fix {fix.issue_key}: {file_path}")
                results.append(
                    ValidationResult(
                        status=ValidationStatus.NEEDS_REVIEW,
                        original_fix=fix,
                        validation_notes="File not found for validation",
                        confidence=0.0,
                    )
                )
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

                result = validator.validate_fix(fix, issue, file_content)
                results.append(result)

            except Exception as e:
                logger.error(f"Error reading file for validation: {e}", exc_info=True)
                results.append(
                    ValidationResult(
                        status=ValidationStatus.NEEDS_REVIEW,
                        original_fix=fix,
                        validation_notes=f"Error reading file: {str(e)}",
                        confidence=0.0,
                    )
                )

        return results
