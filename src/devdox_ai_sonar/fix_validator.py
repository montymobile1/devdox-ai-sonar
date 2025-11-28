"""Fix validation agent for reviewing and improving LLM-generated fixes."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import re

from .logging_config import setup_logging, get_logger
from .models import FixSuggestion, SonarIssue

setup_logging(level='DEBUG', log_file='demo.log')
logger = get_logger(__name__)

try:
    from together import AsyncTogether, Together
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
            validation_notes: str = "",
            concerns: List[str] = None,
            confidence: float = 0.0
    ):
        self.status = status
        self.original_fix = original_fix
        self.modified_fix = modified_fix or original_fix
        self.validation_notes = validation_notes
        self.concerns = concerns or []
        self.confidence = confidence

    @property
    def final_fix(self) -> FixSuggestion:
        """Get the final fix to apply (modified or original)."""
        return self.modified_fix if self.status == ValidationStatus.MODIFIED else self.original_fix

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
            min_confidence_threshold: float = 0.7
    ):
        """
        Initialize the fix validator.

        Args:
            provider: LLM provider ("openai" or "gemini")
            model: Model name (defaults to provider's default)
            api_key: API key (uses environment variables if not provided)
            min_confidence_threshold: Minimum confidence to approve a fix
        """
        self.provider = provider.lower()
        self.min_confidence_threshold = min_confidence_threshold
        if self.provider == "togetherai":
            if not HAS_TOGETHER:
                raise ImportError("Together AI library not installed. Install with: pip install together")
            self.model = model or "gpt-4o"
            self.api_key = api_key or os.getenv("TOGETHER_API_KEY")

            if not self.api_key:
                raise ValueError("Together API key not provided. Set TOGETHER_API_KEY environment variable.")

            self.client = Together(api_key=self.api_key)

        elif self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")

            self.model = model or "gpt-4o"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

            self.client = openai.OpenAI(api_key=self.api_key)

        elif self.provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError("Gemini library not installed. Install with: pip install google-genai")

            self.model = model or "claude-3-5-sonnet-20241022"
            self.api_key = api_key
            if not self.api_key:
                raise ValueError("Gemini API key not provided. Set GEMINI_KEY environment variable.")

            self.client = genai.Client(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini' or 'togetherai'.")

    def validate_fix(
            self,
            fix: FixSuggestion,
            issue: SonarIssue,
            file_content: str,
            context_lines: int = 20
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
            # Extract broader context around the fix
            context = self._extract_validation_context(
                file_content,
                issue.first_line,
                issue.last_line,
                context_lines
            )

            # Generate validation prompt
            prompt = self._create_validation_prompt(fix, issue, context)

            # Call LLM for validation
            validation_response = self._call_llm_validator(prompt)

            if not validation_response:
                logger.warning(f"Failed to validate fix for issue {issue.key}")
                return ValidationResult(
                    status=ValidationStatus.NEEDS_REVIEW,
                    original_fix=fix,
                    validation_notes="Validation failed - manual review required",
                    confidence=0.0
                )

            # Parse validation response
            result = self._parse_validation_response(validation_response, fix, issue)

            return result

        except Exception as e:
            logger.error(f"Error validating fix for issue {issue.key}: {e}", exc_info=True)
            return ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                original_fix=fix,
                validation_notes=f"Validation error: {str(e)}",
                confidence=0.0
            )

    def validate_fixes_batch(
            self,
            fixes: List[Tuple[FixSuggestion, SonarIssue, str]],
            stop_on_rejection: bool = False
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
                logger.warning(f"Stopping batch validation due to rejection of {issue.key}")
                break

        return results

    def _extract_validation_context(
            self,
            file_content: str,
            first_line: int,
            last_line: int,
            context_lines: int
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
        lines = file_content.split('\n')

        # Convert to 0-indexed
        first_idx = first_line - 1
        last_idx = last_line - 1

        # Calculate boundaries with broader context for validation
        start_idx = max(0, first_idx - context_lines)
        end_idx = min(len(lines), last_idx + context_lines + 1)

        context_text = '\n'.join(lines[start_idx:end_idx])
        problem_lines = '\n'.join(lines[first_idx:last_idx + 1])

        return {
            "full_context": context_text,
            "problem_lines": problem_lines,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "issue_start": first_line,
            "issue_end": last_line
        }

    def _create_validation_prompt(
            self,
            fix: FixSuggestion,
            issue: SonarIssue,
            context: Dict[str, Any]
    ) -> str:
        """Create a prompt for fix validation."""

        prompt = f"""You are a senior software engineer conducting a critical code review of an AI-generated fix.
Your job is to validate this fix with extreme scrutiny, checking for:

1. **Correctness**: Does the fix actually solve the issue without introducing bugs?
2. **Security**: Are there any security implications?
3. **Edge Cases**: Does it handle all edge cases?
4. **Best Practices**: Does it follow language-specific best practices?
5. **Breaking Changes**: Will this break existing functionality?
6. **Side Effects**: Are there unintended consequences?

**Original SonarCloud Issue:**
- Rule: {issue.rule}
- Message: {issue.message}
- Severity: {issue.severity}
- Type: {issue.type}
- Lines: {issue.first_line}-{issue.last_line}

**Proposed Fix Details:**
- Confidence: {fix.confidence:.2f}
- Model: {fix.llm_model}
- Explanation: {fix.explanation}

**Original Code:**
```
{fix.original_code}
```

**Proposed Fixed Code:**
```
{fix.fixed_code}
```

**Broader Code Context (lines {context['start_line']}-{context['end_line']}):**
```
{context['full_context']}
```

**Your Task:**
Critically review this fix and provide your assessment.

**Response Format:**
STATUS: [APPROVED|MODIFIED|REJECTED|NEEDS_REVIEW]

CONFIDENCE: [0.0-1.0]

VALIDATION_NOTES:
[Your detailed analysis of the fix]

CONCERNS:
- [List any concerns, one per line, or "None" if no concerns]
- [Use this format even if approving]

IMPROVED_FIX: (only if STATUS is MODIFIED)
```
[Your improved version of the fix]
```

IMPROVED_EXPLANATION: (only if STATUS is MODIFIED)
[Explanation of what you improved and why]

**Critical Guidelines:**
- Be skeptical - catching one bug is worth rejecting ten mediocre fixes
- Consider the broader codebase context
- Think about maintainability and readability
- If uncertain, mark as NEEDS_REVIEW
- MODIFIED status requires an improved fix that's demonstrably better
- APPROVED status requires high confidence (≥0.8) that the fix is correct
"""

        return prompt

    def _call_llm_validator(self, prompt: str) -> Optional[str]:
        """Call LLM for validation."""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior software engineer and security expert specializing in code review. Your reviews are thorough, critical, and focused on preventing bugs and security issues."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                return response.choices[0].message.content

            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return response.text

        except Exception as e:
            logger.error(f"Error calling validator LLM: {e}", exc_info=True)
            return None

    def _parse_validation_response(
            self,
            response_text: str,
            original_fix: FixSuggestion,
            issue: SonarIssue
    ) -> ValidationResult:
        """Parse the validation response from LLM."""



        try:
            # Extract status

            status_match = re.search(
                r'STATUS:\s*(APPROVED|MODIFIED|REJECTED|NEEDS_REVIEW)',
                response_text,
                re.IGNORECASE
            )
            status_str = status_match.group(1).upper() if status_match else "NEEDS_REVIEW"
            status = ValidationStatus(status_str)

            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', response_text)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            confidence = max(0.0, min(1.0, confidence))

            # Extract validation notes
            notes_match = re.search(
                r'VALIDATION_NOTES:\s*(.*?)(?=CONCERNS:|IMPROVED_FIX:|$)',
                response_text,
                re.DOTALL
            )
            validation_notes = notes_match.group(1).strip() if notes_match else ""

            # Extract concerns
            concerns_match = re.search(
                r'CONCERNS:\s*(.*?)(?=IMPROVED_FIX:|IMPROVED_EXPLANATION:|$)',
                response_text,
                re.DOTALL
            )
            concerns_text = concerns_match.group(1).strip() if concerns_match else ""
            concerns = [
                line.strip('- ').strip()
                for line in concerns_text.split('\n')
                if line.strip() and line.strip() != 'None'
            ]

            # Handle MODIFIED status - extract improved fix
            modified_fix = None
            if status == ValidationStatus.MODIFIED:
                improved_code_pattern = r'IMPROVED_FIX:\s*```[a-zA-Z]{0,20}\s*((?:[^`]|`(?!``))*?)\s*```'
                improved_code_match = re.search(
                    improved_code_pattern,
                    response_text,
                    re.DOTALL
                )

                improved_explanation_match = re.search(
                    r'IMPROVED_EXPLANATION:\s*(.*?)(?=$)',
                    response_text,
                    re.DOTALL
                )

                if improved_code_match:
                    improved_code = improved_code_match.group(1).strip()
                    improved_explanation = improved_explanation_match.group(
                        1).strip() if improved_explanation_match else validation_notes

                    # Create modified fix suggestion
                    modified_fix = FixSuggestion(
                        issue_key=original_fix.issue_key,
                        original_code=original_fix.original_code,
                        fixed_code=improved_code,
                        explanation=f"{original_fix.explanation}\n\nValidator Improvement: {improved_explanation}",
                        confidence=confidence,
                        llm_model=f"{original_fix.llm_model} + {self.model} (validated)",
                        rule_description=original_fix.rule_description,
                        file_path=original_fix.file_path,
                        line_number=original_fix.line_number,
                        last_line_number=original_fix.last_line_number
                    )
                else:
                    # If no improved fix provided, treat as NEEDS_REVIEW
                    logger.warning("MODIFIED status but no improved fix found")
                    status = ValidationStatus.NEEDS_REVIEW

            # Apply confidence threshold
            if status == ValidationStatus.APPROVED and confidence < self.min_confidence_threshold:
                status = ValidationStatus.NEEDS_REVIEW
                validation_notes += f"\n\nNote: Confidence {confidence:.2f} is below required threshold {self.min_confidence_threshold}"

            return ValidationResult(
                status=status,
                original_fix=original_fix,
                modified_fix=modified_fix,
                validation_notes=validation_notes,
                concerns=concerns,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error parsing validation response: {e}", exc_info=True)
            return ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                original_fix=original_fix,
                validation_notes=f"Failed to parse validation response: {str(e)}",
                confidence=0.0
            )


def validate_fixes_with_agent(
        fixes: List[FixSuggestion],
        issues: List[SonarIssue],
        project_path: Path,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        min_confidence: float = 0.7
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
        min_confidence_threshold=min_confidence
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
                    confidence=0.0
                )
            )
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
                    confidence=0.0
                )
            )

    return results