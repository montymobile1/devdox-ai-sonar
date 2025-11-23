"""LLM-powered code fixer for SonarCloud issues."""

import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from .fix_validator import FixValidator, ValidationStatus
from .models import SonarIssue, FixSuggestion, FixResult

from .logging_config import setup_logging, get_logger

setup_logging(level='DEBUG',log_file='demo.log')
logger = get_logger(__name__)


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


class LLMFixer:
    """LLM-powered code fixer for SonarCloud issues."""

    def __init__(
            self,
            provider: str = "openai",
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            context_lines: int = 10
    ):
        """
        Initialize the LLM fixer.

        Args:
            provider: LLM provider ("openai" or "gemini")
            model: Model name (defaults to provider's default)
            api_key: API key (uses environment variables if not provided)
            context_lines: Number of lines to include around the issue for context
        """
        self.provider = provider.lower()
        self.context_lines = context_lines

        if self.provider == "openai":
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
            self.api_key = api_key or os.getenv("GEMINI_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key not provided. Set GEMINI_KEY environment variable.")

            self.client = genai.Client(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'.")

    def generate_fix(self, issue: SonarIssue, project_path: Path) -> Optional[FixSuggestion]:
        """
        Generate a fix suggestion for a SonarCloud issue.

        Args:
            issue: SonarCloud issue to fix
            project_path: Path to the project root

        Returns:
            FixSuggestion object or None if fix cannot be generated
        """

        if not issue.file or not issue.first_line or not issue.last_line:
            logger.warning(f"Issue {issue.key} has no file or line number, skipping fix generation")
            return None

        # Read the file and extract context
        file_path = project_path / issue.file
        if not file_path.exists():
            logger.warning(f"File not found for issue {issue.key}: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Get context around the issue
            context = self._extract_context(lines, issue.first_line, issue.last_line, self.context_lines)
            # Generate fix using LLM
            fix_response = self._call_llm(issue, context, file_path.suffix)

            if fix_response:
                logger.info(f"Successfully generated fix for issue {issue.key} with confidence {fix_response['confidence']}")

                return FixSuggestion(
                    issue_key=issue.key,
                    original_code=context["context"],
                    fixed_code=fix_response["fixed_code"],
                    explanation=fix_response["explanation"],
                    confidence=fix_response["confidence"],
                    llm_model=self.model,
                    rule_description=fix_response.get("rule_description"),
                    file_path=str(file_path.relative_to(project_path)),  # Store relative path
                    line_number=issue.first_line,
                    last_line_number=issue.last_line
                )

        except Exception as e:
            logger.error(f"Error generating fix for issue {issue.key}: {e}", exc_info=True)

        return None

    def apply_fixes(
            self,
            fixes: List[FixSuggestion],
            project_path: Path,
            create_backup: bool = True,
            dry_run: bool = False
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
        result = FixResult(
            project_path=project_path,
            total_fixes_attempted=len(fixes)
        )

        # Create backup if requested
        if create_backup and not dry_run:
            backup_path = self._create_backup(project_path)
            result.backup_created = True
            result.backup_path = backup_path
            logger.info(f"Created backup at: {backup_path}")

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

                if self._apply_fixes_to_file(file_path, file_fixes, dry_run):
                    result.successful_fixes.extend(file_fixes)
                    logger.info(f"Successfully applied {len(file_fixes)} fixes to {file_path}")
                else:
                    result.failed_fixes.extend([
                        {"fix": fix, "error": f"Failed to apply fix to file {file_path}"}
                        for fix in file_fixes
                    ])
                    logger.error(f"Failed to apply fixes to {file_path}")
            except Exception as e:
                result.failed_fixes.extend([
                    {"fix": fix, "error": str(e)}
                    for fix in file_fixes
                ])
                logger.error(f"Error processing file {file_path_str}: {e}", exc_info=True)

        return result

    def _extract_context(self, lines: List[str], first_line_number: int, last_line_number: int, context_lines: int) -> Dict[str, Any]:
        """
        Extract context around a problematic line.

        Args:
            lines: All lines in the file
            first_line_number: First line number with the issue (1-indexed)
            last_line_number: Last line number with the issue (1-indexed)
            context_lines: Number of lines to include before/after

        Returns:
            Dictionary with context information
        """
        # Convert to 0-indexed
        first_line_idx = first_line_number - 1
        last_line_idx = last_line_number - 1

        # Calculate context boundaries
        start_idx = max(0, context_lines-first_line_idx  )
        print("start_idx ", start_idx)
        #end_idx = min(len(lines), first_line_idx + context_lines + 1)
        end_idx = min(len(lines), last_line_idx + context_lines + 1)

        context_lines_list = lines[start_idx:end_idx]
        context_lines_list= lines[first_line_idx:last_line_number]

        problem_line = lines[first_line_idx].rstrip() if first_line_idx < len(lines) else ""

        return {
            "context": "".join(context_lines_list),
            "problem_line": problem_line,
            "line_number": first_line_number,
            "start_line": start_idx + 1,
            "end_line": end_idx
        }

    def _call_llm(self, issue: SonarIssue, context: Dict[str, Any], file_extension: str) -> Optional[Dict[str, Any]]:
        """
        Call the LLM to generate a fix.

        Args:
            issue: SonarCloud issue
            context: Code context around the issue
            file_extension: File extension to determine language

        Returns:
            Dictionary with fix information or None
        """
        # Determine programming language
        language = self._get_language_from_extension(file_extension)

        # Prepare prompt
        prompt = self._create_fix_prompt(issue, context, language)

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are a senior software engineer specializing in code quality and SonarCloud rule compliance. Your job is to analyze code issues and provide precise fixes."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                return self._parse_openai_response(response)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return self._parse_anthropic_response(response)
            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return self._parse_gemini_response(response)
        except Exception as e:
            logger.error(f"Error calling {self.provider} LLM: {e}", exc_info=True)
            return None

    def _create_fix_prompt(self, issue: SonarIssue, context: Dict[str, Any], language: str) -> str:
        """Create a prompt for the LLM to generate a fix."""
        prompt = f"""
Analyze this {language} code issue and provide a fix:

**SonarCloud Issue Details:**
- Rule: {issue.rule}
- Message: {issue.message}
- Severity: {issue.severity}
- Type: {issue.type}
- First Line: {issue.first_line}
- Last Line: {issue.last_line}

**Code Context (lines {context['start_line']}-{context['end_line']}):**
```{language}
{context['context']}
```

**Problematic Line {issue.first_line}:**
```{language}
{context['problem_line']}
```

Please provide:
1. **Fixed Code**: The corrected version of the problematic line(s) using best practices and modern conventions based on the issue details and code context.
2. **Explanation**: Why this fix addresses the SonarCloud rule
3. **Confidence**: A score from 0.0 to 1.0 indicating your confidence in this fix

When the issue indicates removing commented-out code or dead code:
- Delete the code completely.
- Do NOT add replacement comments or documentation.
- Produce only the minimal required fix.

Format your response as:
FIXED_CODE:
```{language}
[your fixed code here]
```

EXPLANATION:
[explanation of the fix]

CONFIDENCE: [0.0-1.0]

Focus on:
- Following SonarCloud best practices
- Maintaining code functionality
- Using modern {language} conventions
- Ensuring the fix is minimal and targeted

"""
        return prompt

    def _parse_openai_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse OpenAI API response."""
        try:
            content = response.choices[0].message.content
            return self._extract_fix_from_response(content)
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}", exc_info=True)
            return None

    def _parse_anthropic_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse Anthropic API response."""
        try:
            content = response.content[0].text
            return self._extract_fix_from_response(content)
        except Exception as e:
            logger.error(f"Error parsing Anthropic response: {e}", exc_info=True)
            return None

    def _parse_gemini_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse Gemini API response."""
        try:
            content = response.text
            return self._extract_fix_from_response(content)
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}", exc_info=True)
            return None

    def _extract_fix_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract fix information from LLM response."""
        try:
            # Extract fixed code
            fixed_code_match = re.search(r'FIXED_CODE:\s*```[a-zA-Z]*\s*(.*?)\s*```', content, re.DOTALL)
            fixed_code = fixed_code_match.group(1).strip() if fixed_code_match else ""

            # Extract explanation
            explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=CONFIDENCE:|$)', content, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else ""

            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', content)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            if not fixed_code or not explanation:
                logger.warning("Failed to extract fixed code or explanation from LLM response")
                return None

            return {
                "fixed_code": fixed_code,
                "explanation": explanation,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error extracting fix from response: {e}", exc_info=True)
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
            ".java": "java",
            ".kt": "kotlin",
            ".scala": "scala",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift"
        }
        return language_map.get(extension, "text")

    def _create_backup(self, project_path: Path) -> Path:
        """Create a backup of the project."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = project_path.parent / f"{project_path.name}_backup_{timestamp}"
        shutil.copytree(project_path, backup_path)
        logger.info(f"Created project backup: {backup_path}")
        return backup_path

    def _get_file_from_fix(self, fix: FixSuggestion, project_path: Path) -> Optional[str]:
        """
        Get file path associated with a fix.

        Args:
            fix: FixSuggestion containing issue information
            project_path: Path to the project root

        Returns:
            Absolute file path as string, or None if not found
        """
        # First, try to use the stored file path from the fix
        if fix.file_path:
            file_path = project_path / fix.file_path
            if file_path.exists():
                return str(file_path)

        # Fallback: Try to extract file path from issue key if it follows SonarCloud pattern
        if ':' in fix.issue_key:
            parts = fix.issue_key.split(':')
            # Look for parts that might be file paths
            for part in parts:
                if '/' in part and (part.endswith('.py') or part.endswith('.js') or
                                    part.endswith('.java') or part.endswith('.ts') or
                                    part.endswith('.jsx') or part.endswith('.tsx') or
                                    part.endswith('.kt') or part.endswith('.scala')):
                    file_path = project_path / part
                    if file_path.exists():
                        return str(file_path)

        if fix.original_code and len(fix.original_code.strip()) > 10:
            # Search for files containing this code snippet
            matching_files = self._find_files_with_content(project_path, fix.original_code.strip())
            if matching_files:
                return str(matching_files[0])  # Return first match

        logger.warning(f"Could not determine file path for fix {fix.issue_key}")
        return None

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
        extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.kt', '.scala',
                      '.go', '.rs', '.cpp', '.c', '.cs', '.php', '.rb', '.swift'}

        try:
            # Search through source files
            for file_path in project_path.rglob("*"):
                if (file_path.is_file() and
                        file_path.suffix in extensions and
                        not any(part.startswith('.') for part in file_path.parts) and  # Skip hidden dirs
                        'node_modules' not in file_path.parts and
                        'venv' not in file_path.parts and
                        '__pycache__' not in file_path.parts):

                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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


    def _apply_fixes_to_file(self, file_path: Path, fixes: List[FixSuggestion], dry_run: bool) -> bool:
        """
        Apply fixes by REMOVING the block between line_number and last_line_number,
        and inserting the fixed_code block with correct indentation.
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would apply {len(fixes)} fixes to {file_path}")
            return True

        try:
            # Load file lines
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            applied_fixes = []
            skipped_fixes = []

            # Apply in reverse order to avoid line index shifting
            fixes_sorted = sorted(fixes, key=lambda f: f.line_number or 0, reverse=True)

            for fix in fixes_sorted:
                if not fix.line_number or not fix.last_line_number:
                    logger.warning(f"Fix {fix.issue_key} missing line numbers, skipping")
                    skipped_fixes.append(fix)
                    continue

                start = fix.line_number - 1
                end = fix.last_line_number - 1

                if start < 0 or end >= len(lines) or start > end:
                    logger.warning(f"Fix {fix.issue_key} has invalid line range, skipping")
                    skipped_fixes.append(fix)
                    continue

                # Determine indentation from the first line of the original block
                original_first_line = lines[start]
                leading_ws = original_first_line[:len(original_first_line) - len(original_first_line.lstrip())]

                # Split fixed code into lines
                fixed_lines_raw = fix.fixed_code.split("\n")

                # Apply indentation to every line of the fixed block
                fixed_lines = []
                for fl in fixed_lines_raw:
                    # Preserve relative indentation
                    if fl.strip() == "":
                        fixed_lines.append("\n")
                    else:
                        fixed_lines.append(leading_ws + fl.rstrip() + "\n")



                lines[start:end + 1] = fixed_lines
                applied_fixes.append(fix)

            # Validate final code (optional)
            modified_content = "".join(lines)
            if not self._validate_modified_content(modified_content, file_path):
                logger.error("Validation failed — not writing changes.")
                return False

            # Write file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
            return True

        except Exception as e:
            logger.error(f"Error applying fixes to {file_path}: {e}", exc_info=True)
            return False


    def _validate_modified_content(self, content: str, file_path: Path) -> bool:
        """
        Perform basic validation on modified content to catch obvious corruption.

        This is a safety net that runs BEFORE writing any changes to disk.
        If validation fails, NO changes are written.

        Checks:
        1. Content is not empty
        2. Brackets/parentheses are balanced
        3. Python syntax is valid (for .py files)
        4. No duplicate function definitions

        Args:
            content: Modified file content
            file_path: Path to the file (to determine language)

        Returns:
            True if content passes validation, False otherwise
        """
        try:
            # CHECK 1: File is not empty
            if not content or not content.strip():
                logger.error("Validation failed: Modified content is empty")
                return False

            # # CHECK 2: Basic bracket/parenthesis matching
            # if not self._check_bracket_balance(content):
            #     logger.error("Validation failed: Unbalanced brackets/parentheses/braces")
            #     return False

            # CHECK 3: Python-specific syntax validation
            if file_path.suffix == '.py':
                try:
                    import ast
                    ast.parse(content)

                except SyntaxError as e:
                    logger.error(f"Validation failed: Python syntax error at line {e.lineno}: {e.msg}")

                    return False

            # CHECK 4: No duplicate function/class definitions
            if not self._check_no_duplicate_definitions(content, file_path.suffix):
                logger.error("Validation failed: Detected duplicate function/class definitions")
                return False


            return True

        except Exception as e:
            logger.error(f"Error during content validation: {e}", exc_info=True)
            # On validation error, fail safe - don't write
            return False

    def _check_bracket_balance(self, content: str) -> bool:
        """
        Check if brackets, parentheses, and braces are balanced.

        This is a simple check that doesn't account for strings/comments
        but catches obvious structural corruption.
        """
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

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
        """
        Check for duplicate function/class definitions.

        Duplicate definitions are a strong indicator that code was incorrectly
        inserted/replaced, as this would almost never happen intentionally.
        """
        if file_extension != '.py':
            return True  # Only check Python files for now

        import re

        # Find all function and class definitions
        func_pattern = r'^\s*(async\s+)?def\s+(\w+)\s*\('
        class_pattern = r'^\s*class\s+(\w+)\s*[:\(]'

        definitions = {}  # name -> line numbers where defined

        for line_num, line in enumerate(content.split('\n'), 1):
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

    def apply_fixes_with_validation(
            self,
            fixes: List[FixSuggestion],
            issues: List[SonarIssue],
            project_path: Path,
            create_backup: bool = True,
            dry_run: bool = False,
            use_validator: bool = True,
            validator_provider: str = "openai",
            validator_model: Optional[str] = None,
            validator_api_key: Optional[str] = None,
            min_confidence: float = 0.7
    ) -> FixResult:
        """
        Apply fixes with optional validation by a senior code reviewer agent.

        This method integrates the fix validator to review fixes before application.

        Workflow:
        1. Group fixes by file
        2. If use_validator=True: Validate each fix with AI reviewer
        3. Only apply fixes that pass validation (APPROVED or MODIFIED status)
        4. Use improved line-based replacement logic
        5. Perform syntax validation before writing

        Args:
            fixes: List of fix suggestions
            issues: List of corresponding SonarCloud issues
            project_path: Path to the project root
            create_backup: Whether to create a backup before applying
            dry_run: If True, don't actually modify files
            use_validator: If True, validate fixes with AI reviewer before applying
            validator_provider: LLM provider for validation
            validator_model: LLM model for validation
            validator_api_key: API key for validator
            min_confidence: Minimum confidence threshold for approval

        Returns:
            FixResult with detailed application results
        """


        result = FixResult(
            project_path=project_path,
            total_fixes_attempted=len(fixes)
        )

        # Create backup if requested
        if create_backup and not dry_run:
            backup_path = self._create_backup(project_path)
            result.backup_created = True
            result.backup_path = backup_path
            logger.info(f"Created backup at: {backup_path}")

        # Validate fixes if requested
        validated_fixes = fixes
        if use_validator:
            logger.info("Validating fixes with AI code reviewer...")
            validator = FixValidator(
                provider=validator_provider,
                model=validator_model,
                api_key=validator_api_key,
                min_confidence_threshold=min_confidence
            )

            validation_results = []
            for fix, issue in zip(fixes, issues):
                file_path = project_path / fix.file_path if fix.file_path else None
                if not file_path or not file_path.exists():
                    logger.warning(f"File not found for validation: {file_path}")
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()

                    validation_result = validator.validate_fix(fix, issue, file_content)
                    validation_results.append(validation_result)

                    # Log validation decision
                    if validation_result.status == ValidationStatus.APPROVED:
                        logger.info(f"✓ Fix {fix.issue_key} APPROVED (confidence: {validation_result.confidence:.2f})")
                    elif validation_result.status == ValidationStatus.MODIFIED:
                        logger.info(
                            f"✓ Fix {fix.issue_key} MODIFIED by reviewer (confidence: {validation_result.confidence:.2f})")
                    elif validation_result.status == ValidationStatus.REJECTED:
                        logger.warning(f"✗ Fix {fix.issue_key} REJECTED: {validation_result.validation_notes}")
                    else:
                        logger.warning(f"? Fix {fix.issue_key} NEEDS_REVIEW: {validation_result.validation_notes}")

                except Exception as e:
                    logger.error(f"Error validating fix {fix.issue_key}: {e}", exc_info=True)

            # Filter to only approved/modified fixes
            validated_fixes = [
                vr.final_fix for vr in validation_results
                if vr.should_apply
            ]

            rejected_count = len(fixes) - len(validated_fixes)
            if rejected_count > 0:
                logger.warning(
                    f"Validator rejected/flagged {rejected_count}/{len(fixes)} fixes. "
                    f"Only applying {len(validated_fixes)} validated fixes."
                )

        # Group fixes by file for efficient processing
        fixes_by_file: Dict[str, List[FixSuggestion]] = {}
        for fix in validated_fixes:
            file_key = self._get_file_from_fix(fix, project_path)
            if file_key:
                if file_key not in fixes_by_file:
                    fixes_by_file[file_key] = []
                fixes_by_file[file_key].append(fix)


        # Apply fixes file by file
        for file_path_str, file_fixes in fixes_by_file.items():
            try:
                file_path = Path(file_path_str)

                if self._apply_fixes_to_file(file_path, file_fixes, dry_run):
                    result.successful_fixes.extend(file_fixes)
                else:
                    result.failed_fixes.extend([
                        {"fix": fix, "error": f"Failed validation or application {file_path}"}
                        for fix in file_fixes
                    ])
                    logger.error(f"✗ Failed to apply fixes to {file_path}")
            except Exception as e:
                result.failed_fixes.extend([
                    {"fix": fix, "error": str(e)}
                    for fix in file_fixes
                ])
                logger.error(f"✗ Error processing file {file_path_str}: {e}", exc_info=True)

        return result
