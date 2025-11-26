"""LLM-powered code fixer for SonarCloud issues."""

import os
import re
import shutil
import autopep8
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
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


def calculate_base_indentation(code: str) -> int:
    """
    Calculate the base indentation level from the first non-empty line.

    Args:
        code: Code string

    Returns:
        Number of leading spaces in the first non-empty line
    """
    for line in code.split('\n'):
        if line.strip():
            return len(line) - len(line.lstrip())
    return 0



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

    def generate_fix(self, issue: SonarIssue, project_path: Path,rule_info: Dict[str, Any],modified_content: str="", error_message: str="") -> Optional[FixSuggestion]:
        """
        Generate a fix suggestion for a SonarCloud issue.

        Args:
            issue: SonarCloud issue to fix
            project_path: Path to the project root
            rule_info: Dictionary containing rule information
            modified_content: Optional[str] = None,
            error_message: Optional[str] = None
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
                content_file = f.read()
                lines = content_file.splitlines(keepends=True)


            # Get context around the issue
            if modified_content!="":
                context = modified_content
            else:
                context = self._extract_context(lines, issue.first_line, issue.last_line, self.context_lines)

                if len(lines)<1000:
                    context['context']=content_file

            # Generate fix using LLM
            fix_response = self._call_llm(issue, context, file_path.suffix, rule_info, error_message)
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

        #end_idx = min(len(lines), first_line_idx + context_lines + 1)
        end_idx = min(len(lines), last_line_idx + context_lines + 1)

        context_lines_list= lines[first_line_idx:last_line_number]

        problem_line = lines[first_line_idx].rstrip() if first_line_idx < len(lines) else ""

        return {
            "context": "".join(context_lines_list),
            "problem_line": problem_line,
            "line_number": first_line_number,
            "start_line": start_idx + 1,
            "end_line": end_idx
        }

    def _call_llm(self, issue: SonarIssue, context: Dict[str, Any], file_extension: str, rule_info: Dict[str, Any], error_message: str="") -> Optional[Dict[str, Any]]:
        """
        Call the LLM to generate a fix.

        Args:
            issue: SonarCloud issue
            context: Code context around the issue
            rule_info: Rule info from sonar cloud
            file_extension: File extension to determine language

        Returns:
            Dictionary with fix information or None
        """
        # Determine programming language
        language = self._get_language_from_extension(file_extension)

        # Prepare prompt
        prompt = self._create_fix_prompt(issue, context, rule_info, language, error_message)

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
                    max_tokens=4000
                )
                return self._parse_openai_response(response)


            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return self._parse_gemini_response(response)
        except Exception as e:
            logger.error(f"Error calling {self.provider} LLM: {e}", exc_info=True)
            return None


    def _create_fix_prompt(self, issue: SonarIssue, context: Dict[str, Any], rule_info: Dict[str, Any], language: str,
                        error_message: str = "") -> str:
        """Create a focused prompt for the LLM to generate a fix."""
        error_section = f"\n**Error Context:**\n{error_message}\n" if error_message else ""
        base_indent = calculate_base_indentation(context.get('context', ''))

        # Detect indentation style and count
        indent_style = "spaces" if base_indent > 0 else "detect from code"
        indent_count = base_indent if base_indent > 0 else 0

        # Extract specific rule guidance
        root_cause = rule_info.get('root_cause', 'Code quality issue detected')
        fix_description = rule_info.get('how_to_fix', {}).get('description', 'Apply appropriate fix')
        fix_steps = rule_info.get('how_to_fix', {}).get('steps', [])
        priority = rule_info.get('how_to_fix', {}).get('priority', 'Medium')

        # Format steps as numbered list
        steps_text = ""
        if fix_steps:
            steps_text = "\n".join([f"   {i+1}. {step}" for i, step in enumerate(fix_steps)])

        # Special handling for common rule patterns
        is_unused_code = "unused" in issue.rule.lower() or "unused" in issue.message.lower()
        is_literal_duplication = "duplicating this literal" in issue.message.lower() or "define a constant" in issue.message.lower()
        is_null_check = "null" in issue.rule.lower() or "nullable" in issue.message.lower()

        # Extract literal value for duplication issues
        literal_match = None
        if is_literal_duplication:
            import re
            literal_pattern = r'duplicating this literal "([^"]+)"'
            match = re.search(literal_pattern, issue.message)
            if match:
                literal_match = match.group(1)

        prompt = f"""Fix this {language} code issue following SonarQube rule {issue.rule}.
        
        ## ISSUE ANALYSIS
        **Rule:** {issue.rule} | **Severity:** {issue.severity} | **Type:** {issue.type}
        **Root Cause:** {root_cause}
        **Fix Strategy:** {fix_description}
        
        **Steps to Follow:**
        {steps_text if steps_text else "   1. Analyze the specific issue\n   2. Apply minimal fix\n   3. Preserve functionality"}
        
        **Issue Message:** {issue.message}
        **Location:** Line {issue.first_line}{f"-{issue.last_line}" if issue.last_line != issue.first_line else ""}
        {error_section}
        
        ## CODE CONTEXT (Lines {context['start_line']}-{context['end_line']})
        ```{language}
        {context['context']}
        ```
        
        **Problematic Line {issue.first_line}:**
        ```{language}
        {context['problem_line']}
        ```
        
        ## FIX REQUIREMENTS
        
        **Focus:** Rule {issue.rule} | Priority: {priority} | Minimal surgical change only
        
        ### OUTPUT SCOPE
        - Return EXACTLY lines {context['start_line']}-{context['end_line']} (inclusive)
        - Include ALL lines from this range, even if unchanged
        - NO additional imports, classes, or function headers outside this range
        
        ### FORMATTING
        - Preserve indentation: {indent_count} {indent_style}
        - Maintain original code style and formatting
        - Keep syntax valid and error-free
        
        ### CHANGE STRATEGY
        - Make MINIMAL changes - only what's needed for the rule
        - Preserve variable names unless renaming is the fix
        - Maintain identical functionality and behavior
        - Don't refactor unrelated code
        
        {f'''
        ### UNUSED CODE SPECIAL RULES
        - Remove ONLY the unused variables/imports on the problematic lines
        - If removing a line leaves empty space, maintain proper structure
        - Ensure removal doesn't break syntax or references
        ''' if is_unused_code else ''}
        
        {f'''
        ### LITERAL DUPLICATION SPECIAL RULES
        - Target literal: "{literal_match or '[extract from message]'}"
        - Define a constant with descriptive UPPER_CASE name
        - Replace ALL occurrences of this literal in the provided scope
        - Place constant definition at appropriate scope level (class/function start)
        ''' if is_literal_duplication else ''}
        
        {f'''
        ### NULL CHECK SPECIAL RULES
        - Add proper null/None checks before usage
        - Use appropriate null-safe operators for {language}
        - Ensure defensive programming without changing logic flow
        ''' if is_null_check else ''}
        
        ## VALIDATION CHECKLIST
        Before submitting, verify:
        □ Output contains ONLY lines {context['start_line']}-{context['end_line']}
        □ Indentation preserved exactly ({indent_count} {indent_style})
        □ Syntax is valid {language} code
        □ Functionality unchanged (same inputs → same outputs)
        □ Rule {issue.rule} violation is resolved
        □ No new issues introduced
        
        ## REQUIRED OUTPUT FORMAT
        
        FIXED_CODE:
        ```{language}
        [Exact lines {context['start_line']}-{context['end_line']} with minimal fix applied]
        ```
        
        EXPLANATION:
        [Brief explanation: what changed and why it fixes rule {issue.rule}]
        
        CONFIDENCE: [0.0-1.0]
        - 0.9-1.0: Certain fix is correct and safe
        - 0.7-0.8: Confident but edge cases possible  
        - 0.5-0.6: Fix likely works but needs validation
        - 0.0-0.4: Uncertain or cannot provide safe fix
        
        ---
        **Focus:** Rule {issue.rule} | Priority: {priority} | Minimal surgical change only
        {f"**Critical:** For literal duplication, ALL occurrences must be replaced" if is_literal_duplication else ""}
        {f"**Critical:** Remove unused code without breaking syntax" if is_unused_code else ""}
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
            fixed_code_pattern = r'FIXED_CODE:\s*```[a-zA-Z]{0,20}\s*((?:[^`]|`(?!``))*?)\s*```'
            fixed_code_match = re.search(fixed_code_pattern, content, re.DOTALL)
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

    def _clean_fixed_code(self, fixed_code: str) -> str:
        """
        Clean the fixed code by removing extra quotes and normalizing whitespace.
        """
        # Remove surrounding quotes if present
        cleaned = fixed_code.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        # Replace escaped quotes
        cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")

        # Normalize line endings
        cleaned = cleaned.replace('\\n', '\n')

        return cleaned

    def _process_multi_function_code(self, fixed_code: str, base_indent: str) -> List[str]:
        """
        Process fixed code that may contain multiple functions.
        Handles proper indentation and separation with correct Python nesting.
        """
        lines = fixed_code.splitlines()
        processed_lines = []

        current_function_indent = ""
        inside_function = False
        inside_docstring = False
        docstring_delimiter = None

        for i, line in enumerate(lines):
            if not line.strip():
                # Empty line - keep as is but ensure newline
                processed_lines.append('\n')
                continue

            stripped_line = line.strip()

            # Handle docstring detection
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                delimiter = '"""' if stripped_line.startswith('"""') else "'''"
                if not inside_docstring:
                    # Starting docstring
                    inside_docstring = True
                    docstring_delimiter = delimiter
                    processed_lines.append(base_indent + '    ' + stripped_line + '\n')
                elif stripped_line.endswith(docstring_delimiter):
                    # Ending docstring
                    inside_docstring = False
                    docstring_delimiter = None
                    processed_lines.append(base_indent + '    ' + stripped_line + '\n')
                else:
                    # Middle of docstring
                    processed_lines.append(base_indent + '    ' + stripped_line + '\n')
                continue

            if inside_docstring:
                # Inside docstring - maintain docstring indentation
                processed_lines.append(base_indent + '    ' + stripped_line + '\n')
                continue

            # Function/class definition
            if stripped_line.startswith(('def ', 'class ', '@')):
                inside_function = True
                current_function_indent = base_indent
                processed_lines.append(base_indent + stripped_line + '\n')

            # Function body content
            elif inside_function:
                # Determine the appropriate indentation level
                if stripped_line.startswith(
                        ('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ')):
                    # Control structures - one level deeper than function
                    processed_lines.append(base_indent + '    ' + stripped_line + '\n')

                elif stripped_line.startswith(('return ', 'raise ', 'yield ', 'pass', 'break', 'continue')):
                    # Check if this return is inside a control block by looking at previous lines
                    # Simple heuristic: if the previous non-empty line was a control structure, add extra indent
                    prev_non_empty_idx = i - 1
                    while prev_non_empty_idx >= 0 and not lines[prev_non_empty_idx].strip():
                        prev_non_empty_idx -= 1

                    if prev_non_empty_idx >= 0:
                        prev_line = lines[prev_non_empty_idx].strip()
                        if (prev_line.startswith(
                                ('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:')) or
                                prev_line.endswith(':')):
                            # Inside control block - add extra indentation
                            processed_lines.append(base_indent + '        ' + stripped_line + '\n')
                        else:
                            # Direct function body statement
                            processed_lines.append(base_indent + '    ' + stripped_line + '\n')
                    else:
                        # Default function body level
                        processed_lines.append(base_indent + '    ' + stripped_line + '\n')

                elif stripped_line.endswith(':'):
                    # Likely a control structure we missed
                    processed_lines.append(base_indent + '    ' + stripped_line + '\n')

                else:
                    # Default function body statement
                    processed_lines.append(base_indent + '    ' + stripped_line + '\n')

            else:
                # Not inside a function - treat as module level
                processed_lines.append(base_indent + stripped_line + '\n')

        # Clean up any duplicate return statements at the end
        processed_lines = self._remove_duplicate_returns(processed_lines)

        # Add extra newline at the end for function separation
        if processed_lines and not processed_lines[-1].strip() == '':
            processed_lines.append('\n')

        return processed_lines

    def _remove_duplicate_returns(self, lines: List[str]) -> List[str]:
        """
        Remove duplicate or conflicting return statements.
        """
        cleaned_lines = []
        return_statements = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('return '):
                return_statements.append(line)
            else:
                # If we've collected return statements, only keep the last meaningful one
                if return_statements:
                    # Find the most specific return statement (usually the longest)
                    best_return = max(return_statements, key=lambda x: len(x.strip()))
                    cleaned_lines.append(best_return)
                    return_statements = []
                cleaned_lines.append(line)

        # Handle any remaining return statements
        if return_statements:
            best_return = max(return_statements, key=lambda x: len(x.strip()))
            cleaned_lines.append(best_return)

        return cleaned_lines

    def _apply_fixes_to_file(self, file_path: Path, fixes: List[FixSuggestion], dry_run: bool) -> bool:
        """
        Apply fixes by REMOVING the block between line_number and last_line_number,
        and inserting the fixed_code block with correct indentation.
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would apply {len(fixes)} fixes to {file_path}")
            return True

        try:
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

                # Get the base indentation from the first line of the original block
                original_line = lines[start]

                # Calculate indentation
                indent = original_line[:len(original_line) - len(original_line.lstrip())]

                fixed_code_clean = self._clean_fixed_code(fix.fixed_code)

                # Split into logical blocks (functions) and process each
                new_block = self._process_multi_function_code(fixed_code_clean, indent)



                # CRITICAL FIX: Update lines array directly, don't create separate variable
                lines = (
                        lines[:fix.line_number - 1] +
                        new_block +
                        lines[fix.last_line_number:]
                )

                applied_fixes.append(fix)

            # Validate final code
            modified_content = ''.join(lines)

            validated, message_error, modified_content = self._validate_modified_content(modified_content, file_path)
            if not validated:
                logger.error(f"Validation failed — not writing changes. Error: {message_error}")
                return False

            # Write file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)

            if skipped_fixes:
                logger.warning(f"⚠ Skipped {len(skipped_fixes)} fixes")

            return True

        except Exception as e:
            logger.error(f"Error applying fixes to {file_path}: {e}", exc_info=True)
            return False

    def _validate_modified_content(self, content: str, file_path: Path) -> Union[bool, str]:
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
        message_error = ""
        try:
            # CHECK 1: File is not empty
            if not content or not content.strip():
                message_error="Validation failed: Modified content is empty"
                logger.error(message_error)
                return False,message_error,content

            # # CHECK 2: Basic bracket/parenthesis matching
            # if not self._check_bracket_balance(content):
            #     logger.error("Validation failed: Unbalanced brackets/parentheses/braces")
            #     return False

            # CHECK 3: Python-specific syntax validation
            if file_path.suffix == '.py':
                try:
                    import ast
                    ast.parse(content)
                    validated = True

                except SyntaxError as e:
                    message_error=f"Validation failed: Python syntax error at line {e.lineno}: {e.msg}"
                    logger.error(message_error)
                    validated=False
            if not validated:
                fixed_code = autopep8.fix_code(content, options={"aggressive": 2})
                logger.info(f"fixed code: {fixed_code}")

            # CHECK 4: No duplicate function/class definitions
            if not self._check_no_duplicate_definitions(content, file_path.suffix):
                message_error="Validation failed: Detected duplicate function/class definitions"

                logger.error(message_error)
                return False, message_error,content


            return True,"",content

        except Exception as e:
            message_error=f"Error during content validation: {e}"
            logger.error(message_error, exc_info=True)
            # On validation error, fail safe - don't write
            return False,message_error,content

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

        # Initialize validator if needed (but don't use it upfront)
        validator = None
        if use_validator:
            validator = FixValidator(
                provider=validator_provider,
                model=validator_model,
                api_key=validator_api_key,
                min_confidence_threshold=min_confidence
            )

        # Group fixes by file for efficient processing
        fixes_by_file: Dict[str, List[Tuple[FixSuggestion, SonarIssue]]] = {}
        for fix, issue in zip(fixes, issues):
            file_key = self._get_file_from_fix(fix, project_path)
            if file_key:
                if file_key not in fixes_by_file:
                    fixes_by_file[file_key] = []
                fixes_by_file[file_key].append((fix, issue))

        # Apply fixes file by file
        for file_path_str, file_fix_pairs in fixes_by_file.items():
            try:
                file_path = Path(file_path_str)
                file_fixes = [fix for fix, _ in file_fix_pairs]
                file_issues = [issue for _, issue in file_fix_pairs]

                # STEP 1: Try to apply fixes directly
                logger.info(f"Applying {len(file_fixes)} fixes to {file_path}")

                # Store original content for validator fallback
                original_content = ""
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()

                # Attempt direct application
                success = self._apply_fixes_to_file(file_path, file_fixes, dry_run)
                if success:
                    # Direct application succeeded
                    result.successful_fixes.extend(file_fixes)
                    logger.info(f"✓ Successfully applied {len(file_fixes)} fixes to {file_path}")

                else:
                    # STEP 2: Direct application failed, try AI validator fallback
                    if use_validator and validator:
                        logger.warning(
                            f"Direct fix application failed for {file_path}. Trying AI validator fallback...")

                        # Restore original content
                        if not dry_run and file_path.exists():
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(original_content)

                        # Use validator to fix each problematic fix
                        validator_success_count = 0
                        for fix, issue in file_fix_pairs:
                            try:
                                logger.info(f"Using AI validator for fix {fix.issue_key}")

                                # Get current file content (may have been modified by previous validator fixes)
                                current_content = original_content
                                if file_path.exists():
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        current_content = f.read()

                                validation_result = validator.validate_fix(fix, issue, current_content)

                                # Log validation decision
                                if validation_result.status == ValidationStatus.APPROVED:
                                    logger.info(
                                        f"✓ Fix {fix.issue_key} APPROVED by validator (confidence: {validation_result.confidence:.2f})")
                                    result.successful_fixes.append(fix)
                                    validator_success_count += 1

                                elif validation_result.status == ValidationStatus.MODIFIED:
                                    logger.info(
                                        f"✓ Fix {fix.issue_key} MODIFIED by validator (confidence: {validation_result.confidence:.2f})")

                                    # Apply the improved fix
                                    if validation_result.final_fix:
                                        improved_success = self._apply_fixes_to_file(
                                            file_path,
                                            [validation_result.final_fix],
                                            dry_run
                                        )
                                        if improved_success:
                                            result.successful_fixes.append(validation_result.final_fix)
                                            validator_success_count += 1
                                        else:
                                            result.failed_fixes.append({
                                                "fix": fix,
                                                "error": "Validator improved fix but application still failed"
                                            })
                                    else:
                                        result.failed_fixes.append({
                                            "fix": fix,
                                            "error": "Validator marked as MODIFIED but provided no improved fix"
                                        })

                                elif validation_result.status == ValidationStatus.REJECTED:
                                    logger.warning(
                                        f"✗ Fix {fix.issue_key} REJECTED by validator: {validation_result.validation_notes}")
                                    result.failed_fixes.append({
                                        "fix": fix,
                                        "error": f"Rejected by validator: {validation_result.validation_notes}"
                                    })

                                else:  # NEEDS_REVIEW
                                    logger.warning(
                                        f"? Fix {fix.issue_key} NEEDS_REVIEW: {validation_result.validation_notes}")
                                    result.failed_fixes.append({
                                        "fix": fix,
                                        "error": f"Needs manual review: {validation_result.validation_notes}"
                                    })

                            except Exception as e:
                                logger.error(f"Error in validator fallback for fix {fix.issue_key}: {e}", exc_info=True)
                                result.failed_fixes.append({"fix": fix, "error": f"Validator error: {str(e)}"})

                        logger.info(f"Validator fallback: {validator_success_count}/{len(file_fixes)} fixes successful")

                    else:
                        # No validator available, mark all as failed
                        result.failed_fixes.extend([
                            {"fix": fix, "error": f"Direct application failed and no validator available"}
                            for fix in file_fixes
                        ])
                        logger.error(f"✗ Failed to apply fixes to {file_path} (no validator fallback)")

            except Exception as e:
                result.failed_fixes.extend([
                    {"fix": fix, "error": str(e)}
                    for fix, _ in file_fix_pairs
                ])
                logger.error(f"✗ Error processing file {file_path_str}: {e}", exc_info=True)

        return result

