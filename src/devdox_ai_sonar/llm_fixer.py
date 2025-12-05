"""LLM-powered code fixer for SonarCloud issues."""

import os
import re
import shutil
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import json
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence
from datetime import datetime

from .fix_validator import FixValidator, ValidationStatus

from devdox_ai_sonar.models.sonar import (
    SonarIssue,
    SonarSecurityIssue,
    FixSuggestion,
    FixResult,
)
from devdox_ai_sonar.utils.file_indentation import (
    apply_single_fix,
    write_file_lines,
    read_file_lines,
    calculate_base_indentation,
    normalize_indentation,
)

from .logging_config import setup_logging, get_logger

setup_logging(level="DEBUG", log_file="demo.log")
logger = get_logger(__name__)

try:
    from together import Together  # type: ignore[import]

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

java_extension = ".java"
scala_extension = ".scala"


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
        self.model = model
        self.prompt_dir = Path(__file__).parent / "prompts"
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
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'openai' or 'gemini' or 'togetherai'"
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

    def generate_fix(
        self,
        issue: Union[SonarIssue, SonarSecurityIssue],
        project_path: Path,
        rule_info: Optional[Dict[str, Any]] = None,
        modified_content: str = "",
        error_message: str = "",
    ) -> Optional[FixSuggestion]:
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
            logger.warning(
                f"Issue {issue.key} has no file or line number, skipping fix generation"
            )
            return None

        # Read the file and extract context
        file_path = project_path / issue.file
        if not file_path.exists():
            logger.warning(f"File not found for issue {issue.key}: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content_file = f.read()
                lines = content_file.splitlines(keepends=True)

            # Get context around the issue
            if modified_content != "":
                context_dict = {
                    "context": modified_content,
                    "start_line": issue.first_line,
                    "end_line": issue.last_line,
                }
            else:
                context_dict = self._extract_context(
                    lines, issue.first_line, issue.last_line, self.context_lines
                )
                if "end_line" in context_dict:
                    end_line_value = context_dict.get("end_line")
                    if isinstance(end_line_value, int):
                        issue.last_line = end_line_value

            # Generate fix using LLM
            fix_response = self._call_llm(
                issue, context_dict, file_path.suffix, rule_info, error_message
            )
            if fix_response:
                logger.info(
                    f"Successfully generated fix for issue {issue.key} with confidence {fix_response['confidence']}"
                )
                return FixSuggestion(
                    issue_key=issue.key,
                    original_code=context_dict["context"],
                    fixed_code=fix_response["fixed_code"],
                    helper_code=fix_response.get("helper_code"),
                    placement_helper=fix_response.get("placement_helper"),
                    explanation=fix_response["explanation"],
                    confidence=fix_response["confidence"],
                    llm_model=self.model,
                    rule_description=fix_response.get("rule_description"),
                    file_path=str(
                        file_path.relative_to(project_path)
                    ),  # Store relative path
                    line_number=context_dict.get("start_line"),
                    sonar_line_number=issue.first_line,
                    last_line_number=context_dict.get("end_line"),
                )

        except Exception as e:
            logger.error(
                f"Error generating fix for issue {issue.key}: {e}", exc_info=True
            )

        return None

    def apply_fixes(
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
                    logger.info(
                        f"Successfully applied {len(file_fixes)} fixes to {file_path}"
                    )
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

    def _find_containing_function(
        self, lines: List[str], target_line_idx: int
    ) -> Optional[int]:
        """Find the function that contains the given line index."""
        # Search backwards from the target line to find a function definition
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

    def _is_line_inside_function(
        self, lines: List[str], line_idx: int, function_start_idx: int
    ) -> bool:
        """Check if a line is inside the function starting at function_start_idx."""
        if line_idx < function_start_idx:
            return False

        function_end_idx = self._find_function_end(lines, function_start_idx)
        if function_end_idx is None:
            # If we can't find the end, use heuristic based on indentation
            return self._check_indentation_containment(
                lines, line_idx, function_start_idx
            )

        return line_idx <= function_end_idx

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

    def _extract_context(
        self,
        lines: List[str],
        first_line_number: int,
        last_line_number: int,
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

        # Convert to 0-indexed
        first_line_idx = first_line_number - 1
        last_line_idx = last_line_number - 1

        if first_line_idx >= len(lines):
            return self._get_empty_context(first_line_number)

        problem_line = lines[first_line_idx].rstrip()
        # Check if first line contains function/method definition
        if self._is_function_definition(problem_line):
            function_context = self._extract_complete_function(
                lines, first_line_idx, first_line_idx
            )
            if function_context:
                return function_context

        function_start_idx = self._find_containing_function(lines, first_line_idx)
        if function_start_idx is not None:
            function_context = self._extract_complete_function(
                lines, function_start_idx, first_line_idx
            )
            if function_context:
                return function_context

        # Check if issue spans multiple lines and any line is a function definition
        for line_idx in range(first_line_idx, min(last_line_idx + 1, len(lines))):
            if self._is_function_definition(lines[line_idx]):
                function_context = self._extract_complete_function(
                    lines, line_idx, first_line_idx
                )
                if function_context:
                    return function_context

        # Fall back to normal context extraction
        return self._extract_normal_context(
            lines, first_line_idx, last_line_idx, context_lines
        )

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

    def _find_function_start_with_decorators(
        self, lines: List[str], target_line_idx: int
    ) -> int:
        """
        Find the actual start of a function, including any decorators.

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

    def _extract_complete_function(
        self, lines: List[str], start_idx: int, problem_line_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract complete function/method from start to end, including decorators.

        Args:
            lines: All lines in the file
            start_idx: Starting index (0-indexed) of function definition or where issue was detected

        Returns:
            Context dictionary with complete function or None if not found
        """
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
        function_end = self._find_function_end(lines, function_def_line_idx)

        if function_end is None:
            # If we can't find the end, include reasonable context
            function_end = min(
                len(lines) - 1, function_def_line_idx + 50
            )  # Max 50 lines

        # Extract function lines (including decorators)
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
            "problem_line": problem_line,
        }

    def _find_function_end(self, lines: List[str], start_idx: int) -> Optional[int]:
        """
        Find the end line of a function/method based on language-specific rules.
        """

        if start_idx >= len(lines):
            return None

        start_line = lines[start_idx]

        # Determine language and strategy based on the function definition
        if re.search(r"\bdef\b", start_line):
            # Python function - use indentation
            return self._find_python_function_end(lines, start_idx)
        elif "{" in start_line or re.search(
            r"\bfunction\b|\w+\s*\(.*\)\s*\{", start_line
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
        assignment_match = re.search(r"(\w+)\s*[:=]", function_line)
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

    def _extract_normal_context(
        self,
        lines: List[str],
        first_line_idx: int,
        last_line_idx: int,
        context_lines: int,
    ) -> Dict[str, Any]:
        """
        Extract normal context around the issue (original behavior).
        """
        # Calculate context boundaries
        start_idx = max(0, first_line_idx - context_lines)
        end_idx = min(len(lines), last_line_idx + context_lines + 1)

        context_lines_list = lines[start_idx:end_idx]
        problem_line = (
            lines[first_line_idx].rstrip() if first_line_idx < len(lines) else ""
        )

        return {
            "context": "".join(context_lines_list),
            "problem_line": problem_line,
            "line_number": first_line_idx + 1,  # Convert back to 1-indexed
            "start_line": start_idx + 1,  # Convert back to 1-indexed
            "end_line": end_idx,
            "is_complete_function": False,
        }

    def _get_empty_context(self, line_number: int) -> Dict[str, Any]:
        """
        Return empty context when line number is out of range.
        """
        return {
            "context": "",
            "problem_line": "",
            "line_number": line_number,
            "start_line": line_number,
            "end_line": line_number,
            "is_complete_function": False,
        }

    def _call_llm(
        self,
        issue: Union[SonarIssue, SonarSecurityIssue],
        context: Dict[str, Any],
        file_extension: str,
        rule_info: Optional[Dict[str, Any]] = None,
        error_message: str = "",
    ) -> Optional[Dict[str, Any]]:
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
        prompt = self._create_fix_prompt(
            issue, context, rule_info, language, error_message
        )

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior software engineer specializing in code quality and SonarCloud rule compliance. Your job is to analyze code issues and provide precise fixes.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                )
                return self._parse_openai_response(response)

            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt
                )
                return self._parse_gemini_response(response)
            elif self.provider == "togetherai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior software engineer specializing in code quality and SonarCloud rule compliance. Your job is to analyze code issues and provide precise fixes.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.1,
                )

            return self._parse_togetherai_response(response)
        except Exception as e:
            logger.error(f"Error calling {self.provider} LLM: {e}", exc_info=True)
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

        # If we see many initialization patterns, likely a constructor
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

        # Alternative pattern: "complexity is X, maximum allowed is Y"
        alt_pattern = r"complexity is (\d+).*maximum.*?(\d+)"
        alt_match = re.search(alt_pattern, message, re.IGNORECASE)

        if alt_match:
            return {"current": alt_match.group(1), "target": alt_match.group(2)}

        # Default fallback
        return {"current": "Unknown", "target": "15"}

    def _create_fix_prompt(
        self,
        issue: Union[SonarIssue, SonarSecurityIssue],
        context: Dict[str, Any],
        rule_info: Optional[Dict[str, Any]] = None,
        language: str = "python",
        error_message: str = "",
    ) -> str:
        """Create a concise, focused prompt for the LLM to generate a fix."""

        # 1. Context Setup
        code_chunk = context.get("context", "")
        base_indent = calculate_base_indentation(code_chunk)

        # 2. Strategy Detection
        strategies = [
            f"• PRESERVE indentation ({base_indent} spaces) and existing logic flow.",
            "• Make MINIMAL changes necessary to satisfy the rule.",
        ]

        msg_lower = issue.message.lower()

        # Cognitive Complexity
        if "cognitive complexity" in msg_lower:
            # Extract numbers if available
            comp_info = self._extract_complexity_info(issue.message)
            target = comp_info.get("target", "15")

            strategies.extend(
                [
                    f"• REDUCE complexity to < {target}.",
                    "• EXTRACT logic to new helper methods/functions.",
                    "• CRITICAL: Do NOT define helper functions inside the existing function (No nesting).",
                    "• If the original code snippet calls functions or methods (e.g., validate(), "
                    "normalize(), process_data()), DO NOT create new helper definitions for them."
                    "Assume they already exist in the project and should not be recreated."
                    "• Only extract logic into a helper function if that logic does not already"
                    "correspond to any existing function call present in the snippet."
                    "• Helper functions must be SIMPLE and ATOMIC (do not move complexity, remove it).",
                    "• CRITICAL: Put only the CALL to helper functions in FIXED_SELECTION.",
                    "• CRITICAL: Put only the DEFINITION of helper functions in NEW_HELPER_CODE.",
                    "• NEVER put the same function definition in both sections.",
                    "• If the original function is a class method (uses `self` or `cls`), "
                    "then any helper function placed as a SIBLING MUST also accept `self` or `cls`.",
                    "• If the helper function does NOT use `self` or `cls`, it MUST NOT be placed as a SIBLING.",
                    "• In that case, place NEW_HELPER_CODE in GLOBAL_BOTTOM (utility function), unless it is a constant/import → GLOBAL_TOP.",
                ]
            )
            if self._is_init_method(code_chunk):
                strategies.append(
                    "• Keep __init__ signature intact; extract validation logic to helpers."
                )

        # Unused Code
        elif "unused" in issue.rule.lower() or "unused" in msg_lower:
            strategies.append("• Remove ONLY the specific unused variable/import.")
            strategies.append("• Do not break code that references adjacent lines.")

        # Literal Duplication
        elif "duplicating this literal" in msg_lower:
            match = re.search(r'duplicating this literal "([^"]+)"', issue.message)
            literal = match.group(1) if match else "the repeated value"
            strategies.append(
                f'• Extract the literal "{literal}" to a constant/variable.'
            )
            strategies.append(
                "• Place the constant at the class or module level (not inside the function)."
            )
            strategies.append(
                "• CRITICAL: Put the constant DEFINITION in NEW_HELPER_CODE."
            )
            strategies.append(
                "• CRITICAL: Put the code that USES the constant in FIXED_SELECTION."
            )
            strategies.append(
                "• Define the constant in [NEW_HELPER_CODE] (likely GLOBAL_TOP)."
            )
            strategies.append("• Use the constant in [FIXED_SELECTION].")

        # Null Checks
        elif "null" in issue.rule.lower() or "nullable" in msg_lower:
            strategies.append("• Add defensive null/None checks before usage.")

        # 3. Construct Prompt
        # We join strategies with newlines for a clean list
        strategy_text = "\n".join(strategies)

        template = self.jinja_env.get_template("fix_issues.j2")

        # Prepare context for template
        context = {
            "language": language,
            "issue": issue,
            "rule_info": rule_info,
            "context": context,
            "code_chunk": code_chunk,
            "error_message": error_message,
            "strategy_text": strategy_text,
        }
        # Render enhanced content
        prompt = template.render(**context)
        return prompt.strip()

    def _parse_openai_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parse OpenAI API response."""
        try:
            content = response.choices[0].message.content
            return self._extract_fix_from_response(content)
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}", exc_info=True)
            return None

    def _parse_gemini_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parse Gemini API response."""
        try:
            content = response.text
            return self._extract_fix_from_response(content)
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}", exc_info=True)
            return None

    def _parse_togetherai_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parse Together API response."""
        try:
            content = response.choices[0].message.content

            return self._extract_fix_from_response(content)
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}", exc_info=True)
            return None

    def _extract_using_regex_fallback(self, content: str) -> Optional[Dict[str, Any]]:
        """Fallback extraction using regex for malformed JSON."""
        logger.info(" Using regex fallback extraction...")

        try:
            results = self._apply_regex_patterns(content)
            return self._validate_results(results)
        except Exception as e:
            logger.error(f"Regex fallback failed: {e}")
            return None

    def _apply_regex_patterns(self, content: str) -> Dict[str, Any]:
        patterns = {
            "FIXED_SELECTION": r'"FIXED_SELECTION"\s*:\s*"((?:[^"\\]|\\.)*)"|\'FIXED_SELECTION\'\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            "NEW_HELPER_CODE": r'"NEW_HELPER_CODE"\s*:\s*"((?:[^"\\]|\\.)*)"|\'NEW_HELPER_CODE\'\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            "PLACEMENT": r'"PLACEMENT"\s*:\s*"([^"]*)"|\'PLACEMENT\'\s*:\s*\'([^\']*)\'',
            "EXPLANATION": r'"EXPLANATION"\s*:\s*"((?:[^"\\]|\\.)*)"|\'EXPLANATION\'\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
            "CONFIDENCE": r'"CONFIDENCE"\s*:\s*([0-9]*\.?[0-9]+)',
        }

        results = {}

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
        return value.strip()

    def _validate_results(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not results.get("FIXED_SELECTION"):
            logger.error(" No FIXED_SELECTION found")
            return None

        logger.info(" Regex fallback extraction successful")

        return {
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
        helper_code = str(fix_data.get("NEW_HELPER_CODE", "")).strip()
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

        # Require fixed_code
        if not fixed_code:
            logger.error("❌ FIXED_SELECTION is empty")
            return None

        logger.debug(f"✅ Fields extracted successfully: {len(fixed_code)} chars code")

        return {
            "fixed_code": fixed_code,
            "helper_code": helper_code,
            "placement_helper": placement,
            "explanation": explanation,
            "confidence": confidence,
        }

    def _extract_fix_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Robust extraction that handles various JSON response formats.
        """
        try:
            logger.debug(f"Processing response: {len(content)} chars")

            # Step 1: Try direct JSON parsing first (for well-formed responses)
            cleaned_content = content.strip()
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
                    logger.debug("✅ Direct JSON parsing successful")
                    return self._extract_fields_from_parsed_json(fix_data)
                except json.JSONDecodeError:
                    pass  # Continue to cleaning steps

            # Step 2: Last resort - regex extraction
            logger.info("Using regex fallback extraction")
            return self._extract_using_regex_fallback(content)

        except Exception as e:
            logger.error(f"Error in extraction: {e}", exc_info=True)
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
        shutil.copytree(project_path, backup_path)
        logger.info(f"Created project backup: {backup_path}")
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

    def _apply_fixes_to_file(
        self, file_path: Path, fixes: List[FixSuggestion], dry_run: bool
    ) -> bool:
        if dry_run:
            logger.info(f"[DRY RUN] Would apply {len(fixes)} fixes to {file_path}")
            return True, []

        try:
            lines = read_file_lines(file_path)
            sorted_fixes = sorted(fixes, key=lambda f: f.line_number or 0, reverse=True)

            results = []
            for fix in sorted_fixes:
                result = apply_single_fix(lines, fix)
                results.append(result)

                if not result.success:
                    logger.warning(f"Fix {fix.issue_key} skipped: {result.reason}")

            write_file_lines(file_path, lines)

            successful = sum(1 for r in results if r.success)
            logger.info(f"Applied {successful}/{len(fixes)} fixes to {file_path}")

            return all(r.success for r in results), results

        except Exception as e:
            logger.error(f"Error applying fixes to {file_path}: {e}", exc_info=True)
            return False, []

        except Exception as e:
            logger.error(f"Error applying fixes to {file_path}: {e}", exc_info=True)
            return False

    def _check_bracket_balance(self, content: str) -> bool:
        """
        Check if brackets, parentheses, and braces are balanced.

        This is a simple check that doesn't account for strings/comments
        but catches obvious structural corruption.
        """
        stack = []
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

    def apply_fixes_with_validation(
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
                min_confidence_threshold=min_confidence,
            )

        # Group fixes by file for efficient processing
        fixes_by_file: Dict[
            str, List[Tuple[FixSuggestion, Union[SonarIssue, SonarSecurityIssue]]]
        ] = {}
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

                # STEP 1: Try to apply fixes directly
                logger.info(f"Applying {len(file_fixes)} fixes to {file_path}")

                # Store original content for validator fallback
                original_content = ""
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        original_content = f.read()

                # Attempt direct application
                success = self._apply_fixes_to_file(file_path, file_fixes, dry_run)
                if success:
                    # Direct application succeeded
                    result.successful_fixes.extend(file_fixes)
                    logger.info(
                        f"✓ Successfully applied {len(file_fixes)} fixes to {file_path}"
                    )

                else:
                    # STEP 2: Direct application failed, try AI validator fallback
                    if use_validator and validator:
                        logger.warning(
                            f"Direct fix application failed for {file_path}. Trying AI validator fallback..."
                        )

                        # Restore original content
                        if not dry_run and file_path.exists():
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(original_content)

                        # Use validator to fix each problematic fix
                        validator_success_count = 0
                        for fix, issue in file_fix_pairs:
                            try:
                                logger.info(
                                    f"Using AI validator for fix {fix.issue_key}"
                                )

                                # Get current file content (may have been modified by previous validator fixes)
                                current_content = original_content
                                if file_path.exists():
                                    with open(file_path, "r", encoding="utf-8") as f:
                                        current_content = f.read()

                                validation_result = validator.validate_fix(
                                    fix, issue, current_content
                                )

                                # Log validation decision
                                if (
                                    validation_result.status
                                    == ValidationStatus.APPROVED
                                ):
                                    logger.info(
                                        f"✓ Fix {fix.issue_key} APPROVED by validator (confidence: {validation_result.confidence:.2f})"
                                    )
                                    result.successful_fixes.append(fix)
                                    validator_success_count += 1

                                elif (
                                    validation_result.status
                                    == ValidationStatus.MODIFIED
                                ):
                                    logger.info(
                                        f"✓ Fix {fix.issue_key} MODIFIED by validator (confidence: {validation_result.confidence:.2f})"
                                    )

                                    # Apply the improved fix
                                    if validation_result.final_fix:
                                        improved_success = self._apply_fixes_to_file(
                                            file_path,
                                            [validation_result.final_fix],
                                            dry_run,
                                        )
                                        if improved_success:
                                            result.successful_fixes.append(
                                                validation_result.final_fix
                                            )
                                            validator_success_count += 1
                                        else:
                                            result.failed_fixes.append(
                                                {
                                                    "fix": fix,
                                                    "error": "Validator improved fix but application still failed",
                                                }
                                            )
                                    else:
                                        result.failed_fixes.append(
                                            {
                                                "fix": fix,
                                                "error": "Validator marked as MODIFIED but provided no improved fix",
                                            }
                                        )

                                elif (
                                    validation_result.status
                                    == ValidationStatus.REJECTED
                                ):
                                    logger.warning(
                                        f"✗ Fix {fix.issue_key} REJECTED by validator: {validation_result.validation_notes}"
                                    )
                                    result.failed_fixes.append(
                                        {
                                            "fix": fix,
                                            "error": f"Rejected by validator: {validation_result.validation_notes}",
                                        }
                                    )

                                else:  # NEEDS_REVIEW
                                    logger.warning(
                                        f"? Fix {fix.issue_key} NEEDS_REVIEW: {validation_result.validation_notes}"
                                    )
                                    result.failed_fixes.append(
                                        {
                                            "fix": fix,
                                            "error": f"Needs manual review: {validation_result.validation_notes}",
                                        }
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Error in validator fallback for fix {fix.issue_key}: {e}",
                                    exc_info=True,
                                )
                                result.failed_fixes.append(
                                    {"fix": fix, "error": f"Validator error: {str(e)}"}
                                )

                        logger.info(
                            f"Validator fallback: {validator_success_count}/{len(file_fixes)} fixes successful"
                        )

                    else:
                        # No validator available, mark all as failed
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

            except Exception as e:
                result.failed_fixes.extend(
                    [{"fix": fix, "error": str(e)} for fix, _ in file_fix_pairs]
                )
                logger.error(
                    f"✗ Error processing file {file_path_str}: {e}", exc_info=True
                )

        return result


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


def _try_find_by_content(self, fix: FixSuggestion, project_path: Path) -> Optional[str]:
    """Search for file containing the original code snippet."""
    if not fix.original_code or len(fix.original_code.strip()) <= 10:
        return None

    matching_files = self._find_files_with_content(
        project_path, fix.original_code.strip()
    )

    return str(matching_files[0]) if matching_files else None
