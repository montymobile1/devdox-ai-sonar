from typing import Optional, Dict, Any, List, Tuple, Union
import ast
from pathlib import Path
from devdox_ai_sonar.models.file_structures import ConversionRisk, ConversionAnalysis
from devdox_ai_sonar.utils.async_file_io import AsyncFileReader
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PART 1: CLASS METHOD FINDER (Distinguishes methods from functions)
# ============================================================================


class ClassMethodFinder(ast.NodeVisitor):
    """Find functions and detect if they're class methods."""

    def __init__(self, function_name: str):
        self.function_name = function_name
        self.function_info: Optional[Dict[str, Any]] = None
        self.current_class: Optional[str] = None
        self.class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track when entering a class."""
        self.class_stack.append(node.name)
        self.current_class = node.name
        self.generic_visit(node)
        self.class_stack.pop()
        self.current_class = self.class_stack[-1] if self.class_stack else None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit regular function definitions."""
        if node.name == self.function_name:
            self.function_info = self._extract_function_info(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        if node.name == self.function_name:
            self.function_info = self._extract_function_info(node, is_async=True)
        self.generic_visit(node)

    def _extract_function_info(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool
    ) -> Dict[str, Any]:
        """Extract function information with class context."""
        # Check for @staticmethod or @classmethod
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        is_static = "staticmethod" in decorators
        is_classmethod = "classmethod" in decorators

        # Check if it's an instance method (has 'self' or 'cls' as first param)
        params = [arg.arg for arg in node.args.args]
        has_self = len(params) > 0 and params[0] in ("self", "cls")

        return {
            "name": node.name,
            "is_async": is_async,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "col_offset": node.col_offset,
            "decorators": decorators,
            "parameters": params,
            "class_name": self.current_class,
            "is_method": self.current_class is not None,
            "is_static_method": is_static,
            "is_class_method": is_classmethod,
            "is_instance_method": self.current_class is not None
            and has_self
            and not is_static,
            "full_name": (
                f"{'.'.join(self.class_stack)}.{node.name}"
                if self.class_stack
                else node.name
            ),
        }

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return ast.unparse(decorator)


# ============================================================================
# PART 2: ADVANCED - FIND ALL FUNCTIONS IN A FILE
# ============================================================================


class AllFunctionsFinder(ast.NodeVisitor):
    """Find ALL functions in code (useful for analysis)."""

    def __init__(self) -> None:
        self.functions: List[Dict[str, Any]] = []
        self.class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context."""
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit regular functions."""
        self.functions.append(self._extract_info(node, is_async=False))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async functions."""
        self.functions.append(self._extract_info(node, is_async=True))
        self.generic_visit(node)

    def _extract_info(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool
    ) -> Dict[str, Any]:
        """Extract minimal function info."""
        decorators = [
            d.id if isinstance(d, ast.Name) else ast.unparse(d)
            for d in node.decorator_list
        ]

        return {
            "name": node.name,
            "is_async": is_async,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "class_name": self.class_stack[-1] if self.class_stack else None,
            "is_method": bool(self.class_stack),
            "decorators": decorators,
            "is_static": "staticmethod" in decorators,
        }


class FunctionContainingLineFinder(ast.NodeVisitor):
    """Find the function that contains a specific line number."""

    def __init__(self, target_line: int, source_lines: List[str]):
        self.target_line = target_line
        self.source_lines = source_lines
        self.function_info: Optional[Dict[str, Any]] = None
        self.class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context."""
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check if function contains target line."""
        if node.lineno <= self.target_line <= (node.end_lineno or float("inf")):
            self.function_info = self._extract_info(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check if async function contains target line."""
        if node.lineno <= self.target_line <= (node.end_lineno or float("inf")):
            self.function_info = self._extract_info(node, is_async=True)
        self.generic_visit(node)

    def _extract_info(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool
    ) -> Dict[str, Any]:
        """Extract function type information."""
        decorators = []
        source_lines = self.source_lines

        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                decorators.append(d.id)
            elif isinstance(d, ast.Attribute):
                decorators.append(d.attr)

        is_static = "staticmethod" in decorators
        is_classmethod = "classmethod" in decorators

        params = [arg.arg for arg in node.args.args]
        has_self = len(params) > 0 and params[0] in ("self", "cls")

        # NEW: Extract full definition
        start_line = node.lineno - 1  # Convert to 0-indexed
        definition_lines = []

        # Read lines until we find the closing ':'
        for i in range(start_line, len(source_lines)):
            line = source_lines[i].rstrip("\n")
            definition_lines.append(line)

            # Stop when signature ends
            if line.rstrip().endswith(":"):
                break

        full_definition = "\n".join(definition_lines)

        return {
            "found": True,
            "name": node.name,
            "is_async": is_async,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "class_name": self.class_stack[-1] if self.class_stack else None,
            "is_method": bool(self.class_stack),
            "is_static_method": is_static,
            "is_class_method": is_classmethod,
            "is_instance_method": bool(self.class_stack) and has_self and not is_static,
            "decorators": decorators,
            "parameters": params,
            "has_self_parameter": has_self,
            "definition": source_lines[0],  # ADD THIS
            "full_definition": full_definition,
        }


class FunctionLocator(ast.NodeVisitor):
    """Locates function definitions and their call sites across a codebase."""

    def __init__(self, target_function: str):
        self.target_function = target_function
        self.definitions: List[Dict] = []
        self.calls: List[Dict] = []
        self.current_file: str = ""
        self.current_class: str = ""

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        """Track function definitions."""
        if node.name == self.target_function:
            self.definitions.append(
                {
                    "file": self.current_file,
                    "function": node.name,
                    "line": node.lineno,
                    "col": node.col_offset,
                    "class": self.current_class or None,
                    "decorators": [
                        d.id if isinstance(d, ast.Name) else str(d)
                        for d in node.decorator_list
                    ],
                    "args": [arg.arg for arg in node.args.args],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
            )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context for methods."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Call(self, node: ast.Call) -> None:
        """Track function calls."""
        func_name = None

        # Direct function call: function_a()
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

        # Method call: obj.function_a()
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name == self.target_function:
            self.calls.append(
                {
                    "file": self.current_file,
                    "line": node.lineno,
                    "col": node.col_offset,
                    "context": self._get_context(node),
                }
            )

        self.generic_visit(node)

    def _get_context(self, node: ast.AST) -> str:  # NOSONAR
        """Get the enclosing function/class context for a node."""
        # Walk up the tree to find enclosing function/class
        # This is simplified - you'd need to track parent nodes
        return self.current_class or "module_level"


# ============================================================================
# PART 3: PRACTICAL HELPER FUNCTIONS
# ============================================================================


def find_function(code: str, function_name: str) -> Optional[Dict[str, Any]]:
    """
    Find a function in Python code and return its information.

    Args:
        code: Python source code as string
        function_name: Name of function to find

    Returns:
        Dictionary with function info or None if not found

    Example:
        >>> code = '''
        ... async def fetch_data():
        ...     return "data"
        ... '''
        >>> info = find_function(code, "fetch_data")
        >>> print(info['is_async'])
        True
    """
    try:
        tree = ast.parse(code)
        finder = ClassMethodFinder(function_name)
        finder.visit(tree)
        return finder.function_info
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        return None


def find_all_functions(code: str) -> List[Dict[str, Any]]:
    """
    Find all functions in code.

    Returns:
        List of dictionaries with function information


    """
    try:
        tree = ast.parse(code)
        finder = AllFunctionsFinder()
        finder.visit(tree)
        return finder.functions
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return []


def detect_original_function_type(code: str, target_line: int) -> Dict[str, Any]:
    """
    Detect if the original function at target_line is async/sync and if it's a method.

    This is what you need for your LLM fixer!

    Args:
        code: code of file lines
        target_line: Line number where issue occurs (1-indexed)

    Returns:
        Dictionary with function type information

    Example:
        >>> code =
        ...     "class MyClass:\\n",
        ...     "    async def method(self):\\n",
        ...     "        x = 1\\n"
        ...
        >>> info = detect_original_function_type(code, 2)
        >>> info['is_async']
        True
        >>> info['is_instance_method']
        True
    """

    try:
        source_lines = code.split("\n")
        tree = ast.parse(code)

        # Find function containing target line
        finder = FunctionContainingLineFinder(target_line, source_lines)
        finder.visit(tree)

        if finder.function_info:
            return finder.function_info

        return {
            "found": False,
            "message": f"No function found containing line {target_line}",
        }

    except SyntaxError as e:
        return {"found": False, "error": str(e)}


def find_function_implementations(
    directory: Path, function_name: str, extensions: Tuple[str, ...] = (".py",)
) -> Dict:
    """
    Find all implementations and call sites of a function.

    Args:
        directory: Root directory to search
        function_name: Name of the function to find
        extensions: File extensions to search (default: Python files)

    Returns:
        Dictionary with 'definitions' and 'calls' lists
    """
    locator = FunctionLocator(function_name)

    # results
    _ = {"definitions": [], "calls": []}

    for file_path in directory.rglob("*"):
        if file_path.suffix not in extensions:
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            locator.current_file = str(file_path)
            locator.visit(tree)

        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return {"definitions": locator.definitions, "calls": locator.calls}


class AsyncConversionAnalyzer(ast.NodeVisitor):
    """Analyzes if a function can be safely converted between sync/async."""

    # Async-only operations that block sync conversion
    ASYNC_ONLY_OPERATIONS = {"await", "async with", "async for"}

    # Blocking sync operations (can't be easily made async)
    BLOCKING_SYNC_OPERATIONS = {
        "__init__",
        "__del__",
        "__enter__",
        "__exit__",
        "__getattr__",
        "__setattr__",
        "__getitem__",
        "__setitem__",
    }

    def __init__(self, target_function: str, codebase_root: Path):
        self.target_function = target_function
        self.codebase_root = codebase_root
        self.file_reader = AsyncFileReader()

        # Analysis results
        self.function_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = (
            None
        )
        self.is_async: bool = False
        self.file_path: str = ""

        # Track what the function does
        self.has_await_expressions: List[Dict] = []
        self.has_async_context_managers: List[Dict] = []
        self.has_async_iterations: List[Dict] = []
        self.calls_async_functions: List[Dict] = []
        self.calls_sync_blocking: List[Dict] = []

        # Track how the function is used
        self.called_by: List[Dict] = []
        self.decorated_with: List[str] = []

        # Special cases
        self.is_magic_method: bool = False
        self.is_property: bool = False
        self.is_classmethod: bool = False
        self.is_staticmethod: bool = False
        self.yields: bool = False

        # Internal state
        self.current_file: str = ""
        self.in_target_function: bool = False
        self.current_class: Optional[str] = None

    async def analyze(self) -> ConversionAnalysis:
        """Main analysis entry point."""
        # Step 1: Find the function definition
        await self._find_function_definition()

        if not self.function_node:
            return ConversionAnalysis(
                function_name=self.target_function,
                current_type="unknown",
                target_type="unknown",
                risk_level=ConversionRisk.IMPOSSIBLE,
                blocking_issues=["Function not found in codebase"],
                required_changes=[],
                caller_impact=[],
                internal_calls=[],
                suggestions=[],
            )

        # Step 2: Analyze function internals
        self._analyze_function_body()

        # Step 3: Find all callers
        await self._find_all_callers()

        # Step 4: Determine conversion feasibility
        return self._generate_analysis()

    async def _find_function_definition(self) -> None:
        """Locate the function definition in the codebase."""
        for file_path in self.codebase_root.rglob("*.py"):
            try:
                content = await self.file_reader.read_text(file_path)

                tree = ast.parse(content, filename=str(file_path))
                self.current_file = str(file_path)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == self.target_function:
                            self.function_node = node
                            self.is_async = isinstance(node, ast.AsyncFunctionDef)
                            self.file_path = str(file_path)

                            # Check decorators
                            self._analyze_decorators(node)

                            # Check if it's a magic method
                            self.is_magic_method = node.name.startswith(
                                "__"
                            ) and node.name.endswith("__")

                            return

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    def _analyze_decorators(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        """Extract and categorize decorators."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                dec_name = decorator.id
                self.decorated_with.append(dec_name)

                if dec_name == "property":
                    self.is_property = True
                elif dec_name == "classmethod":
                    self.is_classmethod = True
                elif dec_name == "staticmethod":
                    self.is_staticmethod = True

    def _analyze_function_body(self) -> None:
        """Analyze what the function does internally."""
        if not self.function_node:
            return

        self.in_target_function = True
        self.visit(self.function_node)
        self.in_target_function = False

    def visit_Await(self, node: ast.Await) -> None:
        """Track await expressions."""
        if self.in_target_function:
            self.has_await_expressions.append({"line": node.lineno, "type": "await"})
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Track async context managers."""
        if self.in_target_function:
            self.has_async_context_managers.append(
                {"line": node.lineno, "type": "async with"}
            )
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Track async iterations."""
        if self.in_target_function:
            self.has_async_iterations.append({"line": node.lineno, "type": "async for"})
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        """Track if function is a generator."""
        if self.in_target_function:
            self.yields = True
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """Track yield from expressions."""
        if self.in_target_function:
            self.yields = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track function calls - identify sync/async calls."""
        if self.in_target_function:
            func_name = self._get_call_name(node)

            if func_name:
                # Check if it's a known blocking I/O operation
                if self._is_blocking_io_call(func_name):
                    self.calls_sync_blocking.append(
                        {
                            "line": node.lineno,
                            "function": func_name,
                            "reason": "Blocking I/O operation",
                        }
                    )

                # Check if calling an async function (without await - error!)
                if self._is_likely_async_call(func_name) and not self._is_awaited(node):
                    self.calls_async_functions.append(
                        {"line": node.lineno, "function": func_name, "awaited": False}
                    )

        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract function name from call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _is_blocking_io_call(self, func_name: str) -> bool:
        """Check if function is a known blocking I/O operation."""
        blocking_patterns = {
            "open",
            "read",
            "write",
            "connect",
            "send",
            "recv",
            "sleep",  # time.sleep
            "request",
            "get",
            "post",
            "put",
            "delete",  # requests library
            "execute",
            "fetchall",
            "fetchone",
            "commit",  # database
            "load",
            "dump",  # file operations
        }
        return func_name in blocking_patterns

    def _is_likely_async_call(self, func_name: str) -> bool:
        """Heuristic to detect if a function is likely async."""
        async_patterns = {"async", "await", "aio", "coroutine"}
        return any(pattern in func_name.lower() for pattern in async_patterns)

    def _is_awaited(self, node: ast.Call) -> bool:  # NOSONAR
        """Check if a call is wrapped in await (simplified)."""
        # This is a simplified check - full implementation would need parent tracking
        return False

    async def _find_all_callers(self) -> None:
        """Find all places where this function is called."""
        for file_path in self.codebase_root.rglob("*.py"):
            try:
                content = await self.file_reader.read_text(file_path)

                tree = ast.parse(content, filename=str(file_path))
                self._find_callers_in_tree(tree, str(file_path))

            except Exception:
                logger.warning(
                    "Unexpected error while analyzing %s",
                    file_path,
                    exc_info=True,
                )
                continue

    def _find_callers_in_tree(self, tree: ast.AST, file_path: str) -> None:
        """Find callers in a specific AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)

                if func_name == self.target_function:
                    # Check if it's awaited
                    is_awaited = self._check_if_awaited_in_context(node, tree)

                    self.called_by.append(
                        {
                            "file": file_path,
                            "line": node.lineno,
                            "awaited": is_awaited,
                            "context": self._get_caller_context(node, tree),
                        }
                    )

    def _check_if_awaited_in_context(
        self,
        call_node: ast.Call,
        tree: ast.AST,  # NOSONAR
    ) -> bool:
        """Check if a call is awaited by examining parent nodes."""
        # Simplified - would need proper parent tracking
        for node in ast.walk(tree):
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                if self._get_call_name(node.value) == self.target_function:
                    return True
        return False

    def _get_caller_context(self, node: ast.Call, tree: ast.AST) -> str:  # NOSONAR
        """Get the function/class context of the caller."""
        # Simplified - would need parent tracking
        return "unknown"

    def _generate_analysis(self) -> ConversionAnalysis:
        """Generate final conversion analysis."""
        current_type = "async" if self.is_async else "sync"
        target_type = "sync" if self.is_async else "async"

        blocking_issues: List[str] = []
        required_changes: List[str] = []
        suggestions: List[str] = []

        # Determine conversion direction and feasibility
        if self.is_async:
            # Converting async → sync
            risk = self._analyze_async_to_sync(
                blocking_issues, required_changes, suggestions
            )
        else:
            # Converting sync → async
            risk = self._analyze_sync_to_async(
                blocking_issues, required_changes, suggestions
            )

        return ConversionAnalysis(
            function_name=self.target_function,
            current_type=current_type,
            target_type=target_type,
            risk_level=risk,
            blocking_issues=blocking_issues,
            required_changes=required_changes,
            caller_impact=self.called_by,
            internal_calls=self.calls_sync_blocking + self.calls_async_functions,
            suggestions=suggestions,
        )

    def _analyze_async_to_sync(
        self,
        blocking_issues: List[str],
        required_changes: List[str],
        suggestions: List[str],
    ) -> ConversionRisk:
        """Analyze conversion from async to sync."""

        # BLOCKING: Has await expressions
        if self.has_await_expressions:
            blocking_issues.append(
                f"❌ Function uses {len(self.has_await_expressions)} await expressions "
                f"- cannot be made sync without rewriting"
            )
            return ConversionRisk.IMPOSSIBLE

        # BLOCKING: Has async context managers
        if self.has_async_context_managers:
            blocking_issues.append(
                "❌ Function uses async context managers (async with) "
                "- requires sync equivalents"
            )
            return ConversionRisk.IMPOSSIBLE

        # BLOCKING: Has async iterations
        if self.has_async_iterations:
            blocking_issues.append(
                "❌ Function uses async iterations (async for) "
                "- requires sync equivalents"
            )
            return ConversionRisk.IMPOSSIBLE

        # Check caller impact
        awaited_callers = [c for c in self.called_by if c["awaited"]]

        if awaited_callers:
            required_changes.append(
                f"⚠️  {len(awaited_callers)} callers use 'await' - must remove await calls"
            )
            suggestions.append("Update all callers to call function without 'await'")
            return ConversionRisk.BREAKING

        if self.is_property:
            suggestions.append("✓ Property decorator is compatible with sync")

        suggestions.append(
            "✓ Safe to convert: remove 'async' keyword from function definition"
        )
        return ConversionRisk.SAFE

    def _analyze_sync_to_async(
        self,
        blocking_issues: List[str],
        required_changes: List[str],
        suggestions: List[str],
    ) -> ConversionRisk:
        """Analyze conversion from sync to async."""

        # BLOCKING: Magic methods (except some special cases)
        if self.is_magic_method and self.target_function not in {
            "__aenter__",
            "__aexit__",
            "__aiter__",
            "__anext__",
        }:
            blocking_issues.append(
                f"❌ Magic method '{self.target_function}' cannot be async "
                f"(Python doesn't support async magic methods)"
            )
            return ConversionRisk.IMPOSSIBLE

        # BLOCKING: Property
        if self.is_property:
            blocking_issues.append(
                "❌ Properties cannot be async - use regular async methods instead"
            )
            return ConversionRisk.IMPOSSIBLE

        # BREAKING: Generator functions
        if self.yields:
            blocking_issues.append(
                "⚠️  Generator function - convert to async generator (async def + yield)"
            )
            required_changes.append(
                "Convert to async generator and update all callers to use 'async for'"
            )
            return ConversionRisk.BREAKING

        # Identify blocking I/O that should be made async
        if self.calls_sync_blocking:
            required_changes.append(
                f"⚠️  Function calls {len(self.calls_sync_blocking)} blocking I/O operations"
            )
            for call in self.calls_sync_blocking:
                suggestions.append(
                    f"Line {call['line']}: Replace '{call['function']}' with async equivalent"
                )
            risk_level = ConversionRisk.NEEDS_CHANGES
        else:
            risk_level = ConversionRisk.SAFE

        # Check caller impact
        non_awaited_callers = [c for c in self.called_by if not c["awaited"]]

        if non_awaited_callers:
            required_changes.append(
                f"⚠️  {len(non_awaited_callers)} callers must add 'await' keyword"
            )
            required_changes.append(
                "⚠️  All calling functions may need to become async too (cascading change)"
            )

            for caller in non_awaited_callers[:5]:  # Show first 5
                suggestions.append(
                    f"Update {caller['file']}:{caller['line']} to await this call"
                )

            if len(non_awaited_callers) > 5:
                suggestions.append(
                    f"... and {len(non_awaited_callers) - 5} more callers"
                )

            risk_level = ConversionRisk.BREAKING

        # Success suggestions
        if risk_level == ConversionRisk.SAFE:
            suggestions.append("✓ Add 'async' keyword to function definition")
            suggestions.append(
                "✓ All callers already use 'await' - no breaking changes"
            )
        elif risk_level == ConversionRisk.NEEDS_CHANGES:
            suggestions.append("⚠️  Replace blocking I/O with async equivalents:")
            suggestions.append("   - time.sleep() → asyncio.sleep()")
            suggestions.append("   - requests.get() → aiohttp.get()")
            suggestions.append("   - open() → aiofiles.open()")
            suggestions.append("   - db.execute() → await db.execute()")

        return risk_level
