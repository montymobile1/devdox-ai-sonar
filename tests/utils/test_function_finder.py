"""Tests for devdox_ai_sonar.utils.function_finder module.

Tests the public API: find_function, find_all_functions,
detect_original_function_type, find_function_implementations,
and the AsyncConversionAnalyzer.
"""

import pytest
from pathlib import Path

from devdox_ai_sonar.utils.function_finder import (
    find_function,
    find_all_functions,
    detect_original_function_type,
    find_function_implementations,
    AsyncConversionAnalyzer,
)
from devdox_ai_sonar.models.file_structures import ConversionRisk


# ============================================================================
# FIXTURES - reusable code snippets
# ============================================================================


@pytest.fixture
def simple_sync_function():
    return "def greet(name):\n    return f'Hello {name}'\n"


@pytest.fixture
def simple_async_function():
    return "async def fetch_data():\n    return 'data'\n"


@pytest.fixture
def class_with_methods():
    return (
        "class MyService:\n"
        "    def sync_method(self):\n"
        "        return 1\n"
        "\n"
        "    async def async_method(self):\n"
        "        return 2\n"
        "\n"
        "    @staticmethod\n"
        "    def static_method():\n"
        "        return 3\n"
        "\n"
        "    @classmethod\n"
        "    def class_method(cls):\n"
        "        return 4\n"
    )


@pytest.fixture
def nested_classes():
    return (
        "class Outer:\n"
        "    class Inner:\n"
        "        def inner_method(self):\n"
        "            pass\n"
        "    def outer_method(self):\n"
        "        pass\n"
    )


@pytest.fixture
def code_with_multiple_functions():
    return (
        "import os\n"
        "\n"
        "def func_a():\n"
        "    pass\n"
        "\n"
        "async def func_b(x, y):\n"
        "    return x + y\n"
        "\n"
        "class Foo:\n"
        "    def func_c(self):\n"
        "        pass\n"
    )


# ============================================================================
# TESTS FOR find_function
# ============================================================================


class TestFindFunction:
    """Tests for the find_function public helper."""

    def test_finds_sync_function(self, simple_sync_function):
        result = find_function(simple_sync_function, "greet")

        assert result is not None
        assert result["name"] == "greet"
        assert result["is_async"] is False
        assert result["parameters"] == ["name"]

    def test_finds_async_function(self, simple_async_function):
        result = find_function(simple_async_function, "fetch_data")

        assert result is not None
        assert result["name"] == "fetch_data"
        assert result["is_async"] is True

    def test_returns_none_when_function_not_found(self, simple_sync_function):
        result = find_function(simple_sync_function, "nonexistent")

        assert result is None

    def test_returns_none_on_syntax_error(self):
        result = find_function("def broken(:\n    pass\n", "broken")

        assert result is None

    def test_finds_instance_method(self, class_with_methods):
        result = find_function(class_with_methods, "sync_method")

        assert result is not None
        assert result["is_method"] is True
        assert result["is_instance_method"] is True
        assert result["class_name"] == "MyService"

    def test_finds_async_method(self, class_with_methods):
        result = find_function(class_with_methods, "async_method")

        assert result is not None
        assert result["is_async"] is True
        assert result["is_method"] is True
        assert result["class_name"] == "MyService"

    def test_finds_static_method(self, class_with_methods):
        result = find_function(class_with_methods, "static_method")

        assert result is not None
        assert result["is_static_method"] is True
        assert result["is_instance_method"] is False

    def test_finds_class_method(self, class_with_methods):
        result = find_function(class_with_methods, "class_method")

        assert result is not None
        assert result["is_class_method"] is True
        assert result["parameters"] == ["cls"]

    def test_nested_class_method_has_full_name(self, nested_classes):
        result = find_function(nested_classes, "inner_method")

        assert result is not None
        assert result["full_name"] == "Outer.Inner.inner_method"
        assert result["class_name"] == "Inner"

    def test_outer_method_after_inner_class(self, nested_classes):
        result = find_function(nested_classes, "outer_method")

        assert result is not None
        assert result["class_name"] == "Outer"
        assert result["full_name"] == "Outer.outer_method"

    def test_empty_code_returns_none(self):
        result = find_function("", "anything")

        assert result is None

    def test_function_with_decorators(self):
        code = "@my_decorator\ndef decorated():\n    pass\n"
        result = find_function(code, "decorated")

        assert result is not None
        assert "my_decorator" in result["decorators"]


# ============================================================================
# TESTS FOR find_all_functions
# ============================================================================


class TestFindAllFunctions:
    """Tests for the find_all_functions public helper."""

    def test_finds_all_functions_in_module(self, code_with_multiple_functions):
        results = find_all_functions(code_with_multiple_functions)

        names = [f["name"] for f in results]
        assert "func_a" in names
        assert "func_b" in names
        assert "func_c" in names

    def test_distinguishes_sync_and_async(self, code_with_multiple_functions):
        results = find_all_functions(code_with_multiple_functions)

        by_name = {f["name"]: f for f in results}
        assert by_name["func_a"]["is_async"] is False
        assert by_name["func_b"]["is_async"] is True

    def test_identifies_methods_vs_functions(self, code_with_multiple_functions):
        results = find_all_functions(code_with_multiple_functions)

        by_name = {f["name"]: f for f in results}
        assert by_name["func_a"]["is_method"] is False
        assert by_name["func_c"]["is_method"] is True
        assert by_name["func_c"]["class_name"] == "Foo"

    def test_returns_empty_list_for_no_functions(self):
        code = "x = 1\ny = 2\n"
        results = find_all_functions(code)

        assert results == []

    def test_returns_empty_list_on_syntax_error(self):
        results = find_all_functions("def broken(:\n")

        assert results == []

    def test_empty_code_returns_empty_list(self):
        results = find_all_functions("")

        assert results == []

    def test_detects_static_methods(self):
        code = (
            "class Svc:\n"
            "    @staticmethod\n"
            "    def helper():\n"
            "        pass\n"
        )
        results = find_all_functions(code)

        assert len(results) == 1
        assert results[0]["is_static"] is True


# ============================================================================
# TESTS FOR detect_original_function_type
# ============================================================================


class TestDetectOriginalFunctionType:
    """Tests for detect_original_function_type."""

    def test_detects_async_function_at_target_line(self):
        code = "async def my_func():\n    x = 1\n    return x\n"

        result = detect_original_function_type(code, target_line=1)

        assert result["found"] is True
        assert result["name"] == "my_func"
        assert result["is_async"] is True

    def test_detects_sync_function_at_target_line(self):
        code = "def my_func():\n    return 42\n"

        result = detect_original_function_type(code, target_line=1)

        assert result["found"] is True
        assert result["is_async"] is False

    def test_detects_function_when_line_is_inside_body(self):
        code = "def my_func():\n    x = 1\n    y = 2\n    return x + y\n"

        result = detect_original_function_type(code, target_line=3)

        assert result["found"] is True
        assert result["name"] == "my_func"

    def test_boundary_first_line_of_function(self):
        code = "x = 0\ndef my_func():\n    return 1\n"

        result = detect_original_function_type(code, target_line=2)

        assert result["found"] is True
        assert result["name"] == "my_func"

    def test_boundary_last_line_of_function(self):
        code = "def my_func():\n    x = 1\n    return x\n"

        result = detect_original_function_type(code, target_line=3)

        assert result["found"] is True
        assert result["name"] == "my_func"

    def test_returns_not_found_when_line_outside_any_function(self):
        code = "x = 1\n\ndef my_func():\n    pass\n"

        result = detect_original_function_type(code, target_line=1)

        assert result["found"] is False

    def test_detects_instance_method(self):
        code = (
            "class MyClass:\n"
            "    async def method(self):\n"
            "        x = 1\n"
        )

        result = detect_original_function_type(code, target_line=2)

        assert result["found"] is True
        assert result["is_async"] is True
        assert result["is_instance_method"] is True
        assert result["class_name"] == "MyClass"
        assert result["has_self_parameter"] is True

    def test_detects_static_method(self):
        code = (
            "class MyClass:\n"
            "    @staticmethod\n"
            "    def helper():\n"
            "        pass\n"
        )

        result = detect_original_function_type(code, target_line=3)

        assert result["found"] is True
        assert result["is_static_method"] is True
        assert result["is_instance_method"] is False

    def test_detects_classmethod(self):
        code = (
            "class MyClass:\n"
            "    @classmethod\n"
            "    def create(cls):\n"
            "        pass\n"
        )

        result = detect_original_function_type(code, target_line=3)

        assert result["found"] is True
        assert result["is_class_method"] is True

    def test_returns_error_on_syntax_error(self):
        result = detect_original_function_type("def broken(:\n", target_line=1)

        assert result["found"] is False
        assert "error" in result

    def test_multiline_signature(self):
        code = (
            "def long_func(\n"
            "    param_a,\n"
            "    param_b,\n"
            "):\n"
            "    return param_a + param_b\n"
        )

        result = detect_original_function_type(code, target_line=1)

        assert result["found"] is True
        assert result["name"] == "long_func"
        assert "param_a" in result["parameters"]
        assert "param_b" in result["parameters"]

    def test_includes_full_definition(self):
        code = (
            "def my_func(\n"
            "    x, y\n"
            "):\n"
            "    return x + y\n"
        )

        result = detect_original_function_type(code, target_line=1)

        assert result["found"] is True
        assert "full_definition" in result
        assert "def my_func(" in result["full_definition"]


# ============================================================================
# TESTS FOR find_function_implementations (filesystem search)
# ============================================================================


class TestFindFunctionImplementations:
    """Tests for find_function_implementations using tmp_path fixture."""

    def test_finds_definition_in_file(self, tmp_path):
        py_file = tmp_path / "module.py"
        py_file.write_text("def target_func():\n    pass\n")

        result = find_function_implementations(tmp_path, "target_func")

        assert len(result["definitions"]) == 1
        assert result["definitions"][0]["function"] == "target_func"

    def test_finds_call_sites(self, tmp_path):
        py_file = tmp_path / "module.py"
        py_file.write_text(
            "def target_func():\n"
            "    pass\n"
            "\n"
            "def caller():\n"
            "    target_func()\n"
        )

        result = find_function_implementations(tmp_path, "target_func")

        assert len(result["calls"]) == 1
        assert result["calls"][0]["line"] == 5

    def test_finds_across_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def shared_func():\n    pass\n")
        (tmp_path / "b.py").write_text("def caller():\n    shared_func()\n")

        result = find_function_implementations(tmp_path, "shared_func")

        assert len(result["definitions"]) == 1
        assert len(result["calls"]) == 1

    def test_ignores_non_python_files(self, tmp_path):
        (tmp_path / "data.txt").write_text("def target_func():\n    pass\n")

        result = find_function_implementations(tmp_path, "target_func")

        assert result["definitions"] == []
        assert result["calls"] == []

    def test_returns_empty_when_not_found(self, tmp_path):
        (tmp_path / "module.py").write_text("def other():\n    pass\n")

        result = find_function_implementations(tmp_path, "nonexistent")

        assert result["definitions"] == []
        assert result["calls"] == []

    def test_handles_syntax_error_in_file_gracefully(self, tmp_path):
        (tmp_path / "broken.py").write_text("def broken(:\n    pass\n")
        (tmp_path / "good.py").write_text("def target_func():\n    pass\n")

        result = find_function_implementations(tmp_path, "target_func")

        assert len(result["definitions"]) == 1

    def test_finds_async_definition(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "async def target_func():\n    return 1\n"
        )

        result = find_function_implementations(tmp_path, "target_func")

        assert len(result["definitions"]) == 1
        assert result["definitions"][0]["is_async"] is True

    def test_finds_method_call_via_attribute(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "class Svc:\n"
            "    def do_work(self):\n"
            "        pass\n"
            "\n"
            "svc = Svc()\n"
            "svc.do_work()\n"
        )

        result = find_function_implementations(tmp_path, "do_work")

        assert len(result["definitions"]) == 1
        assert len(result["calls"]) == 1

    def test_empty_directory(self, tmp_path):
        result = find_function_implementations(tmp_path, "anything")

        assert result == {"definitions": [], "calls": []}

    def test_searches_subdirectories(self, tmp_path):
        sub = tmp_path / "pkg" / "sub"
        sub.mkdir(parents=True)
        (sub / "deep.py").write_text("def target_func():\n    pass\n")

        result = find_function_implementations(tmp_path, "target_func")

        assert len(result["definitions"]) == 1


# ============================================================================
# TESTS FOR AsyncConversionAnalyzer
# ============================================================================


class TestAsyncConversionAnalyzer:
    """Tests for AsyncConversionAnalyzer.analyze()."""

    def test_async_func_without_await_is_safe_to_convert(self, tmp_path):
        """Async function with no await/async-with/async-for is safe to make sync."""
        (tmp_path / "module.py").write_text(
            "async def target():\n"
            "    return 42\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.current_type == "async"
        assert result.target_type == "sync"
        assert result.risk_level == ConversionRisk.SAFE

    def test_async_func_with_await_is_impossible(self, tmp_path):
        """Async function using await cannot be trivially converted to sync."""
        (tmp_path / "module.py").write_text(
            "import asyncio\n"
            "async def target():\n"
            "    await asyncio.sleep(1)\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.IMPOSSIBLE
        assert any("await" in issue for issue in result.blocking_issues)

    def test_async_func_with_async_with_is_impossible(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "async def target():\n"
            "    async with open('f') as f:\n"
            "        pass\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.IMPOSSIBLE
        assert any("async context" in issue for issue in result.blocking_issues)

    def test_async_func_with_async_for_is_impossible(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "async def target():\n"
            "    async for item in some_iter():\n"
            "        pass\n"
            "async def some_iter():\n"
            "    yield 1\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.IMPOSSIBLE
        assert any("async iteration" in issue for issue in result.blocking_issues)

    def test_async_func_with_awaited_callers_is_breaking(self, tmp_path):
        """If callers use await, removing async is a breaking change."""
        (tmp_path / "module.py").write_text(
            "async def target():\n"
            "    return 1\n"
            "\n"
            "async def caller():\n"
            "    result = await target()\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.BREAKING
        assert len(result.caller_impact) > 0

    def test_function_not_found_returns_impossible(self, tmp_path):
        (tmp_path / "module.py").write_text("def other():\n    pass\n")

        analyzer = AsyncConversionAnalyzer("nonexistent", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.IMPOSSIBLE
        assert result.current_type == "unknown"
        assert any("not found" in issue for issue in result.blocking_issues)

    def test_sync_func_safe_to_convert_to_async(self, tmp_path):
        """Sync function with no blocking I/O and no callers is safe."""
        (tmp_path / "module.py").write_text(
            "def target():\n"
            "    return 42\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.current_type == "sync"
        assert result.target_type == "async"
        assert result.risk_level == ConversionRisk.SAFE

    def test_sync_magic_method_cannot_be_async(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "class Foo:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
        )

        analyzer = AsyncConversionAnalyzer("__init__", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.IMPOSSIBLE
        assert any("Magic method" in issue for issue in result.blocking_issues)

    def test_sync_property_cannot_be_async(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "class Foo:\n"
            "    @property\n"
            "    def value(self):\n"
            "        return 1\n"
        )

        analyzer = AsyncConversionAnalyzer("value", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.IMPOSSIBLE
        assert any("Properties" in issue for issue in result.blocking_issues)

    def test_sync_generator_is_breaking(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "def target():\n"
            "    yield 1\n"
            "    yield 2\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.BREAKING
        assert any("Generator" in issue for issue in result.blocking_issues)

    def test_sync_func_with_blocking_io_needs_changes(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "import time\n"
            "def target():\n"
            "    time.sleep(1)\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level in (
            ConversionRisk.NEEDS_CHANGES,
            ConversionRisk.BREAKING,
        )

    def test_sync_func_with_non_awaited_callers_is_breaking(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "def target():\n"
            "    return 1\n"
            "\n"
            "def caller():\n"
            "    target()\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.risk_level == ConversionRisk.BREAKING

    def test_handles_file_read_errors_gracefully(self, tmp_path):
        """Analyzer should not crash if a file can't be read."""
        (tmp_path / "good.py").write_text(
            "async def target():\n    return 1\n"
        )
        bad = tmp_path / "bad.py"
        bad.mkdir()  # directory pretending to be .py

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert result.function_name == "target"
        assert result.risk_level == ConversionRisk.SAFE

    def test_detects_decorator_property_flag(self, tmp_path):
        (tmp_path / "module.py").write_text(
            "class Svc:\n"
            "    @staticmethod\n"
            "    def target():\n"
            "        return 1\n"
        )

        analyzer = AsyncConversionAnalyzer("target", tmp_path)
        result = analyzer.analyze()

        assert analyzer.is_staticmethod is True
        assert result.risk_level == ConversionRisk.SAFE
