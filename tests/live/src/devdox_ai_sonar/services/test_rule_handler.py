from pathlib import Path
from unittest.mock import Mock

from devdox_ai_sonar.services.rule_handler import StringLiteralDuplicateHandler
from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.models.file_structures import FixContext


def _make_context(**overrides) -> FixContext:
    """Build a FixContext with sensible defaults, overridable via kwargs."""
    defaults = dict(
        file_path=Path("/project/src/module.py"),
        file_path_tmp=Path("/tmp/module.py"),
        line_range={"first_line": 10, "last_line": 20, "problem_lines": [10, 15]},
        code_content="async def my_func(self, camelCase):\n    return 1\n",
        language="python",
        import_section={"start_line": 1, "end_line": 3},
        class_name=None,
        functions=[{"name": "my_func", "start_line": 10}],
        context_dict={
            "start_line": 10,
            "import_section": {"end_line": 3},
            "functions": [{"name": "my_func", "start_line": 10}],
            "new_context": [
                {"context": "async def my_func(self, camelCase):\n    return 1\n"}
            ],
        },
    )
    defaults.update(overrides)
    return FixContext(**defaults)


def _replacement_blocks(response):
    """Return only SEARCH_REPLACE blocks, filtering out the constants block."""
    return [b for b in response.FIXED_CODE_BLOCKS if b.replacements]


# ============================================================================
# EXISTING CONSTANT REUSE — END-TO-END SCENARIOS
# ============================================================================


class TestExistingConstantReuse:
    """End-to-end tests for StringLiteralDuplicateHandler with existing constants."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()
        self.project_path = Path("/project")

    def _make_issue(self, literal: str, count: int = 3) -> Mock:
        issue = Mock()
        issue.message = (
            f"Define a constant instead of duplicating this literal "
            f'"{literal}" {count} times.'
        )
        issue.first_line = 1
        issue.rule = "python:S1192"
        return issue

    async def test_scenario_1_reuse_existing_constant(self, tmp_path):
        """Scenario 1: Single existing module-level constant — reuse it."""
        source = (
            'APP_JSON = "application/json"\n'
            "\n"
            "def fetch_users():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_orders():\n"
            '    response = post(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_items():\n"
            '    response = put(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # Should reuse APP_JSON — no new constant, no constants block
        assert response.NEW_HELPER_CODE == ""
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].replace == "APP_JSON"

    async def test_scenario_2_no_existing_constant(self, tmp_path):
        """Scenario 2: No existing constant — create a new named constant."""
        source = (
            "def fetch_users():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_orders():\n"
            '    response = post(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_items():\n"
            '    response = put(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # ConstantNamingService generates APPLICATION_JSON from "application/json"
        assert "APPLICATION_JSON" in response.NEW_HELPER_CODE
        assert "'application/json'" in response.NEW_HELPER_CODE
        # Constants block + 3 replacements
        assert response.FIXED_CODE_BLOCKS[0].block_name == "New constants"
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].replace == "APPLICATION_JSON"

    async def test_scenario_3_mixed_usage(self, tmp_path):
        """Scenario 3: Existing constant + some sites already use it."""
        source = (
            'APP_JSON = "application/json"\n'
            "\n"
            "def fetch_users():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_orders():\n"
            '    response = post(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_items():\n"
            '    response = put(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_transport():\n"
            '    response = put(url, headers={"Content-Type": APP_JSON})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        assert response.NEW_HELPER_CODE == ""
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].replace == "APP_JSON"

    async def test_scenario_4_multiple_different_strings(self, tmp_path):
        """Scenario 4: Two different literals — one has constant, one doesn't."""
        source = (
            'CONTENT_TYPE = "application/json"\n'
            "\n"
            "def fetch_users():\n"
            '    response = get("/api/v1/users", headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_orders():\n"
            '    response = get("/api/v1/users", headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch_items():\n"
            '    response = get("/api/v1/users", headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issues = [
            self._make_issue("application/json"),
            self._make_issue("/api/v1/users"),
        ]
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            issues, context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]

        # application/json → reuse CONTENT_TYPE (no new constant for it)
        # /api/v1/users → new named constant (naming service generates a name)
        assert "'/api/v1/users'" in response.NEW_HELPER_CODE
        # Only one new constant should be created
        assert response.NEW_HELPER_CODE.count("=") == 1

        blocks = _replacement_blocks(response)
        replace_names = [b.replacements[0].replace for b in blocks]
        assert "CONTENT_TYPE" in replace_names
        # The /api/v1/users literal gets a generated name (not CONTENT_TYPE)
        non_ct_names = [n for n in replace_names if n != "CONTENT_TYPE"]
        assert len(non_ct_names) == 3
        assert len(set(non_ct_names)) == 1  # all use the same generated name

    async def test_scenario_5_type_annotated_constant(self, tmp_path):
        """Scenario 5: Type-annotated assignment (AnnAssign) — reuse it."""
        source = (
            'APP_JSON: str = "application/json"\n'
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        assert response.NEW_HELPER_CODE == ""
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].replace == "APP_JSON"

    async def test_scenario_6_lowercase_variable_reused(self, tmp_path):
        """Scenario 6: Lowercase variable name — still reuse it."""
        source = (
            'app_json = "application/json"\n'
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        assert response.NEW_HELPER_CODE == ""
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].replace == "app_json"

    async def test_scenario_7_multiple_constants_create_new(self, tmp_path):
        """Scenario 7: Multiple module-level constants with same value — ambiguous, create new."""
        source = (
            'APP_JSON = "application/json"\n'
            'CONTENT_TYPE_JSON = "application/json"\n'
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json", count=5)
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # Naming service generates APPLICATION_JSON (collides with existing APP_JSON
        # and CONTENT_TYPE_JSON, but those are existing constants with the same value,
        # not in existing_module_names since existing_module_names only collects
        # UPPERCASE names from ast.Assign — and both ARE Assign nodes, so they ARE
        # in existing_module_names. The naming service will make_unique.)
        assert "'application/json'" in response.NEW_HELPER_CODE
        assert response.FIXED_CODE_BLOCKS[0].block_name == "New constants"
        blocks = _replacement_blocks(response)
        # 2 definitions + 3 inline = 5
        assert len(blocks) == 5
        # All replacements use the same generated constant name
        const_name = blocks[0].replacements[0].replace
        for block in blocks:
            assert block.replacements[0].replace == const_name

    async def test_scenario_8_class_level_not_reusable(self, tmp_path):
        """Scenario 8: Constant inside a class — NOT reusable, all replaced."""
        source = (
            "class Config:\n"
            '    APP_JSON = "application/json"\n'
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json", count=4)
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        assert "APPLICATION_JSON" in response.NEW_HELPER_CODE
        blocks = _replacement_blocks(response)
        assert len(blocks) == 4
        for block in blocks:
            assert block.replacements[0].replace == "APPLICATION_JSON"

    async def test_scenario_9_function_local_not_reusable(self, tmp_path):
        """Scenario 9: Assignment inside function — NOT reusable, all replaced."""
        source = (
            "def setup():\n"
            '    content_type = "application/json"\n'
            "    return content_type\n"
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json", count=4)
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        assert "APPLICATION_JSON" in response.NEW_HELPER_CODE
        blocks = _replacement_blocks(response)
        assert len(blocks) == 4
        for block in blocks:
            assert block.replacements[0].replace == "APPLICATION_JSON"

    async def test_scenario_10_collection_assignment_not_reusable(self, tmp_path):
        """Scenario 10: String in dict assignment — NOT reusable, all replaced."""
        source = (
            'HEADERS = {"Content-Type": "application/json"}\n'
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json", count=4)
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        assert "APPLICATION_JSON" in response.NEW_HELPER_CODE
        blocks = _replacement_blocks(response)
        assert len(blocks) == 4
        for block in blocks:
            assert block.replacements[0].replace == "APPLICATION_JSON"


# ============================================================================
# MARKDOWN RENDERING — END-TO-END
# ============================================================================


class TestMarkdownRendering:
    """End-to-end test: run handler, render through the Jinja2 template, save output."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()

    def _make_issue(self, literal: str, count: int = 3) -> Mock:
        issue = Mock()
        issue.message = (
            f"Define a constant instead of duplicating this literal "
            f'"{literal}" {count} times.'
        )
        issue.first_line = 1
        issue.rule = "python:S1192"
        issue.severity = "MAJOR"
        issue.file_path = "src/module.py"
        issue.line = 2
        return issue

    async def test_full_markdown_render(self, tmp_path):
        """Run handler → build display blocks → render template → save .md file."""
        source = (
            'APP_JSON = "application/json"\n'
            "\n"
            "def fetch():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch2():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch3():\n"
            '    response = get(url, headers={"Content-Type": "application/json"})\n'
            "\n"
            "def fetch4():\n"
            '    response = post(url, body="1234")\n'
            "\n"
            "def fetch5():\n"
            '    response = post(url, body="1234")\n'
            "\n"
            "def fetch6():\n"
            '    response = post(url, body="1234")\n'
        )
        src_file = tmp_path / "module.py"
        src_file.write_text(source)

        issues = [
            self._make_issue("application/json"),
            self._make_issue("1234"),
        ]
        context = _make_context(file_path=src_file)

        result = await self.handler.generate_fixes(
            issues, context, tmp_path, src_file, llm_caller=None
        )
        assert result is not None
        fix_response = result[0]

        # Build display blocks (converts SEARCH_REPLACE → DIFF with real code)
        display_blocks = LLMFixer._build_display_blocks(
            fix_response.FIXED_CODE_BLOCKS, project_path=tmp_path
        )

        # Render through the Jinja2 template
        from jinja2 import Environment, FileSystemLoader

        templates_dir = (
            Path(__file__).resolve().parents[5]
            / "src"
            / "devdox_ai_sonar"
            / "templates"
        )
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("md.j2")

        rendered = template.render(
            rule="python:S1192",
            severity="MAJOR",
            message='Define a constant instead of duplicating this literal "application/json" 3 times.',
            file_path="src/module.py",
            line=2,
            explanation=fix_response.EXPLANATION,
            suggestion=display_blocks,
            original_code={},
        )

        # Save to a .md file for manual inspection
        output_dir = Path(__file__).resolve().parents[5] / "tests" / "live"
        output_dir.mkdir(parents=True, exist_ok=True)
        md_file = output_dir / "sample_s1192_output.md"
        md_file.write_text(rendered.strip() + "\n")

        # Assertions on the rendered markdown
        assert "## 🔍 Issue: `python:S1192`" in rendered
        assert "### 🧠 Explanation" in rendered
        assert "**File:**" in rendered
        assert "### 🛠 Suggested Fix" in rendered

        # Explanation should contain structured per-literal info
        assert "Created" in rendered
        assert "Reused existing constant" in rendered
        assert "`APP_JSON`" in rendered
        assert "= '1234'" in rendered
        assert "**Lines affected:**" in rendered

        # Suggested Fix should show Original/Fixed code (from display blocks)
        assert "**Original:**" in rendered
        assert "**Fixed:**" in rendered

        # New constants block should appear
        assert "New constants" in rendered


# ============================================================================
# ADDITIONAL EDGE-CASE SCENARIOS — END-TO-END
# ============================================================================


class TestAdditionalEdgeCaseScenarios:
    """Additional edge-case end-to-end scenarios for StringLiteralDuplicateHandler."""

    def setup_method(self):
        self.handler = StringLiteralDuplicateHandler()
        self.project_path = Path("/project")

    def _make_issue(self, literal: str, count: int = 3) -> Mock:
        issue = Mock()
        issue.message = (
            f"Define a constant instead of duplicating this literal "
            f'"{literal}" {count} times.'
        )
        issue.first_line = 1
        issue.rule = "python:S1192"
        return issue

    async def test_scenario_11_string_in_list_comprehension(self, tmp_path):
        """Scenario 11: Strings in list comprehensions are replaced."""
        source = (
            'result1 = [x for x in items if x == "active"]\n'
            'result2 = [x for x in items if x == "active"]\n'
            'result3 = [x for x in items if x == "active"]\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("active")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].search == '"active"'

    async def test_scenario_12_string_in_raise(self, tmp_path):
        """Scenario 12: Strings in raise statements are replaced."""
        source = (
            "def validate_a(x):\n"
            '    raise ValueError("invalid input")\n'
            "\n"
            "def validate_b(x):\n"
            '    raise TypeError("invalid input")\n'
            "\n"
            "def validate_c(x):\n"
            '    raise RuntimeError("invalid input")\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("invalid input")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3

    async def test_scenario_13_same_literal_twice_on_same_line(self, tmp_path):
        """Scenario 13: Two occurrences on the same source line."""
        source = (
            'x = {"key": "value", "other": "value"}\n'
            'y = "value"\n'
            'z = "value"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("value", count=4)
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        blocks = _replacement_blocks(response)
        assert len(blocks) == 4
        # Two of the blocks should be on line 1
        line1_blocks = [b for b in blocks if b.start_line == 1]
        assert len(line1_blocks) == 2

    async def test_scenario_14_naming_collision_same_generated_name(self, tmp_path):
        """Scenario 14: Two literals that would generate the same name get distinct names."""
        source = (
            "def foo():\n"
            '    return "hello world"\n'
            "\n"
            "def bar():\n"
            '    return "hello world"\n'
            "\n"
            "def baz():\n"
            '    return "hello world"\n'
            "\n"
            "def qux():\n"
            '    return "hello/world"\n'
            "\n"
            "def quux():\n"
            '    return "hello/world"\n'
            "\n"
            "def corge():\n"
            '    return "hello/world"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issues = [
            self._make_issue("hello world"),
            self._make_issue("hello/world"),
        ]
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            issues, context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        blocks = _replacement_blocks(response)
        replace_names = {b.replacements[0].replace for b in blocks}
        # Must have 2 distinct constant names
        assert len(replace_names) == 2
        assert len(blocks) == 6

    async def test_scenario_15_annassign_collision_with_naming(self, tmp_path):
        """Scenario 15: AnnAssign UPPERCASE constant — collision risk with naming service."""
        source = (
            'APPLICATION_JSON: str = "something_else"\n'
            "\n"
            "def foo():\n"
            '    return "application/json"\n'
            "\n"
            "def bar():\n"
            '    return "application/json"\n'
            "\n"
            "def baz():\n"
            '    return "application/json"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("application/json")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # New constant is created (AnnAssign with different value is not reused)
        assert response.NEW_HELPER_CODE != ""
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3

    async def test_scenario_16_assign_and_annassign_same_literal(self, tmp_path):
        """Scenario 16: Both Assign and AnnAssign for same literal → ambiguous → create new."""
        source = (
            'X = "value"\n'
            'Y: str = "value"\n'
            "\n"
            "def foo():\n"
            '    return "value"\n'
            "\n"
            "def bar():\n"
            '    return "value"\n'
            "\n"
            "def baz():\n"
            '    return "value"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("value", count=5)
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # 2 existing constants → ambiguous → creates new
        assert response.NEW_HELPER_CODE != ""
        blocks = _replacement_blocks(response)
        # All 5 occurrences (including both definitions) are replaced
        assert len(blocks) == 5

    async def test_scenario_17_only_definition_no_other_occurrences(self, tmp_path):
        """Scenario 17: Only the constant definition exists — returns None."""
        source = 'MY_CONST = "only_here"\n'
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("only_here")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is None

    async def test_scenario_18_string_in_decorator_and_function(self, tmp_path):
        """Scenario 18: String appears in both decorator args and function bodies."""
        source = (
            '@app.route("/api/users")\n'
            "def get_users():\n"
            '    log("Accessing /api/users")\n'
            "\n"
            "def check_route():\n"
            '    return "/api/users"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("/api/users")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        blocks = _replacement_blocks(response)
        # Only the exact string "/api/users" is found, not inside the log message
        # The decorator and return statement have exact matches
        assert len(blocks) >= 2

    async def test_scenario_19_string_in_dunder_all(self, tmp_path):
        """Scenario 19: String in __all__ is replaced (documents current behavior).

        Note: Replacing strings in __all__ with constant references is valid Python
        but may be unexpected. This test documents the handler does NOT exempt __all__.
        """
        source = (
            '__all__ = ["my_module", "other"]\n'
            'x = "my_module"\n'
            'y = "my_module"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("my_module")
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        blocks = _replacement_blocks(response)
        # All 3 occurrences replaced, including __all__
        assert len(blocks) == 3

    async def test_scenario_20_mixed_ast_contexts_with_existing_constant(self, tmp_path):
        """Scenario 20: Existing constant + occurrences in diverse AST contexts."""
        source = (
            'STATUS = "active"\n'
            "\n"
            'items = [x for x in data if x == "active"]\n'
            "\n"
            '@requires("active")\n'
            "def handler():\n"
            '    assert state == "active"\n'
        )
        file = tmp_path / "module.py"
        file.write_text(source)

        issue = self._make_issue("active", count=4)
        context = _make_context(file_path=file)

        result = await self.handler.generate_fixes(
            [issue], context, self.project_path, file, llm_caller=None
        )
        assert result is not None
        response = result[0]
        # Reuses existing STATUS constant
        assert response.NEW_HELPER_CODE == ""
        blocks = _replacement_blocks(response)
        # 3 non-definition occurrences: comprehension, decorator, assert
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].replace == "STATUS"
