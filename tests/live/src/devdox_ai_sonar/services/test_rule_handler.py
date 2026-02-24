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
        """Scenario 2: No existing constant — create STRING_LITERAL_N."""
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
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        assert '"application/json"' in response.NEW_HELPER_CODE
        # Constants block + 3 replacements
        assert response.FIXED_CODE_BLOCKS[0].block_name == "New constants"
        blocks = _replacement_blocks(response)
        assert len(blocks) == 3
        for block in blocks:
            assert block.replacements[0].replace == "STRING_LITERAL_1"

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
        # /api/v1/users → new STRING_LITERAL_1
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        assert '"/api/v1/users"' in response.NEW_HELPER_CODE
        assert "STRING_LITERAL_2" not in response.NEW_HELPER_CODE

        blocks = _replacement_blocks(response)
        replace_names = [b.replacements[0].replace for b in blocks]
        assert "CONTENT_TYPE" in replace_names
        assert "STRING_LITERAL_1" in replace_names

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
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        assert response.FIXED_CODE_BLOCKS[0].block_name == "New constants"
        blocks = _replacement_blocks(response)
        # 2 definitions + 3 inline = 5
        assert len(blocks) == 5
        for block in blocks:
            assert block.replacements[0].replace == "STRING_LITERAL_1"

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
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        blocks = _replacement_blocks(response)
        assert len(blocks) == 4
        for block in blocks:
            assert block.replacements[0].replace == "STRING_LITERAL_1"

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
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        blocks = _replacement_blocks(response)
        assert len(blocks) == 4
        for block in blocks:
            assert block.replacements[0].replace == "STRING_LITERAL_1"

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
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        blocks = _replacement_blocks(response)
        assert len(blocks) == 4
        for block in blocks:
            assert block.replacements[0].replace == "STRING_LITERAL_1"


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
        assert 'STRING_LITERAL_1 = "1234"' in rendered
        assert "**Lines affected:**" in rendered

        # Suggested Fix should show Original/Fixed code (from display blocks)
        assert "**Original:**" in rendered
        assert "**Fixed:**" in rendered

        # New constants block should appear
        assert "New constants" in rendered
