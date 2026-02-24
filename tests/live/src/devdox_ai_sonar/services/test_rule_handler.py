from pathlib import Path
from unittest.mock import Mock

from devdox_ai_sonar.services.rule_handler import StringLiteralDuplicateHandler
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
        # Should reuse APP_JSON — no new constant
        assert response.NEW_HELPER_CODE == ""
        # Should only replace inline literals (3), NOT the definition line
        assert len(response.FIXED_CODE_BLOCKS) == 3
        for block in response.FIXED_CODE_BLOCKS:
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
        assert len(response.FIXED_CODE_BLOCKS) == 3
        for block in response.FIXED_CODE_BLOCKS:
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
        # 3 inline literals (not definition, not the APP_JSON reference)
        assert len(response.FIXED_CODE_BLOCKS) == 3
        for block in response.FIXED_CODE_BLOCKS:
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
        # Should NOT contain a constant def for application/json
        assert "STRING_LITERAL_2" not in response.NEW_HELPER_CODE

        # Check replacement names
        replace_names = [b.replacements[0].replace for b in response.FIXED_CODE_BLOCKS]
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
        assert len(response.FIXED_CODE_BLOCKS) == 3
        for block in response.FIXED_CODE_BLOCKS:
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
        assert len(response.FIXED_CODE_BLOCKS) == 3
        for block in response.FIXED_CODE_BLOCKS:
            assert block.replacements[0].replace == "app_json"

    async def test_scenario_7_multiple_constants_create_new(self, tmp_path):
        """Scenario 7: Multiple module-level constants with same value — create new, replace all."""
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
        # Should create a new constant since there are multiple
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        # Should replace ALL occurrences (2 definitions + 3 inline = 5)
        assert len(response.FIXED_CODE_BLOCKS) == 5
        for block in response.FIXED_CODE_BLOCKS:
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
        # Should create new constant (class-level not reusable)
        assert "STRING_LITERAL_1" in response.NEW_HELPER_CODE
        # All 4 occurrences replaced (1 class attr + 3 inline)
        assert len(response.FIXED_CODE_BLOCKS) == 4
        for block in response.FIXED_CODE_BLOCKS:
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
        # All 4 occurrences (1 local + 3 inline)
        assert len(response.FIXED_CODE_BLOCKS) == 4
        for block in response.FIXED_CODE_BLOCKS:
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
        # All 4 occurrences (1 in dict + 3 inline)
        assert len(response.FIXED_CODE_BLOCKS) == 4
        for block in response.FIXED_CODE_BLOCKS:
            assert block.replacements[0].replace == "STRING_LITERAL_1"
