"""Unit tests for the ``fix_at_line`` tool definition.

Covers the parts that live in ``definition.py``:

* ``FixAtLineAction`` — pydantic schema validation (field constraints and
  the ``end_line >= start_line`` model validator).
* ``FixAtLineTool`` — class-level name derivation, ``create()`` factory,
  and per-file resource declaration used by OpenHands to serialize edits.
* Registration side effect — importing the package must add ``fix_at_line``
  to the OpenHands tool registry.

Strategy: construct the real pydantic models and real ``FixAtLineTool``
instances directly.  No mocks.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from openhands.sdk.tool import DeclaredResources, list_registered_tools

from devdox_ai_sonar.openhands_tools.fix_at_line.definition import (
    FixAtLineAction,
    FixAtLineTool,
)


# ============================================================================
# ACTION SCHEMA VALIDATION
# ============================================================================


class TestFixAtLineAction:
    """Pydantic-level constraints on the action schema."""

    def test_valid_inputs_construct_successfully(self):
        action = FixAtLineAction(
            path="/abs/path.py",
            start_line=1,
            end_line=3,
            old_block="foo",
            new_block="bar",
        )

        assert action.start_line == 1
        assert action.end_line == 3
        assert action.old_block == "foo"
        assert action.new_block == "bar"

    def test_single_line_range_is_valid(self):
        action = FixAtLineAction(
            path="/abs/p.py",
            start_line=5, end_line=5,
            old_block="a", new_block="b",
        )

        assert action.start_line == action.end_line == 5

    @pytest.mark.parametrize("bad_start", [0, -1, -100])
    def test_start_line_below_one_is_rejected(self, bad_start):
        with pytest.raises(ValidationError):
            FixAtLineAction(
                path="/abs/p.py",
                start_line=bad_start, end_line=1,
                old_block="a", new_block="b",
            )

    @pytest.mark.parametrize("bad_end", [0, -1])
    def test_end_line_below_one_is_rejected(self, bad_end):
        with pytest.raises(ValidationError):
            FixAtLineAction(
                path="/abs/p.py",
                start_line=1, end_line=bad_end,
                old_block="a", new_block="b",
            )

    def test_end_line_smaller_than_start_line_is_rejected(self):
        with pytest.raises(ValidationError):
            FixAtLineAction(
                path="/abs/p.py",
                start_line=10, end_line=5,
                old_block="a", new_block="b",
            )


# ============================================================================
# TOOL DEFINITION / FACTORY
# ============================================================================


class TestFixAtLineTool:
    """Class-level behaviour: naming, factory, resource declaration."""

    def test_name_derives_from_class(self):
        """OpenHands auto-derives the tool name via _camel_to_snake."""
        assert FixAtLineTool.name == "fix_at_line"

    def test_create_returns_single_tool_instance(self):
        tools = FixAtLineTool.create(conv_state=None)

        assert len(tools) == 1
        assert isinstance(tools[0], FixAtLineTool)

    def test_create_wires_an_executor(self):
        tool = FixAtLineTool.create(conv_state=None)[0]

        assert tool.executor is not None

    def test_create_rejects_unknown_params(self):
        with pytest.raises(ValueError, match="does not accept parameters"):
            FixAtLineTool.create(conv_state=None, foo="bar")

    def test_declared_resources_locks_target_file(self):
        tool = FixAtLineTool.create(conv_state=None)[0]
        action = FixAtLineAction(
            path="/tmp/some/file.py",
            start_line=1, end_line=1,
            old_block="x", new_block="y",
        )

        resources = tool.declared_resources(action)

        assert isinstance(resources, DeclaredResources)
        assert resources.declared is True
        assert resources.keys == (f"file:{Path('/tmp/some/file.py').resolve()}",)

    def test_declared_resources_normalizes_path(self):
        """Differently-written paths to the same file should produce the same lock key."""
        tool = FixAtLineTool.create(conv_state=None)[0]
        action_a = FixAtLineAction(
            path="/tmp/./some/file.py",
            start_line=1, end_line=1,
            old_block="x", new_block="y",
        )
        action_b = FixAtLineAction(
            path="/tmp/some/file.py",
            start_line=1, end_line=1,
            old_block="x", new_block="y",
        )

        assert tool.declared_resources(action_a).keys == tool.declared_resources(action_b).keys


# ============================================================================
# REGISTRY SIDE EFFECT
# ============================================================================


class TestRegistration:
    """Importing the tool module must register it with OpenHands."""

    def test_tool_is_registered_under_its_name(self):
        assert "fix_at_line" in list_registered_tools()
