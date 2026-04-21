"""Definition of the ``fix_at_line`` OpenHands tool.

``fix_at_line`` replaces an inclusive, 1-indexed range of lines in a text file.
Unlike ``file_editor str_replace`` — which anchors by content and fails when the
same block appears more than once in the file — ``fix_at_line`` anchors by
line number, so duplicated content is never ambiguous.

Contract (mirrors the idioms of the built-in ``file_editor`` tool):

- The agent MUST call ``file_editor view`` first to learn the current content
  of the range it intends to edit.
- ``old_block`` is a safety check: it must match the current contents of
  ``start_line..end_line`` exactly (excluding the trailing newline of the
  block).  If it does not, the edit is rejected and the file is unchanged.
- ``new_block`` is written verbatim.  The agent is responsible for supplying
  correct indentation — the tool does not massage whitespace.
- ``new_block`` may contain fewer or more lines than the original range;
  the file is respliced accordingly (shrink / grow).
- An empty ``new_block`` deletes the range.

This module follows the same layout as ``openhands.tools.file_editor``:
schema + tool definition live here; the executor lives in ``impl.py`` and is
imported lazily inside ``FixAtLineTool.create`` to avoid circular imports.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import Field, model_validator

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


TOOL_DESCRIPTION = """Replace an inclusive, 1-indexed range of lines in a file.  Safe for files with duplicated content.

Prefer this tool over ``file_editor str_replace`` for SonarCloud fixes: the
SonarCloud issue always gives you an exact line range, and ``fix_at_line``
anchors by line number — so duplicated lines elsewhere in the file are never
a problem.

Parameters:
- ``path``: absolute path to the file.
- ``start_line``: 1-indexed inclusive first line of the range to replace.
- ``end_line``: 1-indexed inclusive last line of the range to replace.  Use
  ``end_line == start_line`` to edit exactly one line.
- ``old_block``: the current content of lines ``start_line..end_line``,
  verbatim (including indentation, excluding the final newline of the
  block).  Used as a safety check — if it does not match, the edit is
  rejected and nothing is changed.  You MUST call ``file_editor view`` on
  the file first so you have the current block content.
- ``new_block``: replacement content for the range.  Must include correct
  indentation; no whitespace adjustment is performed.  May contain fewer
  or more lines than the original.  Pass an empty string to delete the
  range.

Workflow:
1. Call ``file_editor view`` with a ``view_range`` covering the target lines.
2. Call ``fix_at_line`` with ``old_block`` = exactly what ``view`` showed
   (minus the ``N<TAB>`` line-number prefix) and ``new_block`` = the
   replacement.
3. If the observation reports a mismatch, call ``view`` again and retry —
   the file has drifted since the last read.
"""


class FixAtLineAction(Action):
    """Schema for a ``fix_at_line`` invocation."""

    path: str = Field(description="Absolute path to the file to edit.")
    start_line: int = Field(
        ge=1,
        description="1-indexed inclusive first line of the range to replace.",
    )
    end_line: int = Field(
        ge=1,
        description=(
            "1-indexed inclusive last line of the range to replace.  Set equal "
            "to ``start_line`` to replace exactly one line."
        ),
    )
    old_block: str = Field(
        description=(
            "Current content of the target line range, verbatim.  Used as a "
            "safety check; the edit is rejected if this does not match."
        ),
    )
    new_block: str = Field(
        description=(
            "Replacement content for the target range.  Supply correct "
            "indentation; no whitespace massaging is performed.  May contain "
            "more or fewer lines than the original.  Empty string deletes "
            "the range."
        ),
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "FixAtLineAction":
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= "
                f"start_line ({self.start_line})"
            )
        return self


class FixAtLineObservation(Observation):
    """Result of a ``fix_at_line`` execution."""

    path: str | None = Field(
        default=None, description="The file path that was edited."
    )
    start_line: int | None = Field(
        default=None, description="First 1-indexed line in the edited range."
    )
    end_line: int | None = Field(
        default=None, description="Last 1-indexed line in the edited range."
    )
    old_block: str | None = Field(
        default=None, description="Block content before the edit."
    )
    new_block: str | None = Field(
        default=None, description="Block content after the edit."
    )


class FixAtLineTool(ToolDefinition[FixAtLineAction, FixAtLineObservation]):
    """Line-range-anchored replacement tool."""

    def declared_resources(self, action: Action) -> DeclaredResources:
        """Declare that this action locks the target file path.

        Mirrors ``FileEditorTool.declared_resources`` so concurrent agents
        editing different files can run in parallel while edits to the same
        file are serialized.
        """
        assert isinstance(action, FixAtLineAction)
        normalized_path = Path(action.path).resolve()
        return DeclaredResources(keys=(f"file:{normalized_path}",), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState | None" = None,
        **params,
    ) -> Sequence[Self]:
        if params:
            raise ValueError("FixAtLineTool does not accept parameters")

        # Lazy import mirrors openhands.tools.file_editor.definition:
        # the executor module imports types from this module, so we defer
        # its import until a tool instance is actually created.
        from devdox_ai_sonar.openhands_tools.fix_at_line.impl import (
            FixAtLineExecutor,
        )

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=FixAtLineAction,
                observation_type=FixAtLineObservation,
                executor=FixAtLineExecutor(),
                annotations=ToolAnnotations(
                    title="fix_at_line",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]


# Register at import time so ``Tool(name=FixAtLineTool.name)`` resolves.
register_tool(FixAtLineTool.name, FixAtLineTool)
