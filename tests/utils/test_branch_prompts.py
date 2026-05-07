"""Tests for utils/branch_prompts.py.

The module is interactive (Rich Prompt/Confirm + a SonarCloud-oriented
ConfigManager). Every test stubs both so the flow runs deterministically
against an in-memory config.
"""

from pathlib import Path

import pytest
import rich_click as click

from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.utils import branch_prompts
from devdox_ai_sonar.utils.branch_prompts import (
    CONFIG_CLONE_TYPE,
    CONFIG_DEFAULT_BRANCH,
    CONFIG_DEFAULT_PULL,
    branch_or_pr,
    branch_or_pr_prompt,
)
from devdox_ai_sonar.utils.exceptions import ValidationError


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    m = ConfigManager(config_path=tmp_path / "config.toml")
    m.create_default_config()
    return m


def _patch_prompts(monkeypatch, *, text_answers=None, confirm_answers=None):
    """Stub Rich's Prompt.ask / Confirm.ask inside the branch_prompts module."""
    texts = iter(text_answers or [])
    confirms = iter(confirm_answers or [])

    class _FakePrompt:
        @staticmethod
        def ask(*args, **kwargs):
            return next(texts, kwargs.get("default", ""))

    class _FakeConfirm:
        @staticmethod
        def ask(*args, **kwargs):
            return next(confirms, kwargs.get("default", False))

    monkeypatch.setattr(branch_prompts, "Prompt", _FakePrompt)
    monkeypatch.setattr(branch_prompts, "Confirm", _FakeConfirm)


# ---------------------------------------------------------------------------
# branch_or_pr — non-interactive resolution from saved defaults
# ---------------------------------------------------------------------------


async def test_branch_or_pr_returns_branch_when_clone_type_is_branch(manager):
    await manager.set_value(CONFIG_CLONE_TYPE, "branch")
    await manager.set_value(CONFIG_DEFAULT_BRANCH, "main")
    await manager.set_value(CONFIG_DEFAULT_PULL, 42)  # ignored in branch mode

    branch, pull = await branch_or_pr(manager)

    assert branch == "main"
    assert pull == 0  # branch mode zeroes the PR number


async def test_branch_or_pr_returns_pr_when_clone_type_is_pr(manager):
    await manager.set_value(CONFIG_CLONE_TYPE, "pr")
    await manager.set_value(CONFIG_DEFAULT_BRANCH, "main")  # ignored in pr mode
    await manager.set_value(CONFIG_DEFAULT_PULL, 42)

    branch, pull = await branch_or_pr(manager)

    assert branch == ""   # pr mode blanks the branch name
    assert pull == 42


# ---------------------------------------------------------------------------
# branch_or_pr_prompt — interactive branch path
# ---------------------------------------------------------------------------


async def test_prompt_branch_path_persists_new_default(monkeypatch, manager):
    """User picks branch mode, supplies a new branch name, and agrees
    to save it as the default. Config reflects both the branch and
    the clone_type change."""
    _patch_prompts(
        monkeypatch,
        text_answers=["feature/my-branch"],
        confirm_answers=[
            False,   # "analyze a pull request?" -> no (take the branch path)
            True,    # "Save this as default for future runs?"
        ],
    )

    branch, pull = await branch_or_pr_prompt(manager)

    assert branch == "feature/my-branch"
    assert pull == 0
    assert await manager.get_value(CONFIG_DEFAULT_BRANCH) == "feature/my-branch"
    assert await manager.get_value(CONFIG_CLONE_TYPE) == "branch"


async def test_prompt_branch_path_declines_to_save_default(
    monkeypatch, manager, capsys
):
    """User declines the 'save this as default' confirm -- the
    config key is left unchanged, but the current-run return value
    still reflects the user's choice."""
    await manager.set_value(CONFIG_DEFAULT_BRANCH, "main")

    _patch_prompts(
        monkeypatch,
        text_answers=["feature/temp"],
        confirm_answers=[
            False,   # analyze a pull request? -> no
            False,   # save as default? -> no
        ],
    )

    branch, pull = await branch_or_pr_prompt(manager)

    assert branch == "feature/temp"
    assert pull == 0
    # Default stayed at "main" -- we did not promote the temp branch.
    assert await manager.get_value(CONFIG_DEFAULT_BRANCH) == "main"


async def test_prompt_branch_invalid_name_aborts(monkeypatch, manager):
    """InputValidator.validate_branch_name raising ValidationError
    must surface as click.Abort (clean CLI exit, not a stack trace)."""
    _patch_prompts(
        monkeypatch,
        text_answers=["not a valid branch name"],
        confirm_answers=[False],  # analyze a pull request? -> no
    )

    def fake_validate(value):
        raise ValidationError(
            "branch names cannot contain spaces", field="branch"
        )

    monkeypatch.setattr(
        branch_prompts.InputValidator,
        "validate_branch_name",
        fake_validate,
    )

    with pytest.raises(click.Abort):
        await branch_or_pr_prompt(manager)


# ---------------------------------------------------------------------------
# branch_or_pr_prompt — interactive pull-request path
# ---------------------------------------------------------------------------


async def test_prompt_pr_path_persists_new_default(monkeypatch, manager):
    """User picks PR mode, supplies a PR number, and saves it as
    default. Config stores both the number and the clone_type flip."""
    _patch_prompts(
        monkeypatch,
        text_answers=["42"],
        confirm_answers=[
            True,   # analyze a pull request? -> yes
            True,   # save as default? -> yes
        ],
    )

    branch, pull = await branch_or_pr_prompt(manager)

    assert branch == ""
    assert pull == 42
    assert await manager.get_value(CONFIG_DEFAULT_PULL) == 42
    assert await manager.get_value(CONFIG_CLONE_TYPE) == "pr"


async def test_prompt_pr_path_with_existing_default_prefills_prompt(
    monkeypatch, manager
):
    """When a PR number is already saved as default, the Prompt
    default argument carries it so the user can press Enter to reuse."""
    await manager.set_value(CONFIG_DEFAULT_PULL, 7)
    captured_defaults = []

    class _CapturingPrompt:
        @staticmethod
        def ask(*args, **kwargs):
            captured_defaults.append(kwargs.get("default"))
            return "7"

    class _Confirm:
        @staticmethod
        def ask(*args, **kwargs):
            return True if "pull request" in (args[0] if args else "") else False

    monkeypatch.setattr(branch_prompts, "Prompt", _CapturingPrompt)
    monkeypatch.setattr(branch_prompts, "Confirm", _Confirm)

    branch, pull = await branch_or_pr_prompt(manager)

    assert pull == 7
    assert "7" in captured_defaults


async def test_prompt_pr_invalid_number_aborts(monkeypatch, manager):
    _patch_prompts(
        monkeypatch,
        text_answers=["not-a-number"],
        confirm_answers=[True],  # analyze a pull request? -> yes
    )

    def fake_validate(value):
        raise ValidationError(
            "PR number must be a positive integer", field="pr"
        )

    monkeypatch.setattr(
        branch_prompts.InputValidator,
        "validate_pull_request_number",
        fake_validate,
    )

    with pytest.raises(click.Abort):
        await branch_or_pr_prompt(manager)


# ---------------------------------------------------------------------------
# _save_as_default_if_changed — internal helper
# ---------------------------------------------------------------------------


async def test_save_as_default_skips_when_value_matches_current(
    monkeypatch, manager
):
    """If the new value matches the saved default, no prompt runs and
    no write happens."""
    prompt_calls = {"n": 0}

    class _Confirm:
        @staticmethod
        def ask(*args, **kwargs):
            prompt_calls["n"] += 1
            return True

    monkeypatch.setattr(branch_prompts, "Confirm", _Confirm)

    await branch_prompts._save_as_default_if_changed(
        manager,
        config_key=CONFIG_DEFAULT_BRANCH,
        new_value="main",
        current_default="main",
        display_name="branch 'main'",
    )

    assert prompt_calls["n"] == 0   # never asked


async def test_save_as_default_writes_without_prompt_when_confirm_false(
    manager
):
    """The confirm=False path updates config silently (used for the
    clone_type flip that mirrors the user's branch-vs-pr choice)."""
    await branch_prompts._save_as_default_if_changed(
        manager,
        config_key=CONFIG_CLONE_TYPE,
        new_value="branch",
        current_default="pr",
        display_name="",
        confirm=False,
    )

    assert await manager.get_value(CONFIG_CLONE_TYPE) == "branch"
