"""Prompts for picking a SonarCloud branch or pull request to analyse.

These helpers only touch the ``sonar.*`` section of ``config.toml``;
nothing LLM-related lives here.
"""

from __future__ import annotations

from typing import Any, Tuple

import rich_click as click
from rich.console import Console
from rich.prompt import Confirm, Prompt

from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.utils.exceptions import ValidationError
from devdox_ai_sonar.utils.validator import InputValidator


__all__ = [
    "CONFIG_DEFAULT_BRANCH",
    "CONFIG_DEFAULT_PULL",
    "CONFIG_CLONE_TYPE",
    "branch_or_pr",
    "branch_or_pr_prompt",
]


console = Console()

CONFIG_DEFAULT_BRANCH = "sonar.default_branch"
CONFIG_DEFAULT_PULL = "sonar.default_pull"
CONFIG_CLONE_TYPE = "sonar.sonar_options.clone_type"


async def branch_or_pr(manager: ConfigManager) -> Tuple[str, int]:
    """Return ``(branch_name, pull_request_number)`` from saved defaults.

    Used when a previous configuration run saved a preferred
    branch/PR; subsequent runs use it non-interactively.
    """
    clone_type = await manager.get_value(CONFIG_CLONE_TYPE)
    default_pull = await manager.get_value(CONFIG_DEFAULT_PULL)
    default_branch = await manager.get_value(CONFIG_DEFAULT_BRANCH)
    if clone_type == "pr":
        default_branch = ""
    else:
        default_pull = 0
    return default_branch, default_pull


async def branch_or_pr_prompt(manager: ConfigManager) -> Tuple[str, int]:
    """Interactive prompt: pick a branch name or a pull-request number."""
    console.print("\n[bold]Choose what to analyze:[/bold]")

    if Confirm.ask(
        "Do you want to analyze a [cyan]pull request[/cyan]?", default=False
    ):
        return await _handle_pull_request_selection(manager)
    return await _handle_branch_selection(manager)


async def _handle_branch_selection(manager: ConfigManager) -> Tuple[str, int]:
    default_branch = (
        await manager.get_value(CONFIG_DEFAULT_BRANCH) or "main"
    )
    branch_input = Prompt.ask("Branch name", default=default_branch)

    try:
        branch = InputValidator.validate_branch_name(branch_input)
    except ValidationError as exc:
        console.print(f"\n[red]❌ {exc.message}[/red]")
        raise click.Abort()

    console.print(f"[green]✓[/green] Analyzing branch: [cyan]{branch}[/cyan]")

    await _save_as_default_if_changed(
        manager,
        config_key=CONFIG_DEFAULT_BRANCH,
        new_value=branch,
        current_default=default_branch,
        display_name=f"branch '{branch}'",
    )
    await _save_as_default_if_changed(
        manager,
        config_key=CONFIG_CLONE_TYPE,
        new_value="branch",
        current_default="pr",
        display_name="",
        confirm=False,
    )
    return branch, 0


async def _handle_pull_request_selection(
    manager: ConfigManager,
) -> Tuple[str, int]:
    default_pull = await manager.get_value(CONFIG_DEFAULT_PULL)
    pr_default = str(default_pull) if default_pull else "0"
    pr_input = Prompt.ask("Pull Request number", default=pr_default)

    try:
        pull_request = InputValidator.validate_pull_request_number(pr_input)
    except ValidationError as exc:
        console.print(f"\n[red]❌ {exc.message}[/red]")
        raise click.Abort()

    console.print(f"[green]✓[/green] Analyzing PR: [cyan]#{pull_request}[/cyan]")

    await _save_as_default_if_changed(
        manager,
        config_key=CONFIG_DEFAULT_PULL,
        new_value=pull_request,
        current_default=default_pull,
        display_name=f"PR #{pull_request}",
    )
    await _save_as_default_if_changed(
        manager,
        config_key=CONFIG_CLONE_TYPE,
        new_value="pr",
        current_default="branch",
        display_name="",
        confirm=False,
    )
    return "", pull_request


async def _save_as_default_if_changed(
    manager: ConfigManager,
    *,
    config_key: str,
    new_value: Any,
    current_default: Any,
    display_name: str,
    confirm: bool = True,
) -> None:
    """Persist ``new_value`` at ``config_key`` if it differs from the
    saved default, optionally asking the user to confirm first."""
    if str(new_value) == str(current_default):
        return
    if confirm and not Confirm.ask(
        "Save this as default for future runs?", default=False
    ):
        return

    await manager.set_value(config_key, new_value)
    manager.save_config()
    if confirm:
        console.print(f"[green]✓[/green] Saved {display_name} as default")
