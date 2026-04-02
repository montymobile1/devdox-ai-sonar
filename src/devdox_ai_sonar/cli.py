"""Command-line interface for SonarCloud Analyzer"""

from datetime import timezone, datetime
import sys
from pathlib import Path
from typing import Optional, List, Any, Sequence, Dict, Tuple, Union, Iterator
import asyncio
import functools
import traceback

from rich.console import Console
from devdox_ai_sonar.utils.ui_prompts import select_from_list
from rich.table import Table
from rich.prompt import Confirm
from rich.panel import Panel
from rich.progress import Progress, TaskID
import rich_click as click
from questionary.prompts.common import Choice
from contextlib import contextmanager


from devdox_ai_sonar import __version__
from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer

from devdox_ai_sonar.services.rule_analyzer import RuleAnalyzer, _load_rules_cache
from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.agent_supervisor import build_supervisor
from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.models.llm import ProviderType
from devdox_ai_sonar.utils.file_indentation import (
    download_latest_version,
    TmpCloneManager,
    sweep_orphaned_tmp_dirs,
)
from devdox_ai_sonar.utils.validator import InputValidator, IssueType
from devdox_ai_sonar.utils.sonar_config import SonarCloudConfigUI
from devdox_ai_sonar.services.configuration import ConfigService, AuthConfig, LLMConfig

from devdox_ai_sonar.models.sonar import (
    AnalysisResult,
    FixSuggestion,
    FixResult,
)
from devdox_ai_sonar.utils.provider_config import (
    ProviderConfigUI,
    ProviderValidator,
    ProviderConfigManager,
)

from devdox_ai_sonar.utils.exceptions import SwitchCommandException
from devdox_ai_sonar.utils.supported_programming_languages import (
    PYTHON,
    IPYTHON,
    is_file_processable,
)
from devdox_ai_sonar.utils.ui import smart_prompt, smart_confirm
from devdox_ai_sonar.utils import constant
from devdox_ai_sonar.config import settings

EXCLUDE_RULE_CONFIG_FIELD = "configuration.exclude_rules"

console = Console()


def _should_skip_file(file_path: Optional[str]) -> bool:
    """Return True if *file_path* should be skipped during processing."""
    return bool(
        file_path
        and not is_file_processable(
            file_path,
            allowed_suffixes=PYTHON.file_extensions | IPYTHON.file_extensions,
            excluded_prefixes={"test_"},
        )
    )


def async_command(f: Any) -> Any:
    """Decorator to run async functions with Click."""

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@contextmanager
def show_progress(
    message: str, total: Optional[int] = None
) -> Iterator[Tuple[Progress, TaskID]]:
    """Context manager for progress display."""
    with Progress() as progress:
        task = progress.add_task(message, total=total)
        try:
            yield progress, task
        finally:
            if not progress.finished:
                progress.remove_task(task)


def _safe_convert_pr(pull_request: str | int | None) -> int:
    """Safely convert PR string to integer."""
    if not pull_request:
        return 0

    if isinstance(pull_request, int):
        return pull_request

    try:
        pr_int = int(pull_request.strip())
        if pr_int < 0:
            console.print("[yellow]⚠ Negative PR numbers not allowed, using 0[/yellow]")
            return 0
        return pr_int
    except (ValueError, AttributeError):
        console.print(f"[yellow]⚠ Invalid PR number '{pull_request}', using 0[/yellow]")
        return 0


def _fallback_command_selector() -> Optional[str]:
    """Fallback text-based command selector when questionary is not available."""
    console.print(
        "\n[bold cyan]═══════════════════════════════════════════════[/bold cyan]"
    )
    console.print(
        "[bold cyan]   DevDox AI Sonar - Command Selection          [/bold cyan]"
    )
    console.print(
        "[bold cyan]═══════════════════════════════════════════════[/bold cyan]\n"
    )

    commands = {
        "1": ("fix_issues", "Fix Issues - Generate and apply LLM-powered fixes"),
        "2": (
            "fix_security_issues",
            "Fix Security Issues - Specialized security fixes",
        ),
        "3": ("analyze", "Analyze Project - Display SonarCloud analysis"),
        "4": ("inspect", "Inspect Project - Analyze local directory structure"),
        "5": ("exit", "Exit"),
    }

    for key, (_, desc) in commands.items():
        console.print(f"[cyan]{key}[/cyan]. {desc}")

    console.print()
    choice = console.input("[bold yellow]Select command (1-5): [/bold yellow]").strip()

    if choice in commands:
        return commands[choice][0]
    return None


# ============================================================================
# INTERACTIVE COMMAND SELECTOR
# ============================================================================


async def show_command_selector_async() -> Optional[str]:
    """
    Show an interactive command selector similar to Claude Code.
    Returns the selected command or None if cancelled.
    """
    try:
        import questionary
        from questionary import Style
    except ImportError:
        console.print(
            "[yellow]⚠ questionary not installed. Using fallback prompt.[/yellow]"
        )
        console.print("[dim]Install with: pip install questionary[/dim]\n")
        return _fallback_command_selector()

    # Define custom style similar to Claude Code
    custom_style = Style(
        [
            ("qmark", constant.BOLD_PURPLE),
            ("question", "bold"),
            ("answer", "fg:#f44336 bold"),
            ("pointer", constant.BOLD_PURPLE),
            ("highlighted", constant.BOLD_PURPLE),
            ("selected", "fg:#cc5454"),
            ("separator", "fg:#cc5454"),
            ("instruction", ""),
            ("text", ""),
        ]
    )

    commands = [
        {
            "name": "➕ Add Provider - Add provider or sonar configuration",
            "value": "add_provider",
            "description": "Add provider configuration",
        },
        {
            "name": "✏️  Update Provider - Update provider or sonar configuration",
            "value": "update_provider",
            "description": "Update SonarCloud provider configuration",
        },
        {
            "name": "🔧 Fix Issues - Generate and apply LLM-powered fixes",
            "value": "fix_issues",
            "description": "Analyze and fix code quality issues",
        },
        {
            "name": "🔒 Fix Security Issues - Specialized security vulnerability fixes",
            "value": "fix_security_issues",
            "description": "Focus on security vulnerabilities",
        },
        {
            "name": "📊 Analyze Project - Display SonarCloud analysis",
            "value": "analyze",
            "description": "View project metrics and issues",
        },
        {
            "name": "🔍 Inspect Project - Analyze local directory structure",
            "value": "inspect",
            "description": "Inspect local project structure",
        },
        {
            "name": "⚙️  Change Parameters Configuration",
            "value": "change_parameters",
            "description": "Change parameters configuration",
        },
        {"name": "❌ Exit", "value": "exit", "description": "Exit the application"},
    ]

    console.clear()

    console.print()
    console.print(
        "[bold cyan]═══════════════════════════════════════════════[/bold cyan]"
    )
    console.print(
        "[bold cyan]   DevDox AI Sonar - Interactive Mode          [/bold cyan]"
    )
    console.print(
        "[bold cyan]═══════════════════════════════════════════════[/bold cyan]"
    )
    console.print()

    try:
        choice = await questionary.select(
            "What would you like to do?",
            choices=[cmd["name"] for cmd in commands],
            style=custom_style,
            use_shortcuts=True,
            use_arrow_keys=True,
            use_jk_keys=True,
        ).ask_async()

        if choice is None:
            return None

        for cmd in commands:
            if cmd["name"] == choice:
                return cmd["value"]

        return None

    except KeyboardInterrupt:
        return None


async def _select_existing_ui(message: str, existing_providers: list) -> str:
    result = await select_from_list(existing_providers, message)
    if not result:
        console.print("[yellow]⚠ Selection cancelled[/yellow]")
        return ""
    return result


def _initialize_managers() -> Tuple[
    ConfigManager,
    ProviderConfigUI,
    ProviderValidator,
    ProviderConfigManager,
    SonarCloudConfigUI,
    ConfigService,
]:
    manager = ConfigManager(config_path=settings.config_file_path)
    ui = ProviderConfigUI()
    validator = ProviderValidator()
    provider_manager = ProviderConfigManager(manager, ui, validator)
    sonar_ui = SonarCloudConfigUI()
    config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
    return manager, ui, validator, provider_manager, sonar_ui, config_service


async def _configure_sonarcloud(
    sonar_ui: SonarCloudConfigUI, config_service: ConfigService
) -> bool:
    auth_config = await config_service.load_auth_config()
    value_exist = config_service.check_all_value_empty(auth_config)
    if not value_exist:
        return True

    sonar_ui.display_welcome()
    sonar_ui.display_step_header(1, 2, "SonarCloud Configuration")

    sonar_config = sonar_ui.configure_sonarcloud(auth_config)
    if not sonar_config:
        console.print("[red]❌ SonarCloud configuration cancelled[/red]")
        return False

    save_success = await config_service.save_config(
        token=sonar_config.token,
        organization=sonar_config.organization,
        project=sonar_config.project,
        project_path=str(sonar_config.project_path),
        git_url=sonar_config.git_url,
    )

    if not save_success:
        click.secho("❌ SonarCloud configuration save failed", fg="red", bold=True)
        return False

    click.secho("\n✓ SonarCloud configuration saved", fg="green", bold=True)
    return True


async def _configure_providers_loop(
    provider_manager: ProviderConfigManager,
    manager: ConfigManager,
    ui: ProviderConfigUI,
    available_providers: list,
) -> None:
    while available_providers:
        console.print(
            f"[cyan]Available providers: {', '.join(available_providers)}[/cyan]\n"
        )
        provider_name = await ui.select_provider_from_list(
            available_providers, "Select a provider to configure"
        )
        if not provider_name:
            break
        success = await _handle_provider_configuration(
            provider_manager, manager, provider_name, available_providers
        )
        if not success:
            console.print(
                f"[yellow]⚠ Configuration of {provider_name} failed or was cancelled[/yellow]\n"
            )
        if _should_stop_configuring(available_providers):
            break


async def _handle_provider_configuration(
    provider_manager: ProviderConfigManager,
    manager: ConfigManager,
    provider_name: str,
    available_providers: list,
) -> bool:
    result = await provider_manager.configure_new_provider(provider_name)
    if not result:
        return False
    await manager.add_provider(
        result["config"], set_as_default=result["set_as_default"]
    )
    available_providers.remove(provider_name)
    console.print(
        f"\n[green]✓ {provider_name.upper()} configured successfully[/green]\n"
    )
    return True


async def change_max_fix(
    manager: ConfigManager, message: str, max_fixes: int, default_max_fixes: int
) -> None:
    max_fixes_str = await smart_prompt(message, default=str(max_fixes))
    try:
        if isinstance(max_fixes_str, list):
            value_str = max_fixes_str[0] if max_fixes_str else str(default_max_fixes)
        else:
            value_str = max_fixes_str
        new_max_fixes = int(value_str)
        if new_max_fixes < 1 or new_max_fixes > default_max_fixes:
            new_max_fixes = settings.DEFAULT_MAX_FIXES
            console.print(
                f"[yellow]Value out of range (1-{default_max_fixes}), "
                f"using default: {settings.DEFAULT_MAX_FIXES}[/yellow]"
            )
    except (ValueError, IndexError):
        new_max_fixes = default_max_fixes
        console.print(
            f"[yellow]Invalid value, using default: {default_max_fixes}[/yellow]"
        )
    await manager.set_value("configuration.max_fixes", new_max_fixes)


def _should_stop_configuring(available_providers: list) -> bool:
    if not available_providers:
        console.print("[green]✅ All providers have been configured[/green]")
        return True
    return not Confirm.ask("Add another provider?", default=False)


def _check_reconfiguration_consent(providers: list) -> bool:
    if not providers or len(providers) == 0:
        return True
    click.secho("\n⚠️  Configuration already exists", fg="yellow", bold=True)
    return False


def _display_operation_header(operation: str) -> None:
    console.print("\n" + "=" * 60)
    console.print(f"[magenta bold]{operation}[/magenta bold]")
    console.print("=" * 60 + "\n")


def _display_completion_message() -> None:
    console.print("=" * 60)
    console.print("[green bold]✨ CONFIGURATION COMPLETE![/green bold]")
    console.print("=" * 60)
    console.print(f"\nConfiguration saved to: {settings.config_file_path}\n")


def _handle_cli_error(error: Exception) -> None:
    if isinstance(error, click.Abort):
        console.print("\n[red]Configuration cancelled[/red]")
        raise
    console.print(f"\n[red]❌ Error during configuration: {str(error)}[/red]")
    raise click.ClickException(str(error))


# ============================================================================
# MAIN CLI COMMANDS
# ============================================================================


async def init_config(
    types: Optional[str] = None,
    severity: Optional[str] = None,
    max_fixes: int = 0,
    apply: bool = False,
    dry_run: bool = False,
) -> None:
    """CLI for config management"""
    try:
        manager, ui, _, provider_manager, sonar_ui, config_service = (
            _initialize_managers()
        )
        manager.create_default_config()
        await manager.load_config()

        providers = await manager.get_value("llm.providers")
        if not _check_reconfiguration_consent(providers):
            return

        if not await _configure_sonarcloud(sonar_ui, config_service):
            raise click.Abort()

        if providers and len(providers) > 0:
            return

        available_providers = await provider_manager.get_available_providers()
        if not available_providers:
            console.print(
                "\n[red]❌ No providers configured. Configuration incomplete.[/red]"
            )
            raise click.Abort()

        await _configure_providers_loop(
            provider_manager, manager, ui, available_providers
        )
        manager.save_config(create_backup=False)
        apply_value = 1 if apply else 0
        dry_run_value = 1 if dry_run else 0
        await change_parameters(
            types,
            severity,
            max_fixes=max_fixes,
            apply=apply_value,
            dry_run=dry_run_value,
        )

    except Exception as e:
        _handle_cli_error(e)


async def add_provider() -> None:
    """CLI command for managing provider configuration"""
    try:
        manager, ui, _, provider_manager, _, _ = _initialize_managers()
        await manager.load_config()
        available_providers = await provider_manager.get_available_providers()
        if not available_providers:
            console.print(
                "[yellow]⚠ All supported providers are already configured[/yellow]"
            )
            raise click.Abort()

        _display_operation_header("🚀 ADD NEW PROVIDER")
        await _configure_providers_loop(
            provider_manager, manager, ui, available_providers
        )
        manager.save_config(create_backup=False)
        _display_completion_message()

    except Exception as e:
        _handle_cli_error(e)


async def update_provider() -> None:
    """CLI command for managing provider configuration"""
    try:
        manager, _, _, provider_manager, _, _ = _initialize_managers()
        await manager.load_config()

        existing_providers = await provider_manager.get_existing_providers()
        if not existing_providers:
            console.print(
                "\n[red]❌ No providers configured. Please add at least one provider first.[/red]"
            )
            raise click.Abort()
        _display_operation_header("🔧 UPDATE EXISTING PROVIDER")
        chosen_provider = await _select_existing_ui(
            "Select the provider to update", existing_providers
        )
        if not chosen_provider:
            raise click.Abort()

        if await provider_manager.update_existing_provider(chosen_provider):
            manager.save_config(create_backup=False)
            console.print(
                f"\n[green]✓ {chosen_provider.upper()} updated successfully[/green]\n"
            )
        else:
            console.print("\n[yellow]⚠ Update cancelled or failed[/yellow]\n")
        _display_completion_message()

    except Exception as e:
        _handle_cli_error(e)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


@click.command()
@click.version_option(__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--command",
    "-c",
    type=click.Choice(
        ["fix_issues", "fix_security_issues", "analyze", "inspect", "fix_multiple"]
    ),
    help="Run specific command directly without interactive mode",
)
@click.option("--sonar-token", type=str, help="Sonar Cloud API token")
@click.option("--sonar-org", type=str, help="Sonar Cloud organization")
@click.option("--sonar-project", type=str, help="Sonar Cloud project")
@click.option("--project-path", type=str, help="Project path to analyze")
@click.option(
    "--llm-provider",
    type=click.Choice(ProviderType.choices(), case_sensitive=True),
    help="LLM provider to use",
)
@click.option("--llm-api-key", type=str, help="LLM API key")
@click.option("--llm-default-model", type=str, help="LLM model to use")
@click.option("--branch", type=str, help="Branch to fix issues on", default="main")
@click.option(
    "--pull-request", type=int, help="Pull request number to fix issues on", default=0
)
@click.option("--types", type=str, help="Comma-separated issue types (for fix_issues)")
@click.option(
    "--severity", type=str, help="Comma-separated severities (for fix_issues)"
)
@click.option(
    "--excluded-rules", type=str, help="Comma-separated excluded rules (for fix_issues)"
)
@click.option(
    "--max-fixes",
    type=click.IntRange(0, settings.MAX_FIXES_LIMIT),
    help=f"Maximum number of fixes (0-{settings.MAX_FIXES_LIMIT})",
)
@click.option(
    "--apply",
    type=click.IntRange(0, 1),
    default=None,
    help="Apply fixes (1 = apply, 0 = preview only)",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be changed without applying fixes"
)
@click.pass_context
@async_command
async def main(
    ctx: click.Context,
    verbose: bool,
    command: Optional[str],
    sonar_token: Optional[str],
    sonar_org: Optional[str],
    sonar_project: Optional[str],
    project_path: Optional[str],
    llm_provider: Optional[str],
    llm_api_key: Optional[str],
    llm_default_model: Optional[str],
    branch: Optional[str],
    pull_request: Optional[int],
    types: Optional[str],
    severity: Optional[str],
    excluded_rules: Optional[str],
    max_fixes: Optional[int],
    apply: Optional[int],
    dry_run: bool = False,
) -> None:
    """
    DevDox AI Sonar - SonarCloud Analyzer with LLM-powered fixes.

    Interactive mode by default. Type '/' during any prompt to switch commands.

    Examples:
        devdox_sonar                           # Interactive mode
        devdox_sonar -c fix_issues            # Run fix_issues directly

    During interactive mode:
        - Type '/' at any prompt to switch to a different command
        - Use arrow keys to navigate menus
        - Press Ctrl+C to cancel current operation
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["options"] = {
        "types": types,
        "severity": severity,
        "max_fixes": max_fixes or 0,
        "apply": apply,
        "dry_run": dry_run,
        "sonar_token": sonar_token,
        "sonar_org": sonar_org,
        "sonar_project": sonar_project,
        "project_path": project_path,
        "llm_provider": llm_provider,
        "llm_api_key": llm_api_key,
        "llm_default_model": llm_default_model,
        "branch": branch,
        "pull_request": pull_request,
        "excluded_rules": excluded_rules,
    }

    sweep_orphaned_tmp_dirs(on_status=lambda msg: console.print(f"[dim]{msg}[/dim]"))

    if command:
        await _execute_command_async(ctx, command)
        return

    await init_config()
    await _run_interactive_mode_async(ctx)


async def _run_interactive_mode_async(ctx: click.Context) -> None:
    while True:
        if await _execute_interactive_iteration_async(ctx):
            return


async def _execute_interactive_iteration_async(ctx: click.Context) -> bool:
    try:
        return await _process_interactive_command_async(ctx)
    except SwitchCommandException:
        _handle_command_switch()
        return False
    except KeyboardInterrupt:
        return await _handle_keyboard_interrupt()
    except Exception as e:
        return await _handle_interactive_error(e)


async def _process_interactive_command_async(ctx: click.Context) -> bool:
    command = await show_command_selector_async()

    if _should_exit_interactive_mode(command):
        _exit_application()
        return True

    await _execute_interactive_command_async(ctx, command)

    if not await _should_continue_to_menu():
        _exit_application()
        return True
    return False


def _should_exit_interactive_mode(command: Optional[str]) -> bool:
    return command is None or command == "exit"


def _handle_command_switch() -> None:
    console.print("\n[yellow]↩ Returning to command menu...[/yellow]\n")


async def _handle_keyboard_interrupt() -> bool:
    console.print("\n\n[yellow]⚠ Interrupted by user[/yellow]")
    try:
        if await smart_confirm("Exit application?", default=False, allow_switch=False):
            sys.exit(0)
        return False
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Force exit...[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error during exit: {e}[/red]")
        sys.exit(1)


async def _handle_interactive_error(error: Exception) -> bool:
    console.print(f"\n[red]❌ Error: {str(error)}[/red]")
    try:
        if not await smart_confirm(
            constant.RETURN_TO_MAIN_MENU, default=True, allow_switch=False
        ):
            sys.exit(1)
        return False
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Exiting after error...[/yellow]")
        sys.exit(1)
    except Exception as confirm_error:
        console.print(f"\n[red]Fatal: Cannot prompt user ({confirm_error})[/red]")
        sys.exit(2)


async def change_field(
    manager: ConfigManager,
    field: str,
    message: str,
    default_value: Optional[Union[str, List[str]]] = None,
    choices: Optional[List[Union[str, Choice]]] = None,
    multiple: bool = True,
    allow_empty: bool = False,
) -> Optional[Union[str, List[str]]]:
    types_input = await smart_prompt(
        message, default=default_value, choices=choices, multiple=multiple
    )
    types = types_input if types_input else None
    if types:
        await manager.set_value(field, types)
    elif allow_empty:
        while not types:
            console.print(constant.EXCLUDE_RULES_EMPTY_ERROR)
            retry_input = await smart_prompt(
                message, default=default_value, choices=choices, multiple=multiple
            )
            types = retry_input if retry_input else None
        await manager.set_value(field, types)
    return types


async def change_parameters(
    types: Optional[str] = None, severity: Optional[str] = None, **kwargs: Any
) -> None:
    """CLI for config management"""
    try:
        manager, _, _, provider_manager, _, _ = _initialize_managers()
        branch, pull_request = await provider_manager.branch_or_pr_prompt()

        if not branch and not pull_request:
            console.print(constant.NO_BRANCH_OR_PR_SPECIFIED)
            raise click.Abort()

        max_fixes = await manager.get_value("configuration.max_fixes") or 0

        await change_max_fix(
            manager,
            f"Maximum fixes to generate (0-{settings.MAX_FIXES_LIMIT})",
            max_fixes,
            settings.MAX_FIXES_LIMIT,
        )
        if not types:
            _ = await change_field(
                manager=manager,
                field="configuration.types",
                message="Issue types (comma-separated, or press Enter to skip)",
                default_value=await manager.get_value("configuration.types"),
                choices=list(InputValidator.VALID_ISSUE_TYPES),
            )

        if not severity:
            _ = await change_field(
                manager=manager,
                field="configuration.severities",
                message="Issue severities (comma-separated, or press Enter to skip)",
                default_value=await manager.get_value("configuration.severities"),
                choices=list(InputValidator.VALID_SEVERITIES),
            )

        choices = [{"name": "yes", "value": 1}, {"name": "no", "value": 0}]
        formatted_choices: List[Union[str, Choice]] = [
            Choice(title=str(choice["name"]), value=choice["value"])
            for choice in choices
        ]

        current_apply = await manager.get_value(constant.CONFIGURATION_APPLY)
        _ = await change_field(
            manager=manager,
            field=constant.CONFIGURATION_APPLY,
            message="Apply fixes of SonarQube (press Enter to skip)",
            default_value=(
                current_apply if current_apply is not None else kwargs.get("apply", 0)
            ),
            choices=formatted_choices,
            multiple=False,
        )

        configuration_backup = await manager.get_value(constant.CONFIGURATION_BACKUP)
        _ = await change_field(
            manager=manager,
            field=constant.CONFIGURATION_BACKUP,
            message="Create backup before apply fixes (press Enter to skip)",
            default_value=(
                configuration_backup
                if configuration_backup is not None
                else kwargs.get("create_backup", 0)
            ),
            choices=formatted_choices,
            multiple=False,
        )

        await provider_manager.skip_tests_prompt()

        console.print(
            "Rules to be excluded  ([yellow]comma-separated[/yellow], "
            "or [bold cyan]NONE[/bold cyan] for no exclusions)"
        )

        _ = await change_field(
            manager=manager,
            field=EXCLUDE_RULE_CONFIG_FIELD,
            message="Exclude rules",
            default_value=(
                await manager.get_value(EXCLUDE_RULE_CONFIG_FIELD)
                if await manager.get_value(EXCLUDE_RULE_CONFIG_FIELD) is not None
                else kwargs.get("exclude_rules", constant.EXCLUDE_NONE)
            ),
            allow_empty=True,
        )
        manager.save_config(create_backup=False)

    except Exception as e:
        _handle_cli_error(e)


def _exit_application() -> None:
    console.print("\n[cyan]👋 Thank you for using DevDox AI Sonar![/cyan]")
    sys.exit(0)


async def _should_continue_to_menu() -> bool:
    result = await smart_confirm(
        constant.RETURN_TO_MAIN_MENU, default=True, allow_switch=True
    )
    return result


async def _execute_interactive_command_async(
    ctx: click.Context, command: Optional[str]
) -> None:
    if command is None:
        return
    console.print(f"\n[bold green]▶ Running: {command}[/bold green]\n")
    await _execute_command_async(ctx, command)
    console.print("\n" + "─" * 50 + "\n")


async def _execute_command_async(ctx: click.Context, command: str) -> None:
    verbose = ctx.obj.get("verbose", False)
    options = ctx.obj.get("options", {})

    try:
        if command == "fix_issues":
            await _run_fix_issues(**options)
        elif command == "fix_security_issues":
            await _run_fix_security_issues(**options)
        elif command == "analyze":
            await _run_analyze()
        elif command == "inspect":
            await _run_inspect()
        elif command == "add_provider":
            await add_provider()
        elif command == "update_provider":
            await update_provider()
        elif command == "change_parameters":
            await change_parameters(**options)
        elif command == "fix_multiple":
            await fix_multiple(**options)
        else:
            click.echo(f"Unknown command: {command}", err=True)
            ctx.exit(1)

    except Exception as e:
        if verbose:
            click.echo(f"Error executing command '{command}': {e}", err=True)
            import traceback

            traceback.print_exc()
        else:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


# ============================================================================
# COMMAND IMPLEMENTATIONS WITH SWITCH SUPPORT
# ============================================================================


async def _run_fix_issues(**kwargs: Any) -> None:
    console.print("\n[bold cyan]🔧 Fix Issues - LLM-Powered Code Fixes[/bold cyan]\n")

    try:
        apply_value = kwargs.get("apply", None)
        dry_run_value = 1 if kwargs.get("dry_run", False) else 0

        auth_config, llm_config, parameters = await _load_and_validate_config(
            use_predefined=True
        )
        fix_params = display_configuration(parameters, dry_run_value, apply_value)

        if not await smart_confirm("Proceed with these settings?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        await _process_and_fix_issues(
            auth_config,
            llm_config,
            fix_params.get("branch", ""),
            pull_request=str(parameters.get("pull_request", 0)),
            fix_params=fix_params,
            issue_type=IssueType.REGULAR,
            download_latest=True,
            system_ask=True,
            check_tmp_path=True,
        )

    except SwitchCommandException:
        console.print(f"\n[yellow]{constant.SWITCH_COMMANDS}[/yellow]")
        traceback.print_exc()
        raise


async def _run_fix_security_issues(**kwargs: Any) -> None:
    console.print("\n[bold cyan]🔒 Fix Security Issues[/bold cyan]\n")

    try:
        auth_config, llm_config, parameters = await _load_and_validate_config(
            use_predefined=True
        )
        apply_value = kwargs.get("apply", None)
        dry_run_value = kwargs.get(
            "dry_run",
            (
                parameters.get("dry_run", 0)
                if parameters.get("dry_run") is not None
                else 0
            ),
        )
        fix_params = display_configuration(parameters, dry_run_value, apply_value)

        if not await smart_confirm("Proceed with security issue fixing?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        await _process_and_fix_issues(
            auth_config,
            llm_config,
            fix_params.get("branch", ""),
            str(fix_params.get("pull_request", 0)),
            fix_params,
            issue_type=IssueType.SECURITY,
            download_latest=True,
            system_ask=True,
            check_tmp_path=True,
        )

    except SwitchCommandException:
        console.print(f"\n[yellow]{constant.SWITCH_COMMANDS}[/yellow]")
        raise


async def _run_analyze() -> None:
    console.print("\n[bold cyan]📊 Analyze SonarCloud Project[/bold cyan]\n")

    try:
        auth_config, _, parameters = await _load_and_validate_config(
            use_predefined=True
        )
        console.print(f"  Project: [cyan]{auth_config.project}[/cyan]")
        console.print(f"  Organization: [cyan]{auth_config.organization}[/cyan]\n")

        limit_response = await smart_prompt(
            f"Max issues to fetch (max: {settings.MAX_FIXES_LIMIT})",
            default=str(parameters.get("max_fixes", settings.MAX_FIXES_LIMIT)),
        )

        try:
            if isinstance(limit_response, list):
                limit_str = (
                    limit_response[0]
                    if limit_response
                    else str(settings.MAX_FIXES_LIMIT)
                )
            else:
                limit_str = limit_response
            limit = int(limit_str)
            if limit <= 0 or limit > settings.MAX_FIXES_LIMIT:
                console.print(
                    f"[yellow]Limit must be between 1 and {settings.MAX_FIXES_LIMIT}, using default[/yellow]"
                )
                limit = settings.MAX_FIXES_LIMIT
        except (ValueError, IndexError):
            console.print("[red]Invalid limit, using default[/red]")
            limit = settings.MAX_FIXES_LIMIT

        analyzer = SonarCloudAnalyzer(auth_config.token, auth_config.organization)

        with show_progress(constant.FETCHING_ISSUES) as (progress, task):
            result = analyzer.get_project_issues(
                project_key=auth_config.project,
                branch=parameters.get("branch", ""),
                max_issues=limit,
                pull_request_number=_safe_convert_pr(parameters.get("pull_request", 0)),
                severities=_validate_severities(parameters.get("severity", "")),
                types=_validate_issue_types(parameters.get("types", "")),
            )

        if result:
            _display_analysis_results(result, limit)

    except SwitchCommandException:
        console.print(f"\n[yellow]{constant.SWITCH_COMMANDS}[/yellow]")
        raise


async def _run_inspect() -> None:
    console.print("\n[bold cyan]🔍 Inspect Local Project[/bold cyan]\n")

    try:
        auth_config, _, _ = await _load_and_validate_config(use_predefined=True)
        analyzer = SonarCloudAnalyzer(auth_config.token, auth_config.organization)
        analysis = analyzer.analyze_project_directory(str(auth_config.project_path))

        console.print(Panel.fit(f"[bold]Project: {auth_config.project_path}[/bold]"))

        table = Table(show_header=True, header_style=constant.BOLD_MAGENTA)
        table.add_column("Property")
        table.add_column("Value")

        table.add_row("Total Files", str(analysis["total_files"]))
        table.add_row("Python Files", str(analysis["python_files"]))
        table.add_row("JavaScript Files", str(analysis["javascript_files"]))
        table.add_row("Java Files", str(analysis["java_files"]))
        table.add_row(
            "Has SonarCloud Config", "✓" if analysis["has_sonar_config"] else "✗"
        )
        table.add_row("Has Git Repository", "✓" if analysis["has_git"] else "✗")

        console.print(table)

        if analysis["potential_source_dirs"]:
            console.print("\n[bold]Potential Source Directories:[/bold]")
            for src_dir in analysis["potential_source_dirs"]:
                console.print(f"  • {src_dir}")

    except SwitchCommandException:
        console.print(f"\n[yellow]{constant.SWITCH_COMMANDS}[/yellow]")
        raise


async def fix_multiple(**kwargs: Any) -> None:
    console.print("\n[bold cyan]🔧 Fix Issues - LLM-Powered Code Fixes[/bold cyan]\n")

    try:
        sonar_token = kwargs.get("sonar_token", None)
        sonar_org = kwargs.get("sonar_org", None)
        sonar_project = kwargs.get("sonar_project", None)
        project_path = kwargs.get("project_path", None)
        llm_provider = kwargs.get("llm_provider", None)
        llm_api_key = kwargs.get("llm_api_key", None)
        llm_default_model = kwargs.get("llm_default_model", None)
        branch = kwargs.get("branch", "main")
        pull_request = kwargs.get("pull_request", 0)

        auth_config = AuthConfig(
            sonar_token, sonar_org, sonar_project, project_path, ""
        )
        llm_config = LLMConfig(
            llm_provider, llm_default_model, llm_api_key, [llm_default_model]
        )

        max_fixes = kwargs.get("max_fixes", settings.MAX_FIXES_LIMIT)
        types_list = kwargs.get("types", "")
        severities = kwargs.get("severity", "")
        excluded_rules = (
            kwargs.get("excluded_rules", "").split(",")
            if kwargs.get("excluded_rules", "")
            else []
        )
        apply_value = kwargs.get("apply", None)
        dry_run_value = 1 if kwargs.get("dry_run", False) else 0

        fix_params = {
            "pull_request": pull_request,
            "branch": branch,
            "max_fixes": max_fixes,
            "types_list": _validate_issue_types(types_list),
            "severities_list": _validate_severities(severities),
            "apply": apply_value,
            "dry_run": dry_run_value,
            "exclude_rules": excluded_rules,
            "create_backup": 0,
        }

        await _process_and_fix_issues(
            auth_config,
            llm_config,
            branch,
            pull_request=str(pull_request),
            fix_params=fix_params,
            issue_type=IssueType.REGULAR,
            download_latest=False,
            system_ask=False,
            check_tmp_path=False,
        )
        await _process_and_fix_issues(
            auth_config,
            llm_config,
            branch,
            pull_request=str(pull_request),
            fix_params=fix_params,
            issue_type=IssueType.SECURITY,
            download_latest=False,
            system_ask=False,
            check_tmp_path=False,
        )

    except SwitchCommandException:
        console.print(f"\n[yellow]{constant.SWITCH_COMMANDS}[/yellow]")
        traceback.print_exc()
        raise


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _load_and_validate_config(
    use_predefined: bool = False,
) -> Tuple[AuthConfig, LLMConfig, Dict[str, Any]]:
    console.print("[dim]Loading configuration...[/dim]")
    manager, _, _, provider_manager, _, config_service = _initialize_managers()

    auth_config_dict = await config_service.load_auth_config()
    if not auth_config_dict:
        console.print(constant.AUTHENTICATION_NOT_FOUND)
        console.print(constant.DEVDOX_SONAR_CONFIG)
        raise click.Abort()
    auth_config = AuthConfig.from_dict(auth_config_dict)
    if not auth_config:
        console.print("[red]❌ Configuration error [/red]")
        raise click.Abort()

    is_valid, error_msg = auth_config.validate()
    if not is_valid:
        console.print(f"[red]❌ Configuration error: {error_msg}[/red]")
        raise click.Abort()

    llm_config = await config_service.load_llm_config(manager)
    if not llm_config:
        console.print("[red]❌ No LLM providers configured[/red]")
        raise click.Abort()

    if use_predefined:
        branch, pull_request = await provider_manager.branch_or_pr()
    else:
        branch, pull_request = await provider_manager.branch_or_pr_prompt()

    if not branch and not pull_request:
        console.print(constant.NO_BRANCH_OR_PR_SPECIFIED)
        raise click.Abort()

    params = await manager.get_value("configuration") or {}
    if params.get("exclude_rules"):
        if str(params["exclude_rules"]) == constant.EXCLUDE_NONE:
            params["exclude_rules"] = []
        else:
            params["exclude_rules"] = str(params["exclude_rules"]).split(",")

    params["branch"] = branch
    params["pull_request"] = pull_request
    params["skip_tests"] = await provider_manager.skip_tests_prompt()

    console.print("[green]✓[/green] Configuration loaded\n")
    return auth_config, llm_config, params


def _validate_issue_types(types_str: Optional[str]) -> Optional[List[str]]:
    if not types_str:
        return None
    return InputValidator.validate_issue_types(types_str)


def _validate_severities(severity_str: Optional[str]) -> Optional[List[str]]:
    if not severity_str:
        return None
    return InputValidator.validate_severities(severity_str)


def display_configuration(
    parameters: Dict[str, Any], dry_run: int, apply: int
) -> Dict[str, Any]:
    console.print("[bold]Configuration:[/bold]")
    pull_request = parameters.get("pull_request", 0)
    branch = parameters.get("branch", "")
    if isinstance(pull_request, int) and pull_request > 0:
        console.print(f"  Pull Request: [cyan]{pull_request}[/cyan]")
    elif branch:
        console.print(f"  Branch: [cyan]{branch}[/cyan]")
    apply_value = apply if apply is not None else parameters.get("apply", 0)
    console.print(f"  Max Fixes: [cyan]{parameters.get('max_fixes')}[/cyan]")
    console.print(f"  Apply: [cyan]{apply_value}[/cyan]")
    console.print(f"  Dry Run: [cyan]{dry_run}[/cyan]")
    skip_tests = parameters.get("skip_tests", True)
    run_tests_label = "No" if skip_tests else "Yes"
    console.print(f"  Run Tests: [cyan]{run_tests_label}[/cyan]")

    return {
        "pull_request": pull_request,
        "branch": branch,
        "max_fixes": parameters.get("max_fixes", 0),
        "types_list": _validate_issue_types(parameters.get("types", "")),
        "severities_list": _validate_severities(parameters.get("severities", "")),
        "apply": apply_value,
        "dry_run": dry_run,
        "exclude_rules": parameters.get("exclude_rules", None),
        "create_backup": 0,
        "skip_tests": skip_tests,
    }


async def _process_and_fix_issues(
    auth_config: AuthConfig,
    llm_config: LLMConfig,
    branch: Optional[str],
    pull_request: Optional[str],
    fix_params: Dict[str, Any],
    issue_type: IssueType = IssueType.REGULAR,
    download_latest: bool = True,
    system_ask: bool = True,
    check_tmp_path: bool = True,
) -> None:
    services = _initialize_fix_services(auth_config, llm_config)

    branch_downloaded = branch
    if download_latest:
        if pull_request and int(pull_request) > 0:
            branch_downloaded = services["analyzer"].get_branch_from_pr(
                project_key=auth_config.project, pull_request=pull_request
            )
        if not branch_downloaded:
            console.print("[red]Could not determine branch to download[/red]")
            raise click.Abort()

    async with TmpCloneManager(
        on_cleanup=lambda p: console.print(
            f"[dim]Cleaning up temporary files: {p}[/dim]"
        )
    ) as tmp_path:
        if download_latest:
            console.print(f"Cloning {auth_config.project} to {tmp_path}")
            downloaded = download_latest_version(
                auth_config.git_url, str(tmp_path), branch_downloaded
            )
            if not downloaded:
                console.print("Not able to download latest version")
                raise click.Abort()

        issues = _fetch_issues_by_type(
            services["analyzer"],
            auth_config,
            branch,
            pull_request,
            fix_params,
            issue_type,
        )
        if not issues:
            msg = (
                "No fixable security issues found"
                if issue_type == IssueType.SECURITY
                else "No fixable issues found"
            )
            console.print(f"[yellow]{msg}[/yellow]")
            return

        total_issues = sum(len(issue_list) for issue_list in issues.values())
        console.print(f"\n[green]✓ Found {total_issues} fixable issues[/green]\n")

        await _process_files_with_agent(
            issues,
            services,
            auth_config,
            fix_params,
            tmp_path,
            agent_max_retries=2,
            system_ask=system_ask,
        )


def _initialize_fix_services(
    auth_config: AuthConfig, llm_config: LLMConfig
) -> Dict[str, Any]:
    console.print("[dim]Initializing services...[/dim]")
    return {
        "analyzer": SonarCloudAnalyzer(auth_config.token, auth_config.organization),
        "ruler": RuleAnalyzer(auth_config.token, auth_config.organization),
        "fixer": LLMFixer(
            provider=llm_config.provider,
            model=llm_config.model,
            api_key=llm_config.api_key,
        ),
    }


async def _process_files_with_agent(
    issues_by_file: Dict[str, List[Any]],
    services: Dict[str, Any],
    auth_config: AuthConfig,
    fix_params: Dict[str, Any],
    tmp_path: Path,
    agent_max_retries: int = 2,
    system_ask: bool = True,
) -> None:
    """
    Process all issues using the LangGraph AgentSupervisor.

    Each issue runs inside a scoped progress spinner. Phase messages from the
    graph nodes are buffered during the spinner (to avoid the Rich live-display
    conflict) and flushed to the terminal only after the spinner closes — each
    on its own line with a blank line before it for readability.
    """
    fixer: LLMFixer = services["fixer"]

    supervisor = build_supervisor(
        provider=fixer.provider,
        model=fixer.model,
        api_key=fixer.api_key,
        use_validator=True,
        validator_provider=fixer.provider,
        validator_model=fixer.model,
        validator_api_key=fixer.api_key,
    )

    all_issues: List[Any] = []
    for _rule, issue_list in issues_by_file.items():
        all_issues.extend(issue_list)

    if not all_issues:
        console.print("[yellow]No issues to process with agent[/yellow]")
        return

    console.print(
        f"\n[bold cyan]🤖 AgentSupervisor — {len(all_issues)} issue(s) "
        f"(max retries per fix: {agent_max_retries})[/bold cyan]\n"
    )

    file_md = (
        Path(str(auth_config.project_path))
        / f"CHANGES_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.md"
    )

    for idx, issue in enumerate(all_issues, 1):
        rule = getattr(issue, "rule", "?")
        file_hint = getattr(issue, "file_path", getattr(issue, "file", "?"))

        console.rule(
            f"[bold]Issue {idx}/{len(all_issues)}[/bold]  "
            f"[yellow]{rule}[/yellow]  [dim]{file_hint}[/dim]"
        )

        # Buffer phase messages while the spinner owns the terminal.
        # Each message is stored as a Rich markup string and printed
        # after the spinner closes so they appear clean on separate lines.
        phase_messages: List[str] = []

        with show_progress("Running fix pipeline…", total=None) as (progress, task):
            result = await supervisor.run(
                issues=[issue],
                project_path=Path(str(auth_config.project_path)),
                tmp_path=tmp_path,
                create_backup=bool(fix_params.get("create_backup", False)),
                dry_run=bool(fix_params.get("dry_run", False)),
                file_md=str(file_md),
                skip_tests=bool(fix_params.get("skip_tests", True)),
                status_reporter=phase_messages.append,
            )
            progress.update(task, completed=1, total=1)
        # ↑ spinner closed — terminal is now free

        # Flush buffered phase messages, each preceded by a blank line.
        for msg in phase_messages:
            console.print()
            console.print(msg)

        _display_fix_results2(result)

        # Only prompt when more issues remain.
        if idx < len(all_issues) and system_ask:
            next_issue = all_issues[idx]
            next_rule = getattr(next_issue, "rule", "?")
            next_file = getattr(
                next_issue, "file_path", getattr(next_issue, "file", "?")
            )
            if not await smart_confirm(
                f"Continue to next issue? {next_rule} in {next_file})",
                default=True,
            ):
                console.print("[yellow]Stopped — remaining issues skipped[/yellow]")
                break


async def _process_files_with_issues(
    issues_by_file: Dict[str, List[Any]],
    services: Dict[str, Any],
    auth_config: AuthConfig,
    fix_params: Dict[str, Any],
    issue_type: IssueType,
    tmp_path: Path,
    system_ask: bool = True,
    check_tmp_path: bool = True,
) -> None:
    md_file_path = (
        Path(str(auth_config.project_path))
        / f"CHANGES_{issue_type.value.upper()}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.md"
    )
    if issue_type == IssueType.SECURITY:
        await _process_security_issues(
            issues_by_file,
            services,
            auth_config,
            fix_params,
            md_file_path,
            tmp_path,
            system_ask,
            check_tmp_path,
        )
    else:
        issues_by_rule_nested = {
            rule_key: {"issue": issues} for rule_key, issues in issues_by_file.items()
        }
        await _process_regular_issues(
            issues_by_rule_nested,
            services,
            auth_config,
            fix_params,
            md_file_path,
            tmp_path,
            system_ask,
            check_tmp_path,
        )


async def _process_regular_issues(
    issues_by_rule: Dict[str, Dict[str, List[Any]]],
    services: Dict[str, Any],
    auth_config: AuthConfig,
    fix_params: Dict[str, Any],
    md_file_path: Path,
    tmp_path: Path,
    system_ask: bool = True,
    check_tmp_path: bool = True,
) -> None:
    total_rules = len(issues_by_rule)
    for rule_num, (rule_key, rule_data) in enumerate(issues_by_rule.items(), 1):
        issues_list = rule_data["issue"]
        console.print(
            f"\n[blue]Processing Rule ({rule_num}/{total_rules}): {rule_key}[/blue]"
        )
        success = await _process_issues_for_rule(
            rule_key,
            issues_list,
            services,
            auth_config,
            fix_params,
            md_file_path,
            tmp_path,
            system_ask,
            check_tmp_path,
        )
        if not success:
            console.print(f"[red]Failed processing {rule_key}, skipping[/red]")


async def _process_issues_for_rule(
    rule_key: str,
    issues_list: List[Any],
    services: Dict[str, Any],
    auth_config: AuthConfig,
    fix_params: Dict[str, Any],
    md_file_path: Path,
    tmp_path: Path,
    system_ask: bool = True,
    check_tmp_path: bool = True,
) -> bool:
    total_issues = len(issues_list)
    for idx, issue in enumerate(issues_list, 1):
        if _should_skip_file(issue.file):
            console.print(
                f"[dim]Skipping {issue.file} "
                f"(only .py files excluding test_ are processed)[/dim]"
            )
            continue

        await _process_single_fix(
            issues=[issue],
            services=services,
            auth_config=auth_config,
            fix_params=fix_params,
            issue_type=IssueType.REGULAR,
            rule_key=rule_key,
            md_file_path=md_file_path,
            tmp_path=tmp_path,
            check_tmp_path=check_tmp_path,
        )

        if not await _should_continue_to_next_issue(
            idx, total_issues, system_ask=system_ask
        ):
            return False

    return True


async def _process_single_fix(
    issues: List[Any],
    services: Dict[str, Any],
    auth_config: AuthConfig,
    fix_params: Dict[str, Any],
    issue_type: IssueType,
    rule_key: str,
    tmp_path: Path,
    md_file_path: Optional[Path] = None,
    check_tmp_path: bool = True,
) -> None:
    fixes = await _generate_fix_for_file(
        issues,
        services,
        auth_config,
        issue_type,
        str(tmp_path),
        rule_key,
        md_file_path,
        check_tmp_path=check_tmp_path,
    )
    if fixes:
        for fix in fixes:
            await handle_fix(fix, issues, services["fixer"], auth_config, fix_params)
    else:
        console.print("[yellow]No fix could be generated[/yellow]")


async def _process_security_issues(
    issues_by_file: Dict[str, List[Any]],
    services: Dict[str, Any],
    auth_config: AuthConfig,
    fix_params: Dict[str, Any],
    md_file_path: Path,
    tmp_path: Path,
    system_ask: bool = True,
    check_tmp_path: bool = True,
) -> None:
    total_files = len(issues_by_file)
    for idx, (file_key, issues) in enumerate(issues_by_file.items(), 1):
        console.print(f"\n[blue]Processing ({idx}/{total_files}): {file_key}[/blue]")
        for _idx_new, issue in enumerate(issues, 1):
            if _should_skip_file(issue.file):
                console.print(
                    f"[dim]Skipping {issue.file} "
                    f"(only .py files excluding test_ are processed)[/dim]"
                )
                continue

            await _process_single_fix(
                issues=[issue],
                services=services,
                auth_config=auth_config,
                fix_params=fix_params,
                issue_type=IssueType.SECURITY,
                rule_key=file_key,
                md_file_path=md_file_path,
                tmp_path=tmp_path,
                check_tmp_path=check_tmp_path,
            )

            if not await _should_continue_to_next_issue(
                idx, total_files, system_ask=system_ask
            ):
                break


async def handle_fix(
    fix: FixSuggestion,
    issues: List[Any],
    fixer: LLMFixer,
    auth_config: AuthConfig,
    fix_params: Dict[str, Any],
) -> None:
    _display_fix_preview(fix, issues)
    if fix_params["apply"]:
        result = await fixer.apply_fixes_with_validation(
            fixes=[fix],
            issues=issues,
            project_path=Path(str(auth_config.project_path)),
            create_backup=fix_params.get("create_backup", False),
            dry_run=fix_params["dry_run"],
            use_validator=True,
            validator_provider=fixer.provider,
            validator_model=fixer.model,
            validator_api_key=fixer.api_key,
        )
        _display_fix_results(result)
    else:
        console.print(f"[dim]{constant.SKIPPED}[/dim]")


async def _generate_fix_for_file(
    issues: List[Any],
    services: Dict[str, Any],
    auth_config: AuthConfig,
    issue_type: IssueType,
    tmp_path: str,
    rule_name: Optional[str] = None,
    md_file_path: Optional[Path] = None,
    check_tmp_path: bool = True,
) -> Optional[List[FixSuggestion]]:
    with show_progress("Generating fixes...", total=len(issues)) as (progress, task):
        if issue_type != IssueType.SECURITY and not rule_name:
            console.print("rule_name is required for non-security issues")
            return None

        file_md_str = str(md_file_path) if md_file_path else ""
        fixer: Any = services["fixer"]
        result: Optional[List[FixSuggestion]] = await fixer.generate_fix_by_file(
            issues=issues,
            project_path=Path(str(auth_config.project_path)),
            tmp_path=Path(tmp_path),
            file_md=file_md_str,
            check_tmp_path=check_tmp_path,
        )
        return result


def _collect_rule_information(
    issues: List[Any],
    ruler: RuleAnalyzer,
) -> Dict[str, Any]:
    """Collect rule information, checking local JSON cache first.

    Falls back to the SonarCloud API only for rules absent from the cache.
    """
    rules_cache = _load_rules_cache()
    rule_info_list: Dict[str, Any] = {}

    for rule_key in issues:
        if rule_key in rules_cache:
            rule_info_list[rule_key] = rules_cache[rule_key]
        else:
            console.print("Cache miss for rule '%s'. Fetching from API.", rule_key)
            rule_info_list[rule_key] = ruler.get_rule_by_key(rule_key)

    return rule_info_list


async def _should_continue_to_next_issue(
    current_idx: int, total_files: int, system_ask: bool = True
) -> bool:
    """Check if should continue to next file (used by legacy rule-based flow)."""
    if current_idx >= total_files:
        return False
    if not system_ask:
        return True
    if not await smart_confirm("Continue to next issue?", default=True):
        console.print("[yellow]Stopped processing remaining files[/yellow]")
        return False
    return True


def _display_fix_preview(fix: FixSuggestion, issues: Sequence[Any]) -> None:
    console.print("\n[bold]Fix Preview:[/bold]")
    console.print(f"File: [cyan]{fix.file_path}[/cyan]")
    console.print(f"Confidence: [cyan]{fix.confidence:.2f}[/cyan]")
    console.print(f"Issues fixed: [cyan]{len(issues)}[/cyan]")
    console.print("\n[bold]Changes:[/bold]")
    if fix.explanation and fix.explanation.strip():
        console.print(
            Panel(fix.explanation, title="Explanation of changed", border_style="green")
        )


def _display_fix_preview2(fix: FixSuggestion) -> None:
    console.print("\n[bold]Fix Preview:[/bold]")
    console.print(f"File: [cyan]{fix.file_path}[/cyan]")
    console.print(f"Confidence: [cyan]{fix.confidence:.2f}[/cyan]")
    console.print("\n[bold]Changes:[/bold]")
    if fix.explanation and fix.explanation.strip():
        console.print(
            Panel(fix.explanation, title="Explanation of changed", border_style="green")
        )


def _display_project_header(result: AnalysisResult) -> None:
    console.print(Panel.fit("[bold]Analysis Results[/bold]"))
    console.print(f"Project: [cyan]{result.project_key}[/cyan]")
    console.print(f"Total Issues: [cyan]{result.total_issues}[/cyan]")


def _display_metrics_section(result: AnalysisResult) -> None:
    if result.metrics:
        console.print("\n[bold]Metrics:[/bold]")
        if result.metrics.lines_of_code:
            console.print(f"  Lines: {result.metrics.lines_of_code:,}")
        if result.metrics.coverage:
            console.print(f"  Coverage: {result.metrics.coverage:.1f}%")
        if result.metrics.bugs:
            console.print(f"  Bugs: {result.metrics.bugs}")
        if result.metrics.vulnerabilities:
            console.print(f"  Vulnerabilities: {result.metrics.vulnerabilities}")


def _display_issues_table(result: AnalysisResult, limit: Optional[int]) -> None:
    if result.issues:
        issues_to_show = result.issues[:limit] if limit else result.issues
        table = Table(show_header=True, header_style=constant.BOLD_MAGENTA)
        table.add_column("Severity", width=10)
        table.add_column("Type", width=15)
        table.add_column("File", width=30)
        table.add_column("Message", width=50)
        for issue in issues_to_show:
            table.add_row(
                issue.severity,
                issue.type,
                issue.file or "N/A",
                (
                    issue.message[:47] + "..."
                    if len(issue.message) > 50
                    else issue.message
                ),
            )
        console.print("\n")
        console.print(table)


def _display_analysis_results(result: AnalysisResult, limit: Optional[int]) -> None:
    _display_project_header(result)
    _display_metrics_section(result)
    _display_issues_table(result, limit)


def _display_fix_results(result: FixResult) -> None:
    console.print("\n[bold]Results:[/bold]")
    console.print(f"Attempted: {result.total_fixes_attempted}")
    console.print(f"Successful: [green]{len(result.successful_fixes)}[/green]")
    console.print(f"Failed: [red]{len(result.failed_fixes)}[/red]")
    console.print(f"Success Rate: {result.success_rate:.1%}")


def _display_fix_results2(result: FixResult) -> None:
    console.print("\n[bold]Results:[/bold]")
    console.print(f"Attempted: {result.total_fixes_attempted}")
    console.print(f"Successful: [green]{len(result.successful_fixes)}[/green]")
    console.print(f"Failed: [red]{len(result.failed_fixes)}[/red]")
    console.print(f"Success Rate: {result.success_rate:.1%}")
    for success in result.successful_fixes:
        _display_fix_preview2(success)


def _fetch_issues_by_type(
    analyzer: SonarCloudAnalyzer,
    auth_config: AuthConfig,
    branch: Optional[str],
    pull_request: Optional[str],
    fix_params: Dict[str, Any],
    issue_type: IssueType,
) -> Dict[str, List[Any]]:
    pr_number = _safe_convert_pr(pull_request) if pull_request else 0
    if issue_type == IssueType.SECURITY:
        with show_progress("Fetching security issues...") as (progress, task):
            return analyzer.get_fixable_security_issues(
                project_key=auth_config.project,
                branch=branch or "",
                pull_request=pr_number,
                max_issues=fix_params["max_fixes"],
                language=PYTHON,
            )
    else:
        with show_progress(constant.FETCHING_ISSUES) as (progress, task):
            return analyzer.get_fixable_issues_by_types(
                project_key=auth_config.project,
                branch=branch or "",
                pull_request=pr_number,
                max_issues=fix_params["max_fixes"],
                severities=fix_params["severities_list"],
                types_list=fix_params["types_list"],
                rules_excluded=fix_params["exclude_rules"],
                group_by="rules",
                languages=[PYTHON.sonar_language_key, IPYTHON.sonar_language_key],
            )


if __name__ == "__main__":
    main()
