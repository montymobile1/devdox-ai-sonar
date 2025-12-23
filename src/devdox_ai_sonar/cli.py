"""Command-line interface for SonarCloud Analyzer with Claude Code-style interaction and mid-workflow command switching."""


import sys
from pathlib import Path
from typing import Optional, List, Any,  Sequence, Dict, Tuple, Union, Set

import questionary
from questionary import ValidationError as QValidationError
from questionary import Validator
from rich.console import Console
import inquirer
from rich.table import Table
from rich.prompt import Confirm
from rich.panel import Panel
from rich.progress import Progress
import rich_click as click
from questionary.prompts.common import Choice
from contextlib import contextmanager


from devdox_ai_sonar import __version__
from devdox_ai_sonar.sonar_analyzer import SonarCloudAnalyzer

from devdox_ai_sonar.services.rule_analyzer import RuleAnalyzer
from devdox_ai_sonar.llm_fixer import LLMFixer
from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.utils.validator import InputValidator
from devdox_ai_sonar.utils.exceptions import ValidationError
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

from devdox_ai_sonar.utils.exceptions import SwitchCommandException, ReturnToMenuException
from devdox_ai_sonar.utils.ui import smart_prompt, smart_confirm
from devdox_ai_sonar.utils import constant
from devdox_ai_sonar.config import settings


console = Console()


# ============================================================================
# EXCEPTIONS FOR COMMAND SWITCHING
# ============================================================================



def _safe_convert_pr(pull_request: Optional[str]) -> int:
    """Safely convert PR string to integer."""
    if not pull_request:
        return 0

    try:
        pr_int = int(pull_request.strip())
        if pr_int < 0:
            console.print("[yellow]⚠ Negative PR numbers not allowed, using 0[/yellow]")
            return 0
        return pr_int
    except (ValueError, AttributeError):
        console.print(f"[yellow]⚠ Invalid PR number '{pull_request}', using 0[/yellow]")
        return 0


@contextmanager
def show_progress(message: str, total: Optional[int] = None):
    """Context manager for progress display."""
    with Progress() as progress:
        task = progress.add_task(message, total=total)
        try:
            yield progress, task
        finally:
            if not progress.finished:
                progress.remove_task(task)


# ============================================================================
# INTERACTIVE PROMPT WITH COMMAND SWITCHING
# ============================================================================










# ============================================================================
# INTERACTIVE COMMAND SELECTOR (Claude Code Style)
# ============================================================================


def show_command_selector() -> Optional[str]:
    """
    Show an interactive command selector similar to Claude Code.
    Returns the selected command or None if cancelled.
    """
    try:
        import questionary
        from questionary import Style
    except ImportError:
        console.print("[yellow]⚠ questionary not installed. Using fallback prompt.[/yellow]")
        console.print("[dim]Install with: pip install questionary[/dim]\n")
        return _fallback_command_selector()

    # Define custom style similar to Claude Code
    custom_style = Style([
        ('qmark', constant.BOLD_PURPLE),
        ('question', 'bold'),
        ('answer', 'fg:#f44336 bold'),
        ('pointer', constant.BOLD_PURPLE),
        ('highlighted', constant.BOLD_PURPLE),
        ('selected', 'fg:#cc5454'),
        ('separator', 'fg:#cc5454'),
        ('instruction', ''),
        ('text', ''),
    ])

    commands = [
        {
            'name': '🔍 Add Provider - Add provider or sonar configuration',
            'value': 'add_provider',
            'description': 'Add provider configuration'
        },
        {
            'name': '🔍 Update Provider - Update provider or sonar configuration',
            'value': 'update_provider',
            'description': 'Update SonarCloud provider configuration'
        },
        {
            'name': '🔧 Fix Issues - Generate and apply LLM-powered fixes',
            'value': 'fix_issues',
            'description': 'Analyze and fix code quality issues'
        },
        {
            'name': '🔒 Fix Security Issues - Specialized security vulnerability fixes',
            'value': 'fix_security_issues',
            'description': 'Focus on security vulnerabilities'
        },
        {
            'name': '📊 Analyze Project - Display SonarCloud analysis',
            'value': 'analyze',
            'description': 'View project metrics and issues'
        },
        {
            'name': '🔍 Inspect Project - Analyze local directory structure',
            'value': 'inspect',
            'description': 'Inspect local project structure'
        },
        {
            'name': '📊 Change Parameters Configuration ',
            'value': 'change_parameters',
            'description': 'Change parameters configuration'
        },
        {
            'name': '❌ Exit',
            'value': 'exit',
            'description': 'Exit the application'
        }
    ]

    # Clear screen for clean display
    console.clear()

    # Show header
    console.print()
    console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   DevDox AI Sonar - Interactive Mode          [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
    console.print()

    try:
        choice = questionary.select(
            "What would you like to do?",
            choices=[cmd['name'] for cmd in commands],
            style=custom_style,
            use_shortcuts=True,
            use_arrow_keys=True,
            use_jk_keys=True,
        ).ask()

        if choice is None:  # User pressed Ctrl+C
            return None

        # Find the command value
        for cmd in commands:
            if cmd['name'] == choice:
                return cmd['value']

        return None

    except KeyboardInterrupt:
        return None


def _fallback_command_selector() -> Optional[str]:
    """Fallback text-based command selector when questionary is not available."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   DevDox AI Sonar - Command Selection          [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]\n")

    commands = {
        '1': ('fix_issues', 'Fix Issues - Generate and apply LLM-powered fixes'),
        '2': ('fix_security_issues', 'Fix Security Issues - Specialized security fixes'),
        '3': ('analyze', 'Analyze Project - Display SonarCloud analysis'),
        '4': ('inspect', 'Inspect Project - Analyze local directory structure'),
        '5': ('exit', 'Exit'),
    }

    for key, (_, desc) in commands.items():
        console.print(f"[cyan]{key}[/cyan]. {desc}")

    console.print()
    choice = console.input("[bold yellow]Select command (1-5): [/bold yellow]").strip()

    if choice in commands:
        return commands[choice][0]
    return None


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


@click.command()
@click.version_option(__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--command", "-c",
    type=click.Choice(['fix_issues', 'fix_security_issues', 'analyze', 'inspect']),
    help="Run specific command directly without interactive mode"
)
@click.option(
    "--types",
    type=str,
    help="Comma-separated issue types (for fix_issues)"
)
@click.option(
    "--severity",
    type=str,
    help="Comma-separated severities (for fix_issues)"
)
@click.option(
    "--max-fixes",
    type=click.IntRange(0, settings.MAX_FIXES_LIMIT),
    help=f"Maximum number of fixes (0-{settings.MAX_FIXES_LIMIT})"
)
@click.option("--apply", is_flag=True, help="Apply fixes to the codebase")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be changed without applying fixes"
)
@click.pass_context
def main(
    ctx: click.Context,
    verbose: bool,
    command: Optional[str],
    types: Optional[str],
    severity: Optional[str],
    max_fixes: Optional[int],
    apply: bool = False,
    dry_run: bool = False
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
    ctx.obj['verbose'] = verbose
    ctx.obj['options'] = {
        'types': types,
        'severity': severity,
        'max_fixes': max_fixes or 0,
        'apply': apply,
        'dry_run': dry_run,
    }

    # If command specified, run it directly
    if command:
        _execute_command(ctx, command)
        return
    init_config()
    # Otherwise, enter interactive mode
    _run_interactive_mode(ctx)

def _select_existing_ui(field_name:str, message:str,existing_providers: list) -> str:
    """Prompt user to select an existing provider.

    Args:
        existing_providers: List of configured provider names

    Returns:
        str: Selected provider name, or empty string if cancelled
    """
    questions = [
        inquirer.List(
            field_name,
            message=message,
            choices=existing_providers,
        )
    ]

    answers = inquirer.prompt(questions)
    if not answers or not answers.get(field_name):
        console.print("[yellow]⚠ Selection cancelled[/yellow]")
        return ""

    return answers[field_name]



def _initialize_managers():
    """Initialize all required manager instances.

    Returns:
        tuple: (ConfigManager, ProviderConfigUI, ProviderValidator,
                ProviderConfigManager, SonarCloudConfigUI, ConfigService)
    """
    manager = ConfigManager(config_path=settings.config_file_path)
    ui = ProviderConfigUI()
    validator = ProviderValidator()
    provider_manager = ProviderConfigManager(manager, ui, validator)
    sonar_ui = SonarCloudConfigUI()
    config_service = ConfigService(sonar_path=Path(settings.auth_file_path))

    return manager, ui, validator, provider_manager, sonar_ui, config_service


def _configure_sonarcloud(sonar_ui: SonarCloudConfigUI, config_service: ConfigService) -> bool:
    """Configure SonarCloud settings.

    Args:
        sonar_ui: SonarCloud UI handler
        config_service: Configuration service

    Returns:
        bool: True if configuration successful, False otherwise
    """

    auth_config = config_service.load_auth_config()

    value_exist = config_service.check_all_value_empty(auth_config)
    if not value_exist:
        return True

    if auth_config is None:
        console.print("[red]❌ Auth configuration not found[/red]")
        return False
    sonar_ui.display_welcome()
    sonar_ui.display_step_header(1, 2, "SonarCloud Configuration")

    sonar_config = sonar_ui.configure_sonarcloud(auth_config)
    if not sonar_config:
        console.print("[red]❌ SonarCloud configuration cancelled[/red]")
        return False

    save_success = config_service.save_config(
        token=sonar_config.token,
        organization=sonar_config.organization,
        project=sonar_config.project,
        project_path=str(sonar_config.project_path),
    )

    if not save_success:
        click.secho("❌ SonarCloud configuration save failed", fg="red", bold=True)
        return False

    click.secho("\n✓ SonarCloud configuration saved", fg="green", bold=True)
    return True


def _configure_providers_loop(
        provider_manager: ProviderConfigManager,
        manager: ConfigManager,
        ui: ProviderConfigUI,
        available_providers: list,
) -> None:
    """Configure providers in a loop until done or cancelled.

    Args:
        provider_manager: Provider configuration manager
        manager: Config manager instance
        ui: Provider UI handler
        available_providers: List of available provider names
    """
    while available_providers:
        console.print(f"[cyan]Available providers: {', '.join(available_providers)}[/cyan]\n")

        provider_name = ui.select_provider_from_list(
            available_providers, "Select a provider to configure"
        )

        if not provider_name:
            break

        success = _handle_provider_configuration(
            provider_manager, manager, provider_name, available_providers
        )

        if not success:
            console.print(
                f"[yellow]⚠ Configuration of {provider_name} failed or was cancelled[/yellow]\n"
            )

        if _should_stop_configuring(available_providers):
            break


def _handle_provider_configuration(
        provider_manager: ProviderConfigManager,
        manager: ConfigManager,
        provider_name: str,
        available_providers: list,
) -> bool:
    """Handle configuration of a single provider.

    Args:
        provider_manager: Provider configuration manager
        manager: Config manager instance
        provider_name: Name of provider to configure
        available_providers: List to remove provider from on success

    Returns:
        bool: True if configuration successful
    """
    result = provider_manager.configure_new_provider(provider_name)

    if not result:
        return False

    manager.add_provider(result["config"], set_as_default=result["set_as_default"])
    available_providers.remove(provider_name)
    console.print(
        f"\n[green]✓ {provider_name.upper()} configured successfully[/green]\n"
    )
    return True


def _should_stop_configuring(available_providers: list) -> bool:
    """Determine if provider configuration loop should stop.

    Args:
        available_providers: List of remaining providers

    Returns:
        bool: True if should stop configuring
    """
    if not available_providers:
        console.print("[green]✅ All providers have been configured[/green]")
        return True

    return not Confirm.ask("Add another provider?", default=False)


def _check_reconfiguration_consent(providers: list) -> bool:
    """Check if user wants to reconfigure existing setup.

    Args:
        providers: List of existing providers

    Returns:
        bool: True if should proceed with reconfiguration
    """
    if not providers or len(providers) == 0:
        return True

    click.secho("\n⚠️  Configuration already exists", fg="yellow", bold=True)


    return False


def _select_existing_provider(existing_providers: list) -> str:
    """Prompt user to select an existing provider.

    Args:
        existing_providers: List of configured provider names

    Returns:
        str: Selected provider name, or empty string if cancelled
    """
    questions = [
        inquirer.List(
            "provider",
            message="Select the provider to update",
            choices=existing_providers,
        )
    ]

    answers = inquirer.prompt(questions)
    if not answers or not answers.get("provider"):
        console.print("[yellow]⚠ Selection cancelled[/yellow]")
        return ""

    return answers["provider"]


def _display_operation_header(operation: str) -> None:
    """Display formatted operation header.

    Args:
        operation: Operation name to display
    """
    console.print("\n" + "=" * 60)
    console.print(f"[magenta bold]{operation}[/magenta bold]")
    console.print("=" * 60 + "\n")


def _display_completion_message() -> None:
    """Display configuration completion message."""
    console.print("=" * 60)
    console.print("[green bold]✨ CONFIGURATION COMPLETE![/green bold]")
    console.print("=" * 60)
    console.print(f"\nConfiguration saved to: {settings.config_file_path}\n")


def _handle_cli_error(error: Exception) -> None:
    """Handle CLI errors with consistent formatting.

    Args:
        error: Exception that occurred

    Raises:
        click.ClickException: Re-raises as click exception
    """
    if isinstance(error, click.Abort):
        console.print("\n[red]Configuration cancelled[/red]")
        raise

    console.print(f"\n[red]❌ Error during configuration: {str(error)}[/red]")
    raise click.ClickException(str(error))


# ============================================================================
# MAIN CLI COMMANDS - Simplified with extracted functions
# ============================================================================





def init_config(  types: Optional[str] = None,
    severity: Optional[str] = None,
    max_fixes: int = 0,
    apply: bool=False,
    dry_run: bool=False):
    """CLI for config management - Complexity: 8 (under limit of 15)"""
    try:
        # Initialize all managers
        manager, ui, _ , provider_manager, sonar_ui, config_service = _initialize_managers()
        manager.create_default_config()
        manager.load_config()


        # Check if reconfiguration is needed
        providers = manager.get_value("llm.providers")
        if not _check_reconfiguration_consent(providers):
            return


        # Configure SonarCloud
        if not _configure_sonarcloud(sonar_ui, config_service):
            raise click.Abort()

        # Skip provider configuration if already exists
        if providers and len(providers) > 0:
            return

        # Configure providers
        available_providers = provider_manager.get_available_providers()
        if not available_providers:
            console.print(
                "\n[red]❌ No providers configured. Configuration incomplete.[/red]"
            )
            raise click.Abort()

        _configure_providers_loop(provider_manager, manager, ui, available_providers)
        manager.save_config(create_backup=False)
        apply_value = 1 if apply else 0
        dry_run_value = 1 if dry_run else 0
        change_parameters(types, severity,max_fixes=max_fixes, apply=apply_value, dry_run=dry_run_value)




    except Exception as e:
        _handle_cli_error(e)


def add_provider():
    """CLI command for managing provider configuration"""
    try:
        # Initialize components
        manager, ui, _, provider_manager, _, _ = _initialize_managers()
        manager.load_config()

        _handle_add_new_provider(provider_manager, manager, ui)
        _display_completion_message()

    except Exception as e:
        _handle_cli_error(e)


def update_provider():
    """CLI command for managing provider configuration - Complexity: 7 (under limit of 15)"""
    try:
        # Initialize components
        manager, _, _, provider_manager, _, _ = _initialize_managers()
        manager.load_config()

        existing_providers = provider_manager.get_existing_providers()
        if not existing_providers:
            console.print(
                "\n[red]❌ No providers configured. Please add at least one provider first.[/red]"
            )
            raise click.Abort()

        _handle_update_existing_provider(provider_manager, manager, existing_providers)

        _display_completion_message()

    except Exception as e:
        _handle_cli_error(e)

def _handle_add_new_provider(
        provider_manager: ProviderConfigManager,
        manager: ConfigManager,
        ui: ProviderConfigUI,
) -> None:
    """Handle adding new provider flow - Complexity: 5"""
    available_providers = provider_manager.get_available_providers()

    if not available_providers:
        console.print(
            "[yellow]⚠ All supported providers are already configured[/yellow]"
        )
        raise click.Abort()

    _display_operation_header("🚀 ADD NEW PROVIDER")
    _configure_providers_loop(provider_manager, manager, ui, available_providers)
    manager.save_config(create_backup=False)


def _handle_update_existing_provider(
        provider_manager: ProviderConfigManager,
        manager: ConfigManager,
        existing_providers: list,
) -> None:
    """Handle updating existing provider flow - Complexity: 4"""
    _display_operation_header("🔧 UPDATE EXISTING PROVIDER")

    chosen_provider = _select_existing_ui("provider","Select the provider to update",existing_providers)
    if not chosen_provider:
        raise click.Abort()

    if provider_manager.update_existing_provider(chosen_provider):
        manager.save_config(create_backup=False)
        console.print(
            f"\n[green]✓ {chosen_provider.upper()} updated successfully[/green]\n"
        )
    else:
        console.print("\n[yellow]⚠ Update cancelled or failed[/yellow]\n")


def _run_interactive_mode(ctx: click.Context) -> None:
    """Run the interactive command selection loop with command switching support."""
    while True:
        try:
            command = show_command_selector()

            if _should_exit_interactive_mode(command):
                _exit_application()
                return

            _execute_interactive_command(ctx, command)

            if not _should_continue_to_menu():
                _exit_application()
                return

        except SwitchCommandException:
            _handle_command_switch()
            continue

        except KeyboardInterrupt:
            if _handle_keyboard_interrupt():
                return

        except Exception as e:
            if _handle_interactive_error(e):
                return


def _should_exit_interactive_mode(command: Optional[str]) -> bool:
    """Check if user wants to exit interactive mode."""
    return command is None or command == 'exit'


def _exit_application() -> None:
    """Exit the application with goodbye message."""
    console.print("\n[cyan]👋 Thank you for using DevDox AI Sonar![/cyan]")
    sys.exit(0)


def _execute_interactive_command(ctx: click.Context, command: str) -> None:
    """Execute a command in interactive mode."""
    console.print(f"\n[bold green]▶ Running: {command}[/bold green]\n")
    _execute_command(ctx=ctx, command=command)
    console.print("\n" + "─" * 50 + "\n")


def _should_continue_to_menu() -> bool:
    """Ask user if they want to return to main menu."""
    return smart_confirm("Return to main menu?", default=True, allow_switch=False)


def _handle_command_switch() -> None:
    """Handle command switching exception."""
    console.print("\n[yellow]↩ Returning to command menu...[/yellow]\n")


def _handle_keyboard_interrupt() -> bool:
    """
    Handle keyboard interrupt (Ctrl+C).

    Returns:
        True if should exit, False if should continue
    """
    console.print("\n\n[yellow]⚠ Interrupted by user[/yellow]")

    try:
        if smart_confirm("Exit application?", default=False, allow_switch=False):
            sys.exit(0)
        return False

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Force exit...[/yellow]")
        sys.exit(130)

    except Exception as e:
        console.print(f"\n[red]Error during exit: {e}[/red]")
        sys.exit(1)


def _handle_interactive_error(error: Exception) -> bool:
    """
    Handle errors in interactive mode.

    Returns:
        True if should exit, False if should continue
    """
    console.print(f"\n[red]❌ Error: {str(error)}[/red]")

    try:
        if not smart_confirm("Return to main menu?", default=True, allow_switch=False):
            sys.exit(1)
        return False

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Exiting after error...[/yellow]")
        sys.exit(1)

    except Exception as confirm_error:
        console.print(f"\n[red]Fatal: Cannot prompt user ({confirm_error})[/red]")
        sys.exit(2)


def  change_field(manager, field,message, default_value, choices:List[str] = None, multiple:bool = True)->Optional[Union[str, List[str]]]:
    types_input = smart_prompt(
        message,
        default=default_value,
        choices=choices,
        multiple=multiple
    )

    types = types_input if types_input else None

    if types:
        manager.set_value(field, types)
    return types

def change_parameters(  types: Optional[str] = None,
    severity: Optional[str] = None, **kwargs):
    """CLI for config management """
    try:


        # Initialize all managers
        manager, _, _ , provider_manager, _, _ = _initialize_managers()
        branch, pull_request = provider_manager.branch_or_pr_prompt()

        if not branch and not pull_request:
            console.print(constant.NO_BRANCH_OR_PR_SPECIFIED)
            raise click.Abort()

        max_fixes = manager.get_value("configuration.max_fixes") or 0

        # Max fixes
        _= change_max_fix(manager,f"Maximum fixes to generate (0-{settings.MAX_FIXES_LIMIT})", max_fixes,settings.MAX_FIXES_LIMIT )

        # Issue types (optional)
        if not types:
            _ = change_field(manager=manager, field="configuration.types",
                             message="Issue types (comma-separated, or press Enter to skip)",
                             default_value=manager.get_value("configuration.types"),

                             choices=list(InputValidator.VALID_ISSUE_TYPES))

        # Severities (optional)
        if not severity:
            _ = change_field(manager=manager, field="configuration.severities",
                             message="Issue severities (comma-separated, or press Enter to skip)",
                             default_value=manager.get_value("configuration.severities"),
                             choices=list(InputValidator.VALID_SEVERITIES))

        choices = [{"name":"yes","value":1},{"name":"no","value":0}]

        formatted_choices = [Choice(title=choice['name'], value=choice['value']) for choice in choices ]


        _ = change_field(manager=manager, field=constant.CONFIGURATION_APPLY,
                         message="Apply fixes of SonarQube (press Enter to skip)",
                        default_value=manager.get_value(constant.CONFIGURATION_APPLY) if manager.get_value(constant.CONFIGURATION_APPLY) is not None else kwargs.get("apply", 0) , # optional
                        choices=formatted_choices,
                        multiple=False)

        _ = change_field(manager=manager, field=constant.CONFIGURATION_BACKUP,
                         message="Create backup before apply fixes (press Enter to skip)",
                         default_value=manager.get_value(constant.CONFIGURATION_BACKUP) if manager.get_value(
                             constant.CONFIGURATION_BACKUP) is not None else kwargs.get("create_backup", 0),  # optional
                         choices=formatted_choices,
                        multiple=False)

        manager.save_config(create_backup=False)

    except Exception as e:
        _handle_cli_error(e)

def  change_max_fix(manager, message , max_fixes,default_max_fixes):

    max_fixes_str = smart_prompt(
                message,
                default=str(max_fixes)
    )
    try:
            max_fixes = int(max_fixes_str)
            if max_fixes < 1 or max_fixes >default_max_fixes:
                max_fixes = settings.DEFAULT_MAX_FIXES
    except ValueError:
                max_fixes = default_max_fixes


    manager.set_value("configuration.max_fixes", max_fixes)

def _execute_command( ctx: click.Context,command: str) -> None:
    """Execute a specific command with command switching support."""
    options = ctx.obj.get('options', {})

    try:
        if command == 'fix_issues':
            _run_fix_issues( **options)
        elif command == 'fix_security_issues':
            _run_fix_security_issues( **options)
        elif command == 'analyze':
            _run_analyze( **options)
        elif command == 'inspect':
            _run_inspect()
        elif command == 'config':
            init_config( **options)
        elif command == 'add_provider':
            add_provider()
        elif command == 'update_provider':
            update_provider()
        elif command == 'change_parameters':
            change_parameters( **options)
        else:
            console.print(f"[red]Unknown command: {command}[/red]")

    except SwitchCommandException:
        console.print("\n[yellow]↩ Switching commands...[/yellow]")
        raise  # Re-raise to be caught by interactive mode loop

    except ValidationError as e:
        console.print(f"\n[red]❌ {e.message}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]", markup=False)
        if ctx.obj.get('verbose'):
            console.print_exception()
        raise


# ============================================================================
# COMMAND IMPLEMENTATIONS WITH SWITCH SUPPORT
# ============================================================================


def _run_fix_issues(
    **kwargs: Any
) -> None:
    """Run the fix_issues command with command switching support."""
    console.print("\n[bold cyan]🔧 Fix Issues - LLM-Powered Code Fixes[/bold cyan]\n")

    try:
        apply_value = 1 if kwargs.get("apply", False) else 0
        dry_run_value = 1 if kwargs.get("dry_run", False) else 0


        # Load and validate configuration
        auth_config, llm_config,parameters = _load_and_validate_config()

        fix_params = display_configuration(parameters, dry_run_value,apply_value)

        # Confirm before proceeding
        if not smart_confirm("Proceed with these settings?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        # Process issues
        _process_and_fix_issues(
            auth_config,
            llm_config,
            fix_params.get("branch",""),
            pull_request=parameters.get("pull_request",0),
            fix_params=fix_params
        )

    except SwitchCommandException:
        console.print("\n[yellow]↩ Switching commands...[/yellow]")
        raise  # Re-raise to be caught by interactive mode loop

def display_configuration(parameters, dry_run,apply):


        console.print("[bold]Configuration:[/bold]")
        pull_request = parameters.get("pull_request",0)
        branch = parameters.get("branch","")
        if isinstance(pull_request, int) and pull_request > 0:
            console.print(f"  Pull Request: [cyan]{pull_request}[/cyan]")
        elif branch:
            console.print(f"  Branch: [cyan]{branch}[/cyan]")
        console.print(f"  Max Fixes: [cyan]{parameters.get('max_fixes')}[/cyan]")
        console.print(f"  Apply: [cyan]{apply}[/cyan]")
        console.print(f"  Dry Run: [cyan]{dry_run}[/cyan]")

        fix_params={
        "pull_request":pull_request,
        "branch":branch,
        'max_fixes':  parameters.get("max_fixes",0),
        'types_list': _validate_issue_types(parameters.get("types","")),
        'severities_list': _validate_severities(parameters.get("severities","")),
        "apply":apply,
        'dry_run': dry_run,
        "create_backup":parameters.get("create_backup",0),
    }

        return fix_params



def _run_fix_security_issues(
    **kwargs: Any
) -> None:
    """Run the fix_security_issues command with command switching support."""
    console.print("\n[bold cyan]🔒 Fix Security Issues[/bold cyan]\n")

    try:
        auth_config, llm_config, parameters= _load_and_validate_config()


        apply_value = kwargs.get("apply", parameters.get("apply", 0) if parameters.get("apply") is not None else 0)
        dry_run_value = kwargs.get("dry_run", parameters.get("dry_run", 0) if parameters.get("dry_run") is not None else 0)

        fix_params = display_configuration(parameters, dry_run_value,apply_value)

        if not smart_confirm("Proceed with security issue fixing?", default=True):
            console.print("[yellow]Cancelled[/yellow]")
            return

        _process_security_issues(
            auth_config,
            llm_config,
            fix_params.get("branch",""),
            fix_params.get("pull_request",0),
            fix_params
        )

    except SwitchCommandException:
        raise


def _run_analyze(
        **kwargs: Any
) -> None:
    """Run the analyze command with command switching support."""
    console.print("\n[bold cyan]📊 Analyze SonarCloud Project[/bold cyan]\n")

    try:

        # Load and validate configuration
        auth_config, _, parameters = _load_and_validate_config()
        console.print(f"  Project: [cyan]{auth_config.project}[/cyan]")
        console.print(f"  Organization: [cyan]{auth_config.organization}[/cyan]\n")


        # Get limit
        limit_str = smart_prompt(
            f"Max issues to fetch (max: {settings.MAX_FIXES_LIMIT})",
            default=str(parameters.get("max_fixes",settings.MAX_FIXES_LIMIT))
        )

        try:
            limit = int(limit_str)
            if limit <= 0 or limit > settings.MAX_FIXES_LIMIT:
                limit = settings.MAX_FIXES_LIMIT
        except ValueError:
            console.print("[red]Invalid limit, using default[/red]")
            limit = settings.MAX_FIXES_LIMIT

        # Fetch and display results
        analyzer = SonarCloudAnalyzer(auth_config.token, auth_config.organization)

        with show_progress("Fetching issues...") as (progress, task):
            result = analyzer.get_project_issues(
                project_key=auth_config.project,
                branch=parameters.get("branch",""),
                max_issues=limit,
                pull_request_number=_safe_convert_pr(parameters.get("pull_request",0)),
                severities=_validate_severities(parameters.get("severity","")),
                types=_validate_issue_types(parameters.get("types","")),
            )

        if result:
            _display_analysis_results(result, limit)

    except SwitchCommandException:
        raise

def _run_inspect() -> None:
    """Run the inspect command."""
    console.print("\n[bold cyan]🔍 Inspect Local Project[/bold cyan]\n")

    try:
        # Use helper instead of inline code
        auth_config, _, _ = _load_and_validate_config()

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
        table.add_row("Has SonarCloud Config", "✓" if analysis["has_sonar_config"] else "✗")
        table.add_row("Has Git Repository", "✓" if analysis["has_git"] else "✗")

        console.print(table)

        if analysis["potential_source_dirs"]:
            console.print("\n[bold]Potential Source Directories:[/bold]")
            for src_dir in analysis["potential_source_dirs"]:
                console.print(f"  • {src_dir}")

    except SwitchCommandException:
        raise
# ============================================================================
# HELPER FUNCTIONS (Reusing from original code)
# ============================================================================


def _load_and_validate_config() -> Tuple[AuthConfig, LLMConfig, Optional[str]]:
    """Load and validate configuration with command switching support."""
    console.print("[dim]Loading configuration...[/dim]")

    config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
    auth_config_dict = config_service.load_auth_config()

    if not auth_config_dict:
        console.print(constant.AUTHENTICATION_NOT_FOUND)
        console.print(constant.DEVDOX_SONAR_CONFIG)
        raise click.Abort()

    auth_config = AuthConfig(**auth_config_dict)
    is_valid, error_msg = auth_config.validate()

    if not is_valid:
        console.print(f"[red]❌ Configuration error: {error_msg}[/red]")
        raise click.Abort()

    # Load LLM config
    manager = ConfigManager(config_path=settings.config_file_path)
    manager.load_config()
    llm_config = config_service.load_llm_config(manager)

    if not llm_config:
        console.print("[red]❌ No LLM providers configured[/red]")
        raise click.Abort()

    # Get branch/PR with command switching support
    ui = ProviderConfigUI()
    validator = ProviderValidator()
    provider_manager = ProviderConfigManager(manager, ui, validator)
    branch, pull_request = provider_manager.branch_or_pr()

    if not branch and not pull_request:
        console.print(constant.NO_BRANCH_OR_PR_SPECIFIED)
        raise click.Abort()
    params = manager.get_value("configuration") or {}

    params['branch'] = branch
    params['pull_request'] = pull_request

    console.print("[green]✓[/green] Configuration loaded\n")
    return auth_config, llm_config,params


def _validate_issue_types(types_str: Optional[str]) -> Optional[List[str]]:
    """Validate issue types."""
    if not types_str:
        return None
    return InputValidator.validate_issue_types(types_str)


def _validate_severities(severity_str: Optional[str]) -> Optional[List[str]]:
    """Validate severities."""
    if not severity_str:
        return None
    return InputValidator.validate_severities(severity_str)


def _process_and_fix_issues(
        auth_config: AuthConfig,
        llm_config: LLMConfig,
        branch: Optional[str],
        pull_request: Optional[str],
        fix_params: Dict[str, Any]
) -> None:
    """Process and fix issues - Refactored."""
    services = _initialize_fix_services(auth_config, llm_config)

    fixable_issues = _fetch_fixable_issues(
        services['analyzer'],
        auth_config,
        branch,
        pull_request,
        fix_params
    )

    if not fixable_issues:
        console.print("[yellow]No fixable issues found[/yellow]")
        return

    console.print(f"\n[green]✓ Found {len(fixable_issues)} fixable issues[/green]\n")

    _process_each_file(
        fixable_issues,
        services,
        auth_config,
        fix_params
    )


def _initialize_fix_services(
        auth_config: AuthConfig,
        llm_config: LLMConfig
) -> Dict[str, Any]:
    """Initialize services for fixing issues."""
    console.print("[dim]Initializing services...[/dim]")

    return {
        'analyzer': SonarCloudAnalyzer(auth_config.token, auth_config.organization),
        'ruler': RuleAnalyzer(auth_config.token, auth_config.organization),
        'fixer': LLMFixer(
            provider=llm_config.provider,
            model=llm_config.model,
            api_key=llm_config.api_key
        )
    }


def _fetch_fixable_issues(
        analyzer: SonarCloudAnalyzer,
        auth_config: AuthConfig,
        branch: Optional[str],
        pull_request: Optional[str],
        fix_params: Dict[str, Any]
) -> Dict[str, List[Any]]:
    """Fetch fixable issues from SonarCloud."""
    with show_progress("Fetching issues...") as (progress, task):
        return analyzer.get_fixable_issues_by_files(
            project_key=auth_config.project,
            branch=branch or "",
            pull_request=int(pull_request) if pull_request else 0,
            max_issues=fix_params['max_fixes'],
            severities=fix_params['severities_list'],
            types_list=fix_params['types_list'],
        )


def _process_each_file(
        fixable_issues: Dict[str, List[Any]],
        services: Dict[str, Any],
        auth_config: AuthConfig,
        fix_params: Dict[str, Any]
) -> None:
    """Process each file with issues."""
    total_files = len(fixable_issues)

    for idx, (file_path, issues) in enumerate(fixable_issues.items(), 1):
        console.print(f"\n[blue]Processing ({idx}/{total_files}): {file_path}[/blue]")

        fix = _generate_fix_for_file(
            issues,
            file_path,
            services,
            auth_config
        )

        if fix:
            _handle_generated_fix(fix, issues, services['fixer'], auth_config, fix_params)
        else:
            console.print("[yellow]No fix could be generated[/yellow]")

        if not _should_continue_to_next_file(idx, total_files):
            break


def _generate_fix_for_file(
        issues: List[Any],
        file_path: str,
        services: Dict[str, Any],
        auth_config: AuthConfig
) -> Optional[FixSuggestion]:
    """Generate fix for a file."""
    with show_progress("Generating fixes...", total=len(issues)) as (progress, task):
        rule_info_list = _collect_rule_information(issues, services['ruler'])

        return services['fixer'].generate_fix_by_file(
            issues,
            Path(str(auth_config.project_path)),
            rule_info_list
        )


def _collect_rule_information(
        issues: List[Any],
        ruler: RuleAnalyzer
) -> Dict[str, Dict[str, str]]:
    """Collect rule information for all issues."""
    rule_info_list = {}
    for issue in issues:
        rule_info = ruler.get_rule_by_key(issue.rule)
        rule_info_list[issue.rule] = rule_info
    return rule_info_list


def _handle_generated_fix(
        fix: FixSuggestion,
        issues: List[Any],
        fixer: LLMFixer,
        auth_config: AuthConfig,
        fix_params: Dict[str, Any]
) -> None:
    """Handle a generated fix."""
    _display_fix_preview(fix, issues)

    if fix_params['apply']:
        result = fixer.apply_fixes_with_validation(
            fixes=[fix],
            issues=issues,
            project_path=Path(str(auth_config.project_path)),
            create_backup=True,
            dry_run=fix_params['dry_run'],
            use_validator=True,
            validator_provider=fixer.provider,
            validator_model=fixer.model,
            validator_api_key=fixer.api_key,
        )
        _display_fix_results(result)
    else:
        console.print("[dim]Skipped[/dim]")


def _should_continue_to_next_file(current_idx: int, total_files: int) -> bool:
    """Check if should continue to next file."""
    if current_idx >= total_files:
        return False

    if not smart_confirm("Continue to next file?", default=True):
        console.print("[yellow]Stopped processing remaining files[/yellow]")
        return False

    return True


def _process_security_issues(
        auth_config: AuthConfig,
        llm_config: LLMConfig,
        branch: Optional[str],
        pull_request: Optional[str],
        fix_params: Dict[str, Any]
) -> None:
    """Process security issues - Refactored."""
    services = _initialize_fix_services(auth_config, llm_config)

    issues_by_file = _fetch_security_issues(
        services['analyzer'],
        auth_config,
        branch,
        pull_request,
        fix_params
    )

    if not issues_by_file:
        console.print("[yellow]No fixable security issues found[/yellow]")
        return

    console.print(f"\n[green]✓ Found {len(issues_by_file)} security issues[/green]\n")

    _process_security_files(
        issues_by_file,
        services,
        auth_config,
        fix_params
    )

def _fetch_security_issues(
    analyzer: SonarCloudAnalyzer,
    auth_config: AuthConfig,
    branch: Optional[str],
    pull_request: Optional[str],
    fix_params: Dict[str, Any]
) -> Dict[str, List[Any]]:
    """Fetch security issues from SonarCloud."""
    with show_progress("Fetching security issues....") as (progress, task):
        return analyzer.get_fixable_security_issues(
            project_key=auth_config.project,
            branch=branch or "",
            pull_request=int(pull_request) if pull_request else 0,
            max_issues=fix_params['max_fixes'],
        )


def _process_security_files(
        issues_by_file: Dict[str, List[Any]],
        services: Dict[str, Any],
        auth_config: AuthConfig,
        fix_params: Dict[str, Any]
) -> None:
    """Process each file with security issues."""
    total_files = len(issues_by_file)

    for idx, (file_path, file_issues) in enumerate(issues_by_file.items(), 1):
        console.print(f"\n[blue]Processing ({idx}/{total_files}): {file_path}[/blue]")

        fix = _generate_security_fix(
            file_issues,
            services,
            auth_config
        )

        if fix:
            _handle_security_fix(fix, file_issues, services['fixer'], auth_config, fix_params)

        if not _should_continue_to_next_file(idx, total_files):
            break


def _handle_security_fix(
        fix: FixSuggestion,
        file_issues: List[Any],
        fixer: LLMFixer,
        auth_config: AuthConfig,
        fix_params: Dict[str, Any]
) -> None:
    """Handle a generated security fix."""
    _display_fix_preview(fix, file_issues)

    if fix_params['apply']:
        result = fixer.apply_fixes_with_validation(
            fixes=[fix],
            issues=file_issues,
            project_path=Path(str(auth_config.project_path)),
            create_backup=fix_params['create_backup'],
            dry_run=fix_params['dry_run'],
            use_validator=True,
            validator_provider=fixer.provider,
            validator_model=fixer.model,
            validator_api_key=fixer.api_key,
        )
        _display_fix_results(result)
    else:
        console.print("[dim]Skipped[/dim]")

def _generate_security_fix(
        file_issues: List[Any],
        services: Dict[str, Any],
        auth_config: AuthConfig
) -> Optional[FixSuggestion]:
    """Generate security fix for a file."""
    with show_progress("Generating fixes....", total=len(file_issues)) as (progress, task):
        rule_info_list = _collect_rule_information(file_issues, services['ruler'])

        return services['fixer'].generate_fix_by_file(
            file_issues,
            Path(str(auth_config.project_path)),
            rule_info_list
        )



def _display_fix_preview(fix: FixSuggestion, issues: Sequence[Any]) -> None:
    """Display a preview of the fix."""
    console.print("\n[bold]Fix Preview:[/bold]")
    console.print(f"File: [cyan]{fix.file_path}[/cyan]")
    console.print(f"Confidence: [cyan]{fix.confidence:.2f}[/cyan]")
    console.print(f"Issues fixed: [cyan]{len(issues)}[/cyan]")

    console.print("\n[bold]Code Changes:[/bold]")
    console.print(Panel(fix.fixed_code[:500] + ("..." if len(fix.fixed_code) > 500 else ""),
                       title="Fixed Code Preview", border_style="green"))


def _display_analysis_results(result: AnalysisResult, limit: Optional[int]) -> None:
    """Display analysis results."""
    console.print(Panel.fit("[bold]Analysis Results[/bold]"))
    console.print(f"Project: [cyan]{result.project_key}[/cyan]")
    console.print(f"Total Issues: [cyan]{result.total_issues}[/cyan]")

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
                issue.message[:47] + "..." if len(issue.message) > 50 else issue.message
            )

        console.print("\n")
        console.print(table)


def _display_fix_results(result: FixResult) -> None:
    """Display fix results."""
    console.print("\n[bold]Results:[/bold]")
    console.print(f"Attempted: {result.total_fixes_attempted}")
    console.print(f"Successful: [green]{len(result.successful_fixes)}[/green]")
    console.print(f"Failed: [red]{len(result.failed_fixes)}[/red]")
    console.print(f"Success Rate: {result.success_rate:.1%}")

    if result.backup_created:
        console.print(f"\n[blue]Backup: {result.backup_path}[/blue]")


if __name__ == "__main__":
    main()