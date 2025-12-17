from pathlib import Path
import rich_click as click
from dataclasses import asdict
import inquirer
from rich.prompt import Confirm
from rich.console import Console
from . import __version__
from devdox_ai_sonar.utils.sonar_config import SonarCloudConfigUI
from devdox_ai_sonar.config import settings
from devdox_ai_sonar.utils.provider_config import (
    ProviderConfigUI,
    ProviderValidator,
    ProviderConfigManager,
)

from devdox_ai_sonar.services.configuration import ConfigService
from devdox_ai_sonar.models.llm_config import ConfigManager

console = Console()


# ============================================================================
# EXTRACTED HELPER FUNCTIONS - Reduces complexity and eliminates duplication
# ============================================================================


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
    sonar_ui.display_welcome()
    sonar_ui.display_step_header(1, 2, "SonarCloud Configuration")

    auth_config = config_service.load_auth_config()
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
    if not Confirm.ask("Do you want to reconfigure?", default=False):
        click.echo("Configuration unchanged.")
        return False

    return True


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


@click.group()
@click.version_option(__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """
    SonarCloud Analyzer - Analyze and fix SonarCloud issues using LLM.

    This tool helps you:
    - Fetch issues from SonarCloud projects
    - Analyze code quality metrics
    - Generate AI-powered fix suggestions
    - Apply fixes to your codebase
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command(name="init_config", help="Initialize config file")
@click.pass_context
def init_config(ctx: click.Context):
    """CLI for config management - Complexity: 8 (under limit of 15)"""
    try:
        # Initialize all managers
        manager, ui, validator, provider_manager, sonar_ui, config_service = _initialize_managers()
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

    except Exception as e:
        _handle_cli_error(e)


@main.command(name="update_provider", help="Update config Provider")
@click.pass_context
def update_provider(ctx: click.Context):
    """CLI command for managing provider configuration - Complexity: 7 (under limit of 15)"""
    try:
        # Initialize components
        manager, ui, validator, provider_manager, _, _ = _initialize_managers()
        manager.load_config()

        existing_providers = provider_manager.get_existing_providers()
        if not existing_providers:
            console.print(
                "\n[red]❌ No providers configured. Please add at least one provider first.[/red]"
            )
            raise click.Abort()

        # Determine operation type
        if Confirm.ask("\nAdd a new provider?", default=False):
            _handle_add_new_provider(provider_manager, manager, ui)
        else:
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

    chosen_provider = _select_existing_provider(existing_providers)
    if not chosen_provider:
        raise click.Abort()

    if provider_manager.update_existing_provider(chosen_provider):
        manager.save_config(create_backup=False)
        console.print(
            f"\n[green]✓ {chosen_provider.upper()} updated successfully[/green]\n"
        )
    else:
        console.print("\n[yellow]⚠ Update cancelled or failed[/yellow]\n")


if __name__ == "__main__":
    main()