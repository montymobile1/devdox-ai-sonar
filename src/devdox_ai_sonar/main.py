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
# setup_theme()


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
    """CLI for config management"""
    try:
        manager = ConfigManager(config_path=settings.config_file_path)
        manager.create_default_config()
        manager.load_config()
        providers = manager.get_value("llm.providers")
        if providers and len(providers) > 0:
            click.secho("\n⚠️  Configuration already exists", fg="yellow", bold=True)
            if not Confirm.ask("Do you want to reconfigure?", default=False):
                click.echo("Configuration unchanged.")
                return
                # Clear existing providers
        sonar_ui = SonarCloudConfigUI()
        validator = ProviderValidator()
        config_service = ConfigService(sonar_path=Path(settings.auth_file_path))
        ui = ProviderConfigUI()
        provider_manager = ProviderConfigManager(manager, ui, validator)

        sonar_ui.display_welcome()

        sonar_ui.display_step_header(1, 2, "SonarCloud Configuration")

        auth_config = config_service.load_auth_config()

        sonar_config = sonar_ui.configure_sonarcloud(auth_config)


        if not sonar_config:
            console.print("[red]❌ SonarCloud configuration cancelled[/red]")
            raise click.Abort()

        save_config= config_service.save_config(
            token=sonar_config.token,
            organization=sonar_config.organization,
            project=sonar_config.project,
            project_path=str(sonar_config.project_path),
        )
        if not save_config:
            click.secho("❌ SonarCloud configuration save failed", fg="red", bold=True)
            click.Abort()
        click.secho("\n✓ SonarCloud configuration saved", fg="green", bold=True)
        if providers and len(providers) > 0:
            return
        providers = provider_manager.get_available_providers()

        if not providers:
            console.print(
                "\n[red]❌ No providers configured. Configuration incomplete.[/red]"
            )
            raise click.Abort()
        while providers:
            console.print(f"[cyan]Available providers: {', '.join(providers)}[/cyan]\n")

            provider_name = ui.select_provider_from_list(
                providers, "Select a provider to configure"
            )

            if not provider_name:
                break

            result = provider_manager.configure_new_provider(provider_name)
            if result:
                manager.add_provider(
                    result["config"], set_as_default=result["set_as_default"]
                )
                providers.remove(provider_name)
                console.print(
                    f"\n[green]✓ {provider_name.upper()} configured successfully[/green]\n"
                )
            else:
                console.print(
                    f"[yellow]⚠ Configuration of {provider_name} failed or was cancelled[/yellow]\n"
                )

            if not providers:
                console.print("[green]✅ All providers have been configured[/green]")
                break

            if not Confirm.ask("Add another provider?", default=False):
                break

        # Save configuration
        manager.save_config(create_backup=False)

    except click.Abort:
        click.secho("\nConfiguration cancelled", fg="red", bold=True)
        raise

    except Exception as e:
        click.secho(f"\n❌ Error during configuration: {str(e)}", fg="red", bold=True)
        raise click.ClickException(str(e))


@main.command(name="update_provider", help="Update config Provider")
@click.pass_context
def update_provider(ctx: click.Context):
    """CLI command for managing provider configuration."""
    try:
        # Initialize components
        manager = ConfigManager(config_path=settings.config_file_path)
        # manager.create_default_config()
        manager.load_config()

        ui = ProviderConfigUI()
        validator = ProviderValidator()
        provider_manager = ProviderConfigManager(manager, ui, validator)

        existing_providers = provider_manager.get_existing_providers()

        if not existing_providers:
            console.print(
                "\n[red]❌ No providers configured. Please add at least one provider first.[/red]"
            )
            raise click.Abort()

        # Determine operation: update existing or add new
        if Confirm.ask("\nAdd a new provider?", default=False):
            # Add new provider flow
            available_providers = provider_manager.get_available_providers()

            if not available_providers:
                console.print(
                    "[yellow]⚠ All supported providers are already configured[/yellow]"
                )
                raise click.Abort()

            console.print("\n" + "=" * 60)
            console.print("[magenta bold]🚀 ADD NEW PROVIDER[/magenta bold]")
            console.print("=" * 60 + "\n")

            while available_providers:
                console.print(
                    f"[cyan]Available providers: {', '.join(available_providers)}[/cyan]\n"
                )

                provider_name = ui.select_provider_from_list(
                    available_providers, "Select a provider to configure"
                )

                if not provider_name:
                    break

                result = provider_manager.configure_new_provider(provider_name)

                if result:
                    manager.add_provider(
                        result["config"], set_as_default=result["set_as_default"]
                    )
                    available_providers.remove(provider_name)
                    console.print(
                        f"\n[green]✓ {provider_name.upper()} configured successfully[/green]\n"
                    )
                else:
                    console.print(
                        f"[yellow]⚠ Configuration of {provider_name} failed or was cancelled[/yellow]\n"
                    )

                if not available_providers:
                    console.print(
                        "[green]✅ All providers have been configured[/green]"
                    )
                    break

                if not Confirm.ask("Add another provider?", default=False):
                    break

            manager.save_config(create_backup=False)

        else:
            # Update existing provider flow
            console.print("\n" + "=" * 60)
            console.print("[magenta bold]🔧 UPDATE EXISTING PROVIDER[/magenta bold]")
            console.print("=" * 60 + "\n")

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
                raise click.Abort()

            chosen_provider = answers["provider"]

            if provider_manager.update_existing_provider(chosen_provider):
                manager.save_config(create_backup=False)
                console.print(
                    f"\n[green]✓ {chosen_provider.upper()} updated successfully[/green]\n"
                )
            else:
                console.print("\n[yellow]⚠ Update cancelled or failed[/yellow]\n")

        # Final success message
        console.print("=" * 60)
        console.print("[green bold]✨ CONFIGURATION COMPLETE![/green bold]")
        console.print("=" * 60)
        console.print(f"\nConfiguration saved to: {settings.config_file_path}\n")

    except click.Abort:
        console.print("\n[red]Configuration cancelled[/red]")
        raise
    except Exception as e:
        console.print(f"\n[red]❌ Error during configuration: {str(e)}[/red]")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
