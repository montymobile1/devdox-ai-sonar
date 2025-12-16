from typing import Optional, List, Dict, Any
from rich.prompt import Confirm, Prompt
from rich.console import Console
from simple_term_menu import TerminalMenu
import rich_click as click
from devdox_ai_sonar.models.llm import ProviderType, ProviderValidator
from devdox_ai_sonar.utils.validator import InputValidator
from devdox_ai_sonar.utils.exceptions import ValidationError

console = Console()


class ProviderConfigUI:
    """Handles user interaction for provider configuration."""

    @staticmethod
    def prompt_for_api_key(provider_name: str) -> Optional[str]:
        """Prompt user for API key with validation."""
        try:
            api_key = Prompt.ask(f"Enter your {provider_name.upper()} API key").strip()
            if not api_key:
                console.print("[red]❌ API key cannot be empty[/red]")
                return None
            return api_key
        except KeyboardInterrupt:
            return None


    @staticmethod
    def select_provider_from_list(providers: List[str], message: str) -> Optional[str]:
        """Select a provider using terminal menu."""
        if not providers:
            return None
        menu = TerminalMenu(
            providers,
            search_key="/",
            search_case_sensitive=False,
            show_search_hint=True,
            show_search_hint_text='Press "/" to search',
            title=message,
            menu_cursor="➤ ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green", "bold"),
        )

        index = menu.show()
        return providers[index] if index is not None else None

    @staticmethod
    def confirm_default(message: str = "Make this the default provider?") -> bool:
        """Ask user to confirm setting as default."""
        try:
            return Confirm.ask(message, default=False)
        except KeyboardInterrupt:
            return False


class ProviderConfigManager:
    """Manages provider configuration operations."""

    def __init__(
        self, config_manager, ui: ProviderConfigUI, validator: ProviderValidator
    ):
        self.config_manager = config_manager
        self.ui = ui
        self.validator = validator

    def get_available_providers(self) -> List[str]:
        """Get list of providers not yet configured."""
        existing_providers = self.config_manager.get_value("llm.providers") or []
        existing_names = {p["name"] for p in existing_providers}
        return [p for p in ProviderType.choices() if p not in existing_names]

    def get_default_provider(self) -> Optional[str]:
        """Get default provider name."""
        return self.config_manager.get_value("llm.default_provider")

    def get_existing_providers(self) -> List[str]:
        """Get list of already configured providers."""
        existing_providers = self.config_manager.get_value("llm.providers") or []
        return [p["name"] for p in existing_providers]

    def configure_new_provider(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Configure a new provider with API key and model selection."""
        try:
            provider_type = ProviderType(provider_name)
        except ValueError:
            console.print(f"[red]❌ Unknown provider: {provider_name}[/red]")
            return None

        # Get API key
        api_key = self.ui.prompt_for_api_key(provider_name)
        if not api_key:
            return None

        # Validate API key
        console.print(f"[cyan]Validating {provider_name.upper()} API key...[/cyan]")
        result = self.validator.validate(provider_type, api_key)

        if not result.success:
            console.print(f"[red]❌ {result.error_message}[/red]")
            return None

        console.print(f"[green]✓ Found {len(result.models)} models[/green]")

        # Select default model
        default_model = self.ui.select_provider_from_list(result.models, provider_name)
        if not default_model:
            console.print("[yellow]⚠ Model selection cancelled[/yellow]")
            return None

        # Ask if should be default provider
        set_as_default = self.ui.confirm_default()

        provider_config = {
            "name": provider_name,
            "api_key": api_key,
            "models": result.models,
            "default_model": default_model,
        }

        return {"config": provider_config, "set_as_default": set_as_default}

    def update_existing_provider(self, provider_name: str) -> bool:
        """Update an existing provider's configuration."""
        providers = self.config_manager.get_value("llm.providers") or []
        provider = next((p for p in providers if p["name"] == provider_name), None)
        if not provider:
            console.print(f"[red]❌ Provider {provider_name} not found[/red]")
            return False

        updates = {}
        current_model = provider.get("default_model", "")
        # Update API key if requested
        if Confirm.ask(f"Change API key for {provider_name.upper()}?", default=False):
            api_key = self.ui.prompt_for_api_key(provider_name)
            if not api_key:
                return False

            # Validate new key
            try:
                provider_type = ProviderType(provider_name)
            except ValueError:
                console.print(f"[red]❌ Unknown provider: {provider_name}[/red]")
                return False

            console.print("[cyan]Validating new API key...[/cyan]")
            result = self.validator.validate(provider_type, api_key)

            if not result.success:
                console.print(f"[red]❌ {result.error_message}[/red]")
                return False

            updates["api_key"] = api_key
            available_models = result.models
        else:
            # Use existing API key to fetch models if needed
            available_models = provider.get("models", [])

        # Update default model if requested

        if not Confirm.ask(f"Keep current model '{current_model}'?", default=True):
            if not available_models:
                # Need to fetch models with existing key
                try:
                    provider_type = ProviderType(provider_name)
                    result = self.validator.validate(
                        provider_type, provider.get("api_key", "")
                    )
                    if result.success:
                        available_models = result.models
                except ValueError:
                    pass

            if available_models:
                new_model = self.ui.select_provider_from_list(available_models, provider_name)
                if new_model:
                    updates["default_model"] = new_model
                    current_model = new_model

        # Update default provider status
        set_as_default = self.ui.confirm_default("Make this the default provider?")

        if set_as_default:
            updates["default_model"] = current_model
        if updates:
            self.config_manager.update_provider(
                provider_name=provider_name,
                updates=updates,
                set_as_default=set_as_default,
            )
            return True

        return False


    def branch_or_pr_prompt(self) -> str:
        """Prompt user for branch or PR number"""
        default_branch = self.config_manager.get_value("sonar.default_branch") or "main"
        default_pull = self.config_manager.get_value("sonar.default_pull")

        # Initialize variables
        branch = ""
        pull_request = 0

        # No command line options, ask user interactively
        console.print("\n[bold]Choose what to analyze:[/bold]")

        if Confirm.ask("Do you want to analyze a [cyan]pull request[/cyan]?", default=False):
            # User wants to use PR
            pr_default = str(default_pull) if default_pull else "0"
            pr_input = Prompt.ask("Pull Request number", default=pr_default)

            try:
                    pull_request = InputValidator.validate_pull_request_number(pr_input)
                    branch = ""
                    console.print(f"[green]✓[/green] Analyzing PR: [cyan]#{pull_request}[/cyan]")

                    # Ask to save as default
                    if str(pull_request) != str(default_pull):
                        if Confirm.ask("Save this PR as default for future runs?", default=False):
                            self.config_manager.set_value("sonar.default_pull", pull_request)
                            self.config_manager.save_config()
                            console.print(f"[green]✓[/green] Saved PR #{pull_request} as default")

            except ValidationError as e:
                    console.print(f"\n[red]❌ {e.message}[/red]")
                    raise click.Abort()

        else:
                # User wants to use branch
                branch_input = Prompt.ask("Branch name", default=default_branch)

                try:
                    branch = InputValidator.validate_branch_name(branch_input)
                    pull_request = 0
                    console.print(f"[green]✓[/green] Analyzing branch: [cyan]{branch}[/cyan]")

                    # Ask to save as default
                    if branch != default_branch:
                        if Confirm.ask("Save this branch as default for future runs?", default=False):
                            self.config_manager.set_value("sonar.default_branch", branch)
                            self.config_manager.save_config()
                            console.print(f"[green]✓[/green] Saved branch '{branch}' as default")

                except ValidationError as e:
                    console.print(f"\n[red]❌ {e.message}[/red]")
                    raise click.Abort()

        return branch, pull_request
