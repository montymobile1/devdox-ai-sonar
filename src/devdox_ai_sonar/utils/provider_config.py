from typing import Optional, List, Dict, Any, Tuple
from rich.prompt import Confirm, Prompt
from rich.console import Console
from simple_term_menu import TerminalMenu
import rich_click as click
from devdox_ai_sonar.models.llm import ProviderType, ProviderValidator
from devdox_ai_sonar.utils.validator import InputValidator
from devdox_ai_sonar.utils.exceptions import ValidationError

console = Console()
CONFIG_PROVIDERS="llm.providers"


class ProviderUpdateContext:
    """Context object for provider update operations."""

    def __init__(self, provider: Dict[str, Any], provider_name: str):
        self.provider = provider
        self.provider_name = provider_name
        self.updates: Dict[str, Any] = {}
        self.current_model = provider.get("default_model", "")
        self.available_models: List[str] = []


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
        existing_providers = self.config_manager.get_value(CONFIG_PROVIDERS) or []
        existing_names = {p["name"] for p in existing_providers}
        return [p for p in ProviderType.choices() if p not in existing_names]

    def get_default_provider(self) -> Optional[str]:
        """Get default provider name."""
        return self.config_manager.get_value("llm.default_provider")

    def get_existing_providers(self) -> List[str]:
        """Get list of already configured providers."""
        existing_providers = self.config_manager.get_value(CONFIG_PROVIDERS) or []
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
        # Find provider
        providers = self.config_manager.get_value(CONFIG_PROVIDERS) or []
        provider = next((p for p in providers if p["name"] == provider_name), None)

        if not provider:
            console.print(f"[red]❌ Provider {provider_name} not found[/red]")
            return False

        # Initialize update context
        ctx = ProviderUpdateContext(provider, provider_name)

        # Handle API key update
        if not self._handle_api_key_update(ctx):
            return False

        # Handle model selection
        self._handle_model_update(ctx)

        # Handle default provider status
        set_as_default = self.ui.confirm_default("Make this the default provider?")

        if set_as_default:
            ctx.updates["default_model"] = ctx.current_model

        # Apply updates if any
        return self._apply_provider_updates(ctx, set_as_default)

    def _handle_model_update(self, ctx: ProviderUpdateContext) -> None:
        """Handle model selection update flow."""
        # Ask if user wants to keep current model
        if Confirm.ask(f"Keep current model '{ctx.current_model}'?", default=True):
            return

        # Need to select new model
        models = self._get_available_models(ctx)
        if not models:
            return

        new_model = self.ui.select_provider_from_list(models, ctx.provider_name)
        if new_model:
            ctx.updates["default_model"] = new_model
            ctx.current_model = new_model

    def _get_available_models(self, ctx: ProviderUpdateContext) -> List[str]:
        """Get available models for provider, fetching if necessary."""
        if ctx.available_models:
            return ctx.available_models

        # Need to fetch models using existing API key
        api_key = ctx.provider.get("api_key", "")
        if not api_key:
            return []

        validation_result = self._validate_provider_api_key(ctx.provider_name, api_key)
        if validation_result and validation_result.success:
            return validation_result.models

        return []

    def _apply_provider_updates(
            self, ctx: ProviderUpdateContext, set_as_default: bool
    ) -> bool:
        """Apply updates to provider configuration."""
        if not ctx.updates:
            return False

        self.config_manager.update_provider(
            provider_name=ctx.provider_name,
            updates=ctx.updates,
            set_as_default=set_as_default,
        )
        return True

    def _handle_api_key_update(self, ctx: ProviderUpdateContext) -> bool:
        """Handle API key update flow. Returns False if update should be cancelled."""
        if not Confirm.ask(f"Change API key for {ctx.provider_name.upper()}?", default=False):
            # Keep existing API key, use existing models
            ctx.available_models = ctx.provider.get("models", [])
            return True

        # Get and validate new API key
        api_key = self.ui.prompt_for_api_key(ctx.provider_name)
        if not api_key:
            return False

        validation_result = self._validate_provider_api_key(ctx.provider_name, api_key)
        if not validation_result:
            return False

        # Update successful
        ctx.updates["api_key"] = api_key
        ctx.available_models = validation_result.models
        return True

    def _validate_provider_api_key(self, provider_name: str, api_key: str):
        """Validate provider API key and return validation result."""
        try:
            provider_type = ProviderType(provider_name)
        except ValueError:
            console.print(f"[red]❌ Unknown provider: {provider_name}[/red]")
            return None

        console.print("[cyan]Validating new API key...[/cyan]")
        result = self.validator.validate(provider_type, api_key)

        if not result.success:
            console.print(f"[red]❌ {result.error_message}[/red]")
            return None

        return result


    def branch_or_pr_prompt(self) -> Tuple[str, int]:
        """Prompt user for branch or PR number"""
        console.print("\n[bold]Choose what to analyze:[/bold]")

        if Confirm.ask(
            "Do you want to analyze a [cyan]pull request[/cyan]?", default=False
        ):
            return self._handle_pull_request_selection()

        return self._handle_branch_selection()

    def _handle_branch_selection(self) -> Tuple[str, int]:
        """Handle branch selection workflow."""
        default_branch = self.config_manager.get_value("sonar.default_branch") or "main"
        branch_input = Prompt.ask("Branch name", default=default_branch)

        try:
            branch = InputValidator.validate_branch_name(branch_input)
            console.print(f"[green]✓[/green] Analyzing branch: [cyan]{branch}[/cyan]")

            self._save_as_default_if_changed(
                config_key="sonar.default_branch",
                new_value=branch,
                current_default=default_branch,
                display_name=f"branch '{branch}'",
            )

            return branch, 0

        except ValidationError as e:
            console.print(f"\n[red]❌ {e.message}[/red]")
            raise click.Abort()

    def _handle_pull_request_selection(self) -> Tuple[str, int]:
        """Handle pull request selection workflow."""
        default_pull = self.config_manager.get_value("sonar.default_pull")
        pr_default = str(default_pull) if default_pull else "0"
        pr_input = Prompt.ask("Pull Request number", default=pr_default)

        try:
            pull_request = InputValidator.validate_pull_request_number(pr_input)
            console.print(f"[green]✓[/green] Analyzing PR: [cyan]#{pull_request}[/cyan]")

            self._save_as_default_if_changed(
                config_key="sonar.default_pull",
                new_value=pull_request,
                current_default=default_pull,
                display_name=f"PR #{pull_request}",
            )

            return "", pull_request

        except ValidationError as e:
            console.print(f"\n[red]❌ {e.message}[/red]")
            raise click.Abort()

    def _save_as_default_if_changed(
            self,
            config_key: str,
            new_value: Any,
            current_default: Any,
            display_name: str,
    ) -> None:
        """Save new value as default if it differs from current default."""
        if str(new_value) == str(current_default):
            return

        if not Confirm.ask(
                "Save this as default for future runs?", default=False
        ):
            return

        self.config_manager.set_value(config_key, new_value)
        self.config_manager.save_config()
        console.print(f"[green]✓[/green] Saved {display_name} as default")
