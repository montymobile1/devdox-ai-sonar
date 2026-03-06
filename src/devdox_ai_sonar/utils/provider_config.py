from typing import Optional, List, Dict, Any, Tuple
from rich.prompt import Confirm, Prompt
from rich.console import Console
from devdox_ai_sonar.utils.ui_prompts import select_from_list
import rich_click as click
from devdox_ai_sonar.models.llm import ProviderType, ProviderValidator
from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.utils.validator import InputValidator
from devdox_ai_sonar.utils.exceptions import ValidationError

console = Console()
CONFIG_PROVIDERS = "llm.providers"
CONFIG_DEFAULT_BRANCH = "sonar.default_branch"
CONFIG_DEFAULT_PULL = "sonar.default_pull"
CONFIG_CLONE_TYPE = "sonar.sonar_options.clone_type"


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
    async def select_provider_from_list(
        providers: List[str], message: str
    ) -> Optional[str]:
        """Select a provider using terminal menu."""
        return await select_from_list(providers, message)

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
        self,
        config_manager: ConfigManager,
        ui: ProviderConfigUI,
        validator: ProviderValidator,
    ):
        self.config_manager = config_manager
        self.ui = ui
        self.validator = validator

    async def get_available_providers(self) -> List[str]:
        """Get list of providers not yet configured."""
        existing_providers_value = await self.config_manager.get_value(CONFIG_PROVIDERS)

        existing_providers: List[Dict[str, Any]] = (
            existing_providers_value if existing_providers_value else []
        )
        existing_names = {p["name"] for p in existing_providers}
        return [p for p in ProviderType.choices() if p not in existing_names]

    async def get_default_provider(self) -> Optional[str]:
        """Get default provider name."""
        value = await self.config_manager.get_value("llm.default_provider")
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"Expected string for llm.default_provider, got {type(value).__name__}"
            )
        return value

    async def get_existing_providers(self) -> List[str]:
        """Get list of already configured providers."""
        existing_providers_value = await self.config_manager.get_value(CONFIG_PROVIDERS)

        existing_providers: List[Dict[str, Any]] = (
            existing_providers_value if existing_providers_value else []
        )

        return [p["name"] for p in existing_providers]

    async def configure_new_provider(
        self, provider_name: str
    ) -> Optional[Dict[str, Any]]:
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
        default_model = await self.ui.select_provider_from_list(
            result.models, provider_name
        )
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

    async def update_existing_provider(self, provider_name: str) -> bool:
        """Update an existing provider's configuration."""
        # Find provider
        providers_value = await self.config_manager.get_value(CONFIG_PROVIDERS)

        providers: List[Dict[str, Any]] = providers_value if providers_value else []
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
        await self._handle_model_update(ctx)

        # Handle default provider status
        set_as_default = self.ui.confirm_default("Make this the default provider?")

        if set_as_default:
            ctx.updates["default_model"] = ctx.current_model

        # Apply updates if any
        return await self._apply_provider_updates(ctx, set_as_default)

    async def _handle_model_update(self, ctx: ProviderUpdateContext) -> None:
        """Handle model selection update flow."""
        # Ask if user wants to keep current model
        if Confirm.ask(f"Keep current model '{ctx.current_model}'?", default=True):
            return

        # Need to select new model
        models = self._get_available_models(ctx)
        if not models:
            return

        new_model = await self.ui.select_provider_from_list(models, ctx.provider_name)
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
            # Ensure models is a list of strings
            models = validation_result.models
            if not isinstance(models, list):
                return []

            # Validate all items are strings
            if not all(isinstance(m, str) for m in models):
                return [str(m) for m in models if m]

            return models
        return []

    async def _apply_provider_updates(
        self, ctx: ProviderUpdateContext, set_as_default: bool
    ) -> bool:
        """Apply updates to provider configuration."""
        if not ctx.updates:
            return False

        await self.config_manager.update_provider(
            provider_name=ctx.provider_name,
            updates=ctx.updates,
            set_as_default=set_as_default,
        )
        return True

    def _handle_api_key_update(self, ctx: ProviderUpdateContext) -> bool:
        """Handle API key update flow. Returns False if update should be cancelled."""
        if not Confirm.ask(
            f"Change API key for {ctx.provider_name.upper()}?", default=False
        ):
            # Keep existing API key, use existing models
            models_value = ctx.provider.get("models", [])

            ctx.available_models = (
                models_value if isinstance(models_value, list) else []
            )
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

    def _validate_provider_api_key(
        self, provider_name: str, api_key: str
    ) -> Optional[Any]:
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

    async def branch_or_pr(self) -> Tuple[str, int]:
        """Prompt user for branch or PR number"""
        clone_type = await self.config_manager.get_value(CONFIG_CLONE_TYPE)
        default_pull = await self.config_manager.get_value(CONFIG_DEFAULT_PULL)
        default_branch = await self.config_manager.get_value(CONFIG_DEFAULT_BRANCH)
        if clone_type == "pr":
            default_branch = ""
        else:
            default_pull = 0
        return default_branch, default_pull

    async def get_params(self) -> Any:
        return await self.config_manager.get_value("sonar.configuration")

    async def branch_or_pr_prompt(self) -> Tuple[str, int]:
        """Prompt user for branch or PR number"""
        console.print("\n[bold]Choose what to analyze:[/bold]")

        if Confirm.ask(
            "Do you want to analyze a [cyan]pull request[/cyan]?", default=False
        ):
            return await self._handle_pull_request_selection()

        return await self._handle_branch_selection()

    async def _handle_branch_selection(self) -> Tuple[str, int]:
        """Handle branch selection workflow."""
        default_branch = (
            await self.config_manager.get_value(CONFIG_DEFAULT_BRANCH) or "main"
        )
        branch_input = Prompt.ask("Branch name", default=default_branch)

        try:
            branch = InputValidator.validate_branch_name(branch_input)
            console.print(f"[green]✓[/green] Analyzing branch: [cyan]{branch}[/cyan]")

            await self._save_as_default_if_changed(
                config_key=CONFIG_DEFAULT_BRANCH,
                new_value=branch,
                current_default=default_branch,
                display_name=f"branch '{branch}'",
            )

            await self._save_as_default_if_changed(
                config_key=CONFIG_CLONE_TYPE,
                new_value="branch",
                current_default="pr",
                display_name="",
                confirm=False,
            )

            return branch, 0

        except ValidationError as e:
            console.print(f"\n[red]❌ {e.message}[/red]")
            raise click.Abort()

    async def _handle_pull_request_selection(self) -> Tuple[str, int]:
        """Handle pull request selection workflow."""
        default_pull = await self.config_manager.get_value(CONFIG_DEFAULT_PULL)
        pr_default = str(default_pull) if default_pull else "0"
        pr_input = Prompt.ask("Pull Request number", default=pr_default)

        try:
            pull_request = InputValidator.validate_pull_request_number(pr_input)
            console.print(
                f"[green]✓[/green] Analyzing PR: [cyan]#{pull_request}[/cyan]"
            )

            await self._save_as_default_if_changed(
                config_key=CONFIG_DEFAULT_PULL,
                new_value=pull_request,
                current_default=default_pull,
                display_name=f"PR #{pull_request}",
            )

            await self._save_as_default_if_changed(
                config_key=CONFIG_CLONE_TYPE,
                new_value="pr",
                current_default="branch",
                display_name="",
                confirm=False,
            )

            return "", pull_request

        except ValidationError as e:
            console.print(f"\n[red]❌ {e.message}[/red]")
            raise click.Abort()

    async def _save_as_default_if_changed(
        self,
        config_key: str,
        new_value: Any,
        current_default: Any,
        display_name: str,
        confirm: bool = True,
    ) -> None:
        """Save new value as default if it differs from current default."""
        if str(new_value) == str(current_default):
            return
        if confirm and not Confirm.ask(
            "Save this as default for future runs?", default=False
        ):
            return

        await self.config_manager.set_value(config_key, new_value)
        self.config_manager.save_config()
        if confirm:
            console.print(f"[green]✓[/green] Saved {display_name} as default")
