from typing import Optional, Dict
from rich.console import Console
from rich.prompt import Confirm, Prompt

from devdox_ai_sonar.models.sonar import SonarCloudConfig

console = Console()


class SonarCloudConfigUI:
    """Handles SonarCloud configuration user interaction."""

    @staticmethod
    def display_welcome() -> None:
        """Display welcome message and wizard introduction."""
        console.print("\n" + "=" * 60)
        console.print(
            "[magenta bold]🚀 DEVDOX SONAR CONFIGURATION WIZARD[/magenta bold]"
        )
        console.print("=" * 60)
        console.print("\nWelcome to DevDox Sonar!")
        console.print("Let's set up your SonarCloud integration and LLM providers.\n")

    @staticmethod
    def display_step_header(step: int, total: int, title: str) -> None:
        """Display step header in the wizard."""
        console.print("=" * 60)
        console.print(f"[green bold]STEP {step}/{total}: {title}[/green bold]")
        console.print("=" * 60 + "\n")

    @staticmethod
    def prompt_with_default(
        prompt_text: str,
        field_name: str,
        default_value: Optional[str] = None,
        required: bool = True,
    ) -> Optional[str]:
        """
        Prompt user for input with optional default value.

        Args:
            prompt_text: The prompt message to display
            default_value: Optional default value to use
            required: Whether the field is required

        Returns:
            User input or None if cancelled
        """
        try:
            if default_value:
                console.print(
                    f"[cyan]Current value of {field_name}: {default_value}  [/cyan]"
                )
                if Confirm.ask("Use current value?", default=True):
                    return default_value

            while True:
                value = Prompt.ask(prompt_text).strip()

                if not value and required:
                    console.print("[red]❌ This field is required[/red]")
                    continue

                if not value and not required:
                    return None

                return value

        except KeyboardInterrupt:
            return None

    @staticmethod
    def configure_sonarcloud(
        existing_config: Dict[str, Optional[str]],
    ) -> Optional[SonarCloudConfig]:
        """
        Configure SonarCloud settings through interactive prompts.

        Args:
            existing_config: Existing configuration to use as defaults (empty dict if none)

        Returns:
            SonarCloudConfig object or None if cancelled
        """

        token = SonarCloudConfigUI.prompt_with_default(
            "SonarCloud authentication token",
            "token",
            existing_config.get("token"),
            required=True,
        )
        if not token:
            return None

        organization = SonarCloudConfigUI.prompt_with_default(
            "SonarCloud organization key",
            "organization",
            existing_config.get("organization"),
            required=True,
        )
        if not organization:
            return None

        project = SonarCloudConfigUI.prompt_with_default(
            "SonarCloud project key",
            "project key",
            existing_config.get("project"),
            required=True,
        )
        if not project:
            return None

        project_path = SonarCloudConfigUI.prompt_with_default(
            "SonarCloud project path",
            "project path",
            existing_config.get("project_path"),
            required=True,
        )
        if not project_path:
            return None

        git_url = SonarCloudConfigUI.prompt_with_default(
            "Git URL for the project",
            "git_url",
            existing_config.get("git_url"),
            required=True,
        )
        if not git_url:
            return None

        config = SonarCloudConfig(
            token=token,
            organization=organization,
            project=project,
            project_path=project_path,
            git_url=git_url,
        )

        # Validate configuration
        is_valid, error_message = config.validate()

        if not is_valid:
            console.print(f"[red]❌ Invalid configuration: {error_message}[/red]")
            return None

        return config
