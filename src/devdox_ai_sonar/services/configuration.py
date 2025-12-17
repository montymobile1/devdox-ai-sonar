# Create src/devdox_ai_sonar/services/config_service.py

from dataclasses import dataclass
from typing import Optional, Tuple,  Dict, Any
from pathlib import Path
import json
from rich.console import Console

from devdox_ai_sonar.models.llm_config import ConfigManager

@dataclass
class AuthConfig:
    token: str
    organization: str
    project: str
    project_path: str

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate all fields are present"""
        if not self.token:
            return False, "Token is required"
        if not self.organization:
            return False, "Organization is required"
        if not self.project:
            return False, "Project key is required"
        if not self.project_path:
            return False, "Project path is required"
        return True, None


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    models: list[str]

console = Console()

class ConfigService:
    """Centralized configuration management"""
    def __init__(self, sonar_path: Path = Path("sonar.toml")):
        self.sonar_path = sonar_path

        self.config: Optional[Dict[str, Any]] = None


    def load_auth_config(self) -> Dict[str, Optional[str]]:
        """
        Load authentication configuration from auth.json file.

        Returns:
            Dictionary with SONAR_TOKEN, SONAR_ORG, and SONAR_PROJ keys.
            Returns empty dict if file doesn't exist or is invalid.
        """
        try:

            if  self.sonar_path .exists():
                with open(self.sonar_path, "r") as f:
                    config = json.load(f)
                    return {
                        "token": config.get("SONAR_TOKEN"),
                        "organization": config.get("SONAR_ORG"),
                        "project": config.get("SONAR_PROJ"),
                        "project_path": config.get("PROJECT_PATH"),
                    }
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[yellow]Warning: Could not read auth.json: {e}[/yellow]")

        return {"token": None, "organization": None, "project": None}

    @staticmethod
    def load_llm_config(manager: ConfigManager) -> Optional[LLMConfig]:
        """Load and validate LLM configuration"""

        providers = manager.get_value("llm.providers")
        if not providers:
            return None

        default_provider = manager.get_value("llm.default_provider")
        default_model = manager.get_value("llm.default_model")
        provider_config = next(
            (p for p in providers if p.get("name") == default_provider),
            None
        )

        if not provider_config:
            return None


        return LLMConfig(
            provider=default_provider,
            model=default_model,
            api_key=provider_config.get("api_key"),
            models = provider_config.get("models",[])
        )

    @staticmethod
    def validate_auth_config(auth_config: dict) -> bool:
        """Validate authentication configuration with detailed error messages."""
        required_fields = ["token", "organization", "project", "project_path"]
        missing_fields = [field for field in required_fields if not auth_config.get(field)]

        if missing_fields:

            return False
        return True


    def save_config(self, token: str, organization: str, project: str, project_path: str):
        """Save authentication configuration"""

        if not validate_token_format(token):
            if (
                    not console.input(
                        "[yellow]Token format seems unusual. Continue? (y/N): [/yellow]"
                    )
                            .lower()
                            .startswith("y")
            ):
                console.print("[red]Cancelled[/red]")
                return False
        success = self.save_complete_config(token, organization, project, project_path)


        return success


    def  save_complete_config(
        self,
        token: str,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        project_path: Optional[str] = None,
        merge: bool = True,
    ) -> bool:
        """
        Save a new token to the auth.json file.

        Args:
            token: SonarCloud authentication token to save
            organization: Optional organization key to save
            project: Optional project key to save
            merge: If True, merge with existing config; if False, overwrite

        Returns:
            True if saved successfully, False otherwise.

        Example:
            # Save only token (merge with existing)
            save_token("squ_new_token")

            # Save all values (merge with existing)
            save_token("squ_new_token", "my-org", "my-project")

            # Overwrite entire file
            save_token("squ_new_token", "my-org", "my-project", merge=False)
        """


        try:
            # Load existing config if merging
            if merge and self.sonar_path.exists():

                existing_config = self.load_auth_config()
            else:
                existing_config = {
                    "token": None,
                    "organization": None,
                    "project": None,
                    "project_path": None,
                }

            # Prepare new config
            new_config = {
                "SONAR_TOKEN": token,
                "SONAR_ORG": (
                    organization
                    if organization is not None
                    else existing_config.get("organization")
                ),
                "SONAR_PROJ": (
                    project if project is not None else existing_config.get("project")
                ),
                "PROJECT_PATH": (
                    project_path
                    if project_path is not None
                    else existing_config.get("project_path")
                ),
            }

            # Write to file
            with open(self.sonar_path, "w") as f:
                json.dump(new_config, f, indent=2)

            # Set secure permissions (Unix/Linux/macOS only)
            try:
                self.sonar_path.chmod(0o600)
            except (OSError, NotImplementedError):
                # Windows doesn't support chmod the same way
                pass

            console.print(f"[green]✓ Token saved to {self.sonar_path}[/green]")
            return True

        except (IOError, PermissionError) as e:
            console.print(f"[red]Error saving token: {e}[/red]")
            return False


def validate_token_format(token: str) -> bool:
    """
    Validate SonarCloud token format.

    Args:
        token: Token to validate

    Returns:
        True if token format appears valid, False otherwise.

    Note:
        SonarCloud tokens typically start with 'squ_' or 'sqa_'
        and are 40+ characters long.
    """
    if not token:
        return False

    # Basic validation
    if len(token) < 20:
        console.print("[yellow]Warning: Token seems too short[/yellow]")
        return False

    return True


