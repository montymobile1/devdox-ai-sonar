# Create src/devdox_ai_sonar/services/config_service.py

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json
from rich.console import Console

from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.utils.async_file_io import AsyncFileReader


@dataclass
class AuthConfig:
    token: str
    organization: str
    project: str
    project_path: str
    git_url: str

    @classmethod
    def from_dict(cls, data: Dict[str, Optional[str]]) -> Optional["AuthConfig"]:
        """Create AuthConfig from dictionary with validation.

        Validates that all required fields are present and non-empty.

        Args:
            data: Dictionary containing configuration values

        Returns:
            AuthConfig instance if valid, None if any field is missing/empty

        Example:
            >>> config_dict = {"token": "abc123", "organization": "myorg", ...}
            >>> auth = AuthConfig.from_dict(config_dict)
            >>> if auth:
            ...     print(f"Loaded config for {auth.organization}")
        """
        # Extract all required fields
        token = data.get("token")
        organization = data.get("organization")
        project = data.get("project")
        project_path = data.get("project_path")
        git_url = data.get("git_url")

        # Validate all are present and non-empty
        if (
            not token
            or not organization
            or not project
            or not project_path
            or not git_url
        ):
            return None

        # Safe to create now - all values validated as non-None, non-empty strings
        return cls(
            token=token,
            organization=organization,
            project=project,
            project_path=project_path,
            git_url=git_url,
        )

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
        if not self.git_url:
            return False, "Git URL is required"
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
        self.file_reader = AsyncFileReader()

        self.config: Optional[Dict[str, Any]] = None

    async def load_auth_config(self) -> Dict[str, Optional[str]]:
        """
        Load authentication configuration from auth.json file.

        Returns:
            Dictionary with SONAR_TOKEN, SONAR_ORG, and SONAR_PROJ keys.
            Returns empty dict if file doesn't exist or is invalid.
        """
        empty_config: Dict[str, Optional[str]] = {
            "token": None,
            "organization": None,
            "project": None,
            "project_path": None,
            "git_url": None,
        }
        try:
            if self.sonar_path.exists():
                config = await self.file_reader.read_json_file(self.sonar_path)

                return {
                        "token": config.get("SONAR_TOKEN"),
                        "organization": config.get("SONAR_ORG"),
                        "project": config.get("SONAR_PROJ"),
                        "project_path": config.get("PROJECT_PATH"),
                        "git_url": config.get("GIT_URL"),
                    }
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read auth.json: {e}[/yellow]")
            return empty_config

        return empty_config

    @staticmethod
    async def load_llm_config(manager: ConfigManager) -> Optional[LLMConfig]:
        """Load and validate LLM configuration"""

        providers = await manager.get_value("llm.providers")
        if not providers:
            return None

        default_provider = await manager.get_value("llm.default_provider")
        default_model = await manager.get_value("llm.default_model")
        provider_config = next(
            (p for p in providers if p.get("name") == default_provider), None
        )

        if not provider_config:
            return None

        return LLMConfig(
            provider=default_provider,
            model=default_model,
            api_key=provider_config.get("api_key"),
            models=provider_config.get("models", []),
        )

    @staticmethod
    def validate_auth_config(auth_config: dict) -> bool:
        """Validate authentication configuration with detailed error messages."""
        required_fields = ["token", "organization", "project", "project_path"]
        missing_fields = [
            field for field in required_fields if not auth_config.get(field)
        ]

        if missing_fields:
            return False
        return True

    def save_config(
        self,
        token: str,
        organization: str,
        project: str,
        project_path: str,
        git_url: str,
    ) -> bool:
        """Save authentication configuration"""

        if not validate_token_format(token) and (
            not console.input(
                "[yellow]Token format seems unusual. Continue? (y/N): [/yellow]"
            )
            .lower()
            .startswith("y")
        ):
            console.print("[red]Cancelled[/red]")
            return False
        success = self.save_complete_config(
            token, organization, project, project_path, git_url
        )

        return success

    async def save_complete_config(
        self,
        token: str,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        project_path: Optional[str] = None,
        git_url: Optional[str] = None,
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
                existing_config = await self.load_auth_config()
            else:
                existing_config = {
                    "token": None,
                    "organization": None,
                    "project": None,
                    "project_path": None,
                    "git_url": None,
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
                "GIT_URL": (
                    git_url if git_url is not None else existing_config.get("git_url")
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

    def check_all_value_empty(self, auth_config: dict) -> bool:
        """Check if any value in the auth_config is empty."""
        required_keys = ["token", "organization", "project", "project_path", "git_url"]

        missing_or_empty = [k for k in required_keys if not auth_config.get(k)]

        if missing_or_empty:
            return True
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
