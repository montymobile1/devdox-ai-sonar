from pathlib import Path
from typing import Optional, Dict, Any, List
import tomli
import tomli_w
import tomlkit
from pydantic import BaseModel
import shutil
from datetime import datetime
from devdox_ai_sonar.models.sonar import SonarType
from devdox_ai_sonar.utils.async_file_io import AsyncFileReader

failed_load_config = "Failed to load configuration"


class AppConfig(BaseModel):
    """Complete application configuration"""

    sonar: Dict[str, Any]
    llm: Dict[str, Any]


class ConfigManager:
    """Manages configuration file with validation and auto-fixing"""

    DEFAULT_CONFIG = {
        "sonar": {"default_branch": "main", "sonar_options": {"clone_type": "branch"}},
        "llm": {"default_provider": "", "default_model": "", "providers": []},
    }

    def __init__(self, config_path: Path = Path("config.toml")):
        self.config_path = config_path
        self.config: Optional[Dict[str, Any]] = None
        self.file_reader = AsyncFileReader()

    def create_backup(self) -> Optional[Path]:
        """Create backup of existing config file"""
        if not self.config_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config_path.with_suffix(f".backup_{timestamp}.toml")
        shutil.copy2(self.config_path, backup_path)
        return backup_path

    def create_default_config(self) -> None:
        """Create default config file if it doesn't exist"""
        if self.config_path.exists():
            return

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write default config
        with open(self.config_path, "wb") as f:
            tomli_w.dump(self.DEFAULT_CONFIG, f)

    async def load_config(self) -> Dict[str, Any]:
        """Load and parse config file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Run with --create-config to create a default configuration."
            )
        self.config = await self.file_reader.read_toml_file(self.config_path)

        return self.config

    async def get_value(self, key_path: str) -> Any:
        """
        Get a value from config using dot notation
        Example: 'llm.default_provider' or 'git.clone_options.clone_type'
        """
        if not self.config:
            await self.load_config()

        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                raise KeyError(f"Cannot traverse non-dict at: {key}")

        return value

    async def set_value(self, key_path: str, new_value: Any, validate: bool = True) -> None:
        """
        Set a value in config using dot notation
        Example: set_value('llm.default_provider', 'anthropic')
        """
        if not self.config:
            await self.load_config()

        if not self.config:
            raise RuntimeError(failed_load_config)

        keys = key_path.split(".")
        current: Dict[str, Any] = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        old_value = current.get(keys[-1])
        current[keys[-1]] = new_value

        # Validate if requested
        if validate:
            is_valid, issues = self.validate_change(key_path, new_value)
            if not is_valid:
                # Revert change
                if old_value is not None:
                    current[keys[-1]] = old_value
                else:
                    del current[keys[-1]]

                raise ValueError(
                    f"Validation failed for {key_path}:\n"
                    + "\n".join(f"  - {issue}" for issue in issues)
                )

    def delete_value(self, key_path: str) -> bool:
        """
        Delete a value from config using dot notation.
        Example: delete_value('configuration.exclude_rules')

        Returns:
            True if the key was deleted, False if it didn't exist
        """
        if not self.config:
            self.load_config()

        if not self.config:
            raise RuntimeError(failed_load_config)

        keys = key_path.split(".")
        current: Dict[str, Any] = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                return False
            current = current[key]
            if not isinstance(current, dict):
                return False

        # Delete the final key if it exists
        if keys[-1] in current:
            del current[keys[-1]]
            return True
        return False

    def validate_change(self, key_path: str, new_value: Any) -> tuple[bool, List[str]]:
        """
        Validate a configuration change
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        if not self.config:
            raise RuntimeError("No configuration to save")

        # Validate default_provider change
        if key_path == "llm.default_provider":
            llm_config = self.config.get("llm", {})

            providers = llm_config.get("providers", [])

            provider_names = [p["name"] for p in providers]
            if new_value not in provider_names:
                issues.append(
                    f"Provider '{new_value}' not found in configured providers. "
                    f"Available: {', '.join(provider_names)}"
                )

        # Validate default_model change
        if key_path == "llm.default_model":
            llm_config = self.config.get("llm", {})

            default_provider = llm_config.get("default_provider")

            providers = llm_config.get("providers", [])
            provider = next(
                (p for p in providers if p["name"] == default_provider),
                None,
            )
            if provider and new_value not in provider["models"]:
                issues.append(
                    f"Model '{new_value}' not available for provider '{default_provider}'. "
                    f"Available: {', '.join(provider['models'])}"
                )

        # Validate clone_type change
        if key_path == "sonar.sonar_options.clone_type":
            valid_types = [t.value for t in SonarType]
            if new_value not in valid_types:
                issues.append(
                    f"Invalid clone_type: '{new_value}'. "
                    f"Must be one of: {', '.join(valid_types)}"
                )

        return len(issues) == 0, issues

    async def update_provider(
        self, provider_name: str, updates: Dict[str, Any], set_as_default: bool = False
    ) -> None:
        """
        Update a specific provider's configuration

        Args:
            provider_name: Name of the provider to update
            updates: Dictionary of fields to update
            set_as_default: Whether to set this as the default provider
        """
        if not self.config:
            await self.load_config()

        if not self.config:
            raise RuntimeError(failed_load_config)

        # Find the provider
        llm_config = self.config.get("llm", {})

        providers: List[Dict[str, Any]] = llm_config.get("providers", [])

        provider_idx = next(
            (i for i, p in enumerate(providers) if p["name"] == provider_name), None
        )

        if provider_idx is None:
            raise ValueError(f"Provider '{provider_name}' not found")

        # Update provider fields
        provider = providers[provider_idx]
        old_values = {}

        for key, value in updates.items():
            old_values[key] = provider.get(key)
            provider[key] = value

        # Set as default if requested
        if set_as_default:
            self.config["llm"]["default_provider"] = provider_name
            self.config["llm"]["default_model"] = provider["default_model"]

    async def add_provider(
        self, provider_config: Dict[str, Any], set_as_default: bool = False
    ) -> None:
        """Add a new provider to the configuration"""
        if not self.config:
            await self.load_config()

        if not self.config:
            raise RuntimeError(failed_load_config)

        provider_name = provider_config.get("name")
        if not provider_name:
            raise ValueError("Provider must have a 'name' field")

        # Check if provider already exists
        if "llm" not in self.config:
            self.config["llm"] = {
                "providers": [],
                "default_provider": "",
                "default_model": "",
            }

        llm_config = self.config["llm"]

        # Initialize providers list if needed
        if "providers" not in llm_config:
            llm_config["providers"] = []

        providers: List[Dict[str, Any]] = llm_config["providers"]

        # Check if provider already exists
        existing_names = [p["name"] for p in providers]

        if provider_name in existing_names:
            raise ValueError(f"Provider '{provider_name}' already exists")
        # Validate required fields
        required = ["name", "api_key", "models"]
        missing = [field for field in required if field not in provider_config]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        # Add provider
        self.config["llm"]["providers"].append(provider_config)

        # Set as default if requested
        if set_as_default:
            self.config["llm"]["default_provider"] = provider_name
            self.config["llm"]["default_model"] = provider_config.get("default_model")

    async def remove_provider(self, provider_name: str) -> None:
        """Remove a provider from the configuration"""
        if not self.config:
            await self.load_config()

        if not self.config:
            raise RuntimeError(failed_load_config)

        llm_config = self.config.get("llm", {})

        # Check if it's the default provider
        if llm_config["default_provider"] == provider_name:
            raise ValueError(
                f"Cannot remove default provider '{provider_name}'. "
                f"Set a different default provider first."
            )

        # Find and remove provider
        providers: List[Dict[str, Any]] = llm_config["providers"]
        original_count = len(providers)

        llm_config["providers"] = [p for p in providers if p["name"] != provider_name]

        if len(llm_config["providers"]) == original_count:
            raise ValueError(f"Provider '{provider_name}' not found")

    async def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers"""
        if not self.config:
            await self.load_config()
        if not self.config:
            raise RuntimeError(failed_load_config)

        llm_config = self.config.get("llm", {})
        providers: List[Dict[str, Any]] = llm_config.get("providers", [])
        return providers

    async def show_provider(self, provider_name: str) -> Dict[str, Any]:
        """Show details of a specific provider"""
        if not self.config:
            await self.load_config()

        if not self.config:
            raise RuntimeError(failed_load_config)

        llm_config = self.config.get("llm", {})
        providers: List[Dict[str, Any]] = llm_config.get("providers", [])

        for provider in providers:
            if provider["name"] == provider_name:
                return provider

        raise ValueError(f"Provider '{provider_name}' not found")

    def save_config(self, create_backup: bool = False) -> None:
        """Save current config to file"""
        if create_backup:
            self.create_backup()

        if not self.config:
            raise RuntimeError("No configuration to save")

        doc = tomlkit.document()

        for section_key, section_value in self.config.items():
            doc[section_key] = self._build_section_value(section_value)

        with open(self.config_path, "w") as f:
            f.write(tomlkit.dumps(doc))

    def _build_section_value(self, section_value: Any) -> Any:
        """Build TOML value for a config section"""
        if not isinstance(section_value, dict):
            return section_value

        section_table = tomlkit.table()

        for key, value in section_value.items():
            section_table[key] = self._build_table_value(key, value)

        return section_table

    def _build_table_value(self, key: str, value: Any) -> Any:
        """Build TOML value for a table entry"""
        if key == "providers" and self._is_provider_array(value):
            return self._build_providers_array(value)

        if isinstance(value, dict):
            return self._build_nested_table(value)

        return value

    def _is_provider_array(self, value: Any) -> bool:
        """Check if value is a non-empty provider array"""
        return isinstance(value, list) and len(value) > 0

    def _build_providers_array(
        self, providers: List[Dict[str, Any]]
    ) -> tomlkit.items.AoT:
        """Build array of tables for providers"""
        aot = tomlkit.aot()

        for provider in providers:
            provider_table = self._build_provider_table(provider)
            aot.append(provider_table)

        return aot

    def _build_provider_table(self, provider: Dict[str, Any]) -> tomlkit.items.Table:
        """Build a single provider table"""
        provider_table = tomlkit.table()

        for key, value in provider.items():
            if isinstance(value, list):
                provider_table[key] = self._build_inline_array(value)
            else:
                provider_table[key] = value

        return provider_table

    def _build_inline_array(self, items: List[Any]) -> tomlkit.items.Array:
        """Build an inline TOML array"""
        arr = tomlkit.array()
        arr.multiline(False)

        for item in items:
            arr.append(item)

        return arr

    def _build_nested_table(self, nested_dict: Dict[str, Any]) -> tomlkit.items.Table:
        """Build a nested TOML table"""
        nested_table = tomlkit.table()

        for key, value in nested_dict.items():
            nested_table[key] = value

        return nested_table
