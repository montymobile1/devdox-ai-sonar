from pathlib import Path
from typing import Optional, Dict, Any, List
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
        "llm": {
            "default_profile": "",
            "profiles": [],
            # Exact-match filters applied to the interactive picker.
            # ``excluded_providers`` removes a provider family (and all
            # its models) from the top-level menu; ``excluded_models``
            # hides individual models in full ``provider/model-id``
            # form. Custom-path input is NOT filtered -- if the user
            # types an excluded model explicitly, we accept it.
            "excluded_providers": [],
            "excluded_models": [],
        },
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
        if self.config is None:
            raise FileNotFoundError(
                f"Config file is empty or invalid: {self.config_path}"
            )
        return self.config

    async def get_value(self, key_path: str) -> Any:
        """
        Get a value from config using dot notation
        Example: 'llm.default_profile' or 'sonar.sonar_options.clone_type'
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

    async def set_value(
        self, key_path: str, new_value: Any, validate: bool = True
    ) -> None:
        """
        Set a value in config using dot notation
        Example: set_value('llm.default_profile', 'my-profile-name')
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

    async def delete_value(self, key_path: str) -> bool:
        """
        Delete a value from config using dot notation.
        Example: delete_value('configuration.exclude_rules')

        Returns:
            True if the key was deleted, False if it didn't exist
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
                return False
            value = current[key]
            if not isinstance(value, dict):
                return False
            current = value

        # Delete the final key if it exists
        if keys[-1] in current:
            del current[keys[-1]]
            return True
        return False

    def validate_change(self, key_path: str, new_value: Any) -> tuple[bool, List[str]]:
        """Return ``(is_valid, issues)`` for a proposed change at ``key_path``."""
        issues = []
        if not self.config:
            raise RuntimeError("No configuration to save")

        if key_path == "sonar.sonar_options.clone_type":
            valid_types = [t.value for t in SonarType]
            if new_value not in valid_types:
                issues.append(
                    f"Invalid clone_type: '{new_value}'. "
                    f"Must be one of: {', '.join(valid_types)}"
                )

        return len(issues) == 0, issues

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
