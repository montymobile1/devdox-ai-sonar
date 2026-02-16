import pytest
import tomli
import tomli_w
import tomlkit
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, mock_open
from devdox_ai_sonar.models.llm_config import AppConfig, ConfigManager
from devdox_ai_sonar.models.sonar import SonarType


# ============================================================================
# TEST CLASS: AppConfig
# ============================================================================

class TestAppConfig:
    """Test AppConfig Pydantic model"""

    def test_app_config_creation(self):
        """Test creating AppConfig"""
        config = AppConfig(
            sonar={"default_branch": "main"},
            llm={"default_provider": "openai"}
        )

        assert config.sonar["default_branch"] == "main"
        assert config.llm["default_provider"] == "openai"

    def test_app_config_validation(self):
        """Test AppConfig validation"""
        with pytest.raises(Exception):  # ValidationError
            AppConfig()  # Missing required fields


# ============================================================================
# TEST CLASS: ConfigManager - Initialization
# ============================================================================

class TestConfigManagerInit:
    """Test ConfigManager initialization"""

    def test_config_manager_creation_default_path(self):
        """Test creating ConfigManager with default path"""
        manager = ConfigManager()

        assert manager.config_path == Path("config.toml")
        assert manager.config is None

    def test_config_manager_creation_custom_path(self):
        """Test creating ConfigManager with custom path"""
        custom_path = Path("/tmp/custom_config.toml")
        manager = ConfigManager(config_path=custom_path)

        assert manager.config_path == custom_path

    def test_config_manager_default_config_structure(self):
        """Test DEFAULT_CONFIG structure"""
        assert "sonar" in ConfigManager.DEFAULT_CONFIG
        assert "llm" in ConfigManager.DEFAULT_CONFIG
        assert "default_branch" in ConfigManager.DEFAULT_CONFIG["sonar"]
        assert "providers" in ConfigManager.DEFAULT_CONFIG["llm"]


# ============================================================================
# TEST CLASS: ConfigManager - Backup
# ============================================================================

class TestConfigManagerBackup:
    """Test backup functionality"""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create temporary config file"""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[sonar]\ndefault_branch = 'main'\n")
        return config_file

    def test_create_backup_success(self, temp_config_file):
        """Test creating backup successfully"""
        manager = ConfigManager(config_path=temp_config_file)

        backup_path = manager.create_backup()

        assert backup_path is not None
        assert backup_path.exists()
        assert "backup" in str(backup_path)

    def test_create_backup_nonexistent_file(self, tmp_path):
        """Test creating backup when config doesn't exist"""
        nonexistent = tmp_path / "nonexistent.toml"
        manager = ConfigManager(config_path=nonexistent)

        backup_path = manager.create_backup()

        assert backup_path is None

    def test_create_backup_timestamp_format(self, temp_config_file):
        """Test backup filename includes timestamp"""
        manager = ConfigManager(config_path=temp_config_file)

        backup_path = manager.create_backup()

        assert backup_path is not None
        # Should contain timestamp in format YYYYMMDD_HHMMSS
        assert "_" in backup_path.stem

    def test_create_backup_preserves_content(self, temp_config_file):
        """Test backup preserves original content"""
        original_content = temp_config_file.read_text()
        manager = ConfigManager(config_path=temp_config_file)

        backup_path = manager.create_backup()

        assert backup_path.read_text() == original_content


# ============================================================================
# TEST CLASS: ConfigManager - Create Default Config
# ============================================================================

class TestConfigManagerCreateDefault:
    """Test default config creation"""

    def test_create_default_config_new_file(self, tmp_path):
        """Test creating default config for new file"""
        config_file = tmp_path / "new_config.toml"
        manager = ConfigManager(config_path=config_file)

        manager.create_default_config()

        assert config_file.exists()

    def test_create_default_config_creates_directory(self, tmp_path):
        """Test creates parent directory if needed"""
        config_file = tmp_path / "nested" / "dir" / "config.toml"
        manager = ConfigManager(config_path=config_file)

        manager.create_default_config()

        assert config_file.parent.exists()
        assert config_file.exists()

    def test_create_default_config_existing_file(self, tmp_path):
        """Test doesn't overwrite existing config"""
        config_file = tmp_path / "existing.toml"
        config_file.write_text("[custom]\nvalue = 123\n")
        original_content = config_file.read_text()

        manager = ConfigManager(config_path=config_file)
        manager.create_default_config()

        # Should not change existing file
        assert config_file.read_text() == original_content

    def test_create_default_config_valid_toml(self, tmp_path):
        """Test created config is valid TOML"""
        config_file = tmp_path / "config.toml"
        manager = ConfigManager(config_path=config_file)

        manager.create_default_config()

        # Should be parseable
        with open(config_file, "rb") as f:
            config = tomli.load(f)

        assert "sonar" in config
        assert "llm" in config


# ============================================================================
# TEST CLASS: ConfigManager - Load Config
# ============================================================================

class TestConfigManagerLoadConfig:
    """Test config loading"""

    @pytest.fixture
    def valid_config_file(self, tmp_path):
        """Create valid config file"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main"},
            "llm": {"default_provider": "openai", "providers": []}
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)
        return config_file

    async def test_load_config_success(self, valid_config_file):
        """Test loading valid config"""
        manager = ConfigManager(config_path=valid_config_file)

        config = await manager.load_config()

        assert config is not None
        assert "sonar" in config
        assert "llm" in config
        assert manager.config == config

    async def test_load_config_nonexistent_file(self, tmp_path):
        """Test loading nonexistent config raises error"""
        nonexistent = tmp_path / "nonexistent.toml"
        manager = ConfigManager(config_path=nonexistent)

        with pytest.raises(FileNotFoundError) as exc_info:
            await manager.load_config()

        assert "not found" in str(exc_info.value).lower()
        assert "--create-config" in str(exc_info.value)

    async def test_load_config_invalid_toml(self, tmp_path):
        """Test loading invalid TOML raises error"""
        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("invalid toml content {{{")

        manager = ConfigManager(config_path=invalid_file)

        with pytest.raises(Exception):  # tomli.TOMLDecodeError
            await manager.load_config()

    async def test_load_config_stores_in_manager(self, valid_config_file):
        """Test loaded config is stored in manager"""
        manager = ConfigManager(config_path=valid_config_file)

        config = await manager.load_config()

        assert manager.config is not None
        assert manager.config == config


# ============================================================================
# TEST CLASS: ConfigManager - Get Value
# ============================================================================

class TestConfigManagerGetValue:
    """Test getting config values"""

    @pytest.fixture
    async def manager_with_config(self, tmp_path):
        """Create manager with loaded config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main", "options": {"clone_type": "branch"}},
            "llm": {
                "default_provider": "openai",
                "providers": [
                    {"name": "openai", "api_key": "key1"},
                    {"name": "gemini", "api_key": "key2"}
                ]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    async def test_get_value_top_level(self, manager_with_config):
        """Test getting top-level value"""
        value = await manager_with_config.get_value("sonar")

        assert value is not None
        assert "default_branch" in value

    async def test_get_value_nested(self, manager_with_config):
        """Test getting nested value"""
        value = await manager_with_config.get_value("sonar.default_branch")

        assert value == "main"

    async def test_get_value_deeply_nested(self, manager_with_config):
        """Test getting deeply nested value"""
        value = await manager_with_config.get_value("sonar.options.clone_type")

        assert value == "branch"

    async def test_get_value_list(self, manager_with_config):
        """Test getting list value"""
        value = await manager_with_config.get_value("llm.providers")

        assert isinstance(value, list)
        assert len(value) == 2

    async def test_get_value_nonexistent_key(self, manager_with_config):
        """Test getting nonexistent key returns None"""
        value = await manager_with_config.get_value("nonexistent.key")

        assert value is None

    async def test_get_value_loads_config_if_needed(self, tmp_path):
        """Test get_value loads config if not already loaded"""
        config_file = tmp_path / "config.toml"
        config_data = {"test": {"value": 42}}
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        # Don't call load_config manually

        value = await manager.get_value("test.value")

        assert value == 42
        assert manager.config is not None

    async def test_get_value_traverse_non_dict_raises(self, manager_with_config):
        """Test traversing non-dict raises KeyError"""
        with pytest.raises(KeyError) as exc_info:
            # Try to traverse through a string value
            await manager_with_config.get_value("sonar.default_branch.invalid")

        assert "non-dict" in str(exc_info.value).lower()


# ============================================================================
# TEST CLASS: ConfigManager - Set Value
# ============================================================================

class TestConfigManagerSetValue:
    """Test setting config values"""

    @pytest.fixture
    async def manager_with_config(self, tmp_path):
        """Create manager with loaded config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main"},
            "llm": {
                "default_provider": "openai",
                "providers": [{"name": "openai", "models": ["gpt-4"], "default_model": "gpt-4"}]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    async def test_set_value_existing_key(self, manager_with_config):
        """Test setting existing key"""
        await manager_with_config.set_value("sonar.default_branch", "develop", validate=False)

        value = await manager_with_config.get_value("sonar.default_branch")
        assert value == "develop"

    async def test_set_value_new_key(self, manager_with_config):
        """Test setting new key"""
        await manager_with_config.set_value("sonar.new_option", "value", validate=False)

        value = await manager_with_config.get_value("sonar.new_option")
        assert value == "value"

    async def test_set_value_creates_nested_structure(self, manager_with_config):
        """Test setting value creates nested dicts as needed"""
        await manager_with_config.set_value("new.nested.key", 123, validate=False)

        value = await manager_with_config.get_value("new.nested.key")
        assert value == 123

    async def test_set_value_with_validation_success(self, manager_with_config):
        """Test setting value with successful validation"""
        await manager_with_config.set_value("llm.default_provider", "openai", validate=True)

        value = await manager_with_config.get_value("llm.default_provider")
        assert value == "openai"

    async def test_set_value_validation_reverts_on_failure(self, manager_with_config):
        """Test failed validation reverts change"""
        original = await manager_with_config.get_value("llm.default_provider")

        with pytest.raises(ValueError):
            # Try to set invalid provider
            await manager_with_config.set_value("llm.default_provider", "nonexistent", validate=True)

        # Value should be unchanged
        current = await manager_with_config.get_value("llm.default_provider")
        assert current == original

    async def test_set_value_validation_deletes_new_key_on_failure(self, manager_with_config):
        """Test validation failure deletes newly created key"""
        # Try to set invalid clone_type
        with pytest.raises(ValueError):
            await manager_with_config.set_value("sonar.sonar_options.clone_type", "invalid_type", validate=True)

        # Key should not exist
        value = await manager_with_config.get_value("sonar.sonar_options.clone_type")
        assert value is None


# ============================================================================
# TEST CLASS: ConfigManager - Delete Value
# ============================================================================

class TestConfigManagerDeleteValue:
    """Test configuration value deletion"""

    @pytest.fixture
    async def manager_with_config(self, tmp_path):
        """Create manager with sample config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main"},
            "llm": {"default_provider": "openai", "default_model": "gpt-4"},
            "configuration": {
                "max_fixes": 10,
                "exclude_rules": "python:S1234,python:S5678"
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)
        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    async def test_delete_value_existing_key(self, manager_with_config):
        """Test deleting an existing key"""
        # Verify key exists
        assert await manager_with_config.get_value("configuration.exclude_rules") is not None

        # Delete the key
        result = await manager_with_config.delete_value("configuration.exclude_rules")

        assert result is True
        assert await manager_with_config.get_value("configuration.exclude_rules") is None

    async def test_delete_value_nonexistent_key(self, manager_with_config):
        """Test deleting a key that doesn't exist"""
        result = await manager_with_config.delete_value("configuration.nonexistent_key")

        assert result is False

    async def test_delete_value_nested_key(self, manager_with_config):
        """Test deleting a nested key"""
        result = await manager_with_config.delete_value("llm.default_model")

        assert result is True
        assert await manager_with_config.get_value("llm.default_model") is None
        # Parent should still exist
        assert await manager_with_config.get_value("llm.default_provider") == "openai"

    async def test_delete_value_top_level_key(self, manager_with_config):
        """Test deleting a top-level section"""
        result = await manager_with_config.delete_value("configuration")

        assert result is True
        assert await manager_with_config.get_value("configuration") is None

    async def test_delete_value_nonexistent_parent(self, manager_with_config):
        """Test deleting when parent path doesn't exist"""
        result = await manager_with_config.delete_value("nonexistent.parent.key")

        assert result is False

    async def test_delete_value_loads_config_if_needed(self, tmp_path):
        """Test that delete_value works when config is loaded beforehand"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "test": {"key": "value"}
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        # load_config is now async, so we must call it explicitly
        await manager.load_config()

        result = await manager.delete_value("test.key")

        assert result is True
        assert await manager.get_value("test.key") is None

    async def test_delete_value_persists_after_save(self, manager_with_config, tmp_path):
        """Test that deleted value is not present after save and reload"""
        await manager_with_config.delete_value("configuration.exclude_rules")
        manager_with_config.save_config()

        # Create new manager and load
        new_manager = ConfigManager(config_path=manager_with_config.config_path)
        await new_manager.load_config()

        assert await new_manager.get_value("configuration.exclude_rules") is None
        # Other values should still exist
        assert await new_manager.get_value("configuration.max_fixes") == 10


# ============================================================================
# TEST CLASS: ConfigManager - Validate Change
# ============================================================================

class TestConfigManagerValidateChange:
    """Test configuration validation"""

    @pytest.fixture
    async def manager_with_providers(self, tmp_path):
        """Create manager with provider config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main", "sonar_options": {"clone_type": "branch"}},
            "llm": {
                "default_provider": "openai",
                "default_model": "gpt-4",
                "providers": [
                    {"name": "openai", "models": ["gpt-4", "gpt-3.5"], "default_model": "gpt-4"},
                    {"name": "anthropic", "models": ["claude-3"], "default_model": "claude-3"}
                ]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    def test_validate_change_valid_provider(self, manager_with_providers):
        """Test validating valid provider change"""
        is_valid, issues = manager_with_providers.validate_change("llm.default_provider", "anthropic")

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_change_invalid_provider(self, manager_with_providers):
        """Test validating invalid provider change"""
        is_valid, issues = manager_with_providers.validate_change("llm.default_provider", "nonexistent")

        assert is_valid is False
        assert len(issues) > 0
        assert "not found" in issues[0].lower()
        assert "nonexistent" in issues[0]

    def test_validate_change_valid_model(self, manager_with_providers):
        """Test validating valid model change"""
        is_valid, issues = manager_with_providers.validate_change("llm.default_model", "gpt-3.5")

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_change_invalid_model(self, manager_with_providers):
        """Test validating invalid model for current provider"""
        is_valid, issues = manager_with_providers.validate_change("llm.default_model", "invalid-model")

        assert is_valid is False
        assert len(issues) > 0
        assert "not available" in issues[0].lower()

    def test_validate_change_valid_clone_type(self, manager_with_providers):
        """Test validating valid clone_type"""
        is_valid, issues = manager_with_providers.validate_change(
            "sonar.sonar_options.clone_type",
            SonarType.BRANCH.value
        )

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_change_invalid_clone_type(self, manager_with_providers):
        """Test validating invalid clone_type"""
        is_valid, issues = manager_with_providers.validate_change(
            "sonar.sonar_options.clone_type",
            "invalid_type"
        )

        assert is_valid is False
        assert len(issues) > 0
        assert "invalid clone_type" in issues[0].lower()

    def test_validate_change_unvalidated_key(self, manager_with_providers):
        """Test keys without validation rules pass"""
        is_valid, issues = manager_with_providers.validate_change("sonar.default_branch", "develop")

        assert is_valid is True
        assert len(issues) == 0


# ============================================================================
# TEST CLASS: ConfigManager - Update Provider
# ============================================================================

class TestConfigManagerUpdateProvider:
    """Test provider update operations"""

    @pytest.fixture
    async def manager_with_providers(self, tmp_path):
        """Create manager with provider config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "llm": {
                "default_provider": "openai",
                "default_model": "gpt-4",
                "providers": [
                    {
                        "name": "openai",
                        "api_key": "old_key",
                        "models": ["gpt-4"],
                        "default_model": "gpt-4"
                    }
                ]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    async def test_update_provider_api_key(self, manager_with_providers):
        """Test updating provider API key"""
        await manager_with_providers.update_provider("openai", {"api_key": "new_key"})

        providers = await manager_with_providers.get_value("llm.providers")
        assert providers[0]["api_key"] == "new_key"

    async def test_update_provider_models(self, manager_with_providers):
        """Test updating provider models"""
        await manager_with_providers.update_provider("openai", {"models": ["gpt-4", "gpt-3.5"]})

        providers = await manager_with_providers.get_value("llm.providers")
        assert "gpt-3.5" in providers[0]["models"]

    async def test_update_provider_set_as_default(self, manager_with_providers):
        """Test setting provider as default during update"""
        # Add another provider first
        await manager_with_providers.add_provider({
            "name": "anthropic",
            "api_key": "key",
            "models": ["claude-3"],
            "default_model": "claude-3"
        })

        # Update and set as default
        await manager_with_providers.update_provider("anthropic", {"api_key": "new_key"}, set_as_default=True)

        assert await manager_with_providers.get_value("llm.default_provider") == "anthropic"
        assert await manager_with_providers.get_value("llm.default_model") == "claude-3"

    async def test_update_provider_nonexistent_raises(self, manager_with_providers):
        """Test updating nonexistent provider raises error"""
        with pytest.raises(ValueError) as exc_info:
            await manager_with_providers.update_provider("nonexistent", {"api_key": "key"})

        assert "not found" in str(exc_info.value).lower()

    async def test_update_provider_multiple_fields(self, manager_with_providers):
        """Test updating multiple provider fields"""
        updates = {
            "api_key": "new_key",
            "models": ["gpt-4", "gpt-3.5"],
            "default_model": "gpt-3.5"
        }
        await manager_with_providers.update_provider("openai", updates)

        providers = await manager_with_providers.get_value("llm.providers")
        provider = providers[0]
        assert provider["api_key"] == "new_key"
        assert provider["default_model"] == "gpt-3.5"
        assert len(provider["models"]) == 2


# ============================================================================
# TEST CLASS: ConfigManager - Add Provider
# ============================================================================

class TestConfigManagerAddProvider:
    """Test adding providers"""

    @pytest.fixture
    async def manager_with_config(self, tmp_path):
        """Create manager with basic config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "llm": {
                "default_provider": "openai",
                "providers": [{"name": "openai", "api_key": "key1", "models": ["gpt-4"]}]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    async def test_add_provider_success(self, manager_with_config):
        """Test adding new provider successfully"""
        new_provider = {
            "name": "anthropic",
            "api_key": "key2",
            "models": ["claude-3"],
            "default_model": "claude-3"
        }
        await manager_with_config.add_provider(new_provider)

        providers = await manager_with_config.get_value("llm.providers")
        assert len(providers) == 2
        assert providers[1]["name"] == "anthropic"

    async def test_add_provider_set_as_default(self, manager_with_config):
        """Test adding provider and setting as default"""
        new_provider = {
            "name": "anthropic",
            "api_key": "key2",
            "models": ["claude-3"],
            "default_model": "claude-3"
        }
        await manager_with_config.add_provider(new_provider, set_as_default=True)

        assert await manager_with_config.get_value("llm.default_provider") == "anthropic"
        assert await manager_with_config.get_value("llm.default_model") == "claude-3"

    async def test_add_provider_duplicate_name_raises(self, manager_with_config):
        """Test adding provider with duplicate name raises error"""
        duplicate_provider = {
            "name": "openai",
            "api_key": "key2",
            "models": ["gpt-4"]
        }

        with pytest.raises(ValueError) as exc_info:
            await manager_with_config.add_provider(duplicate_provider)

        assert "already exists" in str(exc_info.value).lower()

    async def test_add_provider_missing_name_raises(self, manager_with_config):
        """Test adding provider without name raises error"""
        invalid_provider = {
            "api_key": "key2",
            "models": ["claude-3"]
        }

        with pytest.raises(ValueError) as exc_info:
            await manager_with_config.add_provider(invalid_provider)

        assert "name" in str(exc_info.value).lower()

    async def test_add_provider_missing_required_fields_raises(self, manager_with_config):
        """Test adding provider without required fields raises error"""
        invalid_provider = {
            "name": "anthropic",
            "api_key": "key2"
            # Missing 'models'
        }

        with pytest.raises(ValueError) as exc_info:
            await manager_with_config.add_provider(invalid_provider)

        assert "missing required fields" in str(exc_info.value).lower()
        assert "models" in str(exc_info.value).lower()

    async def test_add_provider_to_empty_config(self, tmp_path):
        """Test adding provider to config without llm section"""
        config_file = tmp_path / "config.toml"
        config_data = {"sonar": {"default_branch": "main"}}
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()

        new_provider = {
            "name": "openai",
            "api_key": "key1",
            "models": ["gpt-4"]
        }
        await manager.add_provider(new_provider)

        assert "llm" in manager.config
        assert "providers" in manager.config["llm"]
        assert len(manager.config["llm"]["providers"]) == 1


# ============================================================================
# TEST CLASS: ConfigManager - Remove Provider
# ============================================================================

class TestConfigManagerRemoveProvider:
    """Test removing providers"""

    @pytest.fixture
    async def manager_with_providers(self, tmp_path):
        """Create manager with multiple providers"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "llm": {
                "default_provider": "openai",
                "providers": [
                    {"name": "openai", "api_key": "key1", "models": ["gpt-4"]},
                    {"name": "anthropic", "api_key": "key2", "models": ["claude-3"]}
                ]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    async def test_remove_provider_success(self, manager_with_providers):
        """Test removing non-default provider"""
        await manager_with_providers.remove_provider("anthropic")

        providers = await manager_with_providers.get_value("llm.providers")
        assert len(providers) == 1
        assert providers[0]["name"] == "openai"

    async def test_remove_provider_default_raises(self, manager_with_providers):
        """Test removing default provider raises error"""
        with pytest.raises(ValueError) as exc_info:
            await manager_with_providers.remove_provider("openai")

        assert "cannot remove default" in str(exc_info.value).lower()

    async def test_remove_provider_nonexistent_raises(self, manager_with_providers):
        """Test removing nonexistent provider raises error"""
        with pytest.raises(ValueError) as exc_info:
            await manager_with_providers.remove_provider("nonexistent")

        assert "not found" in str(exc_info.value).lower()


# ============================================================================
# TEST CLASS: ConfigManager - List and Show Providers
# ============================================================================

class TestConfigManagerListShowProviders:
    """Test listing and showing providers"""

    @pytest.fixture
    async def manager_with_providers(self, tmp_path):
        """Create manager with providers"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "llm": {
                "default_provider": "openai",
                "providers": [
                    {"name": "openai", "api_key": "key1", "models": ["gpt-4"]},
                    {"name": "anthropic", "api_key": "key2", "models": ["claude-3"]}
                ]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager

    async def test_list_providers(self, manager_with_providers):
        """Test listing all providers"""
        providers = await manager_with_providers.list_providers()

        assert len(providers) == 2
        assert providers[0]["name"] == "openai"
        assert providers[1]["name"] == "anthropic"

    async def test_show_provider_success(self, manager_with_providers):
        """Test showing specific provider"""
        provider = await manager_with_providers.show_provider("openai")

        assert provider["name"] == "openai"
        assert provider["api_key"] == "key1"
        assert "models" in provider

    async def test_show_provider_nonexistent_raises(self, manager_with_providers):
        """Test showing nonexistent provider raises error"""
        with pytest.raises(ValueError) as exc_info:
            await manager_with_providers.show_provider("nonexistent")

        assert "not found" in str(exc_info.value).lower()


# ============================================================================
# TEST CLASS: ConfigManager - Save Config
# ============================================================================

class TestConfigManagerSaveConfig:
    """Test saving configuration"""

    @pytest.fixture
    async def manager_with_config(self, tmp_path):
        """Create manager with config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main"},
            "llm": {
                "default_provider": "openai",
                "providers": [{"name": "openai", "api_key": "key1", "models": ["gpt-4", "gpt-3.5"]}]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()
        return manager, config_file

    async def test_save_config_writes_file(self, manager_with_config):
        """Test save_config writes to file"""
        manager, config_file = manager_with_config

        # Modify config
        await manager.set_value("sonar.default_branch", "develop", validate=False)

        # Save
        manager.save_config()

        # Read and verify
        with open(config_file, "rb") as f:
            saved_config = tomli.load(f)

        assert saved_config["sonar"]["default_branch"] == "develop"

    async def test_save_config_with_backup(self, manager_with_config):
        """Test save_config creates backup when requested"""
        manager, config_file = manager_with_config

        # Save with backup
        manager.save_config(create_backup=True)

        # Check backup exists
        backups = list(config_file.parent.glob("*.backup_*.toml"))
        assert len(backups) == 1

    async def test_save_config_preserves_structure(self, manager_with_config):
        """Test save_config preserves nested structure"""
        manager, config_file = manager_with_config

        # Add nested structure
        await manager.set_value("sonar.options.clone_type", "branch", validate=False)

        # Save
        manager.save_config()

        # Read and verify
        with open(config_file, "rb") as f:
            saved_config = tomli.load(f)

        assert saved_config["sonar"]["options"]["clone_type"] == "branch"

    async def test_save_config_preserves_arrays(self, manager_with_config):
        """Test save_config preserves arrays correctly"""
        manager, config_file = manager_with_config

        # Save
        manager.save_config()

        # Read and verify
        with open(config_file, "rb") as f:
            saved_config = tomli.load(f)

        providers = saved_config["llm"]["providers"]
        assert len(providers) == 1
        assert isinstance(providers[0]["models"], list)
        assert "gpt-4" in providers[0]["models"]


# ============================================================================
# TEST CLASS: ConfigManager - Private Helper Methods
# ============================================================================

class TestConfigManagerPrivateMethods:
    """Test private helper methods"""

    @pytest.fixture
    def manager(self):
        """Create basic manager"""
        return ConfigManager()

    def test_build_section_value_non_dict(self, manager):
        """Test _build_section_value with non-dict value"""
        result = manager._build_section_value("simple_string")

        assert result == "simple_string"

    def test_build_section_value_dict(self, manager):
        """Test _build_section_value with dict"""
        result = manager._build_section_value({"key": "value"})

        assert isinstance(result, tomlkit.items.Table)

    def test_build_table_value_simple(self, manager):
        """Test _build_table_value with simple value"""
        result = manager._build_table_value("key", "value")

        assert result == "value"

    def test_build_table_value_nested_dict(self, manager):
        """Test _build_table_value with nested dict"""
        result = manager._build_table_value("key", {"nested": "value"})

        assert isinstance(result, tomlkit.items.Table)

    def test_build_table_value_providers_array(self, manager):
        """Test _build_table_value with providers array"""
        providers = [{"name": "openai", "models": ["gpt-4"]}]
        result = manager._build_table_value("providers", providers)

        assert isinstance(result, tomlkit.items.AoT)

    def test_is_provider_array_true(self, manager):
        """Test _is_provider_array returns True for non-empty list"""
        result = manager._is_provider_array([{"name": "openai"}])

        assert result is True

    def test_is_provider_array_false_empty(self, manager):
        """Test _is_provider_array returns False for empty list"""
        result = manager._is_provider_array([])

        assert result is False

    def test_is_provider_array_false_non_list(self, manager):
        """Test _is_provider_array returns False for non-list"""
        result = manager._is_provider_array("not a list")

        assert result is False

    def test_build_providers_array(self, manager):
        """Test _build_providers_array"""
        providers = [
            {"name": "openai", "models": ["gpt-4"]},
            {"name": "anthropic", "models": ["claude-3"]}
        ]
        result = manager._build_providers_array(providers)

        assert isinstance(result, tomlkit.items.AoT)
        assert len(result) == 2

    def test_build_provider_table(self, manager):
        """Test _build_provider_table"""
        provider = {"name": "openai", "api_key": "key", "models": ["gpt-4"]}
        result = manager._build_provider_table(provider)

        assert isinstance(result, tomlkit.items.Table)
        assert result["name"] == "openai"

    def test_build_inline_array(self, manager):
        """Test _build_inline_array"""
        items = ["item1", "item2", "item3"]
        result = manager._build_inline_array(items)

        assert isinstance(result, tomlkit.items.Array)
        assert len(result) == 3
        assert result[0] == "item1"

    def test_build_nested_table(self, manager):
        """Test _build_nested_table"""
        nested = {"key1": "value1", "key2": "value2"}
        result = manager._build_nested_table(nested)

        assert isinstance(result, tomlkit.items.Table)
        assert result["key1"] == "value1"


# ============================================================================
# TEST CLASS: Integration Tests
# ============================================================================

class TestConfigManagerIntegration:
    """Integration tests for complete workflows"""

    async def test_full_provider_management_workflow(self, tmp_path):
        """Test complete provider management workflow"""
        config_file = tmp_path / "config.toml"
        manager = ConfigManager(config_path=config_file)

        # Create default config
        manager.create_default_config()
        await manager.load_config()

        # Add first provider
        await manager.add_provider({
            "name": "openai",
            "api_key": "key1",
            "models": ["gpt-4"],
            "default_model": "gpt-4"
        }, set_as_default=True)

        # Add second provider
        await manager.add_provider({
            "name": "anthropic",
            "api_key": "key2",
            "models": ["claude-3"],
            "default_model": "claude-3"
        })

        # Update provider
        await manager.update_provider("openai", {"api_key": "updated_key"})

        # Save
        manager.save_config()

        # Reload and verify
        manager2 = ConfigManager(config_path=config_file)
        await manager2.load_config()

        providers = await manager2.list_providers()
        assert len(providers) == 2
        assert providers[0]["api_key"] == "updated_key"

    async def test_config_validation_workflow(self, tmp_path):
        """Test validation during config changes"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main"},
            "llm": {
                "default_provider": "openai",
                "providers": [{"name": "openai", "models": ["gpt-4"], "default_model": "gpt-4"}]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()

        # Valid change
        await manager.set_value("llm.default_provider", "openai", validate=True)

        # Invalid change should fail
        with pytest.raises(ValueError):
            await manager.set_value("llm.default_provider", "invalid", validate=True)

        # Config should be unchanged
        assert await manager.get_value("llm.default_provider") == "openai"

    async def test_backup_and_restore_workflow(self, tmp_path):
        """Test backup creation and restoration"""
        config_file = tmp_path / "config.toml"
        config_data = {"sonar": {"default_branch": "main"}}
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        await manager.load_config()

        # Create backup
        backup_path = manager.create_backup()
        assert backup_path is not None

        # Modify config
        await manager.set_value("sonar.default_branch", "develop", validate=False)
        manager.save_config()

        # Verify change
        assert await manager.get_value("sonar.default_branch") == "develop"

        # Restore from backup
        import shutil
        shutil.copy2(backup_path, config_file)

        # Reload and verify restoration
        manager2 = ConfigManager(config_path=config_file)
        await manager2.load_config()
        assert await manager2.get_value("sonar.default_branch") == "main"
