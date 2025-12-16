

import pytest
import tomli
import tomli_w
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, mock_open
from devdox_ai_sonar.models.llm_config import AppConfig, ConfigManager


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

    def test_load_config_success(self, valid_config_file):
        """Test loading valid config"""
        manager = ConfigManager(config_path=valid_config_file)

        config = manager.load_config()

        assert config is not None
        assert "sonar" in config
        assert "llm" in config
        assert manager.config == config

    def test_load_config_nonexistent_file(self, tmp_path):
        """Test loading nonexistent config raises error"""
        nonexistent = tmp_path / "nonexistent.toml"
        manager = ConfigManager(config_path=nonexistent)

        with pytest.raises(FileNotFoundError) as exc_info:
            manager.load_config()

        assert "not found" in str(exc_info.value).lower()

    def test_load_config_invalid_toml(self, tmp_path):
        """Test loading invalid TOML raises error"""
        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("invalid toml content {{{")

        manager = ConfigManager(config_path=invalid_file)

        with pytest.raises(Exception):  # tomli.TOMLDecodeError
            manager.load_config()

    def test_load_config_stores_in_manager(self, valid_config_file):
        """Test loaded config is stored in manager"""
        manager = ConfigManager(config_path=valid_config_file)

        config = manager.load_config()

        assert manager.config is not None
        assert manager.config == config


# ============================================================================
# TEST CLASS: ConfigManager - Get Value
# ============================================================================

class TestConfigManagerGetValue:
    """Test getting config values"""

    @pytest.fixture
    def manager_with_config(self, tmp_path):
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
        manager.load_config()
        return manager

    def test_get_value_top_level(self, manager_with_config):
        """Test getting top-level value"""
        value = manager_with_config.get_value("sonar")

        assert value is not None
        assert "default_branch" in value

    def test_get_value_nested(self, manager_with_config):
        """Test getting nested value"""
        value = manager_with_config.get_value("sonar.default_branch")

        assert value == "main"

    def test_get_value_deeply_nested(self, manager_with_config):
        """Test getting deeply nested value"""
        value = manager_with_config.get_value("sonar.options.clone_type")

        assert value == "branch"

    def test_get_value_list(self, manager_with_config):
        """Test getting list value"""
        value = manager_with_config.get_value("llm.providers")

        assert isinstance(value, list)
        assert len(value) == 2

    def test_get_value_nonexistent_key(self, manager_with_config):
        """Test getting nonexistent key returns None"""
        value = manager_with_config.get_value("nonexistent.key")

        assert value is None

    def test_get_value_loads_config_if_needed(self, tmp_path):
        """Test get_value loads config if not already loaded"""
        config_file = tmp_path / "config.toml"
        config_data = {"test": {"value": 42}}
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        # Don't call load_config manually

        value = manager.get_value("test.value")

        assert value == 42
        assert manager.config is not None

    def test_get_value_traverse_non_dict_raises(self, manager_with_config):
        """Test traversing non-dict raises KeyError"""
        with pytest.raises(KeyError) as exc_info:
            # Try to traverse through a string value
            manager_with_config.get_value("sonar.default_branch.invalid")

        assert "non-dict" in str(exc_info.value).lower()


# ============================================================================
# TEST CLASS: ConfigManager - Set Value
# ============================================================================

class TestConfigManagerSetValue:
    """Test setting config values"""

    @pytest.fixture
    def manager_with_config(self, tmp_path):
        """Create manager with loaded config"""
        config_file = tmp_path / "config.toml"
        config_data = {
            "sonar": {"default_branch": "main"},
            "llm": {
                "default_provider": "openai",
                "providers": [{"name": "openai"}]
            }
        }
        with open(config_file, "wb") as f:
            tomli_w.dump(config_data, f)

        manager = ConfigManager(config_path=config_file)
        manager.load_config()
        return manager

    def test_set_value_existing_key(self, manager_with_config):
        """Test setting existing key"""
        manager_with_config.set_value("sonar.default_branch", "develop", validate=False)

        value = manager_with_config.get_value("sonar.default_branch")
        assert value == "develop"

    def test_set_value_new_key(self, manager_with_config):
        """Test setting new key"""
        manager_with_config.set_value("sonar.new_option", "value", validate=False)

        value = manager_with_config.get_value("sonar.new_option")
        assert value == "value"

    def test_set_value_creates_nested_structure(self, manager_with_config):
        """Test setting value creates nested dicts as needed"""
        manager_with_config.set_value("new.nested.key", 123, validate=False)

        value = manager_with_config.get_value("new.nested.key")
        assert value == 123

    def test_set_value_with_validation_success(self, manager_with_config):
        """Test setting value with successful validation"""
        # This would need actual validation logic
        manager_with_config.set_value("llm.default_provider", "openai", validate=True)

        value = manager_with_config.get_value("llm.default_provider")
        assert value == "openai"

    def test_set_value_validation_reverts_on_failure(self, manager_with_config):
        """Test failed validation reverts change"""
        original = manager_with_config.get_value("llm.default_provider")

        with pytest.raises(ValueError):
            # Try to set invalid provider
            manager_with_config.set_value("llm.default_provider", "nonexistent", validate=True)

        # Value should be unchanged
        current = manager_with_config.get_value("llm.default_provider")
        assert current == original


