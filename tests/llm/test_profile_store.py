"""Tests for llm/profile_store.py.

Every test uses a tmp-path-backed ``ConfigManager`` so TOML round-trips
hit a real file. The tests verify the full cycle: write profile(s),
reload into a fresh manager, read back.
"""

from pathlib import Path

import pytest

from devdox_ai_sonar.llm.errors import ConfigStorageError, ProfileNotFoundError
from devdox_ai_sonar.llm.profile import LLMProfile
from devdox_ai_sonar.llm.profile_store import (
    delete_profile,
    get_default_profile,
    load_profiles,
    save_profile,
    update_profile,
)
from devdox_ai_sonar.models.llm_config import ConfigManager


@pytest.fixture
def fresh_manager(tmp_path: Path) -> ConfigManager:
    """A ConfigManager pointed at a tmp config.toml with the default schema."""
    config_path = tmp_path / "config.toml"
    manager = ConfigManager(config_path=config_path)
    manager.create_default_config()
    return manager


@pytest.fixture
def legacy_manager(tmp_path: Path) -> ConfigManager:
    """A ConfigManager whose config.toml contains the old [[llm.providers]]
    shape that we refuse to migrate."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[llm]\n'
        'default_provider = "openai"\n'
        '\n'
        '[[llm.providers]]\n'
        'name = "openai"\n'
        'api_key = "sk-old"\n'
        'models = ["gpt-4"]\n'
        'default_model = "gpt-4"\n'
    )
    return ConfigManager(config_path=config_path)


# ---------------------------------------------------------------------------
# Happy-path CRUD
# ---------------------------------------------------------------------------


async def test_load_profiles_empty_config_returns_empty(fresh_manager):
    assert await load_profiles(fresh_manager) == []


async def test_save_then_load_round_trips(fresh_manager):
    profile = LLMProfile(name="prod", model="openai/gpt-4o", api_key="sk-x")
    await save_profile(fresh_manager, profile, set_as_default=False)

    # Reload from disk in a fresh manager instance.
    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    profiles = await load_profiles(reloaded)
    assert profiles == [profile]


async def test_save_with_set_as_default_records_default(fresh_manager):
    profile = LLMProfile(name="prod", model="openai/gpt-4o", api_key="sk-x")
    await save_profile(fresh_manager, profile, set_as_default=True)

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    default = await get_default_profile(reloaded)
    assert default == profile


async def test_save_without_set_as_default_leaves_default_empty(fresh_manager):
    profile = LLMProfile(name="p", model="openai/gpt-4o")
    await save_profile(fresh_manager, profile, set_as_default=False)

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    assert await get_default_profile(reloaded) is None


async def test_save_rejects_duplicate_name(fresh_manager):
    profile = LLMProfile(name="prod", model="openai/gpt-4o")
    await save_profile(fresh_manager, profile, set_as_default=False)

    with pytest.raises(ConfigStorageError, match="already exists"):
        await save_profile(fresh_manager, profile, set_as_default=False)


async def test_save_supports_multiple_profiles(fresh_manager):
    profile_a = LLMProfile(name="a", model="openai/gpt-4o", api_key="sk-a")
    profile_b = LLMProfile(name="b", model="gemini/gemini-2.5-flash", api_key="g-b")
    await save_profile(fresh_manager, profile_a, set_as_default=True)
    await save_profile(fresh_manager, profile_b, set_as_default=False)

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    profiles = await load_profiles(reloaded)
    assert {p.name for p in profiles} == {"a", "b"}
    default = await get_default_profile(reloaded)
    assert default == profile_a


async def test_save_preserves_base_url_round_trip(fresh_manager):
    profile = LLMProfile(
        name="vllm",
        model="openai/my-model",
        api_key="",
        base_url="https://vllm.example.com",
    )
    await save_profile(fresh_manager, profile, set_as_default=False)

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    loaded = await load_profiles(reloaded)
    assert loaded == [profile]


# ---------------------------------------------------------------------------
# update_profile
# ---------------------------------------------------------------------------


async def test_update_profile_changes_api_key(fresh_manager):
    await save_profile(
        fresh_manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="sk-old"),
        set_as_default=False,
    )
    await update_profile(fresh_manager, "p", {"api_key": "sk-new"})

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    [loaded] = await load_profiles(reloaded)
    assert loaded.api_key == "sk-new"


async def test_update_profile_rename_follows_default(fresh_manager):
    await save_profile(
        fresh_manager,
        LLMProfile(name="old", model="openai/gpt-4o"),
        set_as_default=True,
    )
    await update_profile(fresh_manager, "old", {"name": "new"})

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    default = await get_default_profile(reloaded)
    assert default is not None
    assert default.name == "new"


async def test_update_profile_rename_collision_raises(fresh_manager):
    await save_profile(
        fresh_manager,
        LLMProfile(name="a", model="openai/gpt-4o"),
        set_as_default=False,
    )
    await save_profile(
        fresh_manager,
        LLMProfile(name="b", model="openai/gpt-4o-mini"),
        set_as_default=False,
    )
    with pytest.raises(ConfigStorageError, match="already in use"):
        await update_profile(fresh_manager, "a", {"name": "b"})


async def test_update_profile_unknown_name_raises(fresh_manager):
    with pytest.raises(ProfileNotFoundError):
        await update_profile(fresh_manager, "nope", {"api_key": "x"})


# ---------------------------------------------------------------------------
# delete_profile
# ---------------------------------------------------------------------------


async def test_delete_profile_removes_entry(fresh_manager):
    await save_profile(
        fresh_manager,
        LLMProfile(name="p", model="openai/gpt-4o"),
        set_as_default=False,
    )
    await delete_profile(fresh_manager, "p")

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    assert await load_profiles(reloaded) == []


async def test_delete_default_profile_clears_default(fresh_manager):
    await save_profile(
        fresh_manager,
        LLMProfile(name="p", model="openai/gpt-4o"),
        set_as_default=True,
    )
    await delete_profile(fresh_manager, "p")

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    assert await get_default_profile(reloaded) is None


async def test_delete_non_default_profile_leaves_default_intact(fresh_manager):
    await save_profile(
        fresh_manager,
        LLMProfile(name="keeper", model="openai/gpt-4o"),
        set_as_default=True,
    )
    await save_profile(
        fresh_manager,
        LLMProfile(name="throwaway", model="gemini/gemini-2.5-flash"),
        set_as_default=False,
    )
    await delete_profile(fresh_manager, "throwaway")

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    default = await get_default_profile(reloaded)
    assert default is not None
    assert default.name == "keeper"


async def test_delete_unknown_profile_raises(fresh_manager):
    with pytest.raises(ProfileNotFoundError):
        await delete_profile(fresh_manager, "nope")


# ---------------------------------------------------------------------------
# Legacy shape rejection
# ---------------------------------------------------------------------------


async def test_load_profiles_rejects_old_providers_shape(legacy_manager):
    with pytest.raises(ConfigStorageError, match="old provider shape"):
        await load_profiles(legacy_manager)


async def test_save_profile_rejects_old_providers_shape(legacy_manager):
    with pytest.raises(ConfigStorageError, match="old provider shape"):
        await save_profile(
            legacy_manager,
            LLMProfile(name="new", model="openai/gpt-4o"),
            set_as_default=False,
        )


async def test_update_profile_rejects_old_providers_shape(legacy_manager):
    with pytest.raises(ConfigStorageError, match="old provider shape"):
        await update_profile(legacy_manager, "anything", {"api_key": "x"})


async def test_delete_profile_rejects_old_providers_shape(legacy_manager):
    with pytest.raises(ConfigStorageError, match="old provider shape"):
        await delete_profile(legacy_manager, "anything")


async def test_get_default_profile_rejects_old_providers_shape(legacy_manager):
    with pytest.raises(ConfigStorageError, match="old provider shape"):
        await get_default_profile(legacy_manager)


# ---------------------------------------------------------------------------
# get_default_profile edge cases
# ---------------------------------------------------------------------------


async def test_get_default_profile_returns_none_when_name_points_nowhere(
    fresh_manager,
):
    """If default_profile points at a name that isn't in the profiles
    list (e.g. after a manual edit), treat it as 'no default' rather
    than raising. That keeps the UI flow recoverable."""
    # Seed a default_profile pointing at a missing name.
    llm_section = await fresh_manager.get_value("llm")
    llm_section["default_profile"] = "ghost"
    fresh_manager.save_config(create_backup=False)

    reloaded = ConfigManager(config_path=fresh_manager.config_path)
    assert await get_default_profile(reloaded) is None
