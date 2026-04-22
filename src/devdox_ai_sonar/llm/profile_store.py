"""CRUD operations on the ``[[llm.profiles]]`` section of config.toml.

This module is the only code that reads or writes the new profile shape
to disk. It sits on top of the existing
:class:`devdox_ai_sonar.models.llm_config.ConfigManager` (which handles
the raw TOML load/save), translating between dict entries and
:class:`LLMProfile` instances and enforcing the schema invariants
(unique names, default_profile consistency).

Old-shape detection
-------------------
If an older ``[[llm.providers]]`` array is encountered, every CRUD entry
point raises :class:`ConfigStorageError` with a recovery hint. The shape
isn't automatically migrated; the user is asked to delete the file and
reconfigure.
"""

from __future__ import annotations

from typing import Any

from devdox_ai_sonar.llm.errors import ConfigStorageError, ProfileNotFoundError
from devdox_ai_sonar.llm.profile import LLMProfile
from devdox_ai_sonar.models.llm_config import ConfigManager


__all__ = [
    "load_profiles",
    "save_profile",
    "update_profile",
    "delete_profile",
    "get_default_profile",
]


_LLM_SECTION = "llm"
_PROFILES_KEY = "profiles"
_DEFAULT_PROFILE_KEY = "default_profile"
_LEGACY_PROVIDERS_KEY = "providers"  # Old [[llm.providers]] shape.
_OLD_SHAPE_MESSAGE = (
    "config.toml uses the old '[[llm.providers]]' schema which is no "
    "longer supported. Delete ~/devdox/config.toml and run "
    "'devdox_sonar' to reconfigure."
)


async def load_profiles(manager: ConfigManager) -> list[LLMProfile]:
    """Read all configured profiles from disk.

    Args:
        manager: Live :class:`ConfigManager` with its config loaded (the
            function will lazily load it if not yet loaded, via
            :meth:`ConfigManager.get_value`).

    Raises:
        ConfigStorageError: If the config file uses the old
            ``[[llm.providers]]`` shape.

    Returns:
        A list of :class:`LLMProfile`, possibly empty.
    """
    llm_section = await _get_llm_section(manager)
    _reject_legacy_shape(llm_section)

    raw_profiles = llm_section.get(_PROFILES_KEY, [])
    if not isinstance(raw_profiles, list):
        raise ConfigStorageError(
            f"Expected list at 'llm.{_PROFILES_KEY}', got {type(raw_profiles).__name__}"
        )

    return [LLMProfile.from_toml_dict(entry) for entry in raw_profiles]


async def save_profile(
    manager: ConfigManager,
    profile: LLMProfile,
    *,
    set_as_default: bool,
) -> None:
    """Append a new profile to the ``[[llm.profiles]]`` array and persist.

    Args:
        manager: Live :class:`ConfigManager`.
        profile: The profile to add. Its ``name`` must not collide with
            an existing profile's name.
        set_as_default: When ``True``, also set
            ``[llm].default_profile = profile.name``.

    Raises:
        ConfigStorageError: If a profile with the same name already
            exists, or if the config file uses the old shape.
    """
    llm_section = await _get_llm_section(manager)
    _reject_legacy_shape(llm_section)

    existing = llm_section.get(_PROFILES_KEY, [])
    if any(entry.get("name") == profile.name for entry in existing):
        raise ConfigStorageError(
            f"Profile named {profile.name!r} already exists"
        )

    existing.append(profile.to_toml_dict())
    llm_section[_PROFILES_KEY] = existing

    if set_as_default:
        llm_section[_DEFAULT_PROFILE_KEY] = profile.name

    manager.save_config(create_backup=False)


async def update_profile(
    manager: ConfigManager,
    name: str,
    updates: dict[str, Any],
) -> None:
    """Apply field-level updates to an existing profile.

    ``updates`` is a dict of profile fields to overwrite. When the ``name``
    field appears in ``updates``, the profile is renamed; if it was the
    default, ``[llm].default_profile`` is updated to track the rename.

    Args:
        manager: Live :class:`ConfigManager`.
        name: Existing profile name to locate.
        updates: Mapping of field name to new value. Unrecognised keys
            are written through unchanged (TOML tolerates extras; user
            trust is fine here).

    Raises:
        ProfileNotFoundError: If no profile with ``name`` is found.
        ConfigStorageError: If the rename collides with another existing
            profile's name, or if the config file uses the old shape.
    """
    llm_section = await _get_llm_section(manager)
    _reject_legacy_shape(llm_section)

    profiles = llm_section.get(_PROFILES_KEY, [])
    if not isinstance(profiles, list):
        raise ConfigStorageError(
            f"Expected list at 'llm.{_PROFILES_KEY}', got {type(profiles).__name__}"
        )

    index = _find_profile_index(profiles, name)
    if index is None:
        raise ProfileNotFoundError(f"Profile {name!r} not found")

    new_name = updates.get("name", name)
    if new_name != name and _find_profile_index(profiles, new_name) is not None:
        raise ConfigStorageError(
            f"Cannot rename {name!r} -> {new_name!r}: name already in use"
        )

    # Apply updates. Optional fields (base_url) with value None are
    # removed from the TOML entry rather than written as null, because
    # tomlkit has no representation for None and an empty base_url means
    # "use the provider default" -- i.e. the field's absence is its
    # semantically correct state.
    entry = profiles[index]
    for key, value in updates.items():
        if key == "base_url" and value is None:
            entry.pop("base_url", None)
        else:
            entry[key] = value

    # Keep default_profile consistent on rename.
    current_default = llm_section.get(_DEFAULT_PROFILE_KEY)
    if current_default == name and new_name != name:
        llm_section[_DEFAULT_PROFILE_KEY] = new_name

    manager.save_config(create_backup=False)


async def delete_profile(manager: ConfigManager, name: str) -> None:
    """Remove a profile from the ``[[llm.profiles]]`` array.

    If the deleted profile was the default, ``[llm].default_profile``
    is cleared (set to empty string).

    Raises:
        ProfileNotFoundError: If no profile with ``name`` is found.
        ConfigStorageError: If the config file uses the old shape.
    """
    llm_section = await _get_llm_section(manager)
    _reject_legacy_shape(llm_section)

    profiles = llm_section.get(_PROFILES_KEY, [])
    index = _find_profile_index(profiles, name)
    if index is None:
        raise ProfileNotFoundError(f"Profile {name!r} not found")

    del profiles[index]

    if llm_section.get(_DEFAULT_PROFILE_KEY) == name:
        llm_section[_DEFAULT_PROFILE_KEY] = ""

    manager.save_config(create_backup=False)


async def get_default_profile(manager: ConfigManager) -> LLMProfile | None:
    """Return the profile currently marked as default, if any.

    Returns ``None`` when no default is set or when the name pointed to
    by ``default_profile`` no longer matches a real entry (a mild
    inconsistency we tolerate by treating it as "no default").
    """
    llm_section = await _get_llm_section(manager)
    _reject_legacy_shape(llm_section)

    default_name = llm_section.get(_DEFAULT_PROFILE_KEY, "")
    if not default_name:
        return None

    profiles = llm_section.get(_PROFILES_KEY, [])
    for entry in profiles:
        if entry.get("name") == default_name:
            return LLMProfile.from_toml_dict(entry)

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_llm_section(manager: ConfigManager) -> dict[str, Any]:
    """Return the mutable ``[llm]`` table, creating it if absent.

    Mutating the returned dict updates the in-memory config held by
    ``manager``; call ``manager.save_config()`` to persist.
    """
    section = await manager.get_value(_LLM_SECTION)
    if section is None:
        # No [llm] section at all; create one so callers can write.
        if manager.config is None:
            # Should be impossible after get_value, but be explicit.
            raise ConfigStorageError("Config not loaded")
        manager.config[_LLM_SECTION] = {}
        section = manager.config[_LLM_SECTION]
    if not isinstance(section, dict):
        raise ConfigStorageError(
            f"Expected dict at 'llm', got {type(section).__name__}"
        )
    return section


def _reject_legacy_shape(llm_section: dict[str, Any]) -> None:
    """Raise if the ``[[llm.providers]]`` array is present.

    The new schema uses ``[[llm.profiles]]``; if the reader encounters
    ``providers`` instead, the user is on an older config file that the
    new code cannot interpret. We fail loudly with a recovery hint
    rather than trying to guess their intent.
    """
    if _LEGACY_PROVIDERS_KEY in llm_section:
        raise ConfigStorageError(_OLD_SHAPE_MESSAGE)


def _find_profile_index(profiles: list[dict[str, Any]], name: str) -> int | None:
    for i, entry in enumerate(profiles):
        if entry.get("name") == name:
            return i
    return None
