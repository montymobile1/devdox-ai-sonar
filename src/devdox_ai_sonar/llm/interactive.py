"""Interactive add / update flows for LLM profiles.

Both entry points walk a shared sequence of primitives (pick provider,
pick model, collect key, validate) wired together through
:mod:`selection`, :mod:`validation`, and :mod:`profile_store`.

add_profile_flow (the wizard)
-----------------------------
1. Build provider menu from the catalog adapters.
2. Ask the user to pick one.
3. Pick a model from that provider's model list.
4. Validate the resulting model string via
   :func:`validation.validate_model_string`.
5. Prompt for an API key (blank allowed).
6. Live-probe the credentials via :func:`validation.probe_api_key`.
7. Save as a new :class:`LLMProfile` (name prompted, default offered).
8. Loop: "Add another?"

update_profile_flow (Option B sequential prompts)
-------------------------------------------------
Walks every field in order ("Keep name?", "Change key?", "Keep model?",
"Keep base_url?", "Set as default?"), accumulates the requested
changes in a :class:`ProfileUpdateContext`, runs *one* live probe of
the combined final state, and commits atomically. Either all changes
land or none do.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console
from rich.prompt import Confirm, Prompt

from devdox_ai_sonar.llm import selection, validation
from devdox_ai_sonar.llm.errors import (
    InvalidModelFormatError,
    KeyProbeError,
    ProfileNotFoundError,
    UnknownProviderError,
)
from devdox_ai_sonar.llm.profile import LLMProfile
from devdox_ai_sonar.llm.profile_store import (
    get_default_profile,
    load_profiles,
    save_profile,
    update_profile,
)
from devdox_ai_sonar.models.llm_config import ConfigManager
from devdox_ai_sonar.utils.ui_prompts import select_from_list


__all__ = [
    "ProfileUpdateContext",
    "add_profile_flow",
    "update_profile_flow",
]


_console = Console()


@dataclass
class ProfileUpdateContext:
    """Pending state for an Option-B sequential update.

    Attributes:
        profile: The original profile loaded from disk.
        updates: Field -> new-value mapping; passed to
            :func:`profile_store.update_profile` at the end.
        new_api_key: Candidate API key (when the user chose to rotate
            it). Held separately from ``updates`` so it can participate
            in the live probe without being applied until probe success.
        set_as_default: Whether to mark the profile as default after
            successful probe + commit.
    """

    profile: LLMProfile
    updates: dict[str, Any] = field(default_factory=dict)
    new_api_key: Optional[str] = None
    set_as_default: bool = False

    def final_model(self) -> str:
        return self.updates.get("model", self.profile.model)

    def final_base_url(self) -> Optional[str]:
        if "base_url" in self.updates:
            return self.updates["base_url"]
        return self.profile.base_url

    def final_api_key(self) -> str:
        if self.new_api_key is not None:
            return self.new_api_key
        return self.profile.api_key


# ---------------------------------------------------------------------------
# add_profile_flow
# ---------------------------------------------------------------------------


async def add_profile_flow(manager: ConfigManager) -> None:
    """Run the add-profile wizard, looping on "Add another?"."""
    while True:
        profile = await _collect_and_validate_profile()
        if profile is None:
            _console.print("[yellow]⚠ Profile configuration cancelled[/yellow]")
            break

        set_default = _confirm_default()
        try:
            await save_profile(manager, profile, set_as_default=set_default)
        except Exception as exc:
            _console.print(f"[red]❌ Could not save profile: {exc}[/red]")
            break

        _console.print(f"[green]✓ Profile '{profile.name}' saved[/green]")

        if not _confirm_add_another():
            break


async def _collect_and_validate_profile() -> Optional[LLMProfile]:
    """Walk steps 1-6 of the wizard; return a ready-to-save profile or None."""
    picked_provider = await _prompt_for_provider()
    if picked_provider is None:
        return None

    model_id = await _prompt_for_model(picked_provider)
    if model_id is None:
        return None
    model = f"{picked_provider}/{model_id}"

    validated = _validate_or_report(model, base_url=None)
    if validated is None:
        return None

    api_key = _prompt_for_api_key(picked_provider)
    if not _probe_or_report(model, api_key, base_url=None):
        return None

    name = _prompt_profile_name(default=_suggest_profile_name(model))
    if name is None:
        return None

    return LLMProfile(name=name, model=model, api_key=api_key, base_url=None)


# ---------------------------------------------------------------------------
# update_profile_flow  (Option B -- sequential prompts)
# ---------------------------------------------------------------------------


async def update_profile_flow(manager: ConfigManager) -> None:
    """Walk every updatable field in order, probe once, commit atomically."""
    profiles = await load_profiles(manager)
    if not profiles:
        _console.print(
            "[yellow]⚠ No profiles configured; use add_provider first[/yellow]"
        )
        return

    target = await _prompt_pick_profile(profiles)
    if target is None:
        return

    current_default = await get_default_profile(manager)
    is_currently_default = (
        current_default is not None and current_default.name == target.name
    )

    ctx = ProfileUpdateContext(profile=target)

    _sequential_name_prompt(ctx, existing_names={p.name for p in profiles})
    _sequential_api_key_prompt(ctx)
    await _sequential_model_prompt(ctx)
    _sequential_base_url_prompt(ctx)
    _sequential_default_prompt(ctx, currently_default=is_currently_default)

    if not (ctx.updates or ctx.new_api_key is not None or ctx.set_as_default):
        _console.print("[yellow]⚠ No changes requested[/yellow]")
        return

    # Single live probe of the final combined state.
    if not _probe_or_report(
        ctx.final_model(),
        ctx.final_api_key(),
        ctx.final_base_url(),
    ):
        _console.print("[red]❌ Update cancelled; no changes written[/red]")
        return

    if ctx.new_api_key is not None:
        ctx.updates["api_key"] = ctx.new_api_key

    try:
        await update_profile(manager, target.name, ctx.updates)
        if ctx.set_as_default:
            await _set_default(manager, ctx.updates.get("name", target.name))
    except ProfileNotFoundError as exc:
        _console.print(f"[red]❌ {exc}[/red]")
        return

    _console.print("[green]✓ Profile updated[/green]")


async def _set_default(manager: ConfigManager, name: str) -> None:
    """Persist ``[llm].default_profile = name``."""
    llm_section = await manager.get_value("llm")
    if isinstance(llm_section, dict):
        llm_section["default_profile"] = name
        manager.save_config(create_backup=False)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


async def _prompt_for_provider() -> Optional[str]:
    """Step 1: provider picker. Returns the picked name or None on cancel."""
    menu = selection.build_provider_menu()
    names = [entry.name for entry in menu]
    picked = await select_from_list(names, "Select an LLM provider")
    return picked


async def _prompt_for_model(provider: str) -> Optional[str]:
    """Step 2: pick a model id from a known provider's catalog."""
    menu = selection.build_model_menu(provider)
    if not menu:
        _console.print(
            f"[red]❌ No models available for provider '{provider}'[/red]"
        )
        return None

    ids = [entry.model_id for entry in menu]
    picked = await select_from_list(ids, f"Select a {provider} model")
    return picked


def _prompt_for_api_key(family_label: str) -> str:
    """Collect an API key. Blank is allowed (endpoints that accept no auth)."""
    try:
        return Prompt.ask(
            f"API key for {family_label} (press Enter if not required)",
            default="",
            password=True,
        ).strip()
    except KeyboardInterrupt:
        return ""


def _confirm_default(
    message: str = "Set this as the default profile?",
) -> bool:
    try:
        return Confirm.ask(message, default=False)
    except KeyboardInterrupt:
        return False


def _confirm_add_another() -> bool:
    try:
        return Confirm.ask("Add another profile?", default=False)
    except KeyboardInterrupt:
        return False


def _prompt_profile_name(default: str) -> Optional[str]:
    try:
        return Prompt.ask("Profile name", default=default).strip() or None
    except KeyboardInterrupt:
        return None


def _suggest_profile_name(model: str) -> str:
    """Default profile name: the provider family (before the first '/')."""
    separator = model.find("/")
    return model[:separator] if separator != -1 else model


# ---------------------------------------------------------------------------
# Update-flow helpers (Option B)
# ---------------------------------------------------------------------------


async def _prompt_pick_profile(
    profiles: list[LLMProfile],
) -> Optional[LLMProfile]:
    names = [p.name for p in profiles]
    picked = await select_from_list(names, "Select a profile to update")
    if picked is None:
        return None
    return next((p for p in profiles if p.name == picked), None)


def _sequential_name_prompt(
    ctx: ProfileUpdateContext, *, existing_names: set[str]
) -> None:
    if Confirm.ask(f"Keep current name '{ctx.profile.name}'?", default=True):
        return
    new_name = Prompt.ask("New profile name").strip()
    if not new_name:
        _console.print("[yellow]⚠ Name unchanged (empty input)[/yellow]")
        return
    if new_name in existing_names - {ctx.profile.name}:
        _console.print(
            f"[red]❌ '{new_name}' already in use; keeping current name[/red]"
        )
        return
    ctx.updates["name"] = new_name


def _sequential_api_key_prompt(ctx: ProfileUpdateContext) -> None:
    if not Confirm.ask("Change the API key?", default=False):
        return
    ctx.new_api_key = Prompt.ask(
        "New API key (press Enter if not required)",
        default="",
        password=True,
    ).strip()


async def _sequential_model_prompt(ctx: ProfileUpdateContext) -> None:
    if Confirm.ask(
        f"Keep current model '{ctx.profile.model}'?", default=True
    ):
        return
    picked_provider = await _prompt_for_provider()
    if picked_provider is None:
        return
    model_id = await _prompt_for_model(picked_provider)
    if model_id is None:
        return
    ctx.updates["model"] = f"{picked_provider}/{model_id}"
    ctx.updates["base_url"] = None


def _sequential_base_url_prompt(ctx: ProfileUpdateContext) -> None:
    if "base_url" in ctx.updates:
        return
    current = ctx.profile.base_url or "(none)"
    if Confirm.ask(f"Keep current base URL '{current}'?", default=True):
        return
    new_base_url = Prompt.ask(
        "New base URL (press Enter to clear)", default=""
    ).strip()
    ctx.updates["base_url"] = new_base_url or None


def _sequential_default_prompt(
    ctx: ProfileUpdateContext, *, currently_default: bool
) -> None:
    if currently_default:
        return
    if Confirm.ask("Set this as the default profile?", default=False):
        ctx.set_as_default = True


# ---------------------------------------------------------------------------
# Validation + probe helpers shared by both flows
# ---------------------------------------------------------------------------


def _validate_or_report(
    model: str, base_url: Optional[str]
) -> Optional[validation.ValidatedModel]:
    result = validation.validate_model_string(model, base_url)
    if result.is_ok():
        return result.unwrap()
    err = result.error
    if isinstance(err, InvalidModelFormatError):
        _console.print(f"[red]❌ Invalid model format: {err.reason}[/red]")
    elif isinstance(err, UnknownProviderError):
        _console.print(
            f"[red]❌ Unknown provider prefix in '{err.model}'. "
            "Pick a provider from the menu.[/red]"
        )
    else:
        _console.print(f"[red]❌ Validation failed: {err}[/red]")
    return None


def _probe_or_report(model: str, api_key: str, base_url: Optional[str]) -> bool:
    _console.print("[cyan]Validating credentials...[/cyan]")
    result = validation.probe_api_key(model, api_key, base_url)
    if result.is_ok():
        _console.print("[green]✓ Credentials accepted[/green]")
        return True

    err = result.error
    assert isinstance(err, KeyProbeError)
    if err.kind == "auth":
        _console.print("[red]❌ Key rejected by the provider[/red]")
    elif err.kind == "connection":
        _console.print(
            "[red]❌ Could not reach endpoint (check base_url / network)[/red]"
        )
    elif err.kind == "bad_request":
        _console.print(
            "[red]❌ Endpoint rejected the request (model may be invalid)[/red]"
        )
    elif err.kind == "rate_limit":
        _console.print(
            "[red]❌ Endpoint rate-limited the probe; try again shortly[/red]"
        )
    else:
        _console.print(f"[red]❌ Probe failed: {err.detail}[/red]")
    return False
