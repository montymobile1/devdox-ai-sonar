"""Interactive add / update flows for LLM profiles.

Both entry points walk a shared sequence of primitives (pick provider,
pick model, collect key, validate) wired together through
:mod:`selection`, :mod:`validation`, and :mod:`profile_store`.

add_profile_flow (the wizard)
-----------------------------
1. Build provider menu from the catalog adapters.
2. Ask the user to pick one (or the ``"🔧 Custom"`` escape hatch for
   self-hosted / non-curated endpoints).
3. If not Custom: pick a model from that provider's model list.
   If Custom: prompt for a fully-qualified ``provider/model`` string
   and an optional ``base_url``.
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
from typing import Any, Literal, Optional

from rich.console import Console
from rich.prompt import Confirm, Prompt

from devdox_ai_sonar.llm import exclusions, selection, validation
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
    "offer_legacy_llm_recovery",
    "update_profile_flow",
]


_console = Console()

# UI label for the "type your own model string" escape hatch and the
# sentinel returned when the user picks it. The sentinel is a
# synthetic-looking string so it cannot collide with a real litellm
# provider name.
_CUSTOM_LABEL = "🔧 Custom"
_CUSTOM_SENTINEL = "__custom__"
_VERIFIED_MARK = "⭐ "

# Mode-picker (Step 0) labels. The wizard's top-level choice is now
# "curated catalogue" vs "custom endpoint"; before Step 0 existed, the
# 🔧 Custom option was the last entry of the provider picker and easy
# to miss.
_MODE_CURATED_LABEL = "📋 Pick from curated list"
_MODE_CUSTOM_LABEL = "🔧 Custom endpoint"
_MODE_CURATED = "curated"
_MODE_CUSTOM = "custom"

# Selecting ``_BACK_LABEL`` in a menu (or typing ``back`` at a text
# prompt inside the custom flow) pops one level up the wizard.
_BACK_LABEL = "← Back"
_BACK_SENTINEL = "__back__"
_VERIFIED_LEGEND = (
    "[dim]⭐ = verified by OpenHands "
    "(tested and known to work well for code fixes)[/dim]"
)

# Per-invocation flag: the ⭐ legend is educational, not a repeating
# status indicator. Show it the first time a picker with stars opens,
# then stay quiet for the rest of that add/update flow. Reset to False
# at the top of each top-level wizard entry point so the NEXT
# invocation explains the symbol to a user who might be new.
_legend_shown_this_run: bool = False


def _show_verified_legend_once(menu_has_stars: bool) -> None:
    """Print the ⭐ legend at most once per wizard invocation.

    ``menu_has_stars`` short-circuits: no stars on screen means no
    symbol to explain, so we stay silent regardless of state.
    """
    global _legend_shown_this_run
    if not menu_has_stars or _legend_shown_this_run:
        return
    _console.print(_VERIFIED_LEGEND)
    _legend_shown_this_run = True

# Maximum credential-shaped (auth / unknown / connection) probe failures
# tolerated in a single API-key prompt loop before the wizard gives up
# and asks the user to verify their credentials. Bounded so a confused
# user, a misbehaving auth endpoint, or a non-interactive run cannot
# keep the wizard stuck forever.
_MAX_API_KEY_ATTEMPTS = 3

# Process-wide toggle set by the CLI entry point via :func:`set_verbose`.
# When true, probe-failure details (litellm exception class, provider
# identifier, HTTP status, raw detail) are printed under the failure
# headline. When false (the default) those are suppressed -- the
# headline alone is enough information for a regular user.
_VERBOSE: bool = False


def set_verbose(flag: bool) -> None:
    """Toggle whether probe-failure provenance is shown to the user.

    Called once from the CLI entry point (mirroring the ``--verbose`` /
    ``-v`` flag). Left as a process-wide mutable rather than threaded
    through every function because it's a display concern that touches
    deep helpers; tests can still flip it via ``monkeypatch.setattr``.
    """
    global _VERBOSE
    _VERBOSE = flag


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
# Legacy [[llm.providers]] recovery
# ---------------------------------------------------------------------------


async def offer_legacy_llm_recovery(manager: ConfigManager) -> bool:
    """Detect old ``[[llm.providers]]`` and offer to remove it in place.

    Reads ``manager.config['llm']`` directly (not via profile_store, which
    would raise on the legacy shape) and, if the old key is present,
    prompts the user. On consent, removes only the ``providers`` key
    from the ``[llm]`` table and persists. All other config — SonarCloud
    auth, project paths, anything outside ``[llm]`` — is left untouched.

    Returns:
        True if the caller should continue (no legacy detected, or
        recovery completed). False if the user declined recovery.
    """
    llm_section = (manager.config or {}).get("llm") or {}
    if "providers" not in llm_section:
        return True

    _console.print(
        "[yellow]Old [[llm.providers]] schema detected in config.toml. "
        "The [llm] section needs to be reset; your other settings will "
        "be preserved.[/yellow]"
    )
    if not Confirm.ask("Reset just the [llm] section now?", default=True):
        return False

    del manager.config["llm"]["providers"]
    manager.save_config(create_backup=False)
    _console.print("[green]✓ [llm] section reset.[/green]")
    return True


# ---------------------------------------------------------------------------
# add_profile_flow
# ---------------------------------------------------------------------------


async def add_profile_flow(manager: ConfigManager) -> None:
    """Run the add-profile wizard, looping on "Add another?"."""
    global _legend_shown_this_run
    _legend_shown_this_run = False
    while True:
        # Re-read existing names on every iteration -- a just-saved
        # profile in the previous pass becomes a duplicate for the
        # next one.
        existing_names = frozenset(
            p.name for p in await load_profiles(manager)
        )
        attempt = await _collect_and_validate_profile(
            existing_names=existing_names
        )
        if attempt.kind == "cancelled":
            _console.print("[yellow]⚠ Profile configuration cancelled[/yellow]")
            break
        if attempt.kind == "abandoned":
            # User explicitly abandoned this attempt (rate-limit "c");
            # offer "add another?" so they can pick a different
            # provider without being dumped back to the CLI root.
            if not _confirm_add_another():
                break
            continue

        profile = attempt.profile
        assert profile is not None  # kind == "created" implies profile

        set_default = _confirm_default()
        try:
            await save_profile(manager, profile, set_as_default=set_default)
        except Exception as exc:
            _console.print(f"[red]❌ Could not save profile: {exc}[/red]")
            break

        _console.print(f"[green]✓ Profile '{profile.name}' saved[/green]")

        if not _confirm_add_another():
            break


@dataclass(frozen=True)
class _AttemptResult:
    """Outcome of one walk through the add-profile wizard.

    * ``kind == "created"`` -- a fully-validated :class:`LLMProfile`
      is in ``profile``; caller persists it.
    * ``kind == "cancelled"`` -- hard abort (Ctrl+C, empty picker,
      etc.); caller exits the wizard.
    * ``kind == "abandoned"`` -- soft abort (rate-limit "c"); caller
      should skip the save but still offer "add another?".
    """

    kind: Literal["created", "cancelled", "abandoned"]
    profile: Optional[LLMProfile] = None


async def _collect_and_validate_profile(
    existing_names: frozenset[str] = frozenset(),
) -> _AttemptResult:
    """Walk the add-profile wizard, handling back-nav and restart-on-bad-model.

    The outer ``while True`` exists so a model-shaped probe failure
    (the endpoint doesn't serve the picked model) can send the user
    back to the mode/provider/model pickers instead of trapping them
    on the API-key prompt. Normal success, explicit cancel, and
    explicit abandon all break out immediately; the caller
    distinguishes them via :class:`_AttemptResult.kind`.
    """
    while True:
        gathered = await _gather_model_and_base_url()
        if gathered is None:
            return _AttemptResult(kind="cancelled")
        model, base_url, family_label = gathered

        validated = _validate_or_report(model, base_url)
        if validated is None:
            return _AttemptResult(kind="cancelled")

        result = _prompt_for_api_key_until_valid(family_label, model, base_url)
        if result.kind == "cancelled":
            return _AttemptResult(kind="cancelled")
        if result.kind == "abandoned":
            return _AttemptResult(kind="abandoned")
        if result.kind == "restart_model_selection":
            continue
        # result.kind == "validated"
        api_key = result.key or ""

        name = _prompt_profile_name(
            default=_suggest_profile_name(model),
            existing_names=existing_names,
        )
        if name is None:
            return _AttemptResult(kind="cancelled")

        return _AttemptResult(
            kind="created",
            profile=LLMProfile(
                name=name, model=model, api_key=api_key, base_url=base_url
            ),
        )


# ---------------------------------------------------------------------------
# update_profile_flow  (Option B -- sequential prompts)
# ---------------------------------------------------------------------------


async def update_profile_flow(manager: ConfigManager) -> None:
    """Walk every updatable field in order, probe once, commit atomically."""
    global _legend_shown_this_run
    _legend_shown_this_run = False
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
    await _sequential_model_prompt(
        ctx, used_models=frozenset(p.model for p in profiles)
    )
    _sequential_base_url_prompt(ctx)
    _sequential_default_prompt(ctx, currently_default=is_currently_default)

    if not (ctx.updates or ctx.new_api_key is not None or ctx.set_as_default):
        _console.print("[yellow]⚠ No changes requested[/yellow]")
        return

    # Single live probe of the final combined state. Option-B updates
    # intentionally do NOT enter the retry / save-on-rate-limit loop --
    # that belongs to the add-profile flow. Here, any failure aborts
    # the update and leaves the persisted profile untouched.
    outcome = _probe_and_handle(
        ctx.final_model(),
        ctx.final_api_key(),
        ctx.final_base_url(),
    )
    if outcome != "accepted":
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


async def _gather_model_and_base_url(
    used_models: frozenset[str] = frozenset(),
) -> Optional[tuple[str, Optional[str], str]]:
    """Run Step 0 (mode) + Step 1/2 (selection) with single-level back.

    Returns ``(full_model_string, base_url, family_label)`` or ``None``
    if the user cancelled at any point. ``family_label`` is what the
    subsequent API-key prompt shows the user ("openai", "custom
    endpoint", etc.).

    ``used_models`` is the set of full ``provider/model-id`` strings
    already pinned by saved profiles. Matching entries in the model
    picker are labelled ``(current)`` and cannot be selected. The add
    flow passes the default empty set (no blocking); the update flow
    passes every profile's model so the user can't land on a duplicate.

    Back navigation is strictly one level at a time:

    * Back from the provider picker  -> mode picker
    * Back from the model picker     -> provider picker
    * Back from the custom-input flow-> mode picker

    Ctrl+C at any screen cancels the whole wizard (returns ``None``).
    """
    while True:
        mode = await _prompt_for_mode()
        if mode is None:
            return None

        if mode == _MODE_CUSTOM:
            custom_result = await _prompt_for_custom_model(used_models)
            if custom_result is None:
                return None  # cancelled
            if custom_result == _BACK_SENTINEL:
                continue  # back to mode picker
            model, base_url = custom_result
            return model, base_url, "custom endpoint"

        # mode == _MODE_CURATED
        curated_result = await _gather_curated_model(used_models)
        if curated_result is None:
            return None  # cancelled
        if curated_result == _BACK_SENTINEL:
            continue  # back to mode picker
        provider, model_id = curated_result
        return f"{provider}/{model_id}", None, provider


async def _gather_curated_model(
    used_models: frozenset[str] = frozenset(),
) -> Optional[tuple[str, str] | str]:
    """Walk the provider -> model pickers with back-nav between them.

    Returns ``(provider, model_id)`` on a complete selection,
    :data:`_BACK_SENTINEL` when the user backs out of the provider
    picker (caller returns to the mode picker), or ``None`` when the
    user cancels the wizard (Ctrl+C).
    """
    while True:
        picked_provider = await _prompt_for_provider()
        if picked_provider is None:
            return None
        if picked_provider == _BACK_SENTINEL:
            return _BACK_SENTINEL

        model_id = await _prompt_for_model(picked_provider, used_models)
        if model_id is None:
            return None
        if model_id == _BACK_SENTINEL:
            # Back from model picker -> re-enter provider picker.
            continue
        return picked_provider, model_id


async def _prompt_for_mode() -> Optional[str]:
    """Step 0: pick "curated catalogue" vs "custom endpoint".

    This screen is the wizard's top level; a Ctrl+C cancels the whole
    wizard. There is no ``← Back`` here -- you can't back out of the
    root.

    Returns :data:`_MODE_CURATED`, :data:`_MODE_CUSTOM`, or ``None``.
    """
    labels_to_value = {
        _MODE_CURATED_LABEL: _MODE_CURATED,
        _MODE_CUSTOM_LABEL: _MODE_CUSTOM,
    }
    display = [_MODE_CURATED_LABEL, _MODE_CUSTOM_LABEL]
    picked = await select_from_list(
        display, "How would you like to configure the model?"
    )
    if picked is None:
        return None
    return labels_to_value.get(picked)


async def _prompt_for_provider() -> Optional[str]:
    """Step 1: provider picker inside the curated-list branch.

    A ``← Back`` entry at the end of the list returns to the mode
    picker without cancelling the wizard. Verified providers get a
    ⭐ prefix so users can spot the curated ones at a glance.

    Returns the provider name, :data:`_BACK_SENTINEL`, or ``None``.
    """
    menu = selection.build_provider_menu(
        excluded_providers=exclusions.EXCLUDED_PROVIDERS,
        included_providers=exclusions.INCLUDED_PROVIDERS,
    )
    labels_to_value: dict[str, str] = {}
    display: list[str] = []
    for entry in menu:
        label = f"{_VERIFIED_MARK}{entry.name}" if entry.verified else entry.name
        labels_to_value[label] = entry.name
        display.append(label)
    labels_to_value[_BACK_LABEL] = _BACK_SENTINEL
    display.append(_BACK_LABEL)

    _show_verified_legend_once(any(entry.verified for entry in menu))
    picked = await select_from_list(display, "Select an LLM provider")
    if picked is None:
        return None
    return labels_to_value.get(picked)


async def _prompt_for_custom_model(
    used_models: frozenset[str] = frozenset(),
) -> Optional[tuple[str, Optional[str]] | str]:
    """Typed-input flow for self-hosted / non-curated endpoints.

    Prompts for a fully-qualified model string in ``provider/model-name``
    form and an optional base URL. Typing ``back`` (case-insensitive)
    at the model prompt returns :data:`_BACK_SENTINEL` so the wizard
    pops up to the mode picker. Ctrl+C cancels the wizard entirely.

    If the typed model is already in ``used_models`` (another saved
    profile pins it), we reject the entry and re-prompt. Mirrors the
    block enforced in the curated picker.
    """
    _console.print(
        "[cyan]Custom endpoint: enter a model string in "
        "'provider/model-name' form (e.g. 'vllm/my-model', "
        "'openai/my-finetune'). Type 'back' to return to the mode "
        "picker.[/cyan]"
    )
    while True:
        try:
            model = Prompt.ask("Model").strip()
        except KeyboardInterrupt:
            return None
        if model.lower() == "back":
            return _BACK_SENTINEL
        if not model:
            _console.print("[red]❌ Model cannot be empty[/red]")
            return _BACK_SENTINEL
        if model in used_models:
            _console.print(
                "[yellow]⚠ That model is already configured by a saved "
                "profile. Pick a different one.[/yellow]"
            )
            continue
        break

    try:
        base_url_raw = Prompt.ask(
            "Base URL (press Enter to use the provider default)", default=""
        ).strip()
    except KeyboardInterrupt:
        return None
    base_url = base_url_raw or None
    return model, base_url


async def _prompt_for_model(
    provider: str, used_models: frozenset[str] = frozenset()
) -> Optional[str]:
    """Step 2: pick a model id from a known provider's catalog.

    Verified entries are prefixed with ⭐ for the display; the returned
    value is the bare model id (without the mark). A ``← Back`` entry
    returns to the provider picker.

    Entries whose full ``provider/model-id`` is in ``used_models`` are
    rendered with a ``(current)`` suffix and cannot be selected --
    picking one prints an explanation and re-enters the same picker.
    The add flow passes an empty set (no blocking); the update flow
    passes every saved profile's model so the user can't land on a
    duplicate.
    """
    menu = selection.build_model_menu(
        provider,
        excluded_models=exclusions.EXCLUDED_MODELS,
        included_models=exclusions.INCLUDED_MODELS,
    )
    if not menu:
        _console.print(
            f"[red]❌ No models available for provider '{provider}'[/red]"
        )
        return _BACK_SENTINEL

    labels_to_value: dict[str, str] = {}
    blocked_labels: set[str] = set()
    display: list[str] = []
    for entry in menu:
        base = (
            f"{_VERIFIED_MARK}{entry.model_id}"
            if entry.verified
            else entry.model_id
        )
        full = f"{provider}/{entry.model_id}"
        label = f"{base} (current)" if full in used_models else base
        labels_to_value[label] = entry.model_id
        if full in used_models:
            blocked_labels.add(label)
        display.append(label)
    labels_to_value[_BACK_LABEL] = _BACK_SENTINEL
    display.append(_BACK_LABEL)

    _show_verified_legend_once(any(entry.verified for entry in menu))
    while True:
        picked = await select_from_list(display, f"Select a {provider} model")
        if picked is None:
            return None
        if picked in blocked_labels:
            _console.print(
                "[yellow]⚠ That model is already configured by a saved "
                "profile. Pick a different one.[/yellow]"
            )
            continue
        return labels_to_value.get(picked)


@dataclass(frozen=True)
class _KeyLoopResult:
    """Outcome of the API-key-collection sub-step.

    Four possibilities matter to the wizard's outer loop:

    * ``kind == "validated"`` -- the probe succeeded (or the user saved
      through a rate-limit); ``key`` holds the accepted credential.
    * ``kind == "cancelled"`` -- hard abort (Ctrl+C / EOF at the key
      prompt, or at the rate-limit sub-prompt). Wizard exits.
    * ``kind == "abandoned"`` -- user explicitly picked "c" at the
      rate-limit sub-prompt. Drop this profile but keep the wizard
      alive so "add another?" runs.
    * ``kind == "restart_model_selection"`` -- the probe revealed a
      model-shaped failure (bad model ID, deprecated, not in the
      endpoint's catalogue). Looping on the same key-prompt is useless
      here; the outer wizard restarts from provider/model selection.
    """

    kind: Literal[
        "validated", "cancelled", "abandoned", "restart_model_selection"
    ]
    key: Optional[str] = None


def _prompt_for_api_key_until_valid(
    family_label: str,
    model: str,
    base_url: Optional[str],
) -> _KeyLoopResult:
    """Prompt for an API key and loop until the probe resolves or the
    outer wizard needs to take over.

    A credentials-shaped failure (auth / unknown / connection) keeps
    the user on this prompt -- just enter a different key -- but is
    bounded by :data:`_MAX_API_KEY_ATTEMPTS` so a stuck user or
    misbehaving auth endpoint cannot keep the wizard running forever.
    A model-shaped failure (not_found / bad_request) propagates up as
    ``restart_model_selection`` so the wizard re-runs provider and
    model selection; it's pointless to keep typing keys when the
    model itself is the problem. Rate-limit has its own three-way
    ``save / retry / cancel`` sub-prompt.
    """
    for attempt in range(1, _MAX_API_KEY_ATTEMPTS + 1):
        api_key = _prompt_for_api_key(family_label)
        if api_key is None:
            return _KeyLoopResult(kind="cancelled")

        outcome = _probe_and_handle(model, api_key, base_url)
        if outcome == "accepted":
            return _KeyLoopResult(kind="validated", key=api_key)
        if outcome == "cancel":
            return _KeyLoopResult(kind="cancelled")
        if outcome == "abandon":
            return _KeyLoopResult(kind="abandoned")
        if outcome == "pick_different_model":
            return _KeyLoopResult(kind="restart_model_selection")
        # outcome == "try_new_key"
        if attempt < _MAX_API_KEY_ATTEMPTS:
            _console.print(
                "[yellow]Enter a different key, or press Ctrl+C to cancel.[/yellow]"
            )

    _console.print(
        f"[red]❌ {_MAX_API_KEY_ATTEMPTS} keys rejected. "
        "Please verify your credentials and try again.[/red]"
    )
    return _KeyLoopResult(kind="cancelled")


def _prompt_for_api_key(family_label: str) -> Optional[str]:
    """Step 4: collect an API key.

    Returns the entered key (possibly empty, to cover endpoints that
    accept unauthenticated requests), or ``None`` when the user
    interrupts with Ctrl+C / EOF. The empty-string-vs-None distinction
    is what lets the caller loop on a rejected key without treating an
    explicit abort as a retry.
    """
    try:
        return Prompt.ask(
            f"API key for {family_label} "
            "[dim](input hidden; press Enter if not required)[/dim]",
            password=True,
            show_default=False,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        return None


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


def _prompt_profile_name(
    default: str, *, existing_names: frozenset[str] = frozenset()
) -> Optional[str]:
    """Prompt for a profile name, re-asking on duplicates.

    Returns the accepted name, or ``None`` if the user cancels with
    Ctrl+C / EOF. A duplicate against ``existing_names`` prints a
    short error and re-enters the prompt instead of letting the user
    walk further into the wizard only to hit a save-time collision.
    """
    while True:
        try:
            name = Prompt.ask("Profile name", default=default).strip() or None
        except KeyboardInterrupt:
            return None
        if name is None:
            return None
        if name in existing_names:
            _console.print(
                f"[red]❌ A profile named '{name}' already exists. "
                "Pick a different name.[/red]"
            )
            continue
        return name


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
        "New API key [dim](input hidden; press Enter if not required)[/dim]",
        password=True,
        show_default=False,
    ).strip()


async def _sequential_model_prompt(
    ctx: ProfileUpdateContext,
    *,
    used_models: frozenset[str] = frozenset(),
) -> None:
    if Confirm.ask(
        f"Keep current model '{ctx.profile.model}'?", default=True
    ):
        return
    # Delegate to the same mode-picker + back-nav state machine the
    # add flow uses. ``used_models`` includes *every* saved profile's
    # model -- including this profile's own current model -- so the
    # picker blocks duplicates. The user already declined
    # "keep current" above, so blocking the current model here is
    # intentional, not a regression.
    gathered = await _gather_model_and_base_url(used_models=used_models)
    if gathered is None:
        return
    new_model, new_base_url, _family = gathered
    ctx.updates["model"] = new_model
    ctx.updates["base_url"] = new_base_url


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


ProbeOutcome = Literal[
    "accepted",
    "cancel",            # hard abort -- wizard exits immediately
    "abandon",           # soft abort -- drop this profile but keep the
                         # wizard alive so "add another?" can run
    "try_new_key",
    "pick_different_model",
]

# Short human labels for the well-known upstream providers, to help the
# reader tell "I got rate-limited by Google" apart from "I got
# rate-limited by our own gateway". Absent entries fall back to just
# the raw provider identifier from litellm.
_PROVIDER_HUMAN_NAMES: dict[str, str] = {
    "openai": "OpenAI API",
    "gemini": "Google Gemini API",
    "anthropic": "Anthropic API",
    "together_ai": "TogetherAI API",
    "openrouter": "OpenRouter API",
    "groq": "Groq API",
    "cohere": "Cohere API",
    "azure": "Azure OpenAI",
    "mistral": "Mistral API",
    "deepseek": "DeepSeek API",
    "ollama": "Ollama (local/self-hosted)",
    "vllm": "vLLM (self-hosted)",
    "litellm_proxy": "LiteLLM proxy",
}


def _probe_and_handle(
    model: str, api_key: str, base_url: Optional[str]
) -> ProbeOutcome:
    """Run the live probe and decide what the caller should do next.

    Returns one of four sentinels:
    * ``"accepted"`` -- key is good (or user explicitly saved through
      a rate-limited probe); caller should persist the profile.
    * ``"try_new_key"`` -- probe failed in a way that points at the
      credentials (auth, connection, unknown); caller should re-prompt
      for a different key.
    * ``"pick_different_model"`` -- probe failed in a way that points
      at the model (not_found, bad_request); caller should restart
      the wizard from the provider/model picker rather than loop on
      the same key-prompt.
    * ``"cancel"`` -- user chose to abort the whole wizard.
    """
    _console.print("[cyan]Validating credentials...[/cyan]")
    result = validation.probe_api_key(model, api_key, base_url)
    if result.is_ok():
        _console.print("[green]✓ Credentials accepted[/green]")
        return "accepted"

    err = result.error
    assert isinstance(err, KeyProbeError)
    _print_probe_failure(err)

    if err.kind == "rate_limit":
        return _handle_rate_limited(model, api_key, base_url)

    if err.kind in ("not_found", "bad_request"):
        # Model-shaped failure: the key isn't the problem. Restart at
        # the provider/model picker instead of looping on the key.
        return "pick_different_model"

    # auth / connection / unknown -- let the user type a different key.
    return "try_new_key"


def _print_probe_failure(err: KeyProbeError) -> None:
    """Concise headline, plus dimmed provenance only in verbose mode."""
    _HEADLINES: dict[str, str] = {
        "auth": "[red]❌ Key rejected by the provider[/red]",
        "connection": (
            "[red]❌ Could not reach endpoint "
            "(check base_url / network)[/red]"
        ),
        "bad_request": (
            "[red]❌ Endpoint rejected the request (model may be invalid)[/red]"
        ),
        "not_found": (
            "[red]❌ Model not served by this endpoint "
            "(deprecated or unknown)[/red]"
        ),
        "rate_limit": "[yellow]⚠ Endpoint rate-limited the probe.[/yellow]",
    }
    _console.print(_HEADLINES.get(err.kind, "[red]❌ Probe failed[/red]"))

    # Provenance is only shown in verbose mode. A regular user doesn't
    # need to stare at litellm exception-class paths.
    if _VERBOSE:
        provenance = _format_provenance(err)
        if provenance:
            _console.print(provenance)


def _format_provenance(err: KeyProbeError) -> str:
    if not _VERBOSE:
        return ""
    lines = []
    if err.exception_class:
        lines.append(f"source:   {err.exception_class}")
    if err.provider:
        human = _PROVIDER_HUMAN_NAMES.get(err.provider)
        pretty = f"{err.provider} ({human})" if human else err.provider
        lines.append(f"provider: {pretty}")
    if err.status_code is not None:
        lines.append(f"status:   HTTP {err.status_code}")
    if err.detail:
        one_line = err.detail.splitlines()[0]
        lines.append(f"detail:   {one_line}")
    if not lines:
        return ""
    indented = "\n".join(f"  {line}" for line in lines)
    return f"[dim]{indented}[/dim]"


def _handle_rate_limited(
    model: str, api_key: str, base_url: Optional[str]
) -> ProbeOutcome:
    """Three-way prompt for a rate-limited probe.

    The provider's auth check runs before quota enforcement on every
    mainstream endpoint we've seen, so a 429 strongly implies the key
    is valid and the endpoint is just throttling. The prompt reflects
    that: ``save`` is the default, with explicit ``retry`` and
    ``cancel`` options.
    """
    _console.print(
        "[dim]Auth checks usually run before rate-limiting, so the key "
        "is almost certainly valid; the endpoint is just throttling.[/dim]"
    )
    # NB: the bracketed key letters below use explicit bold markup
    # rather than literal ``[s]`` / ``[r]`` / ``[c]`` -- Rich would
    # otherwise interpret the bare brackets as unknown style tags and
    # strip them, leaving the user to stare at ``ave anyway /
    # etry probe now / ancel``.
    prompt_text = (
        "[bold green]S[/bold green]ave anyway / "
        "[bold green]R[/bold green]etry probe now / "
        "[bold green]C[/bold green]ancel"
    )
    while True:
        try:
            choice = Prompt.ask(
                prompt_text,
                choices=["s", "r", "c"],
                default="s",
                show_choices=False,
            )
        except (KeyboardInterrupt, EOFError):
            return "cancel"

        if choice == "s":
            _console.print(
                "[green]✓ Profile accepted without successful probe[/green]"
            )
            return "accepted"
        if choice == "c":
            # Explicit "abandon this profile". The wizard should still
            # ask "Add another?" so the user doesn't get kicked out of
            # the CLI after spending time on picker + key.
            return "abandon"

        # choice == "r": re-run the probe with the same key.
        _console.print("[cyan]Retrying...[/cyan]")
        result = validation.probe_api_key(model, api_key, base_url)
        if result.is_ok():
            _console.print("[green]✓ Credentials accepted[/green]")
            return "accepted"

        err = result.error
        assert isinstance(err, KeyProbeError)
        _print_probe_failure(err)
        if err.kind != "rate_limit":
            # The retry surfaced a different failure (commonly: the
            # rate-limit cleared and now auth actually fails).
            # Fall through to the caller's "try a different key" loop.
            return "try_new_key"
        # Still rate-limited -- loop and re-ask.
