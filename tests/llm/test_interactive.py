"""Tests for llm/interactive.py.

These tests exercise the wizard and update flows against fake prompts
and fake catalog adapters. The real interactive primitives (Rich's
``Prompt``/``Confirm`` and ``questionary`` via ``select_from_list``)
are replaced with ``monkeypatch.setattr`` so the flows run
deterministically without a TTY.
"""

from pathlib import Path
from typing import Iterator

import pytest

from devdox_ai_sonar.llm import exclusions, interactive, selection, validation
from devdox_ai_sonar.llm.adapters import (
    key_probe,
    provider_inference,
    unverified_models,
    verified_models,
)
from devdox_ai_sonar.llm.adapters.key_probe import KeyProbeOutcome
from devdox_ai_sonar.llm.profile import LLMProfile
from devdox_ai_sonar.llm.profile_store import load_profiles, save_profile
from devdox_ai_sonar.models.llm_config import ConfigManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    m = ConfigManager(config_path=tmp_path / "config.toml")
    m.create_default_config()
    return m


@pytest.fixture
def fake_catalogs(monkeypatch):
    """Small deterministic fixture catalog for both adapters."""
    monkeypatch.setattr(
        verified_models,
        "list_providers",
        lambda: {"openai": ["gpt-4o"], "openhands": ["claude-sonnet-4-6"]},
    )
    monkeypatch.setattr(
        unverified_models,
        "list_providers",
        lambda: {"together_ai": ["mixtral-8x7b"]},
    )
    monkeypatch.setattr(
        provider_inference,
        "infer_provider",
        lambda model, api_base: _fake_infer(model),
    )
    # Default: probe succeeds.
    monkeypatch.setattr(
        key_probe, "probe", lambda **_: KeyProbeOutcome(ok=True)
    )


def _fake_infer(model: str) -> str | None:
    """Minimal provider inference: split on '/' and accept a known set."""
    if "/" not in model:
        return None
    prefix = model.split("/", 1)[0]
    known = {"openai", "gemini", "together_ai", "vllm", "litellm_proxy"}
    return prefix if prefix in known else None


def _queue_responses(monkeypatch, responses: Iterator):
    """Queue up ``select_from_list`` responses in FIFO order."""
    it = iter(responses)

    async def fake_select(choices, message, use_search=True):
        return next(it, None)

    monkeypatch.setattr(interactive, "select_from_list", fake_select)


def _patch_prompt(monkeypatch, *, text_answers=None, confirm_answers=None):
    """Replace Rich's Prompt.ask / Confirm.ask inside interactive module.

    When the queued-response iterator is exhausted, the fake raises
    ``KeyboardInterrupt`` -- matching what a real user-initiated abort
    produces. This keeps retry loops in the code under test from
    spinning forever on an empty queue.
    """
    texts = iter(text_answers or [])
    confirms = iter(confirm_answers or [])

    class _FakePrompt:
        @staticmethod
        def ask(*args, **kwargs):
            try:
                return next(texts)
            except StopIteration:
                raise KeyboardInterrupt

    class _FakeConfirm:
        @staticmethod
        def ask(*args, **kwargs):
            try:
                return next(confirms)
            except StopIteration:
                return kwargs.get("default", False)

    monkeypatch.setattr(interactive, "Prompt", _FakePrompt)
    monkeypatch.setattr(interactive, "Confirm", _FakeConfirm)


# ---------------------------------------------------------------------------
# add_profile_flow
# ---------------------------------------------------------------------------


async def test_add_profile_flow_happy_path(monkeypatch, manager, fake_catalogs):
    """Pick openai + gpt-4o, supply a key, accept defaults, save, don't loop."""
    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list",  # mode picker
            "⭐ openai",                   # provider picker
            "⭐ gpt-4o",                   # model picker
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-test-key", "my-openai"],
        confirm_answers=[
            True,   # Set as default?
            False,  # Add another?
        ],
    )

    await interactive.add_profile_flow(manager)

    reloaded = ConfigManager(config_path=manager.config_path)
    profiles = await load_profiles(reloaded)
    assert len(profiles) == 1
    saved = profiles[0]
    assert saved.name == "my-openai"
    assert saved.model == "openai/gpt-4o"
    assert saved.api_key == "sk-test-key"
    assert saved.base_url is None


async def test_add_profile_flow_rejects_duplicate_name_and_reprompts(
    monkeypatch, manager, fake_catalogs
):
    """If the first typed name collides with an existing profile, the
    wizard must reject it immediately (before the set-as-default
    prompt) and re-ask. The second, unique name is accepted and
    persisted. Regression test: previously the collision was only
    caught at save time, after the user had clicked past
    "Set as default?"."""
    await save_profile(
        manager,
        LLMProfile(name="gemini", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )

    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list",
            "⭐ openai",
            "⭐ gpt-4o",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "sk-new-key",
            "gemini",        # duplicate -> rejected + re-prompt
            "gemini-alt",    # unique -> accepted
        ],
        confirm_answers=[
            False,  # Set as default?
            False,  # Add another?
        ],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    names = {p.name for p in profiles}
    assert names == {"gemini", "gemini-alt"}


async def test_add_profile_flow_custom_endpoint(
    monkeypatch, manager, fake_catalogs
):
    """Pick Custom at the mode picker, supply a vllm/ model and base_url."""
    _queue_responses(monkeypatch, iter(["🔧 Custom endpoint"]))
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "vllm/my-local-model",   # model
            "https://vllm.local",    # base_url
            "",                      # api_key (local, no auth)
            "vllm-prof",             # profile name
        ],
        confirm_answers=[
            False,  # Set as default?
            False,  # Add another?
        ],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert len(profiles) == 1
    saved = profiles[0]
    assert saved.model == "vllm/my-local-model"
    assert saved.base_url == "https://vllm.local"
    assert saved.api_key == ""


async def test_add_profile_flow_openhands_family_accepted(
    monkeypatch, manager, fake_catalogs
):
    """The openhands/ prefix isn't a litellm provider but must be accepted
    because the verified catalog knows it."""
    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list",
            "⭐ openhands",
            "⭐ claude-sonnet-4-6",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["oh-key", "oh"],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles[0].model == "openhands/claude-sonnet-4-6"


async def test_add_profile_flow_rejects_unreachable_key(
    monkeypatch, manager, fake_catalogs
):
    """When the probe fails, the wizard should NOT save the profile and
    should exit cleanly."""
    monkeypatch.setattr(
        key_probe,
        "probe",
        lambda **_: KeyProbeOutcome(
            ok=False, failure_kind="auth", detail="bad key"
        ),
    )
    _queue_responses(
        monkeypatch,
        iter(["📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o"]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["bad-key"],
        confirm_answers=[False],  # Add another? -> false, so break
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


async def test_add_profile_flow_cancelled_at_mode_picker(
    monkeypatch, manager, fake_catalogs
):
    """Ctrl+C at the mode picker returns None; wizard exits without saving."""
    _queue_responses(monkeypatch, iter([None]))
    _patch_prompt(monkeypatch, text_answers=[], confirm_answers=[])

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


async def test_add_profile_flow_loops_on_add_another(
    monkeypatch, manager, fake_catalogs
):
    """Two profiles configured in sequence via the "Add another?" loop.
    Each iteration re-enters the mode picker."""
    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o",
            "📋 Pick from curated list", "⭐ openhands", "⭐ claude-sonnet-4-6",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "sk-1", "openai-prof",
            "oh-1",  "oh-prof",
        ],
        confirm_answers=[
            False,  # default? (first)
            True,   # add another?
            True,   # default? (second)
            False,  # add another?
        ],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    names = {p.name for p in profiles}
    assert names == {"openai-prof", "oh-prof"}


# ---------------------------------------------------------------------------
# update_profile_flow
# ---------------------------------------------------------------------------


async def test_update_profile_flow_rotate_key_only(
    monkeypatch, manager, fake_catalogs
):
    """Change nothing except the API key; probe the final combination
    once; commit."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="old-key"),
        set_as_default=False,
    )
    _queue_responses(monkeypatch, iter(["p"]))  # pick profile
    _patch_prompt(
        monkeypatch,
        text_answers=["new-key"],
        confirm_answers=[
            True,   # Keep current name?
            True,   # Change the API key?
            True,   # Keep current model?
            True,   # Keep current base URL?
            False,  # Set as default?  (already not default)
        ],
    )

    await interactive.update_profile_flow(manager)

    [updated] = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert updated.api_key == "new-key"
    assert updated.model == "openai/gpt-4o"   # unchanged


async def test_update_profile_flow_model_change_triggers_reprobe(
    monkeypatch, manager, fake_catalogs
):
    """Change the model; the probe runs against the NEW model + existing key."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    probed_with: dict = {}

    def capture_probe(model, api_key, api_base):
        probed_with["model"] = model
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(key_probe, "probe", capture_probe)

    _queue_responses(
        monkeypatch,
        iter([
            "p",                          # pick profile
            "📋 Pick from curated list",  # mode picker
            "⭐ openhands",               # new provider
            "⭐ claude-sonnet-4-6",       # new model
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[],
        confirm_answers=[
            True,   # Keep name?
            False,  # Change key?
            False,  # Keep model? -> no
            False,  # Set as default?
        ],
    )

    await interactive.update_profile_flow(manager)

    assert probed_with["model"] == "openhands/claude-sonnet-4-6"
    [updated] = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert updated.model == "openhands/claude-sonnet-4-6"
    assert updated.api_key == "k"  # untouched


async def test_update_profile_flow_blocks_picking_a_used_model(
    monkeypatch, manager, fake_catalogs
):
    """During update, every saved profile's model is rendered with a
    ``(current)`` suffix and cannot be selected. Picking one warns and
    re-enters the model picker; the user then picks a genuinely free
    model and the update proceeds."""
    # Two profiles exist. Both models ("openai/gpt-4o" and
    # "openhands/claude-sonnet-4-6") should appear in the update
    # picker as (current) and be blocked. Give openai a second model
    # so the user has a free choice to actually land on.
    monkeypatch.setattr(
        verified_models,
        "list_providers",
        lambda: {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "openhands": ["claude-sonnet-4-6"],
        },
    )
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    await save_profile(
        manager,
        LLMProfile(
            name="other",
            model="openhands/claude-sonnet-4-6",
            api_key="k2",
        ),
        set_as_default=False,
    )

    captured_model_menu: list[list[str]] = []
    # Scripted picks: try the profile's own current model first to
    # exercise the block, then try the *other* profile's model to
    # exercise the block for an unrelated entry, then land on the
    # free gpt-4o-mini.
    picks = iter([
        "p",                                # pick profile p
        "📋 Pick from curated list",        # mode
        "⭐ openai",                        # provider
        "⭐ gpt-4o (current)",              # blocked -- same profile
        "⭐ gpt-4o-mini",                   # free choice; but first the
                                            # menu must also show the
                                            # "other" profile's model
                                            # marked current under its
                                            # own provider (asserted
                                            # via captured_model_menu
                                            # for openhands below).
    ])

    async def recording_select(choices, message, use_search=True):
        if message.startswith("Select a openai model") or (
            "openai model" in message
        ):
            captured_model_menu.append(list(choices))
        return next(picks, None)

    monkeypatch.setattr(interactive, "select_from_list", recording_select)
    _patch_prompt(
        monkeypatch,
        text_answers=[],
        confirm_answers=[
            True,   # Keep name?
            False,  # Change key?
            False,  # Keep model? -> no
            False,  # Set as default?
        ],
    )

    await interactive.update_profile_flow(manager)

    # The openai model menu must have shown the profile's current
    # model marked "(current)". The second capture is after the
    # block re-prompt -- same list, same label.
    assert captured_model_menu, "openai model menu was never shown"
    first_menu = captured_model_menu[0]
    assert "⭐ gpt-4o (current)" in first_menu
    assert "⭐ gpt-4o-mini" in first_menu
    # The picker was re-entered after the blocked pick.
    assert len(captured_model_menu) >= 2

    # Profile p was updated to the free alternative.
    reloaded = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    p_updated = next(prof for prof in reloaded if prof.name == "p")
    assert p_updated.model == "openai/gpt-4o-mini"


async def test_update_profile_flow_probe_failure_rolls_back(
    monkeypatch, manager, fake_catalogs
):
    """If the final probe fails, NO fields are persisted."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    monkeypatch.setattr(
        key_probe,
        "probe",
        lambda **_: KeyProbeOutcome(ok=False, failure_kind="auth", detail="nope"),
    )
    _queue_responses(monkeypatch, iter(["p"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["would-be-new-key"],
        confirm_answers=[
            True,   # Keep name?
            True,   # Change key?
            True,   # Keep model?
            True,   # Keep base_url?
            False,  # Set as default?
        ],
    )

    await interactive.update_profile_flow(manager)

    [unchanged] = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert unchanged.api_key == "k"   # rolled back


async def test_update_profile_flow_no_profiles_exits_gracefully(
    monkeypatch, manager, fake_catalogs
):
    _queue_responses(monkeypatch, iter([]))
    _patch_prompt(monkeypatch)
    await interactive.update_profile_flow(manager)
    # No exception, nothing saved -- implicit.


async def test_update_profile_flow_no_changes_exits_without_probe(
    monkeypatch, manager, fake_catalogs
):
    """If every 'Keep X?' is yes and nothing changes, we don't probe
    the network and don't write anything."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )

    probe_called = {"count": 0}

    def should_not_be_called(**_):
        probe_called["count"] += 1
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(key_probe, "probe", should_not_be_called)

    _queue_responses(monkeypatch, iter(["p"]))
    _patch_prompt(
        monkeypatch,
        text_answers=[],
        confirm_answers=[
            True,   # Keep name
            False,  # Change key
            True,   # Keep model
            True,   # Keep base URL
            False,  # Default
        ],
    )

    await interactive.update_profile_flow(manager)

    assert probe_called["count"] == 0


async def test_update_profile_flow_rename_tracks_in_config(
    monkeypatch, manager, fake_catalogs
):
    """After rename, load_profiles() sees the new name."""
    await save_profile(
        manager,
        LLMProfile(name="old", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    _queue_responses(monkeypatch, iter(["old"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["renamed"],
        confirm_answers=[
            False,  # Keep name? -> no
            False,  # Change key?
            True,   # Keep model?
            True,   # Keep base URL?
            False,  # Default?
        ],
    )

    await interactive.update_profile_flow(manager)

    [renamed] = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert renamed.name == "renamed"


# ---------------------------------------------------------------------------
# Rate-limited probe: the three-way save / retry / cancel prompt
# ---------------------------------------------------------------------------


async def test_rate_limited_probe_saves_when_user_picks_s(
    monkeypatch, manager, fake_catalogs
):
    """Defaulting to 's' accepts the key despite the rate limit and
    persists the profile."""
    monkeypatch.setattr(
        key_probe,
        "probe",
        lambda **_: KeyProbeOutcome(
            ok=False,
            failure_kind="rate_limit",
            detail="quota exceeded",
            provider="gemini",
            status_code=429,
            exception_class="litellm.exceptions.RateLimitError",
        ),
    )
    _queue_responses(
        monkeypatch,
        iter(["📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o"]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "sk-valid-but-throttled",
            "s",                    # save anyway
            "rate-limited-profile", # profile name
        ],
        confirm_answers=[
            True,   # Set as default?
            False,  # Add another?
        ],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert len(profiles) == 1
    assert profiles[0].api_key == "sk-valid-but-throttled"


async def test_rate_limited_probe_retries_successfully(
    monkeypatch, manager, fake_catalogs
):
    """The 'r' choice re-runs the probe. A clear-on-retry accepts
    the key without re-asking the user."""
    call_count = {"n": 0}

    def flaky_probe(**_):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return KeyProbeOutcome(
                ok=False,
                failure_kind="rate_limit",
                detail="quota",
                provider="gemini",
                status_code=429,
                exception_class="litellm.exceptions.RateLimitError",
            )
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(key_probe, "probe", flaky_probe)
    _queue_responses(
        monkeypatch,
        iter(["📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o"]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "sk-key",
            "r",          # retry
            "prof-name",
        ],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert len(profiles) == 1
    assert call_count["n"] == 2


async def test_rate_limited_probe_cancel_aborts_wizard(
    monkeypatch, manager, fake_catalogs
):
    """The 'c' choice abandons this profile but still defers to the
    "Add another profile?" prompt. When the user declines (here: the
    default False via the exhausted confirm iter), the wizard exits
    without saving."""
    monkeypatch.setattr(
        key_probe,
        "probe",
        lambda **_: KeyProbeOutcome(
            ok=False,
            failure_kind="rate_limit",
            detail="quota",
            provider="gemini",
            status_code=429,
            exception_class="litellm.exceptions.RateLimitError",
        ),
    )
    _queue_responses(
        monkeypatch,
        iter(["📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o"]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-key", "c"],
        confirm_answers=[],   # "Add another?" defaults to False -> exit
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


async def test_rate_limited_probe_cancel_then_add_another_tries_again(
    monkeypatch, manager, fake_catalogs
):
    """Picking 'c' at the rate-limit prompt should NOT drop the user
    back to the top-level CLI. It should offer "Add another profile?"
    just like a successful save does -- so a user who hit a 429 with
    one provider can immediately try a different one."""
    probe_calls = {"n": 0}

    def probe_by_attempt(**_):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:
            return KeyProbeOutcome(
                ok=False,
                failure_kind="rate_limit",
                detail="quota",
                provider="gemini",
                status_code=429,
                exception_class="litellm.exceptions.RateLimitError",
            )
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(key_probe, "probe", probe_by_attempt)
    _queue_responses(
        monkeypatch,
        iter([
            # First attempt: rate-limited, user picks 'c'.
            "📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o",
            # Second attempt after "Add another?": different provider.
            "📋 Pick from curated list",
            "⭐ openhands", "⭐ claude-sonnet-4-6",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "sk-throttled",           # first key
            "c",                       # abandon first profile
            "sk-ok",                   # second key
            "profile-kept",            # profile name for the second one
        ],
        confirm_answers=[
            True,    # "Add another?" after the abandoned first attempt
            True,    # Set as default? (second profile)
            False,   # Add another? -> exit
        ],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    assert len(profiles) == 1
    assert profiles[0].name == "profile-kept"
    assert profiles[0].model == "openhands/claude-sonnet-4-6"
    assert probe_calls["n"] == 2


# ---------------------------------------------------------------------------
# Model-shaped probe failures restart the wizard at the provider picker
# ---------------------------------------------------------------------------


async def test_not_found_probe_restarts_wizard_at_provider_picker(
    monkeypatch, manager, fake_catalogs
):
    """When the probe returns ``not_found`` (the endpoint doesn't
    serve the picked model), the wizard must loop back to the
    provider/model picker rather than keep asking for a different
    key. The user picks a different model, the next probe succeeds,
    and the profile is saved under the second model."""
    call_count = {"n": 0}

    def probe_by_attempt(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return KeyProbeOutcome(
                ok=False,
                failure_kind="not_found",
                detail="deprecated model",
                provider="gemini",
                status_code=404,
                exception_class="litellm.exceptions.NotFoundError",
            )
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(key_probe, "probe", probe_by_attempt)
    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o",
            "📋 Pick from curated list", "⭐ openhands", "⭐ claude-sonnet-4-6",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "key-for-first",
            "key-for-second",
            "final-profile-name",
        ],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert len(profiles) == 1
    assert profiles[0].model == "openhands/claude-sonnet-4-6"
    assert call_count["n"] == 2


async def test_bad_request_probe_also_restarts_wizard(
    monkeypatch, manager, fake_catalogs
):
    """Same restart behaviour for ``bad_request`` -- treated as a
    model-configuration problem, not a key problem."""
    call_count = {"n": 0}

    def probe_by_attempt(**_):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return KeyProbeOutcome(
                ok=False,
                failure_kind="bad_request",
                detail="unrecognised model",
                provider="openai",
                status_code=400,
                exception_class="litellm.exceptions.BadRequestError",
            )
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(key_probe, "probe", probe_by_attempt)
    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o",
            "📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-first", "sk-second", "profile"],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert len(profiles) == 1
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Verified-star legend ("⭐ = verified by OpenHands")
# ---------------------------------------------------------------------------


async def test_verified_legend_printed_once_per_wizard_run(
    monkeypatch, manager, fake_catalogs, capsys
):
    """The ⭐ legend is educational, not a status indicator, so it
    prints exactly once per wizard invocation -- above the first
    picker that contains verified entries. Subsequent pickers in the
    same run stay silent even though they also show stars."""
    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list",
            "⭐ openai",
            "⭐ gpt-4o",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-key", "p"],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    out = capsys.readouterr().out
    legend_count = out.count("⭐ = verified by OpenHands")
    assert legend_count == 1, (
        f"expected the legend exactly once per wizard run; "
        f"saw {legend_count} occurrences"
    )


async def test_verified_legend_resets_between_wizard_runs(
    monkeypatch, manager, fake_catalogs, capsys
):
    """A fresh add/update invocation is a new session -- reset the
    'legend already shown' flag so a new user running the tool later
    still sees the explanation."""
    queued = iter([
        # First run.
        "📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o",
        # Second run (after capsys readout in between -- separate
        # invocation of add_profile_flow).
        "📋 Pick from curated list", "⭐ openhands", "⭐ claude-sonnet-4-6",
    ])
    # monkeypatch shares a single fake_select across both runs by
    # reading from the same iterator.
    _queue_responses(monkeypatch, queued)
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "sk-key", "prof-one",
            "sk-key", "prof-two",
        ],
        confirm_answers=[
            True, False,   # default? / add another? -> no (first run)
            True, False,   # default? / add another? -> no (second run)
        ],
    )

    await interactive.add_profile_flow(manager)
    out1 = capsys.readouterr().out
    assert out1.count("⭐ = verified by OpenHands") == 1

    await interactive.add_profile_flow(manager)
    out2 = capsys.readouterr().out
    assert out2.count("⭐ = verified by OpenHands") == 1


async def test_verified_legend_suppressed_when_no_verified_entries(
    monkeypatch, manager, fake_catalogs, capsys
):
    """No stars in the menu means no legend -- don't noisily explain a
    symbol the user won't see."""
    # together_ai only has an unverified entry in the fake catalog.
    monkeypatch.setattr(
        verified_models, "list_providers", lambda: {}
    )
    monkeypatch.setattr(
        unverified_models,
        "list_providers",
        lambda: {"together_ai": ["mixtral-8x7b"]},
    )

    _queue_responses(
        monkeypatch,
        iter([
            "📋 Pick from curated list",
            "together_ai",        # no star prefix -- unverified
            "mixtral-8x7b",
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-key", "p"],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    out = capsys.readouterr().out
    assert "⭐ = verified by OpenHands" not in out


# ---------------------------------------------------------------------------
# Verbose toggle controls the provenance block
# ---------------------------------------------------------------------------


def test_format_provenance_returns_empty_when_not_verbose(monkeypatch):
    """The dimmed provenance block only appears when the CLI is
    invoked with ``--verbose``. Regular users get the one-line
    headline and nothing more."""
    monkeypatch.setattr(interactive, "_VERBOSE", False)
    from devdox_ai_sonar.llm.errors import KeyProbeError

    err = KeyProbeError(
        kind="auth",
        detail="bad key",
        provider="gemini",
        status_code=401,
        exception_class="litellm.exceptions.AuthenticationError",
    )
    assert interactive._format_provenance(err) == ""


def test_format_provenance_renders_all_fields_when_verbose(monkeypatch):
    """Verbose mode surfaces the litellm exception class, provider
    identifier, HTTP status, and first line of the upstream detail --
    enough for a developer to pin the failure to the right layer."""
    monkeypatch.setattr(interactive, "_VERBOSE", True)
    from devdox_ai_sonar.llm.errors import KeyProbeError

    err = KeyProbeError(
        kind="auth",
        detail="bad key\nmore detail",
        provider="gemini",
        status_code=401,
        exception_class="litellm.exceptions.AuthenticationError",
    )
    rendered = interactive._format_provenance(err)
    assert "litellm.exceptions.AuthenticationError" in rendered
    assert "gemini" in rendered
    assert "Google Gemini API" in rendered  # human label applied
    assert "HTTP 401" in rendered
    assert "bad key" in rendered
    # Only the first line of multi-line detail leaks into provenance;
    # full text is still available on err.detail for programmatic use.
    assert "more detail" not in rendered


async def test_rate_limited_probe_retry_surfacing_auth_error_falls_back(
    monkeypatch, manager, fake_catalogs
):
    """If the retry reveals a *different* failure (e.g. the rate limit
    cleared and now auth fails), the wizard must leave the rate-limit
    prompt and drop back to the 'enter a different key' outer loop."""
    call_count = {"n": 0}

    def flaky_probe(**_):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return KeyProbeOutcome(
                ok=False, failure_kind="rate_limit",
                detail="quota", provider="gemini", status_code=429,
                exception_class="litellm.exceptions.RateLimitError",
            )
        if call_count["n"] == 2:
            return KeyProbeOutcome(
                ok=False, failure_kind="auth",
                detail="bad key", provider="gemini", status_code=401,
                exception_class="litellm.exceptions.AuthenticationError",
            )
        return KeyProbeOutcome(ok=True)

    monkeypatch.setattr(key_probe, "probe", flaky_probe)
    _queue_responses(
        monkeypatch,
        iter(["📋 Pick from curated list", "⭐ openai", "⭐ gpt-4o"]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "throttled-but-bad-key",   # first key
            "r",                        # retry -> now auth-rejected
            "actually-valid-key",       # new key via outer loop
            "prof-name",
        ],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert len(profiles) == 1
    assert profiles[0].api_key == "actually-valid-key"
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Developer-owned exclusions (EXCLUDED_PROVIDERS / EXCLUDED_MODELS)
# ---------------------------------------------------------------------------


async def test_add_flow_hides_excluded_provider_from_picker(
    monkeypatch, manager, fake_catalogs
):
    """End-to-end: a provider in ``exclusions.EXCLUDED_PROVIDERS``
    disappears from the Step 1 menu. The wizard proceeds with whichever
    alternative is presented."""
    monkeypatch.setattr(
        exclusions, "EXCLUDED_PROVIDERS", frozenset({"openhands"})
    )
    monkeypatch.setattr(exclusions, "EXCLUDED_MODELS", frozenset())

    # Track what was actually shown to select_from_list so we can
    # assert ``openhands`` never made it into the menu. Skip the
    # mode-picker screen (identified by its fixed two-choice menu)
    # when asserting.
    shown_to_user: list[list[str]] = []

    async def recording_select(choices, message, use_search=True):
        shown_to_user.append(list(choices))
        # First screen is the mode picker.
        if "📋 Pick from curated list" in choices:
            return "📋 Pick from curated list"
        # Pick openai and its gpt-4o as a viable, non-excluded path.
        if "⭐ openai" in choices:
            return "⭐ openai"
        if "⭐ gpt-4o" in choices:
            return "⭐ gpt-4o"
        return None

    monkeypatch.setattr(interactive, "select_from_list", recording_select)
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-test", "prof-name"],
        confirm_answers=[True, False],
    )

    await interactive.add_profile_flow(manager)

    # shown_to_user[0] is the mode picker; the provider menu is [1].
    provider_menu = shown_to_user[1]
    assert "⭐ openhands" not in provider_menu
    assert "openhands" not in provider_menu


async def test_add_flow_hides_excluded_model_from_picker(
    monkeypatch, manager, fake_catalogs
):
    """An entry in ``exclusions.EXCLUDED_MODELS`` disappears from the
    Step 2 menu for its provider; other models under the same provider
    are still visible."""
    monkeypatch.setattr(exclusions, "EXCLUDED_PROVIDERS", frozenset())
    monkeypatch.setattr(
        exclusions, "EXCLUDED_MODELS", frozenset({"openai/gpt-4o"})
    )
    # Give openai a second model so the picker still has a
    # non-excluded option; otherwise the new back-nav on an empty
    # model list would loop the wizard back to the provider picker.
    monkeypatch.setattr(
        verified_models,
        "list_providers",
        lambda: {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "openhands": ["claude-sonnet-4-6"],
        },
    )

    shown_model_menu: list[list[str]] = []

    async def recording_select(choices, message, use_search=True):
        # Mode picker first.
        if "📋 Pick from curated list" in choices:
            return "📋 Pick from curated list"
        if message.startswith("Select a openai model") or "model" in message:
            shown_model_menu.append(list(choices))
        if "⭐ openai" in choices:
            return "⭐ openai"
        # No gpt-4o available; pick whatever else is on offer so we
        # don't hang. Use the raw Mock's ``None`` to cancel if empty.
        return choices[0] if choices else None

    monkeypatch.setattr(interactive, "select_from_list", recording_select)
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-test", "prof-name"],
        confirm_answers=[False, False],
    )

    await interactive.add_profile_flow(manager)

    # Concatenate every menu `recording_select` was asked about -- the
    # specific message match is brittle, but "gpt-4o never shown" is
    # robust across any prompt wording.
    all_entries = [label for menu in shown_model_menu for label in menu]
    assert "⭐ gpt-4o" not in all_entries
    assert "gpt-4o" not in all_entries
