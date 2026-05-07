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

from devdox_ai_sonar.llm import adapters, interactive
from devdox_ai_sonar.llm.adapters import KeyProbeOutcome
from devdox_ai_sonar.llm.errors import (
    InvalidModelFormatError,
    KeyProbeError,
    ProfileNotFoundError,
    UnknownProviderError,
)
from devdox_ai_sonar.llm.profile import LLMProfile
from devdox_ai_sonar.llm.profile_store import (
    load_profiles,
    save_profile,
    update_profile,
)
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
    """Small deterministic fixture catalog for both adapter functions."""
    monkeypatch.setattr(
        adapters,
        "list_verified_models",
        lambda: {"openai": ["gpt-4o"], "openhands": ["claude-sonnet-4-6"]},
    )
    monkeypatch.setattr(
        adapters,
        "list_unverified_models",
        lambda: {"together_ai": ["mixtral-8x7b"]},
    )
    monkeypatch.setattr(
        adapters,
        "infer_provider",
        lambda model, api_base: _fake_infer(model),
    )
    # Default: probe succeeds.
    monkeypatch.setattr(
        adapters, "probe", lambda **_: KeyProbeOutcome(ok=True)
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
            "openai",          # provider picker
            "gpt-4o",          # model picker
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


async def test_add_profile_flow_custom_endpoint(
    monkeypatch, manager, fake_catalogs
):
    """Pick the 🔧 Custom option and supply a vllm/ model with a base URL.
    The saved profile should carry both the typed model string and the
    base URL verbatim."""
    _queue_responses(monkeypatch, iter(["🔧 Custom"]))
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

    profiles = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    assert len(profiles) == 1
    saved = profiles[0]
    assert saved.model == "vllm/my-local-model"
    assert saved.base_url == "https://vllm.local"
    assert saved.api_key == ""


async def test_add_profile_flow_custom_endpoint_rejects_empty_model(
    monkeypatch, manager, fake_catalogs
):
    """An empty model string at the custom prompt aborts the wizard
    cleanly without saving."""
    _queue_responses(monkeypatch, iter(["🔧 Custom"]))
    _patch_prompt(
        monkeypatch,
        text_answers=[""],   # model -> empty -> abort
        confirm_answers=[],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    assert profiles == []


async def test_update_profile_flow_switch_to_custom_endpoint(
    monkeypatch, manager, fake_catalogs
):
    """Update an existing profile to point at a custom endpoint. The
    model string + base_url are both persisted."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="sk"),
        set_as_default=False,
    )

    _queue_responses(
        monkeypatch,
        iter([
            "p",           # pick profile
            "🔧 Custom",   # new provider -> custom
        ]),
    )
    _patch_prompt(
        monkeypatch,
        text_answers=[
            "vllm/new-local",     # custom model
            "https://vllm.co",    # base_url
        ],
        confirm_answers=[
            True,    # Keep name?
            False,   # Change key?
            False,   # Keep model? -> no
            False,   # Set as default?
        ],
    )

    await interactive.update_profile_flow(manager)

    [updated] = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    assert updated.model == "vllm/new-local"
    assert updated.base_url == "https://vllm.co"


async def test_add_profile_flow_openhands_family_accepted(
    monkeypatch, manager, fake_catalogs
):
    """The openhands/ prefix isn't a litellm provider but must be accepted
    because the verified catalog knows it."""
    _queue_responses(
        monkeypatch,
        iter([
            "openhands",
            "claude-sonnet-4-6",
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
        adapters,
        "probe",
        lambda **_: KeyProbeOutcome(
            ok=False, failure_kind="auth", detail="bad key"
        ),
    )
    _queue_responses(monkeypatch, iter(["openai", "gpt-4o"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["bad-key"],
        confirm_answers=[False],  # Add another? -> false, so break
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


async def test_add_profile_flow_cancelled_at_provider_picker(
    monkeypatch, manager, fake_catalogs
):
    """Ctrl+C at the provider picker returns None; wizard exits without saving."""
    _queue_responses(monkeypatch, iter([None]))
    _patch_prompt(monkeypatch, text_answers=[], confirm_answers=[])

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


async def test_add_profile_flow_loops_on_add_another(
    monkeypatch, manager, fake_catalogs
):
    """Two profiles configured in sequence via the "Add another?" loop."""
    _queue_responses(
        monkeypatch,
        iter([
            "openai", "gpt-4o",
            "openhands", "claude-sonnet-4-6",
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

    monkeypatch.setattr(adapters, "probe", capture_probe)

    _queue_responses(
        monkeypatch,
        iter([
            "p",                    # pick profile
            "openhands",            # new provider
            "claude-sonnet-4-6",    # new model
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
        adapters,
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

    monkeypatch.setattr(adapters, "probe", should_not_be_called)

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
# add_profile_flow — additional branch coverage
# ---------------------------------------------------------------------------


async def test_add_profile_flow_save_failure_breaks_loop(
    monkeypatch, manager, fake_catalogs
):
    """If save_profile raises, the wizard prints a short error and stops
    looping -- it does not swallow the exception or keep asking."""

    async def failing_save(*_, **__):
        raise RuntimeError("disk is on fire")

    monkeypatch.setattr(interactive, "save_profile", failing_save)
    _queue_responses(
        monkeypatch, iter(["openai", "gpt-4o"])
    )
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-x", "my-prof"],
        confirm_answers=[False],   # Set as default?
    )

    # No exception should escape the flow.
    await interactive.add_profile_flow(manager)


async def test_add_profile_flow_cancelled_at_model_picker(
    monkeypatch, manager, fake_catalogs
):
    """Ctrl+C at the model picker (picked_provider OK, model_id None)
    cancels the wizard without saving."""
    _queue_responses(monkeypatch, iter(["openai", None]))
    _patch_prompt(monkeypatch, text_answers=[], confirm_answers=[])

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


async def test_add_profile_flow_cancelled_at_name_prompt(
    monkeypatch, manager, fake_catalogs
):
    """Ctrl+C at the profile-name prompt cancels the wizard (name is
    None, so no profile is created)."""

    class _PromptAbort:
        @staticmethod
        def ask(*args, **kwargs):
            raise KeyboardInterrupt

    _queue_responses(monkeypatch, iter(["openai", "gpt-4o"]))
    monkeypatch.setattr(interactive, "Prompt", _PromptAbort)

    class _FakeConfirm:
        @staticmethod
        def ask(*args, **kwargs):
            return kwargs.get("default", False)

    monkeypatch.setattr(interactive, "Confirm", _FakeConfirm)

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


# ---------------------------------------------------------------------------
# update_profile_flow — additional branch coverage
# ---------------------------------------------------------------------------


async def test_update_profile_flow_cancelled_at_profile_picker(
    monkeypatch, manager, fake_catalogs
):
    """With at least one profile saved, cancelling the profile picker
    (None) exits cleanly without errors."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    _queue_responses(monkeypatch, iter([None]))
    _patch_prompt(monkeypatch, text_answers=[], confirm_answers=[])

    await interactive.update_profile_flow(manager)

    [unchanged] = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    assert unchanged.api_key == "k"


async def test_update_profile_flow_profile_not_found_error(
    monkeypatch, manager, fake_catalogs
):
    """If the backing update_profile raises ProfileNotFoundError (e.g.
    the file was modified under us between load and save), the flow
    surfaces the error and returns."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )

    async def raising_update(*_, **__):
        raise ProfileNotFoundError("simulated race")

    monkeypatch.setattr(interactive, "update_profile", raising_update)

    _queue_responses(monkeypatch, iter(["p"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["new-key"],
        confirm_answers=[True, True, True, True, False],
    )

    # Should NOT raise.
    await interactive.update_profile_flow(manager)


async def test_update_profile_flow_set_as_default_after_rename(
    monkeypatch, manager, fake_catalogs
):
    """A rename + set-as-default updates both the profile name and the
    [llm].default_profile pointer in config.toml."""
    await save_profile(
        manager,
        LLMProfile(name="old-name", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    _queue_responses(monkeypatch, iter(["old-name"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["new-name"],
        confirm_answers=[
            False,  # Keep name? -> no
            False,  # Change key?
            True,   # Keep model?
            True,   # Keep base URL?
            True,   # Set as default?
        ],
    )

    await interactive.update_profile_flow(manager)

    reloaded = ConfigManager(config_path=manager.config_path)
    llm_section = await reloaded.get_value("llm")
    assert llm_section.get("default_profile") == "new-name"


# ---------------------------------------------------------------------------
# Menu / prompt edge cases
# ---------------------------------------------------------------------------


async def test_prompt_for_model_empty_catalog_returns_none(
    monkeypatch, manager, fake_catalogs, capsys
):
    """When a provider has no models at all, the picker prints a short
    error and the caller gets None back. Exercised via the add-flow."""
    monkeypatch.setattr(
        adapters,
        "list_verified_models",
        lambda: {"empty-provider": []},
    )
    monkeypatch.setattr(adapters, "list_unverified_models", lambda: {})

    _queue_responses(monkeypatch, iter(["empty-provider"]))
    _patch_prompt(monkeypatch, text_answers=[], confirm_answers=[])

    await interactive.add_profile_flow(manager)

    out = capsys.readouterr().out
    assert "No models available" in out


async def test_sequential_name_empty_input_keeps_current(
    monkeypatch, manager, fake_catalogs, capsys
):
    """Entering an empty new name leaves the profile's name unchanged
    and surfaces a short warning."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    _queue_responses(monkeypatch, iter(["p"]))
    _patch_prompt(
        monkeypatch,
        text_answers=[""],   # new name -> empty
        confirm_answers=[
            False,  # Keep name? -> no (trigger new-name prompt)
            False,  # Change key?
            True,   # Keep model?
            True,   # Keep base URL?
            False,  # Default?
        ],
    )

    await interactive.update_profile_flow(manager)

    [unchanged] = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    assert unchanged.name == "p"
    assert "unchanged" in capsys.readouterr().out.lower()


async def test_sequential_name_collision_keeps_current(
    monkeypatch, manager, fake_catalogs, capsys
):
    """Renaming to a name another profile already uses is rejected
    inline; the original name is kept."""
    await save_profile(
        manager,
        LLMProfile(name="alpha", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    await save_profile(
        manager,
        LLMProfile(name="beta", model="openhands/claude-sonnet-4-6", api_key="k"),
        set_as_default=False,
    )

    _queue_responses(monkeypatch, iter(["alpha"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["beta"],   # collide with other profile
        confirm_answers=[
            False,  # Keep name? -> no
            False,  # Change key?
            True,   # Keep model?
            True,   # Keep base URL?
            False,  # Default?
        ],
    )

    await interactive.update_profile_flow(manager)

    reloaded = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    names = {p.name for p in reloaded}
    assert names == {"alpha", "beta"}  # neither renamed
    assert "already in use" in capsys.readouterr().out


async def test_sequential_base_url_change_path(
    monkeypatch, manager, fake_catalogs
):
    """When the user declines 'Keep current base URL?' and supplies a
    new value, the update carries that base_url forward."""
    await save_profile(
        manager,
        LLMProfile(
            name="p",
            model="openai/gpt-4o",
            api_key="k",
            base_url="https://old.example.com",
        ),
        set_as_default=False,
    )
    _queue_responses(monkeypatch, iter(["p"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["https://new.example.com"],
        confirm_answers=[
            True,   # Keep name?
            False,  # Change key?
            True,   # Keep model?
            False,  # Keep base URL? -> no
            False,  # Default?
        ],
    )

    await interactive.update_profile_flow(manager)

    [updated] = await load_profiles(
        ConfigManager(config_path=manager.config_path)
    )
    assert updated.base_url == "https://new.example.com"


async def test_sequential_default_promotes_non_default_profile(
    monkeypatch, manager, fake_catalogs
):
    """A non-default profile can be promoted to default via the final
    'Set as default?' prompt."""
    await save_profile(
        manager,
        LLMProfile(name="p", model="openai/gpt-4o", api_key="k"),
        set_as_default=False,
    )
    _queue_responses(monkeypatch, iter(["p"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["new-key"],
        confirm_answers=[
            True,   # Keep name?
            True,   # Change key?
            True,   # Keep model?
            True,   # Keep base URL?
            True,   # Set as default?  <-- the branch we're exercising
        ],
    )

    await interactive.update_profile_flow(manager)

    reloaded = ConfigManager(config_path=manager.config_path)
    llm_section = await reloaded.get_value("llm")
    assert llm_section.get("default_profile") == "p"


# ---------------------------------------------------------------------------
# Error-rendering branches (validation + probe failure messages)
# ---------------------------------------------------------------------------


def test_validate_or_report_invalid_model_format(monkeypatch, capsys):
    """An InvalidModelFormatError is rendered as a specific message
    and the helper returns None so the caller knows to abort."""
    monkeypatch.setattr(
        interactive.validation,
        "validate_model_string",
        lambda model, base_url: interactive.validation.Err(
            InvalidModelFormatError(model=model, reason="empty")
        ),
    )
    result = interactive._validate_or_report("", base_url=None)
    assert result is None
    out = capsys.readouterr().out
    assert "Invalid model format" in out
    assert "empty" in out


def test_validate_or_report_unknown_provider(monkeypatch, capsys):
    monkeypatch.setattr(
        interactive.validation,
        "validate_model_string",
        lambda model, base_url: interactive.validation.Err(
            UnknownProviderError(model=model)
        ),
    )
    result = interactive._validate_or_report("bogus/xyz", base_url=None)
    assert result is None
    out = capsys.readouterr().out
    assert "Unknown provider prefix" in out
    assert "bogus/xyz" in out


def test_probe_or_report_prints_validating_header(monkeypatch, capsys):
    """The probe helper always prints 'Validating credentials...' at
    the start, regardless of outcome."""
    monkeypatch.setattr(
        interactive.validation,
        "probe_api_key",
        lambda model, api_key, base_url: interactive.validation.Ok(None),
    )
    assert interactive._probe_or_report("openai/gpt-4o", "sk", None) is True
    out = capsys.readouterr().out
    assert "Validating credentials" in out
    assert "accepted" in out.lower()


@pytest.mark.parametrize(
    "failure_kind, expected_phrase",
    [
        ("auth", "Key rejected"),
        ("connection", "Could not reach endpoint"),
        ("bad_request", "Endpoint rejected the request"),
        ("rate_limit", "rate-limited"),
        ("unknown", "Probe failed"),
    ],
)
def test_probe_or_report_failure_messages(
    monkeypatch, capsys, failure_kind, expected_phrase
):
    """Each distinct probe failure kind surfaces a distinct message so
    the user can tell auth from connection from rate-limit."""
    monkeypatch.setattr(
        interactive.validation,
        "probe_api_key",
        lambda model, api_key, base_url: interactive.validation.Err(
            KeyProbeError(kind=failure_kind, detail="upstream detail")
        ),
    )
    result = interactive._probe_or_report("openai/gpt-4o", "sk", None)
    assert result is False
    out = capsys.readouterr().out
    assert expected_phrase in out
