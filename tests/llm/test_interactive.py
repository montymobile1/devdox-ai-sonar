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

from devdox_ai_sonar.llm import interactive, selection, validation
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
            "⭐ openai",          # provider picker
            "⭐ gpt-4o",          # model picker
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
    """Pick Custom, supply a vllm/ model and base_url."""
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
    _queue_responses(monkeypatch, iter(["⭐ openai", "⭐ gpt-4o"]))
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
            "⭐ openai", "⭐ gpt-4o",
            "⭐ openhands", "⭐ claude-sonnet-4-6",
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
            "p",                    # pick profile
            "⭐ openhands",         # new provider
            "⭐ claude-sonnet-4-6", # new model
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
    _queue_responses(monkeypatch, iter(["⭐ openai", "⭐ gpt-4o"]))
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
    _queue_responses(monkeypatch, iter(["⭐ openai", "⭐ gpt-4o"]))
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
    """The 'c' choice stops the wizard without saving."""
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
    _queue_responses(monkeypatch, iter(["⭐ openai", "⭐ gpt-4o"]))
    _patch_prompt(
        monkeypatch,
        text_answers=["sk-key", "c"],
        confirm_answers=[],
    )

    await interactive.add_profile_flow(manager)

    profiles = await load_profiles(ConfigManager(config_path=manager.config_path))
    assert profiles == []


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
            "⭐ openai", "⭐ gpt-4o",              # first attempt
            "⭐ openhands", "⭐ claude-sonnet-4-6", # second attempt
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
            "⭐ openai", "⭐ gpt-4o",
            "⭐ openai", "⭐ gpt-4o",
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
    _queue_responses(monkeypatch, iter(["⭐ openai", "⭐ gpt-4o"]))
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
