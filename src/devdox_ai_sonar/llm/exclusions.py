"""Developer-owned filter lists for the LLM wizard menus.

These lists are part of the codebase, NOT user-facing configuration.
The intended audience is the maintainers of devdox-sonar: as providers
deprecate models, surface new ones that don't fit the product
(image-only, TTS-only, etc.), or introduce curated shortlists, edit
the constants below and ship a release.

Two filter shapes, both dev-owned
---------------------------------
* **Exclusion** (denylist): remove specific entries. Everything not
  named stays visible. Good for hiding a handful of broken or
  off-product items out of the full catalogue.
* **Inclusion** (allowlist): show only specific entries. Everything
  not named is hidden. Good for curating a small shortlist when the
  full catalogue would overwhelm the picker.

The lists cooperate via a simple rule: **inclusion narrows, exclusion
always wins.** See :func:`check_profile_against_filters` for the
precise combination applied to saved profiles.

Effect on the UI
----------------
* Entries in :data:`EXCLUDED_PROVIDERS` hide the provider family
  entirely from the Step 1 picker.
* Entries in :data:`EXCLUDED_MODELS` are full ``provider/model-id``
  strings. The named model is removed from the Step 2 picker for its
  provider.
* Entries in :data:`INCLUDED_PROVIDERS` (if non-empty) restrict the
  Step 1 picker to just those providers; any unlisted provider
  disappears even when not in :data:`EXCLUDED_PROVIDERS`.
* Entries in :data:`INCLUDED_MODELS` (if non-empty) restrict the
  Step 2 picker to just those full ``provider/model-id`` strings.
* The 🔧 Custom free-text path ignores **both** exclusion and
  inclusion lists -- it's an escape hatch for advanced users who know
  what they're doing.

What does NOT happen automatically
----------------------------------
* Existing saved profiles in ``~/devdox/config.toml`` that point at
  newly-filtered entries are left alone. The user is warned at run
  time (see :func:`check_profile_against_filters`) but not forced to
  change anything -- their configuration is theirs.
* No automatic migration. If you flag a model as deprecated or narrow
  the inclusion list, the next run of a non-matching profile prints a
  warning and proceeds. The user can run ``update_provider`` when
  they're ready.

How to use
----------
Import the sets directly; they are intentionally plain
``frozenset[str]`` so they can be filtered in-place::

    from devdox_ai_sonar.llm.exclusions import (
        EXCLUDED_MODELS, EXCLUDED_PROVIDERS,
        INCLUDED_MODELS, INCLUDED_PROVIDERS,
    )

Inclusion lists start empty; when empty they are treated as "no
constraint" and only exclusion filtering applies. Add entries with a
comment pointing at the reason (provider announcement URL, deprecation
date, internal ticket) so the next maintainer can tell when to remove
them.
"""

from typing import Optional

from devdox_ai_sonar.llm.profile import LLMProfile


__all__ = [
    "EXCLUDED_PROVIDERS",
    "EXCLUDED_MODELS",
    "INCLUDED_PROVIDERS",
    "INCLUDED_MODELS",
    "check_profile_against_filters",
]


# ---------------------------------------------------------------------------
# Exclusions. Start with a curated denylist of non-chat entries.
# ---------------------------------------------------------------------------


EXCLUDED_PROVIDERS: frozenset[str] = frozenset({
    # Non-text providers that cannot serve chat completions for code
    # fixes. Including them in the picker would just produce 400/405
    # responses when the user eventually picks a model from one of
    # them and hits the probe.
    "aws_polly",          # text-to-speech only
    "elevenlabs",         # text-to-speech only
    "deepgram",           # speech-to-text
    "assemblyai",         # speech-to-text
    "stability",          # image generation
    "black_forest_labs",  # image generation (FLUX family)
    "recraft",            # image generation
    "runwayml",           # video generation
    "fal_ai",             # image / video inference service
    "voyage",             # embeddings only
})


EXCLUDED_MODELS: frozenset[str] = frozenset({
    # Gemini 1.x: Google removed the entire family from the v1beta
    # public API. Probe hits produce HTTP 404 NotFoundError, but
    # litellm still carries them in its catalogue so they'd otherwise
    # appear in the "advanced" menu.
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-1.0-pro",
    "gemini/gemini-pro",

    # Gemini image-generation variants. Produce pixels, not text --
    # useless for a code fixer.
    "gemini/imagen-3.0-generate-001",
    "gemini/imagen-3.0-fast-generate-001",
    "gemini/imagen-3.0-generate-002",
    "gemini/imagen-4.0-generate-001",
    "gemini/imagen-4.0-fast-generate-001",
    "gemini/imagen-4.0-ultra-generate-001",
    "gemini/gemini-2.0-flash-exp-image-generation",
    "gemini/gemini-2.5-flash-image",
    "gemini/gemini-3-pro-image-preview",
    "gemini/gemini-3.1-flash-image-preview",

    # Gemini video-generation variants (Veo).
    "gemini/veo-2.0-generate-001",
    "gemini/veo-3.1-generate-001",
    "gemini/veo-3.1-generate-preview",
    "gemini/veo-3.1-lite-generate-preview",
    "gemini/veo-3.1-fast-generate-001",
    "gemini/veo-3.1-fast-generate-preview",

    # Gemini music-generation variants (Lyria).
    "gemini/lyria-3-pro-preview",
    "gemini/lyria-3-clip-preview",

    # Gemini embedding models. Return vectors, not chat completions.
    "gemini/gemini-embedding-001",
    "gemini/gemini-embedding-2-preview",

    # Gemini TTS previews. Audio output only.
    "gemini/gemini-2.5-flash-preview-tts",
    "gemini/gemini-2.5-pro-preview-tts",

    # Gemini native-audio / live variants. Duplex audio streams, not
    # request/response code fixing.
    "gemini/gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini/gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini/gemini-2.5-flash-native-audio-latest",
    "gemini/gemini-live-2.5-flash-preview-native-audio-09-2025",
    "gemini/gemini-3.1-flash-live-preview",

    # Gemini non-chat specialists. Trained for different tasks
    # (browser automation, robotics embeddings) than code patching.
    "gemini/gemini-2.5-computer-use-preview-10-2025",
    "gemini/gemini-robotics-er-1.5-preview",

    # OpenAI Sora (video generation).
    "openai/sora-2",
    "openai/sora-2-pro",
    "openai/sora-2-pro-high-res",

    # OpenAI safeguard variants. Moderation classifiers -- return
    # safe/unsafe labels, not code.
    "openai/gpt-oss-safeguard-120b",
    "openai/gpt-oss-safeguard-20b",
})


# ---------------------------------------------------------------------------
# Dev-owned allowlists. Empty by default -- treated as "no constraint".
# ---------------------------------------------------------------------------


INCLUDED_PROVIDERS: frozenset[str] = frozenset()
"""When non-empty, restrict the Step 1 picker to these providers only.

Empty (the current default) means the constraint is not applied and
only :data:`EXCLUDED_PROVIDERS` filters the picker.
"""


INCLUDED_MODELS: frozenset[str] = frozenset()
"""When non-empty, restrict the Step 2 picker to these full
``provider/model-id`` strings only.

Empty (the current default) means the constraint is not applied and
only :data:`EXCLUDED_MODELS` filters the picker.
"""


# ---------------------------------------------------------------------------
# Runtime check for already-saved profiles.
# ---------------------------------------------------------------------------


def check_profile_against_filters(profile: LLMProfile) -> Optional[str]:
    """Return a warning message if the profile is filtered out by the
    current exclusion/inclusion lists.

    The intent is *informational*: callers print the returned string
    (if any) and proceed. We never refuse to run a profile the user
    saved earlier -- their configuration is theirs to adjust.

    Checking order (most-specific first):

    1. Exclusion wins outright: a profile whose model is in
       :data:`EXCLUDED_MODELS` gets the model-specific warning.
    2. Provider-family exclusion: the family is in
       :data:`EXCLUDED_PROVIDERS`.
    3. Inclusion miss (model): :data:`INCLUDED_MODELS` is non-empty
       but the profile's model isn't listed.
    4. Inclusion miss (provider): :data:`INCLUDED_PROVIDERS` is
       non-empty but the profile's family isn't listed.

    Preferring specificity means a single profile only ever produces
    one warning, even if it also technically misses a broader list.

    Args:
        profile: The resolved profile about to be used for a run.

    Returns:
        A human-readable warning, or ``None`` when the profile passes
        every applicable filter.
    """
    if profile.model in EXCLUDED_MODELS:
        return (
            f"Profile '{profile.name}' uses '{profile.model}', which is "
            "on the devdox-sonar exclusion list (likely deprecated or "
            "unsupported). The run will proceed, but consider "
            "'devdox_sonar -c update_provider' to switch to a current "
            "model."
        )

    family = profile.family()
    if family in EXCLUDED_PROVIDERS:
        return (
            f"Profile '{profile.name}' uses provider '{family}', which "
            "is on the devdox-sonar exclusion list. The run will "
            "proceed, but consider 'devdox_sonar -c update_provider' "
            "to switch to a different provider."
        )

    # Inclusion: a profile passes if ANY non-empty allowlist accepts
    # it. A model-specific match (more specific) clears a broader
    # provider miss -- e.g. a dev that explicitly names
    # ``anthropic/claude-4`` on ``INCLUDED_MODELS`` alongside an
    # ``INCLUDED_PROVIDERS={"openai"}`` provider allowlist is carving
    # out an exception, not fencing themselves in.
    if INCLUDED_MODELS and profile.model in INCLUDED_MODELS:
        return None
    if INCLUDED_PROVIDERS and family in INCLUDED_PROVIDERS:
        return None

    # Neither inclusion rule matched. Warn against the most specific
    # miss: if the dev set a model allowlist at all, the user's
    # non-match there is the actionable guidance.
    if INCLUDED_MODELS:
        return (
            f"Profile '{profile.name}' uses '{profile.model}', which "
            "is not on the devdox-sonar curated model list. The run "
            "will proceed, but consider 'devdox_sonar -c "
            "update_provider' to switch to a curated model."
        )
    if INCLUDED_PROVIDERS:
        return (
            f"Profile '{profile.name}' uses provider '{family}', which "
            "is not on the devdox-sonar curated provider list. The run "
            "will proceed, but consider 'devdox_sonar -c "
            "update_provider' to switch to a curated provider."
        )

    return None
