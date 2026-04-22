"""Developer-owned lists of LLM providers and models to hide from the wizard.

These lists are part of the codebase, NOT user-facing configuration.
The intended audience is the maintainers of devdox-sonar: as providers
deprecate models or surface new ones that don't fit the product
(image-only, TTS-only, etc.), add or remove entries here and ship a
release.

Effect on the UI
----------------
* Entries in :data:`EXCLUDED_PROVIDERS` hide the provider family
  entirely from the Step 1 picker -- and by extension every model
  under it.
* Entries in :data:`EXCLUDED_MODELS` are full ``provider/model-id``
  strings. The named model is removed from the Step 2 picker for its
  provider. Other models under the same provider are untouched.
* The 🔧 Custom free-text path also honours both lists: the wizard
  rejects user input that matches with an actionable message and
  restarts at the provider picker.

What does NOT happen automatically
----------------------------------
* Existing saved profiles in ``~/devdox/config.toml`` that point at a
  newly-excluded model are left alone. The user is warned at run time
  (see :func:`check_profile_against_exclusions`) but not forced to
  change anything -- their configuration is theirs.
* No automatic migration. If you flag a model as deprecated here, the
  next invocation of the user's existing profile will print a
  warning and proceed. The user can run ``update_provider`` when
  they're ready.

How to use
----------
Import the sets directly; they are intentionally plain
``frozenset[str]`` so they can be filtered in-place::

    from devdox_ai_sonar.llm.exclusions import (
        EXCLUDED_MODELS, EXCLUDED_PROVIDERS,
    )

Start both lists empty. Add entries with a comment pointing at the
reason (provider announcement URL, deprecation date, internal ticket)
so the next maintainer can tell when to remove them.
"""

from __future__ import annotations

from typing import Optional

from devdox_ai_sonar.llm.profile import LLMProfile


__all__ = [
    "EXCLUDED_PROVIDERS",
    "EXCLUDED_MODELS",
    "check_profile_against_exclusions",
]


# ---------------------------------------------------------------------------
# The two dev-owned lists. Start empty; add entries as needed.
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
    # public API. Confirmed against ListModels on 2026-04-23 -- no
    # 1.* entries served. Probe hits produce HTTP 404 NotFoundError,
    # but litellm still carries them in its catalogue so they'd
    # otherwise appear in the "advanced" menu.
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
# Runtime check for already-saved profiles.
# ---------------------------------------------------------------------------


def check_profile_against_exclusions(profile: LLMProfile) -> Optional[str]:
    """Return a warning message if the profile uses an excluded entry.

    The intent is *informational*: callers print the returned string
    (if any) and proceed. We never refuse to run a profile the user
    saved earlier -- their configuration is theirs to adjust.

    Args:
        profile: The resolved profile about to be used for a run.

    Returns:
        A human-readable warning, or ``None`` when the profile is
        clear. The check prefers the more specific message (matching
        a full model entry) over the coarser family-level one, so a
        user excluding a single model doesn't get two warnings at
        once for the same profile.
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

    return None
