"""Thin adapters around the OpenHands SDK and litellm.

Each module in this sub-package wraps exactly one external call. These are
the only files in devdox-ai-sonar that import ``openhands.sdk`` or ``litellm``;
every other module depends on these adapters so that tests can substitute them
with ``monkeypatch.setattr`` without touching the real SDK.

Adapters
--------
- ``verified_models``      API #1 — curated provider/model catalogue.
- ``unverified_models``    API #2 — full litellm catalogue minus Bedrock.
- ``provider_inference``   API #4 — validate/route a ``provider/model`` string.
- ``key_probe``            ``litellm.check_valid_key`` — live credential check.
"""
