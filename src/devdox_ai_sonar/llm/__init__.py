"""LLM configuration, selection, validation, and profile management.

This package is the single internal gateway between devdox-ai-sonar and the
external OpenHands SDK / litellm libraries. Only modules inside
``devdox_ai_sonar.llm.adapters`` are permitted to import from
``openhands.sdk`` or ``litellm``; everything else in this package depends on
those adapters (which makes tests fast, offline, and easy to mock via
``monkeypatch.setattr``).

Sub-modules
-----------
- ``adapters``    Thin wrappers around OpenHands SDK / litellm calls.
- ``selection``   Pure functions that build provider/model menus.
- ``validation``  Model-string and API-key validation.
- ``profile``     ``LLMProfile`` dataclass.
- ``profile_store`` TOML I/O for ``[[llm.profiles]]``.
- ``env_profile`` Build a profile from environment variables.
- ``interactive`` Interactive add/update profile flows.
- ``errors``      Custom exceptions and branchable error types.
"""
