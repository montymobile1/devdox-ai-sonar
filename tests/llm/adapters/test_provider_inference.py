"""Tests for the provider_inference adapter (API #4)."""

import pytest

from devdox_ai_sonar.llm.adapters import provider_inference


@pytest.mark.parametrize(
    ("model", "api_base", "expected"),
    [
        # Fully-qualified, well-known providers.
        ("openai/gpt-4o", None, "openai"),
        ("gemini/gemini-2.5-flash", None, "gemini"),
        ("anthropic/claude-sonnet-4-6", None, "anthropic"),
        ("together_ai/meta-llama/Llama-3-70b-chat", None, "together_ai"),
        ("openrouter/anthropic/claude-sonnet-4", None, "openrouter"),
        ("ollama/llama3", None, "ollama"),
        # Bare OpenAI model — auto-inferred via litellm's cost map.
        ("gpt-4o", None, "openai"),
        # Custom-endpoint prefixes: require api_base in practice, but the
        # resolver doesn't mind if one isn't supplied.
        ("vllm/qwen-72b", "https://my-vllm.example.com", "vllm"),
        ("azure/gpt-4", "https://my-azure.openai.azure.com", "azure"),
    ],
)
def test_known_prefixes_resolve(model, api_base, expected):
    assert provider_inference.infer_provider(model=model, api_base=api_base) == expected


@pytest.mark.parametrize(
    "model",
    [
        "bogus/whatever",
        "my-company/my-model",
        "claude-3-5-sonnet-20241022",  # Bare anthropic — not auto-attributed.
        "not-a-real-model",
    ],
)
def test_unknown_or_unresolvable_returns_none(model):
    assert provider_inference.infer_provider(model=model, api_base=None) is None


def test_openhands_prefix_is_not_routable_by_litellm():
    """The ``openhands/*`` namespace is handled by the OpenHands Cloud
    gateway rather than a litellm adapter. Validation callers must treat
    this ``None`` specially by falling back to the verified-models map.
    """
    assert provider_inference.infer_provider(
        model="openhands/gpt-5.4", api_base=None
    ) is None
