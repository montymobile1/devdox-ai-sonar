"""The ``LLMProfile`` dataclass -- one configured LLM entry.

A profile is the unit of LLM configuration the rest of the application
passes around. It contains everything needed to instantiate an
``openhands.sdk.LLM``: the fully-qualified model string, an optional API
key, and an optional custom endpoint URL.

Profiles are serialised as entries of the ``[[llm.profiles]]`` TOML array
in ``~/devdox/config.toml``. The :meth:`to_toml_dict` / :meth:`from_toml_dict`
methods are the canonical serialisation format.
"""

from dataclasses import dataclass
from typing import Any


__all__ = ["LLMProfile"]


@dataclass
class LLMProfile:
    """A single configured LLM.

    Attributes:
        name: User-supplied label for this profile. Unique within the
            ``[[llm.profiles]]`` array; also what
            ``[llm].default_profile`` references.
        model: Fully-qualified model string in the OpenHands / litellm
            convention, e.g. ``"openai/gpt-4o"``, ``"gemini/gemini-2.5-flash"``,
            ``"vllm/my-local-model"``.
        api_key: API key for whichever provider the model routes to. The
            empty string is permitted -- it indicates "endpoint accepts
            unauthenticated requests" (e.g. a local Ollama).
        base_url: Optional custom endpoint URL. Only set when the user
            chose the "Custom" path during configuration, or when their
            endpoint differs from the provider's default.
    """

    name: str
    model: str
    api_key: str = ""
    base_url: str | None = None

    def to_toml_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for a TOML array-of-tables entry.

        ``base_url`` is omitted from the output when ``None`` so that
        profiles configured against a provider's default endpoint do not
        carry a noisy empty field.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "api_key": self.api_key,
        }
        if self.base_url is not None:
            payload["base_url"] = self.base_url
        return payload

    @classmethod
    def from_toml_dict(cls, data: dict[str, Any]) -> "LLMProfile":
        """Construct from a TOML array-of-tables entry.

        Raises:
            KeyError: If either ``name`` or ``model`` is missing. These two
                are required; ``api_key`` defaults to empty and ``base_url``
                defaults to ``None`` when absent.
        """
        return cls(
            name=data["name"],
            model=data["model"],
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url"),
        )

    def family(self) -> str:
        """Return the provider family prefix of :attr:`model`.

        For ``"openai/gpt-4o"`` -> ``"openai"``; for
        ``"together_ai/meta-llama/Llama-3-70b-chat"`` -> ``"together_ai"``.
        If the model has no ``/`` separator, the whole model string is
        returned (this case is uncommon after validation but possible for
        bare-model auto-inferred strings like ``"gpt-4o"``).
        """
        separator_index = self.model.find("/")
        if separator_index == -1:
            return self.model
        return self.model[:separator_index]
