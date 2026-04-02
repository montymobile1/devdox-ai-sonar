from enum import Enum
from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether


API_KEY_EMPTY = "API key cannot be empty"
NO_MODELS_FOUND = "No models found for this API key"


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    TOGETHERAI = "togetherai"
    OPENROUTER = "openrouter"

    @classmethod
    def choices(cls) -> List[str]:
        """Get list of provider names for menus."""
        return [p.value for p in cls]


@dataclass
class ProviderValidationResult:
    """Result of provider API key validation."""

    success: bool
    models: List[str]
    error_message: Optional[str] = None

    @classmethod
    def success_result(cls, models: List[str]) -> "ProviderValidationResult":
        """Create a successful validation result."""
        return cls(success=True, models=models, error_message=None)

    @classmethod
    def failure_result(cls, error: str) -> "ProviderValidationResult":
        """Create a failed validation result."""
        return cls(success=False, models=[], error_message=error)


class ProviderValidator:
    """Validates API keys for different LLM providers."""

    @staticmethod
    def validate_openai(api_key: str) -> ProviderValidationResult:
        """Validate OpenAI API key and fetch available models."""
        if not api_key or not api_key.strip():
            return ProviderValidationResult.failure_result(API_KEY_EMPTY)

        try:
            # ChatOpenAI wraps openai.OpenAI; .client exposes the underlying SDK client.
            wrapper = ChatOpenAI(api_key=api_key, model="gpt-4o")
            models = wrapper.client.models.list().data
            model_list = [model.id for model in models]

            if not model_list:
                return ProviderValidationResult.failure_result(NO_MODELS_FOUND)

            return ProviderValidationResult.success_result(model_list)

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "authentication" in error_msg
                or "api key" in error_msg
                or "401" in error_msg
            ):
                return ProviderValidationResult.failure_result(
                    "Invalid API key - authentication failed"
                )
            return ProviderValidationResult.failure_result(
                f"Unexpected error: {str(e)}"
            )

    @staticmethod
    def validate_gemini(api_key: str) -> ProviderValidationResult:
        """Validate Gemini API key and fetch available models."""
        if not api_key or not api_key.strip():
            return ProviderValidationResult.failure_result(API_KEY_EMPTY)

        try:
            # ChatGoogleGenerativeAI wraps google.genai.Client;
            # .client exposes the underlying SDK client for model listing.
            wrapper = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", google_api_key=api_key
            )
            models = wrapper.client.models.list()
            model_list = [
                model.name.replace("models/", "")
                for model in models
                if model.name is not None
            ]

            if not model_list:
                return ProviderValidationResult.failure_result(NO_MODELS_FOUND)

            return ProviderValidationResult.success_result(model_list)

        except Exception as e:
            return ProviderValidationResult.failure_result(
                f"Validation error: {str(e)}"
            )

    @staticmethod
    def validate_togetherai(api_key: str) -> ProviderValidationResult:
        """Validate TogetherAI API key and fetch available models."""
        if not api_key or not api_key.strip():
            return ProviderValidationResult.failure_result(API_KEY_EMPTY)

        try:
            # ChatTogether wraps together.Together;
            # .client exposes the underlying SDK client for model listing.
            wrapper = ChatTogether(
                model="meta-llama/Llama-3-70b-chat-hf", together_api_key=api_key
            )
            models = wrapper.client.models.list()
            model_list = [model.id for model in models]

            if not model_list:
                return ProviderValidationResult.failure_result(NO_MODELS_FOUND)

            return ProviderValidationResult.success_result(model_list)

        except Exception as e:
            return ProviderValidationResult.failure_result(
                f"Validation error: {str(e)}"
            )

    @staticmethod
    def validate_openrouter(api_key: str) -> ProviderValidationResult:
        """Validate OpenRouter API key and fetch available models."""
        if not api_key or not api_key.strip():
            return ProviderValidationResult.failure_result(API_KEY_EMPTY)

        try:
            # OpenRouter is OpenAI-compatible — use ChatOpenAI with base_url override.
            # .client exposes the underlying openai.OpenAI SDK client.
            wrapper = ChatOpenAI(
                api_key=api_key,
                model="anthropic/claude-sonnet-4",
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://devdox.ai",
                    "X-Title": "DevDox AI Sonar",
                },
            )
            models = wrapper.client.models.list().data
            model_list = [model.id for model in models]

            if not model_list:
                return ProviderValidationResult.failure_result(NO_MODELS_FOUND)

            return ProviderValidationResult.success_result(model_list)

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "authentication" in error_msg
                or "api key" in error_msg
                or "401" in error_msg
            ):
                return ProviderValidationResult.failure_result(
                    "Invalid API key - authentication failed"
                )
            return ProviderValidationResult.failure_result(
                f"Unexpected error: {str(e)}"
            )

    @classmethod
    def validate(cls, provider: ProviderType, api_key: str) -> ProviderValidationResult:
        """Validate API key for any provider type."""
        validators = {
            ProviderType.OPENAI: cls.validate_openai,
            ProviderType.GEMINI: cls.validate_gemini,
            ProviderType.TOGETHERAI: cls.validate_togetherai,
            ProviderType.OPENROUTER: cls.validate_openrouter,
        }

        validator = validators.get(provider)
        if not validator:
            return ProviderValidationResult.failure_result(
                f"Unknown provider: {provider}"
            )

        return validator(api_key)


class LLMProviderConfig(BaseModel):
    """Single LLM provider configuration"""

    name: str
    api_key: str
    base_url: str
    models: list[str]
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000
    timeout: int = 30
