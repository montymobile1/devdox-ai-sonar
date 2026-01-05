from enum import Enum
from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel
from google import genai
import openai
from openai import OpenAI
from together import Together

API_KEY_EMPTY = "API key cannot be empty"
NO_MODELS_FOUND = "No models found for this API key"


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    TOGETHERAI = "togetherai"

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
            client = OpenAI(api_key=api_key)
            models = client.models.list().data
            model_list = [model.id for model in models]

            if not model_list:
                return ProviderValidationResult.failure_result(NO_MODELS_FOUND)

            return ProviderValidationResult.success_result(model_list)

        except openai.AuthenticationError:
            return ProviderValidationResult.failure_result(
                "Invalid API key - authentication failed"
            )
        except openai.APIError as e:
            return ProviderValidationResult.failure_result(f"API error: {str(e)}")
        except ImportError:
            return ProviderValidationResult.failure_result(
                "OpenAI package not installed"
            )
        except Exception as e:
            return ProviderValidationResult.failure_result(
                f"Unexpected error: {str(e)}"
            )

    @staticmethod
    def validate_gemini(api_key: str) -> ProviderValidationResult:
        """Validate Gemini API key and fetch available models."""
        if not api_key or not api_key.strip():
            return ProviderValidationResult.failure_result(API_KEY_EMPTY)

        try:
            client = genai.Client(api_key=api_key)
            models = client.models.list()
            model_list = [
                model.name.replace("models/", "")
                for model in models
                if model.name is not None
            ]
            if not model_list:
                return ProviderValidationResult.failure_result(NO_MODELS_FOUND)

            return ProviderValidationResult.success_result(model_list)

        except ImportError:
            return ProviderValidationResult.failure_result(
                "Google GenAI package not installed"
            )
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
            client = Together(api_key=api_key)
            models = client.models.list()
            model_list = [model.id for model in models]

            if not model_list:
                return ProviderValidationResult.failure_result(NO_MODELS_FOUND)

            return ProviderValidationResult.success_result(model_list)

        except ImportError:
            return ProviderValidationResult.failure_result(
                "Together package not installed"
            )
        except Exception as e:
            return ProviderValidationResult.failure_result(
                f"Validation error: {str(e)}"
            )

    @classmethod
    def validate(cls, provider: ProviderType, api_key: str) -> ProviderValidationResult:
        """Validate API key for any provider type."""
        validators = {
            ProviderType.OPENAI: cls.validate_openai,
            ProviderType.GEMINI: cls.validate_gemini,
            ProviderType.TOGETHERAI: cls.validate_togetherai,
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
