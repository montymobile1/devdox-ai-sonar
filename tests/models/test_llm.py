import pytest
from unittest.mock import Mock, patch, MagicMock
import openai

from devdox_ai_sonar.models.llm import (
    ProviderType,
    ProviderValidationResult,
    ProviderValidator,
    LLMProviderConfig,
)


# ============================================================================
# TEST CLASS: ProviderType
# ============================================================================

class TestProviderType:
    """Test ProviderType enum"""

    def test_provider_type_values(self):
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.GEMINI.value == "gemini"
        assert ProviderType.TOGETHERAI.value == "togetherai"
        assert ProviderType.OPENROUTER.value == "openrouter"

    def test_provider_type_from_string(self):
        assert ProviderType("openai") == ProviderType.OPENAI
        assert ProviderType("gemini") == ProviderType.GEMINI
        assert ProviderType("togetherai") == ProviderType.TOGETHERAI
        assert ProviderType("openrouter") == ProviderType.OPENROUTER

    def test_provider_type_choices(self):
        choices = ProviderType.choices()

        assert isinstance(choices, list)
        assert "openai" in choices
        assert "gemini" in choices
        assert "togetherai" in choices
        assert "openrouter" in choices
        assert len(choices) == 4

    def test_provider_type_choices_all_strings(self):
        choices = ProviderType.choices()
        assert all(isinstance(choice, str) for choice in choices)

    def test_provider_type_enum_iteration(self):
        providers = list(ProviderType)

        assert len(providers) == 4
        assert ProviderType.OPENAI in providers
        assert ProviderType.GEMINI in providers
        assert ProviderType.TOGETHERAI in providers
        assert ProviderType.OPENROUTER in providers

    def test_provider_type_equality(self):
        assert ProviderType.OPENAI == ProviderType.OPENAI
        assert ProviderType.OPENAI != ProviderType.GEMINI

    def test_provider_type_string_representation(self):
        assert str(ProviderType.OPENAI) == "ProviderType.OPENAI"


# ============================================================================
# TEST CLASS: ProviderValidationResult
# ============================================================================

class TestProviderValidationResult:
    """Test ProviderValidationResult dataclass"""

    def test_validation_result_creation_success(self):
        result = ProviderValidationResult(
            success=True, models=["gpt-4", "gpt-3.5-turbo"], error_message=None
        )

        assert result.success is True
        assert len(result.models) == 2
        assert result.error_message is None

    def test_validation_result_creation_failure(self):
        result = ProviderValidationResult(
            success=False, models=[], error_message="Invalid API key"
        )

        assert result.success is False
        assert result.models == []
        assert result.error_message == "Invalid API key"

    def test_success_result_class_method(self):
        models = ["model-1", "model-2", "model-3"]
        result = ProviderValidationResult.success_result(models)

        assert result.success is True
        assert result.models == models
        assert result.error_message is None

    def test_success_result_empty_models(self):
        result = ProviderValidationResult.success_result([])

        assert result.success is True
        assert result.models == []

    def test_failure_result_class_method(self):
        error = "Authentication failed"
        result = ProviderValidationResult.failure_result(error)

        assert result.success is False
        assert result.models == []
        assert result.error_message == error

    def test_failure_result_empty_error(self):
        result = ProviderValidationResult.failure_result("")

        assert result.success is False
        assert result.error_message == ""

    def test_validation_result_is_dataclass(self):
        result = ProviderValidationResult.success_result(["model"])
        assert hasattr(result, "__dataclass_fields__")

    def test_validation_result_default_error_message(self):
        result = ProviderValidationResult(success=True, models=["model"])
        assert result.error_message is None


# ============================================================================
# Helpers
# ============================================================================

def _make_chat_openai_mock(model_ids: list[str]) -> Mock:
    """Build a ChatOpenAI mock whose .client.models.list().data returns model mocks."""
    mock_wrapper = Mock()
    mock_wrapper.client.models.list.return_value.data = [
        Mock(id=mid) for mid in model_ids
    ]
    return mock_wrapper


def _make_chat_together_mock(model_ids: list[str]) -> Mock:
    """Build a ChatTogether mock whose .client.models.list() returns model mocks."""
    mock_wrapper = Mock()
    mock_wrapper.client.models.list.return_value = [Mock(id=mid) for mid in model_ids]
    return mock_wrapper


def _make_chat_gemini_mock(model_names: list[str]) -> Mock:
    """Build a ChatGoogleGenerativeAI mock whose .client.models.list() returns model mocks."""
    mock_wrapper = Mock()
    mock_wrapper.client.models.list.return_value = [Mock(name=n) for n in model_names]
    return mock_wrapper


# ============================================================================
# TEST CLASS: ProviderValidator - OpenAI
# ============================================================================

class TestProviderValidatorOpenAI:
    """Test ProviderValidator OpenAI validation — uses ChatOpenAI wrapper."""

    def test_validate_openai_empty_key(self):
        result = ProviderValidator.validate_openai("")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_validate_openai_whitespace_key(self):
        result = ProviderValidator.validate_openai("   ")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openai_success(self, mock_chat_openai):
        mock_chat_openai.return_value = _make_chat_openai_mock(["gpt-4", "gpt-4o"])

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is True
        assert "gpt-4" in result.models
        assert result.error_message is None

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openai_multiple_models(self, mock_chat_openai):
        mock_chat_openai.return_value = _make_chat_openai_mock(
            [f"model-{i}" for i in range(5)]
        )

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is True
        assert len(result.models) == 5

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openai_no_models(self, mock_chat_openai):
        mock_chat_openai.return_value = _make_chat_openai_mock([])

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is False
        assert "No models found" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openai_authentication_error(self, mock_chat_openai):
        """AuthenticationError str contains 'authentication' — triggers string check."""
        mock_wrapper = Mock()
        mock_wrapper.client.models.list.side_effect = openai.AuthenticationError(
            "Invalid key", response=Mock(status_code=401), body=None
        )
        mock_chat_openai.return_value = mock_wrapper

        result = ProviderValidator.validate_openai("invalid-key")

        assert result.success is False

        assert "invalid key" in result.error_message.lower()

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openai_import_error(self, mock_chat_openai):
        """ImportError is caught by the generic handler — error_message contains the text."""
        mock_chat_openai.side_effect = ImportError("No module named 'openai'")

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is False
        # Generic handler: "Unexpected error: No module named 'openai'"
        assert "No module named 'openai'" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openai_unexpected_error(self, mock_chat_openai):
        mock_chat_openai.side_effect = Exception("Unexpected error")

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is False
        assert "Unexpected error" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openai_constructs_wrapper_with_correct_params(self, mock_chat_openai):
        """Verify ChatOpenAI is called with the expected arguments."""
        mock_chat_openai.return_value = _make_chat_openai_mock(["gpt-4o"])

        ProviderValidator.validate_openai("my-key")

        mock_chat_openai.assert_called_once_with(api_key="my-key", model="gpt-4o")


# ============================================================================
# TEST CLASS: ProviderValidator - Gemini
# ============================================================================

class TestProviderValidatorGemini:
    """Test ProviderValidator Gemini validation — uses ChatGoogleGenerativeAI wrapper."""

    def test_validate_gemini_empty_key(self):
        result = ProviderValidator.validate_gemini("")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_validate_gemini_whitespace_key(self):
        result = ProviderValidator.validate_gemini("   ")

        assert result.success is False

  

    @patch("devdox_ai_sonar.models.llm.ChatGoogleGenerativeAI")
    def test_validate_gemini_no_models(self, mock_chat_gemini):
        mock_chat_gemini.return_value = _make_chat_gemini_mock([])

        result = ProviderValidator.validate_gemini("test-key")

        assert result.success is False
        assert "No models found" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatGoogleGenerativeAI")
    def test_validate_gemini_unexpected_error(self, mock_chat_gemini):
        mock_chat_gemini.side_effect = Exception("Connection error")

        result = ProviderValidator.validate_gemini("test-key")

        assert result.success is False
        assert "Validation error" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatGoogleGenerativeAI")
    def test_validate_gemini_constructs_wrapper_with_correct_params(self, mock_chat_gemini):
        """Verify ChatGoogleGenerativeAI is called with the expected arguments."""
        mock_chat_gemini.return_value = _make_chat_gemini_mock(["models/gemini-pro"])

        ProviderValidator.validate_gemini("my-key")

        mock_chat_gemini.assert_called_once_with(
            model="gemini-1.5-flash", google_api_key="my-key"
        )



# ============================================================================
# TEST CLASS: ProviderValidator - TogetherAI
# ============================================================================

class TestProviderValidatorTogetherAI:
    """Test ProviderValidator TogetherAI validation — uses ChatTogether wrapper."""

    def test_validate_togetherai_empty_key(self):
        result = ProviderValidator.validate_togetherai("")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    @patch("devdox_ai_sonar.models.llm.ChatTogether")
    def test_validate_togetherai_success(self, mock_chat_together):
        mock_chat_together.return_value = _make_chat_together_mock(["mixtral-8x7b"])

        result = ProviderValidator.validate_togetherai("test-key")

        assert result.success is True
        assert "mixtral-8x7b" in result.models

    @patch("devdox_ai_sonar.models.llm.ChatTogether")
    def test_validate_togetherai_multiple_models(self, mock_chat_together):
        mock_chat_together.return_value = _make_chat_together_mock(
            [f"model-{i}" for i in range(4)]
        )

        result = ProviderValidator.validate_togetherai("test-key")

        assert result.success is True
        assert len(result.models) == 4

    @patch("devdox_ai_sonar.models.llm.ChatTogether")
    def test_validate_togetherai_no_models(self, mock_chat_together):
        mock_chat_together.return_value = _make_chat_together_mock([])

        result = ProviderValidator.validate_togetherai("test-key")

        assert result.success is False
        assert "No models found" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatTogether")
    def test_validate_togetherai_unexpected_error(self, mock_chat_together):
        mock_chat_together.side_effect = Exception("Connection refused")

        result = ProviderValidator.validate_togetherai("test-key")

        assert result.success is False
        assert "Validation error" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatTogether")
    def test_validate_togetherai_import_error(self, mock_chat_together):
        """ImportError from missing langchain_together is caught by the generic handler."""
        mock_chat_together.side_effect = ImportError("No module named 'langchain_together'")

        result = ProviderValidator.validate_togetherai("test-key")

        assert result.success is False
        assert "langchain_together" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatTogether")
    def test_validate_togetherai_constructs_wrapper_with_correct_params(
        self, mock_chat_together
    ):
        """Verify ChatTogether is called with the expected arguments."""
        mock_chat_together.return_value = _make_chat_together_mock(["mixtral-8x7b"])

        ProviderValidator.validate_togetherai("my-key")

        mock_chat_together.assert_called_once_with(
            model="meta-llama/Llama-3-70b-chat-hf", together_api_key="my-key"
        )


# ============================================================================
# TEST CLASS: ProviderValidator - OpenRouter
# ============================================================================

class TestProviderValidatorOpenRouter:
    """Test ProviderValidator OpenRouter validation — uses ChatOpenAI with base_url."""

    def test_validate_openrouter_empty_key(self):
        result = ProviderValidator.validate_openrouter("")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_validate_openrouter_whitespace_key(self):
        result = ProviderValidator.validate_openrouter("   ")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openrouter_success(self, mock_chat_openai):
        mock_chat_openai.return_value = _make_chat_openai_mock(
            ["anthropic/claude-sonnet-4"]
        )

        result = ProviderValidator.validate_openrouter("test-key")

        assert result.success is True
        assert "anthropic/claude-sonnet-4" in result.models
        assert result.error_message is None

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openrouter_constructs_wrapper_with_correct_params(
        self, mock_chat_openai
    ):
        """Verify ChatOpenAI is called with base_url and default_headers for OpenRouter."""
        mock_chat_openai.return_value = _make_chat_openai_mock(["anthropic/claude-sonnet-4"])

        ProviderValidator.validate_openrouter("test-key")

        mock_chat_openai.assert_called_once_with(
            api_key="test-key",
            model="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://devdox.ai",
                "X-Title": "DevDox AI Sonar",
            },
        )

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openrouter_multiple_models(self, mock_chat_openai):
        mock_chat_openai.return_value = _make_chat_openai_mock(
            [f"provider/model-{i}" for i in range(5)]
        )

        result = ProviderValidator.validate_openrouter("test-key")

        assert result.success is True
        assert len(result.models) == 5

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openrouter_no_models(self, mock_chat_openai):
        mock_chat_openai.return_value = _make_chat_openai_mock([])

        result = ProviderValidator.validate_openrouter("test-key")

        assert result.success is False
        assert "No models found" in result.error_message


    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openrouter_api_error(self, mock_chat_openai):
        """openai.APIError is caught by the generic handler."""
        mock_wrapper = Mock()
        mock_wrapper.client.models.list.side_effect = openai.APIError(
            "Server error", request=Mock(), body=None
        )
        mock_chat_openai.return_value = mock_wrapper

        result = ProviderValidator.validate_openrouter("test-key")

        assert result.success is False
        # Generic handler produces: "Unexpected error: Server error"
        assert "Server error" in result.error_message

    @patch("devdox_ai_sonar.models.llm.ChatOpenAI")
    def test_validate_openrouter_unexpected_error(self, mock_chat_openai):
        mock_chat_openai.side_effect = Exception("Unexpected error")

        result = ProviderValidator.validate_openrouter("test-key")

        assert result.success is False
        assert "Unexpected error" in result.error_message


# ============================================================================
# TEST CLASS: ProviderValidator - Generic validate()
# ============================================================================

class TestProviderValidatorGeneric:
    """Test ProviderValidator generic validate method"""

    @patch.object(ProviderValidator, "validate_openai")
    def test_validate_routes_to_openai(self, mock_validate):
        mock_validate.return_value = ProviderValidationResult.success_result(["model"])

        result = ProviderValidator.validate(ProviderType.OPENAI, "test-key")

        mock_validate.assert_called_once_with("test-key")
        assert result.success is True

    @patch.object(ProviderValidator, "validate_gemini")
    def test_validate_routes_to_gemini(self, mock_validate):
        mock_validate.return_value = ProviderValidationResult.success_result(["model"])

        ProviderValidator.validate(ProviderType.GEMINI, "test-key")

        mock_validate.assert_called_once_with("test-key")

    @patch.object(ProviderValidator, "validate_togetherai")
    def test_validate_routes_to_togetherai(self, mock_validate):
        mock_validate.return_value = ProviderValidationResult.success_result(["model"])

        ProviderValidator.validate(ProviderType.TOGETHERAI, "test-key")

        mock_validate.assert_called_once_with("test-key")

    @patch.object(ProviderValidator, "validate_openrouter")
    def test_validate_routes_to_openrouter(self, mock_validate):
        mock_validate.return_value = ProviderValidationResult.success_result(["model"])

        result = ProviderValidator.validate(ProviderType.OPENROUTER, "test-key")

        mock_validate.assert_called_once_with("test-key")
        assert result.success is True

    def test_validate_unknown_provider(self):
        class FakeProvider:
            value = "unknown"

        result = ProviderValidator.validate(FakeProvider(), "test-key")

        assert result.success is False
        assert "Unknown provider" in result.error_message


# ============================================================================
# TEST CLASS: LLMProviderConfig
# ============================================================================

class TestLLMProviderConfig:
    """Test LLMProviderConfig Pydantic model"""

    def test_llm_provider_config_creation(self):
        config = LLMProviderConfig(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com",
            models=["gpt-4", "gpt-3.5-turbo"],
        )

        assert config.name == "openai"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com"
        assert len(config.models) == 2

    def test_llm_provider_config_default_values(self):
        config = LLMProviderConfig(
            name="gemini",
            api_key="key",
            base_url="https://api.gemini.com",
            models=[],
        )

        assert config.max_requests_per_minute == 60
        assert config.max_tokens_per_minute == 100000
        assert config.timeout == 30

    def test_llm_provider_config_custom_limits(self):
        config = LLMProviderConfig(
            name="openai",
            api_key="key",
            base_url="url",
            models=[],
            max_requests_per_minute=120,
            max_tokens_per_minute=200000,
            timeout=60,
        )

        assert config.max_requests_per_minute == 120
        assert config.max_tokens_per_minute == 200000
        assert config.timeout == 60

    def test_llm_provider_config_validation(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMProviderConfig(name="openai", api_key="key")  # missing base_url, models

    def test_llm_provider_config_to_dict(self):
        config = LLMProviderConfig(
            name="openai", api_key="key", base_url="url", models=["model"]
        )

        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["name"] == "openai"
        assert "max_requests_per_minute" in data

    def test_llm_provider_config_from_dict(self):
        data = {
            "name": "gemini",
            "api_key": "key",
            "base_url": "url",
            "models": ["gemini-pro"],
        }

        config = LLMProviderConfig(**data)

        assert config.name == "gemini"
        assert config.models == ["gemini-pro"]