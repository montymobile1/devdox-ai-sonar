

import pytest
from unittest.mock import Mock, patch, MagicMock
import openai
from devdox_ai_sonar.models.llm import (
    ProviderType,
    ProviderValidationResult,
    ProviderValidator,
    LLMProviderConfig
)


# ============================================================================
# TEST CLASS: ProviderType
# ============================================================================

class TestProviderType:
    """Test ProviderType enum"""

    def test_provider_type_values(self):
        """Test all provider type values"""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.GEMINI.value == "gemini"
        assert ProviderType.TOGETHERAI.value == "togetherai"

    def test_provider_type_from_string(self):
        """Test creating ProviderType from string"""
        assert ProviderType("openai") == ProviderType.OPENAI
        assert ProviderType("gemini") == ProviderType.GEMINI
        assert ProviderType("togetherai") == ProviderType.TOGETHERAI

    def test_provider_type_choices(self):
        """Test choices() class method"""
        choices = ProviderType.choices()

        assert isinstance(choices, list)
        assert "openai" in choices
        assert "gemini" in choices
        assert "togetherai" in choices
        assert len(choices) == 3

    def test_provider_type_choices_all_strings(self):
        """Test all choices are strings"""
        choices = ProviderType.choices()

        assert all(isinstance(choice, str) for choice in choices)

    def test_provider_type_enum_iteration(self):
        """Test iterating over provider types"""
        providers = list(ProviderType)

        assert len(providers) == 3
        assert ProviderType.OPENAI in providers
        assert ProviderType.GEMINI in providers
        assert ProviderType.TOGETHERAI in providers

    def test_provider_type_equality(self):
        """Test provider type equality"""
        assert ProviderType.OPENAI == ProviderType.OPENAI
        assert ProviderType.OPENAI != ProviderType.GEMINI

    def test_provider_type_string_representation(self):
        """Test string representation"""
        assert str(ProviderType.OPENAI) == "ProviderType.OPENAI"


# ============================================================================
# TEST CLASS: ProviderValidationResult
# ============================================================================

class TestProviderValidationResult:
    """Test ProviderValidationResult dataclass"""

    def test_validation_result_creation_success(self):
        """Test creating successful validation result"""
        result = ProviderValidationResult(
            success=True,
            models=["gpt-4", "gpt-3.5-turbo"],
            error_message=None
        )

        assert result.success is True
        assert len(result.models) == 2
        assert result.error_message is None

    def test_validation_result_creation_failure(self):
        """Test creating failed validation result"""
        result = ProviderValidationResult(
            success=False,
            models=[],
            error_message="Invalid API key"
        )

        assert result.success is False
        assert result.models == []
        assert result.error_message == "Invalid API key"

    def test_success_result_class_method(self):
        """Test success_result class method"""
        models = ["model-1", "model-2", "model-3"]
        result = ProviderValidationResult.success_result(models)

        assert result.success is True
        assert result.models == models
        assert result.error_message is None

    def test_success_result_empty_models(self):
        """Test success_result with empty models list"""
        result = ProviderValidationResult.success_result([])

        assert result.success is True
        assert result.models == []

    def test_failure_result_class_method(self):
        """Test failure_result class method"""
        error = "Authentication failed"
        result = ProviderValidationResult.failure_result(error)

        assert result.success is False
        assert result.models == []
        assert result.error_message == error

    def test_failure_result_empty_error(self):
        """Test failure_result with empty error message"""
        result = ProviderValidationResult.failure_result("")

        assert result.success is False
        assert result.error_message == ""

    def test_validation_result_is_dataclass(self):
        """Test that ProviderValidationResult is a dataclass"""
        result = ProviderValidationResult.success_result(["model"])

        assert hasattr(result, '__dataclass_fields__')

    def test_validation_result_default_error_message(self):
        """Test default error_message is None"""
        result = ProviderValidationResult(success=True, models=["model"])

        assert result.error_message is None


# ============================================================================
# TEST CLASS: ProviderValidator - OpenAI
# ============================================================================

class TestProviderValidatorOpenAI:
    """Test ProviderValidator OpenAI validation"""

    def test_validate_openai_empty_key(self):
        """Test OpenAI validation with empty API key"""
        result = ProviderValidator.validate_openai("")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_validate_openai_whitespace_key(self):
        """Test OpenAI validation with whitespace-only key"""
        result = ProviderValidator.validate_openai("   ")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    @patch('devdox_ai_sonar.models.llm.OpenAI')
    def test_validate_openai_success(self, mock_openai):
        """Test successful OpenAI validation"""
        # Mock client and models
        mock_client = Mock()
        mock_model = Mock()
        mock_model.id = "gpt-4"
        mock_client.models.list.return_value.data = [mock_model]
        mock_openai.return_value = mock_client

        result = ProviderValidator.validate_openai("test-key")
        print("result ", result )
        assert result.success is True
        assert "gpt-4" in result.models
        assert result.error_message is None

    @patch('devdox_ai_sonar.models.llm.OpenAI')
    def test_validate_openai_multiple_models(self, mock_openai):
        """Test OpenAI validation returns multiple models"""
        mock_client = Mock()
        models_data = [Mock(id=f"model-{i}") for i in range(5)]
        mock_client.models.list.return_value.data = models_data
        mock_openai.return_value = mock_client

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is True
        assert len(result.models) == 5

    @patch('devdox_ai_sonar.models.llm.OpenAI')
    def test_validate_openai_no_models(self, mock_openai):
        """Test OpenAI validation with no models"""
        mock_client = Mock()
        mock_client.models.list.return_value.data = []
        mock_openai.return_value = mock_client

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is False
        assert "No models found" in result.error_message

    @patch('devdox_ai_sonar.models.llm.OpenAI')
    def test_validate_openai_authentication_error(self, mock_openai_class):
        """Test OpenAI authentication error"""
        mock_client = Mock()
        mock_response = Mock(status_code=401)
        error = openai.AuthenticationError(
            "Invalid key",
            response=mock_response,
            body=None,
        )
        # Raise exception when models.list() is called
        mock_client.models.list.side_effect = error
        mock_openai_class.return_value = mock_client

        result = ProviderValidator.validate_openai("invalid-key")
        assert result.success is False
        assert "authentication failed" in result.error_message.lower()

    @patch('devdox_ai_sonar.models.llm.OpenAI')
    def test_validate_openai_authentication_error(self, mock_openai_class):
        """Test OpenAI authentication error"""
        mock_client = Mock()
        mock_response = Mock(status_code=401)
        error = openai.AuthenticationError(
            "Invalid key",
            response=mock_response,
            body=None,
        )
        # Raise exception when models.list() is called
        mock_client.models.list.side_effect = error
        mock_openai_class.return_value = mock_client

        result = ProviderValidator.validate_openai("invalid-key")
        assert result.success is False
        assert "authentication failed" in result.error_message.lower()

    @patch('devdox_ai_sonar.models.llm.OpenAI')
    def test_validate_openai_import_error(self, mock_openai_class):
        """Test OpenAI validation when package not installed"""
        # Simulate ImportError when trying to instantiate OpenAI client
        mock_openai_class.side_effect = ImportError("No module named 'openai'")

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is False
        assert "not installed" in result.error_message.lower()

    @patch('devdox_ai_sonar.models.llm.OpenAI')
    def test_validate_openai_unexpected_error(self, mock_openai):
        """Test OpenAI validation with unexpected error"""
        mock_openai.side_effect = Exception("Unexpected error")

        result = ProviderValidator.validate_openai("test-key")

        assert result.success is False
        assert "Unexpected error" in result.error_message


# ============================================================================
# TEST CLASS: ProviderValidator - Gemini
# ============================================================================

class TestProviderValidatorGemini:
    """Test ProviderValidator Gemini validation"""

    def test_validate_gemini_empty_key(self):
        """Test Gemini validation with empty API key"""
        result = ProviderValidator.validate_gemini("")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_validate_gemini_whitespace_key(self):
        """Test Gemini validation with whitespace-only key"""
        result = ProviderValidator.validate_gemini("   ")

        assert result.success is False

    @patch('devdox_ai_sonar.models.llm.genai')
    def test_validate_gemini_success(self, mock_genai):
        """Test successful Gemini validation"""
        mock_client = Mock()
        mock_model = Mock()
        mock_model.name = "models/gemini-pro"
        mock_client.models.list.return_value = [mock_model]
        mock_genai.Client.return_value = mock_client

        result = ProviderValidator.validate_gemini("test-key")

        assert result.success is True
        assert "gemini-pro" in result.models

    @patch('devdox_ai_sonar.models.llm.genai')
    def test_validate_gemini_strips_models_prefix(self, mock_genai):
        """Test Gemini validation strips 'models/' prefix"""
        mock_client = Mock()
        mock_model1 = Mock()
        mock_model1.name = "models/gemini-pro"
        mock_model2 = Mock()
        mock_model2.name = "models/gemini-flash"

        mock_client.models.list.return_value = [mock_model1, mock_model2]
        mock_genai.Client.return_value = mock_client

        result = ProviderValidator.validate_gemini("test-key")

        assert result.success is True
        assert "gemini-pro" in result.models
        assert "gemini-flash" in result.models
        assert all("models/" not in model for model in result.models)

    @patch('devdox_ai_sonar.models.llm.genai')
    def test_validate_gemini_no_models(self, mock_genai):
        """Test Gemini validation with no models"""
        mock_client = Mock()
        mock_client.models.list.return_value = []
        mock_genai.Client.return_value = mock_client

        result = ProviderValidator.validate_gemini("test-key")

        assert result.success is False
        assert "No models found" in result.error_message


    @patch('devdox_ai_sonar.models.llm.genai')
    def test_validate_gemini_unexpected_error(self, mock_genai):
        """Test Gemini validation with unexpected error"""
        mock_genai.Client.side_effect = Exception("Connection error")

        result = ProviderValidator.validate_gemini("test-key")

        assert result.success is False
        assert "Validation error" in result.error_message


# ============================================================================
# TEST CLASS: ProviderValidator - TogetherAI
# ============================================================================

class TestProviderValidatorTogetherAI:
    """Test ProviderValidator TogetherAI validation"""

    def test_validate_togetherai_empty_key(self):
        """Test TogetherAI validation with empty API key"""
        result = ProviderValidator.validate_togetherai("")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    @patch('devdox_ai_sonar.models.llm.Together')
    def test_validate_togetherai_success(self, mock_together):
        """Test successful TogetherAI validation"""
        mock_client = Mock()
        mock_model = Mock()
        mock_model.id = "mixtral-8x7b"
        mock_client.models.list.return_value = [mock_model]
        mock_together.return_value = mock_client

        result = ProviderValidator.validate_togetherai("test-key")

        assert result.success is True
        assert "mixtral-8x7b" in result.models

    @patch('devdox_ai_sonar.models.llm.Together')
    def test_validate_togetherai_no_models(self, mock_together):
        """Test TogetherAI validation with no models"""
        mock_client = Mock()
        mock_client.models.list.return_value = []
        mock_together.return_value = mock_client

        result = ProviderValidator.validate_togetherai("test-key")

        assert result.success is False

    def test_validate_togetherai_import_error(self):
        """Test TogetherAI validation when package not installed"""
        with patch.dict('sys.modules', {'together': None}):
            result = ProviderValidator.validate_togetherai("test-key")

            if not result.success:
                assert "not installed" in result.error_message.lower()


# ============================================================================
# TEST CLASS: ProviderValidator - Generic validate()
# ============================================================================

class TestProviderValidatorGeneric:
    """Test ProviderValidator generic validate method"""

    @patch.object(ProviderValidator, 'validate_openai')
    def test_validate_routes_to_openai(self, mock_validate):
        """Test validate() routes to validate_openai"""
        mock_validate.return_value = ProviderValidationResult.success_result(["model"])

        result = ProviderValidator.validate(ProviderType.OPENAI, "test-key")

        mock_validate.assert_called_once_with("test-key")
        assert result.success is True

    @patch.object(ProviderValidator, 'validate_gemini')
    def test_validate_routes_to_gemini(self, mock_validate):
        """Test validate() routes to validate_gemini"""
        mock_validate.return_value = ProviderValidationResult.success_result(["model"])

        result = ProviderValidator.validate(ProviderType.GEMINI, "test-key")

        mock_validate.assert_called_once_with("test-key")

    @patch.object(ProviderValidator, 'validate_togetherai')
    def test_validate_routes_to_togetherai(self, mock_validate):
        """Test validate() routes to validate_togetherai"""
        mock_validate.return_value = ProviderValidationResult.success_result(["model"])

        result = ProviderValidator.validate(ProviderType.TOGETHERAI, "test-key")

        mock_validate.assert_called_once_with("test-key")

    def test_validate_unknown_provider(self):
        """Test validate() with unknown provider"""

        # Create a fake provider type
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
        """Test creating LLMProviderConfig"""
        config = LLMProviderConfig(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com",
            models=["gpt-4", "gpt-3.5-turbo"]
        )

        assert config.name == "openai"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com"
        assert len(config.models) == 2

    def test_llm_provider_config_default_values(self):
        """Test default values for optional fields"""
        config = LLMProviderConfig(
            name="gemini",
            api_key="key",
            base_url="https://api.gemini.com",
            models=[]
        )

        assert config.max_requests_per_minute == 60
        assert config.max_tokens_per_minute == 100000
        assert config.timeout == 30

    def test_llm_provider_config_custom_limits(self):
        """Test custom rate limits"""
        config = LLMProviderConfig(
            name="openai",
            api_key="key",
            base_url="url",
            models=[],
            max_requests_per_minute=120,
            max_tokens_per_minute=200000,
            timeout=60
        )

        assert config.max_requests_per_minute == 120
        assert config.max_tokens_per_minute == 200000
        assert config.timeout == 60

    def test_llm_provider_config_validation(self):
        """Test Pydantic validation"""
        with pytest.raises(Exception):  # ValidationError
            LLMProviderConfig(
                name="openai",
                api_key="key"
                # Missing required fields
            )

    def test_llm_provider_config_to_dict(self):
        """Test converting to dictionary"""
        config = LLMProviderConfig(
            name="openai",
            api_key="key",
            base_url="url",
            models=["model"]
        )

        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["name"] == "openai"
        assert "max_requests_per_minute" in data

    def test_llm_provider_config_from_dict(self):
        """Test creating from dictionary"""
        data = {
            "name": "gemini",
            "api_key": "key",
            "base_url": "url",
            "models": ["gemini-pro"]
        }

        config = LLMProviderConfig(**data)

        assert config.name == "gemini"
        assert config.models == ["gemini-pro"]


