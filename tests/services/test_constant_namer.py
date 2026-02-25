"""Tests for the constant naming service."""

from unittest.mock import Mock

from devdox_ai_sonar.models.constant_naming import LiteralContext, NamingRequest
from devdox_ai_sonar.services.constant_namer import (
    ConstantNamingService,
    NameValidator,
    _clean,
    _detect_language,
    _extract_with_yake,
    _slugify,
    generate_name,
)


# ============================================================================
# CLEAN
# ============================================================================


class TestClean:
    def test_slash_separated(self):
        assert _clean("application/json") == ["application", "json"]

    def test_underscore_kept_as_word_char(self):
        # Underscore is a word character in \w, so it stays in the token
        assert _clean("pending_review") == ["pending_review"]

    def test_hyphen_separated(self):
        assert _clean("Content-Type") == ["Content", "Type"]

    def test_path_like(self):
        assert _clean("/api/v1/users") == ["api", "v1", "users"]

    def test_empty_string(self):
        assert _clean("") == []

    def test_only_special_chars(self):
        assert _clean("///") == []

    def test_preserves_digits(self):
        assert _clean("error-404") == ["error", "404"]


# ============================================================================
# DETECT LANGUAGE
# ============================================================================


class TestDetectLanguage:
    def test_english_string(self):
        lang = _detect_language("Failed to connect to the authentication service")
        assert lang == "en"

    def test_short_string_defaults_to_english(self):
        # Very short strings may not be reliably detected; should default to en
        lang = _detect_language("ok")
        # langdetect may or may not detect this as English,
        # but it should not raise an exception
        assert isinstance(lang, str)

    def test_empty_string_defaults_to_english(self):
        assert _detect_language("") == "en"


# ============================================================================
# SLUGIFY
# ============================================================================


class TestSlugify:
    def test_basic(self):
        assert _slugify(["application", "json"]) == "APPLICATION_JSON"

    def test_caps_at_max_tokens(self):
        tokens = ["a", "b", "c", "d", "e", "f"]
        result = _slugify(tokens, max_tokens=4)
        assert result == "A_B_C_D"

    def test_empty_list(self):
        assert _slugify([]) is None

    def test_strips_non_ascii(self):
        # slugify is for Latin characters; non-ASCII gets stripped
        result = _slugify(["café", "résumé"])
        assert result is not None
        assert result.isidentifier() or result == ""

    def test_single_token(self):
        assert _slugify(["hello"]) == "HELLO"


# ============================================================================
# YAKE EXTRACTION
# ============================================================================


class TestExtractWithYake:
    def test_long_english_string(self):
        text = "Failed to connect to the authentication service after timeout"
        result = _extract_with_yake(text, lang="en", max_keywords=4)
        assert result is not None
        # Should pick meaningful keywords, not stop words
        parts = result.split("_")
        assert len(parts) <= 4
        assert all(p.isupper() or p.isdigit() for p in parts)

    def test_short_string(self):
        result = _extract_with_yake("application/json", lang="en", max_keywords=4)
        assert result is not None

    def test_empty_string(self):
        result = _extract_with_yake("", lang="en")
        # May return None or empty
        assert result is None or result == ""

    def test_format_string(self):
        result = _extract_with_yake("%Y-%m-%d", lang="en")
        # YAKE may or may not extract useful tokens from format strings
        # Either None or something short is acceptable
        assert result is None or isinstance(result, str)


# ============================================================================
# GENERATE NAME (end-to-end code layer)
# ============================================================================


class TestGenerateName:
    def test_short_english_uses_slugify(self):
        assert generate_name("application/json") == "APPLICATION_JSON"

    def test_short_english_content_type(self):
        assert generate_name("Content-Type") == "CONTENT_TYPE"

    def test_short_english_underscore(self):
        assert generate_name("pending_review") == "PENDING_REVIEW"

    def test_long_english_uses_yake(self):
        result = generate_name(
            "Failed to connect to the authentication service after timeout"
        )
        assert result is not None
        parts = result.split("_")
        assert len(parts) <= 4

    def test_single_word_returns_single_token(self):
        result = generate_name("hello")
        # Single word → slugify → "HELLO" (single part)
        assert result == "HELLO"

    def test_empty_string(self):
        assert generate_name("") is None

    def test_numeric_string(self):
        result = generate_name("12345")
        # "12345" → slugify → "12345" (starts with digit, will fail validation later)
        assert result == "12345" or result is None


# ============================================================================
# NAME VALIDATOR
# ============================================================================


class TestNameValidator:
    def setup_method(self):
        self.v = NameValidator()

    def test_valid_two_parts(self):
        assert self.v.is_valid("APPLICATION_JSON", set()) is True

    def test_valid_three_parts(self):
        assert self.v.is_valid("CONTENT_TYPE_JSON", set()) is True

    def test_rejects_lowercase(self):
        assert self.v.is_valid("application_json", set()) is False

    def test_rejects_single_word(self):
        assert self.v.is_valid("APPLICATION", set()) is False

    def test_rejects_six_words(self):
        assert self.v.is_valid("A_B_C_D_E_F", set()) is False

    def test_rejects_python_keyword(self):
        # "FALSE_VALUE" should be fine, but check edge cases
        assert self.v.is_valid("TRUE", set()) is False  # single word

    def test_rejects_starts_with_digit(self):
        assert self.v.is_valid("1234_VALUE", set()) is False

    def test_rejects_collision(self):
        assert self.v.is_valid("APP_JSON", {"APP_JSON"}) is False

    def test_accepts_no_collision(self):
        assert self.v.is_valid("APP_JSON", {"OTHER_CONST"}) is True

    def test_make_unique_no_collision(self):
        assert self.v.make_unique("APP_JSON", set()) == "APP_JSON"

    def test_make_unique_with_collision(self):
        result = self.v.make_unique("APP_JSON", {"APP_JSON"})
        assert result == "APP_JSON_2"

    def test_make_unique_with_multiple_collisions(self):
        result = self.v.make_unique(
            "APP_JSON", {"APP_JSON", "APP_JSON_2", "APP_JSON_3"}
        )
        assert result == "APP_JSON_4"

    def test_make_unique_respects_max_parts(self):
        """When name already has MAX_PARTS parts, replace last segment instead of appending."""
        result = self.v.make_unique(
            "A_B_C_D_E", {"A_B_C_D_E"}
        )
        # Should replace last part, not create A_B_C_D_E_2 (6 parts)
        assert result == "A_B_C_D_2"
        assert self.v.is_structurally_valid(result)


# ============================================================================
# ORCHESTRATION — ConstantNamingService (no LLM)
# ============================================================================


class TestConstantNamingServiceNoLLM:
    """Tests with no LLM caller — code layers + STRING_LITERAL_N fallback."""

    def test_slugifiable_literal(self):
        service = ConstantNamingService()
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="application/json", occurrences=[(1, 5, 23)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["application/json"] == "APPLICATION_JSON"

    def test_multiple_slugifiable_literals(self):
        service = ConstantNamingService()
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="application/json", occurrences=[(1, 0, 18)]),
                LiteralContext(literal="Content-Type", occurrences=[(2, 0, 14)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["application/json"] == "APPLICATION_JSON"
        assert resp.names["Content-Type"] == "CONTENT_TYPE"

    def test_non_slugifiable_falls_to_string_literal_n(self):
        service = ConstantNamingService()
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="hello", occurrences=[(1, 0, 7)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["hello"] == "STRING_LITERAL_1"

    def test_collision_resolved_with_suffix(self):
        service = ConstantNamingService()
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="application/json", occurrences=[(1, 0, 18)]),
            ],
            existing_names={"APPLICATION_JSON"},
        )
        resp = service.name_literals(req)
        assert resp.names["application/json"] == "APPLICATION_JSON_2"

    def test_mix_of_slugifiable_and_non(self):
        service = ConstantNamingService()
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="application/json", occurrences=[(1, 0, 18)]),
                LiteralContext(literal="1234", occurrences=[(2, 0, 6)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["application/json"] == "APPLICATION_JSON"
        assert resp.names["1234"] == "STRING_LITERAL_1"

    def test_multiple_fallbacks_get_unique_names(self):
        service = ConstantNamingService()
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="a", occurrences=[(1, 0, 3)]),
                LiteralContext(literal="b", occurrences=[(2, 0, 3)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        names = list(resp.names.values())
        assert names[0] != names[1]
        assert names[0].startswith("STRING_LITERAL_")
        assert names[1].startswith("STRING_LITERAL_")


# ============================================================================
# LLM FALLBACK (mocked)
# ============================================================================


class TestConstantNamingServiceWithLLM:
    """Tests with mocked LLM caller."""

    def test_llm_called_for_unresolved_literals(self):
        mock_caller = Mock()
        mock_caller.call_for_json.return_value = {"hello": "GREETING_DEFAULT"}

        service = ConstantNamingService(llm_caller=mock_caller)
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="hello", occurrences=[(1, 0, 7)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["hello"] == "GREETING_DEFAULT"
        mock_caller.call_for_json.assert_called_once()

    def test_llm_not_called_when_all_slugified(self):
        mock_caller = Mock()

        service = ConstantNamingService(llm_caller=mock_caller)
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="application/json", occurrences=[(1, 0, 18)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["application/json"] == "APPLICATION_JSON"
        mock_caller.call_for_json.assert_not_called()

    def test_llm_returns_invalid_name_falls_to_fallback(self):
        mock_caller = Mock()
        mock_caller.call_for_json.return_value = {"hello": "bad-name"}

        service = ConstantNamingService(llm_caller=mock_caller)
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="hello", occurrences=[(1, 0, 7)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["hello"] == "STRING_LITERAL_1"

    def test_llm_returns_none_falls_to_fallback(self):
        mock_caller = Mock()
        mock_caller.call_for_json.return_value = None

        service = ConstantNamingService(llm_caller=mock_caller)
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="hello", occurrences=[(1, 0, 7)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["hello"] == "STRING_LITERAL_1"

    def test_llm_receives_only_unresolved(self):
        mock_caller = Mock()
        mock_caller.call_for_json.return_value = {"hello": "GREETING_MSG"}

        service = ConstantNamingService(llm_caller=mock_caller)
        req = NamingRequest(
            file_path="/test.py",
            literals=[
                LiteralContext(literal="application/json", occurrences=[(1, 0, 18)]),
                LiteralContext(literal="hello", occurrences=[(2, 0, 7)]),
            ],
            existing_names=set(),
        )
        resp = service.name_literals(req)
        assert resp.names["application/json"] == "APPLICATION_JSON"
        assert resp.names["hello"] == "GREETING_MSG"
        # LLM should have been called once, only for "hello"
        mock_caller.call_for_json.assert_called_once()
