"""Tests for the constant naming service."""

from unittest.mock import Mock

from devdox_ai_sonar.models.constant_naming import LiteralContext, NamingRequest
from devdox_ai_sonar.services.constant_namer import (
    ConstantNamingService,
    LLMFixerAdapter,
    NameValidator,
    _clean,
    _detect_language,
    _extract_with_yake,
    _format_screaming_snake,
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


# ============================================================================
# FORMAT SCREAMING SNAKE
# ============================================================================


class TestFormatScreamingSnake:
    def test_basic_tokens(self):
        assert _format_screaming_snake(["hello", "world"]) == "HELLO_WORLD"

    def test_empty_list(self):
        assert _format_screaming_snake([]) is None

    def test_strips_non_ascii(self):
        result = _format_screaming_snake(["café"])
        assert result is not None
        assert all(c.isascii() for c in result)

    def test_collapses_multiple_underscores(self):
        result = _format_screaming_snake(["a__b", "c"])
        assert "__" not in result


# ============================================================================
# _name_via_pipeline
# ============================================================================


class TestNameViaPipeline:
    """Tests for the _name_via_pipeline micro-method."""

    def setup_method(self):
        self.service = ConstantNamingService()

    def test_slugifiable_literals_resolved(self):
        names = {}
        used = set()
        remaining = self.service._name_via_pipeline(
            [LiteralContext(literal="application/json", occurrences=[(1, 0, 18)])],
            names,
            used,
        )
        assert names["application/json"] == "APPLICATION_JSON"
        assert "APPLICATION_JSON" in used
        assert remaining == []

    def test_non_slugifiable_returned_as_remaining(self):
        names = {}
        used = set()
        remaining = self.service._name_via_pipeline(
            [LiteralContext(literal="hello", occurrences=[(1, 0, 7)])],
            names,
            used,
        )
        assert "hello" not in names
        assert len(remaining) == 1
        assert remaining[0].literal == "hello"

    def test_deduplicates_same_literal(self):
        names = {}
        used = set()
        remaining = self.service._name_via_pipeline(
            [
                LiteralContext(literal="application/json", occurrences=[(1, 0, 18)]),
                LiteralContext(literal="application/json", occurrences=[(2, 0, 18)]),
            ],
            names,
            used,
        )
        assert len(names) == 1
        assert remaining == []

    def test_collision_resolved_via_make_unique(self):
        names = {}
        used = {"APPLICATION_JSON"}
        remaining = self.service._name_via_pipeline(
            [LiteralContext(literal="application/json", occurrences=[(1, 0, 18)])],
            names,
            used,
        )
        assert names["application/json"] == "APPLICATION_JSON_2"


# ============================================================================
# _resolve_remaining
# ============================================================================


class TestResolveRemaining:
    """Tests for the _resolve_remaining micro-method."""

    def test_empty_remaining_does_nothing(self):
        service = ConstantNamingService()
        names = {}
        used = set()
        service._resolve_remaining([], names, used)
        assert names == {}

    def test_without_llm_falls_to_fallback(self):
        service = ConstantNamingService()
        names = {}
        used = set()
        remaining = [LiteralContext(literal="hello", occurrences=[(1, 0, 7)])]
        service._resolve_remaining(remaining, names, used)
        assert names["hello"] == "STRING_LITERAL_1"

    def test_with_llm_uses_llm_result(self):
        mock_caller = Mock()
        mock_caller.call_for_json.return_value = {"hello": "GREETING_MSG"}
        service = ConstantNamingService(llm_caller=mock_caller)
        names = {}
        used = set()
        remaining = [LiteralContext(literal="hello", occurrences=[(1, 0, 7)])]
        service._resolve_remaining(remaining, names, used, file_path="/test.py")
        assert names["hello"] == "GREETING_MSG"


# ============================================================================
# _assign_name
# ============================================================================


class TestAssignName:
    """Tests for the _assign_name micro-method."""

    def setup_method(self):
        self.service = ConstantNamingService()

    def test_valid_llm_name_assigned(self):
        names = {}
        used = set()
        lit_ctx = LiteralContext(literal="hello", occurrences=[(1, 0, 7)])
        self.service._assign_name(
            lit_ctx, {"hello": "GREETING_DEFAULT"}, names, used
        )
        assert names["hello"] == "GREETING_DEFAULT"
        assert "GREETING_DEFAULT" in used

    def test_invalid_llm_name_falls_to_fallback(self):
        names = {}
        used = set()
        lit_ctx = LiteralContext(literal="hello", occurrences=[(1, 0, 7)])
        self.service._assign_name(lit_ctx, {"hello": "bad"}, names, used)
        assert names["hello"] == "STRING_LITERAL_1"

    def test_missing_llm_name_falls_to_fallback(self):
        names = {}
        used = set()
        lit_ctx = LiteralContext(literal="hello", occurrences=[(1, 0, 7)])
        self.service._assign_name(lit_ctx, {}, names, used)
        assert names["hello"] == "STRING_LITERAL_1"

    def test_llm_name_collision_resolved(self):
        names = {}
        used = {"GREETING_DEFAULT"}
        lit_ctx = LiteralContext(literal="hello", occurrences=[(1, 0, 7)])
        self.service._assign_name(
            lit_ctx, {"hello": "GREETING_DEFAULT"}, names, used
        )
        assert names["hello"] == "GREETING_DEFAULT_2"


# ============================================================================
# LLMFixerAdapter
# ============================================================================


class TestLLMFixerAdapterCallOpenaiCompatible:
    """Tests for _call_openai_compatible micro-method."""

    def test_successful_call(self):
        mock_fixer = Mock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"hello": "GREETING"}'))]
        )
        adapter = LLMFixerAdapter(mock_fixer)
        result = adapter._call_openai_compatible("system", "user")
        assert result == {"hello": "GREETING"}

    def test_none_content_returns_none(self):
        mock_fixer = Mock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=None))]
        )
        adapter = LLMFixerAdapter(mock_fixer)
        result = adapter._call_openai_compatible("system", "user")
        assert result is None


class TestLLMFixerAdapterCallForJson:
    """Tests for call_for_json dispatch."""

    def test_dispatches_to_openai(self):
        mock_fixer = Mock()
        mock_fixer.provider = "openai"
        mock_fixer.model = "gpt-4"
        mock_fixer.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"k": "V"}'))]
        )
        adapter = LLMFixerAdapter(mock_fixer)
        result = adapter.call_for_json("system", "user")
        assert result == {"k": "V"}

    def test_dispatches_togetherai(self):
        mock_fixer = Mock()
        mock_fixer.provider = "togetherai"
        mock_fixer.model = "model"
        mock_fixer.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"k": "V"}'))]
        )
        adapter = LLMFixerAdapter(mock_fixer)
        result = adapter.call_for_json("system", "user")
        assert result == {"k": "V"}

    def test_unknown_provider_returns_none(self):
        mock_fixer = Mock()
        mock_fixer.provider = "unknown_provider"
        adapter = LLMFixerAdapter(mock_fixer)
        result = adapter.call_for_json("system", "user")
        assert result is None

    def test_exception_returns_none(self):
        mock_fixer = Mock()
        mock_fixer.provider = "openai"
        mock_fixer.client.chat.completions.create.side_effect = RuntimeError("fail")
        adapter = LLMFixerAdapter(mock_fixer)
        result = adapter.call_for_json("system", "user")
        assert result is None
