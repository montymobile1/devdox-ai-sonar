"""Decoupled constant naming service for python:S1192 fixes.

Generates meaningful SCREAMING_SNAKE_CASE names for duplicated string literals
using a pipeline: clean → detect language → slugify/YAKE → LLM fallback.
"""

import keyword
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

import yake
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langdetect import detect
from langdetect import DetectorFactory
from openhands.sdk import Message, TextContent
from pydantic import TypeAdapter

from devdox_ai_sonar.models.constant_naming import (
    LiteralContext,
    NamingRequest,
    NamingResponse,
)

DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

_names_adapter: TypeAdapter[Dict[str, str]] = TypeAdapter(Dict[str, str])

MAX_WORDS_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Step 1 — Clean
# ---------------------------------------------------------------------------


def _clean(literal: str) -> List[str]:
    """Split the literal on non-word characters, preserving Unicode letters."""
    tokens = re.split(r"[^\w]+", literal)
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Step 2 — Detect language
# ---------------------------------------------------------------------------


def _detect_language(literal: str) -> str:
    """Return ISO 639-1 language code. Defaults to ``"en"`` on failure."""
    try:
        lang: str = detect(literal)
        return lang
    except Exception:
        return "en"


# ---------------------------------------------------------------------------
# Step 3 — Format tokens as SCREAMING_SNAKE_CASE
# ---------------------------------------------------------------------------


def _format_screaming_snake(tokens: List[str]) -> Optional[str]:
    """Join tokens into a SCREAMING_SNAKE_CASE name, stripping non-ASCII."""
    if not tokens:
        return None
    name = "_".join(tokens).upper()
    name = re.sub(r"[^A-Z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else None


# ---------------------------------------------------------------------------
# Step 4 — Slugify (short strings)
# ---------------------------------------------------------------------------


def _slugify(tokens: List[str], max_tokens: int = MAX_WORDS_THRESHOLD) -> Optional[str]:
    """Join the first *max_tokens* tokens as SCREAMING_SNAKE_CASE."""
    return _format_screaming_snake(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Step 5 — YAKE keyword extraction (long strings)
# ---------------------------------------------------------------------------


def _extract_with_yake(
    literal: str,
    lang: str = "en",
    max_keywords: int = MAX_WORDS_THRESHOLD,
) -> Optional[str]:
    """Use YAKE to pick the most important keywords and form a name."""
    try:
        kw_extractor = yake.KeywordExtractor(
            lan=lang,
            n=1,
            top=max_keywords,
            dedupLim=0.9,
            windowsSize=1,
        )
        keywords = kw_extractor.extract_keywords(literal)
    except Exception:
        return None

    return _format_screaming_snake([kw[0] for kw in keywords])


# ---------------------------------------------------------------------------
# Combined name generator
# ---------------------------------------------------------------------------


def generate_name(
    literal: str,
    max_words: int = MAX_WORDS_THRESHOLD,
) -> Optional[str]:
    """Generate a SCREAMING_SNAKE_CASE name from a literal value.

    Pipeline:
    1. Clean the string into tokens.
    2. If short (≤ *max_words* tokens) → direct slugify (language-agnostic).
    3. Otherwise → detect language → YAKE keyword extraction.

    Short strings are always slugified because ``langdetect`` is unreliable
    for short technical strings and YAKE may drop useful tokens.

    Returns ``None`` when no valid name can be produced.
    """
    tokens = _clean(literal)

    if len(tokens) <= max_words:
        return _slugify(tokens, max_tokens=max_words)

    lang = _detect_language(literal)
    return _extract_with_yake(literal, lang=lang, max_keywords=max_words)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_SCREAMING_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")


class NameValidator:
    """Validates and deduplicates generated constant names."""

    MIN_PARTS = 2
    MAX_PARTS = 5

    def is_structurally_valid(self, name: str) -> bool:
        """Return ``True`` when *name* has valid structure (ignoring collisions)."""
        if not name:
            return False
        if not _SCREAMING_SNAKE_RE.match(name):
            return False
        parts = name.split("_")
        if len(parts) < self.MIN_PARTS or len(parts) > self.MAX_PARTS:
            return False
        if not name.isidentifier():
            return False
        if keyword.iskeyword(name) or keyword.iskeyword(name.lower()):
            return False
        return True

    def is_valid(self, name: str, existing_names: Set[str]) -> bool:
        """Return ``True`` when *name* passes all validation rules."""
        return self.is_structurally_valid(name) and name not in existing_names

    def make_unique(self, name: str, existing_names: Set[str]) -> str:
        """Append a numeric suffix to resolve collisions.

        If appending would exceed ``MAX_PARTS``, the last segment is replaced
        with the numeric suffix instead of appending a new one.
        """
        if name not in existing_names:
            return name
        parts = name.split("_")
        if len(parts) >= self.MAX_PARTS:
            base = "_".join(parts[: self.MAX_PARTS - 1])
        else:
            base = name
        counter = 2
        while f"{base}_{counter}" in existing_names:
            counter += 1
        return f"{base}_{counter}"


# ---------------------------------------------------------------------------
# LLM caller protocol + adapter
# ---------------------------------------------------------------------------


class LLMCaller(Protocol):
    """Protocol satisfied by any object that can make a simple JSON LLM call."""

    def call_for_json(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[Dict[str, str]]: ...


class LLMFixerAdapter:
    """Adapts an ``LLMFixer`` instance to the :class:`LLMCaller` protocol.

    Reuses the fixer's already-constructed OpenHands ``LLM`` client to
    run a single JSON-mode completion for constant naming. No per-provider
    dispatch: OpenHands/litellm routes based on the model's prefix.
    """

    def __init__(self, llm_fixer: Any) -> None:
        self._fixer = llm_fixer

    def call_for_json(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[Dict[str, str]]:
        try:
            response = self._fixer.client.completion(
                messages=[
                    Message(
                        role="system",
                        content=[TextContent(text=system_prompt)],
                    ),
                    Message(
                        role="user",
                        content=[TextContent(text=user_prompt)],
                    ),
                ],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.05,
            )
        except Exception:
            logger.error("LLM call for constant naming failed", exc_info=True)
            return None

        content = _extract_completion_content(response)
        if content is None:
            return None

        try:
            return _names_adapter.validate_json(content)
        except Exception:
            logger.error(
                "Constant-naming response did not match expected shape",
                exc_info=True,
            )
            return None


def _extract_completion_content(response: Any) -> Optional[str]:
    """Pull text content from an OpenHands ``LLMResponse`` or raw ``ModelResponse``.

    Mirrors the same-named helper in ``fix_validator`` -- kept private to
    this module so the two call sites can evolve independently without a
    shared utility drift.
    """
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)

    try:
        choices = response.choices
        if not choices:
            return None
        litellm_message = choices[0].message
        litellm_content = getattr(litellm_message, "content", None)
        return litellm_content if isinstance(litellm_content, str) else None
    except (AttributeError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class ConstantNamingService:
    """Generate meaningful SCREAMING_SNAKE_CASE names for duplicated literals.

    Usage::

        service = ConstantNamingService(llm_caller=LLMFixerAdapter(fixer))
        response = service.name_literals(request)
        # response.names == {"application/json": "APPLICATION_JSON", ...}
    """

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        max_words: int = MAX_WORDS_THRESHOLD,
    ) -> None:
        self._llm_caller = llm_caller
        self._max_words = max_words
        self._validator = NameValidator()
        self._prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(self._prompt_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=select_autoescape(["html", "xml"]),
        )

    # -- public API ---------------------------------------------------------

    def name_literals(self, request: NamingRequest) -> NamingResponse:
        """Main entry point. Accepts all literals for a file, returns mapping."""
        names: Dict[str, str] = {}
        used: Set[str] = set(request.existing_names)

        remaining = self._name_via_pipeline(request.literals, names, used)
        self._resolve_remaining(remaining, names, used, file_path=request.file_path)

        return NamingResponse(names=names)

    def _name_via_pipeline(
        self,
        literals: List[LiteralContext],
        names: Dict[str, str],
        used: Set[str],
    ) -> List[LiteralContext]:
        """Try the clean → slugify/YAKE pipeline; return literals that failed."""
        remaining: List[LiteralContext] = []
        seen: Set[str] = set()

        for lit_ctx in literals:
            if lit_ctx.literal in seen:
                continue
            seen.add(lit_ctx.literal)

            name = generate_name(lit_ctx.literal, max_words=self._max_words)
            if name and self._validator.is_structurally_valid(name):
                name = self._validator.make_unique(name, used)
                names[lit_ctx.literal] = name
                used.add(name)
            else:
                remaining.append(lit_ctx)

        return remaining

    def _resolve_remaining(
        self,
        remaining: List[LiteralContext],
        names: Dict[str, str],
        used: Set[str],
        file_path: str = "",
    ) -> None:
        """Resolve literals that the pipeline could not name."""
        if not remaining:
            return

        llm_names: Dict[str, str] = {}
        if self._llm_caller:
            llm_names = self._call_llm_for_names(remaining, used, file_path)

        for lit_ctx in remaining:
            self._assign_name(lit_ctx, llm_names, names, used)

    def _assign_name(
        self,
        lit_ctx: LiteralContext,
        llm_names: Dict[str, str],
        names: Dict[str, str],
        used: Set[str],
    ) -> None:
        """Assign a single literal its name from LLM results or fallback."""
        llm_name = llm_names.get(lit_ctx.literal)
        if llm_name and self._validator.is_structurally_valid(llm_name):
            llm_name = self._validator.make_unique(llm_name, used)
            names[lit_ctx.literal] = llm_name
            used.add(llm_name)
        else:
            fallback = self._generic_fallback(used)
            names[lit_ctx.literal] = fallback
            used.add(fallback)

    # -- LLM fallback -------------------------------------------------------

    def _call_llm_for_names(
        self,
        literals: List[LiteralContext],
        used_names: Set[str],
        file_path: str = "",
    ) -> Dict[str, str]:
        """Batch all remaining literals into a single LLM call."""
        if self._llm_caller is None:
            return {}

        try:
            system_tpl = self._jinja_env.get_template("python/constant_namer_system.j2")
            user_tpl = self._jinja_env.get_template("python/constant_namer_user.j2")
        except Exception as e:
            logger.error("Failed to load naming prompt templates: %s", e)
            return {}

        system_prompt = system_tpl.render()
        user_prompt = user_tpl.render(
            file_path=file_path,
            literals=literals,
            used_names=sorted(used_names),
        )

        result = self._llm_caller.call_for_json(system_prompt, user_prompt)
        if not isinstance(result, dict):
            return {}
        return result

    # -- ultimate fallback --------------------------------------------------

    @staticmethod
    def _generic_fallback(used_names: Set[str]) -> str:
        """Generate ``STRING_LITERAL_N`` as the last resort."""
        counter = 1
        name = f"STRING_LITERAL_{counter}"
        while name in used_names:
            counter += 1
            name = f"STRING_LITERAL_{counter}"
        return name
