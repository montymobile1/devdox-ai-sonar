"""Decoupled constant naming service for python:S1192 fixes.

Generates meaningful SCREAMING_SNAKE_CASE names for duplicated string literals
using a pipeline: clean → detect language → slugify/YAKE → LLM fallback.
"""

import json
import keyword
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

import yake
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langdetect import detect

from devdox_ai_sonar.models.constant_naming import (
    LiteralContext,
    NamingRequest,
    NamingResponse,
)

logger = logging.getLogger(__name__)

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
        return detect(literal)
    except Exception:
        return "en"


# ---------------------------------------------------------------------------
# Step 3 — Slugify (short English strings)
# ---------------------------------------------------------------------------


def _slugify(tokens: List[str], max_tokens: int = MAX_WORDS_THRESHOLD) -> Optional[str]:
    """Join the first *max_tokens* tokens as SCREAMING_SNAKE_CASE."""
    tokens = tokens[:max_tokens]
    if not tokens:
        return None
    name = "_".join(tokens).upper()
    name = re.sub(r"[^A-Z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else None


# ---------------------------------------------------------------------------
# Step 4 — YAKE keyword extraction (long or non-English strings)
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

    tokens = [kw[0] for kw in keywords]
    if not tokens:
        return None
    name = "_".join(tokens).upper()
    name = re.sub(r"[^A-Z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else None


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
    2. Detect language.
    3. If English and short (≤ *max_words* tokens) → direct slugify.
    4. Otherwise → YAKE keyword extraction.

    Returns ``None`` when no valid name can be produced.
    """
    tokens = _clean(literal)
    lang = _detect_language(literal)

    if lang == "en":
        if len(tokens) <= max_words:
            return _slugify(tokens, max_tokens=max_words)
        else:
            return _extract_with_yake(literal, lang="en", max_keywords=max_words)
    else:
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

    @staticmethod
    def make_unique(name: str, existing_names: Set[str]) -> str:
        """Append a numeric suffix to resolve collisions."""
        if name not in existing_names:
            return name
        counter = 2
        while f"{name}_{counter}" in existing_names:
            counter += 1
        return f"{name}_{counter}"


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

    Reuses the existing provider infrastructure (client, model, API key)
    but makes a simpler JSON-only call.
    """

    def __init__(self, llm_fixer: Any) -> None:
        self._fixer = llm_fixer

    def call_for_json(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[Dict[str, str]]:
        try:
            if self._fixer.provider == "openai":
                response = self._fixer.client.chat.completions.create(
                    model=self._fixer.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=2000,
                    temperature=0.05,
                )
                content = response.choices[0].message.content
                return json.loads(content)  # type: ignore[arg-type]

            elif self._fixer.provider == "gemini":
                from google.genai import types  # local to avoid hard dep

                response = self._fixer.client.models.generate_content(
                    model=self._fixer.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                    ),
                )
                return json.loads(response.text)  # type: ignore[arg-type]

            elif self._fixer.provider in ("togetherai", "openrouter"):
                response = self._fixer.client.chat.completions.create(
                    model=self._fixer.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.05,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)  # type: ignore[arg-type]

            else:
                logger.error(
                    f"Unknown provider for constant naming: {self._fixer.provider}"
                )
                return None

        except Exception as e:
            logger.error(f"LLM call for constant naming failed: {e}", exc_info=True)
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
        self._max_words = max(max_words, MAX_WORDS_THRESHOLD)
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
        remaining: List[LiteralContext] = []

        for lit_ctx in request.literals:
            name = generate_name(lit_ctx.literal, max_words=self._max_words)
            if name and self._validator.is_structurally_valid(name):
                name = self._validator.make_unique(name, used)
                names[lit_ctx.literal] = name
                used.add(name)
            else:
                remaining.append(lit_ctx)

        if remaining and self._llm_caller:
            llm_names = self._call_llm_for_names(remaining, used)
            for lit_ctx in remaining:
                llm_name = llm_names.get(lit_ctx.literal)
                if llm_name and self._validator.is_structurally_valid(llm_name):
                    llm_name = self._validator.make_unique(llm_name, used)
                    names[lit_ctx.literal] = llm_name
                    used.add(llm_name)
                else:
                    fallback = self._generic_fallback(used)
                    names[lit_ctx.literal] = fallback
                    used.add(fallback)
        elif remaining:
            for lit_ctx in remaining:
                fallback = self._generic_fallback(used)
                names[lit_ctx.literal] = fallback
                used.add(fallback)

        return NamingResponse(names=names)

    # -- LLM fallback -------------------------------------------------------

    def _call_llm_for_names(
        self,
        literals: List[LiteralContext],
        used_names: Set[str],
    ) -> Dict[str, str]:
        """Batch all remaining literals into a single LLM call."""
        assert self._llm_caller is not None

        try:
            system_tpl = self._jinja_env.get_template("python/constant_namer_system.j2")
            user_tpl = self._jinja_env.get_template("python/constant_namer_user.j2")
        except Exception as e:
            logger.error(f"Failed to load naming prompt templates: {e}")
            return {}

        system_prompt = system_tpl.render()
        user_prompt = user_tpl.render(
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
