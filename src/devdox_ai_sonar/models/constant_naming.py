"""Models for the constant naming service."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from pydantic import BaseModel, Field


@dataclass
class LiteralContext:
    """Context about a single duplicated literal across a file."""

    literal: str
    occurrences: List[Tuple[int, int, int]] = field(default_factory=list)


class NamingRequest(BaseModel):
    """Input to the constant naming service — one per file."""

    file_path: str = Field(..., description="Absolute path to the source file")
    literals: List[LiteralContext] = Field(
        ..., description="All duplicated literals with their occurrences"
    )
    existing_names: Set[str] = Field(
        default_factory=set,
        description="Names already used in the file (module-level constants, etc.)",
    )


class NamingResponse(BaseModel):
    """Output from the constant naming service."""

    names: Dict[str, str] = Field(
        ...,
        description="Mapping of literal value to SCREAMING_SNAKE_CASE constant name",
    )
