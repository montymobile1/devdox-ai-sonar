from typing import TypeVar, Generic, Union, Optional, List
from dataclasses import dataclass
from devdox_ai_sonar.utils import constant

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


@dataclass
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass
class Err(Generic[E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> None:
        raise self.error


Result = Union[Ok[T], Err[E]]


@dataclass
class PromptConfig:
    """Configuration for smart prompt behavior."""

    message: str
    default: Optional[Union[str, List[str]]] = None
    choices: Optional[List[str]] = None
    allow_switch: bool = True
    multiple: bool = False

    def get_display_message(self) -> str:
        """Build the full message with optional switch hint."""
        if self.allow_switch:
            return f"{self.message}\n[dim](Type '/' to switch commands)[/dim]"
        return self.message


@dataclass
class ConfirmConfig:
    """Configuration for confirmation prompt."""

    message: str
    default: bool
    allow_switch: bool

    def get_display_message(self) -> str:
        """Build message with optional switch hint."""
        if self.allow_switch:
            return f"{self.message}\n[dim](Type '/' to switch commands)[/dim]"
        return self.message

    def get_default_choice(self) -> str:
        """Get the default choice label."""
        return "Yes" if self.default else "No"

    def get_questionary_choices(self) -> list[str]:
        """Get choice list for questionary including optional switch."""
        choices = ["Yes", "No"]
        if self.allow_switch:
            choices.append(f"{constant.SWITCH_COMMAND_TRIGGER} Switch Command")
        return choices
