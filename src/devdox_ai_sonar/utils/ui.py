from typing import Optional, List, Union, Set
from rich.console import Console
from rich.prompt import Prompt
import questionary
from questionary.prompts.common import Choice
from devdox_ai_sonar.utils.result import PromptConfig, ConfirmConfig

from devdox_ai_sonar.utils.exceptions import SwitchCommandException, ReturnToMenuException
from devdox_ai_sonar.utils import constant

console = Console()

def smart_prompt(
    message: str,
    default: Optional[Union[str, List[str]]] = None,
    choices: Optional[List[str]] = None,
    allow_switch: bool = True,
    multiple:bool=False
) -> Union[str, List[str]]:
    """
    Enhanced prompt that allows switching commands by typing '/'.

    Args:
        message: Prompt message to display
        default: Default value
        choices: List of valid choices (if applicable)
        allow_switch: Whether to allow command switching

    Returns:
        User's input

    Raises:
        SwitchCommandException: If user types '/' to switch commands
    """
    config = PromptConfig(message, default, choices, allow_switch, multiple)
    try:
        result = _prompt_with_questionary(config)
    except ImportError:
        result = _prompt_with_rich_fallback(config)

    _check_for_switch_command(result, allow_switch)
    return result

def _check_for_switch_command(result: Union[str, List[str], None], allow_switch: bool) -> None:
    """
    Check if user requested command switch.

    Args:
        result: The prompt result to check
        allow_switch: Whether switching is allowed

    Raises:
        SwitchCommandException: If switch command detected
    """
    if not allow_switch:
        return

    # Handle None result (user cancelled)
    if result is None:
        return

    # For text/select prompts (string result)
    if isinstance(result, str) and result.strip() == constant.SWITCH_COMMAND_TRIGGER:
        raise SwitchCommandException()

    # For checkbox prompts (list result) - check if any item is the trigger
    if isinstance(result, list) and constant.SWITCH_COMMAND_TRIGGER in result:
        raise SwitchCommandException()


def _prompt_with_rich_fallback(config: PromptConfig) -> Union[str, List[str]]:
    """
    Fallback to rich.prompt when questionary is unavailable.

    Note: Rich doesn't support multiple selection, so we fall back to single select.
    """

    if config.allow_switch:
        console.print(f"[dim](Type '{constant.SWITCH_COMMAND_TRIGGER}' to switch commands)[/dim]")

    if config.multiple:
        console.print("[yellow]⚠ Multiple selection not available in fallback mode[/yellow]")

    if config.choices:
        return Prompt.ask(config.message, choices=config.choices, default=config.default)

    return Prompt.ask(config.message, default=config.default)


def _prompt_with_questionary(config: PromptConfig) -> Union[str, List[str]]:
        """
        Execute prompt using questionary library.

        Raises:
            ImportError: If questionary is not available
        """
        display_message = config.get_display_message()

        if not config.choices:

            return _questionary_text_prompt(display_message, config.default)

        if config.multiple:
            return _questionary_checkbox_prompt(display_message, config.choices, config.default)

        return _questionary_select_prompt(display_message, config.choices, config.default)


def _questionary_text_prompt(message: str, default: Optional[str]) -> str:
    """Simple text input prompt."""
    return questionary.text(message, default=default or "").ask()

def _questionary_select_prompt(
        message: str,
        choices: List[str],
        default: Optional[str]
) -> str:
    """Single selection dropdown prompt."""
    return questionary.select(message, choices=choices, default=default).ask()

def _questionary_checkbox_prompt(
        message: str,
        choices: List[str],
        default: Optional[List[str]]
) -> List[str]:
    """
    Multiple selection checkbox prompt.

    Args:
        message: Display message
        choices: Available options
        default: Comma-separated list of pre-selected choices
    """
    checked_items = _parse_default_choices(default)

    choice_objects = [
        Choice(value=choice, title=choice, checked=choice in checked_items)
        for choice in choices
    ]

    return questionary.checkbox(message, choices=choice_objects).ask()






def _parse_default_choices(default: list[str]) -> Set[str]:
    """
    Parse default value into a set of selected choices.

    Args:
        default: Comma-separated string of default choices (e.g., "opt1,opt2")

    Returns:
        Set of choice values to pre-select
    """
    if not default:
        return set()

    # Split by comma and strip whitespace
    return {choice.strip() for choice in default if choice.strip()}


def _confirm_with_questionary(config: ConfirmConfig) -> str:
    """
    Execute confirmation using questionary library.

    Returns:
        Selected choice: "Yes", "No", or "/ Switch Command"

    Raises:
        ImportError: If questionary is not available
    """
    import questionary

    return questionary.select(
        config.get_display_message(),
        choices=config.get_questionary_choices(),
        default=config.get_default_choice()
    ).ask()


def _confirm_with_console_fallback(config: ConfirmConfig) -> str:
    """
    Fallback confirmation using console input.

    Returns:
        User's input (may be empty, "y", "yes", "n", "no", or "/")
    """
    if config.allow_switch:
        console.print(
            f"[dim](Type '{constant.SWITCH_COMMAND_TRIGGER}' to switch commands)[/dim]"
        )

    prompt = _build_console_prompt(config.message, config.default)
    return console.input(prompt).strip().lower()


def _build_console_prompt(message: str, default: bool) -> str:
    """
    Build prompt string with Y/N indicators based on default.

    Examples:
        default=True  -> "Continue? [Y/n]: "
        default=False -> "Continue? [y/N]: "
    """
    yes_indicator = "Y" if default else "y"
    no_indicator = "n" if default else "N"
    return f"{message} [{yes_indicator}/{no_indicator}]: "


def _parse_confirmation_result(result: Optional[str], default: bool) -> bool:
    """
    Parse user's confirmation response into boolean.

    Args:
        result: User's input or selected choice
        default: Default value for empty input

    Returns:
        True for affirmative responses, False otherwise
    """
    if not result or result.strip() == "":
        return default

    result_lower = result.lower()
    return result_lower in ("y", "yes")



def smart_confirm(
        message: str,
        default: bool = True,
        allow_switch: bool = True
) -> bool:
    """
    Enhanced confirmation that allows switching commands.

    Args:
        message: Confirmation message
        default: Default value (True for Yes, False for No)
        allow_switch: Whether to allow command switching

    Returns:
        User's confirmation (True for Yes, False for No)

    Raises:
        SwitchCommandException: If user types '/' to switch commands
    """
    config = ConfirmConfig(message, default, allow_switch)

    try:
        result = _confirm_with_questionary(config)
    except ImportError:
        result = _confirm_with_console_fallback(config)

    _check_for_switch_command(result, allow_switch)
    return _parse_confirmation_result(result, default)

