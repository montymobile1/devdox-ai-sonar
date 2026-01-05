from typing import Optional, List, Union, Set
from rich.console import Console
from rich.prompt import Prompt
import questionary
from questionary.prompts.common import Choice
from devdox_ai_sonar.utils.result import PromptConfig, ConfirmConfig

from devdox_ai_sonar.utils.exceptions import (
    SwitchCommandException,
)
from devdox_ai_sonar.utils import constant

console = Console()


def smart_prompt(
    message: str,
    default: Optional[Union[str, List[str]]] = None,
    choices: Optional[List[Union[str, Choice]]] = None,
    allow_switch: bool = True,
    multiple: bool = False,
) -> Union[str, List[str]]:
    """Enhanced prompt that allows switching commands by typing '/'.

    Args:
        message: Prompt message to display
        default: Default value (string or list of strings)
        choices: List of valid choices (strings or Choice objects)
        allow_switch: Whether to allow command switching with '/'
        multiple: Whether to allow multiple selections

    Returns:
        User's input (string or list of strings)

    Raises:
        SwitchCommandException: If user types '/' to switch commands
    """
    choice_strings = _normalize_choices(choices)

    config = PromptConfig(message, default, choice_strings, allow_switch, multiple)

    try:
        result = _prompt_with_questionary(config)
    except ImportError:
        result = _prompt_with_rich_fallback(config)

    _check_for_switch_command(result, allow_switch)

    return result


def _normalize_choices(
    choices: Optional[List[Union[str, Choice]]],
) -> Optional[List[str]]:
    """Convert a list of Choice objects or strings into a list of strings."""
    if not choices:
        return None

    normalized: List[str] = []
    for choice in choices:
        if isinstance(choice, Choice):
            value = getattr(choice, "value", choice)
            normalized.append(str(value))
        else:
            normalized.append(str(choice))
    return normalized


def _check_for_switch_command(
    result: Union[str, List[str], None], allow_switch: bool
) -> None:
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
        console.print(
            f"[dim](Type '{constant.SWITCH_COMMAND_TRIGGER}' to switch commands)[/dim]"
        )

    if config.multiple:
        console.print(
            "[yellow]⚠ Multiple selection not available in fallback mode[/yellow]"
        )

    default_value = config.default if config.default is not None else ""

    if config.choices:
        return Prompt.ask(config.message, choices=config.choices, default=default_value)

    return Prompt.ask(config.message, default=default_value)


def _prompt_with_questionary(config: PromptConfig) -> Union[str, List[str]]:
    """
    Execute prompt using questionary library.

    Raises:
        ImportError: If questionary is not available
    """
    display_message = config.get_display_message()

    if not config.choices:
        text_default: str = config.default if isinstance(config.default, str) else ""

        return _questionary_text_prompt(display_message, text_default)

    if config.multiple:
        checkbox_default: Optional[List[str]] = (
            config.default if isinstance(config.default, list) else None
        )

        return _questionary_checkbox_prompt(
            display_message, config.choices, checkbox_default
        )
    select_default: Optional[str] = (
        config.default if isinstance(config.default, str) else None
    )

    return _questionary_select_prompt(display_message, config.choices, select_default)


def _questionary_select_prompt(
    message: str,
    choices: List[str],
    default_value: Optional[str] = None,
) -> str:
    """
    Prompt user with single selection.

    Args:
        message: The prompt message
        choices: List of available choices
        default_value: Default selected value

    Returns:
        Selected choice as string
    """
    if default_value is not None:
        default = default_value

    elif choices:
        default = choices[0]

    else:
        default = ""

    result = questionary.select(message, choices=choices, default=default).ask()
    # Ensure we return a string, not Any
    return str(result) if result is not None else ""


def _questionary_text_prompt(
    message: str,
    default_value: Optional[str] = None,
) -> str:
    """
    Prompt user for text input.

    Args:
        message: The prompt message
        default_value: Default text value

    Returns:
        User input as string
    """
    result = questionary.text(message, default=default_value or "").ask()

    # Ensure we return a string, not Any
    return str(result) if result is not None else ""


def _questionary_checkbox_prompt(
    message: str, choices: List[str], default_value: Optional[List[str]] = None
) -> List[str]:
    """
    Multiple selection checkbox prompt.

    Args:
        message: Display message
        choices: Available options
       default_value: List of pre-selected choices



    Returns:

        List of selected choices
    """

    # Create Choice objects with checked status based on defaults

    choice_objects = []

    default_set = set(default_value) if default_value else set()

    for choice_str in choices:
        choice_obj = Choice(
            title=choice_str, value=choice_str, checked=(choice_str in default_set)
        )

        choice_objects.append(choice_obj)

    result = questionary.checkbox(message, choices=choice_objects).ask()

    # Ensure we always return a list, never None

    return list(result) if result is not None else []


def _parse_default_choices(default: Optional[List[str]]) -> Set[str]:
    """
    Parse default value into a set of selected choices.

    Args:
        default: Comma-separated string of default choices (e.g., "opt1,opt2")

    Returns:
          default: List of default choices
    """
    if not default:
        return set()

    return {choice.strip() for choice in default if choice and choice.strip()}


def _confirm_with_questionary(config: ConfirmConfig) -> str:
    """
    Execute confirmation using questionary library.

    Returns:
        Selected choice: "yes", "no"
    Raises:
        ImportError: If questionary is not available
    """
    import questionary

    result = questionary.confirm(config.message, default=config.default).ask()
    if result:
        return "yes"
    return "no"


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

    default_value = config.default if config.default is not None else ""
    prompt = _build_console_prompt(config.message, default_value)
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
    message: str, default: bool = True, allow_switch: bool = True
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
