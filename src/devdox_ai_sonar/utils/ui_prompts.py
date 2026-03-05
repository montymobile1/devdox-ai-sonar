"""Cross-platform UI prompt utilities.

Wraps questionary to provide a consistent interface for interactive
CLI prompts. All interactive menu/selection logic should go through
this module — if the underlying library ever needs to change, only
this file needs updating.
"""

from typing import List, Optional

import questionary
from questionary import Style

# Shared style for menu prompts
_MENU_STYLE = Style([
    ("pointer", "fg:green bold"),
    ("highlighted", "fg:green bold"),
])


def select_from_list(
    choices: List[str],
    message: str,
    use_search: bool = True,
) -> Optional[str]:
    """Present an interactive menu and return the selected item.

    Args:
        choices: List of string options to display.
        message: Prompt message shown above the menu.
        use_search: Enable type-to-filter search (default True).

    Returns:
        The selected string, or None if cancelled (Ctrl+C) or empty list.
    """
    if not choices:
        return None
    return questionary.select(
        message,
        choices=choices,
        use_search_filter=use_search,
        use_arrow_keys=True,
        use_jk_keys=not use_search,
        pointer="➤ ",
        instruction="(Type to filter)" if use_search else None,
        style=_MENU_STYLE,
    ).ask()
