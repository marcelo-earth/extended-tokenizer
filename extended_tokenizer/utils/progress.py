"""Progress bars and logging utilities."""

from __future__ import annotations

import logging
import sys
from typing import Any

from tqdm import tqdm


def get_progress_bar(
    iterable: Any | None = None,
    total: int | None = None,
    desc: str | None = None,
    disable: bool = False,
    unit: str = "it",
    **kwargs: Any,
) -> tqdm:
    """Create a progress bar with consistent styling.

    Args:
        iterable: Optional iterable to wrap
        total: Total number of iterations
        desc: Description prefix
        disable: Whether to disable the progress bar
        unit: Unit name for iterations
        **kwargs: Additional arguments to pass to tqdm

    Returns:
        tqdm progress bar instance
    """
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        disable=disable,
        unit=unit,
        ncols=100,
        file=sys.stderr,
        **kwargs,
    )


def setup_logging(
    level: int = logging.INFO,
    format_str: str | None = None,
) -> logging.Logger:
    """Set up logging for the package.

    Args:
        level: Logging level
        format_str: Optional custom format string

    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    return logging.getLogger("extended_tokenizer")
