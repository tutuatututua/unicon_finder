"""Central logging configuration for the project.

Import and call ``get_logger(__name__)`` inside modules instead of using ``print``.

Environment variable LOG_LEVEL can override the default (INFO).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_CONFIGURED = False


def _configure_root(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger, configuring root logger on first use.

    Parameters
    ----------
    name : str, optional
        Logger name, usually ``__name__`` from the caller.
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    _configure_root(level)
    return logging.getLogger(name if name else "unicon2")


__all__ = ["get_logger"]
