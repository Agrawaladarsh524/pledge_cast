"""Logging setup - PLAN.md sec.10: "Stdlib logging -> console + rotating file".

One setup function, called once per process entry point (scripts, API, dashboard).
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

_CONFIGURED = False


def setup_logging(settings: Settings | None = None, *, force: bool = False) -> logging.Logger:
    """Configure root logging from config. Idempotent unless ``force``.

    Returns the ``pledgecast`` root logger.
    """
    global _CONFIGURED

    if settings is None:
        from config import get_settings

        settings = get_settings()

    root = logging.getLogger()
    if _CONFIGURED and not force:
        return logging.getLogger("pledgecast")

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(settings.log_level)
    formatter = logging.Formatter(
        fmt=settings.logging.format,
        datefmt=settings.logging.date_format,
    )

    if settings.logging.console_enabled:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)
        console.setLevel(settings.log_level)
        root.addHandler(console)

    if settings.logging.file_enabled:
        settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.paths.logs_dir / settings.logging.file_name,
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(settings.log_level)
        root.addHandler(file_handler)

    # Third-party chatter that would otherwise drown a 6,000-file download.
    for noisy in ("urllib3", "matplotlib", "shap", "numba", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger("pledgecast")


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name if name.startswith("pledgecast") else f"pledgecast.{name}")


__all__ = ["setup_logging", "get_logger"]
