"""
ai_waf_v2.utils.logging
-------------------
Centralised logging setup using Rich for pretty console output and
a plain rotating file handler for persistent logs.

Usage
-----
    from ai_waf_v2.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("Training started", extra={"step": 0, "lr": 1e-4})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Return a named logger with a Rich console handler and an optional
    file handler.

    Calling get_logger with the same name multiple times is safe — handlers
    are only added once.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.
    level : int
        Logging level. Default: INFO.
    log_file : str | Path | None
        If provided, also write logs to this file (plain text, rotating,
        max 10 MB × 3 backups).
    """
    logger = logging.getLogger(name)

    # Don't add handlers if they're already attached (e.g. called twice)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler ──────────────────────────────
    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(
            level=level,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        # Fallback if Rich is not installed
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    logger.addHandler(console_handler)

    # ── File handler (optional) ──────────────────────
    if log_file is not None:
        from logging.handlers import RotatingFileHandler

        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def configure_root(level: int = logging.WARNING) -> None:
    """
    Silence noisy third-party loggers (transformers, datasets, etc.)
    while keeping ai_waf_v2. loggers at INFO.

    Call once at the entry point of every stage script.
    """
    logging.getLogger().setLevel(level)
    for noisy in ("transformers", "datasets", "tokenizers",
                  "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.ERROR)