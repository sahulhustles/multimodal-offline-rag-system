"""Structured logging configuration for the Multimodal RAG System."""

import logging
import sys

from backend.config import settings

_CONFIGURED = False


def setup_logging() -> None:
    """Configure structured logging for the application.

    Uses a human-readable format with timestamp, level, logger name, and message.
    Quietens noisy third-party libraries.
    Safe to call multiple times — only configures on the first call.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Quieten noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance."""
    return logging.getLogger(name)
