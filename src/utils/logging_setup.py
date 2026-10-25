"""Logging configuration with secret redaction."""

from __future__ import annotations

import logging

from src.utils.secrets import redact

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RedactingFilter(logging.Filter):
    """Scrub registered secrets from log messages and exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            exc.args = tuple(redact(str(a)) if isinstance(a, str) else a for a in exc.args)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a redacting stream handler to the ``src`` logger (idempotent)."""
    logger = logging.getLogger("src")
    logger.setLevel(level)
    if any(getattr(h, "_rfp_handler", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(RedactingFilter())
    handler._rfp_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger whose records pass through the redacting filter."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger
