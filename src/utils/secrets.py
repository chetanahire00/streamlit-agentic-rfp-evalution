"""Secret masking and redaction.

API keys are registered here as soon as they are entered so that the logging
filter and error formatting can scrub them from any text.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_SECRETS: set[str] = set()
_MIN_SECRET_LEN = 6


def mask_secret(secret: str | None) -> str:
    """Return a display-safe form such as ``sk-…abcd``.

    Short or empty values are fully masked.
    """
    if not secret:
        return "(none)"
    if len(secret) < 12:
        return "…" + "*" * 4
    return f"{secret[:3]}…{secret[-4:]}"


def register_secret(secret: str | None) -> None:
    """Remember ``secret`` so :func:`redact` removes it from text."""
    if secret and len(secret) >= _MIN_SECRET_LEN:
        with _LOCK:
            _SECRETS.add(secret)


def clear_secrets() -> None:
    """Forget every registered secret (used by tests)."""
    with _LOCK:
        _SECRETS.clear()


def redact(text: str, extra: str | None = None) -> str:
    """Replace every registered secret (and ``extra``) in ``text`` with its mask."""
    with _LOCK:
        secrets = set(_SECRETS)
    if extra and len(extra) >= _MIN_SECRET_LEN:
        secrets.add(extra)
    for secret in sorted(secrets, key=len, reverse=True):
        if secret in text:
            text = text.replace(secret, mask_secret(secret))
    return text
