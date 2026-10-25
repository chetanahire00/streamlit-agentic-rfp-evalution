"""Typed application errors with stable codes and user-safe messages.

Every error raised to the UI carries a ``code`` from :class:`ErrorCode` and a
``user_message`` that is safe to display (it never contains secrets, file paths
or tracebacks). Technical detail goes to the log only.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Catalogue of error codes (documented in docs/TECHNICAL.md)."""

    # Input validation
    CRITERIA_NONE_ACTIVE = "E101"
    CRITERIA_WEIGHT_TOTAL = "E102"
    CRITERIA_INVALID_FIELD = "E103"
    CRITERIA_DUPLICATE_NAME = "E104"
    FILE_TOO_FEW = "E201"
    FILE_NOT_PDF = "E202"
    FILE_TOO_LARGE = "E203"
    FILE_ENCRYPTED = "E204"
    FILE_CORRUPT = "E205"
    FILE_DUPLICATE = "E206"
    FILE_EMPTY = "E207"
    META_NAME_REQUIRED = "E301"
    META_NAME_DUPLICATE = "E302"
    META_DATE_REQUIRED = "E303"
    META_DATE_FUTURE = "E304"
    META_RATING_INVALID = "E305"
    # LLM
    LLM_AUTH = "E401"
    LLM_MODEL_NOT_FOUND = "E402"
    LLM_RATE_LIMIT = "E403"
    LLM_TIMEOUT = "E404"
    LLM_NETWORK = "E405"
    LLM_SERVER = "E406"
    LLM_BAD_REQUEST = "E407"
    LLM_UNKNOWN = "E408"
    LLM_CONFIG = "E409"
    # Runtime
    RUN_ABORTED = "E501"
    PERSISTENCE = "E502"


class AppError(Exception):
    """Base class for all expected application errors.

    Args:
        code: Stable error code.
        user_message: Friendly message that is safe to show in the UI.
    """

    def __init__(self, code: ErrorCode, user_message: str) -> None:
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.user_message}"


class ValidationError(AppError):
    """User input (criteria, files, metadata) failed validation."""


class DocumentError(AppError):
    """A PDF could not be accepted or read."""


class LLMError(AppError):
    """An LLM call failed.

    Args:
        code: One of the ``LLM_*`` codes.
        user_message: Friendly message.
        retryable: Whether retrying the same request may succeed.
    """

    def __init__(self, code: ErrorCode, user_message: str, *, retryable: bool = False) -> None:
        super().__init__(code, user_message)
        self.retryable = retryable


class RunAbortedError(AppError):
    """The whole batch had to stop (for example, an invalid API key)."""


class PersistenceError(AppError):
    """Writing to SQLite failed; the transaction was rolled back."""
