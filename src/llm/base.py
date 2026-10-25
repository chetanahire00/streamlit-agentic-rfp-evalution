"""Provider-neutral LLM interface, retry policy and error mapping.

Each adapter implements :meth:`LLMProvider._complete` with its SDK's native JSON
or structured-output mode. This base class adds:

* exponential-backoff retries for transient errors (rate limit, timeout,
  network, 5xx). SDK-level retries are disabled so the policy lives in one place;
* mapping of every SDK exception to an :class:`LLMError` with a friendly
  message that never contains the API key.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.utils.errors import ErrorCode, LLMError
from src.utils.logging_setup import get_logger
from src.utils.secrets import mask_secret, redact, register_secret

log = get_logger(__name__)

PING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


@dataclass(frozen=True)
class LLMConfig:
    """Everything needed to call one provider/model.

    ``api_key`` is excluded from ``repr`` so it never appears in logs or tracebacks.
    """

    provider: str
    model: str
    api_key: str = field(default="", repr=False)
    temperature: float = 0.0
    max_tokens: int = 8_000
    timeout_s: float = 60.0
    max_retries: int = 2
    supports_temperature: bool = True
    provider_label: str = ""

    @property
    def label(self) -> str:
        """Human-readable provider name."""
        return self.provider_label or self.provider

    def masked_key(self) -> str:
        """Display-safe key, e.g. ``sk-…abcd``."""
        return mask_secret(self.api_key)


_RETRYABLE = {ErrorCode.LLM_RATE_LIMIT, ErrorCode.LLM_TIMEOUT, ErrorCode.LLM_NETWORK, ErrorCode.LLM_SERVER}


def friendly_message(code: ErrorCode, config: LLMConfig) -> str:
    """User-facing text for an LLM error code (never includes the key)."""
    p, m = config.label, config.model
    return {
        ErrorCode.LLM_AUTH: f"{p} rejected the API key (invalid or unauthorised). "
                            "Check the key in the sidebar and try again.",
        ErrorCode.LLM_MODEL_NOT_FOUND: f"Model '{m}' was not found on {p}, or your key cannot use it. "
                                       "Choose another model.",
        ErrorCode.LLM_RATE_LIMIT: f"{p} rate limit reached. Wait a minute and try again.",
        ErrorCode.LLM_TIMEOUT: f"The request to {p} timed out after {config.timeout_s:g}s. Try again or "
                               "raise the timeout under Advanced.",
        ErrorCode.LLM_NETWORK: f"Could not reach {p}. Check your internet connection and try again.",
        ErrorCode.LLM_SERVER: f"{p} returned a server error. Try again shortly.",
        ErrorCode.LLM_BAD_REQUEST: f"{p} rejected the request for model '{m}'. Try another model or "
                                   "lower Max output tokens.",
        ErrorCode.LLM_CONFIG: f"{p} is not configured correctly.",
    }.get(code, f"Unexpected error while calling {p}.")


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify_error(exc: BaseException, config: LLMConfig) -> LLMError:
    """Map any SDK exception to an :class:`LLMError` (works across all four SDKs).

    Uses the HTTP status where available, then exception class names (all SDKs
    name their timeout/connection errors accordingly), then message keywords.
    """
    if isinstance(exc, LLMError):
        return exc
    names = " ".join(c.__name__.lower() for c in type(exc).__mro__)
    text = redact(str(exc), config.api_key).lower()
    status = _status_of(exc)

    if "timeout" in names or isinstance(exc, TimeoutError):
        code = ErrorCode.LLM_TIMEOUT
    elif status in (401, 403) or "authentication" in names or "permissiondenied" in names:
        code = ErrorCode.LLM_AUTH
    elif status == 404 or "notfound" in names:
        code = ErrorCode.LLM_MODEL_NOT_FOUND
    elif status == 429 or "ratelimit" in names:
        code = ErrorCode.LLM_RATE_LIMIT
    elif status is not None and status >= 500:
        code = ErrorCode.LLM_SERVER
    elif "connection" in names or isinstance(exc, ConnectionError):
        code = ErrorCode.LLM_NETWORK
    elif status == 400 and ("api key" in text or "api_key" in text):
        code = ErrorCode.LLM_AUTH  # Gemini reports a bad key as 400 API_KEY_INVALID
    elif status == 400 and "model" in text and ("not found" in text or "does not exist" in text):
        code = ErrorCode.LLM_MODEL_NOT_FOUND
    elif status is not None and 400 <= status < 500:
        code = ErrorCode.LLM_BAD_REQUEST
    else:
        code = ErrorCode.LLM_UNKNOWN
    return LLMError(code, friendly_message(code, config), retryable=code in _RETRYABLE)


class LLMProvider(ABC):
    """Abstract provider: ``complete_json(system, user, schema) -> str``.

    Args:
        config: Provider/model/key and limits.
        client: Pre-built SDK client (tests inject fakes; ``None`` builds a real one).
        sleep: Backoff sleep function (tests inject a no-op).
    """

    provider_id: str = "base"

    def __init__(self, config: LLMConfig, client: Any | None = None,
                 sleep: Callable[[float], None] = time.sleep, backoff_base_s: float = 1.0) -> None:
        if self.requires_key and not config.api_key:
            raise LLMError(ErrorCode.LLM_AUTH, f"No API key provided for {config.label}.")
        register_secret(config.api_key)
        self.config = config
        self._sleep = sleep
        self._backoff = backoff_base_s
        self._client = client if client is not None else self._build_client()

    requires_key: bool = True

    @abstractmethod
    def _build_client(self) -> Any:
        """Create the SDK client with the configured key and timeout."""

    @abstractmethod
    def _complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        """One raw request; return the model's text (expected to be JSON)."""

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        """Call the model and return its raw JSON text, retrying transient errors.

        Raises:
            LLMError: mapped, user-safe error after retries are exhausted.
        """
        attempts = max(0, self.config.max_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._complete(system, user, schema)
            except Exception as exc:  # noqa: BLE001 - every SDK error is mapped below
                err = classify_error(exc, self.config)
                log.warning("%s/%s call failed (attempt %d/%d, %s): %s", self.config.provider,
                            self.config.model, attempt, attempts, err.code.value,
                            redact(f"{type(exc).__name__}: {exc}", self.config.api_key))
                if not err.retryable or attempt == attempts:
                    raise err from None
                self._sleep(self._backoff * 2 ** (attempt - 1))
        raise LLMError(ErrorCode.LLM_UNKNOWN, friendly_message(ErrorCode.LLM_UNKNOWN, self.config))

    def test_connection(self) -> None:
        """Make a minimal JSON call; raise :class:`LLMError` on failure."""
        text = self.complete_json(
            "You are a connectivity check. Reply with JSON only.",
            'Return exactly this JSON object: {"ok": true}',
            PING_SCHEMA,
        )
        try:
            json.loads(_strip_fence(text))
        except (ValueError, TypeError) as exc:
            raise LLMError(ErrorCode.LLM_BAD_REQUEST,
                           f"{self.config.label} answered, but not with JSON. Try another model.") from exc

    def _temperature_kwargs(self) -> dict[str, float]:
        """``{"temperature": t}`` only for models that accept it."""
        return {"temperature": self.config.temperature} if self.config.supports_temperature else {}


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[1] if "\n" in t else t
    return t
