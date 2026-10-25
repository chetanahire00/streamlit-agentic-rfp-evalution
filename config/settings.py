"""Application-wide settings and limits.

Every tunable limit lives here so the UI, tools and tests share one source of
truth. Values can be overridden with environment variables where noted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPT_VERSION = "v1.0"
"""Version of the evaluation prompt templates. Part of the LLM cache key."""


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of runtime limits.

    Attributes:
        db_path: SQLite database file (env ``RFP_DB_PATH``).
        providers_file: YAML file listing providers and models.
        max_file_mb: Largest accepted PDF upload, in megabytes.
        min_text_chars: Below this many extracted characters a PDF is ``NO_TEXT``.
        max_pdf_chars: Default number of document characters sent to the LLM.
        min_suppliers: Peer comparison needs at least this many suppliers.
        experience_min: Lowest allowed historical experience rating.
        experience_max: Highest allowed historical experience rating.
        weight_total: Required sum of active criterion weights.
        weight_tolerance: Allowed deviation from ``weight_total``.
        evidence_match_threshold: rapidfuzz score (0-100) that counts as a grounded quote.
        max_evidence_words: Evidence quotes are truncated to this many words.
        max_justification_chars: Longer justifications are truncated.
        max_summary_chars: Longer overall summaries are truncated.
        max_risks: At most this many risks are kept.
        max_risk_chars: Each risk is truncated to this many characters.
        decimals: Rounding applied to every computed value before comparison.
        default_timeout_s: Default LLM request timeout.
        default_max_retries: Default retry count for transient LLM errors.
        default_max_tokens: Default LLM output token cap.
        backoff_base_s: First retry delay; doubles on each attempt.
    """

    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("RFP_DB_PATH", PROJECT_ROOT / "data" / "rfp_evaluator.db")
        )
    )
    providers_file: Path = PROJECT_ROOT / "config" / "providers.yaml"
    max_file_mb: float = 10.0
    min_text_chars: int = 200
    max_pdf_chars: int = 60_000
    min_suppliers: int = 2
    experience_min: float = 0.0
    experience_max: float = 5.0
    weight_total: float = 100.0
    weight_tolerance: float = 0.01
    evidence_match_threshold: float = 85.0
    max_evidence_words: int = 40
    max_justification_chars: int = 1_200
    max_summary_chars: int = 2_000
    max_risks: int = 10
    max_risk_chars: int = 300
    decimals: int = 4
    default_timeout_s: float = 60.0
    default_max_retries: int = 2
    default_max_tokens: int = 8_000
    backoff_base_s: float = 1.0

    @property
    def max_file_bytes(self) -> int:
        """Upload size limit in bytes."""
        return int(self.max_file_mb * 1024 * 1024)


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` (reads environment overrides)."""
    return Settings()
