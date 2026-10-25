"""Pydantic v2 schemas shared by every layer.

The flow of data is::

    Criterion ──┐
    SupplierInput ─► ExtractedDocument ─► (LLM JSON) ─► SupplierScorecard
                                                          │
                       RankingResult ◄── ranking_tool ◄───┘
                            │
                        RunResult  (persisted + exported)
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(StrEnum):
    """Lifecycle status of an RFP run."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class SupplierStatus(StrEnum):
    """Outcome of evaluating one supplier."""

    OK = "OK"
    NO_TEXT = "NO_TEXT"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    LLM_FAILED = "LLM_FAILED"


class Criterion(BaseModel):
    """One evaluation criterion as stored in ``evaluation_criteria``."""

    model_config = ConfigDict(frozen=True)

    criterion_id: int
    name: str
    description: str = ""
    weight: float
    max_score: float
    is_active: bool = True


class WarningRecord(BaseModel):
    """A structured warning produced by validation, extraction or ranking."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    supplier_name: str | None = None
    criterion_id: int | None = None


class SupplierInput(BaseModel):
    """User-entered metadata plus the uploaded file for one supplier."""

    supplier_name: str
    submission_date: date
    experience_rating: float
    filename: str
    file_sha256: str
    file_bytes: bytes = Field(repr=False, exclude=True)


class ExtractedDocument(BaseModel):
    """Clean text extracted from a PDF, with metadata."""

    text: str
    page_count: int
    char_count: int
    extraction_method: str
    truncated: bool = False
    original_char_count: int = 0


class CriterionScore(BaseModel):
    """Validated, normalised result for one criterion of one supplier."""

    criterion_id: int
    criterion_name: str
    score: float
    max_score: float
    weight: float
    justification: str
    evidence: str
    evidence_page: int | None = None
    evidence_unverified: bool = False


class SupplierScorecard(BaseModel):
    """Everything known about one supplier after validation (before ranking)."""

    supplier_name: str
    submission_date: date
    experience_rating: float
    filename: str
    file_sha256: str
    status: SupplierStatus
    criteria: list[CriterionScore]
    risks: list[str] = Field(default_factory=list)
    overall_summary: str = ""
    warnings: list[WarningRecord] = Field(default_factory=list)
    page_count: int = 0
    char_count: int = 0
    extraction_method: str = ""
    from_cache: bool = False


class CriterionMetric(BaseModel):
    """Peer metrics for one supplier on one criterion."""

    criterion_id: int
    criterion_name: str
    score: float
    max_score: float
    weight: float
    weighted_score: float
    benchmark: float
    gap: float
    relative_pct: float


class SupplierRanking(BaseModel):
    """Final ranked position of one supplier."""

    supplier_name: str
    submission_date: date
    experience_rating: float
    absolute_score: float
    ppi: float
    final_rank: int
    metrics: list[CriterionMetric]


class RankingResult(BaseModel):
    """Output of the ranking tool."""

    rankings: list[SupplierRanking]
    benchmarks: dict[int, float]
    tie_break_explanations: list[str]
    warnings: list[WarningRecord]


class RunResult(BaseModel):
    """A complete RFP run: the shape that is persisted and exported."""

    rfp_run_id: str
    created_at: str
    status: RunStatus
    provider: str
    model: str
    prompt_version: str
    criteria_snapshot: list[Criterion]
    scorecards: list[SupplierScorecard]
    ranking: RankingResult
    warnings: list[WarningRecord] = Field(default_factory=list)

    def all_warnings(self) -> list[WarningRecord]:
        """Run-level, ranking and per-supplier warnings in a stable order."""
        out = list(self.warnings) + list(self.ranking.warnings)
        for card in self.scorecards:
            out.extend(card.warnings)
        return out
