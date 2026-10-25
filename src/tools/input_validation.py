"""Input validation for criteria, uploaded files and supplier metadata.

Validators return lists of :class:`Issue` instead of raising, so the UI can
show every problem inline at once. ``Issue.blocking`` marks problems that must
be fixed before evaluation can start.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from config.settings import Settings
from src.db.repository import CriterionDraft
from src.models.schemas import Criterion
from src.utils.errors import ErrorCode

PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class Issue:
    """One validation problem.

    Attributes:
        code: Stable error code.
        message: Friendly, specific message (cause and how to fix).
        field: Which field or item it refers to (e.g. ``"row 2"``, a filename).
        blocking: Whether it prevents evaluation/saving.
    """

    code: ErrorCode
    message: str
    field: str = ""
    blocking: bool = True


@dataclass(frozen=True)
class FileCandidate:
    """An uploaded file before metadata is attached."""

    filename: str
    data: bytes


@dataclass(frozen=True)
class SupplierMeta:
    """Metadata typed by the user for one uploaded file."""

    file_key: str
    supplier_name: str | None
    submission_date: date | None
    experience_rating: float | None


# ── criteria ──────────────────────────────────────────────────────────────────
def validate_criteria_drafts(drafts: Sequence[CriterionDraft]) -> list[Issue]:
    """Save-time checks: names, weights, max scores, duplicates."""
    issues: list[Issue] = []
    seen: dict[str, int] = {}
    for i, d in enumerate(drafts, start=1):
        label = f"row {i}"
        name = (d.name or "").strip()
        if not name:
            issues.append(Issue(ErrorCode.CRITERIA_INVALID_FIELD, "Criterion name is required.", label))
        else:
            key = name.casefold()
            if key in seen:
                issues.append(Issue(ErrorCode.CRITERIA_DUPLICATE_NAME,
                                    f"'{name}' duplicates row {seen[key]}. Criterion names must be unique.",
                                    label))
            else:
                seen[key] = i
        if not _finite(d.weight) or d.weight < 0:
            issues.append(Issue(ErrorCode.CRITERIA_INVALID_FIELD,
                                f"Weight for '{name or label}' must be a number ≥ 0.", label))
        if not _finite(d.max_score) or d.max_score <= 0:
            issues.append(Issue(ErrorCode.CRITERIA_INVALID_FIELD,
                                f"Max score for '{name or label}' must be greater than 0.", label))
    return issues


def validate_active_criteria(criteria: Sequence[Criterion], settings: Settings) -> list[Issue]:
    """Evaluation-time checks: at least one active criterion, weights total 100."""
    active = [c for c in criteria if c.is_active]
    if not active:
        return [Issue(ErrorCode.CRITERIA_NONE_ACTIVE,
                      "No active criteria. Activate at least one criterion in the Criteria tab.")]
    total = round(sum(c.weight for c in active), 6)
    if abs(total - settings.weight_total) > settings.weight_tolerance:
        return [Issue(ErrorCode.CRITERIA_WEIGHT_TOTAL,
                      f"Active criterion weights total {total:g}, but must total "
                      f"{settings.weight_total:g}. Adjust the weights in the Criteria tab.")]
    return []


def weight_total(drafts: Sequence[CriterionDraft]) -> float:
    """Sum of weights of active drafts (for the live indicator)."""
    return round(sum(d.weight for d in drafts if d.is_active and _finite(d.weight)), 6)


# ── files ─────────────────────────────────────────────────────────────────────
def check_pdf_bytes(filename: str, data: bytes, settings: Settings) -> list[Issue]:
    """Size and magic-byte checks for one upload (content checks happen at extraction)."""
    if not data:
        return [Issue(ErrorCode.FILE_EMPTY, f"'{filename}' is empty.", filename)]
    if len(data) > settings.max_file_bytes:
        size_mb = len(data) / (1024 * 1024)
        return [Issue(ErrorCode.FILE_TOO_LARGE,
                      f"'{filename}' is {size_mb:.1f} MB; the limit is {settings.max_file_mb:g} MB.", filename)]
    # Some writers prepend a few bytes before the header; the PDF spec allows it within 1 KB.
    if PDF_MAGIC not in data[:1024]:
        return [Issue(ErrorCode.FILE_NOT_PDF,
                      f"'{filename}' is not a PDF (its content does not start with a PDF header).",
                      filename)]
    return []


def find_duplicate_files(hashes: Sequence[tuple[str, str]]) -> list[Issue]:
    """Flag files whose SHA-256 matches an earlier upload.

    Args:
        hashes: ``(filename, sha256)`` pairs in upload order.
    """
    issues: list[Issue] = []
    first: dict[str, str] = {}
    for filename, digest in hashes:
        if digest in first:
            issues.append(Issue(ErrorCode.FILE_DUPLICATE,
                                f"'{filename}' has identical content to '{first[digest]}'. "
                                "Remove one of them.", filename))
        else:
            first[digest] = filename
    return issues


def check_file_count(n_files: int, settings: Settings) -> list[Issue]:
    """Peer comparison needs at least ``settings.min_suppliers`` valid PDFs."""
    if n_files < settings.min_suppliers:
        return [Issue(ErrorCode.FILE_TOO_FEW,
                      f"Upload at least {settings.min_suppliers} valid supplier PDFs "
                      f"(currently {n_files}); peer benchmarking needs more than one supplier.")]
    return []


# ── metadata ─────────────────────────────────────────────────────────────────
def normalise_name(name: str | None) -> str:
    """Trim and collapse internal whitespace."""
    return " ".join((name or "").split())


def validate_metadata(metas: Sequence[SupplierMeta], settings: Settings, today: date) -> list[Issue]:
    """Required fields, unique names (case-insensitive), no future dates, rating range.

    Args:
        metas: One entry per accepted file.
        settings: Rating range.
        today: Injected for testability (no hidden clock).
    """
    issues: list[Issue] = []
    seen: dict[str, str] = {}
    lo, hi = settings.experience_min, settings.experience_max
    for m in metas:
        name = normalise_name(m.supplier_name)
        if not name:
            issues.append(Issue(ErrorCode.META_NAME_REQUIRED, "Supplier name is required.", m.file_key))
        else:
            key = name.casefold()
            if key in seen:
                issues.append(Issue(ErrorCode.META_NAME_DUPLICATE,
                                    f"Supplier name '{name}' is already used in this batch "
                                    "(names are compared case-insensitively).", m.file_key))
            else:
                seen[key] = m.file_key
        if m.submission_date is None:
            issues.append(Issue(ErrorCode.META_DATE_REQUIRED, "Submission date is required.", m.file_key))
        elif m.submission_date > today:
            issues.append(Issue(ErrorCode.META_DATE_FUTURE,
                                f"Submission date {m.submission_date.isoformat()} is in the future.",
                                m.file_key))
        r = m.experience_rating
        if r is None or not _finite(r) or not lo <= r <= hi:
            issues.append(Issue(ErrorCode.META_RATING_INVALID,
                                f"Experience rating must be a number from {lo:g} to {hi:g}.", m.file_key))
    return issues


def _finite(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
