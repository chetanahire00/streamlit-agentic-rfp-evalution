"""Shared fixtures and builders for the test suite (no network access)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.db.repository import Repository
from src.models.schemas import Criterion, CriterionScore, SupplierScorecard, SupplierStatus
from src.utils.secrets import clear_secrets


@pytest.fixture(autouse=True)
def _reset_secrets() -> None:
    clear_secrets()


@pytest.fixture()
def repo(tmp_path: Path) -> Repository:
    """A freshly initialised repository on a temporary SQLite file."""
    r = Repository(tmp_path / "test.db")
    r.initialize()
    return r


def make_criteria(weights: tuple[float, ...] = (30, 20, 20, 20, 10), max_score: float = 10) -> list[Criterion]:
    """Criteria with ids 1..N and the given weights."""
    names = ["Technical", "Implementation", "Commercial", "Security", "Support", "Extra1", "Extra2"]
    return [
        Criterion(criterion_id=i + 1, name=names[i], description=f"desc {i + 1}", weight=w,
                  max_score=max_score, is_active=True)
        for i, w in enumerate(weights)
    ]


def make_scorecard(
    name: str,
    scores: dict[int, float],
    criteria: list[Criterion],
    *,
    submitted: date = date(2026, 1, 10),
    rating: float = 3.0,
    status: SupplierStatus = SupplierStatus.OK,
) -> SupplierScorecard:
    """A validated scorecard with the given per-criterion scores."""
    return SupplierScorecard(
        supplier_name=name,
        submission_date=submitted,
        experience_rating=rating,
        filename=f"{name}.pdf",
        file_sha256=f"sha-{name}",
        status=status,
        criteria=[
            CriterionScore(criterion_id=c.criterion_id, criterion_name=c.name,
                           score=scores.get(c.criterion_id, 0.0), max_score=c.max_score,
                           weight=c.weight, justification="j", evidence="e")
            for c in criteria
        ],
    )


@pytest.fixture()
def criteria() -> list[Criterion]:
    return make_criteria()
