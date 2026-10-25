"""Criteria rules: weights, active set, max score, duplicates, snapshot isolation."""

from __future__ import annotations

from config.settings import Settings
from src.db.repository import CriterionDraft, Repository
from src.models.schemas import Criterion
from src.tools.input_validation import validate_active_criteria, validate_criteria_drafts, weight_total
from src.utils.errors import ErrorCode
from tests.conftest import make_criteria

S = Settings()


def draft(name: str, weight: float = 20, max_score: float = 10, active: bool = True,
          cid: int | None = None) -> CriterionDraft:
    return CriterionDraft(cid, name, "", weight, max_score, active)


def test_weights_must_total_100() -> None:
    issues = validate_active_criteria(make_criteria((30, 20, 20, 20)), S)
    assert issues[0].code is ErrorCode.CRITERIA_WEIGHT_TOTAL
    assert "90" in issues[0].message


def test_weight_tolerance() -> None:
    assert validate_active_criteria(make_criteria((33.33, 33.33, 33.34)), S) == []
    assert validate_active_criteria(make_criteria((33.33, 33.33, 33.335)), S) == []
    assert validate_active_criteria(make_criteria((33.3, 33.3, 33.3)), S) != []


def test_inactive_criteria_ignored_in_total() -> None:
    inactive = Criterion(criterion_id=9, name="Off", weight=50, max_score=10, is_active=False)
    crits = [*make_criteria((30, 20, 20, 20, 10)), inactive]
    assert validate_active_criteria(crits, S) == []


def test_no_active_criteria() -> None:
    crits = [c.model_copy(update={"is_active": False}) for c in make_criteria()]
    assert validate_active_criteria(crits, S)[0].code is ErrorCode.CRITERIA_NONE_ACTIVE
    assert validate_active_criteria([], S)[0].code is ErrorCode.CRITERIA_NONE_ACTIVE


def test_invalid_max_score_and_negative_weight() -> None:
    issues = validate_criteria_drafts([draft("A", max_score=0), draft("B", weight=-1),
                                       draft("C", max_score=-2)])
    assert [i.code for i in issues] == [ErrorCode.CRITERIA_INVALID_FIELD] * 3


def test_non_finite_values_rejected() -> None:
    issues = validate_criteria_drafts([draft("A", weight=float("nan")), draft("B", max_score=float("inf"))])
    assert len(issues) == 2


def test_duplicate_names_rejected() -> None:
    issues = validate_criteria_drafts([draft("Security"), draft(" security ")])
    assert [i.code for i in issues] == [ErrorCode.CRITERIA_DUPLICATE_NAME]


def test_empty_name_rejected() -> None:
    assert validate_criteria_drafts([draft("  ")])[0].code is ErrorCode.CRITERIA_INVALID_FIELD


def test_weight_total_indicator() -> None:
    assert weight_total([draft("A", 60), draft("B", 40), draft("C", 25, active=False)]) == 100


def test_snapshot_isolation(repo: Repository) -> None:
    """Editing criteria after a run starts must not change that run's snapshot."""
    snapshot = repo.list_criteria(active_only=True)
    repo.create_run("run-x", "2026-10-25T10:00:00+00:00", "demo", "m", "v1.0", snapshot)
    edited = [CriterionDraft(c.criterion_id, c.name + " (edited)", c.description, c.weight, 5, c.is_active)
              for c in repo.list_criteria()]
    repo.save_criteria(edited)
    assert all(c.name.endswith("(edited)") for c in repo.list_criteria())
    from src.db.connection import connect
    conn = connect(repo.db_path)
    rows = conn.execute("SELECT name, max_score FROM run_criteria_snapshot WHERE rfp_run_id='run-x' "
                        "ORDER BY criterion_id").fetchall()
    conn.close()
    assert [r[0] for r in rows] == [c.name for c in snapshot]
    assert all(r[1] == 10 for r in rows)
