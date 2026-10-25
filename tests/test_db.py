"""SQLite schema, seeding, persistence round-trip and rollback."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.init_db import main as init_main
from src.db.connection import connect, init_db, transaction
from src.db.repository import CriterionDraft, Repository
from src.models.schemas import RankingResult, RunResult, RunStatus, SupplierRanking
from src.utils.errors import PersistenceError
from tests.conftest import make_scorecard

EXPECTED_TABLES = {
    "evaluation_criteria", "rfp_runs", "run_criteria_snapshot", "supplier_results", "llm_cache",
}


def _ranking_for(cards) -> RankingResult:  # type: ignore[no-untyped-def]
    return RankingResult(
        rankings=[
            SupplierRanking(supplier_name=c.supplier_name, submission_date=c.submission_date,
                            experience_rating=c.experience_rating, absolute_score=50.0 - i,
                            ppi=90.0 - i, final_rank=i + 1, metrics=[])
            for i, c in enumerate(cards)
        ],
        benchmarks={1: 9.0},
        tie_break_explanations=["A above B: equal PPI 90; earlier submission date."],
        warnings=[],
    )


def _run(repo: Repository, criteria, run_id: str = "run-1") -> RunResult:  # type: ignore[no-untyped-def]
    repo.create_run(run_id, "2026-10-25T10:00:00+00:00", "demo", "heuristic-v1", "v1.0", criteria)
    cards = [make_scorecard("Alpha", {1: 8}, criteria), make_scorecard("Beta", {1: 6}, criteria)]
    return RunResult(
        rfp_run_id=run_id, created_at="2026-10-25T10:00:00+00:00", status=RunStatus.COMPLETED,
        provider="demo", model="heuristic-v1", prompt_version="v1.0", criteria_snapshot=criteria,
        scorecards=cards, ranking=_ranking_for(cards),
    )


def test_schema_created(repo: Repository) -> None:
    conn = connect(repo.db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert names >= EXPECTED_TABLES


def test_seed_matches_brief(repo: Repository) -> None:
    crits = repo.list_criteria()
    assert [c.weight for c in crits] == [30, 20, 20, 20, 10]
    assert all(c.max_score == 10 and c.is_active for c in crits)
    assert crits[0].name == "Technical Capability"


def test_seed_is_idempotent(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "x.db")
    assert repo.initialize() == 5
    assert repo.initialize() == 0
    assert len(repo.list_criteria()) == 5


def test_init_script_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "script.db"
    assert init_main(["--db", str(db)]) == 0
    assert init_main(["--db", str(db)]) == 0
    assert len(Repository(db).list_criteria()) == 5


def test_save_criteria_insert_update_delete(repo: Repository) -> None:
    crits = repo.list_criteria()
    drafts = [CriterionDraft(c.criterion_id, c.name, c.description, c.weight, c.max_score, c.is_active)
              for c in crits[:3]]
    drafts[0] = CriterionDraft(crits[0].criterion_id, "Tech v2", "d", 50, 10, True)
    drafts.append(CriterionDraft(None, "Sustainability", "green", 10, 5, False))
    repo.save_criteria(drafts)
    after = repo.list_criteria()
    assert [c.name for c in after] == ["Tech v2", "Implementation Plan", "Commercial Value", "Sustainability"]
    assert repo.list_criteria(active_only=True)[-1].name == "Commercial Value"


def test_save_criteria_name_swap(repo: Repository) -> None:
    a, b = repo.list_criteria()[:2]
    repo.save_criteria([
        CriterionDraft(a.criterion_id, b.name, "", 50, 10, True),
        CriterionDraft(b.criterion_id, a.name, "", 50, 10, True),
    ])
    names = [c.name for c in repo.list_criteria()]
    assert names == [b.name, a.name]


def test_save_criteria_constraint_violation_rolls_back(repo: Repository) -> None:
    before = repo.list_criteria()
    with pytest.raises(PersistenceError):
        repo.save_criteria([CriterionDraft(None, "Bad", "", -5, 10, True)])
    assert repo.list_criteria() == before


def test_run_round_trip(repo: Repository, criteria) -> None:  # type: ignore[no-untyped-def]
    result = _run(repo, criteria)
    repo.save_run_results(result)
    loaded = repo.load_run("run-1")
    assert loaded is not None
    assert loaded.status is RunStatus.COMPLETED
    assert [c.supplier_name for c in loaded.scorecards] == ["Alpha", "Beta"]
    assert loaded.ranking.tie_break_explanations == result.ranking.tie_break_explanations
    assert loaded.criteria_snapshot == criteria
    assert loaded.ranking.benchmarks == {1: 9.0}
    summaries = repo.list_runs()
    assert summaries[0].rfp_run_id == "run-1" and summaries[0].supplier_count == 2


def test_load_unknown_or_incomplete_run(repo: Repository, criteria) -> None:  # type: ignore[no-untyped-def]
    assert repo.load_run("nope") is None
    repo.create_run("r2", "2026-10-25T10:00:00+00:00", "demo", "m", "v1.0", criteria)
    assert repo.load_run("r2") is None


def test_save_results_rolls_back_on_failure(repo: Repository, criteria) -> None:  # type: ignore[no-untyped-def]
    result = _run(repo, criteria)
    # A scorecard with no matching ranking triggers a failure mid-transaction.
    result.scorecards.append(make_scorecard("Ghost", {}, criteria))
    with pytest.raises(PersistenceError):
        repo.save_run_results(result)
    conn = connect(repo.db_path)
    n = conn.execute("SELECT COUNT(*) FROM supplier_results").fetchone()[0]
    status = conn.execute("SELECT status FROM rfp_runs WHERE rfp_run_id='run-1'").fetchone()[0]
    conn.close()
    assert n == 0
    assert status == "CREATED"


def test_unique_supplier_per_run(repo: Repository, criteria) -> None:  # type: ignore[no-untyped-def]
    result = _run(repo, criteria)
    repo.save_run_results(result)
    conn = connect(repo.db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO supplier_results (rfp_run_id, supplier_name, submission_date, experience_rating,"
            " file_sha256, absolute_score, ppi, final_rank, status, result_json)"
            " VALUES ('run-1','Alpha','2026-01-01',1,'x',0,0,9,'OK','{}')"
        )
    conn.close()


def test_transaction_helper_rolls_back(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    with pytest.raises(RuntimeError), transaction(conn):
        conn.execute("DELETE FROM evaluation_criteria")
        raise RuntimeError("boom")
    assert conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0] == 5
    conn.close()


def test_status_update_and_cache(repo: Repository, criteria) -> None:  # type: ignore[no-untyped-def]
    repo.create_run("r3", "2026-10-25T10:00:00+00:00", "demo", "m", "v1.0", criteria)
    repo.update_run_status("r3", RunStatus.RUNNING)
    repo.update_run_status("r3", RunStatus.FAILED, [])
    assert repo.list_runs()[0].status == "FAILED"
    assert repo.cache_get("k") is None
    repo.cache_put("k", "demo", "m", "v1.0", '{"a":1}')
    repo.cache_put("k", "demo", "m", "v1.0", '{"a":2}')  # first write wins
    assert repo.cache_get("k") == '{"a":1}'
