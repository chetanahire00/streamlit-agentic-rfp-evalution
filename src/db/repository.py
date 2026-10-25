"""Repository: the only module that issues SQL.

All statements use ``?`` placeholders (no string formatting of values), so user
input can never alter a query. Each public method opens its own short-lived
connection, which is safe across Streamlit's script threads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.db.connection import connect, init_db, transaction, utc_now_iso
from src.models.schemas import (
    Criterion,
    RankingResult,
    RunResult,
    RunStatus,
    SupplierRanking,
    SupplierScorecard,
    WarningRecord,
)
from src.utils.errors import ErrorCode, PersistenceError
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CriterionDraft:
    """A criterion row as edited in the UI (``criterion_id`` is None for new rows)."""

    criterion_id: int | None
    name: str
    description: str
    weight: float
    max_score: float
    is_active: bool


@dataclass(frozen=True)
class RunSummary:
    """One row of the run-history selector."""

    rfp_run_id: str
    created_at: str
    status: str
    provider: str
    model: str
    supplier_count: int


class Repository:
    """Data access for criteria, runs, results and the LLM cache.

    Args:
        db_path: SQLite file path (``":memory:"`` is not supported because each
            call opens a new connection; tests use a temporary file).
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # ── setup ─────────────────────────────────────────────────────────────
    def initialize(self) -> int:
        """Create the schema and seed default criteria. Idempotent."""
        with self._conn() as conn:
            return init_db(conn)

    # ── criteria ──────────────────────────────────────────────────────────
    def list_criteria(self, *, active_only: bool = False) -> list[Criterion]:
        """Return criteria ordered by id (optionally only active ones)."""
        sql = (
            "SELECT criterion_id, name, description, weight, max_score, is_active "
            "FROM evaluation_criteria"
        )
        if active_only:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY criterion_id"
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [
            Criterion(
                criterion_id=r["criterion_id"],
                name=r["name"],
                description=r["description"],
                weight=r["weight"],
                max_score=r["max_score"],
                is_active=bool(r["is_active"]),
            )
            for r in rows
        ]

    def save_criteria(self, drafts: Sequence[CriterionDraft]) -> None:
        """Replace the criteria table with ``drafts`` in one transaction.

        Rows whose id is missing from ``drafts`` are deleted, existing ids are
        updated and rows without an id are inserted. Callers validate first
        (see ``input_validation.validate_criteria_drafts``); the schema's
        CHECK/UNIQUE constraints are the last line of defence.
        """
        now = utc_now_iso()
        keep_ids = [d.criterion_id for d in drafts if d.criterion_id is not None]
        try:
            with self._conn() as conn, transaction(conn):
                existing = {r[0] for r in conn.execute("SELECT criterion_id FROM evaluation_criteria")}
                for cid in sorted(existing - set(keep_ids)):
                    conn.execute("DELETE FROM evaluation_criteria WHERE criterion_id = ?", (cid,))
                # Rename pass: free names first so swapping names between rows cannot
                # violate UNIQUE mid-transaction.
                for d in drafts:
                    if d.criterion_id is not None and d.criterion_id in existing:
                        conn.execute(
                            "UPDATE evaluation_criteria SET name = ? WHERE criterion_id = ?",
                            (f"__tmp__{d.criterion_id}", d.criterion_id),
                        )
                for d in drafts:
                    params = (d.name.strip(), d.description.strip(), float(d.weight),
                              float(d.max_score), int(d.is_active), now)
                    if d.criterion_id is not None and d.criterion_id in existing:
                        conn.execute(
                            "UPDATE evaluation_criteria SET name = ?, description = ?, weight = ?, "
                            "max_score = ?, is_active = ?, updated_at = ? WHERE criterion_id = ?",
                            (*params, d.criterion_id),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO evaluation_criteria "
                            "(name, description, weight, max_score, is_active, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            params,
                        )
        except sqlite3.Error as exc:
            log.exception("Saving criteria failed")
            raise PersistenceError(
                ErrorCode.PERSISTENCE, "Could not save the criteria. No changes were made."
            ) from exc

    # ── runs ──────────────────────────────────────────────────────────────
    def create_run(
        self,
        run_id: str,
        created_at: str,
        provider: str,
        model: str,
        prompt_version: str,
        snapshot: Sequence[Criterion],
    ) -> None:
        """Insert a ``CREATED`` run and its criteria snapshot atomically."""
        try:
            with self._conn() as conn, transaction(conn):
                conn.execute(
                    "INSERT INTO rfp_runs (rfp_run_id, created_at, status, provider, model, "
                    "prompt_version) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, created_at, RunStatus.CREATED.value, provider, model, prompt_version),
                )
                conn.executemany(
                    "INSERT INTO run_criteria_snapshot "
                    "(rfp_run_id, criterion_id, name, description, weight, max_score) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [(run_id, c.criterion_id, c.name, c.description, c.weight, c.max_score)
                     for c in snapshot],
                )
        except sqlite3.Error as exc:
            log.exception("Creating run %s failed", run_id)
            raise PersistenceError(ErrorCode.PERSISTENCE, "Could not create the run record.") from exc

    def update_run_status(
        self, run_id: str, status: RunStatus, warnings: Sequence[WarningRecord] | None = None
    ) -> None:
        """Set a run's status (and optionally its run-level warnings)."""
        with self._conn() as conn:
            if warnings is None:
                conn.execute("UPDATE rfp_runs SET status = ? WHERE rfp_run_id = ?",
                             (status.value, run_id))
            else:
                conn.execute(
                    "UPDATE rfp_runs SET status = ?, warnings_json = ? WHERE rfp_run_id = ?",
                    (status.value, _dump([w.model_dump() for w in warnings]), run_id),
                )

    def save_run_results(self, result: RunResult) -> None:
        """Write all supplier results and the final run status in ONE transaction.

        Raises:
            PersistenceError: on any SQLite error; nothing is written.
        """
        by_name = {r.supplier_name: r for r in result.ranking.rankings}
        ranking_meta = {
            "benchmarks": {str(k): v for k, v in result.ranking.benchmarks.items()},
            "tie_break_explanations": result.ranking.tie_break_explanations,
            "warnings": [w.model_dump() for w in result.ranking.warnings],
        }
        try:
            with self._conn() as conn, transaction(conn):
                conn.execute("DELETE FROM supplier_results WHERE rfp_run_id = ?", (result.rfp_run_id,))
                for card in result.scorecards:
                    rank = by_name[card.supplier_name]
                    conn.execute(
                        "INSERT INTO supplier_results (rfp_run_id, supplier_name, submission_date, "
                        "experience_rating, file_sha256, absolute_score, ppi, final_rank, status, "
                        "result_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            result.rfp_run_id,
                            card.supplier_name,
                            card.submission_date.isoformat(),
                            card.experience_rating,
                            card.file_sha256,
                            rank.absolute_score,
                            rank.ppi,
                            rank.final_rank,
                            card.status.value,
                            _dump({
                                "scorecard": card.model_dump(mode="json"),
                                "ranking": rank.model_dump(mode="json"),
                            }),
                        ),
                    )
                conn.execute(
                    "UPDATE rfp_runs SET status = ?, warnings_json = ?, ranking_json = ? "
                    "WHERE rfp_run_id = ?",
                    (
                        result.status.value,
                        _dump([w.model_dump() for w in result.warnings]),
                        _dump(ranking_meta),
                        result.rfp_run_id,
                    ),
                )
        except (sqlite3.Error, KeyError) as exc:
            log.exception("Persisting run %s failed; rolled back", result.rfp_run_id)
            raise PersistenceError(
                ErrorCode.PERSISTENCE, "Could not save the run results. Nothing was written."
            ) from exc

    def list_runs(self, limit: int = 50) -> list[RunSummary]:
        """Most recent runs first, for the history selector."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.rfp_run_id, r.created_at, r.status, r.provider, r.model, "
                "COUNT(s.result_id) AS n FROM rfp_runs r "
                "LEFT JOIN supplier_results s ON s.rfp_run_id = r.rfp_run_id "
                "GROUP BY r.rfp_run_id ORDER BY r.created_at DESC, r.rfp_run_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [RunSummary(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def load_run(self, run_id: str) -> RunResult | None:
        """Rebuild a :class:`RunResult` from SQLite, or ``None`` if unknown/incomplete."""
        with self._conn() as conn:
            run = conn.execute("SELECT * FROM rfp_runs WHERE rfp_run_id = ?", (run_id,)).fetchone()
            if run is None or run["ranking_json"] is None:
                return None
            snap = conn.execute(
                "SELECT criterion_id, name, description, weight, max_score FROM run_criteria_snapshot "
                "WHERE rfp_run_id = ? ORDER BY criterion_id",
                (run_id,),
            ).fetchall()
            rows = conn.execute(
                "SELECT result_json FROM supplier_results WHERE rfp_run_id = ? ORDER BY final_rank",
                (run_id,),
            ).fetchall()
        meta = json.loads(run["ranking_json"])
        payloads = [json.loads(r[0]) for r in rows]
        ranking = RankingResult(
            rankings=[SupplierRanking.model_validate(p["ranking"]) for p in payloads],
            benchmarks={int(k): v for k, v in meta["benchmarks"].items()},
            tie_break_explanations=meta["tie_break_explanations"],
            warnings=[WarningRecord.model_validate(w) for w in meta["warnings"]],
        )
        scorecards = [SupplierScorecard.model_validate(p["scorecard"]) for p in payloads]
        scorecards.sort(key=lambda c: c.supplier_name.casefold())
        return RunResult(
            rfp_run_id=run["rfp_run_id"],
            created_at=run["created_at"],
            status=RunStatus(run["status"]),
            provider=run["provider"],
            model=run["model"],
            prompt_version=run["prompt_version"],
            criteria_snapshot=[
                Criterion(criterion_id=s[0], name=s[1], description=s[2], weight=s[3],
                          max_score=s[4], is_active=True)
                for s in snap
            ],
            scorecards=scorecards,
            ranking=ranking,
            warnings=[WarningRecord.model_validate(w) for w in json.loads(run["warnings_json"])],
        )

    # ── LLM cache ─────────────────────────────────────────────────────────
    def cache_get(self, key: str) -> str | None:
        """Return the cached raw LLM response for ``key``, if any."""
        with self._conn() as conn:
            row = conn.execute("SELECT response_text FROM llm_cache WHERE cache_key = ?",
                               (key,)).fetchone()
        return row[0] if row else None

    def cache_put(self, key: str, provider: str, model: str, prompt_version: str, text: str) -> None:
        """Store a raw LLM response (first write wins, so results never drift)."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO llm_cache "
                "(cache_key, provider, model, prompt_version, response_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, provider, model, prompt_version, text, utc_now_iso()),
            )


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
