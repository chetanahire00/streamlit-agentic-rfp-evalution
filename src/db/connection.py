"""SQLite connection handling, schema creation and seeding."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

SEED_CRITERIA: list[tuple[str, str, float, float]] = [
    ("Technical Capability", "Architecture, integrations, scalability, technical fit", 30.0, 10.0),
    ("Implementation Plan", "Timeline, milestones, staffing, risk plan", 20.0, 10.0),
    ("Commercial Value", "Pricing clarity, total cost, assumptions", 20.0, 10.0),
    ("Security & Compliance", "Controls, certifications, privacy, auditability", 20.0, 10.0),
    ("Support & Experience", "Support model, similar projects, references", 10.0, 10.0),
]
"""The five criteria from the project brief: (name, description, weight, max_score)."""


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection in autocommit mode with foreign keys enforced.

    Autocommit (``isolation_level=None``) lets :func:`transaction` issue an
    explicit ``BEGIN`` so each unit of work is one atomic transaction.
    """
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run the enclosed statements in one transaction; roll back on any error."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def init_db(conn: sqlite3.Connection) -> int:
    """Create the schema and seed the brief's criteria if the table is empty.

    Idempotent: running it again changes nothing.

    Returns:
        Number of criteria inserted (0 when already seeded).
    """
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    count = conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0]
    if count:
        return 0
    now = utc_now_iso()
    with transaction(conn):
        conn.executemany(
            "INSERT OR IGNORE INTO evaluation_criteria "
            "(name, description, weight, max_score, is_active, updated_at) VALUES (?, ?, ?, ?, 1, ?)",
            [(n, d, w, m, now) for n, d, w, m in SEED_CRITERIA],
        )
    log.info("Seeded %d evaluation criteria", len(SEED_CRITERIA))
    return len(SEED_CRITERIA)
