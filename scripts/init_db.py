"""Create the SQLite schema and seed the five criteria from the brief.

Idempotent: running it again leaves an existing database unchanged.

Usage::

    python scripts/init_db.py [--db PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.db.repository import Repository
from src.utils.logging_setup import configure_logging, get_logger

log = get_logger("src.scripts.init_db")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=get_settings().db_path, help="SQLite file path")
    args = parser.parse_args(argv)
    configure_logging(logging.INFO)
    repo = Repository(args.db)
    inserted = repo.initialize()
    active = repo.list_criteria(active_only=True)
    log.info("Database ready at %s (%d seeded now, %d active criteria, weight total %.2f)",
             args.db, inserted, len(active), sum(c.weight for c in active))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
