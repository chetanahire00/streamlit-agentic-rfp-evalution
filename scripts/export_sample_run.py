"""Run the four sample proposals through the full pipeline and export the JSON.

Uses the offline Demo provider by default, so no API key is needed. Pass
``--provider``/``--model`` (with the matching ``*_API_KEY`` environment
variable) to produce the sample from a real LLM instead.

Usage::

    python scripts/export_sample_run.py [--out sample_output/sample_run.json]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROJECT_ROOT, get_settings
from src.agents.orchestrator import Orchestrator, export_json
from src.db.repository import Repository
from src.llm.base import LLMConfig
from src.llm.catalog import load_catalog
from src.models.schemas import SupplierInput
from src.ui.input_tab import SAMPLE_META
from src.utils.hashing import sha256_bytes
from src.utils.logging_setup import configure_logging, get_logger

log = get_logger("src.scripts.export_sample_run")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "sample_output" / "sample_run.json")
    parser.add_argument("--provider", default="demo")
    parser.add_argument("--model", default="heuristic-v1")
    args = parser.parse_args(argv)
    configure_logging(logging.INFO)

    settings = get_settings()
    provider = next(p for p in load_catalog(settings.providers_file) if p.id == args.provider)
    api_key = os.environ.get(provider.secret_name or "", "")
    config = LLMConfig(provider=provider.id, model=args.model, api_key=api_key,
                       supports_temperature=provider.supports_temperature, provider_label=provider.label)

    suppliers = []
    for filename, name, submitted, rating in SAMPLE_META:
        data = (PROJECT_ROOT / "sample_data" / filename).read_bytes()
        suppliers.append(SupplierInput(supplier_name=name, submission_date=submitted, experience_rating=rating,
                                       filename=filename, file_sha256=sha256_bytes(data), file_bytes=data))

    with tempfile.TemporaryDirectory() as tmp:
        repo = Repository(Path(tmp) / "export.db")
        repo.initialize()
        result = Orchestrator(repo, settings).run(suppliers, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(export_json(result) + "\n", encoding="utf-8")
    log.info("Exported run %s (%s) to %s", result.rfp_run_id, result.status.value, args.out)
    for r in result.ranking.rankings:
        log.info("  #%d %-16s PPI %.4f  absolute %.4f", r.final_rank, r.supplier_name, r.ppi, r.absolute_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
