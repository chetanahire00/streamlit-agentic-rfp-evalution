-- Agentic RFP Evaluator: SQLite schema. Idempotent (safe to run on every start).

CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    description  TEXT    NOT NULL DEFAULT '',
    weight       REAL    NOT NULL CHECK (weight >= 0),
    max_score    REAL    NOT NULL CHECK (max_score > 0),
    is_active    INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    updated_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id     TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL
                   CHECK (status IN ('CREATED', 'RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED')),
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    warnings_json  TEXT NOT NULL DEFAULT '[]',
    ranking_json   TEXT
);

CREATE TABLE IF NOT EXISTS run_criteria_snapshot (
    rfp_run_id   TEXT    NOT NULL REFERENCES rfp_runs (rfp_run_id) ON DELETE CASCADE,
    criterion_id INTEGER NOT NULL,
    name         TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    weight       REAL    NOT NULL,
    max_score    REAL    NOT NULL,
    PRIMARY KEY (rfp_run_id, criterion_id)
);

CREATE TABLE IF NOT EXISTS supplier_results (
    result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id        TEXT    NOT NULL REFERENCES rfp_runs (rfp_run_id) ON DELETE CASCADE,
    supplier_name     TEXT    NOT NULL,
    submission_date   TEXT    NOT NULL,
    experience_rating REAL    NOT NULL,
    file_sha256       TEXT    NOT NULL,
    absolute_score    REAL    NOT NULL,
    ppi               REAL    NOT NULL,
    final_rank        INTEGER NOT NULL,
    status            TEXT    NOT NULL,
    result_json       TEXT    NOT NULL,
    UNIQUE (rfp_run_id, supplier_name)
);

-- Raw LLM responses keyed by hash(pdf_text + criteria_snapshot + provider + model + prompt_version).
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key      TEXT PRIMARY KEY,
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    response_text  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_supplier_results_run ON supplier_results (rfp_run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created ON rfp_runs (created_at);
