-- scripts/init_schema.sql
-- PostgreSQL schema for Judicial AI Data Pipeline
-- Auto-executed by Docker on first startup (mounted as initdb script)

-- ── Extensions ──────────────────────────────────────────────────
-- (pg_trgm for potential text search; not required for vector search)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── Cases table ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cases (
    id                      SERIAL PRIMARY KEY,
    case_id                 VARCHAR(200) UNIQUE NOT NULL,
    file_path               TEXT NOT NULL,
    file_name               VARCHAR(500),
    file_hash               VARCHAR(64) UNIQUE NOT NULL,

    -- Case identifiers
    case_number             VARCHAR(500),
    case_type               VARCHAR(100),
    year                    INTEGER,

    -- Dates
    judgment_date           DATE,
    filing_date             DATE,

    -- Parties
    petitioner              TEXT,
    respondent              TEXT,

    -- Bench
    bench                   TEXT,
    judges                  JSONB DEFAULT '[]',

    -- Legal references
    acts_cited              JSONB DEFAULT '[]',
    citations               JSONB DEFAULT '[]',

    -- Outcome
    outcome                 VARCHAR(100),

    -- Stats
    page_count              INTEGER,
    word_count              INTEGER,
    extraction_method       VARCHAR(50),
    extraction_confidence   FLOAT,

    -- Timestamps
    created_at              TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at              TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_cases_year       ON cases (year);
CREATE INDEX IF NOT EXISTS idx_cases_case_type  ON cases (case_type);
CREATE INDEX IF NOT EXISTS idx_cases_outcome    ON cases (outcome);
CREATE INDEX IF NOT EXISTS idx_cases_file_hash  ON cases (file_hash);

-- GIN index for JSONB fields (search within arrays)
CREATE INDEX IF NOT EXISTS idx_cases_judges     ON cases USING gin (judges);
CREATE INDEX IF NOT EXISTS idx_cases_acts       ON cases USING gin (acts_cited);
CREATE INDEX IF NOT EXISTS idx_cases_citations  ON cases USING gin (citations);

-- ── Chunks table ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id                      SERIAL PRIMARY KEY,
    case_id                 VARCHAR(200) NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    chunk_index             INTEGER NOT NULL,
    text                    TEXT NOT NULL,
    char_count              INTEGER,
    approx_token_count      INTEGER,
    total_chunks            INTEGER,
    faiss_id                BIGINT DEFAULT -1,

    created_at              TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),

    CONSTRAINT unique_chunk UNIQUE (case_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_case_id   ON chunks (case_id);
CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id  ON chunks (faiss_id);

-- ── Pipeline runs table ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                      SERIAL PRIMARY KEY,
    run_id                  VARCHAR(36) UNIQUE NOT NULL,
    status                  VARCHAR(20) DEFAULT 'running',
    total_files             INTEGER DEFAULT 0,
    processed_files         INTEGER DEFAULT 0,
    failed_files            INTEGER DEFAULT 0,
    skipped_files           INTEGER DEFAULT 0,
    started_at              TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    completed_at            TIMESTAMP WITHOUT TIME ZONE,
    error_log               TEXT
);

-- ── Trigger: auto-update updated_at on cases ──────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cases_updated_at ON cases;
CREATE TRIGGER cases_updated_at
    BEFORE UPDATE ON cases
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
