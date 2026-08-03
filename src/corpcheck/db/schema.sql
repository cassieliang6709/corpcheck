-- =============================================================================
-- Financial Research RAG Pipeline – PostgreSQL Schema
-- Requires: pgvector extension  (CREATE EXTENSION IF NOT EXISTS vector)
--           pg_trgm  (for trigram GIN indexes on text search)
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- =============================================================================
-- 1. companies
-- =============================================================================
CREATE TABLE IF NOT EXISTS companies (
    ticker          TEXT        PRIMARY KEY,
    name            TEXT        NOT NULL,
    sector          TEXT        NOT NULL,
    industry        TEXT,
    market_cap      NUMERIC,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_sector
    ON companies (sector);

-- =============================================================================
-- 2. filings
-- =============================================================================
CREATE TABLE IF NOT EXISTS filings (
    id              BIGSERIAL   PRIMARY KEY,
    ticker          TEXT        NOT NULL REFERENCES companies (ticker) ON DELETE CASCADE,
    filing_type     TEXT        NOT NULL,   -- '10-K', '10-Q', '8-K'
    fiscal_year     SMALLINT,
    period          TEXT,                   -- e.g. 'Q1', 'Q2', 'annual'
    filed_date      DATE,
    period_of_report DATE,
    accession_number TEXT,
    cik             TEXT,
    source_url      TEXT,
    local_path      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, filing_type, fiscal_year, period)
);

ALTER TABLE filings
    ADD COLUMN IF NOT EXISTS period_of_report DATE;

ALTER TABLE filings
    ADD COLUMN IF NOT EXISTS accession_number TEXT;

ALTER TABLE filings
    ADD COLUMN IF NOT EXISTS cik TEXT;

CREATE INDEX IF NOT EXISTS idx_filings_ticker
    ON filings (ticker);
CREATE INDEX IF NOT EXISTS idx_filings_type_year
    ON filings (filing_type, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_filings_filed_date
    ON filings (filed_date);
CREATE INDEX IF NOT EXISTS idx_filings_period_of_report
    ON filings (period_of_report);
CREATE INDEX IF NOT EXISTS idx_filings_accession
    ON filings (accession_number);

-- =============================================================================
-- 3. chunks  (SEC filing text chunks with vector + FTS)
-- =============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL   PRIMARY KEY,
    filing_id       BIGINT      NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    ticker          TEXT        NOT NULL,
    sector          TEXT        NOT NULL,
    filing_type     TEXT        NOT NULL,
    fiscal_year     SMALLINT,
    period          TEXT,
    filed_date      DATE,
    section_name    TEXT,                   -- 'MD&A', 'Risk Factors', etc.
    chunk_index     INTEGER,
    content         TEXT        NOT NULL,
    char_count      INTEGER     NOT NULL,
    token_count     INTEGER     NOT NULL,
    numeric_token_count INTEGER NOT NULL DEFAULT 0,
    number_density  REAL        NOT NULL DEFAULT 0,
    data_signal_score REAL      NOT NULL DEFAULT 0,
    is_quantitative BOOLEAN     NOT NULL DEFAULT FALSE,
    content_kind    TEXT        NOT NULL DEFAULT 'narrative',
    chunk_strategy  TEXT        NOT NULL DEFAULT 'sentence_pack',
    display_title   TEXT,
    chunk_group_key TEXT,
    structure_meta  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(384),
    content_tsv     TSVECTOR,
    source_url      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS filed_date DATE;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS chunk_index INTEGER;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS numeric_token_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS number_density REAL NOT NULL DEFAULT 0;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS data_signal_score REAL NOT NULL DEFAULT 0;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS is_quantitative BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_kind TEXT NOT NULL DEFAULT 'narrative';

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS chunk_strategy TEXT NOT NULL DEFAULT 'sentence_pack';

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS display_title TEXT;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS chunk_group_key TEXT;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS structure_meta JSONB NOT NULL DEFAULT '{}'::jsonb;

UPDATE chunks c
SET filed_date = f.filed_date
FROM filings f
WHERE c.filing_id = f.id
  AND c.filed_date IS NULL;

WITH chunk_positions AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY filing_id
            ORDER BY id
        ) - 1 AS inferred_chunk_index
    FROM chunks
    WHERE chunk_index IS NULL
)
UPDATE chunks c
SET chunk_index = cp.inferred_chunk_index
FROM chunk_positions cp
WHERE c.id = cp.id;

ALTER TABLE chunks
    ALTER COLUMN chunk_index SET NOT NULL;

WITH ranked_chunks AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY filing_id, chunk_index
            ORDER BY id
        ) AS row_num
    FROM chunks
)
DELETE FROM chunks c
USING ranked_chunks rc
WHERE c.id = rc.id
  AND rc.row_num > 1;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chunks_filing_id_chunk_index_key'
          AND conrelid = 'chunks'::regclass
    ) THEN
        ALTER TABLE chunks
            ADD CONSTRAINT chunks_filing_id_chunk_index_key
            UNIQUE (filing_id, chunk_index);
    END IF;
END
$$;

-- Vector similarity index (IVFFlat – good for recall at scale)
-- lists = sqrt(row_count) is a reasonable heuristic; for the expected corpus
-- size, 256 is a better default than 100.
DO $$
DECLARE
    index_reloptions TEXT[];
BEGIN
    SELECT c.reloptions
    INTO index_reloptions
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relname = 'idx_chunks_embedding_ivfflat'
      AND n.nspname = current_schema();

    IF index_reloptions IS NULL THEN
        EXECUTE '
            CREATE INDEX idx_chunks_embedding_ivfflat
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 256)
        ';
    ELSIF NOT ('lists=256' = ANY(index_reloptions)) THEN
        EXECUTE 'DROP INDEX idx_chunks_embedding_ivfflat';
        EXECUTE '
            CREATE INDEX idx_chunks_embedding_ivfflat
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 256)
        ';
    END IF;
END
$$;

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
    ON chunks USING GIN (content_tsv);

-- Btree lookups
CREATE INDEX IF NOT EXISTS idx_chunks_ticker
    ON chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_chunks_filing_id
    ON chunks (filing_id);
CREATE INDEX IF NOT EXISTS idx_chunks_filing_chunk_index
    ON chunks (filing_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_filed_date
    ON chunks (filed_date);
CREATE INDEX IF NOT EXISTS idx_chunks_sector_type
    ON chunks (sector, filing_type);
CREATE INDEX IF NOT EXISTS idx_chunks_fiscal_year
    ON chunks (fiscal_year);
CREATE INDEX IF NOT EXISTS idx_chunks_section
    ON chunks (section_name);
CREATE INDEX IF NOT EXISTS idx_chunks_quantitative
    ON chunks (is_quantitative, data_signal_score DESC);

-- Auto-populate tsvector on insert/update
CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector(
        'english',
        COALESCE(NEW.section_name, '') || ' ' || COALESCE(NEW.content, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_chunks_tsv ON chunks;
CREATE TRIGGER trig_chunks_tsv
    BEFORE INSERT OR UPDATE OF section_name, content ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();

-- =============================================================================
-- 4. market_data  (daily OHLCV + fundamentals from yfinance)
-- =============================================================================
CREATE TABLE IF NOT EXISTS market_data (
    ticker          TEXT        NOT NULL REFERENCES companies (ticker) ON DELETE CASCADE,
    date            DATE        NOT NULL,
    open            NUMERIC,
    high            NUMERIC,
    low             NUMERIC,
    close           NUMERIC,
    adj_close       NUMERIC,
    volume          BIGINT,
    market_cap      NUMERIC,
    pe_ratio        NUMERIC,
    pb_ratio        NUMERIC,
    ps_ratio        NUMERIC,
    dividend_yield  NUMERIC,
    beta            NUMERIC,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_market_data_date
    ON market_data (date);
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date
    ON market_data (ticker, date DESC);

-- =============================================================================
-- 5. financials  (quarterly income statement / balance sheet / cash flow)
-- =============================================================================
CREATE TABLE IF NOT EXISTS financials (
    id                      BIGSERIAL   PRIMARY KEY,
    ticker                  TEXT        NOT NULL REFERENCES companies (ticker) ON DELETE CASCADE,
    fiscal_year             SMALLINT    NOT NULL,
    period                  TEXT        NOT NULL,   -- 'Q1','Q2','Q3','Q4','annual'
    period_end_date         DATE,
    -- Income statement
    revenue                 NUMERIC,
    gross_profit            NUMERIC,
    operating_income        NUMERIC,
    net_income              NUMERIC,
    eps_basic               NUMERIC,
    eps_diluted             NUMERIC,
    ebitda                  NUMERIC,
    -- Balance sheet
    total_assets            NUMERIC,
    total_liabilities       NUMERIC,
    total_debt              NUMERIC,
    shareholders_equity     NUMERIC,
    cash_and_equivalents    NUMERIC,
    -- Cash flow
    operating_cash_flow     NUMERIC,
    capex                   NUMERIC,
    free_cash_flow          NUMERIC,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, fiscal_year, period)
);

CREATE INDEX IF NOT EXISTS idx_financials_ticker
    ON financials (ticker);
CREATE INDEX IF NOT EXISTS idx_financials_year_period
    ON financials (fiscal_year, period);

-- =============================================================================
-- 6. macro_indicators  (FRED series)
-- =============================================================================
CREATE TABLE IF NOT EXISTS macro_indicators (
    date            DATE        NOT NULL,
    indicator_id    TEXT        NOT NULL,   -- FRED series ID e.g. 'DFF'
    series_name     TEXT        NOT NULL,
    value           NUMERIC,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (date, indicator_id)
);

CREATE INDEX IF NOT EXISTS idx_macro_indicator_id
    ON macro_indicators (indicator_id, date);
CREATE INDEX IF NOT EXISTS idx_macro_date
    ON macro_indicators (date);

-- =============================================================================
-- 7. news_articles
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_articles (
    id              BIGSERIAL   PRIMARY KEY,
    ticker          TEXT        REFERENCES companies (ticker) ON DELETE SET NULL,
    title           TEXT        NOT NULL,
    content         TEXT,
    summary         TEXT,
    author          TEXT,
    published_date  TIMESTAMPTZ,
    source_url      TEXT        UNIQUE,
    source          TEXT,                   -- 'yahoo_rss', 'reuters_rss', etc.
    embedding       vector(384),
    content_tsv     TSVECTOR,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_embedding_ivfflat
    ON news_articles USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_news_content_tsv
    ON news_articles USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_news_ticker
    ON news_articles (ticker);
CREATE INDEX IF NOT EXISTS idx_news_published_date
    ON news_articles (published_date DESC);

CREATE OR REPLACE FUNCTION news_tsv_trigger() RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector(
        'english',
        COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, '') || ' ' || COALESCE(NEW.summary, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_news_tsv ON news_articles;
CREATE TRIGGER trig_news_tsv
    BEFORE INSERT OR UPDATE OF title, content, summary ON news_articles
    FOR EACH ROW EXECUTE FUNCTION news_tsv_trigger();

-- =============================================================================
-- 8. news_chunks
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_chunks (
    id              BIGSERIAL   PRIMARY KEY,
    news_article_id BIGINT      NOT NULL REFERENCES news_articles (id) ON DELETE CASCADE,
    ticker          TEXT        REFERENCES companies (ticker) ON DELETE SET NULL,
    published_date  TIMESTAMPTZ,
    source          TEXT,
    chunk_index     INTEGER     NOT NULL,
    content         TEXT        NOT NULL,
    token_count     INTEGER     NOT NULL,
    numeric_token_count INTEGER NOT NULL DEFAULT 0,
    number_density  REAL        NOT NULL DEFAULT 0,
    data_signal_score REAL      NOT NULL DEFAULT 0,
    is_quantitative BOOLEAN     NOT NULL DEFAULT FALSE,
    content_kind    TEXT        NOT NULL DEFAULT 'narrative',
    chunk_strategy  TEXT        NOT NULL DEFAULT 'article_sentence_pack',
    display_title   TEXT,
    chunk_group_key TEXT,
    structure_meta  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(384),
    content_tsv     TSVECTOR,
    source_url      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (news_article_id, chunk_index)
);

ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS numeric_token_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS number_density REAL NOT NULL DEFAULT 0;
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS data_signal_score REAL NOT NULL DEFAULT 0;
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS is_quantitative BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS content_kind TEXT NOT NULL DEFAULT 'narrative';
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS chunk_strategy TEXT NOT NULL DEFAULT 'article_sentence_pack';
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS display_title TEXT;
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS chunk_group_key TEXT;
ALTER TABLE news_chunks
    ADD COLUMN IF NOT EXISTS structure_meta JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_news_chunks_embedding_ivfflat
    ON news_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_news_chunks_content_tsv
    ON news_chunks USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_news_chunks_ticker
    ON news_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_news_chunks_article_id
    ON news_chunks (news_article_id);
CREATE INDEX IF NOT EXISTS idx_news_chunks_published_date
    ON news_chunks (published_date DESC);

CREATE OR REPLACE FUNCTION news_chunks_tsv_trigger() RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_news_chunks_tsv ON news_chunks;
CREATE TRIGGER trig_news_chunks_tsv
    BEFORE INSERT OR UPDATE OF content ON news_chunks
    FOR EACH ROW EXECUTE FUNCTION news_chunks_tsv_trigger();

-- =============================================================================
-- 9. earnings_transcripts
-- =============================================================================
CREATE TABLE IF NOT EXISTS earnings_transcripts (
    id              BIGSERIAL   PRIMARY KEY,
    ticker          TEXT        NOT NULL REFERENCES companies (ticker) ON DELETE CASCADE,
    fiscal_year     SMALLINT    NOT NULL,
    quarter         SMALLINT    NOT NULL,   -- 1..4
    content         TEXT        NOT NULL,
    published_date  DATE,
    source_url      TEXT        UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, fiscal_year, quarter)
);

CREATE INDEX IF NOT EXISTS idx_transcripts_ticker
    ON earnings_transcripts (ticker);
CREATE INDEX IF NOT EXISTS idx_transcripts_year_quarter
    ON earnings_transcripts (fiscal_year, quarter);

-- =============================================================================
-- 10. transcript_chunks
-- =============================================================================
CREATE TABLE IF NOT EXISTS transcript_chunks (
    id              BIGSERIAL   PRIMARY KEY,
    transcript_id   BIGINT      NOT NULL REFERENCES earnings_transcripts (id) ON DELETE CASCADE,
    ticker          TEXT        NOT NULL,
    fiscal_year     SMALLINT    NOT NULL,
    quarter         SMALLINT    NOT NULL,
    section         TEXT,                   -- 'prepared_remarks', 'qa', 'closing'
    chunk_index     INTEGER     NOT NULL,
    content         TEXT        NOT NULL,
    token_count     INTEGER,
    numeric_token_count INTEGER NOT NULL DEFAULT 0,
    number_density  REAL        NOT NULL DEFAULT 0,
    data_signal_score REAL      NOT NULL DEFAULT 0,
    is_quantitative BOOLEAN     NOT NULL DEFAULT FALSE,
    content_kind    TEXT        NOT NULL DEFAULT 'narrative',
    chunk_strategy  TEXT        NOT NULL DEFAULT 'speaker_turn',
    display_title   TEXT,
    chunk_group_key TEXT,
    structure_meta  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(384),
    content_tsv     TSVECTOR,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS numeric_token_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS number_density REAL NOT NULL DEFAULT 0;
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS data_signal_score REAL NOT NULL DEFAULT 0;
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS is_quantitative BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS content_kind TEXT NOT NULL DEFAULT 'narrative';
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS chunk_strategy TEXT NOT NULL DEFAULT 'speaker_turn';
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS display_title TEXT;
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS chunk_group_key TEXT;
ALTER TABLE transcript_chunks
    ADD COLUMN IF NOT EXISTS structure_meta JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'transcript_chunks_transcript_id_chunk_index_key'
          AND conrelid = 'transcript_chunks'::regclass
    ) THEN
        ALTER TABLE transcript_chunks
            ADD CONSTRAINT transcript_chunks_transcript_id_chunk_index_key
            UNIQUE (transcript_id, chunk_index);
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_tc_embedding_ivfflat
    ON transcript_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_tc_content_tsv
    ON transcript_chunks USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_tc_ticker
    ON transcript_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_tc_transcript_id
    ON transcript_chunks (transcript_id);
CREATE INDEX IF NOT EXISTS idx_tc_section
    ON transcript_chunks (section);

CREATE OR REPLACE FUNCTION transcript_chunks_tsv_trigger() RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_tc_tsv ON transcript_chunks;
CREATE TRIGGER trig_tc_tsv
    BEFORE INSERT OR UPDATE OF content ON transcript_chunks
    FOR EACH ROW EXECUTE FUNCTION transcript_chunks_tsv_trigger();

-- =============================================================================
-- Helper view: chunk search results with filing metadata
-- =============================================================================
DROP VIEW IF EXISTS v_chunk_search;
CREATE VIEW v_chunk_search AS
SELECT
    c.id,
    c.ticker,
    co.name            AS company_name,
    c.sector,
    c.filing_type,
    c.fiscal_year,
    c.period,
    c.section_name,
    c.numeric_token_count,
    c.number_density,
    c.data_signal_score,
    c.is_quantitative,
    c.content_kind,
    c.chunk_strategy,
    c.display_title,
    c.chunk_group_key,
    c.structure_meta,
    c.content,
    c.token_count,
    c.embedding,
    c.content_tsv,
    c.source_url,
    c.filed_date
FROM chunks c
JOIN companies co ON co.ticker = c.ticker;

DROP VIEW IF EXISTS v_retrieval_chunks;
CREATE VIEW v_retrieval_chunks AS
SELECT
    'sec'::text         AS source_type,
    c.id::text          AS chunk_id,
    c.ticker,
    co.name             AS company_name,
    c.sector,
    c.filing_type,
    c.fiscal_year,
    c.period::text      AS period_label,
    c.section_name      AS section_name,
    c.content_kind,
    c.chunk_strategy,
    c.display_title,
    c.chunk_group_key,
    c.structure_meta,
    c.numeric_token_count,
    c.number_density,
    c.data_signal_score,
    c.is_quantitative,
    c.content,
    c.token_count,
    c.embedding,
    c.content_tsv,
    c.source_url,
    c.filed_date::date  AS event_date
FROM chunks c
JOIN companies co ON co.ticker = c.ticker
UNION ALL
SELECT
    'news'::text        AS source_type,
    nc.id::text         AS chunk_id,
    nc.ticker,
    COALESCE(co.name, nc.ticker) AS company_name,
    co.sector,
    NULL::text          AS filing_type,
    NULL::smallint      AS fiscal_year,
    NULL::text          AS period_label,
    'news_article'::text AS section_name,
    nc.content_kind,
    nc.chunk_strategy,
    nc.display_title,
    nc.chunk_group_key,
    nc.structure_meta,
    nc.numeric_token_count,
    nc.number_density,
    nc.data_signal_score,
    nc.is_quantitative,
    nc.content,
    nc.token_count,
    nc.embedding,
    nc.content_tsv,
    nc.source_url,
    nc.published_date::date AS event_date
FROM news_chunks nc
LEFT JOIN companies co ON co.ticker = nc.ticker
UNION ALL
SELECT
    'transcript'::text  AS source_type,
    tc.id::text         AS chunk_id,
    tc.ticker,
    COALESCE(co.name, tc.ticker) AS company_name,
    co.sector,
    NULL::text          AS filing_type,
    tc.fiscal_year,
    ('Q' || tc.quarter)::text AS period_label,
    tc.section,
    tc.content_kind,
    tc.chunk_strategy,
    tc.display_title,
    tc.chunk_group_key,
    tc.structure_meta,
    tc.numeric_token_count,
    tc.number_density,
    tc.data_signal_score,
    tc.is_quantitative,
    tc.content,
    tc.token_count,
    tc.embedding,
    tc.content_tsv,
    et.source_url,
    et.published_date::date AS event_date
FROM transcript_chunks tc
LEFT JOIN earnings_transcripts et ON et.id = tc.transcript_id
LEFT JOIN companies co ON co.ticker = tc.ticker;
