"""
postgres_schema.py
Canonical PostgreSQL schema for PMC strain-mention extraction.
Dave Carre  08/20/2026

** PMC announced a major change to its article-distribution system targeted for August 26, 2026. 
Legacy dataset files were removed, so code based on older PMC directory structures must be deprecated.
The current code is a fully up-to-date, alternative solution that accommodates the 
new PMC Dataset inventory, structure, formats, and naming conventions. ** 
Search PMC for articles mentioning strain aliases, fetch matching article-version
XML objects from the public pmc-oa-opendata S3 bucket, and capture bibliographic
metadata alongside every detected mention.

Run this module directly, or use db_init_extended.py as the initializer/CSV seeder.
Keep DDL in this one file so the two scripts cannot drift into competing schemas.
"""
from __future__ import annotations

import os
import sys


DB_CONN_STRING = os.getenv(
    "PMC_DB_CONN_STRING",
    "host=localhost dbname=pubmed_analytics user=postgres password=secure_password port=5432",
)

SCHEMA_SQL = r"""
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS jax_products (
    jax_id VARCHAR(64) PRIMARY KEY,
    product_name VARCHAR(255),
    expected_status VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE jax_products ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pipeline_status_enum') THEN
        CREATE TYPE pipeline_status_enum AS ENUM (
            'PENDING_LLM', 'PROCESSED', 'HUMAN_REVIEW', 'FALSE_POSITIVE', 'FAILED'
        );
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS publication_mentions (
    mention_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mention_fingerprint TEXT,
    jax_id VARCHAR(64) NOT NULL REFERENCES jax_products(jax_id) ON DELETE CASCADE,
    s3_uri TEXT NOT NULL,
    pmcid VARCHAR(16),
    article_version VARCHAR(16),
    publication_title TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    publication_name TEXT NOT NULL DEFAULT '',
    publication_date TEXT NOT NULL DEFAULT '',
    matched_text TEXT NOT NULL,
    matched_alias TEXT NOT NULL DEFAULT '',
    context_window TEXT NOT NULL,
    is_valid_jax_mention BOOLEAN DEFAULT FALSE,
    product_status VARCHAR(32),
    confidence_score NUMERIC(4,3),
    ai_justification TEXT,
    pipeline_status pipeline_status_enum NOT NULL DEFAULT 'PENDING_LLM',
    error_log TEXT,
    extracted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Idempotent migrations for databases created by either earlier schema script.
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS mention_fingerprint TEXT;
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS pmcid VARCHAR(16);
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS article_version VARCHAR(16);
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS publication_title TEXT NOT NULL DEFAULT '';
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS author TEXT NOT NULL DEFAULT '';
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS publication_name TEXT NOT NULL DEFAULT '';
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS publication_date TEXT NOT NULL DEFAULT '';
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS matched_alias TEXT NOT NULL DEFAULT '';
ALTER TABLE publication_mentions ADD COLUMN IF NOT EXISTS extracted_at TIMESTAMPTZ;

-- Backfill legacy rows. Their older dedupe key did not preserve the alias or metadata.
UPDATE publication_mentions
SET matched_alias = COALESCE(NULLIF(matched_alias, ''), matched_text)
WHERE matched_alias IS NULL OR matched_alias = '';
UPDATE publication_mentions
SET mention_fingerprint = md5(concat_ws(chr(31), jax_id, s3_uri, matched_alias, matched_text, context_window))
WHERE mention_fingerprint IS NULL OR mention_fingerprint = '';

-- Old data may contain duplicate legacy rows. Make legacy fingerprints unique before
-- adding the unique index; newly scanned rows use SHA-256 fingerprints from Python.
WITH ranked AS (
    SELECT mention_id, mention_fingerprint,
           row_number() OVER (PARTITION BY mention_fingerprint ORDER BY created_at, mention_id) AS rn
    FROM publication_mentions
)
UPDATE publication_mentions AS p
SET mention_fingerprint = p.mention_fingerprint || '-' || p.mention_id::text
FROM ranked AS r
WHERE p.mention_id = r.mention_id AND r.rn > 1;
ALTER TABLE publication_mentions ALTER COLUMN mention_fingerprint SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pub_mentions_jax_id ON publication_mentions(jax_id);
CREATE INDEX IF NOT EXISTS idx_pub_mentions_pipeline_status ON publication_mentions(pipeline_status);
CREATE INDEX IF NOT EXISTS idx_pub_mentions_pmcid ON publication_mentions(pmcid);
CREATE INDEX IF NOT EXISTS idx_pub_mentions_publication_date ON publication_mentions(publication_date);
CREATE UNIQUE INDEX IF NOT EXISTS uq_publication_mentions_fingerprint
    ON publication_mentions(mention_fingerprint);

CREATE TABLE IF NOT EXISTS human_review_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mention_id UUID NOT NULL REFERENCES publication_mentions(mention_id) ON DELETE CASCADE,
    reviewer_identity VARCHAR(128) NOT NULL,
    previous_ai_status VARCHAR(32) DEFAULT 'HUMAN_REVIEW',
    final_human_override_status VARCHAR(32) NOT NULL,
    reviewer_notes TEXT,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_audit_log_mention_id ON human_review_audit_log(mention_id);

CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_pub_mentions_modtime ON publication_mentions;
CREATE TRIGGER update_pub_mentions_modtime
    BEFORE UPDATE ON publication_mentions
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

DROP TRIGGER IF EXISTS update_jax_products_modtime ON jax_products;
CREATE TRIGGER update_jax_products_modtime
    BEFORE UPDATE ON jax_products
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();
"""


def initialize_database(conn_string: str = DB_CONN_STRING) -> None:
    """Create or migrate the canonical schema in one transaction."""
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("PostgreSQL schema initialization requires psycopg v3 (pip install 'psycopg[binary]').") from exc
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
        print("[SUCCESS] PostgreSQL schema initialized/migrated: jax_products, publication_mentions, and human_review_audit_log.")
    except Exception as exc:
        print(f"[CRITICAL] PostgreSQL schema initialization failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    initialize_database()
