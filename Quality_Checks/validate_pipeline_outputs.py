"""Validate scanner output fields and context-window bounds without value-by-value matching.

Usage:
    set PMC_DB_CONN_STRING=host=localhost dbname=pubmed_analytics user=postgres password=... port=5432
    python validate_pipeline_outputs.py --jsonl output/publications_mentions_YYYYMMDD_HHMMSS.jsonl

This script checks live PostgreSQL schema/rows and JSONL structure. It does not compare
individual mention values between the two destinations, as requested for this validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
A
import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the environment configuration file in the same
# directory as this Python script.
PROJECT_DIR = Path(__file__).resolve().parent

ENV_FILE = PROJECT_DIR / "project-pubmed-pipeline.env"

# Load environment variables from the configuration file.
# Existing operating-system environment variables take precedence.
load_dotenv(dotenv_path=ENV_FILE)

DB_CONN_STRING = os.getenv("PMC_DB_CONN_STRING", "")
COMPARABLE_FIELDS = {
    "jax_id", "s3_uri", "pmcid", "article_version", "publication_title", "author",
    "publication_name", "publication_date", "matched_text", "matched_alias", "context_window",
}


def validate_jsonl(path: Path) -> dict:
    """Check that all JSONL records use a stable field set and have bounded context."""
    total = 0
    missing_context = 0
    missing_fields = {field: 0 for field in COMPARABLE_FIELDS}
    context_lengths: list[int] = []
    violations: list[tuple[int, int, int]] = []
    field_sets: set[tuple[str, ...]] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            field_sets.add(tuple(sorted(record.keys())))
            for field in COMPARABLE_FIELDS:
                if field not in record or record[field] is None or record[field] == "":
                    missing_fields[field] += 1
            context = record.get("context_window") or ""
            matched = record.get("matched_text") or ""
            if not context.strip():
                missing_context += 1
                continue
            context_lengths.append(len(context))
            # 500 characters before + match + 500 after + six wrapper dots.
            if not context.startswith("...") or not context.endswith("...") or len(context) > 1006 + len(matched):
                violations.append((line_no, len(context), len(matched)))

    return {
        "records": total,
        "unique_field_sets": len(field_sets),
        "field_sets": [list(fields) for fields in field_sets],
        "missing_fields": missing_fields,
        "missing_context": missing_context,
        "context_min_chars": min(context_lengths) if context_lengths else None,
        "context_max_chars": max(context_lengths) if context_lengths else None,
        "context_bound_violations": violations[:25],
        "context_bound_violation_count": len(violations),
    }


def validate_database() -> dict:
    """Inspect required live table columns and the newest 25 rows without value matching."""
    if not DB_CONN_STRING:
        raise RuntimeError("Set PMC_DB_CONN_STRING before running the live PostgreSQL checks.")
    with psycopg.connect(DB_CONN_STRING, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'publication_mentions'
                ORDER BY ordinal_position
            """)
            columns = {row["column_name"]: dict(row) for row in cur.fetchall()}
            if "context_window" in columns and "matched_text" in columns and "created_at" in columns:
                cur.execute("""
                    SELECT COUNT(*) AS row_count,
                           COUNT(*) FILTER (WHERE context_window IS NULL OR btrim(context_window) = '') AS missing_context,
                           MIN(length(context_window)) AS min_context_chars,
                           MAX(length(context_window)) AS max_context_chars,
                           COUNT(*) FILTER (
                               WHERE context_window IS NOT NULL
                                 AND (length(context_window) > 1006 + length(COALESCE(matched_text, ''))
                                      OR left(context_window, 3) <> '...'
                                      OR right(context_window, 3) <> '...')
                           ) AS context_bound_violations
                    FROM (
                        SELECT * FROM public.publication_mentions
                        ORDER BY created_at DESC LIMIT 25
                    ) AS recent
                """)
                sample = dict(cur.fetchone())
            else:
                sample = {"status": "NOT CHECKED: required context/ordering columns are missing"}

            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'strain_alias_catalog'
                ) AS alias_catalog_exists
            """)
            alias_exists = cur.fetchone()["alias_catalog_exists"]
            if alias_exists:
                cur.execute("""
                    SELECT COUNT(*) AS alias_catalog_rows,
                           COUNT(*) FILTER (WHERE mgi_candidate_count IS NULL) AS aliases_without_mgi_counts,
                           COUNT(*) FILTER (WHERE jax_candidate_count > 1) AS multi_jax_candidate_aliases
                    FROM public.strain_alias_catalog
                """)
                alias_catalog = dict(cur.fetchone())
            else:
                alias_catalog = {"status": "MISSING: run postgres_schema.py and the scanner catalog build"}

    required = COMPARABLE_FIELDS | {
        "mention_id", "mention_fingerprint", "pipeline_status", "is_ambiguous_in_jax_set",
        "ambiguity_metadata_quality", "normalized_alias", "candidate_identifiers",
        "n_candidates_jax_set", "n_candidates_all_mgi", "jax_candidate_count",
        "mgi_candidate_count",
    }
    return {
        "columns": columns,
        "missing_required_columns": sorted(required - set(columns)),
        "context_window_nullable": columns.get("context_window", {}).get("is_nullable"),
        "recent_25_row_context_check": sample,
        "alias_catalog": alias_catalog,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to the scanner's JSONL output file")
    args = parser.parse_args()
    jsonl_path = Path(args.jsonl).expanduser().resolve()
    if not jsonl_path.exists():
        parser.error(f"JSONL file does not exist: {jsonl_path}")

    jsonl_result = validate_jsonl(jsonl_path)
    database_result = validate_database()
    print("=== JSONL structural/context validation ===")
    print(json.dumps(jsonl_result, indent=2, ensure_ascii=False))
    print("=== PostgreSQL schema/context/catalog validation ===")
    print(json.dumps(database_result, indent=2, ensure_ascii=False, default=str))

    failed = (
        jsonl_result["unique_field_sets"] != 1
        or jsonl_result["missing_context"] > 0
        or jsonl_result["context_bound_violation_count"] > 0
        or bool(database_result["missing_required_columns"])
        or database_result["recent_25_row_context_check"]["missing_context"] > 0
        or database_result["recent_25_row_context_check"]["context_bound_violations"] > 0
    )
    print("RESULT:", "FAIL — inspect the diagnostics above" if failed else "PASS — schema and context checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
