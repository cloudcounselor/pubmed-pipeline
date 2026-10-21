"""
db_init_extended.py
Initialize/migrate the PostgreSQL schema and optionally seed named JAX products.
Dave Carre  08/20/2026

** PMC announced a major change to its article-distribution system targeted for August 26, 2026. 
Legacy dataset files were removed, so code based on older PMC directory structures must be deprecated.
The current code is a fully up-to-date, alternative solution that accommodates the 
new PMC Dataset inventory, structure, formats, and naming conventions. ** 
Search PMC for articles mentioning strain aliases, fetch matching article-version
XML objects from the public pmc-oa-opendata S3 bucket, and capture bibliographic
metadata alongside every detected mention.

REQUIRED ONCE PER DATABASE (and again after schema changes): run this initializer or
run postgres_schema.py directly. The scanner intentionally does not create/migrate
DDL. CSV seeding is OPTIONAL because the scanner itself upserts JAX IDs and product
names before inserting publication_mentions; use --csv here for preloading or
validation before a scan, not as a prerequisite for every scan.

postgres_schema.py remains the single source of truth for DDL. This file contains no
competing CREATE TABLE/index/trigger definitions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from postgres_schema import DB_CONN_STRING, initialize_database


def seed_canonical_jax_ids(csv_path: Path, conn_string: str = DB_CONN_STRING) -> int:
    """Upsert JAX IDs and strain names from the same CSV consumed by the scanner."""
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("CSV seeding requires psycopg v3 (pip install 'psycopg[binary]').") from exc
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "jax_id" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'jax_id' column")
    # Keep the name-selection policy aligned with the scanner. An explicit
    # product_name wins; otherwise use the canonical/official strain name, then
    # the alternate JAX strain name, then common_name. Duplicate JAX IDs are
    # collapsed while retaining the first non-empty product name encountered.
    name_columns = ("product_name", "official_strain_name", "jax_strain_name", "common_name")
    products: dict[str, str] = {}
    for _, row in df.iterrows():
        jax_id = str(row.get("jax_id", "")).strip()
        if not jax_id:
            continue
        product_name = next(
            (str(row.get(col, "")).strip() for col in name_columns
             if str(row.get(col, "")).strip()
             and str(row.get(col, "")).strip().lower() not in {"nan", "none", "not applicable", "not specified"}),
            "",
        )
        if jax_id not in products or (not products[jax_id] and product_name):
            products[jax_id] = product_name
    rows = list(products.items())
    query = """
        INSERT INTO jax_products (jax_id, product_name)
        VALUES (%s, %s)
        ON CONFLICT (jax_id) DO UPDATE
          SET product_name = COALESCE(NULLIF(EXCLUDED.product_name, ''), jax_products.product_name);
    """
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
    print(f"[SUCCESS] Seeded/updated {len(rows)} unique JAX IDs from {csv_path}.")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize/migrate PostgreSQL schema; optionally preload named JAX products.")
    parser.add_argument("--csv", help="Optional Strains.csv / strains_abridged.csv to seed jax_products.product_name after schema initialization")
    parser.add_argument("--conn-string", default=os.getenv("PMC_DB_CONN_STRING", DB_CONN_STRING), help="PostgreSQL connection string (prefer PMC_DB_CONN_STRING environment variable)")
    args = parser.parse_args()
    try:
        initialize_database(args.conn_string)
        if args.csv:
            seed_canonical_jax_ids(Path(args.csv).expanduser().resolve(), args.conn_string)
        return 0
    except Exception as exc:
        print(f"[ERROR] Initialization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
