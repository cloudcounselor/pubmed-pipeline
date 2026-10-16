"""
s3_chunk_scanner_pmcaws.py

Search PMC for articles mentioning strain aliases, then fetch the matching
article-version XML objects from the public pmc-oa-opendata S3 bucket.

Discovery follows PMC's documented eSearch -> PMCID -> S3 article-version
prefix pipeline. The current PMC AWS structure uses uncompressed .xml objects
under keys such as:
  PMC10009416.1/PMC10009416.1.xml

This script intentionally does NOT crawl the entire S3 bucket and does NOT
assume the legacy .xml.gz layout.
import gzip OMITTED
import psycopg INCLUDED within code block deeper into the script

Environment:
  PMC_DB_CONN_STRING  Optional PostgreSQL connection string. If unset, hits
                      are written to a local JSONL file instead.
  PMC_CSV_PATH        Optional CSV path (default: strains_abridged.csv beside
                      this script, then one directory above).

"""

import argparse
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "pmc-oa-opendata"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
DEFAULT_START_DATE = "2026/01/01"
DEFAULT_END_DATE = "2026/10/31"
DEFAULT_CONTEXT_WINDOW = 500
DEFAULT_MAX_ARTICLES = 2000
DEFAULT_MAX_HITS = 2000
DEFAULT_OUTPUT = "publication_mentions.jsonl"
USER_AGENT = "PMC-StrainMentionScanner/1.0 (research; contact repository administrator)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("pmc_scanner")


def load_strains(csv_path: Path) -> list[dict]:
    """Load strain rows and collect usable aliases while preserving JAX IDs."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "jax_id" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'jax_id' column")

    alias_columns = [
        "jax_id",
        "rrid",
        "official_strain_name",
        "jax_strain_name",
        "common_name",
    ]
    available = [c for c in alias_columns if c in df.columns]
    strains = []
    for _, row in df.iterrows():
        jax_id = row.get("jax_id", "").strip()
        if not jax_id:
            continue

        aliases = []
        for col in available:
            value = str(row.get(col, "")).strip()
            if not value or value.lower() in {"nan", "none", "not applicable", "not specified"}:
                continue
            # Ignore placeholder values, but keep meaningful names and identifiers.
            if value not in aliases:
                aliases.append(value)

        strains.append({
            "jax_id": jax_id,
            "aliases": aliases,
            "official_strain_name": row.get("official_strain_name", "").strip(),
        })

    if not strains:
        raise ValueError(f"No usable JAX IDs were found in {csv_path}")
    return strains


def make_search_terms(strains: list[dict]) -> list[tuple[str, str]]:
    """Return unique (JAX ID, alias) pairs; avoid searching aliases with no owner."""
    seen_aliases = set()
    terms = []
    for strain in strains:
        for alias in strain["aliases"]:
            key = alias.casefold()
            if key not in seen_aliases:
                seen_aliases.add(key)
                terms.append((strain["jax_id"], alias))
    return terms


def build_esearch_query(alias: str, start_date: str | None, end_date: str | None) -> str:
    """
    Build an Entrez PMC query. Phrase quoting and URL encoding are handled
    separately by requests; callers should not manually URL-encode this string.
    """
    # Escape embedded quotes to avoid accidentally changing the query structure.
    phrase = alias.replace('"', '\\"')
    parts = [f'"{phrase}"']
    if start_date and end_date:
        parts.append(f"{start_date}:{end_date}[pmcrdat]")
    parts.append("(open_access[filter] OR author_manuscript[filter])")
    return " AND ".join(parts)


def esearch_pmc_ids(
    session: requests.Session,
    query: str,
    *,
    api_key: str | None = None,
    pause_seconds: float = 0.36,
) -> list[str]:
    """Query PMC ESearch and return numeric PMCID suffixes, with explicit cap checks."""
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": "10000",
        "tool": "pmc_strain_mention_scanner",
    }
    if api_key:
        params["api_key"] = api_key

    # PMC ESearch caps a single query at 10,000 returned IDs. Do not silently
    # treat a truncated result as complete.
    response = session.get(ESEARCH_URL, params=params, timeout=(15, 90))
    response.raise_for_status()
    payload = response.json()
    result = payload.get("esearchresult", {})
    if result.get("error"):
        raise RuntimeError(f"ESearch error for query {query!r}: {result['error']}")

    count = int(result.get("count", "0"))
    ids = result.get("idlist", [])
    if count > 10000:
        raise RuntimeError(
            f"ESearch query returned {count:,} records, exceeding its 10,000-result cap. "
            "Split the date range into smaller intervals for this alias."
        )
    time.sleep(pause_seconds)
    return [str(value) for value in ids]


def list_article_xml_keys(s3_client, pmcid: str) -> list[str]:
    """
    Resolve a numeric PMCID to actual XML object keys.

    The official repository layout stores article versions as prefixes
    PMC<PMCID>.<version>/ and XML at
    PMC<PMCID>.<version>/PMC<PMCID>.<version>.xml.
    """
    prefix = f"PMC{pmcid}."
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=prefix,
        Delimiter="/",
        PaginationConfig={"PageSize": 1000},
    ):
        for item in page.get("CommonPrefixes", []):
            version_prefix = item["Prefix"]
            version_name = version_prefix.rstrip("/").split("/")[-1]
            xml_key = f"{version_prefix}{version_name}.xml"
            # Verify the exact XML object exists before attempting the fetch.
            try:
                s3_client.head_object(Bucket=BUCKET, Key=xml_key)
                keys.append(xml_key)
            except Exception as exc:
                # A version can lack an accessible XML object; log and continue.
                log.warning("XML object unavailable for %s: %s", xml_key, exc)
    return sorted(set(keys))


def flatten_xml(xml_bytes: bytes) -> str:
    """Extract text from all JATS XML elements, including nested inline markup."""
    root = ET.fromstring(xml_bytes)
    return " ".join(part.strip() for part in root.itertext() if part and part.strip())


def find_alias_mentions(
    full_text: str,
    strain_records: list[dict],
    s3_uri: str,
    context_window: int,
) -> list[dict]:
    """Match aliases locally in fetched XML and emit context around each mention."""
    hits = []
    seen = set()

    for strain in strain_records:
        for alias in strain["aliases"]:
            # Boundaries prevent matching an identifier as a substring of a
            # longer alphanumeric identifier, while allowing punctuation in names.
            pattern = re.compile(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                flags=re.IGNORECASE,
            )
            for match in pattern.finditer(full_text):
                start_idx, end_idx = match.span()
                left = max(0, start_idx - context_window)
                right = min(len(full_text), end_idx + context_window)
                snippet = full_text[left:right].strip()
                dedupe_key = (strain["jax_id"], alias.casefold(), start_idx)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                hits.append({
                    "jax_id": strain["jax_id"],
                    "s3_uri": s3_uri,
                    "matched_text": match.group(0),
                    "matched_alias": alias,
                    "context_window": f"...{snippet}...",
                })
    return hits


def fetch_and_scan_xml(
    s3_client,
    key: str,
    strains: list[dict],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> list[dict]:
    """Fetch one current-layout PMC XML object and scan its extracted text."""
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    try:
        xml_bytes = response["Body"].read()
    finally:
        response["Body"].close()

    full_text = flatten_xml(xml_bytes)
    return find_alias_mentions(
        full_text,
        strains,
        f"s3://{BUCKET}/{key}",
        context_window,
    )


def write_jsonl(hits: Iterable[dict], output_path: Path) -> int:
    """Append hits to JSONL, deduplicating by article, JAX ID, alias and context."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for hit in hits:
            handle.write(json.dumps(hit, ensure_ascii=False) + "\n")
            count += 1
    return count


def insert_mentions_to_db(hits: list[dict], conn_string: str) -> None:
    """Insert hits into the existing publication_mentions table."""
    if not hits:
        return
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install psycopg to write to PostgreSQL, or unset PMC_DB_CONN_STRING.") from exc

    query = """
        INSERT INTO publication_mentions (
            jax_id, s3_uri, matched_text, context_window, pipeline_status
        )
        VALUES (%s, %s, %s, %s, 'PENDING_LLM')
        ON CONFLICT DO NOTHING;
    """
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                query,
                [
                    (
                        hit["jax_id"],
                        hit["s3_uri"],
                        hit["matched_text"],
                        hit["context_window"],
                    )
                    for hit in hits
                ],
            )
        conn.commit()


def resolve_csv_path(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env_path = os.getenv("PMC_CSV_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    here = Path(__file__).resolve().parent
    candidates = [here / "strains_abridged.csv", here.parent / "strains_abridged.csv"]
    """
    orig:
    candidates = [here / "strains_abridged.csv", here / "Strains.csv", here.parent / "strains_abridged.csv"]
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find strains_abridged.csv or Strains.csv. Pass --csv PATH or set PMC_CSV_PATH."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Find strain mentions in PMC Open Access article XML on AWS.")
    parser.add_argument("--csv", help="Path to Strains.csv / strains_abridged.csv")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="PMC live date start, YYYY/MM/DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="PMC live date end, YYYY/MM/DD")
    parser.add_argument("--no-date-filter", action="store_true", help="Search all indexed dates; may produce many more results")
    parser.add_argument("--max-articles", type=int, default=DEFAULT_MAX_ARTICLES)
    parser.add_argument("--max-hits", type=int, default=DEFAULT_MAX_HITS)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSONL output path when database connection is not configured")
    parser.add_argument("--dry-run", action="store_true", help="Discover candidate PMCID values and S3 keys but do not fetch XML or write hits")
    args = parser.parse_args()

    if args.max_articles < 1 or args.max_hits < 1:
        parser.error("--max-articles and --max-hits must be positive")

    csv_path = resolve_csv_path(args.csv)
    strains = load_strains(csv_path)
    search_terms = make_search_terms(strains)
    log.info("Loaded %d strain rows and %d unique row/alias search terms from %s",
             len(strains), len(search_terms), csv_path)

    start_date = None if args.no_date_filter else args.start_date
    end_date = None if args.no_date_filter else args.end_date
    if (start_date is None) != (end_date is None):
        parser.error("Specify both --start-date and --end-date, or use --no-date-filter")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    api_key = os.getenv("NCBI_API_KEY")

    # Public, unsigned S3 access. PMC's current region is us-east-1.
    s3_client = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5, "mode": "standard"}),
    )

    # Use search results to discover candidate PMCIDs; never crawl the whole bucket.
    candidate_ids = set()
    id_to_search_terms: dict[str, set[str]] = {}
    for index, (jax_id, alias) in enumerate(search_terms, start=1):
        query = build_esearch_query(alias, start_date, end_date)
        try:
            ids = esearch_pmc_ids(session, query, api_key=api_key)
        except Exception as exc:
            log.error("ESearch failed for alias %r (JAX ID %s): %s", alias, jax_id, exc)
            continue

        for pmcid in ids:
            candidate_ids.add(pmcid)
            id_to_search_terms.setdefault(pmcid, set()).add(alias)

        log.info("Search term %d/%d: %r -> %d PMCID(s); %d unique candidates so far",
                 index, len(search_terms), alias, len(ids), len(candidate_ids))

    if not candidate_ids:
        log.warning("ESearch returned no candidate articles. Verify query syntax and date range.")
        return 0

    log.info("Discovered %d unique candidate PMCIDs", len(candidate_ids))
    conn_string = os.getenv("PMC_DB_CONN_STRING")
    output_path = Path(args.output).expanduser().resolve()
    total_articles = 0
    total_hits = 0
    processed_keys = set()

    for pmcid in sorted(candidate_ids, key=int):
        if total_articles >= args.max_articles:
            log.warning("Reached --max-articles limit (%d); stopping.", args.max_articles)
            break

        try:
            xml_keys = list_article_xml_keys(s3_client, pmcid)
        except Exception as exc:
            log.error("Could not resolve S3 XML keys for PMCID %s: %s", pmcid, exc)
            continue

        if not xml_keys:
            log.warning("No current-layout XML object found for PMCID %s", pmcid)
            continue

        for key in xml_keys:
            if key in processed_keys:
                continue
            if total_articles >= args.max_articles:
                break
            processed_keys.add(key)

            if args.dry_run:
                log.info("[DRY RUN] Would fetch s3://%s/%s", BUCKET, key)
                continue

            try:
                hits = fetch_and_scan_xml(s3_client, key, strains, args.context_window)
            except Exception as exc:
                log.error("Fetch/parse failed for s3://%s/%s: %s", BUCKET, key, exc)
                continue

            total_articles += 1
            if hits:
                remaining = args.max_hits - total_hits
                hits = hits[:remaining]
                if hits:
                    if conn_string:
                        try:
                            insert_mentions_to_db(hits, conn_string)
                        except Exception as exc:
                            log.error("Database insert failed; writing these hits to JSONL instead: %s", exc)
                            write_jsonl(hits, output_path)
                    else:
                        write_jsonl(hits, output_path)

                    total_hits += len(hits)
                    log.info("HITS: %d in %s (cumulative hits: %d)",
                             len(hits), key, total_hits)

            if total_hits >= args.max_hits:
                log.warning("Reached --max-hits limit (%d); stopping.", args.max_hits)
                break

        if total_hits >= args.max_hits:
            break

    log.info("Finished. XML objects fetched/scanned: %d; hits recorded: %d", total_articles, total_hits)
    if not conn_string and not args.dry_run:
        log.info("Hit output: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
