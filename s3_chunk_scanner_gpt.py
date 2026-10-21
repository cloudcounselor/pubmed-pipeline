"""
s3_chunk_scanner_gpt.py
Dave Carre  08/20/2026

** PMC announced a major change to its article-distribution system targeted for August 26, 2026. 
Legacy dataset files were removed, so code based on older PMC directory structures must be deprecated.
The current code is a fully up-to-date, alternative solution that accommodates the 
new PMC Dataset inventory, structure, formats, and naming conventions. ** 
Search PMC for articles mentioning strain aliases, fetch matching article-version
XML objects from the public pmc-oa-opendata S3 bucket, and capture bibliographic
metadata alongside every detected mention.

Search PMC for articles mentioning strain aliases, fetch matching article-version
XML objects from the public pmc-oa-opendata S3 bucket, and capture bibliographic
metadata alongside every detected mention.

Discovery follows PMC's documented eSearch -> PMCID -> S3 article-version
prefix pipeline. The current PMC AWS structure uses uncompressed .xml objects
under keys such as:
  PMC10009416.1/PMC10009416.1.xml

This script intentionally does NOT crawl the entire S3 bucket and does NOT
assume the legacy .xml.gz layout.

Environment:
  PMC_DB_CONN_STRING  Optional PostgreSQL connection string. If unset, hits
                      are written to a local JSONL file instead.
  PMC_CSV_PATH        Optional CSV path (default: strains_abridged.csv beside
                      this script, then one directory above).
"""

# Standard-library modules handle command-line options, JSONL output, logging,
# regular expressions, timing, XML parsing, paths, and type hints.
import argparse
import json
import logging
from datetime import date, datetime, timedelta
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

# Third-party libraries provide the PMC HTTP client, CSV processing, and S3 access.
import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config

# Repository/API settings and conservative defaults for a bounded first run.
BUCKET = "pmc-oa-opendata"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
DEFAULT_START_DATE = "2026/01/01"
DEFAULT_END_DATE = "2026/10/31"
DEFAULT_CONTEXT_WINDOW = 500
DEFAULT_MAX_ARTICLES = 2000
DEFAULT_MAX_HITS = 2000
DEFAULT_OUTPUT = "publications_mentions.jsonl"
ESEARCH_RESULT_CAP = 10000
USER_AGENT = "PMC-StrainMentionScanner/1.0 (research; contact repository administrator)"

# Keep progress and errors visible without printing full article contents.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("pmc_scanner")


# Give every execution a shared timestamp so its JSONL results and log are easy to pair.
def build_run_paths(output_arg: str, run_timestamp: str) -> tuple[Path, Path]:
    """Return timestamped JSONL and log paths in their dedicated subdirectories."""
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    logs_dir = script_dir / "logs"

    # Keep the requested base filename, but route all result files to /output.
    requested_output = Path(output_arg).expanduser()
    output_name = requested_output.name
    output_suffix = requested_output.suffix or ".jsonl"
    output_stem = requested_output.stem if requested_output.suffix else output_name
    stamped_output = output_dir / f"{output_stem}_{run_timestamp}{output_suffix}"
    log_path = logs_dir / f"s3_chunk_scanner_{run_timestamp}.log"
    return stamped_output, log_path


# Add a file handler for this run while retaining the existing console logging.
def configure_run_file_logging(log_path: Path) -> None:
    """Write the same timestamped progress messages to a per-run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)


# Read the CSV and build one record per JAX ID, including searchable aliases.
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


# Deduplicate aliases so repeated names do not trigger redundant API searches.
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


# Build the PMC indexed-search query; requests handles URL encoding separately.
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


# Ask PMC ESearch for matching numeric PMCID values and detect truncated results.
def esearch_pmc_ids(
    session: requests.Session,
    query: str,
    *,
    api_key: str | None = None,
    pause_seconds: float = 0.36,
    expected_count: int | None = None,
) -> list[str]:
    """Query PMC ESearch and return numeric PMCID suffixes with completeness checks."""
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
    ids = [str(value) for value in result.get("idlist", [])]
    if count > ESEARCH_RESULT_CAP:
        raise RuntimeError(
            f"ESearch ID request reports {count:,} matching records, exceeding the "
            f"{ESEARCH_RESULT_CAP:,}-result cap. Refusing potentially truncated results."
        )
    if expected_count is not None and count != expected_count:
        raise RuntimeError(
            f"ESearch count changed between preflight and ID retrieval: "
            f"preflight={expected_count:,}, retrieval response={count:,}. "
            "Refusing to treat this interval as complete; retry the run."
        )
    if len(ids) != count:
        raise RuntimeError(
            f"ESearch completeness check failed: API count={count:,}, "
            f"IDs returned={len(ids):,}. Refusing to use an incomplete ID list."
        )
    invalid_ids = [value for value in ids if not value.isdigit()]
    if invalid_ids:
        raise RuntimeError(f"ESearch returned non-numeric PMC identifiers: {invalid_ids[:3]!r}")
    time.sleep(pause_seconds)
    return ids


def esearch_result_count(
    session: requests.Session,
    query: str,
    *,
    api_key: str | None = None,
    pause_seconds: float = 0.36,
) -> int:
    """Ask ESearch for the count only, before requesting any result IDs."""
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": "0",
        "tool": "pmc_strain_mention_scanner",
    }
    if api_key:
        params["api_key"] = api_key

    response = session.get(ESEARCH_URL, params=params, timeout=(15, 90))
    response.raise_for_status()
    result = response.json().get("esearchresult", {})
    if result.get("error"):
        raise RuntimeError(f"ESearch error for query {query!r}: {result['error']}")
    count = int(result.get("count", "0"))
    time.sleep(pause_seconds)
    return count


def discover_pmc_ids_within_cap(
    session: requests.Session,
    alias: str,
    start_date: str,
    end_date: str,
    *,
    api_key: str | None = None,
) -> list[str]:
    """Partition a date range until each ESearch result set is within the API cap.

    Each accepted interval is preflight-counted immediately before ID retrieval.
    The returned list is deduplicated across intervals; a partition summary is
    logged so interval totals are not confused with unique PMCID totals.
    """
    start = date.fromisoformat(start_date.replace("/", "-"))
    end = date.fromisoformat(end_date.replace("/", "-"))
    if start > end:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")

    discovered: set[str] = set()
    pending: list[tuple[date, date]] = [(start, end)]
    accepted_intervals: list[tuple[str, str, int, int]] = []
    split_count = 0
    expected_total = 0
    returned_total = 0
    duplicate_ids_across_intervals = 0
    initial_count: int | None = None

    while pending:
        interval_start, interval_end = pending.pop(0)
        start_text = interval_start.strftime("%Y/%m/%d")
        end_text = interval_end.strftime("%Y/%m/%d")
        query = build_esearch_query(alias, start_text, end_text)
        count = esearch_result_count(session, query, api_key=api_key)
        if initial_count is None:
            initial_count = count
        log.info(
            "ESearch preflight | alias=%r | interval=%s..%s | matching_records=%s | cap=%s",
            alias, start_text, end_text, f"{count:,}", f"{ESEARCH_RESULT_CAP:,}",
        )

        if count > ESEARCH_RESULT_CAP:
            if interval_start == interval_end:
                raise RuntimeError(
                    f"A single-day ESearch interval for alias {alias!r} ({start_text}) "
                    f"still returns {count:,} records, above the {ESEARCH_RESULT_CAP:,}-result "
                    "cap; automatic date splitting cannot reduce this interval further."
                )
            midpoint = interval_start + (interval_end - interval_start) // 2
            right_start = midpoint + timedelta(days=1)
            split_count += 1
            pending[0:0] = [(interval_start, midpoint), (right_start, interval_end)]
            log.info(
                "ESearch partition split | alias=%r | oversized_interval=%s..%s | "
                "split_after=%s | left=%s..%s | right=%s..%s",
                alias, start_text, end_text, midpoint.strftime("%Y/%m/%d"),
                start_text, midpoint.strftime("%Y/%m/%d"),
                right_start.strftime("%Y/%m/%d"), end_text,
            )
            continue

        # Safety gate: never make an ID request for an interval whose preflight
        # count exceeds the API cap. Kept at the call site as a second guard.
        if count > ESEARCH_RESULT_CAP:
            raise RuntimeError(
                f"Safety gate blocked ID retrieval for {alias!r} {start_text}..{end_text}: "
                f"preflight count {count:,} exceeds cap {ESEARCH_RESULT_CAP:,}."
            )
        if count == 0:
            accepted_intervals.append((start_text, end_text, count, 0))
            continue

        ids = esearch_pmc_ids(
            session, query, api_key=api_key, expected_count=count
        )
        if len(ids) > ESEARCH_RESULT_CAP:
            raise RuntimeError(
                f"Safety gate failed for {alias!r} {start_text}..{end_text}: "
                f"retrieved {len(ids):,} IDs, above cap {ESEARCH_RESULT_CAP:,}."
            )
        before = len(discovered)
        discovered.update(ids)
        newly_added = len(discovered) - before
        duplicates_here = len(ids) - newly_added
        duplicate_ids_across_intervals += duplicates_here
        expected_total += count
        returned_total += len(ids)
        accepted_intervals.append((start_text, end_text, count, len(ids)))
        log.info(
            "ESearch interval accepted | alias=%r | interval=%s..%s | "
            "preflight_count=%s | ids_retrieved=%s | new_unique_ids=%s | "
            "overlap_duplicates=%s",
            alias, start_text, end_text, f"{count:,}", f"{len(ids):,}",
            f"{newly_added:,}", f"{duplicates_here:,}",
        )

    interval_summary = "; ".join(
        f"{left}..{right}: count={count:,}, ids={got:,}"
        for left, right, count, got in accepted_intervals
    ) or "none"
    log.info(
        "ESearch partition summary | alias=%r | requested_window=%s..%s | "
        "initial_preflight_count=%s | splits=%d | accepted_intervals=%d | "
        "sum_of_accepted_interval_counts=%s | total_ids_retrieved=%s | "
        "unique_pmcids=%s | duplicate_ids_across_intervals=%s | intervals=[%s]",
        alias, start.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d"),
        f"{(initial_count or 0):,}", split_count, len(accepted_intervals),
        f"{expected_total:,}", f"{returned_total:,}", f"{len(discovered):,}",
        f"{duplicate_ids_across_intervals:,}", interval_summary,
    )
    return sorted(discovered, key=int)


# Resolve a PMCID to the actual version-specific XML keys present in public S3.
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


# Flatten all XML text, including nested inline elements, for deterministic matching.
def flatten_xml(xml_bytes: bytes) -> str:
    """Extract text from all JATS XML elements, including nested inline markup."""
    root = ET.fromstring(xml_bytes)
    return " ".join(part.strip() for part in root.itertext() if part and part.strip())


# Extract bibliographic fields from the same JATS XML used for mention matching.
# Metadata is returned as plain strings so it can be attached to each hit record.
def extract_article_metadata(xml_bytes: bytes) -> dict:
    """Return title, first-author label, journal name, and publication date."""
    root = ET.fromstring(xml_bytes)

    # JATS elements may be namespaced; compare their local names for portability.
    def local_name(element):
        return element.tag.split("}")[-1] if isinstance(element.tag, str) else ""

    # Search descendants, not only direct children, because JATS nests metadata.
    def first_element(parent, name):
        return next(
            (element for element in parent.iter() if local_name(element) == name),
            None,
        )

    # Preserve text from nested tags while removing excess whitespace.
    def element_text(element):
        if element is None:
            return ""
        return " ".join(
            part.strip()
            for part in element.itertext()
            if part and part.strip()
        )

    # Article metadata holds the title, contributors, and article publication dates.
    article_meta = first_element(root, "article-meta")
    journal_meta = first_element(root, "journal-meta")
    if article_meta is None:
        # Some XML records may not contain standard JATS article metadata.
        return {
            "title": "",
            "author": "",
            "publication_name": "",
            "publication_date": "",
        }

    # Read the article title, retaining text nested in italic or other inline tags.
    title = element_text(first_element(article_meta, "article-title"))

    # Use the first contributor explicitly identified as an author.
    first_author = next(
        (
            element
            for element in article_meta.iter()
            if local_name(element) == "contrib"
            and element.attrib.get("contrib-type", "").lower() == "author"
        ),
        None,
    )
    author = ""
    if first_author is not None:
        surname = element_text(first_element(first_author, "surname"))
        given_names = element_text(first_element(first_author, "given-names"))
        # Find the first alphabetic character so compound or punctuated names work.
        initial = next((char.upper() for char in given_names if char.isalpha()), "")
        if surname:
            author = f"{surname}{initial}. et al"

    # The journal title belongs to journal-meta rather than article-meta.
    publication_name = ""
    if journal_meta is not None:
        publication_name = element_text(first_element(journal_meta, "journal-title"))

    # Prefer electronic publication dates, then print dates, then other pub-date entries.
    pub_dates = [
        element for element in article_meta.iter() if local_name(element) == "pub-date"
    ]

    def date_priority(element):
        # PMC/JATS records use slightly different attribute names across records.
        date_type = (
            element.attrib.get("pub-type", "")
            or element.attrib.get("date-type", "")
        ).lower()
        if date_type in {"epub", "electronic"}:
            return 0
        if date_type in {"ppub", "print"}:
            return 1
        return 2

    pub_dates.sort(key=date_priority)
    publication_date = ""
    for pub_date in pub_dates:
        year = element_text(first_element(pub_date, "year"))
        month = element_text(first_element(pub_date, "month"))
        day = element_text(first_element(pub_date, "day"))
        # Keep partial dates valid when only a year or year/month is supplied.
        if year:
            publication_date = year
            if month:
                publication_date += f"-{month.zfill(2)}"
            if day:
                publication_date += f"-{day.zfill(2)}"
            break

    return {
        "title": title,
        "author": author,
        "publication_name": publication_name,
        "publication_date": publication_date,
    }


# Match each strain alias against fetched article text and attach article metadata.
def find_alias_mentions(
    full_text: str,
    strain_records: list[dict],
    s3_uri: str,
    article_metadata: dict,
    context_window: int,
) -> list[dict]:
    """Match aliases and emit bibliographic fields plus context for every mention."""
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
                # Store bibliographic metadata before the matched phrase and context.
                # Empty strings are used when an article omits a metadata field.
                hits.append({
                    "jax_id": strain["jax_id"],
                    "s3_uri": s3_uri,
                    "publication_title": article_metadata.get("title", ""),
                    "author": article_metadata.get("author", ""),
                    "publication_name": article_metadata.get("publication_name", ""),
                    "publication_date": article_metadata.get("publication_date", ""),
                    "matched_text": match.group(0),
                    "matched_alias": alias,
                    "context_window": f"...{snippet}...",
                })
    return hits


# Fetch one XML object, extract its metadata and text, then find strain mentions.
def fetch_and_scan_xml(
    s3_client,
    key: str,
    strains: list[dict],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> list[dict]:
    """Fetch one PMC XML object and return matching mentions with article metadata."""
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    try:
        # Read the object once; both metadata extraction and text extraction use these bytes.
        xml_bytes = response["Body"].read()
    finally:
        # Always close the streaming response, including when parsing later fails.
        response["Body"].close()

    # Extract bibliographic details before matching so every hit can carry them.
    article_metadata = extract_article_metadata(xml_bytes)
    full_text = flatten_xml(xml_bytes)

    # Pass the extracted metadata into the matching function with the article's S3 URI.
    return find_alias_mentions(
        full_text,
        strains,
        f"s3://{BUCKET}/{key}",
        article_metadata,
        context_window,
    )


# Write one JSON object per line so results can be inspected or reprocessed easily.
def write_jsonl(hits: Iterable[dict], output_path: Path) -> int:
    """Append hits to JSONL, deduplicating by article, JAX ID, alias and context."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for hit in hits:
            handle.write(json.dumps(hit, ensure_ascii=False) + "\n")
            count += 1
    return count


# Optional database sink; this existing insert remains unchanged by metadata changes.
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


# Resolve the CSV path from a command-line argument, environment variable, or defaults.
def resolve_csv_path(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env_path = os.getenv("PMC_CSV_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    here = Path(__file__).resolve().parent
    candidates = [here / "strains_abridged.csv", here / "Strains.csv", here.parent / "strains_abridged.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find strains_abridged.csv or Strains.csv. Pass --csv PATH or set PMC_CSV_PATH."
    )


# Coordinate discovery, S3 key resolution, fetching, matching, and output limits.
def main() -> int:
    # Expose limits and date filters so small diagnostic runs can precede a full scan.
    parser = argparse.ArgumentParser(description="Find strain mentions in PMC Open Access article XML on AWS.")
    parser.add_argument("--csv", help="Path to Strains.csv / strains_abridged.csv")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="PMC live date start, YYYY/MM/DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="PMC live date end, YYYY/MM/DD")
    parser.add_argument("--no-date-filter", action="store_true", help="Search all indexed dates; may produce many more results")
    parser.add_argument("--max-articles", type=int, default=DEFAULT_MAX_ARTICLES)
    parser.add_argument("--max-hits", type=int, default=DEFAULT_MAX_HITS)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Base JSONL filename; a unique run timestamp is appended")
    parser.add_argument("--dry-run", action="store_true", help="Discover candidate PMCID values and S3 keys but do not fetch XML or write hits")
    args = parser.parse_args()

    # Create a high-resolution run ID; JSONL and log files share this exact suffix.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path, log_path = build_run_paths(args.output, run_timestamp)
    configure_run_file_logging(log_path)
    log.info("Run ID: %s", run_timestamp)
    log.info("Run log: %s", log_path)

    if args.max_articles < 1 or args.max_hits < 1:
        parser.error("--max-articles and --max-hits must be positive")

    # Load the input strain catalogue and create the alias list used for discovery.
    csv_path = resolve_csv_path(args.csv)
    strains = load_strains(csv_path)
    search_terms = make_search_terms(strains)
    log.info("Loaded %d strain rows and %d unique row/alias search terms from %s",
             len(strains), len(search_terms), csv_path)

    start_date = None if args.no_date_filter else args.start_date
    end_date = None if args.no_date_filter else args.end_date
    if (start_date is None) != (end_date is None):
        parser.error("Specify both --start-date and --end-date, or use --no-date-filter")

    # Reuse one HTTP session for PMC API calls and identify the research client.
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    api_key = os.getenv("NCBI_API_KEY")

    # Public, unsigned S3 access. PMC's current region is us-east-1.
    s3_client = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5, "mode": "standard"}),
    )

    # Use indexed PMC search to discover candidate PMCIDs; never crawl the whole bucket.
    candidate_ids = set()
    id_to_search_terms: dict[str, set[str]] = {}
    for index, (jax_id, alias) in enumerate(search_terms, start=1):
        try:
            if start_date and end_date:
                # Treat the requested dates as the outer search window. Preflight
                # counts and subdivide oversized intervals before fetching IDs.
                ids = discover_pmc_ids_within_cap(
                    session, alias, start_date, end_date, api_key=api_key
                )
            else:
                # --no-date-filter is an explicit request not to partition by date.
                # If it exceeds the API cap, the existing guard fails safely.
                query = build_esearch_query(alias, start_date, end_date)
                count = esearch_result_count(session, query, api_key=api_key)
                if count > ESEARCH_RESULT_CAP:
                    raise RuntimeError(
                        f"Unbounded ESearch query for alias {alias!r} returned "
                        f"{count:,} records, exceeding the {ESEARCH_RESULT_CAP:,}-result cap. "
                        "Run with a start/end date range to enable automatic splitting."
                    )
                ids = esearch_pmc_ids(
                    session, query, api_key=api_key, expected_count=count
                ) if count else []
        except Exception as exc:
            log.error("ESearch failed for alias %r (JAX ID %s): %s", alias, jax_id, exc)
            continue

        for pmcid in ids:
            candidate_ids.add(pmcid)
            id_to_search_terms.setdefault(pmcid, set()).add(alias)

        log.info(
            "Search term %d/%d | alias=%r | pmcids_for_this_alias=%s | "
            "cumulative_unique_pmcids_across_all_aliases=%s",
            index, len(search_terms), alias, f"{len(ids):,}", f"{len(candidate_ids):,}",
        )

    # Stop early if discovery produced no IDs; there is nothing to fetch in this run.
    if not candidate_ids:
        log.warning("ESearch returned no candidate articles. Verify query syntax and date range.")
        return 0

    log.info(
        "Discovery complete | cumulative_unique_pmcids_across_all_aliases=%s | "
        "search_terms_attempted=%d",
        f"{len(candidate_ids):,}", len(search_terms),
    )
    conn_string = os.getenv("PMC_DB_CONN_STRING")
    total_articles = 0
    total_hits = 0
    processed_keys = set()

    # Resolve each unique PMCID to versioned XML keys and process within configured limits.
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

            # Dry-run mode verifies discovery and key resolution without downloading XML.
            if args.dry_run:
                log.info("[DRY RUN] Would fetch s3://%s/%s", BUCKET, key)
                continue

            # Fetch the article, extract metadata/text, and run deterministic alias matching.
            try:
                hits = fetch_and_scan_xml(s3_client, key, strains, args.context_window)
            except Exception as exc:
                log.error("Fetch/parse failed for s3://%s/%s: %s", BUCKET, key, exc)
                continue

            total_articles += 1
            # Persist only actual matches, respecting the global hit limit.
            if hits:
                remaining = args.max_hits - total_hits
                hits = hits[:remaining]
                # Persist only actual matches, respecting the global hit limit.
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

    # Summarize the run and identify the JSONL destination when used.
    log.info("Finished. XML objects fetched/scanned: %d; hits recorded: %d", total_articles, total_hits)
    if not conn_string and not args.dry_run:
        log.info("Hit output: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
