



"""Asynchronous coordinator for the OpenAI/LangGraph mention-evaluation stage.

This module claims PENDING_LLM rows, sends each row through compiled_graph, and
persists the resulting classification. It never performs PMC searching; that work
belongs to s3_chunk_scanner_gpt.py. Database credentials and OpenAI settings are
read from environment variables rather than embedded in source code.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any
import time
from concurrent.futures import ThreadPoolExecutor
import psycopg
from psycopg.rows import dict_row

from langgraph_orchestrator import AgentState, JAXProductEvaluation, compiled_graph


# ==========================================
# 1. RUNTIME CONFIGURATION AND LOGGING
# ==========================================
DB_CONN_STRING = os.getenv(
    "PMC_DB_CONN_STRING",
    "host=localhost dbname=pubmed_analytics user=postgres password=secure_password port=5432",
)
BATCH_SIZE = max(1, int(os.getenv("PMC_LLM_BATCH_SIZE", "50")))
MAX_CONCURRENCY = max(1, int(os.getenv("PMC_LLM_MAX_CONCURRENCY", "5")))
CLAIM_STALE_MINUTES = max(1, int(os.getenv("PMC_LLM_CLAIM_STALE_MINUTES", "30")))

# Log to console and a persistent file so failures can be diagnosed after a run.
logging.basicConfig(
    level=os.getenv("PMC_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] [PIPELINE] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline_performance.log", mode="a", encoding="utf-8")],
)
logger = logging.getLogger("PipelineCoordinator")


# ==========================================
# 2. ATOMIC BATCH CLAIMING
# ==========================================
def claim_pending_batch(batch_size: int) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim a batch so concurrent coordinator processes do not duplicate work.

    A claim token and timestamp are used instead of adding an enum value to the pipeline
    status. If a process crashes, a later run can reclaim the row after the stale timeout.
    """
    claim_token = str(uuid.uuid4())
    claim_sql = """
        WITH candidates AS (
            SELECT mention_id
            FROM publication_mentions
            WHERE pipeline_status = 'PENDING_LLM'
              AND (llm_claimed_at IS NULL
                   OR llm_claimed_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 minute'))
            ORDER BY created_at, mention_id
            FOR UPDATE SKIP LOCKED
            LIMIT %s
        ), claimed AS (
            UPDATE publication_mentions AS p
            SET llm_claimed_at = CURRENT_TIMESTAMP,
                llm_claim_token = %s
            FROM candidates AS c
            WHERE p.mention_id = c.mention_id
            RETURNING p.mention_id, p.jax_id, p.s3_uri, p.pmcid,
                      p.publication_title, p.author, p.publication_name, p.publication_date,
                      p.matched_text, p.matched_alias, p.context_window,
                      p.normalized_term, p.n_candidates_jax_set, p.n_candidates_all_mgi,
                      p.is_ambiguous_in_jax_set, p.ambiguity_metadata_quality,
                      p.normalized_alias, p.candidate_identifiers
        )
        SELECT claimed.*,
               a.total_occurrences AS alias_total_occurrences,
               a.distinct_articles AS alias_distinct_articles,
               a.jax_candidate_count AS catalog_jax_candidate_count,
               a.mgi_candidate_count AS catalog_mgi_candidate_count,
               a.candidate_identifiers AS catalog_candidate_identifiers
        FROM claimed
        LEFT JOIN strain_alias_catalog AS a ON a.normalized_alias = claimed.normalized_alias
    """
    # psycopg commits the claim transaction when this connection context exits normally.
    with psycopg.connect(DB_CONN_STRING, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(claim_sql, (CLAIM_STALE_MINUTES, batch_size, claim_token))
            records = [dict(row) for row in cur.fetchall()]
    return claim_token, records


# ==========================================
# 3. PER-MENTION ASYNC WORKER
# ==========================================
async def process_single_mention(
    semaphore: asyncio.Semaphore,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Run one synchronous LangGraph invocation off the event loop under a semaphore."""
    async with semaphore:
        # Pass the filtered match and its known ambiguity metadata to the graph.
        initial_state: AgentState = {
            "mention_id": str(record["mention_id"]),
            "jax_id": record["jax_id"],
            "s3_uri": record["s3_uri"],
            "pmcid": record.get("pmcid") or "",
            "publication_title": record.get("publication_title") or "",
            "author": record.get("author") or "",
            "publication_name": record.get("publication_name") or "",
            "publication_date": record.get("publication_date") or "",
            "matched_text": record.get("matched_text") or "",
            "matched_alias": record.get("matched_alias") or "",
            "context_window": record.get("context_window") or "",
            "normalized_term": record.get("normalized_term") or "",
            "n_candidates_jax_set": record.get("catalog_jax_candidate_count", record.get("n_candidates_jax_set")),
            "n_candidates_all_mgi": record.get("catalog_mgi_candidate_count", record.get("n_candidates_all_mgi")),
            "is_ambiguous_in_jax_set": record.get("is_ambiguous_in_jax_set"),
            "ambiguity_metadata_quality": record.get("ambiguity_metadata_quality") or "UNKNOWN",
            "normalized_alias": record.get("normalized_alias") or "",
            "candidate_identifiers": record.get("catalog_candidate_identifiers") or record.get("candidate_identifiers") or [],
            "alias_total_occurrences": record.get("alias_total_occurrences"),
            "alias_distinct_articles": record.get("alias_distinct_articles"),
            "evaluation": None,
            "routing_action": "",
            "error": "",
        }
        started = time.perf_counter()
        try:
            # compiled_graph.invoke is synchronous, so asyncio.to_thread keeps the event loop free.
            final_state = await asyncio.to_thread(compiled_graph.invoke, initial_state)
            latency_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "LLM_COMPLETE | mention_id=%s | jax_id=%s | action=%s | latency_ms=%.2f",
                record["mention_id"], record["jax_id"], final_state.get("routing_action"), latency_ms,
            )
            return {"mention_id": str(record["mention_id"]), "success": True,
                    "final_state": final_state, "latency_ms": latency_ms}
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.exception("LLM_WORKER_FAILURE | mention_id=%s", record["mention_id"])
            return {"mention_id": str(record["mention_id"]), "success": False,
                    "error": f"{type(exc).__name__}: {exc}", "latency_ms": latency_ms}


# ==========================================
# 4. PERSIST CLASSIFICATION RESULTS
# ==========================================
def write_batch_results_to_db(results: list[dict[str, Any]], claim_token: str) -> None:
    """Persist evaluations and clear only claims owned by this batch token."""
    update_sql = """
        UPDATE publication_mentions
        SET is_valid_jax_mention = %s,
            product_status = %s,
            confidence_score = %s,
            ai_justification = %s,
            pipeline_status = %s,
            error_log = %s,
            extracted_at = CURRENT_TIMESTAMP,
            llm_claimed_at = NULL,
            llm_claim_token = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE mention_id = %s AND llm_claim_token = %s
    """
    status_map = {
        "UPSERT_TO_POSTGRES": "PROCESSED",
        "MARK_FOR_HUMAN_REVIEW": "HUMAN_REVIEW",
        "SKIP_OR_DISCARD": "FALSE_POSITIVE",
        "LOG_ERROR": "FAILED",
    }
    started = time.perf_counter()
    try:
        with psycopg.connect(DB_CONN_STRING) as conn:
            with conn.cursor() as cur:
                for result in results:
                    state = result.get("final_state") or {}
                    evaluation = state.get("evaluation")
                    if isinstance(evaluation, dict):
                        evaluation = JAXProductEvaluation.model_validate(evaluation)

                    if result["success"]:
                        action = state.get("routing_action", "LOG_ERROR")
                        # Do not silently treat an unknown action as successful processing.
                        db_status = status_map.get(action, "FAILED")
                        if action not in status_map:
                            error_text = f"Unexpected routing action: {action!r}"
                        else:
                            error_text = state.get("error") or None
                        values = (
                            evaluation.is_valid_jax_mention if evaluation else False,
                            evaluation.product_status if evaluation else None,
                            evaluation.confidence_score if evaluation else None,
                            evaluation.justification if evaluation else None,
                            db_status, error_text,
                            result["mention_id"], claim_token,
                        )
                    else:
                        values = (False, None, None, None, "FAILED", result.get("error"),
                                  result["mention_id"], claim_token)
                    cur.execute(update_sql, values)
                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"Expected to update one claimed mention {result['mention_id']}; "
                            f"updated {cur.rowcount}. Claim token may have expired or changed."
                        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("DB_WRITE_SUCCESS | batch_size=%d | latency_ms=%.2f", len(results), elapsed_ms)
    except Exception:
        logger.exception("ABEND | DB_WRITE_FAILURE | batch_size=%d", len(results))
        # Re-raise so the coordinator stops rather than reporting an unpersisted batch as done.
        raise


# ==========================================
# 5. MAIN PIPELINE LOOP
# ==========================================
async def orchestrate_pipeline_loop() -> None:
    """Repeatedly claim batches, evaluate concurrently, and persist the results."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    total_processed = 0
    started = time.perf_counter()
    logger.info("Starting OpenAI/LangGraph evaluation loop | batch_size=%d | concurrency=%d",
                BATCH_SIZE, MAX_CONCURRENCY)

    while True:
        # Claim rows in a short transaction; do not hold database locks during API calls.
        claim_token, records = await asyncio.to_thread(claim_pending_batch, BATCH_SIZE)
        if not records:
            duration = time.perf_counter() - started
            throughput = total_processed / duration if duration else 0.0
            logger.info("PIPELINE_COMPLETE | processed=%d | seconds=%.2f | rows_per_second=%.2f",
                        total_processed, duration, throughput)
            return

        results = await asyncio.gather(*(process_single_mention(semaphore, row) for row in records))
        # A failed persistence operation is fatal for this loop; don't count this batch as done.
        await asyncio.to_thread(write_batch_results_to_db, results, claim_token)
        total_processed += len(results)
        elapsed = time.perf_counter() - started
        logger.info("THROUGHPUT | batch=%d | cumulative=%d | rows_per_second=%.2f",
                    len(results), total_processed, total_processed / elapsed if elapsed else 0.0)


if __name__ == "__main__":
    # Entry point used when this file is run directly from the project environment.
    asyncio.run(orchestrate_pipeline_loop())
