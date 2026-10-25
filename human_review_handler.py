"""Transactional human-review gate for the PubMed/JAX publication pipeline.

Integration contract (based on the supplied scripts):
  publication_mentions: mention_id, pipeline_status, product_status,
                        confidence_score
  human_review_audit_log: mention_id, reviewer_identity, previous_ai_status,
                          final_human_override_status, reviewer_notes

Only records already placed in HUMAN_REVIEW may be resolved. The AI confidence
score is retained as evidence; a human decision is not an AI confidence score.
Set PMC_DB_CONN_STRING in the environment before launching the application.
"""

import logging
import os
from typing import Optional

import psycopg

LOGGER = logging.getLogger(__name__)

# These are the only accepted decisions. Never silently accept an unknown
# decision: doing so could incorrectly clear a record from the review queue.
DECISION_TO_STATUS = {
    "APPROVED_ACTIVE": ("PROCESSED", "ACTIVE"),
    "APPROVED_HISTORICAL": ("PROCESSED", "HISTORICAL"),
    "REJECTED": ("FALSE_POSITIVE", "FALSE_POSITIVE"),
}


class HumanReviewError(RuntimeError):
    """Base exception for a failed review; transaction is rolled back."""


class ReviewConflictError(HumanReviewError):
    """The mention is missing or has already left the HUMAN_REVIEW queue."""


def get_db_conn_string() -> str:
    """Read the same database setting used by the pipeline; no embedded secrets."""
    conn_string = os.getenv("PMC_DB_CONN_STRING", "").strip()
    if not conn_string:
        raise HumanReviewError(
            "PMC_DB_CONN_STRING is not set. Load the pipeline .env file "
            "or set this environment variable before starting Streamlit."
        )
    return conn_string


def resolve_human_review_item(
    mention_id: str,
    decision: str,
    reviewer: str,
    notes: Optional[str] = None,
) -> bool:
    """Resolve exactly one queued mention and audit it in a single transaction.

    Args:
        mention_id: Existing primary key in publication_mentions.
        decision: APPROVED_ACTIVE, APPROVED_HISTORICAL, or REJECTED.
        reviewer: Nonempty reviewer identifier, supplied by the UI.
        notes: Optional evidence and rationale; never interpolated into SQL.

    Returns:
        True after both SQL statements commit successfully.

    Raises:
        ValueError: Invalid decision, ID, or reviewer.
        ReviewConflictError: The mention is not pending review anymore.
        psycopg.Error: Database operation fails (no partial commit).
    """
    # Validate all user inputs before opening a database transaction. The
    # mapping below defines the only permitted downstream classification.
    if decision not in DECISION_TO_STATUS:
        raise ValueError(f"Unsupported review decision: {decision!r}")
    if not str(mention_id).strip():
        raise ValueError("A mention_id is required.")
    if not reviewer or not reviewer.strip():
        raise ValueError("A reviewer identity is required for auditability.")

    pipeline_status, product_status = DECISION_TO_STATUS[decision]

    # UPDATE ... RETURNING supplies the *actual* previous queue status from
    # the record we updated. The WHERE clause implements optimistic queue
    # ownership: only one reviewer can transition HUMAN_REVIEW -> resolved.
    # PostgreSQL row locking ensures competing updates cannot both succeed.
    update_sql = """
        UPDATE publication_mentions
           SET pipeline_status = %s,
               product_status = %s
         WHERE mention_id = %s
           AND pipeline_status = 'HUMAN_REVIEW'
        RETURNING mention_id
    """

    # The audit entry uses the exact column names in the supplied scripts.
    # We preserve the AI confidence score in publication_mentions rather than
    # falsely replacing it with 1.0 after a human decision.
    audit_sql = """
        INSERT INTO human_review_audit_log
            (mention_id, reviewer_identity, previous_ai_status,
             final_human_override_status, reviewer_notes)
        VALUES (%s, %s, %s, %s, %s)
    """

    # psycopg's connection context automatically commits on successful exit
    # and rolls back if any statement fails. Thus a classification cannot be
    # committed without its corresponding audit record.
    try:
        with psycopg.connect(get_db_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(update_sql, (pipeline_status, product_status, mention_id))
                if cur.fetchone() is None:
                    raise ReviewConflictError(
                        "This finding is no longer pending human review "
                        "(or its mention_id does not exist). Refresh the queue."
                    )
                cur.execute(
                    audit_sql,
                    (mention_id, reviewer.strip(), "HUMAN_REVIEW", decision,
                     notes.strip() if notes else None),
                )
    except (psycopg.Error, HumanReviewError):
        LOGGER.exception("Human review transaction failed for mention_id=%s", mention_id)
        raise

    LOGGER.info("Review committed: mention_id=%s decision=%s", mention_id, decision)
    return True
