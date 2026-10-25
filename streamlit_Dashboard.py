"""Streamlit human-review dashboard for ambiguous PubMed/JAX findings.

Launch from the project directory with:
    streamlit run Streamlit_Dashboard.py

Requirements: streamlit, psycopg[binary], PostgreSQL schema already created,
PMC_DB_CONN_STRING set in the environment, and human_review_handler.py in the
same directory. This UI delegates ALL writes to the transaction handler.
"""

import html
import re

import psycopg
from psycopg.rows import dict_row
import streamlit as st

from human_review_handler import (
    HumanReviewError,
    ReviewConflictError,
    get_db_conn_string,
    resolve_human_review_item,
)

# Streamlit requires page configuration before other st.* UI calls.
st.set_page_config(
    page_title="JAX ID Pipeline: Human Review Desk",
    layout="wide",
    initial_sidebar_state="expanded",
)


def highlight_keyword(text: str | None, keyword: str | None) -> str:
    """HTML-escape publication text, then highlight literal case-insensitive hits.

    Publication snippets are external/untrusted content. Escaping before adding
    our own <mark> tags prevents snippets from injecting arbitrary HTML or JS.
    A plain escaped snippet is shown when no keyword is available.
    """
    safe_text = html.escape(text or "")
    if not keyword:
        return safe_text
    safe_keyword = html.escape(keyword)
    if not safe_keyword:
        return safe_text
    return re.sub(
        re.escape(safe_keyword),
        lambda match: '<mark style="background-color:#fcf8e3; padding:2px; '
                      'font-weight:bold">' + match.group(0) + '</mark>',
        safe_text,
        flags=re.IGNORECASE,
    )


def fetch_pending_items() -> list[dict]:
    """Read only the unresolved review queue, least confident records first.

    The selected columns are those already referenced by the original UI;
    no new schema fields are assumed. Null confidence scores sort last.
    """
    query = """
        SELECT mention_id, jax_id, s3_uri, matched_text, context_window,
               confidence_score, ai_justification
          FROM publication_mentions
         WHERE pipeline_status = 'HUMAN_REVIEW'
         ORDER BY confidence_score ASC NULLS LAST, mention_id
    """
    with psycopg.connect(get_db_conn_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()


def process_triage_action(mention_id: str, decision: str, notes: str, reviewer: str) -> None:
    """Delegate a reviewer action to the shared atomic database handler.

    Success is shown only after the transaction commits. A conflict instructs
    the reviewer to refresh rather than overwriting another person's work.
    """
    try:
        resolve_human_review_item(mention_id, decision, reviewer, notes)
    except (ValueError, HumanReviewError, psycopg.Error) as exc:
        st.error(f"Review could not be saved: {exc}")
    else:
        st.session_state["review_notice"] = f"Saved review decision: {decision}."
        st.rerun()


# Display title and any successful action message preserved across reruns.
st.title("JAX Product Citations: Human Review Workspace")
st.caption("Inspect AI-flagged ambiguous matches before final classification.")
if notice := st.session_state.pop("review_notice", None):
    st.success(notice)

# Never display a database failure as an empty queue. These are materially
# different operational states and must be distinguishable to the reviewer.
try:
    pending_items = fetch_pending_items()
except (psycopg.Error, HumanReviewError) as exc:
    st.error(f"Unable to load the review queue: {exc}")
    st.stop()

# Sidebar is the queue navigator. Use mention_id as the radio value so two
# publications with the same JAX ID/confidence remain individually selectable.
with st.sidebar:
    st.header(f"Pending review ({len(pending_items)})")
    reviewer = st.text_input(
        "Reviewer identity (required)",
        placeholder="Your work username or reviewer ID",
        help="This value is written to the audit history. Use an attributable ID.",
    )
    if pending_items:
        by_id = {str(item["mention_id"]): item for item in pending_items}

        def label_for(mention_key: str) -> str:
            item = by_id[mention_key]
            score = item["confidence_score"]
            score_label = "N/A" if score is None else f"{float(score):.0%}"
            return f"JAX {item['jax_id']} | AI {score_label} | {mention_key}"

        selected_id = st.radio(
            "Choose a finding", list(by_id), format_func=label_for
        )
    else:
        selected_id = None
        st.info("No findings currently await human review.")

if selected_id is not None:
    # Two-column layout keeps the source evidence adjacent to the model's
    # reasoning so the reviewer can compare both before making a decision.
    current_record = by_id[selected_id]
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Publication context")
        formatted_snippet = highlight_keyword(
            current_record["context_window"], current_record["matched_text"]
        )
        st.markdown(
            '<div style="background-color:#f7f9fa; padding:18px; '
            'border-radius:8px; border-left:4px solid #2b5c8f; '
            f'line-height:1.6">{formatted_snippet}</div>',
            unsafe_allow_html=True,
        )
        st.caption("S3 source URI")
        st.code(current_record["s3_uri"] or "Not available")

    with right:
        st.subheader("AI evaluation")
        st.write("**JAX ID:**", str(current_record["jax_id"]))
        st.write("**Matched text:**", current_record["matched_text"] or "Not available")
        score = current_record["confidence_score"]
        st.write("**Original AI confidence:**", "N/A" if score is None else f"{float(score):.3f}")
        st.write("**AI justification:**")
        st.info(current_record["ai_justification"] or "No justification was recorded.")

    # Explicit actions correspond to the handler's allowlisted decision map.
    # Notes are required to preserve the rationale for every human override.
    st.divider()
    st.subheader("Human review decision")
    notes = st.text_area(
        "Verification evidence and rationale (required)",
        key=f"notes_{selected_id}",
        placeholder="Explain the evidence supporting your classification.",
    )
    can_submit = bool(reviewer.strip() and notes.strip())
    if not can_submit:
        st.caption("Enter a reviewer identity and review rationale to enable decisions.")

    active, historical, reject = st.columns(3)
    with active:
        if st.button("Approve: active product", use_container_width=True,
                     type="primary", disabled=not can_submit):
            process_triage_action(selected_id, "APPROVED_ACTIVE", notes, reviewer)
    with historical:
        if st.button("Approve: historical reference", use_container_width=True,
                     disabled=not can_submit):
            process_triage_action(selected_id, "APPROVED_HISTORICAL", notes, reviewer)
    with reject:
        if st.button("Reject: false positive", use_container_width=True,
                     disabled=not can_submit):
            process_triage_action(selected_id, "REJECTED", notes, reviewer)
