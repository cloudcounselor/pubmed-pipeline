
"""LangGraph workflow for probabilistic evaluation of deterministic PMC matches.

The upstream scanner remains responsible for searching PMC and extracting exact
matches. This module sends only one already-filtered mention package to OpenAI,
validates the response with Pydantic, and applies deterministic routing rules.
Database writes are intentionally owned by pipeline_coordinator.py.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the environment file in the same directory
# as this Python script.
BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / "project-pubmed-pipeline.env"

# Load environment variables from the custom file.
if not ENV_FILE.is_file():
    raise FileNotFoundError(
        f"Environment file not found: {ENV_FILE}"
    )

load_dotenv(dotenv_path=ENV_FILE)

# Verify that the OpenAI API key is available.
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not loaded from .env file."
    )

print("OpenAI API key successfully loaded.")

# ==========================================
# 1. STRUCTURED OUTPUT CONTRACT
# ==========================================
class JAXProductEvaluation(BaseModel):
    """Validated semantic interpretation returned by the OpenAI model."""

    is_valid_jax_mention: bool = Field(
        description=(
            "True only when the supplied context supports a genuine Jackson Laboratory "
            "(JAX) mouse strain/product mention; false for unrelated identifiers or matches."
        )
    )
    product_status: Literal["ACTIVE", "HISTORICAL", "FALSE_POSITIVE", "AMBIGUOUS"] = Field(
        description=(
            "ACTIVE for current use, HISTORICAL for explicit legacy use, FALSE_POSITIVE "
            "for an unrelated match, or AMBIGUOUS when the evidence is insufficient or conflicting."
        )
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence from 0.0 to 1.0; this never overrides deterministic ambiguity rules.",
    )
    justification: str = Field(
        min_length=1,
        description="Concise evidence-based explanation grounded only in the supplied text and metadata.",
    )


# ==========================================
# 2. LANGGRAPH STATE INTERFACE
# ==========================================
class AgentState(TypedDict, total=False):
    """Fields passed between the extraction and deterministic routing nodes."""

    mention_id: str
    jax_id: str
    s3_uri: str
    pmcid: str
    publication_title: str
    author: str
    publication_name: str
    publication_date: str
    matched_text: str
    matched_alias: str
    context_window: str
    normalized_term: str
    n_candidates_jax_set: int | None
    n_candidates_all_mgi: int | None
    is_ambiguous_in_jax_set: bool | None
    ambiguity_metadata_quality: str
    normalized_alias: str
    candidate_identifiers: list[str]
    alias_total_occurrences: int | None
    alias_distinct_articles: int | None
    evaluation: JAXProductEvaluation | None
    routing_action: str
    error: str


# ==========================================
# 3. OPENAI EXTRACTION NODE
# ==========================================
def extraction_node(state: AgentState) -> dict[str, Any]:
    """Ask OpenAI to interpret one deterministic match using a strict Pydantic schema."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"evaluation": None, "error": "OPENAI_API_KEY is not set."}

    # A missing context window is an evidence gap, not a reason to guess. Return a
    # structured ambiguous result so the deterministic router sends it to human review.
    if not state.get("context_window", "").strip():
        return {
            "evaluation": JAXProductEvaluation(
                is_valid_jax_mention=True,
                product_status="AMBIGUOUS",
                confidence_score=0.0,
                justification="No context_window is available in PostgreSQL; the mention cannot be evaluated safely.",
            ),
            "error": "",
        }

    # The model is configurable so a tested model can be selected without editing code.
    # Temperature zero reduces variability; Pydantic validates the structured response.
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
        api_key=api_key,
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    )
    structured_llm = llm.with_structured_output(JAXProductEvaluation)

    # Supply the model with evidence and ambiguity metadata, not a search task.
    # Explicit IDs are stronger identifiers than common strain names or synonyms.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a biomedical evidence-classification component in a data pipeline.
You do NOT search the web, infer unseen facts, or change the supplied JAX target ID.
Evaluate only the candidate text and metadata supplied in this request.

Classify the mention as:
- ACTIVE: context supports current use of the target JAX mouse strain/product.
- HISTORICAL: context explicitly describes prior/legacy use of the target product.
- FALSE_POSITIVE: the matched string is unrelated to the target JAX strain/product.
- AMBIGUOUS: context is insufficient/conflicting, or the name alone cannot distinguish candidates.

An explicit catalog identifier such as JAX:000664 or MGI:3028467 is unambiguous as an identifier.
A common name/synonym with multiple JAX candidates is ambiguous; do not guess which candidate
is intended. Treat verified catalog candidate counts as deterministic metadata. A missing count means UNKNOWN,
not zero and not proof of ambiguity. Alias frequency in publications is supporting context only:
a frequent alias is not automatically ambiguous, and a rare alias is not automatically unique.
Do not infer ACTIVE merely because a paper mentions a strain name. Explain the evidence briefly.
Return only the requested structured fields."""),
        ("human", """Target JAX ID: {jax_id}
Matched text: {matched_text}
Matched alias: {matched_alias}
Normalized term: {normalized_term}
JAX candidates for normalized term: {n_candidates_jax_set}
MGI candidates for normalized term: {n_candidates_all_mgi}
Ambiguous within JAX set (true/false/unknown): {is_ambiguous_in_jax_set}
Catalog metadata quality: {ambiguity_metadata_quality}
Candidate identifiers: {candidate_identifiers}
Total distinct alias occurrences in corpus: {alias_total_occurrences}
Distinct articles containing alias: {alias_distinct_articles}
Article title: {publication_title}
First author: {author}
Publication/journal: {publication_name}
Publication date: {publication_date}
Source S3 URI: {s3_uri}

Candidate context (the only text to interpret):
{context_window}"""),
    ])

    chain = prompt | structured_llm
    try:
        result = chain.invoke({
            "jax_id": state.get("jax_id", ""),
            "matched_text": state.get("matched_text", ""),
            "matched_alias": state.get("matched_alias", ""),
            "normalized_term": state.get("normalized_term", ""),
            "n_candidates_jax_set": state.get("n_candidates_jax_set"),
            "n_candidates_all_mgi": state.get("n_candidates_all_mgi"),
            "is_ambiguous_in_jax_set": state.get("is_ambiguous_in_jax_set"),
            "ambiguity_metadata_quality": state.get("ambiguity_metadata_quality", "UNKNOWN"),
            "candidate_identifiers": state.get("candidate_identifiers", []),
            "alias_total_occurrences": state.get("alias_total_occurrences"),
            "alias_distinct_articles": state.get("alias_distinct_articles"),
            "publication_title": state.get("publication_title", ""),
            "author": state.get("author", ""),
            "publication_name": state.get("publication_name", ""),
            "publication_date": state.get("publication_date", ""),
            "s3_uri": state.get("s3_uri", ""),
            "context_window": state.get("context_window", ""),
        })
        # with_structured_output normally returns the Pydantic model. Re-validate defensively.
        if not isinstance(result, JAXProductEvaluation):
            result = JAXProductEvaluation.model_validate(result)
        return {"evaluation": result, "error": ""}
    except Exception as exc:
        # Return the failure in graph state; coordinator persists FAILED status and error_log.
        return {"evaluation": None, "error": f"OpenAI evaluation failed: {type(exc).__name__}: {exc}"}


# ==========================================
# 4. DETERMINISTIC ROUTING AND SAFETY RULES
# ==========================================
_EXPLICIT_ID_PATTERN = re.compile(r"\b(?:JAX:\d{6}|MGI:\d{7})\b", re.IGNORECASE)


def routing_and_logging_node(state: AgentState) -> dict[str, Any]:
    """Apply business rules after model evaluation; rules outrank model confidence."""
    if state.get("error"):
        return {"routing_action": "LOG_ERROR"}

    evaluation = state.get("evaluation")
    if evaluation is None:
        return {"routing_action": "LOG_ERROR", "error": "No validated evaluation was returned."}

    # Hard gate: ambiguous common terms must go to a human even if the LLM says ACTIVE.
    # Explicit catalog IDs are not treated as ambiguous merely because a related name is.
    # Only the actual matched token/alias establishes that the search target was an ID.
    # An unrelated ID elsewhere in the context must not bypass the ambiguity gate.
    matched_identifier = " ".join((state.get("matched_text", ""), state.get("matched_alias", "")))
    has_explicit_id = bool(_EXPLICIT_ID_PATTERN.search(matched_identifier))
    candidate_count = state.get("n_candidates_jax_set")
    metadata_says_ambiguous = state.get("is_ambiguous_in_jax_set") is True
    # Fail closed when candidate-count metadata is absent or malformed: without a
    # confirmed single-candidate term, the pipeline cannot safely disambiguate a name.
    # If a common alias lacks a verified count, it is not proven ambiguous; it is
    # simply not safe to auto-approve. Keep that distinction in the metadata and route.
    candidate_count_is_not_unique_or_unknown = candidate_count != 1
    if not has_explicit_id and (metadata_says_ambiguous or candidate_count_is_not_unique_or_unknown):
        return {"routing_action": "MARK_FOR_HUMAN_REVIEW"}

    if not evaluation.is_valid_jax_mention or evaluation.product_status == "FALSE_POSITIVE":
        return {"routing_action": "SKIP_OR_DISCARD"}
    if evaluation.product_status in ("ACTIVE", "HISTORICAL"):
        return {"routing_action": "UPSERT_TO_POSTGRES"}
    # The remaining status is AMBIGUOUS, which is always sent to the review queue.
    return {"routing_action": "MARK_FOR_HUMAN_REVIEW"}


# ==========================================
# 5. COMPILE THE LANGGRAPH WORKFLOW
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("extractor", extraction_node)
workflow.add_node("router", routing_and_logging_node)
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "router")
workflow.add_edge("router", END)

# External modules import this compiled graph and invoke it once per mention.
compiled_graph = workflow.compile()


# ==========================================
# 6. OPTIONAL MANUAL SMOKE TEST
# ==========================================
if __name__ == "__main__":
    # This is a local smoke-test example. It makes a real OpenAI API call if invoked.
    sample_state: AgentState = {
        "jax_id": "000664",
        "s3_uri": "s3://pmc-oa-opendata/example.xml",
        "matched_text": "C57BL/6J",
        "matched_alias": "C57BL/6J",
        "normalized_term": "C57BL/6J",
        "n_candidates_jax_set": 1,
        "n_candidates_all_mgi": 4,
        "is_ambiguous_in_jax_set": False,
        "ambiguity_metadata_quality": "CATALOG_COMPLETE_MGI_AVAILABLE",
        "normalized_alias": "c57bl6j",
        "candidate_identifiers": ["JAX:000664", "MGI:3028467"],
        "alias_total_occurrences": 12,
        "alias_distinct_articles": 8,
        "context_window": "Mice of the C57BL/6J strain were obtained from The Jackson Laboratory.",
        "evaluation": None,
        "routing_action": "",
        "error": "",
    }
    print("Running LangGraph OpenAI evaluation smoke test...")
    print(compiled_graph.invoke(sample_state))
