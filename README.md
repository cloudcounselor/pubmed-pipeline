# "PubMed" Product Citation Discovery Pipeline

**Python · An organization's Lexicon of nomenclature · PubMed Central Open Access repository · AWS S3 · PostgreSQL · LangGraph · OpenAI GPT · Human Review**

## Executive synopsis

A medical researcher writes *C57BL/6J* in the methods section of a scientific paper. Somewhere else, a product catalog lists the same genetic strain using a different formal identifier. Are these "mentions" referring to the same animal model at the center of a revolutionary discovery in cancer treatment? Did the researcher actually use the genetic strain, or merely mention it? And how could anyone answer those questions across tens of thousands of products and a vast, growing body of scientific literature?

That is the problem this project tackles. It combines fast, predictable software with GPT-based interpretation and a human checkpoint, turning scattered references in scientific papers into traceable **candidate product citations**—and, when the evidence supports it, corroborated findings about product use.

The interesting engineering decision is **where *not* to use AI**. Most of the workload is ordinary data engineering. Only the harder questions need language-model reasoning, and a comparatively small number of "hard calls" require a person.

## Background

*Building an AI-powered solution to uncover true research impact*
    The Use Case:  A non-profit research powerhouse with global reach needed a more reliable way to cope with inconsistent nomenclature references in order to identify scientific publications citing use of its products.
    A cohesive, scalable, AI-driven approach was developed in 2023 to automatically discover, attribute, and validate these citations while minimizing manual literature review (typically a Herculean task).
    The solution centered on an AWS-native, agentic AI pipeline combining a controlled lexicon of identifiers, structured publication data, deterministic text matching, semantic search, Amazon Bedrock-based contextual analysis, OpenSearch indexing, and agentic orchestration.  An AI agent was given instructions to dynamically investigate ambiguous references, select additional search strategies, interpret semantic context, assess evidence, resolve deadlocks, and assign confidence scores.  An explicit human-review gate was employed to resolve the ambiguous edge cases specifically flagged by the AI.  The solution had been utilized with AWS machine learning services that are now deprecated (since August 2026).  The original solution is now fully rebuilt and operational using Python deterministic scripts, PostgreSQL, LangGraph agentic orchestration, and OpenAI GPT.

 The current project develops that capability for The Jackson Laboratory (JAX) and the mouse-strain identifiers and aliases that describe its living products. The engineering solution discovers candidate mentions in the PubMed Central (PMC) Open Access corpus, associates them with publication metadata and surrounding text, and evaluates whether a mention qualifies as a specific, catalog-product association.

The design deliberately separates three kinds of work. 
**1) Deterministic Python logic** performs low-cost, repeatable discovery, string matching, metadata extraction, alias normalization, candidate enumeration, and database operations. 
**2) Agentic AI**, orchestrated by **LangGraph** and using **OpenAI GPT**, interprets the scientific context of selected candidates where literal matching alone is insufficient. 
**3) A human-review gate** handles ambiguous or insufficiently-supported cases; the system should not silently turn uncertain model judgments into authoritative product citations.

PostgreSQL is the operational system of record; timestamped JSONL files preserve scanner output as a supplementary export. A deterministic alias catalog records known alias-to-JAX relationships and corpus-level frequency measures, allowing the AI stage to receive evidence about ambiguity before AI reasoning commences. This architecture can be operated repeatedly as an **ongoing literature-surveillance process** across thousands of biomaterials catalog items, subject to source coverage, API limits, catalog maintenance, and operational scheduling. 

Here is the logical workflow:

1. **Start with a catalog.** A source such as `Strains.csv` supplies JAX product identifiers and the names or aliases that authors might use.
2. **Find candidate publications.** Python uses NCBI E-utilities to plan searches and discover PMC records, then retrieves applicable open-access article content from the public PMC dataset on AWS S3.
3. **Capture the evidence.** The scanner identifies matching text, extracts publication metadata, and saves up to **500 characters before and 500 characters after** each match. It does not send whole articles to GPT by default.
4. **Ask what the name could mean.** The deterministic alias catalog reports which JAX products share a normalized alias, what identifier evidence is available, and how often that alias appears in the scanned corpus.
5. **Evaluate context selectively.** Eligible records enter a PostgreSQL queue. A coordinator invokes a LangGraph workflow backed by OpenAI GPT, supplying the snippet and candidate evidence for a structured interpretation.
6. **Keep uncertainty visible.** A sufficiently supported result can be persisted; ambiguous or weakly supported cases are routed to `HUMAN_REVIEW`. Evaluation failures are errors, not negative scientific findings.
7. **Build a useful record.** The outcome remains connected to the publication, the matched language, the proposed catalog product, the model's explanation, and any human disposition.

## Architecture at a glance

text inputs
Catalog identifiers and aliases (e.g., Strains.csv)


[1] s3 chunking script for Deterministic discovery and PubMed Central article retrieval
    E-utilities query planning / partitioning; PMC OA article XML


[2] Deterministic matching and evidence extraction
    alias matches; publication metadata; +/- 500 characters of context

[3] PostgreSQL: publication_mentions + strain_alias_catalog
    durable evidence, candidate counts, work status, audit structures
                |
                v
[4] Coordinator + LangGraph + OpenAI GPT
    claim eligible rows; assess context; structured decision and rationale
    A supported decision becomes a persisted result.     
    Ambiguous or weak evidence is routed to the HUMAN_REVIEW queue for both reviewer disposition + audit.


## Technical deep dive

### 1. Product and publication discovery — deterministic

1. Load the configured JAX product IDs and their aliases from the catalog input (historically `Strains.csv`). A matched alias is a **candidate association**, not proof that the paper used that product.
2. Use NCBI E-utilities (ESearch/EFetch) to identify candidate PMC publications, including partitioning of large search result sets so individual ESearch result windows remain within the workflow's 10,000-result constraint. Retrieve applicable open-access article content from the public `s3://pmc-oa-opendata` source rather than assuming arbitrary S3 object paths.
3. Parse article XML to extract available publication title, first-author summary, journal, publication date, PMCID/article version, and source URI. Metadata completeness depends on the source article.
4. Apply deterministic alias matching to article text. For each hit, preserve the matched text and alias and extract **up to 500 characters before and 500 after the match** (characters, not words). These snippets control downstream token volume and retain nearby scientific evidence.
5. Emit a timestamped JSONL export and persist mentions in PostgreSQL. Repeated scans should rely on the implemented fingerprints/upsert constraints and should be verified for idempotent behavior with the live schema.

### 2. Alias intelligence — deterministic

1. Normalize alias strings consistently (lowercase, with non-alphanumeric characters removed) for grouping and lookup, while retaining the original spellings.
2. Build `public.strain_alias_catalog`, keyed by `normalized_alias`. Store `original_aliases`, `jax_candidate_count`, nullable `mgi_candidate_count`, `candidate_identifiers`, `total_occurrences`, and `distinct_articles`.
3. Count distinct JAX candidates associated with each normalized alias. **MGI counts are available only when explicit MGI identifiers are present in the configured catalog input**; a missing count means *unknown*, not zero or a completed search of the entire MGI registry.
4. Refresh corpus-frequency statistics from the mentions stored in PostgreSQL. These are counts over the **scanned corpus**, not all published biomedical literature. The occurrence measure deduplicates identical article/match/context combinations; it is not an exact positional count when repeated occurrences have identical contexts.
5. Pass candidate and frequency evidence into the AI stage. Alias popularity and uniqueness are useful signals, but neither independently proves product usage.

### 3. Persistence and workflow state — deterministic

1. `public.publication_mentions` stores the extracted evidence and operational fields; PostgreSQL is the authoritative store. JSONL is a supplementary record of scanner output, not the queue's source of truth.
2. `postgres_schema.py` creates and evolves schema objects, including the explicit `context_window` migration, nullable ambiguity indicator, metadata-quality fields, alias catalog, and review/audit structures.
3. The coordinator claims pending records in batches using PostgreSQL row locking (`FOR UPDATE SKIP LOCKED`) to reduce duplicate claims among concurrent workers. Claiming, processing, and final-state persistence must be checked together in integration tests.
4. Distinguish **not evaluated**, **evaluated negative**, **evaluation error**, and **requires review**. Missing ambiguity evidence must not be silently converted to a confident `False`.

### 4. Scientific interpretation — probabilistic / agentic AI

1. `pipeline_coordinator.py` retrieves eligible mentions, manages concurrency, and invokes the LangGraph workflow in `langgraph_orchestrator.py`.
2. LangGraph coordinates stateful evaluation and routing. The language model is **OpenAI GPT** (configured through the OpenAI API), not Claude. The graph receives the publication snippet, alias, proposed JAX ID, available candidate identifiers/counts, and supporting metadata.
3. The model interprets contextual clues—such as provenance, experimental use, alternative strain meanings, and contradictory evidence—and returns a structured assessment with confidence and explanation. This is **probabilistic judgment**, not an authoritative lookup.
4. Deterministic rules should continue to control eligibility, missing-data handling, routing constraints, and persistence. In particular, missing context or non-unique/unknown candidate evidence should favor review rather than unsupported acceptance.
5. Keep the model's rationale and decision traceable to the cited passage. GPT output does not establish that an article used a product unless the evidence and applicable review policy support that conclusion.

### 5. Human review — explicit quality gate

1. Route uncertain, ambiguous, or insufficiently evidenced evaluations to `HUMAN_REVIEW` rather than automatically accepting them.
2. A reviewer should see the publication citation, proposed JAX ID, exact alias, context window, candidate alternatives, AI recommendation, confidence, and rationale.
3. The review process should record the reviewer disposition and an audit history (`human_review_audit_log`), preserving the distinction between AI suggestion and human confirmation.
4. A Streamlit review dashboard is part of the intended local workflow. Its precise availability, approval/rejection behavior, and container startup depend on the deployed UI and Docker configuration; do not assume a one-command demo or verified dashboard functionality from these README files alone.
5. Human decisions can inform future evaluation rules and quality testing, but no automatic retraining or feedback-learning mechanism is claimed here.

### 6. Quality assurance, scale, and operating model

1. Run `validate_pipeline_outputs.py` against a scanner JSONL file and the live PostgreSQL schema. It checks expected fields, context-window constraints, and database/catalog presence; it does **not** establish scientific classification accuracy.
2. Start AI testing with a small, deliberately selected batch covering unique aliases, multi-candidate aliases, unavailable MGI metadata, weak context, and prior errors. **Batch size alone is not a run-wide cap** if the coordinator loops until the queue is empty; enforce a separate maximum-records limit before cost-controlled tests.
3. Manually label a reference sample and measure precision, recall, false positives, review referral rate, and error rate. Review failures separately from valid negative findings.
4. For ongoing operation, schedule incremental publication discovery, refresh alias/catalog inputs, deduplicate previously seen evidence, process bounded AI queues, and staff the human-review checkpoint. Scheduling, alerting, and operational SLAs are **deployment extensions**, not demonstrated features of the current code.
5. The same architecture can be generalized beyond mouse strains to **thousands of biomaterials products**, provided each product family has maintained identifiers/aliases, appropriate disambiguation rules, and domain-specific human review.

## Principal modules and data assets

| Component | Responsibility |
|---|---|
| `s3_chunk_scanner_gpt.py` | Catalog loading, PMC discovery/retrieval, deterministic matching, context and metadata extraction, JSONL/DB writes, alias catalog updates |
| `postgres_schema.py` | PostgreSQL tables, migrations, indexes, workflow/review audit structures |
| `langgraph_orchestrator.py` | GPT-backed structured assessment, state graph, routing logic |
| `pipeline_coordinator.py` | Batch claims, concurrent evaluation, state persistence and execution logging |
| `validate_pipeline_outputs.py` | JSONL, live schema, context-window and catalog checks |
| `public.publication_mentions` | Authoritative mention evidence and evaluation state |
| `public.strain_alias_catalog` | Deterministic alias relationships and scanned-corpus statistics |
| `human_review_audit_log` | Human-review history, as supported by the schema and review workflow |
| Streamlit / Docker Compose configuration | Local UI and service deployment; verify actual filenames and running services in the checked-out repository |

## Practical local operation

The project has been exercised from Windows PowerShell with PostgreSQL running in Docker Desktop. Exact service names, credentials, and optional UI services depend on the local Compose configuration.

```powershell
# Start the database services described by your actual compose file.
docker compose up -d

# Apply database schema/migrations (requires PMC_DB_CONN_STRING).
python postgres_schema.py

# Validate a known scanner export and the live database.
python validate_pipeline_outputs.py --jsonl "output\YOUR_TIMESTAMPED_EXPORT.jsonl"

# Inspect queue counts before invoking paid model calls.
docker exec -it pubmed_postgres psql -U postgres -d pubmed_analytics -c "SELECT pipeline_status, COUNT(*) FROM public.publication_mentions GROUP BY pipeline_status ORDER BY pipeline_status;"

# Process eligible rows only after verifying API configuration and limits.
python pipeline_coordinator.py
```

The Python processes need `PMC_DB_CONN_STRING`; GPT evaluation also needs `OPENAI_API_KEY`. A project-specific `.env` file is effective only when explicitly loaded by the script or shell. Keep credentials out of source control. `processed=0` from the coordinator means no rows were processed during that run—not that model classification was tested. Before a paid run, verify the eligible queue and a **total-run** cap; concurrency and per-batch size are not spending limits.

## Boundaries and interpretation

- **Citation candidate vs. confirmed use:** A matching string may be incidental, historical, comparative, or ambiguous. Report these categories separately.
- **Provenance:** Preserve the PMCID, article metadata, source URI, matched alias, context, candidate set, AI rationale, and review disposition where available.
- **Intended outcome:** a repeatable, auditable pipeline that turns large-scale scientific literature into product-level citation intelligence while reserving expensive reasoning for cases that need it and human judgment for cases that remain uncertain.
