# PubMed Citation Discovery Pipeline

A solution, both deterministic and AI agentic, that scours the scientific literature for key terms and Identifiers to be used for trend analysis in the biomedical research sciences.

# JAX Mouse Strain Reference

A name-resolution reference for mouse strains that carry a Jackson Laboratory (JAX) stock number, built from Mouse Genome Informatics (MGI) data. 

# Constituents

Python · PubMed Central Open Access · AWS S3 · PostgreSQL · LangGraph · OpenAI GPT · Human Review

# Executive synopsis

Biomedical product suppliers need a reliable means to learn when, where, and how their catalog items are cited or used in published research. This project develops that capability for the mouse-strain identifiers and aliases of The Jackson Laboratory (JAX). The solution discovers candidate mentions in the PubMed Central (PMC) Open Access corpus, associates them with publication metadata and surrounding text, and evaluates whether a mention qualifies as a specific catalog-product association.

The design deliberately separates three kinds of work. Deterministic Python logic performs inexpensive, repeatable discovery, string matching, metadata extraction, alias normalization, candidate enumeration, and database operations. Agentic AI, orchestrated by LangGraph and using OpenAI GPT, interprets the scientific context of selected candidates where literal matching alone is insufficient. A human-review gate handles ambiguous or insufficiently supported cases; the system should not silently turn uncertain model judgments into authoritative product citations.

PostgreSQL is the operational system of record; timestamped JSONL files preserve scanner output as a supplementary export. A deterministic alias catalog records known alias-to-JAX relationships and corpus-level frequency measures, allowing the AI stage to receive evidence about ambiguity before reasoning. This architecture can be operated repeatedly as an ongoing literature-surveillance process across thousands of biomaterials catalog items, subject to source coverage, API limits, catalog maintenance, and operational scheduling. It is a pipeline under iterative validation—not a claim that all publications have been scanned, that every mention is a confirmed use, or that unattended production scheduling is already deployed.

# Architecture at a glance

Catalog identifiers and aliases (e.g., Strains.csv)
                |
                v
[1] Deterministic discovery and PMC article retrieval
    E-utilities query planning / partitioning; PMC OA article XML
                |
                v
[2] Deterministic matching and evidence extraction
    alias matches; publication metadata; +/- 500 characters of context
                |
                +----> timestamped JSONL export
                |
                v
[3] PostgreSQL: publication_mentions + strain_alias_catalog
    durable evidence, candidate counts, work status, audit structures
                |
                v
[4] Coordinator + LangGraph + OpenAI GPT
    claim eligible rows; assess context; structured decision and rationale
                |
          +-----+------------------+
          |                        |
          v                        v
    supported decision      ambiguous / weak evidence
          |                        |
          v                        v
    persisted result         HUMAN_REVIEW queue
                                   |
                                   v
                            reviewer disposition + audit

# Technical deep dive

A. Product and publication discovery — deterministic

1. Load the configured JAX product IDs and their aliases from the catalog input (historically Strains.csv). A matched alias is a candidate association, not proof that the paper used that product.
2. Use NCBI E-utilities (ESearch/EFetch) to identify candidate PMC publications, including partitioning of large search result sets so individual ESearch result windows remain within the workflow's 10,000-result constraint. Retrieve applicable open-access article content from the public s3://pmc-oa-opendata source rather than assuming arbitrary S3 object paths.
3. Parse article XML to extract available publication title, first-author summary, journal, publication date, PMCID/article version, and source URI. Metadata completeness depends on the source article.
4. Apply deterministic alias matching to article text. For each hit, preserve the matched text and alias and extract up to 500 characters before and 500 after the match (characters, not words). These snippets control downstream token volume and retain nearby scientific evidence.
5. Emit a timestamped JSONL export and persist mentions in PostgreSQL. Repeated scans should rely on the implemented fingerprints/upsert constraints and should be verified for idempotent behavior with the live schema.

B. Alias intelligence — deterministic

1. Normalize alias strings consistently (lowercase, with non-alphanumeric characters removed) for grouping and lookup, while retaining the original spellings.
2. Build public.strain_alias_catalog, keyed by normalized_alias. Store original_aliases, jax_candidate_count, nullable mgi_candidate_count, candidate_identifiers, total_occurrences, and distinct_articles.
3. Count distinct JAX candidates associated with each normalized alias. MGI counts are available only when explicit MGI identifiers are present in the configured catalog input; a missing count means *unknown*, not zero or a completed search of the entire MGI registry.
4. Refresh corpus-frequency statistics from the mentions stored in PostgreSQL. These are counts over the scanned corpus, not all published biomedical literature. The occurrence measure deduplicates identical article/match/context combinations; it is not an exact positional count when repeated occurrences have identical contexts.
5. Pass candidate and frequency evidence into the AI stage. Alias popularity and uniqueness are useful signals, but neither independently proves product usage.

C. Persistence and workflow state — deterministic

1. public.publication_mentions stores the extracted evidence and operational fields; PostgreSQL is the authoritative store. JSONL is a supplementary record of scanner output, not the queue's source of truth.
2. postgres_schema.py creates and evolves schema objects, including the explicit context_window migration, nullable ambiguity indicator, metadata-quality fields, alias catalog, and review/audit structures.
3. The coordinator claims pending records in batches using PostgreSQL row locking (FOR UPDATE SKIP LOCKED) to reduce duplicate claims among concurrent workers. Claiming, processing, and final-state persistence must be checked together in integration tests.
4. Distinguish not evaluated, evaluated negative, evaluation error, and requires review. Missing ambiguity evidence must not be silently converted to a confident False.

D. Scientific interpretation — probabilistic / agentic AI

1. pipeline_coordinator.py retrieves eligible mentions, manages concurrency, and invokes the LangGraph workflow in langgraph_orchestrator.py.
2. LangGraph coordinates stateful evaluation and routing. The language model is OpenAI GPT (configured through the OpenAI API), not Claude. The graph receives the publication snippet, alias, proposed JAX ID, available candidate identifiers/counts, and supporting metadata.
3. The model interprets contextual clues—such as provenance, experimental use, alternative strain meanings, and contradictory evidence—and returns a structured assessment with confidence and explanation. This is probabilistic judgment, not an authoritative lookup.
4. Deterministic rules should continue to control eligibility, missing-data handling, routing constraints, and persistence. In particular, missing context or non-unique/unknown candidate evidence should favor review rather than unsupported acceptance.
5. Keep the model's rationale and decision traceable to the cited passage. GPT output does not establish that an article used a product unless the evidence and applicable review policy support that conclusion.

E. Human review — explicit quality gate

1. Route uncertain, ambiguous, or insufficiently evidenced evaluations to HUMAN_REVIEW rather than automatically accepting them.
2. A reviewer should see the publication citation, proposed JAX ID, exact alias, context window, candidate alternatives, AI recommendation, confidence, and rationale.
3. The review process should record the reviewer disposition and an audit history (human_review_audit_log), preserving the distinction between AI suggestion and human confirmation.
4. A Streamlit review dashboard is part of the intended local workflow. Its precise availability, approval/rejection behavior, and container startup depend on the deployed UI and Docker configuration; do not assume a one-command demo or verified dashboard functionality from these README files alone.
5. Human decisions can inform future evaluation rules and quality testing, but no automatic retraining or feedback-learning mechanism is claimed here.

F. Quality assurance, scale, and operating model

1. Run validate_pipeline_outputs.py against a scanner JSONL file and the live PostgreSQL schema. It checks expected fields, context-window constraints, and database/catalog presence; it does not establish scientific classification accuracy.
2. Start AI testing with a small, deliberately selected batch covering unique aliases, multi-candidate aliases, unavailable MGI metadata, weak context, and prior errors. Batch size alone is not a run-wide cap if the coordinator loops until the queue is empty; enforce a separate maximum-records limit before cost-controlled tests.
3. Manually label a reference sample and measure precision, recall, false positives, review referral rate, and error rate. Review failures separately from valid negative findings.
4. For ongoing operation, schedule incremental publication discovery, refresh alias/catalog inputs, deduplicate previously seen evidence, process bounded AI queues, and staff the human-review checkpoint. Scheduling, alerting, and operational SLAs are deployment extensions, not demonstrated features of the current code.
5. The same architecture can be generalized beyond mouse strains to thousands of biomaterials products, provided each product family has maintained identifiers/aliases, appropriate disambiguation rules, and domain-specific human review.

# Principal modules and data assets

| Component | Responsibility |
|---|---|
| s3_chunk_scanner_gpt.py | Catalog loading, PMC discovery/retrieval, deterministic matching, context and metadata extraction, JSONL/DB writes, alias catalog updates |
| postgres_schema.py | PostgreSQL tables, migrations, indexes, workflow/review audit structures |
| langgraph_orchestrator.py | GPT-backed structured assessment, state graph, routing logic |
| pipeline_coordinator.py | Batch claims, concurrent evaluation, state persistence and execution logging |
| validate_pipeline_outputs.py | JSONL, live schema, context-window and catalog checks |
| public.publication_mentions | Authoritative mention evidence and evaluation state |
| public.strain_alias_catalog | Deterministic alias relationships and scanned-corpus statistics |
| human_review_audit_log | Human-review history, as supported by the schema and review workflow |
| Streamlit / Docker Compose configuration | Local UI and service deployment; verify actual filenames and running services in the checked-out repository |

# Practical local operation

The project has been exercised from Windows PowerShell with PostgreSQL running in Docker Desktop. Exact service names, credentials, and optional UI services depend on the local Compose configuration.

Start the database services described by your actual compose file.
docker compose up -d

Apply database schema/migrations (requires PMC_DB_CONN_STRING).
python postgres_schema.py

Validate a known scanner export and the live database.
python validate_pipeline_outputs.py --jsonl "output\YOUR_TIMESTAMPED_EXPORT.jsonl"

Inspect queue counts before invoking paid model calls.
docker exec -it pubmed_postgres psql -U postgres -d pubmed_analytics -c "SELECT pipeline_status, COUNT(*) FROM public.publication_mentions GROUP BY pipeline_status ORDER BY pipeline_status;"

Process eligible rows only after verifying API configuration and limits.
python pipeline_coordinator.py

The Python processes need PMC_DB_CONN_STRING; GPT evaluation also needs OPENAI_API_KEY. A project-specific .env file is effective only when explicitly loaded by the script or shell. Keep credentials out of source control. processed=0 from the coordinator means no rows were processed during that run—not that model classification was tested. Before a paid run, verify the eligible queue and a total-run cap; concurrency and per-batch size are not spending limits.

# Boundaries and interpretation

- Citation candidate vs. confirmed use: A matching string may be incidental, historical, comparative, or ambiguous. Report these categories separately.
- Coverage: PMC Open Access is not the entirety of PubMed or biomedical publishing; a missing hit does not establish non-use.
- Provenance: Preserve the PMCID, article metadata, source URI, matched alias, context, candidate set, AI rationale, and review disposition where available.
- Current maturity: The scanner, alias catalog, PostgreSQL validation, and orchestration components have been implemented and iteratively tested. Production-scale accuracy, fully unattended scheduling, and a complete dashboard approval workflow should be claimed only after separate verification.

# Intended outcome: a repeatable, auditable pipeline that turns large-scale scientific literature into product-level citation intelligence while reserving expensive reasoning for cases that need it and human judgment for cases that remain uncertain.
