# pubmed-pipeline
A solution, both deterministic and AI agentic, that scours the scientific literature for key terms and Identifiers to be used for trend analysis in the biomedical research sciences.

# JAX Mouse Strain Reference

A name-resolution reference for mouse strains that carry a Jackson Laboratory (JAX) stock number,
built from Mouse Genome Informatics (MGI) data according to the plan in
*"AI IO pubmed-pipeline for JAX IDs.docx"* (Passes 1–4).

**Data as of 2026-10-08** (date of the MGI strain-summary export). Pipeline version 1.0.0.

---

## 1. Read this first: what this dataset is and is not

| | |
|---|---|
| **Is** | 17,509 MGI strain records that carry a `JAX:<6-digit>` stock ID, with MGI IDs, JAX IDs, derived RRIDs, official names, MGI synonyms, strain types/attributes, other repository IDs, and name-parsed background/gene/allele fields. Every row traces to named source files. |
| **Is not (yet)** | Verified against live JAX or IMSR web pages. `jax_strain_name`, `common_name`, `description`, availability and `active_jax_stock` are **empty / "unknown"** because Pass 4's web step was not performed (see §4). |
| **Caution** | 17,509 is **not** "the JAX catalog of ~13,000 currently available strains". It is every MGI strain record that lists a JAX stock ID, which can include retired or historical stocks. Active status cannot be determined from these sources. |
| **Caution** | `gene` / `allele` / `background` are **parsed from the strain name text** by regular expressions. They have not been checked against MGI marker or allele reports. |
| **Caution** | `rrid` and `jax_url` are **constructed by rule** from the JAX number. The pattern was confirmed on one record (JAX:000664, via the RRID registry). The other 17,508 were not individually checked. |

---

## 2. Contents

Highlighting the key coding modules:
1) s3_chunk_scanner_gpt.py     > Searches PubMed Central for articles mentioning strain aliases from a CSV, automatically partitions oversized date ranges, and retrieves matching article XML from the public PMC AWS S3 repository. Extracts bibliographic metadata and contextual strain mentions, then records the results in timestamped JSONL output files or an optional PostgreSQL database, with run-specific logs and completeness checks.
2) postgres_schema.py     > This script uses psycopg (v3) to create a highly optimized database schema. It features proper indexing, foreign keys, constraints to prevent duplicate entries, and a status tracking column to manage weekly batch workflows.
3) db_init_extended.py    > Optional. Preloads or refreshes catalog entries before a scan.  This script remains useful as a deployment and administration entry point. It delegates all DDL to postgres_schema.py, which remains the single source of truth for the database schema. You do not need to run both scripts to initialize the database: use the initializer or run the schema script directly.
4) langgraph_orchestrator.py
5) pipeline_coordinator.py
6) DockerFile
7) docker-compose.yml
8) test_pipeline.py
9) Streamlit_Dashboard.py
10) human_review_handler.py

```
JAX_Mouse_Strain_Reference/
├── README.md                          this file
├── JAX_Mouse_Strain_Reference.xlsx    workbook: Central_Identifiers, Search_Aliases, Source_Files, QA_Checks (+ README sheet)
├── strains.csv                        central identifier table            17,509 rows
├── strain_aliases.csv                 source-attested names + provenance  26,519 rows
├── strain_identifiers.csv             all IDs, long form                  73,015 rows
├── strain_genes.csv                   parsed genes                        14,565 rows
├── strain_alleles.csv                 parsed alleles/transgenes           17,800 rows
├── strain_sources.csv                 provenance per strain               52,523 rows
├── strain_search_terms.csv            search-optimised alias table       219,056 rows  (= Search_Aliases sheet)
├── pipeline/
│   ├── jax_strain_pipeline.py         the ENTIRE deterministic process (one file, one function per step)
│   └── requirements.txt               pandas, openpyxl, xlsxwriter
├── supporting/
│   ├── source_files.csv               input files + SHA-256
│   ├── qa_checks.csv / qa_summary.json   integrity checks and run metrics
│   ├── enrichment_queue.csv           17,509 JAX stocks awaiting Pass 4 web enrichment (jax_stock_number, rrid, jax_url)
│   └── mgi_master_strains.csv.gz      all 141,018 MGI strains merged from Passes 1–2 (not only JAX)
├── enrichment/                        (empty) drop jax_enrichment.csv here after Pass 4 web step; see §7
└── data/raw/                          put input files here (not shipped)
```

CSV files are UTF-8 **with BOM** so that Excel on Windows displays `×` and `ß` correctly. Identifiers such as `000664` are text; keep them as text if you re-import.

---

## 3. How each step was done: code vs. judgment

Everything marked **CODE** is in `pipeline/jax_strain_pipeline.py`, runs without AI, and gives byte-identical CSV output for identical input (verified by running twice and comparing a SHA-256 of all seven CSVs).

### 3a. Deterministic steps (in the script)

| # | Plan pass | Step | Function | What it does |
|---|---|---|---|---|
| 0 | 1 | Fetch (optional, `--fetch`) | `fetch_reports` | Downloads `MGI_Strain.rpt` and `MGI_Nonstandard_Strain.rpt` from `https://www.informatics.jax.org/downloads/reports/`, uses ETag/Last-Modified to skip unchanged files, keeps dated archive copies, refuses responses under 100 KB, **stops with an error** on any HTTP/network failure. *Only network-using step.* |
| 1 | 1, 2 | Load and validate | `load_rpt`, `load_summary`, `find_summary` | `.rpt`: exactly 3 tab fields per line, well-formed `MGI:n` IDs, no duplicates. Export: exact expected 6 columns, no duplicate IDs, ≥100,000 rows (catches truncated exports). Export date is read from the file name. Any violation **stops the run**. |
| 2 | 1, 2 | Clean names | `clean_name` | Unicode NFC; removes the `Â` mojibake byte and non-breaking spaces found in 4 `.rpt` names; strips outer whitespace (7 names). `×` and `ß` are preserved. 11 names changed in this run. |
| 3 | 1, 2 | Merge into MGI master | `build_master` | Outer-join `.rpt` (standard + nonstandard) with the export on MGI ID. **Official name rule:** `.rpt` name wins; export name only if the ID is absent from both `.rpt` files. QA proves every remaining name difference is explained by the export dropping non-ASCII characters (0 unexplained). |
| 4 | 3 | Isolate JAX records | `isolate_jax` | Splits the export's `IDs` field on `\|`; a record is JAX only if a token **exactly** matches `JAX:\d{6}`. Counts malformed tokens, multi-JAX records, and JAX numbers shared by several MGI records (all 0 in this run). |
| 5 | 4 (rule part) | Derive IDs/URLs | `build_tables` | `rrid = RRID:IMSR_JAX:<n>`; `RRID:MGI:<n>`; `jax_url = https://www.jax.org/strain/<n>`; `source_url = https://www.informatics.jax.org/strain/<MGI id>`; `canonical_id = STRAIN-<8-digit MGI number>` (stable across runs; independent of row order). Flagged `derived_rule`, `verified_at_source = no`. |
| 6 | 4 (hook) | Merge web enrichment if present | `load_enrichment` | Reads `enrichment/jax_enrichment.csv` if it exists, validates columns and that every `jax_id` exists in the catalog, then fills `jax_strain_name`, `common_name`, `active_jax_stock`, `status` and adds aliases. **No such file exists yet.** Tested with a synthetic file, then discarded. |
| 7 | 5 | Parse background / gene / allele from name | `parse_strain_name` | Regex logic for `Gene<allele>` notation, `Tg(...)`/`TgN(...)` transgenes, `Chr N<...>` chromosome substitutions, rearrangements (`Ts`, `In`, `Df`, …), compound alleles (`Ltb/Tnf/Lta`), backgrounds with spaces/semicolons, and digit-initial RIKEN genes. Allele class hint from the superscript (`tm`, `em`, `Gt(`). **Heuristic; see §5.** |
| 8 | – | Build tables | `build_tables` | Produces strains, aliases, identifiers, genes, alleles, sources. |
| 9 | – | Normalise and type names | `normalize_term`, `synonym_form` | `normalized_term` = lower-case alphanumerics only (`C57BL/6J` → `c57bl6j`). `synonym_form_heuristic` labels each MGI synonym `nomenclature_like` or `informal_like` by shape (MGI does not type synonyms itself). |
| 10 | – | Search-optimised table | `build_search_terms` | One row per searchable term: official names, synonyms, `000664`, `JAX:000664`, `JAX 000664`, `JAX 664`, `IMSR_JAX:000664`, `IMSR_JAX:664`, `RRID:IMSR_JAX:000664`, `MGI:n` variants, `RRID:MGI:n`. Adds **ambiguity counts** (see §6). |
| 11 | – | Integrity checks | `integrity_checks` | 20 checks (unique keys, foreign keys, ID formats, RRID/JAX consistency, every strain has a name alias and a source row, no unexplained name differences). Failure of a hard check **stops the run before any output is written**. |
| 12 | – | Change detection | `diff_against_previous` | Compares the new `strains.csv` with the previous one (added / removed / changed fields), archives the old file in `supporting/history/`, writes `supporting/changes_vs_previous_run.csv`. |
| 13 | – | Write outputs | `write_csv`, `write_excel`, `main` | CSVs, workbook, QA files, enrichment queue, master vocabulary. |

### 3b. Steps I (Claude) performed that are **not** in the code

| Action | Why it matters |
|---|---|
| Read the plan document and extracted the pass structure and target schemas. | The code follows the plan's table layouts. The plan itself was taken as given except for the corrections in §4. |
| Exploratory profiling of the three input files (row counts, overlaps, ID formats, encoding). | **Findings were converted into code checks** so they re-run every time: export-vs-`.rpt` name reconciliation, JAX token validation, shared/multiple JAX ID counts, schema checks. The exploration itself is not repeatable code. |
| Design decisions (listed in §4). | These are encoded as rules, but the decision to adopt them was mine and you may want to change some. |
| Two web searches: (1) confirmed the MGI report location and file names; (2) confirmed on one record that `RRID:IMSR_JAX:000664` resolves to `https://www.jax.org/strain/000664`, and that the RRID registry also lists the unpadded alternate `IMSR_JAX:664`. | This is the **only** verification of derived RRIDs/URLs. The MGI strain-page URL pattern (`/strain/MGI:n`) was **not** checked. |
| Ad hoc tests: determinism (two runs identical), change detection (simulated edit), enrichment merge (synthetic file), failure paths (network off, bad enrichment ID, changed export column name). | Run once by hand; **not packaged as a test suite**. |
| Reviewed parser output against known examples and iterated the regexes (3 rounds). | Reduced "partial" parses from 190 to 18. Spot-checked, not exhaustively verified. |

### 3c. Not done

* **Pass 4 web enrichment** (JAX product names, common names, descriptions, availability) — no network in this session, and I would not ship untested scraping code.
* **Validation of genes/alleles against MGI** — requires marker/allele reports not supplied (see §8).
* **Automated download of the Pass 2 strain-summary export** — it was supplied as a manual web export; I did not verify any programmatic endpoint for it.
* **Live testing of `--fetch`** — written, but the sandbox network was off. The failure path was tested; the success path was not.

---

## 4. Departures from, and corrections to, the plan

1. **Scope is JAX records only** (17,509), as the plan's Pass 3 specifies. The full 141,018-strain MGI vocabulary is kept in `supporting/mgi_master_strains.csv.gz` so the scope can be changed without re-downloading.
2. **"~13,000 JAX strains"** (plan) vs **17,509** found. MGI lists JAX IDs for more stocks than are presently orderable. The difference cannot be explained from MGI data alone.
3. **The strain-summary export is defective in two ways** the plan did not anticipate: its worksheet declares dimension `A1` (so read-only readers see only the header; the pipeline uses pandas to read it fully), and it **drops every non-ASCII character** (`×`, `ß`). The `.rpt` files keep them, hence the "`.rpt` name wins" rule.
4. **Substring `JAX` matching is wrong.** Two records have `MMRRC:…-JAX` tokens (MMRRC stock numbers held at JAX). Only exact `JAX:nnnnnn` tokens count.
5. **Substrain-stripped aliases were deliberately not generated.** The plan's example lists `C57BL/6` as an alias of `C57BL/6J`. `C57BL/6` and `C57BL/6J` are different strains, so the pipeline normalises punctuation only (`C57BL/6J` → `c57bl6j`) and never drops suffixes.
6. **Bare numbers (`664`) are not indexed**; `JAX 664` and `IMSR_JAX:664` are. A bare number collides with too many other identifiers.
7. **`jax_strain_name`, `common_name` are left empty, not copied from the MGI name.** Copying would present MGI data as if confirmed from JAX.
8. **`last_verified` = MGI export date**, with `verification_level = MGI_export_crosscheck_only`. It does not mean JAX/IMSR was consulted.
9. **Confidence values are policy labels, not probabilities:** 1.00 stated by MGI file; 0.95 rule-derived `RRID:IMSR_JAX`; 0.90 formatting variants and `RRID:MGI`; 0.85 / 0.75 / 0.50 name-parse (normal / complex symbol / failed).

---

## 5. Table guide

**`strains.csv`** — one row per strain. Key columns: `canonical_id`, `mgi_id`, `jax_id`, `rrid`, `official_strain_name`, `jax_strain_name` *(empty until Pass 4b)*, `common_name` *(empty)*, `strain_type` (MGI `.rpt` type), `strain_attributes` (export, `|`-separated), `background`, `gene`, `allele`, `nomenclature_status` (`standard` / `nonstandard` / `unknown_not_in_rpt`), `active_jax_stock` (`unknown`), `status` (`mgi_listed_jax_stock_unverified`), `other_repository_ids`, `name_parse_status` (`parsed` 15,988 / `no_allele_notation` 1,503 / `partial` 18), `source`, `source_url`, `jax_url`, `last_verified`, `verification_level`.

**`strain_aliases.csv`** — names attested by a source: `alias_type` = `official_mgi_name` (17,509) or `mgi_synonym` (9,010); plus `jax_*` types once enrichment exists. Columns: `alias_id`, `canonical_id`, `alias`, `alias_type`, `synonym_form_heuristic`, `normalized_term`, `source`, `source_record`, `confidence`, `date_verified`.

**`strain_search_terms.csv` / `Search_Aliases` sheet** — the lookup table for an agent: `search_term`, `canonical_strain_id`, `match_type`, `normalized_term`, `source`, `confidence`, plus `mgi_id`, `jax_id`, `official_strain_name` (so no join is needed) and ambiguity fields.

**`strain_identifiers.csv`** — one row per ID: `id_type` (`MGI_STRAIN`, `JAX_STOCK`, `RRID_IMSR_JAX`, `RRID_MGI`, `REPOSITORY_OR_DB_ID`, `MGI_ADDITIONAL_ACCESSION`), `namespace`, `id_value`, `id_curie`, `derivation` (`attested` / `derived_rule`), `verified_at_source`. Other repositories present: MMRRC 2,283; MPD 641; JPGA 33; APB 8; ORNL 4; unprefixed 3. Seven `MGI_ADDITIONAL_ACCESSION` rows are extra MGI numbers on a record; the source does not say whether they are secondary or merged IDs.

**`strain_genes.csv` / `strain_alleles.csv`** — parsed from the name. Alleles carry `kind` (`allele_superscript` 14,618; `transgene` 2,770; `rearrangement` 291; `chromosome_substitution` 121), `allele_class_hint`, `confidence`, `needs_marker_validation = yes`, and **empty** `mgi_allele_id` / `mgi_marker_id` columns reserved for Pass 5.

**`strain_sources.csv`** — per strain: one row per contributing source (`.rpt`, export, `pipeline rules`), the fields it contributed, data date, and `attested` vs `derived_rule`.

---

## 6. How to use the ambiguity fields

`n_candidates_jax_set` = number of distinct JAX strains sharing the same `normalized_term`. `n_candidates_all_mgi` = number across **all** 141,018 MGI strains (names and synonyms). `is_ambiguous_in_jax_set = yes` for 464 rows (221 distinct terms).

Example: `B6` is an MGI synonym for two JAX strains and five MGI strains; `C57BL/6J` normalises to the same term as three other MGI records. **An agent should not auto-resolve ambiguous terms**; it should present candidates or ask. IDs (`JAX:000664`, `MGI:3028467`) are unambiguous.

---

## 7. Running and scheduling

```
pip install -r pipeline/requirements.txt
python pipeline/jax_strain_pipeline.py            # build from files in data/raw
python pipeline/jax_strain_pipeline.py --fetch    # download the two .rpt files first (needs network)
```

Inputs in `data/raw/`: `MGI_Strain.rpt`, `MGI_Nonstandard_Strain.rpt` (the `*_rpt.txt` spellings also work), and `MGIstrains_YYYYMMDD_HHMMSS.xlsx` (newest used). Exit code 2 = pipeline stopped on a validation failure; the message names the cause.

Weekly schedule on Windows (MGI regenerates reports weekly):

```
schtasks /Create /SC WEEKLY /D MON /ST 06:00 /TN "JAXRef" /TR "python C:\path\JAX_Mouse_Strain_Reference\pipeline\jax_strain_pipeline.py --fetch"
```

**Weakest link for unattended runs:** the strain-summary export is still a manual download. If it is more than 14 days old the script logs a warning but continues, so a scheduled run can combine fresh `.rpt` files with a stale export. New strains appear in the `.rpt` files first; their synonyms and JAX IDs only arrive with a fresh export. The `diff` step reports what changed each run.

**Pass 4b contract** (when the web step exists): produce `enrichment/jax_enrichment.csv` with columns `jax_id, jax_strain_name, common_name, description, availability_status, active_jax_stock, extra_synonyms (|-separated), source_url, retrieved_date`, one row per JAX number (`supporting/enrichment_queue.csv` lists the targets). The next pipeline run merges it and fails if any `jax_id` is unknown or duplicated.

---

## 8. Recommended next steps

1. **Pass 4b:** retrieve JAX/IMSR product pages for the queued stocks (rate-limited, respecting site terms) to fill names, common names, availability. Consider checking whether JAX or IMSR offers a bulk export first.
2. **Pass 5 (gene/allele validation):** MGI's reports page lists `MRK_List2.rpt` (current marker symbols) and allele reports such as `MGI_PhenotypicAllele.rpt`. Joining parsed symbols to these would populate `mgi_marker_id` / `mgi_allele_id` and expose any mis-parsed names.
3. Save the ad hoc tests as a regression suite.
4. Decide whether non-JAX repositories (MMRRC, etc.) should enter the reference; the master vocabulary already has them.

---

## 9. Known limitations

* 2,173 MGI strains appear in the `.rpt` files but not in the export (118 standard, 2,055 nonstandard). Their ID lists are unavailable, so whether they carry JAX IDs is **unknown**; they are not in this dataset. Four JAX records appear in the export only; their names come from the export.
* 18 strains are `partial` (an allele symbol could not be separated from a complex legacy background, e.g. marker-interval congenics). Another 1,503 have no `<…>`/`Tg(…)` notation (inbred, recombinant inbred, wild-derived, etc.); 12 of those carry a targeted/endonuclease attribute but a name pattern the parser does not recognise (e.g. `Isr(...)`).
* `synonym_form_heuristic` is a shape guess; some synonyms will be mislabelled.
* A JAX stock number appearing once in the export does not make it a single biological strain, per the plan's own caution; this dataset keeps the strain (MGI) and stock (JAX) identifiers as separate fields.
