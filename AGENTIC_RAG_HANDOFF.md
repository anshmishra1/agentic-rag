# Agentic RAG — Project Handoff (for a new assistant conversation)

Use this file at the start of a new chat instead of re-explaining the project. It reflects the current source, measured evaluations, and real runtime evidence; source and Git state remain authoritative.

## What this is
A LangGraph-orchestrated corrective RAG application — PDF/image/audio ingestion, hybrid retrieval, adaptive routing, hallucination correction. Built as Ansh's primary portfolio/interview piece. Repo: `anshmishra1/AgenticRAG` (also a fresh empty repo `anshmishra1/agentic-rag` was set up for a clean push). Runs locally on Windows, GPU-accelerated (RTX 2050, CUDA 13.0, torch 2.13).

## IMPORTANT: two divergent project histories exist
A separately-maintained `PROJECT.md` handoff doc (written for local-Ollama/other-tool sessions) describes a **simpler** architecture with no hybrid search, no cross-encoder, no `correct_generation` node — and explicitly lists those as "not yet done" / "don't add them just because they sound advanced." **That document is stale relative to what's actually running.** A real application log from 2026-09-17 confirms the advanced architecture below (dense+sparse query representation, `route_after_retrieval_assessment`, `grounding_diagnosis`/`correction_attempted`) is the live one. Trust logs/source over either handoff doc when they conflict.

## Current architecture (confirmed live)
```
contextualize_question → (control? → record_turn/END) → retrieve
retrieve → assess_retrieval (3-way: generate | grade_documents | rewrite_query)
grade_documents → (relevant → generate) | (irrelevant → rewrite_query while retries remain, then abstain)
rewrite_query → retrieve when novel | abstain when empty or repeated
generate → check_hallucination
check_hallucination → grounded→END | insufficient_evidence→rewrite_query (or abstain when retries are exhausted) | unsupported→correct_generation
correct_generation → check_hallucination (one more pass, single correction budget)
```
- **Retrieval:** Pinecone (dotproduct metric index, required for hybrid), dense + BM25 sparse retrieved separately, fused via RRF, then locally reranked by the pinned Ettin 32M model (CPU/CUDA auto-selection). Overview chunks (whole-doc summaries generated at ingest) are retrieved separately from content chunks for structural questions.
- **Document scoping:** every doc gets a SHA-256 content-hash `document_id`; retrieval/BM25/vectors are scoped to it; re-ingestion overwrites rather than duplicates (deterministic vector IDs, content-hash based).
- **Index migrations:** the active Pinecone index name is the retrieval schema identity. New chunking, embedding, sparse encoding, metadata, or vector-ID behavior uses a numeric versioned index such as `agentic-rag-hybrid-v2`. PostgreSQL keys BM25 and document registry state by `(index_name, document_id)`. V1 is an archived evaluation/migration reference; active local work is V2 plus the isolated V3 schema.
- **Versioned chunking:** `structure_aware_v2` is the current runtime default. It uses layout-preserving PDF extraction, detects headings and bullets, packs complete sentence units beneath the 240-token cap, and stores section plus page-range metadata. The registry records this strategy and prevents two strategies from sharing one index. `token_window_v1` remains only for historical reproducibility and legacy-row interpretation.
- **Current evaluation decision:** reviewed retrieval evidence improved from 76% to 92% in the final five passages with v2, so chunking and embeddings remain fixed while citation normalization, abstention classification, and difficult-passage selection are repaired. See `docs/PROJECT_PROGRESS.md` and `docs/CHUNKING_V2_BROAD_EVALUATION.md`.
- **Citations:** generation context uses deterministic per-answer labels (`[S1]`, `[S2]`, ...); valid labels referenced by the final answer are returned by the API and displayed in Streamlit with filename/page or chunk-type provenance.
- **Retrieval confidence gate** (`policies/retrieval.py`): absolute score floor + relative shape (top/mean ratio, gap ratio) + overview-dominance special case → decides whether to skip the LLM grader entirely.
- **LLM providers:** tiered fallback chain (fast tier for classification/grading/rewriting, primary tier for final generation only), across Groq/Cerebras/NVIDIA/OpenRouter/Bedrock(optional), configurable order.
- **Conversation memory:** LangGraph Postgres checkpointer (Neon), thread_id-scoped.
- **Packaging:** Python 3.12 dependencies are locked by `uv.lock`; Docker builds from the lock file, runs as a non-root user, and Compose provides a health-checked local PostgreSQL service on host port 5442.
- **Model and parser provisioning:** Docker pins exact Hugging Face revisions for the MiniLM embedding model and Ettin 32M reranker, plus exact PyMuPDF, PyMuPDF4LLM, PyMuPDF Layout, and RapidOCR versions. CI verifies the selected parser with networking disabled. API and frontend share one image, while only the API receives GPU access. Embedding and reranking use the 4 GiB RTX 2050; PDF layout parsing, RapidOCR, and ordinary application work remain on CPU. CI uses the smaller CPU target.
- **V3 persistence and query boundary:** PostgreSQL stores retrieval-schema identity, artifact/search-record metadata, page-to-region parent links, locators, BM25 state, and fail-closed `parsed`/`ready` status. The `v3_assets` Compose volume stores binary page and region assets. The isolated V3 writer and reader use explicit dependencies and document-prefixed IDs without modifying the V2 vector store. `POST /v3/ingest` remains opt-in; `/query` accepts `retrieval_version`, reuses the existing Ettin reranker and corrective graph for ready V3 manifests, returns structured cited evidence, and keeps V2/V3 checkpoints separate. Streamlit can select both versions and preview cited artifacts. PR #34 merged; the isolated V3 index exists; a controlled scanned-PDF ingest/query returned a grounded, cited OCR answer and valid preview, then removed all test vectors, manifest rows, and assets.
- **V3 evidence-selection repair (offline validated):** the next schema splits only oversized searchable proxies at 240 MiniLM tokens while keeping source artifacts and citation locators whole. Query planning carries `text|visual|table|formula|mixed`; structured questions reserve two existing context slots for matching RRF candidates plus one same-page caption/explanation, and a filtered artifact lookup runs only when the ordinary candidate pool lacks the requested form. Ettin and Groq remain text-proxy models, so raw-pixel or spatial claims still require an explicit representation or must be qualified/abstained. The complete 312-test network-disabled suite passed; fresh V3 ingestion and live application comparison remain required.
- **Grounding verification:** structured JSON verdicts distinguish grounded answers, insufficient evidence, and unsupported generation. Unsupported-claim text is passed into one constrained correction; exhausted evidence retries produce a deterministic abstention rather than speculative generation.
- **Observability:** fixed — no more unbounded `trace` accumulation in checkpointed state (was growing forever, printing the whole session's history every turn); now logger-based, with per-request + session-level PerformanceTracker via contextvars (not a module global anymore).

## Key files
`graph/{state,nodes,edges,builder}.py`, `retrieval/{vectorstore,v3_writer,v3_reader,reranker,sparse}.py`, `policies/{retrieval,generation,conversation,grounding,calibrate_retrieval}.py`, `ingestion/{loaders,chunking,pipeline,registry,contracts,asset_store,manifest_store,v3_pipeline,v3_runtime}.py`, `llm/provider.py`, `api/{main,evidence}.py`, `app/{streamlit_app,streamlit_support}.py`, `core/timing.py`.

## Real bugs fixed this project (don't re-litigate these)
Duplicate Pinecone vectors (was: random UUID vector IDs → fixed: content-hash IDs), overview chunk buried by absolute-floor check running before overview-dominance check, refusal answers ("I don't know") wrongly flagged hallucinated, `rewrite_query` losing half of compound questions because it only saw the last attempt not the original question, `classify_query_intent` shadowed-duplicate-function bug, unbounded `trace` state growth, module-global (not request-scoped) performance tracker, mislabeled "RRF" scores that were actually raw Pinecone hybrid scores.

## Open/unresolved right now
- V3 ingestion and querying passed one controlled live fixture. Broad matched
  retrieval and bounded answer evaluation across the reviewed corpus remains.
  V3 deletion remains intentionally hidden until vector, manifest, and asset
  removal is coordinated in the product API.
- The V3 evaluation suite is source reviewed: 49 cases across seven documents,
  including seven controlled/natural pairs and an 18-case bounded core. It
  covers ambiguity, follow-ups, figures/tables/formulas/OCR, false premises,
  absent evidence, routing, citations, formats, and latency. The corpus is now
  ingested; exact artifact and search-record labels remain pending review; see
  `docs/V3_END_TO_END_EVALUATION.md`.
- The V3 retrieval-only executor is implemented. It reuses the production
  hybrid reader and Ettin reranker, emits no passage text, and records exact
  record/artifact IDs, provenance, per-stage ranks/scores, production
  selection, and latency without calling an LLM. Its first 46-case corpus run
  completed without errors. Page-only labels undercounted valid evidence and
  two follow-up cases lacked prior-turn context, so its preliminary rates are
  diagnostic rather than promotion metrics.
- Retrieval score thresholds remain provisional. Cross-encoder scores are
  ranking signals, not calibrated probabilities.
- PyMuPDF4LLM/Layout packages report AGPL 3.0 or commercial licensing. Local
  evaluation can continue, but public AWS deployment requires an explicit
  compliant licensing decision.
- RAGAS remains legacy/dead evaluation code; current decisions use reviewed
  passage labels, retrieval ranks, citations, answer status, and bounded live
  checks.
- Streaming responses, richer progress reporting, authentication, upload
  limits, and rate limiting remain product/deployment work.
- Agentic tool use such as web search and calculators has not been added; the
  current system is a corrective LangGraph workflow.

## User's standing preference
Show changed/new file contents directly in chat rather than re-zipping the whole project; only zip for the initial scaffold or when explicitly asked.
