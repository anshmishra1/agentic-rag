# Agentic RAG

A production-oriented corrective Retrieval-Augmented Generation system built on
LangGraph. Rather than a fixed retrieve-then-generate pipeline, the graph
adaptively routes each query through relevance grading, query rewriting,
grounding verification, and constrained regeneration. The current V3 release
candidate extends that flow with artifact-aware evidence for text, figures,
tables, formulas, and OCR-derived content.

## Current project status

The V2 comparison baseline uses the versioned `agentic-rag-hybrid-v2` index with
`structure_aware_v2` chunking. In the reviewed v1/v2 evaluation, v2 raised
relevant-evidence recall in the application's final five passages from 76% to
92% and improved final ranking MRR from 0.647 to 0.703 without a material
retrieval-latency increase. Structure-aware chunking remains fixed for that
baseline while embeddings remain unchanged.

V3 is now a connected release candidate beside that stable V2 baseline. The
selected PyMuPDF Layout parser, conditional RapidOCR fallback, typed artifact
and token-safe search-record contracts, content-addressed asset store,
transactional PostgreSQL manifests, isolated Pinecone reader/writer, and
`POST /v3/ingest` route are connected through the API and Streamlit. The current
network-disabled offline suite passes 314 tests.

Seven hash-verified documents are ready in `agentic-rag-hybrid-v3` under the
current schema. PostgreSQL and Pinecone agree on 2,208 search records linked to
2,203 stable source artifacts. A targeted public-API replay of the former
FlashAttention Figure 1 miss now returns cited page-2 text plus figure evidence
with an answered, grounded verdict. The broader manual release-candidate run
across routing, natural questions, formats, negative cases, citations, and
latency remains the promotion gate. V2 remains available as the comparison and
rollback baseline.

A source-reviewed V3 acceptance suite is now ready with 49 cases over seven
documents. Its 18-case core bounds the first Groq-backed run; all retrievable
cases run retrieval-only first. See
[`docs/V3_END_TO_END_EVALUATION.md`](docs/V3_END_TO_END_EVALUATION.md) for the
questions, expected evidence, execution order, and promotion method. Exact
machine-readable expectations live in
[`eval_set/v3_end_to_end_questions.json`](eval_set/v3_end_to_end_questions.json).

The active v2 reliability repair separates the current information need from a
pure reformat of the previous answer, canonicalizes provider citation variants,
and prevents unsupported motivation or exhaustive-document claims during
generation and correction. Its eleven-case acceptance contract records route,
retrieval-query, citation, status, retry, and correction evidence without
storing document passages.
See [`docs/PROJECT_PROGRESS.md`](docs/PROJECT_PROGRESS.md) for the completed
milestones, measured evidence, current limitations, and upcoming work. Detailed
v1/v2 results are in
[`docs/CHUNKING_V2_BROAD_EVALUATION.md`](docs/CHUNKING_V2_BROAD_EVALUATION.md).
The reliability method and release gates are in
[`docs/V2_ANSWER_RELIABILITY_ACCEPTANCE.md`](docs/V2_ANSWER_RELIABILITY_ACCEPTANCE.md).

## Architecture

```
contextualize_question
        |
        +-- control query -----------------------------> record_turn -> END
        |
        +-- general-model query ----------------------> generate_general_answer
        |                                                       |
        |                                                       v
        |                                                  record_turn -> END
        |
        +-- unresolved source ------------------------> request_clarification
        |                                                       |
        |                                                       v
        |                                                  record_turn -> END
        |
        v
     retrieve  <----------------------------------------------+
        |                                                      |
        v                                                      |
  assess_retrieval                                             |
        |                                                      |
   +----+----+------------------+                              |
   |         |                  |                              |
strong   ambiguous            weak                              |
   |         |                  |                               |
   |         v                  +-------------------------------+
   |   grade_documents                    rewrite_query
   |         |                                  ^
   |    +--------+--------+                    |
   | relevant contradicted irrelevant --(retry)+
   |    |         |                            |
   +----+---------+                            |
        |
        v
    generate
        |
        v
  check_hallucination
        |
   +------------+------------------------------+
   |            |                              |
grounded   insufficient_evidence           unsupported
   |         |                 |                |
   v    retries left     retries exhausted      v
record_turn     |                 |       correct_generation
   |            v                 v              |
  END      rewrite_query       abstain           v
               |                 |       check_hallucination
               v                 v        (one final pass)
            retrieve        record_turn -> END
```

**The core idea:** every routing decision is made by the cheapest mechanism that can be trusted for that decision — a deterministic check where the signal is reliable, a fast-tier LLM call where judgment is needed but cheaply, a full LLM call only where quality genuinely matters (final answer generation).

## Key features

- **Retrieval confidence gating** — a three-way policy (`generate` / `grade` / `rewrite_query`) decides whether an LLM answerability grader is necessary. Strong score distributions skip grading; any non-empty candidate set below the provisional score floor goes to semantic grading instead of a blind rewrite. The grader distinguishes directly answerable evidence, counterevidence that can correct a false premise, and irrelevant evidence. Both answerable classes reach generation; only missing or semantically irrelevant evidence triggers retrieval correction. An uncertain grader result proceeds to generation and the stricter grounding verifier remains the final acceptance gate. Rewrites receive the latest failure reason and stop with an abstention if they repeat an attempted query.
- **Hybrid retrieval** — dense (embedding) and sparse (BM25) retrieval run independently and are fused via Reciprocal Rank Fusion, then reranked with a cross-encoder for final relevance scoring. Within the existing five content slots, the first two RRF candidates are retained as a recall safeguard; cross-encoder scores still order the selected passages. `RERANK_CONTENT_RRF_RESERVE` defaults to `2` and can be set to `0` to use pure cross-encoder selection. This is a rank-slot rule, not a score cutoff. BM25 is fit per document at ingestion time, not globally, matching the document-scoped retrieval model.
- **Direct-first answer evidence** — retrieval keeps its global score order for confidence assessment, while answer construction preserves all five selected content passages and appends at most one whole-document overview. Generation, correction, citation labels, and grounding verification use the same selection, so an overview cannot undo the reranker's content recall safeguard.
- **Document-scoped retrieval** — every document gets a stable, content-derived ID (a hash of its bytes), so retrieval, BM25 encoding, and vector storage are all scoped to a specific document rather than the whole corpus. Re-ingestion upserts the current vectors, updates the document's BM25 state, then removes obsolete vectors for that document.
- **Whole-document overview chunks** — a summary generated at ingestion time and retrieved separately from content chunks, so structural questions ("what does this document cover") aren't left to chunk-level semantic search, which is the wrong granularity for that kind of question.
- **Corrective grounding handling** — a structured verifier distinguishes a grounded answer, insufficient evidence, and unsupported generation. Verification sees only the passages cited by the answer and a semantic projection that removes rendering-only fences, borders, connectors, table separators, and math delimiters while retaining factual text in diagram nodes, table cells, and equations. Missing or unknown labels and uncited factual list items route to correction without an LLM verdict. Evidence problems route back to retrieval and end in an explicit abstention when retries are exhausted; a relevance grade of irrelevant also abstains after the retrieval retry budget. Generation problems receive one constrained correction using the verifier's unsupported-claim list. Malformed verifier output fails closed instead of silently approving an answer.
- **Multi-provider LLM fallback** — requests fall through a configurable provider chain (Groq, Cerebras, NVIDIA NIM, OpenRouter, optionally AWS Bedrock), split into a "primary" tier (used only for final answer generation) and a "fast" tier (used for classification-style calls: grading, rewriting, hallucination checking). A 429 places that provider/model on a bounded cooldown, honors `Retry-After` when supplied, and moves to a distinct fallback instead of repeating the same quota-consuming request.
- **Typed semantic routing** — the UI explicitly separates selected-document, general-model, and automatic source modes. First-turn questions in an explicit source mode are planned locally; Auto mode and conversational follow-ups use the bounded fast-tier planner because they require semantic judgment. Auto mode keeps a selected document when its title names the question topic, preserves complete standalone search text, and still permits clearly unrelated general questions. Requested answer presentation is independent from source evidence: asking for a table or flowchart does not require the document to contain that artifact.
- **GPU-accelerated local reranking** — the cross-encoder and embedding model run locally (auto-detecting CUDA if available), so reranking adds no external API cost or quota pressure.

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM providers | Groq, Cerebras, NVIDIA NIM, OpenRouter, AWS Bedrock (optional) |
| Vector store | Pinecone (dotproduct metric, required for hybrid dense+sparse vectors) |
| Sparse retrieval | BM25 (pinecone-text), fit per document |
| Reranking | Cross-encoder (sentence-transformers), CUDA-accelerated when available |
| Embeddings | HuggingFace sentence-transformers |
| V3 PDF parsing | PyMuPDF4LLM/Layout plus conditional RapidOCR |
| V3 artifact metadata | PostgreSQL manifests plus content-addressed local assets |
| Conversation persistence | Postgres (Neon), via LangGraph's checkpointer |
| Document registry | Postgres |
| Backend | FastAPI |
| Frontend | Streamlit |

## Project structure

```
src/agentic_rag/
  config.py                 Typed settings (provider keys, models, thresholds, timeouts)
  llm/
    provider.py               Tiered (fast/primary) provider fallback chain
    vision.py                  Image captioning for multimodal ingestion
  ingestion/
    loaders.py                 PDF / image / audio loaders
    chunking.py                 Text splitting, document-id tagging
    pipeline.py                  Ingestion orchestration: load -> chunk -> overview -> upsert
    registry.py                  Postgres-backed document registry
    contracts.py                 V3 artifact, search-record, locator, and schema identity
    asset_store.py               Content-addressed local binary asset storage
    manifest_store.py            Transactional V3 PostgreSQL metadata and lineage
    v3_pipeline.py               Parsed -> indexed -> ready coordination
    v3_runtime.py                Lazy opt-in V3 parser/storage/index composition
  retrieval/
    vectorstore.py              Dense + BM25 + RRF retrieval against Pinecone
    v3_writer.py                 Isolated V3 text-proxy vector writer
    reranker.py                  Cross-encoder reranking (GPU-aware)
    sparse.py                     Per-document BM25 fit/serialize
  policies/
    retrieval.py                 Retrieval confidence gate (generate / grade / rewrite)
    generation.py                 Refusal detection, context/history budget limits
    conversation.py               Closed control-command recognition
    query_planning.py              Typed semantic source, relationship, and format plan
    answer_format.py               Shared math and textual-flowchart presentation rules
    grounding.py                   Hallucination-check verdict parsing
    calibrate_retrieval.py         Empirical threshold calibration against labeled examples
  graph/
    state.py                     Domain-split state contracts (request/query/retrieval/evidence/answer/control)
    nodes.py                      All graph node functions
    edges.py                      Conditional routing logic
    builder.py                    Graph assembly
  api/
    main.py                       FastAPI app: /query, /ingest, /documents, /health
  observability/
    trace.py                      Structured logging configuration
  core/
    timing.py                     Per-request performance tracking (context-scoped)
app/
  streamlit_app.py               Frontend: upload, document selection, chat
scripts/
  create_hybrid_index.py         One-time Pinecone index setup (dotproduct metric)
```

## Setup

```bash
cp .env.example .env   # fill in provider keys, Pinecone key, Postgres URL
uv sync --frozen
```

Create a new versioned Pinecone index (must use `dotproduct` for hybrid search):

```bash
python scripts/create_hybrid_index.py --name agentic-rag-hybrid-v2
```

This is a live Pinecone operation. Run it only after the corresponding
chunking or embedding change passes offline evaluation. Set
`PINECONE_INDEX_NAME` to the chosen version before ingestion. Existing
`agentic-rag-hybrid` data remains untouched for rollback.

The PostgreSQL document registry is index-aware. On first use it assigns
pre-versioning rows to the legacy `agentic-rag-hybrid` index and changes the
registry identity from `document_id` to `(index_name, document_id)`. Document
listing, BM25 lookup, and deletion then operate only on the active index. The
legacy identity remains for migration and historical evaluation; current local
runtime work starts from v2 and develops v3 separately.
The registry also records the chunking strategy and rejects ingestion when an
index already contains a different strategy.

The archived v1 control used `CHUNKING_STRATEGY=token_window_v1`: plain PDF
extraction followed by 240-token windows with 40-token overlap. The current v2
runtime uses `CHUNKING_STRATEGY=structure_aware_v2`: layout-preserving PDF
extraction, heading detection, sentence/bullet units, section-aware packing up
to the same 240-token embedding limit, and section/page-range metadata. V1 is
kept in source only for reproducible historical comparisons and legacy-row
interpretation.

## Running locally

```bash
uv run uvicorn agentic_rag.api.main:app --reload      # backend, :8000
uv run streamlit run app/streamlit_app.py              # frontend, :8501
```

On Windows, `scripts/run_server.ps1` starts the backend with the project
virtual environment and writes versioned logs under `logs/`.

## Running with Docker Compose

After creating `.env`, use the local runner from PowerShell or Command Prompt:

```powershell
python scripts/local_docker.py start
python scripts/local_docker.py status
python scripts/local_docker.py stop
```

`start` reuses existing images and containers. On a CUDA-capable local machine,
the runner exposes the GPU only to the API and runs embedding plus Ettin
reranking on CUDA. Use `start --cpu` for a portable recovery run. The runner creates a temporary
Compose override that writes API run files to `logs/runs/` on the host. It
keeps provider settings in `.env` and does not read that file itself. Model
downloads are disabled at runtime because the pinned embedding and reranking
models are already included in the application image. The NLTK `punkt_tab` and
`stopwords` resources required by Pinecone BM25 are also provisioned and
verified during the build, so the first query does not download them. For a quota-limited
diagnostic run, use `start --bounded` (two graph retries, one attempt per LLM
call). `--build api`, `--build frontend`, and `--build all` each rebuild the
one shared application image before starting both services. The runner's
`stop` command retains containers and the PostgreSQL volume.

The default local Docker target contains CUDA PyTorch, the exactly pinned
PyMuPDF4LLM/Layout plus RapidOCR parser, the MiniLM embedding, and the Ettin
reranker in one shared application image. Only model operations that support
CUDA run on the GPU; PDF layout parsing, RapidOCR, BM25, storage, API work,
PostgreSQL, and Streamlit remain on CPU. A process-local lock serializes local
model inference so the 4 GiB GPU is not loaded by overlapping operations.
The smaller `cpu-runtime` Docker target is used by CI and non-CUDA verification.
The Dockerfile keeps uv's package cache outside image layers, so later builds
can reuse downloads without shipping the cache in the image. A source change
still requires rebuilding the affected service once; routine starts reuse the
existing image. Pull requests and `main` run the locked Docker build, runtime
import check, baked-model and parser offline checks, and offline unit suite in
`.github/workflows/offline.yml` without provider credentials or network access
during tests.

The image pins exact Hugging Face revisions for
`sentence-transformers/all-MiniLM-L6-v2` and
`cross-encoder/ettin-reranker-32m-v1`. The first image build downloads those
public model files into a reusable Docker layer. API startup and CI load them
with Hugging Face networking disabled, so a host cache is no longer required.
Changing the embedding model or revision is an index migration and requires a
new versioned `PINECONE_INDEX_NAME` plus document re-ingestion. Changing only
the reranker does not alter stored vectors, but it does require a matched rank,
latency, and score-distribution comparison before routing thresholds are
trusted.

The V3 parser identity is also exact: PyMuPDF, PyMuPDF4LLM, and PyMuPDF Layout
1.28.2 with RapidOCR 3.9.2, 144 DPI evidence renders, 150 DPI conditional OCR,
and ignored repeating page headers/footers. A network-disabled build check
rejects missing or different package versions before the image can be used.
These Artifex packages report dual AGPL 3.0/commercial licensing; production
distribution or hosted deployment remains gated on choosing a compliant path.

V3 uses a separate retrieval schema and index from V2. The `POST /v3/ingest`
boundary lazily composes the selected parser, asset store, manifest store,
shared embedding runtime, document-local BM25 encoder, and isolated V3 index
writer. Importing the API or using the V2 `POST /ingest` route does not
construct V3 external clients. Its PostgreSQL manifest store
persists schema identity, artifacts, page-to-region parent links, locators,
search records, and `parsed`/`ready` state transactionally. Binary page and
region assets use the dedicated `v3_assets` Compose volume; Pinecone payloads
contain searchable text and compact provenance, never binary content. The V3
writer receives its index explicitly and writes document-prefixed deterministic
IDs before pruning stale IDs. A failed vector write leaves the manifest
non-ready, so future retrieval can fail closed. The route has offline regression
coverage, and its DDL, lineage, parsed-state hiding, ready promotion, and schema
guard passed against a disposable PostgreSQL 16 instance. The complete
seven-document corpus has since been migrated into the token-safe schema, and
the targeted Figure 1 replay verified the public query and evidence path.
The V3 reader, unified document listing, version-scoped graph invocation,
structured evidence response, safe artifact-preview route, and Streamlit V2/V3
selection are implemented and covered offline. V3 deletion stays unavailable
until vector, manifest, and asset cleanup can be coordinated atomically.

For direct Compose use, `docker compose up -d --no-build` starts existing
images, but it does not add the local runner's GPU reservation or host log
folder.

The frontend is available on port 8501, the API on port 8000, and PostgreSQL
is exposed to the host on port 5442. API startup warms both the embedding and
cross-encoder models; their exact revisions are already baked into the image.

The API reads provider settings from `.env`. For a Groq-only local run, set
these values in your own `.env` (along with `GROQ_API_KEY` and Pinecone
credentials):

```dotenv
PROVIDER_ORDER=groq
GROQ_MODEL=openai/gpt-oss-120b
GROQ_FAST_MODEL=openai/gpt-oss-20b
PRIMARY_LLM_MAX_TOKENS=2048
FAST_LLM_MAX_TOKENS=512
GROQ_VERIFIER_REASONING_EFFORT=low
RELEVANCE_GRADER_MAX_TOKENS=512
QUERY_REWRITE_MAX_TOKENS=512
GROUNDING_VERIFIER_MAX_TOKENS=1024
DEBUG=false
```

These model IDs are hosted by Groq and use the Groq key, not a direct OpenAI
API key. Groq free-tier rate limits still apply across requests. Changing only
these settings requires recreating the API container, not rebuilding its image.

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Purpose |
|---|---|
| `PROVIDER_ORDER` | Fallback order across LLM providers, e.g. `cerebras,groq,nvidia,openrouter` |
| `PRIMARY_LLM_MAX_TOKENS`, `FAST_LLM_MAX_TOKENS` | Output caps for primary and fast model calls (defaults: 2048 and 512) |
| `SEMANTIC_ROUTER_MAX_TOKENS` | Output cap for the typed semantic query plan (default: 256) |
| `RELEVANCE_GRADER_MAX_TOKENS`, `QUERY_REWRITE_MAX_TOKENS` | Dedicated caps for the short retrieval-decision tasks (defaults: 512 each) |
| `GROQ_VERIFIER_REASONING_EFFORT` | Groq GPT-OSS effort for bounded fast routing, grading, rewriting, and verification calls; `low` by default (historical variable name retained) |
| `GROUNDING_VERIFIER_MAX_TOKENS` | Verifier-only completion cap (default: 1024) |
| `LLM_RATE_LIMIT_COOLDOWN_SECONDS` | Default provider/model cooldown after HTTP 429 when `Retry-After` is absent (default: 60 seconds) |
| `LLM_RATE_LIMIT_MAX_COOLDOWN_SECONDS` | Maximum accepted cooldown from configuration or `Retry-After` (default: 300 seconds) |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Vector store connection and retrieval schema version; use a name such as `agentic-rag-hybrid-v2` for migrations |
| `CHUNKING_STRATEGY` | `structure_aware_v2` for the current runtime; archived `token_window_v1` exists only for historical comparison; one strategy per index |
| `V3_SEARCH_RECORD_MAX_TOKENS` | Maximum MiniLM-token length for one V3 searchable representation; oversized proxies split while their source artifact remains whole (default: 240) |
| `V3_REQUIRED_EVIDENCE_SLOTS` | Existing five-record answer-context slots reserved for the explicitly requested figure/table/formula lane; this is not a score threshold (default: 2) |
| `V3_EVIDENCE_COMPANION_SLOTS` | Nearby same-page text/caption slots retained for a selected structured artifact inside the same context budget (default: 1) |
| `POSTGRES_URL` | Conversation checkpointing + document registry |
| `EMBEDDING_MODEL`, `EMBEDDING_MODEL_REVISION` | Dense embedding model and immutable Hugging Face revision baked into Docker |
| `EMBEDDING_DEVICE` | `auto` (default), `cuda`, or `cpu`; the local Docker runner selects CUDA unless `--cpu` is used |
| `CROSS_ENCODER_MODEL`, `CROSS_ENCODER_MODEL_REVISION` | Reranker model and immutable Hugging Face revision baked into Docker; defaults to pinned Ettin 32M |
| `CROSS_ENCODER_BATCH_SIZE` | Local reranking batch size; defaults to the locally measured value of 8 |
| `MODEL_LOCAL_FILES_ONLY` | Prevent runtime model downloads; Docker sets this to `true` |
| `MAX_GENERATION_CONTEXT_DOCUMENTS`, `MAX_GENERATION_CONTEXT_OVERVIEWS` | Direct-passage and supplemental-overview answer-context caps (defaults: 5 and 1) |
| `RETRIEVAL_MIN_TOP_SCORE` | Provisional absolute score floor for retrieval routing (`RETRIEVAL_STRONG_TOP_SCORE` is currently unused) |
| `CROSS_ENCODER_DEVICE` | `auto` (default), `cuda`, or `cpu`; the local Docker runner selects CUDA unless `--cpu` is used |
| `MAX_RETRIES` | Cap on rewrite/retry loop iterations |
| `DEBUG` | Verbose logging (candidate-level retrieval/rerank tables) |

The grounding verifier has its own 1024-token cap because a bounded Groq
GPT-OSS 20B diagnostic consumed 510 of 512 completion tokens in reasoning
and returned empty answer text with `finish_reason=length`. Groq GPT-OSS
semantic routing, relevance grading, query rewriting, and verification use
low reasoning effort so their small visible results fit within bounded task
budgets. An identical, separate post-change probe
returned valid verdict JSON with 70 completion tokens and
`finish_reason=stop`. Empty verifier text still fails closed, with only
finish reason and token counts added to diagnostics. This verifies the
provider-call mechanism, not a full application answer.

Retrieval thresholds in `config.py` are provisional. The calibration command
requires reviewed positive and negative query/document pairs; it no longer
uses the old hard-coded document ID.

To audit the available evaluation questions and source files without loading
models or contacting providers, run:

```bash
python -m agentic_rag.evaluation.dataset_audit --pdf-dir PDF
```

The 80-question set contains 70 single-document questions and 10
cross-document questions. It has answer references but no verified relevant
chunk IDs or negative query/document labels, so it cannot yet measure
retrieval hit rates or justify score thresholds. Add those labels and collect
scores from the current hybrid retrieval path before changing thresholds.
The offline review set and instructions are in
[`docs/RETRIEVAL_EVAL_LABELING.md`](docs/RETRIEVAL_EVAL_LABELING.md). A reviewed
46-pair pilot is available for retrieval-only score collection; the original
48 proposed pairs remain separate, and chunk-level labels are still pending.

The calibration input is a JSON list with `query`, 64-character `document_id`,
and boolean `should_match` on every row. Offline rows also need `top_score`.
After indexing the reviewed source documents, collect scores explicitly:

```bash
python -m agentic_rag.policies.calibrate_retrieval labeled_pairs.json --live --scores-output scores.json
python -m agentic_rag.policies.calibrate_retrieval scores.json
```

The first command reads PostgreSQL and queries Pinecone but does not call an
LLM; the second analyzes saved scores offline. The score snapshot also records
the routing metrics and ranked chunk identities without document text. With
development/holdout labels, only development rows can suggest a floor; the
holdout rows report errors at that fixed candidate. The analyzer reports no
candidate when either development class has fewer than five examples or their
scores overlap. Ratio, gap, and overview settings need separate routing
evaluation before changing configuration.

The evaluation history, index-consistency diagnosis, and cleanup safety gate
are documented in [`docs/RETRIEVAL_EVALUATION_RATIONALE.md`](docs/RETRIEVAL_EVALUATION_RATIONALE.md).
The V3 schema uses token-safe artifact search records. It preserves complete
figures, tables, formulas, page assets, and source locators while splitting only
an oversized searchable text proxy beneath the embedding model's input limit.
Query planning carries a typed evidence requirement so structured-source
questions retain a bounded artifact lane without changing ordinary text
selection. The separate V3 end-to-end suite covers parser evidence, retrieval and
reranking, routing, answer formats, citations, verdicts, and latency. Its
retrieval-only executor reuses the production V3 reader and Ettin reranker,
records IDs, provenance, dense/sparse/RRF/reranker signals, and stage latency,
and never calls an LLM provider. It requires ready V3 manifests and an explicit
`--live` flag:

```bash
python scripts/evaluate_v3_retrieval.py --live --output logs/evaluation/v3-retrieval.json
```

Reviewed positive cases are scored against page, artifact-type, and grounding
labels. Unanswerable cases are retained in the report but excluded from
positive hit-rate denominators because they do not identify a relevant record.
See [`docs/V3_END_TO_END_EVALUATION.md`](docs/V3_END_TO_END_EVALUATION.md).

## API

- `POST /query` - `{question, session_id, document_id, source_mode, retrieval_version}` returns the answer plus routing fields, the planned evidence requirement, grounding status, citations, contexts, and additive structured V3 evidence
- `POST /ingest` - V2 multipart upload returning `{filename, document_id, index_name, chunking_strategy, chunks_indexed}` per file
- `POST /v3/ingest` - opt-in V3 PDF ingestion returning parser, schema, artifact, search-record, vector, and ready-state metadata
- `GET /v3/documents/{document_id}/artifacts/{artifact_id}/asset` - return a ready cited artifact preview without exposing its storage key
- `GET /documents` — list ready V3 manifests and existing V2 registry rows through one versioned selection contract
- `GET /health` — liveness check
