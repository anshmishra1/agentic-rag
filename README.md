# Agentic RAG

A production-grade, corrective Retrieval-Augmented Generation system built on LangGraph. Rather than a fixed retrieve-then-generate pipeline, the graph adaptively routes each query through relevance grading, query rewriting, hallucination checking, and constrained regeneration — deciding at each step whether the evidence is strong enough to proceed, or whether the query, the retrieval, or the answer itself needs correcting first.

## Current project status

The current retrieval schema uses the versioned `agentic-rag-hybrid-v2` index
with `structure_aware_v2` chunking. In the reviewed v1/v2 evaluation, v2 raised
relevant-evidence recall in the application's final five passages from 76% to
92% and improved final ranking MRR from 0.647 to 0.703 without a material
retrieval-latency increase. Structure-aware chunking is therefore fixed for the
next development stage while embeddings remain unchanged.

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
   |    +----+----+                             |
   |  relevant  irrelevant --------(retries left)+
   |    |
   +----+
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

- **Retrieval confidence gating** — a three-way policy (`generate` / `grade` / `rewrite_query`) decides whether an LLM relevance grader is necessary. Strong score distributions skip grading; any non-empty candidate set below the provisional score floor goes to semantic grading instead of a blind rewrite. Only missing or semantically irrelevant evidence triggers retrieval correction. An uncertain grader result proceeds to generation and the stricter grounding verifier remains the final acceptance gate. Rewrites receive the latest failure reason and stop with an abstention if they repeat an attempted query.
- **Hybrid retrieval** — dense (embedding) and sparse (BM25) retrieval run independently and are fused via Reciprocal Rank Fusion, then reranked with a cross-encoder for final relevance scoring. Within the existing five content slots, the first two RRF candidates are retained as a recall safeguard; cross-encoder scores still order the selected passages. `RERANK_CONTENT_RRF_RESERVE` defaults to `2` and can be set to `0` to use pure cross-encoder selection. This is a rank-slot rule, not a score cutoff. BM25 is fit per document at ingestion time, not globally, matching the document-scoped retrieval model.
- **Document-scoped retrieval** — every document gets a stable, content-derived ID (a hash of its bytes), so retrieval, BM25 encoding, and vector storage are all scoped to a specific document rather than the whole corpus. Re-ingestion upserts the current vectors, updates the document's BM25 state, then removes obsolete vectors for that document.
- **Whole-document overview chunks** — a summary generated at ingestion time and retrieved separately from content chunks, so structural questions ("what does this document cover") aren't left to chunk-level semantic search, which is the wrong granularity for that kind of question.
- **Corrective grounding handling** — a structured verifier distinguishes a grounded answer, insufficient evidence, and unsupported generation. Verification sees only the passages cited by the answer; missing or unknown labels and uncited factual list items route to correction without an LLM verdict. Evidence problems route back to retrieval and end in an explicit abstention when retries are exhausted; a relevance grade of irrelevant also abstains after the retrieval retry budget. Generation problems receive one constrained correction using the verifier's unsupported-claim list. Malformed verifier output fails closed instead of silently approving an answer.
- **Multi-provider LLM fallback** — requests fall through a configurable provider chain (Groq, Cerebras, NVIDIA NIM, OpenRouter, optionally AWS Bedrock), split into a "primary" tier (used only for final answer generation) and a "fast" tier (used for classification-style calls: grading, rewriting, hallucination checking) to control cost and rate-limit pressure.
- **Typed semantic routing** — the UI explicitly separates selected-document, general-model, and automatic source modes. A bounded fast-tier planner resolves conversational follow-ups and verification requests into a validated query plan; ambiguous automatic choices ask for clarification instead of silently mixing document evidence with model knowledge.
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
  retrieval/
    vectorstore.py              Dense + BM25 + RRF retrieval against Pinecone
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
listing, BM25 lookup, and deletion then operate only on the active index. This
allows the same source document to be evaluated independently in v1 and v2.
The registry also records the chunking strategy and rejects ingestion when an
index already contains a different strategy.

The v1 control uses `CHUNKING_STRATEGY=token_window_v1`: plain PDF extraction
followed by the existing 240-token windows with 40-token overlap. The v2
treatment uses `CHUNKING_STRATEGY=structure_aware_v2`: layout-preserving PDF
extraction, heading detection, sentence/bullet units, section-aware packing up
to the same 240-token embedding limit, and section/page-range metadata. Keep
the embedding model and the other retrieval settings fixed for the first v1
versus v2 comparison.

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

`start` reuses existing images and containers. The runner creates a temporary
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

The Docker image uses the CPU-only PyTorch wheel on Linux, matching its CPU
embedding and reranking runtime. Local Windows installs retain the CUDA wheel.
The Dockerfile keeps uv's package cache outside image layers, so later builds
can reuse downloads without shipping the cache in the image. A source change
still requires rebuilding the affected service once; routine starts reuse the
existing image. Pull requests and `main` run the locked Docker build, runtime
import check, baked-model offline load check, and offline unit suite in
`.github/workflows/offline.yml` without provider credentials or network access
during tests.

The image pins exact Hugging Face revisions for
`sentence-transformers/all-MiniLM-L6-v2` and
`cross-encoder/ms-marco-MiniLM-L6-v2`. The first image build downloads those
public model files into a reusable Docker layer. API startup and CI load them
with Hugging Face networking disabled, so a host cache is no longer required.
Changing either model or revision is an index migration and requires a new
versioned `PINECONE_INDEX_NAME` plus document re-ingestion.

For direct Compose use, `docker compose up -d --no-build` starts existing
images, but it does not mount the host model cache or log folder.

The frontend is available on port 8501, the API on port 8000, and PostgreSQL
is exposed to the host on port 5442. API startup warms the cross-encoder model,
so its first health check can take several minutes when the model cache is
empty.

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
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Vector store connection and retrieval schema version; use a name such as `agentic-rag-hybrid-v2` for migrations |
| `CHUNKING_STRATEGY` | `token_window_v1` for the baseline or `structure_aware_v2` for the section-aware treatment; one strategy per index |
| `POSTGRES_URL` | Conversation checkpointing + document registry |
| `EMBEDDING_MODEL`, `EMBEDDING_MODEL_REVISION` | Dense embedding model and immutable Hugging Face revision baked into Docker |
| `CROSS_ENCODER_MODEL`, `CROSS_ENCODER_MODEL_REVISION` | Reranker model and immutable Hugging Face revision baked into Docker |
| `MODEL_LOCAL_FILES_ONLY` | Prevent runtime model downloads; Docker sets this to `true` |
| `RETRIEVAL_MIN_TOP_SCORE` | Provisional absolute score floor for retrieval routing (`RETRIEVAL_STRONG_TOP_SCORE` is currently unused) |
| `CROSS_ENCODER_DEVICE` | `auto` (default), `cuda`, or `cpu` |
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

## API

- `POST /query` - `{question, session_id, document_id, source_mode}` returns the answer plus `answer_source`, `query_relationship`, `information_need_source`, `response_format`, grounding status, citations, and contexts
- `POST /ingest` - multipart file upload returns `{filename, document_id, index_name, chunking_strategy, chunks_indexed}` per file
- `GET /documents` — list of ingested documents and their metadata
- `GET /health` — liveness check
