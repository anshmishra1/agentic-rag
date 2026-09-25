# Agentic RAG

A production-grade, corrective Retrieval-Augmented Generation system built on LangGraph. Rather than a fixed retrieve-then-generate pipeline, the graph adaptively routes each query through relevance grading, query rewriting, hallucination checking, and constrained regeneration — deciding at each step whether the evidence is strong enough to proceed, or whether the query, the retrieval, or the answer itself needs correcting first.

## Architecture

```
contextualize_question
        |
        +-- control query -----------------------------> record_turn -> END
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

- **Retrieval confidence gating** — a three-way policy (`generate` / `grade` / `rewrite_query`) decides whether an LLM relevance grader is even necessary, based on an absolute score floor plus the relative shape of the score distribution. Strong evidence skips grading entirely; weak evidence skips straight to a rewrite instead of wasting a generation attempt.
- **Hybrid retrieval** — dense (embedding) and sparse (BM25) retrieval run independently and are fused via Reciprocal Rank Fusion, then reranked with a cross-encoder for final relevance scoring. BM25 is fit per document at ingestion time, not globally, matching the document-scoped retrieval model.
- **Document-scoped retrieval** — every document gets a stable, content-derived ID (a hash of its bytes), so retrieval, BM25 encoding, and vector storage are all scoped to a specific document rather than the whole corpus. Re-ingesting a document overwrites its existing vectors instead of duplicating them.
- **Whole-document overview chunks** — a summary generated at ingestion time and retrieved separately from content chunks, so structural questions ("what does this document cover") aren't left to chunk-level semantic search, which is the wrong granularity for that kind of question.
- **Corrective grounding handling** — a structured verifier distinguishes a grounded answer, insufficient evidence, and unsupported generation. Evidence problems route back to retrieval and end in an explicit abstention when retries are exhausted; generation problems receive one constrained correction using the verifier's unsupported-claim list. Malformed verifier output fails closed instead of silently approving an answer.
- **Multi-provider LLM fallback** — requests fall through a configurable provider chain (Groq, Cerebras, NVIDIA NIM, OpenRouter, optionally AWS Bedrock), split into a "primary" tier (used only for final answer generation) and a "fast" tier (used for classification-style calls: grading, rewriting, hallucination checking) to control cost and rate-limit pressure.
- **Conversation memory** — multi-turn context via a LangGraph Postgres checkpointer, with a per-turn query classifier distinguishing new questions, follow-ups, and control utterances (e.g. "thanks", "stop").
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
    conversation.py               Query-intent classification (new / follow-up / control)
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

Create the Pinecone index (must use `dotproduct` metric for hybrid search to work):

```bash
python scripts/create_hybrid_index.py
```

## Running locally

```bash
uv run uvicorn agentic_rag.api.main:app --reload      # backend, :8000
uv run streamlit run app/streamlit_app.py              # frontend, :8501
```

On Windows, `scripts/run_server.ps1` starts the backend with the project
virtual environment and writes versioned logs under `logs/`.

## Running with Docker Compose

After creating `.env`, build and start the API, Streamlit frontend, and local
PostgreSQL service:

```bash
docker compose up --build
```

The first command builds the images. For later runs with unchanged code, use
`docker compose start` to restart the same containers, or `docker compose up -d`
if they have not been created yet. Use `docker compose stop` when finished so
the containers and their downloaded model cache remain available. Rebuild with
`docker compose up --build` only after changing code or dependencies.

The frontend is available on port 8501, the API on port 8000, and PostgreSQL
is exposed to the host on port 5442. API startup warms the cross-encoder model,
so its first health check can take several minutes when the model cache is
empty.

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Purpose |
|---|---|
| `PROVIDER_ORDER` | Fallback order across LLM providers, e.g. `cerebras,groq,nvidia,openrouter` |
| `PRIMARY_LLM_MAX_TOKENS`, `FAST_LLM_MAX_TOKENS` | Output caps for primary and fast model calls (defaults: 4096 and 1024) |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Vector store connection |
| `POSTGRES_URL` | Conversation checkpointing + document registry |
| `RETRIEVAL_MIN_TOP_SCORE`, `RETRIEVAL_STRONG_TOP_SCORE` | Absolute score gates for the retrieval confidence policy |
| `CROSS_ENCODER_DEVICE` | `auto` (default), `cuda`, or `cpu` |
| `MAX_RETRIES` | Cap on rewrite/retry loop iterations |
| `DEBUG` | Verbose logging (candidate-level retrieval/rerank tables) |

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

The calibration input is a JSON list with `query`, 64-character `document_id`,
and boolean `should_match` on every row. Offline rows also need `top_score`.
After indexing the reviewed source documents, collect scores explicitly:

```bash
python -m agentic_rag.policies.calibrate_retrieval labeled_pairs.json --live --scores-output scores.json
python -m agentic_rag.policies.calibrate_retrieval scores.json
```

The first command reads PostgreSQL and queries Pinecone but does not call an
LLM; the second analyzes saved scores offline. It reports a candidate score
floor only when at least five examples in each class separate cleanly.
Validate any candidate on a separate holdout set before changing configuration.

## API

- `POST /query` — `{question, session_id, document_id}` → `{answer, grounded, answer_status, grounding_diagnosis, verification_exhausted, citations, contexts}`
- `POST /ingest` — multipart file upload → `{filename, document_id, chunks_indexed}` per file
- `GET /documents` — list of ingested documents and their metadata
- `GET /health` — liveness check
