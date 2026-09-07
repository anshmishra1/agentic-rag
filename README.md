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
   +----+------------------+------------------+
   |                        |                  |
grounded            insufficient_evidence   unsupported
   |                (retries left)              |
   v                        |                   v
record_turn -> END    rewrite_query      correct_generation
                                                 |
                                                 v
                                        check_hallucination
                                          (one verification pass)
```

**The core idea:** every routing decision is made by the cheapest mechanism that can be trusted for that decision — a deterministic check where the signal is reliable, a fast-tier LLM call where judgment is needed but cheaply, a full LLM call only where quality genuinely matters (final answer generation).

## Key features

- **Retrieval confidence gating** — a three-way policy (`generate` / `grade` / `rewrite_query`) decides whether an LLM relevance grader is even necessary, based on an absolute score floor plus the relative shape of the score distribution. Strong evidence skips grading entirely; weak evidence skips straight to a rewrite instead of wasting a generation attempt.
- **Hybrid retrieval** — dense (embedding) and sparse (BM25) retrieval run independently and are fused via Reciprocal Rank Fusion, then reranked with a cross-encoder for final relevance scoring. BM25 is fit per document at ingestion time, not globally, matching the document-scoped retrieval model.
- **Document-scoped retrieval** — every document gets a stable, content-derived ID (a hash of its bytes), so retrieval, BM25 encoding, and vector storage are all scoped to a specific document rather than the whole corpus. Re-ingesting a document overwrites its existing vectors instead of duplicating them.
- **Whole-document overview chunks** — a summary generated at ingestion time and retrieved separately from content chunks, so structural questions ("what does this document cover") aren't left to chunk-level semantic search, which is the wrong granularity for that kind of question.
- **Corrective hallucination handling** — a failed grounding check is diagnosed as either an *evidence problem* (routes back to retrieval) or a *generation problem* (routes to a constrained regeneration pass), rather than blindly retrying the same step. A single correction budget prevents infinite loops, and an explicit `verification_exhausted` flag is surfaced to the caller rather than silently serving an unverified answer as if it were fine.
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
  calibrate_retrieval.py         Retrieval threshold calibration runner
```

## Setup

```bash
cp .env.example .env   # fill in provider keys, Pinecone key, Postgres URL
pip install -r requirements.txt
```

Create the Pinecone index (must use `dotproduct` metric for hybrid search to work):

```bash
python scripts/create_hybrid_index.py
```

## Running locally

```bash
uvicorn agentic_rag.api.main:app --reload      # backend, :8000
streamlit run app/streamlit_app.py              # frontend, :8501
```

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Purpose |
|---|---|
| `PROVIDER_ORDER` | Fallback order across LLM providers, e.g. `cerebras,groq,nvidia,openrouter` |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Vector store connection |
| `POSTGRES_URL` | Conversation checkpointing + document registry |
| `RETRIEVAL_MIN_TOP_SCORE`, `RETRIEVAL_STRONG_TOP_SCORE` | Absolute score gates for the retrieval confidence policy |
| `CROSS_ENCODER_DEVICE` | `auto` (default), `cuda`, or `cpu` |
| `MAX_RETRIES` | Cap on rewrite/retry loop iterations |
| `DEBUG` | Verbose logging (candidate-level retrieval/rerank tables) |

Retrieval thresholds are calibrated empirically — see `scripts/calibrate_retrieval.py` — rather than chosen arbitrarily; they should be re-run whenever the retrieval or reranking mechanism changes, since the calibration is specific to whatever scoring signal is currently in use.

## API

- `POST /query` — `{question, session_id, document_id}` → `{answer, grounded, verification_exhausted}`
- `POST /ingest` — multipart file upload → `{filename, document_id, chunks_indexed}` per file
- `GET /documents` — list of ingested documents and their metadata
- `GET /health` — liveness check
