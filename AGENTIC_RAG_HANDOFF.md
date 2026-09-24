# Agentic RAG — Project Handoff (for a new Claude conversation)

Upload this file at the start of a new chat instead of re-explaining the project. It reflects the actual current architecture, confirmed against real application logs, not assumptions.

## What this is
A LangGraph-orchestrated corrective RAG application — PDF/image/audio ingestion, hybrid retrieval, adaptive routing, hallucination correction. Built as Ansh's primary portfolio/interview piece. Repo: `anshmishra1/AgenticRAG` (also a fresh empty repo `anshmishra1/agentic-rag` was set up for a clean push). Runs locally on Windows, GPU-accelerated (RTX 2050, CUDA 13.0, torch 2.13).

## IMPORTANT: two divergent project histories exist
A separately-maintained `PROJECT.md` handoff doc (written for local-Ollama/other-tool sessions) describes a **simpler** architecture with no hybrid search, no cross-encoder, no `correct_generation` node — and explicitly lists those as "not yet done" / "don't add them just because they sound advanced." **That document is stale relative to what's actually running.** A real application log from 2026-09-17 confirms the advanced architecture below (dense+sparse query representation, `route_after_retrieval_assessment`, `grounding_diagnosis`/`correction_attempted`) is the live one. Trust logs/source over either handoff doc when they conflict.

## Current architecture (confirmed live)
```
contextualize_question → (control? → record_turn/END) → retrieve
retrieve → assess_retrieval (3-way: generate | grade_documents | rewrite_query)
grade_documents → (relevant → generate) | (irrelevant → rewrite_query, capped by max_retries)
rewrite_query → retrieve (loop)
generate → check_hallucination
check_hallucination → grounded→END | insufficient_evidence→rewrite_query (or abstain when retries are exhausted) | unsupported→correct_generation
correct_generation → check_hallucination (one more pass, single correction budget)
```
- **Retrieval:** Pinecone (dotproduct metric index, required for hybrid), dense + BM25 sparse retrieved separately, fused via RRF, then cross-encoder reranked (ms-marco-MiniLM, CUDA-accelerated). Overview chunks (whole-doc summaries generated at ingest) retrieved separately from content chunks for structural questions.
- **Document scoping:** every doc gets a SHA-256 content-hash `document_id`; retrieval/BM25/vectors are scoped to it; re-ingestion overwrites rather than duplicates (deterministic vector IDs, content-hash based).
- **Citations:** generation context uses deterministic per-answer labels (`[S1]`, `[S2]`, ...); valid labels referenced by the final answer are returned by the API and displayed in Streamlit with filename/page or chunk-type provenance.
- **Retrieval confidence gate** (`policies/retrieval.py`): absolute score floor + relative shape (top/mean ratio, gap ratio) + overview-dominance special case → decides whether to skip the LLM grader entirely.
- **LLM providers:** tiered fallback chain (fast tier for classification/grading/rewriting, primary tier for final generation only), across Groq/Cerebras/NVIDIA/OpenRouter/Bedrock(optional), configurable order.
- **Conversation memory:** LangGraph Postgres checkpointer (Neon), thread_id-scoped.
- **Grounding verification:** structured JSON verdicts distinguish grounded answers, insufficient evidence, and unsupported generation. Unsupported-claim text is passed into one constrained correction; exhausted evidence retries produce a deterministic abstention rather than speculative generation.
- **Observability:** fixed — no more unbounded `trace` accumulation in checkpointed state (was growing forever, printing the whole session's history every turn); now logger-based, with per-request + session-level PerformanceTracker via contextvars (not a module global anymore).

## Key files
`graph/{state,nodes,edges,builder}.py`, `retrieval/{vectorstore,reranker,sparse}.py`, `policies/{retrieval,generation,conversation,grounding,calibrate_retrieval}.py`, `ingestion/{loaders,chunking,pipeline,registry}.py`, `llm/provider.py`, `api/main.py`, `app/streamlit_app.py`, `core/timing.py`.

## Real bugs fixed this project (don't re-litigate these)
Duplicate Pinecone vectors (was: random UUID vector IDs → fixed: content-hash IDs), overview chunk buried by absolute-floor check running before overview-dominance check, refusal answers ("I don't know") wrongly flagged hallucinated, `rewrite_query` losing half of compound questions because it only saw the last attempt not the original question, `classify_query_intent` shadowed-duplicate-function bug, unbounded `trace` state growth, module-global (not request-scoped) performance tracker, mislabeled "RRF" scores that were actually raw Pinecone hybrid scores.

## Open/unresolved right now
- **Live bug, just found (2026-09-17 log + Streamlit screenshot):** the query-intent classifier misclassifies "Can you elaborate?" / "Can you elaborate on X?" as a control/conversation-ending utterance (same bucket as "thank you"), routing straight to `record_turn` and skipping retrieval+generation entirely — returns a generic "You're welcome, ask another question" for what should be a real follow-up. Root cause not yet found — need `query_classifier.py` (structured-output LLM classifier, not the old regex version) to diagnose. Related cosmetic bug: control-routed turns show "Could not fully verify against context," which is misleading since no generation/check ever ran.
- RAGAS evaluation harness exists but has never actually been run — still effectively dead code.
- No OCR path for scanned/image-only PDFs (PyPDFLoader only extracts embedded text).
- Streaming responses + UI status indicator during retrieval — planned in detail, not implemented.
- Agentic tool-use (web search, calculator, RAG-as-tool) — not started; corrective RAG loop is the current "agentic" ceiling.
- A transient Pinecone DNS resolution error appeared once in the latest log during upsert — likely transient, watch for recurrence.

## User's standing preference
Show changed/new file contents directly in chat rather than re-zipping the whole project; only zip for the initial scaffold or when explicitly asked.
