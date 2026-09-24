# AGENTS.md — AgenticRAG

Read this first. Make small, verified, reversible changes. For history and past decisions, read `docs/CODEX_SESSION_LOG.md`.

## Project
Document-scoped corrective RAG app.
- FastAPI backend, Streamlit UI, LangGraph orchestration
- Pinecone hybrid retrieval (dense + per-document BM25, weighted RRF) → local HF cross-encoder rerank
- PostgreSQL: document registry, BM25 state, LangGraph checkpoints
- LLM fallback chain: Cerebras → Groq → NVIDIA → OpenRouter (Bedrock optional); primary tier = answers/overviews, fast tier = classify/grade/rewrite/verify
- Inputs: PDF (embedded text only, no OCR), PNG/JPEG (vision caption), MP3/WAV/M4A (Groq Whisper)

## Where things live
- Real entry point: `src/agentic_rag/api/main.py` (root `main.py` is a placeholder)
- UI: `app/streamlit_app.py`
- Graph: `src/agentic_rag/graph/{builder,edges,nodes,state}.py`
- Pure decision logic (keep nodes thin): `src/agentic_rag/policies/`
- Retrieval: `retrieval/` · Ingestion: `ingestion/` · Providers: `llm/provider.py`
- Tests: `tests/unit/` (offline, default). Older scripts directly under `tests/` are stale or need live services and are excluded from default pytest.
- Docs: `README.md`, `AGENTIC_RAG_HANDOFF.md`, `tests/README.md`, `docs/CODEX_SESSION_LOG.md`

## Commands (Windows / PowerShell)
```
$env:PYTHONDONTWRITEBYTECODE='1'; .\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```
Default `pytest` runs only the offline unit suite (38 tests at last count). If the `.venv` interpreter fails, report it. Do not install or upgrade dependencies without asking.

## Hard rules
- Never read, print, or modify secrets in `.env`. Known issue: it has `DEBUG=release`, which is invalid for the boolean `Settings.debug`. Do not "fix" it; unit tests isolate it in `tests/unit/conftest.py`.
- No live services during verification (Pinecone, PostgreSQL, LLM providers, model downloads) unless I explicitly ask.
- Git: check branch and `git status` before editing. Treat existing uncommitted changes as mine and preserve them.
- One concern per branch (`fix/`, `refactor/`, `test/`, `docs/`, `feature/`). Small commits with intent prefixes (`fix:`, `test:`, `refactor:`, `docs:`). Never mix fixes, refactors, and formatting.
- Review gate: after implementing and verifying a task, leave its changes uncommitted and keep the task branch checked out. Report the changed files and the exact diff command, then wait for my approval before staging or committing. Request separate authorization before pushing or opening a PR.
- Tests are executable regression contracts, not repository decoration. Run every relevant test that the available environment supports. If the environment blocks execution, report the exact blocker and do not describe the affected behavior as fully verified.
- Never push, open a PR, merge, delete branches, force-push, or `reset --hard` unless I ask. Never merge without my review.
- When source, tests, config, and docs disagree: source wins first; afterwards all four must agree.
- Before a non-trivial change, send a brief: problem, evidence in code, files affected, intended behavior, risks, verification plan. Wait for my approval.
- After a change, report: tests run, results, what was not tested, remaining limitations, whether docs match runtime.
- `nodes.py`/`builder.py` contain encoding-damaged docstring characters. Use small targeted patches, not large ones.

## Design invariants
- Retrieval is scoped to one selected document. The cross-encoder score is a ranking signal, not a calibrated probability, and the routing thresholds are placeholders until calibrated.
- Retrieval retries are bounded by `MAX_RETRIES` (default 2). Corrective regeneration runs at most once.
- Grounding verdicts are `grounded`, `insufficient_evidence`, `unsupported`:
  - `insufficient_evidence` → rewrite/retrieve while retries remain, then deterministic abstention
  - `unsupported` → one constrained regeneration using the listed unsupported claims
  - malformed verifier output → fail closed, never treated as grounded
  - a model refusal is an abstention, not "grounded"
- API exposes `answer_status` and `grounding_diagnosis`; Streamlit labels abstentions separately from failed verification.
- Do not overwrite the original question; keep query rewrites as separate attempts.
- Do not store observability data or full `Document` objects in durable checkpoint state.
- Keep `RAGState` flat: no nested state groups or subgraphs until a genuinely separable agentic unit exists. "State cleanup" means declaring every field that is used and removing dead or legacy ones, not restructuring.
- Add new state fields before removing legacy ones; remove only after all references are gone.
- Changing embedding model, chunking, sparse encoding, metadata, or vector-ID logic is an index migration (use a versioned index name).

## Known open issues
- Duplicate query-intent classifier: `nodes.py` shadows the policy version (misses e.g. `thanks!`)
- Re-ingestion overwrites Pinecone vectors but inserts duplicate registry rows
- Graph state migration incomplete (legacy flat fields mixed with typed state)
- No citations yet (`citations` declared but never populated)
- Packaging/deploy drift: Dockerfile copies missing `requirements.txt`; README references missing `.env.example` and misnamed scripts (`create_hybird_index.py` typo); old `create_index.py` makes a cosine index (incompatible with sparse hybrid); `scripts/run_server.ps1` cds into `scripts/`
- Stale legacy tests; no auth, upload limits, or rate limiting; logs may contain document text (treat as sensitive)

## Agreed order of work
1. Query-intent consolidation → 2. idempotent registry → 3. graph-state cleanup → 4. citations → 5. packaging/Docker repair → 6. retrieval calibration and evaluation baseline → 7. cloud (AWS), only after the above.

## Context hygiene
- Prefer `rg` and line ranges over dumping whole files.
- At the end of each task, append a short entry to `docs/CODEX_SESSION_LOG.md` (what changed, why, tests, commit/PR, next step).
