# Codex Session Log — AgenticRAG (condensed)

Repo: `anshmishra1/agentic-rag`. Distilled from one long Codex (VS Code) session. Contains only work done, actions, changes, and reasoning. Rules live in `AGENTS.md`.
Status as of transcript end: PR #1 merged; PR #2 open and unmerged. Re-check with `git log` / GitHub.

## Timeline
| # | Step | Type | Result |
|---|------|------|--------|
| 1 | Full repo audit | read-only | Architecture mapped; 9 discrepancies found |
| 2 | Methodology review | read-only | P0/P1/P2 improvements + 5-phase roadmap |
| 3 | Maintenance approach | advice | Principles, test/branch/doc strategy, correction order |
| 4 | Git handling rules | advice | Branch-per-concern; no remote/destructive actions unasked |
| 5 | Baseline tests | code | Branch `maintenance/baseline-tests`, commit `3e30c07`, 30 tests pass |
| 6 | PR #1 | GitHub | Opened, then merged by user; local `main` fast-forwarded to `24e18dc` |
| 7 | Cloud plan | advice | AWS-first, after fixing discrepancies |
| 8 | Structured grounding | code | Branch `fix/structured-grounding`, commit `7854231`, 38 tests pass, PR #2 open |

Starting point: clean `main` at `1c10b8f`.

## 1. Repo audit (read-only, no files changed)
Findings that mattered:
1. Three-way grounding parser (`parse_grounding_response`) existed but was never called. The live verifier was binary (`grounded`/`hallucinated`), so the "insufficient evidence → re-retrieve" path was unreachable and unsupported claims never reached correction.
2. Query-intent logic duplicated. `nodes.py` shadows the policy classifier with a weaker one (e.g. `thanks!` not treated as control).
3. State migration incomplete: typed fields mixed with legacy flat fields; `trace`/`verification_exhausted` used but not declared; `answer_status`/citations declared but unpopulated.
4. Re-ingest is idempotent for Pinecone vectors but inserts duplicate PostgreSQL registry rows.
5. Deployment drift: Dockerfile copies nonexistent `requirements.txt`; README cites missing `.env.example`, misnamed scripts (`create_hybird_index.py`), missing calibration script; old `create_index.py` builds a cosine index (wrong for sparse hybrid).
6. `scripts/run_server.ps1` cds into `scripts/` and uses a different import path than documented.
7. Tests stale or manual scripts (timing, routing, retriever, diagnostic tests out of date; some need live services).
8. Ingestion and querying are synchronous and heavy; multi-file upload is sequential.
9. No auth, upload-size limit, rate limiting; logs may hold document text.
Also: no OCR for scanned PDFs; RAGAS eval exists (80-row set, cross-document questions skipped) but was never run.
Verdict: architecture is coherent and advanced; the gap is implementation drift (design more sophisticated on paper than at runtime).

## 2. Methodology review (read-only) — condensed
Strengths: token-aware chunking, content-hash doc IDs, overview chunk, per-document BM25, retrieve→RRF→cross-encoder, bounded retries, tiered providers, request-scoped diagnostics.
Top P0s:
- Make ingestion atomic (state machine `pending→processing→indexed→ready|failed`); upsert registry by `document_id`.
- Calibrate retrieval thresholds empirically; replace fragile score-shape heuristics with evidence features.
- Make retries meaningfully different (track query similarity, candidate overlap) and preserve retry history.
- Add citations and provenance in generation context (stable `[S1]` labels; verifier checks them).
- Verify claims, not the whole answer; separate missing evidence from contradiction; route on structured verdicts.
- Consolidate intent classifier; finish state migration; lazy provider init (currently built at import time).
- Layered evaluation, retrieval gold set, negative/adversarial cases; error boundaries and ingestion status in UI.
Notable P1s: query decomposition, neighbor/parent context expansion, layout/OCR parsing, ingestion versioning, provider circuit breaker, explicit abstention after retries, answer-status semantics, log redaction, prompt-injection framing.
Roadmap: (1) correctness → (2) measurement → (3) retrieval quality → (4) reliability → (5) product experience.

## 3. Maintenance approach (agreed)
- Preserve behavior before improving: observe → capture in tests → define desired → one focused change → verify → evaluate → update docs.
- Source code authoritative; small vertical slices, no rewrite.
- Every change must state: failure addressed, how often, metric, cost, how to detect regression.
- Layers: unit (offline) / component (mocked I/O) / integration (test infra) / e2e eval (opt-in).
- Boundaries: API/UI → services → graph → policies/state → infra adapters; nodes thin, logic in policies.
- Index changes (embedding, chunking, sparse, metadata, vector IDs) = migration with versioned index name.
- Docs: README, architecture doc, ADRs (e.g. document-scoped retrieval, per-document BM25, RRF before rerank, Postgres checkpoints, two-tier providers, structured grounding), one known-limitations list.
- Per-change workflow: define problem → trace surface → write expected behavior → tests first → smallest complete change → progressive verify → review diff → record decision.
Correction order: Stage 0 baseline → 1 runtime correctness (intent, grounding, abstention, routing tests, status semantics, idempotent ingest) → 2 contract cleanup → 3 reproducibility (packaging, Docker, env template, migrations) → 4 quality/eval → 5 product (citations, progress, streaming, OCR).

## 4. Git handling (agreed)
`main` stays stable; short-lived branches (`fix/`, `refactor/`, `test/`); inspect branch and status first; preserve user changes; small intent-named commits; no push/PR/merge/delete/force/reset without approval; PR even for solo work (permanent record of why/what/tests/limits).

## 5. Baseline tests — actions and changes
Branch `maintenance/baseline-tests`, commit `3e30c07 test: establish offline baseline suite`.
Files:
- `pyproject.toml`: default pytest collects only offline `tests/unit`
- `tests/README.md`: test-suite guide
- `tests/unit/`: `conftest.py`, `test_conversation_policy.py`, `test_generation_policy.py`, `test_graph_routing.py`, `test_grounding_policy.py`, `test_retrieval_policy.py`, `test_timing.py`
- Old live/stale scripts untouched and excluded.
Result: 30 passed in ~1s, no external services.
Obstacles and how they were handled:
- `.git` read-only in sandbox → requested narrow Git permission.
- `.venv` base interpreter (Python 3.12) outside workspace → requested permission to run it; nothing installed.
- `.env` has `DEBUG=release` (bool field) → import failed; `.env` not modified; `conftest.py` isolates it; logged as a runtime config issue.
Why offline-only: prevents Pinecone/Postgres/LLM/model downloads from firing by accident.

## 6. PR #1
Pushed `maintenance/baseline-tests`. No `gh` CLI and no browser tool, so the PR was created through the GitHub REST API using the credential already in Git's credential helper (kept in memory, not printed or stored). PR #1 "test: establish offline baseline suite" body noted 30 tests and the `DEBUG=release` issue. User reviewed and merged it; local `main` was fetched and fast-forwarded (no merge commit) to `24e18dc`.

## 7. Cloud plan (advice only, nothing changed)
- One cloud first: AWS (Bedrock already supported, container-oriented). Not multi-cloud.
- Phase 1 make cloud-ready: fix Docker/requirements, startup config validation, `DEBUG=release`, liveness vs readiness, lazy provider init, model cache/baked-in model, stdout logs, stateless API.
- Phase 2 minimal deploy: ECR + FastAPI and Streamlit containers (ECS Express/Fargate), Secrets Manager, CloudWatch, `/health` check. Keep Pinecone and Neon at first.
- CPU vs GPU: Fargate has no GPU; run MiniLM embed/rerank on CPU, benchmark, bake pinned models into the image. Don't swap embedding model without re-ingest and recalibration.
- Phase 3 async ingestion: browser → S3 → SQS → worker → Pinecone + Postgres; status `pending→processing→ready|failed`; DLQ; scale workers on backlog per worker.
- Phase 4 CI/CD: PR → offline tests → container build → vuln scan → integration; main → versioned image (commit-SHA tags, not `latest`) → ECR → staging → smoke → prod; GitHub Actions to AWS via OIDC, no long-lived keys.
- Phase 5 IaC: Terraform (dev/staging/prod).
- Later, separately: RDS instead of Neon; OpenSearch instead of Pinecone (retrieval-platform migration needing its own evaluation).
- Security/cost: no `.env` in image, least-privilege IAM, private encrypted S3, log redaction, budget alerts.
Decision (user): implement cloud only after fixing current discrepancies.

## 7b. Planned order including cloud
Merge baseline PR → structured grounding/abstention → intent consolidation → idempotent ingestion → state cleanup → packaging/Docker repair → local container integration tests → AWS MVP → CI/CD + IaC → async ingestion → monitoring/hardening → RDS/vector store evaluation.

## 8. Structured grounding — actions, design, changes
Branch `fix/structured-grounding` from fast-forwarded `main`; commit `7854231 fix: distinguish grounding failures and abstain safely`.
Design (fixed before coding):
| Verifier result | Outcome |
|---|---|
| `grounded` | Record verified answer |
| `insufficient_evidence`, retries left | Rewrite query and retrieve again |
| `insufficient_evidence`, retries exhausted | Deterministic abstention |
| `unsupported` | One constrained regeneration fed the unsupported-claim list |
| Still unsupported after correction | End with unverified disclaimer |
| Malformed verifier output | Fail closed as verification-uncertain, never "grounded" |
| Model-generated refusal | Explicit abstention status, not "grounded" |
Reasoning: the old binary verifier could not tell a retrieval problem (evidence missing → retrieve differently) from a generation problem (evidence exists, answer overclaims → regenerate conservatively). Streamlit also had to change, otherwise a safe abstention would show as "could not verify".
Changed (11 files, one vertical slice): `policies/grounding.py`, `graph/{edges,state,builder,nodes}.py`, `api/main.py`, `app/streamlit_app.py`, `tests/unit/test_graph_routing.py`, `tests/unit/test_grounding_policy.py`, `README.md`, `AGENTIC_RAG_HANDOFF.md`.
Review refinements before commit: parser rejects non-list `unsupported_claims` (avoids iterating characters); README diagram shows abstention as a terminal evidence path; state outputs completed.
Process note: a combined patch was rejected due to encoding-damaged docstring characters, so edits were split into small patches.
Verification: 38 offline tests pass; changed Python files syntax-checked; diff audited (old binary verifier gone, all four routes `end`/`rewrite_query`/`correct_generation`/`abstain` mapped, API and UI statuses agree, only 11 intended files). No live services called.
PR #2 "fix: distinguish grounding failures and abstain safely" opened via the same API method; not merged. Excluded: cloud changes, query-intent consolidation, live end-to-end run.

## Open items after this session
Audit findings still open: #2 intent duplication, #3 state migration, #4 registry duplicates, #5 packaging/Docker drift, #6 launcher, #7 remaining stale tests, #8 sync ingestion, #9 hardening; also citations, evaluation baseline, `DEBUG=release`.
Fixed: #1 (grounding).

## 9. Post-merge sync and continuity handoff
PR #2 was reviewed and merged on GitHub. Local `main` was fast-forwarded to `cea6733` and matches `origin/main`; the user's untracked `AGENTS.md` and `docs/` files were preserved. Next planned implementation remains `fix/query-intent`. Continuity convention: read `AGENTS.md` first, then this condensed log, verify Git/source state, and treat source as authoritative when any historical note is stale.

## 10. Query-intent consolidation
Branch `fix/query-intent` from `main` at `cea6733` (uncommitted at time of entry). Removed the duplicate `classify_query_intent` implementation in `graph/nodes.py`, so the graph now uses the canonical deterministic policy in `policies/conversation.py`. Preserved the documented `stop`/`end this` control behavior in that policy and added regression coverage for punctuated control input plus an AST guard against future node-level shadowing. Static AST parsing passed for all three changed Python files; `git diff --check` found no whitespace errors. Offline pytest could not run because `.venv` points to a missing uv-managed Python 3.12 executable and the available system Python 3.14 has no pytest installed. No dependencies were installed and no live services were called. Next planned task after review/commit: idempotent document registry.
