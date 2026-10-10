# AgenticRAG project progress

Updated: 2026-10-09

This document is the concise project status. Detailed chronological decisions
remain in `docs/CODEX_SESSION_LOG.md`; evaluation methods and measurements live
in the linked reports.

## Delivered foundations

- Consolidated query-intent classification and bounded retry behavior.
- Added structured grounding outcomes: `grounded`, `insufficient_evidence`, and
  `unsupported`, with deterministic abstention after exhausted evidence retries.
- Made document registry writes idempotent and index-aware.
- Completed the flat graph-state contract and added answer citations.
- Repaired packaging and Docker startup, reduced the application image, and
  baked pinned embedding and cross-encoder models into the image.
- Added a reusable local Docker runner and persistent run logs.
- Added versioned retrieval indexes so chunking, embedding, sparse encoding,
  metadata, and vector-ID migrations can be evaluated without corrupting the
  previous retrieval schema.
- Added reproducible, source-reviewed retrieval evaluation datasets and reports.

## Semantic answer-source routing (local review)

The former phrase-based new-question/follow-up classifier has been replaced by
a constrained, typed query plan inside the existing LangGraph flow. The UI now
offers Selected document, General model, and Auto modes. The plan records the
answer source, conversational relationship, requested response format, and a
standalone retrieval query. Ambiguous Auto requests ask for clarification.

Document answers continue through retrieval, citations, correction, and
grounding. General-model answers skip Pinecone and document verification and
are labeled explicitly in both the API and UI. Closed controls remain
deterministic. Shared generation instructions require valid LaTeX, symbol
definitions, a plain-language explanation for formulas, and fenced text for
requested textual flowcharts.

The design and its boundaries are documented in
`docs/QUERY_ROUTING_DESIGN.md`. Forty-three focused tests and 147 additional
supported offline tests pass. Three existing chunking tests remain blocked on
the host because its cache lacks the pinned Hugging Face tokenizer that is
baked into the Docker image. PR #31 merged the router at `2407a13`; its bounded
live validation remains pending. No live providers, Pinecone, PostgreSQL, or
`.env` were used for the implementation verification.

## Structure-aware chunking v2

The v1 control uses plain PDF extraction followed by 240-token windows with
40-token overlap. The v2 treatment keeps the same chunk limit, overlap setting,
embedding model, sparse retrieval, reranker, and questions, while changing only
PDF extraction and chunk construction:

- layout-preserving PDF text extraction;
- conservative heading and bullet recognition;
- sentence-level units packed within section boundaries;
- section and page-range metadata;
- duplicate-text removal before content-addressed vector IDs are generated.

The PostgreSQL registry records the chunking strategy and prevents different
strategies from sharing one Pinecone index.

## Measured v1/v2 result

The broad evaluation used eight document styles. Its 30 new reviewed cases
contained 20 ordinary answerable questions, five false-premise questions, and
five unanswerable questions. Twenty-five source-supported cases were evaluated
against each index through Pinecone, document-specific BM25, weighted RRF, and
the local cross-encoder.

| Final retrieval metric | v1 | v2 |
|---|---:|---:|
| Reviewed evidence in final five passages | 19/25 (76%) | 23/25 (92%) |
| Reviewed evidence in top three | 17/25 (68%) | 20/25 (80%) |
| Mean reciprocal rank | 0.647 | 0.703 |

Six matched full-API cases increased reviewed evidence in final context from
1/6 with v1 to 5/6 with v2. Strict answered-and-grounded outcomes increased
from 1/6 to 3/6. The strict gold metric is conservative because an answer can be
supported by an alternate passage outside the manually anchored gold set.

These results support adopting `structure_aware_v2` for the next local stage.
They do not establish that every query improved: v1 retained one additional
top-one hit, and difficult evidence can still be removed during final selection.

Detailed evidence: `docs/CHUNKING_V2_BROAD_EVALUATION.md`.

## Current repair queue

### Answer routing reliability (local review checkpoint)

The latest manual run proved that response-format classification was not the
main cause of missing output: a FlashAttention flowchart was planned,
retrieved, and generated, then discarded after an insufficient-evidence
verdict and an empty Groq rewrite. The repair now semantically grades any
non-empty below-floor retrieval, treats malformed grading as `uncertain`, and
lets the citation-aware grounding verifier make the final acceptance decision.
Groq short decision tasks use low reasoning effort and bounded task-specific
caps; required empty content triggers provider fallback, and hidden Groq SDK
retries are disabled.

First-turn format detection, retrieval-query cleanup, cited ASCII diagrams,
and guarded prior-question reuse for format-only follow-ups cover the flowchart
and LaTeX paths without adding per-format graph branches. The plan now records
whether execution uses the current turn or the previous information need; both
the model plan and a narrow full-message grammar must agree before the prior
question can replace the current one. A request to find a document diagram,
table, or equation remains a current evidence request. Auto routing now receives
the selected document filename from the index-scoped registry instead of only
a boolean document flag.

Citation normalization now accepts alternate source brackets, optional line
suffixes, and invisible Unicode formatting characters inside markers. Exact
Start/End/Yes/No labels are exempt only inside fenced text diagrams; factual
nodes remain checked. Generation and correction explicitly avoid unsupported
author motivation, causality, maximality, and claims about what the whole
document does not contain.

The 2026-10-08 eleven-question run is frozen as a sanitized acceptance contract
with route, query-lineage, citation, status, retry, correction, and forbidden-
claim gates. The method is documented in
`docs/V2_ANSWER_RELIABILITY_ACCEPTANCE.md`; no passage or provider-prompt text is
stored in the committed manifest.

The reliability-focused suite passes 65 tests, and the complete offline suite
passes 180 tests with model networking disabled. One existing Starlette
dependency deprecation warning remains. A single candidate image built from
this worktree passed installed-package imports, CPU-only PyTorch, baked
embedding/reranker model verification, baked NLTK verification, FastAPI import,
and the same 180-test suite with container networking disabled. Its 3.92 GB
logical size shares 3.919 GB with the existing local image and adds about 643 KB
of unique image data. Infrastructure startup and live acceptance remain unrun.
No live service was contacted.

### 1. Normalize generated citations

The model sometimes returns line-qualified markers such as
`【S1†L7-L9】`. The application currently recognizes canonical `[S1]` and the
unqualified alternate form, so a supported answer can lose its citations and
be routed to correction. Normalize known provider variations to `[S#]`, then
validate every label against the supplied source catalog. This boundary should
remain provider-independent.

### 2. Classify abstentions reliably

An evidence-limitation answer beginning with a typographic `I don’t know` was
safe but labeled `answered` and `grounded`. Normalize Unicode punctuation and
recognize explicit evidence-based abstentions without relying on the existing
short response-length limit. Deterministic status checks should run before the
semantic grounding verifier.

### 3. Improve final evidence selection

For the DPO partition case, v2 recovered the reviewed passage into the candidate
pool, but the cross-encoder ranked it tenth and the five-passage context dropped
it. Evaluate blended RRF/cross-encoder ordering, score-margin selection, section
or neighbor expansion, and diversity constraints against the saved labels. Do
not tune cross-encoder scores as if they were calibrated probabilities.

### 4. Make runtime budgets reproducible

- Disable SDK-level provider retries so the application retry budget is the
  single authoritative limit.
- NLTK `punkt_tab` and `stopwords` are now provisioned during the image build
  and verified in the network-disabled CI runtime gate. The reviewed repair is
  committed on `fix/nltk-runtime-assets`; Docker/CI image validation is pending.
- Prefer direct content citations over overview citations when both support the
  answer.

## Verification gates

1. Repair citation normalization and abstention classification with offline
   regression tests.
2. Rerun only the vLLM block-size, Chinchilla false-premise, and unanswerable
   vLLM BLEU cases through the full API.
3. Repair final passage selection against the saved DPO partition and vLLM
   false-premise cases, then evaluate the complete reviewed retrieval set.
4. Keep embeddings fixed until these correctness defects are closed. Compare
   alternative embedding or reranker models afterward using the same labels.
5. Continue with CI/CD, AWS deployment, OCR and richer parsing, and multimodal
   expansion after the retrieval and answer-contract gates pass.

## Evaluation artifacts

- `docs/CHUNKING_PILOT.md` - reproducible labeling and comparison workflow.
- `docs/CHUNKING_V1_V2_RETRIEVAL_REPORT.md` - initial three-document comparison.
- `docs/CHUNKING_V2_BROAD_EVALUATION.md` - broad retrieval and full-API evidence.
- `eval_set/chunking_broad_reviewed_cases.json` - reviewed broad cases.
- `eval_set/chunking_broad_v1_pairs.json` - v1 chunk labels.
- `eval_set/chunking_broad_v2_pairs.json` - v2 chunk labels.

The full response payloads and extracted passage catalogs remain under ignored
`logs/evaluation/` because they can contain document text.
