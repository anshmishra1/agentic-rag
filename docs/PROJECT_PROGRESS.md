# AgenticRAG project progress

Updated: 2026-10-16

This document is the concise project status. Detailed chronological decisions
remain in `docs/CODEX_SESSION_LOG.md`; evaluation methods and measurements live
in the linked reports.

## Journey at a glance

| Stage | Problem addressed | Result and evidence |
|---|---|---|
| Runtime correctness | Duplicated intent logic, incomplete graph state, unsafe grounding routes, duplicate registry rows | One canonical intent policy, declared flat state, structured grounding/correction/abstention, and idempotent index-aware registry writes |
| Reproducible local runtime | Packaging drift, missing model assets, slow rebuilds, first-query downloads, oversized images | Locked Docker builds, baked Hugging Face and NLTK assets, shared API/frontend image, GPU/CPU targets, reusable local runner, and CI offline gates |
| Retrieval measurement | Stale Pinecone vectors, missing passage labels, unmeasured reranking, provisional thresholds | Stale vectors removed after a hash-locked audit; reviewed labels and reports added; Ettin plus the RRF recall guard improved measured ranking while thresholds remain explicitly provisional |
| V2 evidence quality | Weak chunk boundaries, citation variants, unsafe answer verdicts, repeated rewrites, format and follow-up routing failures | Structure-aware V2 chunking improved reviewed final-five recall from 76% to 92%; citations, retry novelty, semantic routing, verifier budgets, and answer contracts were repaired and regression tested |
| V3 evidence model | V2 could not represent figures, tables, formulas, OCR, or exact artifact lineage | Typed artifacts, search records, source locators, grounding lineage, stable identities, and a content-addressed asset boundary were implemented |
| Parser selection | Full Docling was accurate but too slow; vanilla PyMuPDF was fast but missed OCR and structured evidence | A fixed ten-case bakeoff selected PyMuPDF4LLM/Layout plus conditional RapidOCR: 54/54 screening checks, deterministic IDs, and about 14x lower mean case time than full CPU Docling |
| V3 packaged boundary | Parser and model drift, GPU startup defects, partial cross-store writes | Exact parser/model versions are baked into the promoted image; PostgreSQL manifests, isolated V3 vector writing, and lazy `POST /v3/ingest` coordination are implemented |
| Current checkpoint | Needed a token-safe, evidence-aware V3 release candidate and proof that a required visual artifact survives final ranking | PR #39 merged; seven hash-verified documents are ready under the new schema with PostgreSQL/Pinecone agreement on 2,208 records, and the former FlashAttention Figure 1 miss now returns cited page-2 text plus figure evidence with an answered, grounded verdict |
| V3 acceptance definition | Prior manual prompts mixed retrieval, generation, and routing failures and did not cover realistic ambiguity consistently | A hash-locked, source-reviewed 49-case suite now spans seven documents, seven controlled/natural pairs, seven negative cases, and an 18-case bounded end-to-end core |

V2 remains the stable comparison baseline. V3 is now a connected release
candidate rather than only a storage/parser prototype: its token-safe corpus is
freshly indexed, the targeted visual-evidence regression passed through the
public API, and the Streamlit container startup path is repaired. V3 remains
opt-in until the broader manual release-candidate run records routing, answer,
citation, negative-query, format, and latency outcomes across the reviewed
corpus.

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

## Multimodal retrieval V3 (release-candidate checkpoint)

V3 implementation has started with the storage-neutral contracts required to
keep source artifacts separate from searchable representations. The new model
defines stable artifacts for text blocks, figures, tables, formulas, page
images, standalone images, and audio segments; searchable records for native
text, OCR, captions, table Markdown, formula LaTeX, transcripts, and page
summaries; explicit grounding lineage; and additive answer-facing evidence
references.

A complete retrieval-schema fingerprint now covers parser identity/options,
chunking, representation contract, embedding identity/dimension, sparse
encoding, vector-ID rules, and an optional visual embedding. A path-safe,
content-addressed local asset store provides the development boundary that can
later be replaced by S3. A compatibility adapter can wrap current V2 chunks
without changing their text.

The second work package now has a common parser adapter/run contract, a
hash-pinned ten-case corpus across six exact PDFs, pure scoring and repeatability
checks, and executable audit and Docling-run CLIs. It covers multi-column text,
figures, tables, formulas, algorithm blocks, and a deterministic image-only OCR
fixture. The audit confirms that all six expected files match their recorded
SHA-256 digests.

A disposable Docling 2.129.0 image now embeds the selected layout, table,
formula, and RapidOCR models and runs without networking. All ten CPU cases
parsed successfully. Nine unchanged cases met 49/49 checks; review found that
the LoRA expectation targeted the wrong page, and its corrected focused run met
5/5 checks. Representative figure crops, tables, formulas, and the image-only
OCR page were visually reviewed. Native and OCR provenance is now separated:
the LoRA page produced 11 native text records plus 2 OCR-only figure labels,
while the scanned fixture produced 5 OCR text records and no native text.

The measured full run took 475.3 seconds, with a 3.33 GiB process peak. Formula
heavy FlashAttention and DPO pages dominated at about 196 and 133 seconds. A
bounded CUDA target on the 4 GiB RTX 2050 reduced those cases to 42.15 and 32.25
seconds, a 4.66x and 4.11x speedup, while meeting all 10 checks and reserving at
most 1,948 MiB CUDA memory. A repeated FlashAttention run reproduced all 31
artifact IDs and 30 search-record IDs. One artifact differed between the older
CPU and CUDA outputs, so the selected accelerator is now part of parser
provenance and the retrieval-schema fingerprint. This prevents mixed CPU/GPU
records in one V3 index.

The Docling GPU candidate established that the 4 GiB RTX 2050 could safely
accelerate local inference, but Docling is no longer packaged in the selected
application image. The production GPU target now uses CUDA for MiniLM embedding
and Ettin reranking; PyMuPDF Layout, RapidOCR, storage, and ordinary application
work remain on CPU. Only the API container receives GPU access. Local inference
uses a process lock to prevent overlapping allocations on the 4 GiB device, and
CI retains a smaller CPU target without creating a second retained local image.
A Windows path overflow found by the parser evaluation was repaired by making
physical asset keys shorter and opaque; full identity remains in the contracts,
and the longest verified path is 216 characters.

The PyMuPDF 1.28.2 control is now implemented behind the same contracts and was
run twice across all ten cases with networking disabled. Both runs met 42/54
screening checks and reproduced every artifact and search-record ID. They
averaged 1.24 seconds per case, 13.75 seconds wall time, and at most 168.2 MiB
process memory. Against the Docling CPU evidence this is about 37 times faster
per case and about 20 times lighter, but the misses cover all standalone formula
artifacts, OCR text, and several vector figures. Visual review also found tiny
decorative image blocks labeled as figures and diagram regions proposed as
tables.

PyMuPDF4LLM 1.28.2 plus PyMuPDF Layout 1.28.2 and RapidOCR 3.9.2 is now
implemented as a third adapter. It emits ordered text and headings, structured
table Markdown, vector and raster figure crops, formula-region crops, clean page
images, and native-versus-OCR lineage. Two final network-disabled runs each met
54/54 checks without adapter warnings and reproduced every artifact and search
record ID. They averaged 3.33 seconds per case, 36.77 seconds wall time, and at
most 800.1 MiB process memory. This is about 14 times faster per case and 4.2
times lighter than the full Docling CPU run.

The implementation repaired three parser-boundary defects found by real
evaluation: missing spaces between coordinate-separated spans, line-break
hyphenation, and doubled evidence images caused by rendering the in-memory PDF
after OCR inserted text. Layout/OCR parsing now uses a separate document object
from page and crop rendering. Representative architecture diagrams, formula
crops, the OCR fixture, and its table and diagram regions were visually reviewed
after the fix.

The measured decision is PyMuPDF Layout as the V3 integration candidate.
Docling remains a measured fallback for a future reviewed miss instead of an
unproven default triage layer. Formula regions are citable but are not claimed
to be normalized LaTeX.

The first production-boundary package now pins PyMuPDF, PyMuPDF4LLM, and
PyMuPDF Layout 1.28.2 plus RapidOCR 3.9.2 in `pyproject.toml` and `uv.lock`.
The parser rejects any package-version drift, exposes reviewed render/OCR
settings, and has a safe identity check that runs with networking disabled.
The Docker definition removes Docling and its model-download layers from the
selected application runtime; CUDA remains for MiniLM embedding and Ettin
reranking, while PDF layout and OCR remain on CPU. Ettin's CUDA path invokes a
Triton kernel during API warmup, so the GPU target includes `gcc` and `libc6-dev`
for Triton's runtime launcher compilation and returns to the non-root application
user before Python installation and execution.

The promoted GPU image passed exact parser and baked-model verification, CUDA
visibility on the RTX 2050, an offline API import with placeholder constructor
values, Compose validation without resolving the private environment file, and
the complete 253-test offline suite. An internal-only PostgreSQL/API/Streamlit
smoke returned healthy endpoints with no OOM. The only test warning remains the
existing Starlette `BlockingPortal` dependency deprecation. Detailed parser
evidence is in `docs/V3_PARSER_COMPARISON.md`.

The Docling GPU feasibility question is answered for the two pages that
dominated CPU time. The consolidated image also passed a co-residency run with
embedding and Ettin already loaded: the FlashAttention formula case met 5/5
checks and peak reserved CUDA memory was 2,188 MiB.

The same 15 reviewed reranker cases preserved every Ettin support rank, Hit@5
of 12/12, and MRR of 0.840 while mean time for 15 candidates fell from 0.850
seconds on CPU to 0.142 seconds on CUDA, about 6.0x faster. The decisive V2 DPO
case remained rank 4. On the same 225 saved passages, batched MiniLM embedding
fell from 7.73 to 1.34 seconds, about 5.8x faster. A 15-query batch was slightly
slower on CUDA (0.079 versus 0.054 seconds), showing that small-query launch
overhead remains and that GPU benefit comes mainly from batched ingestion and
reranking.

The current image `sha256:0085357025005521f2072591d527b7d4ae498782a63d81dd65a56a83293230a3`
is 12.80 GB logical and is tagged `agentic-rag-app:local`. The obsolete 15.32 GB
Docling checkpoint and temporary candidate tag were removed. Docker retains only
the application image, `postgres:16`, and the named PostgreSQL data volume. The
27.85 GB BuildKit cache is intentionally retained because it contains the costly
locked dependencies, models, and CUDA archives that make subsequent rebuilds
fast; D: still had 194.8 GiB free after cleanup. Source and UI files are copied
after the compiler and CUDA layers, and the final project-only install does not
mutate the shared dependency cache. An unchanged image rebuild therefore reused
every layer and completed in 2.89 seconds instead of repeating the large CUDA
export.

The V3 persistence and indexing boundary is now implemented with a separate,
opt-in API boundary. `POST /v3/ingest` lazily constructs the selected parser,
asset store, transactional manifest repository, shared embedding runtime,
document-local BM25 encoder, and isolated V3 index writer. The default
`POST /ingest` route remains V2 and cannot initialize the V3 runtime. PostgreSQL
DDL and a transactional repository persist the
complete retrieval schema, document manifest, artifact metadata, parent-child
page relationships, locators, searchable records, BM25 state, and explicit
`parsed`/`ready` status. Re-ingestion replaces one document manifest atomically;
schema mismatches fail before deletion, and only ready manifests are exposed to
future retrieval. Binary page and region content remains in the
content-addressed `v3_assets` volume rather than PostgreSQL or Pinecone.

The isolated V3 writer embeds only `SearchRecord` text, validates the exact
schema dimension and parser identity, writes deterministic document-prefixed
vector IDs with compact artifact provenance, and prunes obsolete IDs only after
all current batches succeed. Its coordinator marks a manifest ready only after
the vector write completes. It does not import the V2 vector store or global
settings, and the current V2 application path is unchanged. Forty-four focused
tests and the complete 265-test offline suite passed at the preceding persistence
checkpoint. Seven regression tests were added for the new runtime and API
boundary. The focused ingestion/API/persistence selection passed 24 tests, and
the complete network-disabled offline suite passed 272 tests with the existing
Starlette deprecation warning. AST, diff, and whitespace checks also pass. A
disposable PostgreSQL 16 smoke then created the V3 schema, persisted two linked
artifacts and one search record, kept the parsed manifest hidden, promoted it to
ready, and rejected a different retrieval schema without replacing the ready
document. Pinecone V3 index creation and live ingestion were not run.

The query boundary now validates a ready V3 manifest before retrieval, scopes
dense and sparse candidates to the selected document and retrieval schema,
fuses them with weighted RRF, and reuses the existing Ettin reranker and graph.
The API keeps V2 and V3 conversation checkpoints separate, lists ready V3 and
existing V2 documents together, returns structured evidence only for cited V3
records, and serves stored previews through a safe artifact route. Streamlit
routes PDFs to V3, preserves V2 for current image/audio inputs, tracks the
retrieval version with each selection, and renders cited artifact metadata and
page/region previews. V3 deletion is deferred until vectors, manifest rows, and
binary assets can be removed as one coordinated operation. The focused query/UI
selection passed 45 tests. With the retrieval-only evaluator contracts, the
complete offline suite now passes 299 tests.

V1 is no longer an active runtime target; its source path remains only for
historical reproducibility and legacy metadata. The isolated V3 index and one
controlled full-path smoke are complete. The retrieval-only executor now
reuses the production reader, shared candidate deduplication, and Ettin
reranker while recording exact IDs, provenance, ranks, scores, production
selection, and latency without provider calls. The next gate is to ingest the
frozen corpus and inspect that report before the bounded Groq core. Design,
measurement limits, and promotion gates are documented in
`docs/MULTIMODAL_V3_ARCHITECTURE.md`.

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
validation was followed by logged user-run V2 sessions covering Auto,
selected-document, formatting, and follow-up behavior. Their failures drove the
answer-routing acceptance repair recorded below. No live providers, Pinecone,
PostgreSQL, or `.env` were used for the original implementation verification.

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

## Current V3 execution sequence

1. **Review and publish the V3 boundary - complete.** PR #34 merged at
   `b594ef4`; the locked CPU workflow and local cached GPU build passed.
2. **Connect artifact-aware retrieval and presentation - complete locally.**
   The V3 reader, existing reranker/graph integration, additive API evidence,
   unified document selection, safe asset route, and Streamlit preview path are
   implemented. Live retrieval quality is intentionally not claimed yet.
3. **Create an isolated V3 index - complete.** `agentic-rag-hybrid-v3` uses the
   current 384-dimensional embedding and `dotproduct` metric; V2 was untouched.
4. **Run one controlled V3 ingestion/query - complete.** The scanned fixture
   produced 8 artifacts, 7 search records/vectors, ready status, a grounded and
   cited OCR answer, structured locator evidence, and a valid preview. Cleanup
   removed 3 binary assets, 1 manifest, and all test vectors.
5. **Generate a reviewed question set for each selected document - complete.**
   The 49-case suite pairs controlled and natural wording and records expected
   pages, evidence types, concepts, statuses, citations, formats, and retry
   bounds. Exact artifact IDs remain pending controlled V3 ingestion.
6. **Evaluate in two passes.** Run the larger set through retrieval only, then
   run a bounded end-to-end subset through Groq to measure answers, citations,
   verdicts, follow-ups, formatting, and latency without wasting provider quota.
7. **Promote or repair from evidence.** Compare matched V2/V3 results. Promote
   V3 only if text retrieval does not regress and its additional evidence types,
   citations, abstentions, storage, and latency meet the documented gates.
8. **Ship a runnable cloud version.** Once the local demo boundary is stable,
   move to CI/CD and an AWS MVP while keeping parser licensing, secrets,
   observability, upload limits, and rollback explicit.

## Planned live evaluation

The exact test documents and questions are now frozen in
`eval_set/v3_end_to_end_questions.json`, with the execution guide in
`docs/V3_END_TO_END_EVALUATION.md`. Each document has controlled and natural
user wording so the evaluation can distinguish retrieval quality from language
sensitivity. The suite has 49 cases; the first provider-backed run is limited to
18 core cases.

Per-document coverage will include:

- direct factual lookup and ordinary paraphrases;
- vague or underspecified questions that a real user might type;
- conversational follow-ups, pronouns, corrections, and “are you sure?” turns;
- section-level synthesis and relationships across distant pages;
- figure, table, formula, diagram, caption, and OCR-dependent questions where
  those artifacts actually exist;
- requested bullets, tables, ASCII flowcharts, and mathematical formatting;
- false premises, absent evidence, and wrong-document questions that should
  abstain rather than generate;
- citation checks against the exact page, region, artifact, and grounding text.

For every query, the evidence record should capture the expected artifact,
pre-rerank and post-rerank position, final context inclusion, answer status,
grounding diagnosis, citation validity, retry/correction counts, and cold/warm
latency. This separates four failure classes: parser extraction, retrieval or
reranking, generation or formatting, and routing or verdict assignment.

## V2 repair findings and outcomes

The following defects were discovered through earlier live runs and retained as
regression boundaries for V3. Their repairs are present in the current local
worktree or merged baseline; they are no longer the active development queue.

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
logical size shared 3.919 GB with the existing local image and added about 643
KB of unique image data. That package checkpoint did not contact a live service;
later logged user runs supplied the remaining behavioral evidence and the V3
evaluation retains these cases as regression boundaries.

### 1. Normalize generated citations

Provider variants such as `【S1†L7-L9】`, alternate brackets, and invisible
Unicode characters are normalized to canonical `[S#]` markers before labels
are checked against the supplied source catalog. The repair is
provider-independent and retains fail-closed handling for unknown labels.

### 2. Classify abstentions reliably

Unicode-aware evidence-limitation detection now recognizes explicit long and
short abstentions before semantic grounding. Safe “I don’t know from the
provided evidence” answers therefore cannot be published as answered and
grounded solely because of punctuation or response length.

### 3. Improve final evidence selection

The reranker was changed to the measured Ettin 32M model, answer construction
keeps direct passages ahead of overviews, and a bounded RRF reserve prevents a
single cross-encoder miss from removing every lexical candidate. The reviewed
15-case comparison retained Hit@5 of 12/12 and MRR 0.840 on CUDA while cutting
mean reranking time to 0.142 seconds. Broader V3 evaluation remains necessary;
cross-encoder scores are still ranking signals rather than probabilities.

### 4. Make runtime budgets reproducible

- SDK-level Groq retries are disabled so the graph owns the retry budget.
- Task-specific caps and low reasoning effort keep routing, grading, rewriting,
  and verification responses inside predictable bounds.
- NLTK `punkt_tab` and `stopwords`, embedding and reranker revisions, and the
  selected parser are baked into the image and verified without networking.
- Direct content is selected before the optional whole-document overview when
  both support the answer.

## Verification gates

1. Keep the 299-test network-disabled suite and disposable PostgreSQL smoke
   green while integrating live V3 storage.
2. Keep the isolated V3 index separate from the populated V2 index.
3. Preserve the successful parser-to-ready and live query smoke as the minimum
   regression boundary.
4. Run the reviewed corpus through retrieval-only evaluation before expanding
   the bounded Groq run.
5. Preserve the now-frozen page-level evidence and abstention labels; attach
   exact artifact and search-record IDs after controlled V3 ingestion.
6. Compare V2 and V3 on the same documents and information needs, reporting
   retrieval, citation, verdict, latency, storage, and failure evidence.
7. Decide V3 promotion and the AWS deployment boundary from those results.

## Evaluation artifacts

- `docs/CHUNKING_PILOT.md` - reproducible labeling and comparison workflow.
- `docs/CHUNKING_V1_V2_RETRIEVAL_REPORT.md` - initial three-document comparison.
- `docs/CHUNKING_V2_BROAD_EVALUATION.md` - broad retrieval and full-API evidence.
- `eval_set/chunking_broad_reviewed_cases.json` - reviewed broad cases.
- `eval_set/chunking_broad_v1_pairs.json` - v1 chunk labels.
- `eval_set/chunking_broad_v2_pairs.json` - v2 chunk labels.
- `eval_set/v3_end_to_end_questions.json` - V3 page, artifact, grounding,
  answer, citation, routing, format, and latency expectations.
- `scripts/evaluate_v3_retrieval.py` - explicit retrieval-only V3 evidence
  collection with no provider calls or passage text in its report.

The full response payloads and extracted passage catalogs remain under ignored
`logs/evaluation/` because they can contain document text.
