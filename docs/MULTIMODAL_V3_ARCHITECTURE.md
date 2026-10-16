# Multimodal retrieval V3

Status: the V3 contracts, measured parser choice, selected-parser packaging,
transactional artifact/search-record persistence, hybrid reader/writer, graph
dispatch, structured API evidence, and Streamlit previews are implemented and
merged. PyMuPDF4LLM/Layout with conditional RapidOCR is selected from the fixed
parser comparison; Docling remains measured fallback evidence. The token-safe
seven-document corpus is ready under schema
`multimodal_v3:441428c54fceb4fd`, and PostgreSQL/Pinecone agree on 2,208 search
records linked to 2,203 source artifacts. The former FlashAttention Figure 1
selection miss now passes through the public API with cited visual and text
evidence. The broader manual release-candidate run remains pending.

## Why V3 exists

V2 improved text retrieval by preserving headings, sentences, and page ranges,
but it still treats every source as searchable text:

- a PDF is page text without figures, table structure, formula objects, or OCR;
- an image becomes one generated caption and the original image is discarded;
- an audio file becomes one transcript without timestamped evidence;
- a citation points to a retrieved text chunk rather than a stable source
  artifact.

V3 separates original source artifacts from the representations used to find
them. This makes it possible to retrieve through text while displaying and
citing the original figure, table, formula, page crop, image, or audio segment.

## Logical architecture

```text
source file
  -> versioned parser
  -> Artifact manifest
       |-- original/derived bytes -> AssetStore
       `-- searchable representations -> SearchRecord[]
                                              |
                                              v
                                    versioned retrieval index
                                              |
                                              v
                                       EvidenceRef[]
                                              |
                                              v
                                  generation and verification
```

V3 is one logical retrieval schema that may use multiple physical stores:

```text
multimodal_v3
  - text proxy index: agentic-rag-hybrid-v3
  - optional visual index added only after a measured text-proxy miss
  - PostgreSQL artifact and ingestion registry
  - local asset volume, later replaced by S3
```

V2 remains the live comparison/control path. V1 is retained only as archived
evaluation history and legacy-schema compatibility; it is not an active
runtime target.

## Work Package 1 contracts

Implemented in `src/agentic_rag/ingestion/`:

| Contract | Responsibility |
|---|---|
| `Artifact` | Stable source unit with kind, locator, content digest, derivation, and optional asset key |
| `SearchRecord` | Searchable text representation linked to exactly one artifact |
| `EvidenceRef` | Answer-facing provenance without passage text or binary payloads |
| `RetrievalSchema` | Complete parser, chunking, representation, embedding, sparse, and vector-ID identity |
| `AssetStore` | Storage boundary for original and derived binary assets |

Supported artifact kinds are text blocks, figures, tables, formulas, page
images, standalone images, and timestamped audio segments. Search
representations include native text, OCR, captions, table Markdown, formula
LaTeX, transcripts, and page summaries.

Every representation records its grounding basis, such as direct text,
OCR-derived text, table extraction, formula extraction, a vision description,
or an audio transcript. The application can therefore avoid describing a
caption-only answer as direct visual verification.

### Stable identity

- Artifact IDs derive from the document, artifact kind, source locator, parent,
  and exact content digest.
- Search-record IDs additionally include representation type, searchable text,
  grounding basis, and retrieval schema.
- Retrieval schema fingerprints change when parser revision/options, chunking,
  representation contract, embedding model/revision/dimension, sparse encoder,
  vector-ID contract, or optional visual embedding changes.
- The Docling accelerator is a parser option. CPU and CUDA inference cannot
  silently write into the same V3 schema because one formula-heavy page emitted
  a different artifact identity across devices.
- Reranker configuration is deliberately excluded because reranking can be
  evaluated against existing vectors without re-ingestion.

These rules prevent incompatible parser or embedding experiments from silently
sharing one index.

### Source locations

PDF locations use one-based page ranges and optional normalized bounding boxes.
Audio locations use millisecond intervals. A locator cannot mix page and audio
coordinates. These invariants make citations testable and allow the UI to open
the exact source asset later.

### Local assets

`LocalAssetStore` writes content-addressed files beneath one configured root.
Keys contain a validated document ID, artifact ID, and full SHA-256 digest.
Writes are atomic and idempotent, path traversal is rejected, and document
deletion cannot escape the store root. Binary bytes are not placed in
Pinecone, PostgreSQL rows, LangGraph checkpoints, or answer payloads.

### V2 compatibility

`legacy_document_to_v3` wraps a current LangChain `Document` as one text
artifact and one search record without changing its searchable content. This
adapter is a migration aid and does not replace original visual or audio assets
that V2 already discarded.

## Work Package 2 parser evaluation harness

The parser decision is measured against a fixed, hash-pinned page corpus before
either candidate can enter live ingestion. `eval_set/v3_parser_bakeoff_cases.json`
contains ten cases across six files: research-paper pages with multi-column
text, architecture figures, tables, formulas, and algorithm blocks, plus a
deterministic image-only PDF for the OCR path. Research PDFs remain outside the
repository; the manifest records only their filenames, exact SHA-256 digests,
page selections, purposes, and review expectations.

Both candidates must implement the same `ParserAdapter` boundary and emit a
`ParseRun` containing only normalized `Artifact` and `SearchRecord` contracts.
The harness checks:

- expected artifact kinds and required searchable terms per selected page;
- selected-page coverage and artifact/search-record lineage;
- deterministic artifact and record IDs across repeated runs;
- elapsed time, peak memory, and parser warning counts;
- corpus identity before a comparison begins.

The summary deliberately excludes extracted passage text. Visual correctness
still requires rendering the selected pages and reviewing artifact bounding
boxes and crops; count and term checks cannot prove layout accuracy by
themselves.

The corpus audit passes for all ten cases and six exact files. The first Docling
measurements used a separate evaluation image to protect V2. The verified
dependency and model set is now consolidated into the default GPU application
target, while V3 remains disconnected from live ingestion. Vanilla PyMuPDF and
PyMuPDF4LLM with Layout now have equivalent adapters and repeated complete runs
against the same manifest.

### Docling candidate evidence

The disposable image pins Docling 2.129.0, RapidOCR 3.9.2, CPU-only PyTorch,
and the exact layout, accurate-table, formula, and English OCR model files. It
runs with networking disabled after the build. The final image digest is
`sha256:2b0c24032682be92b9eee5a8be2c462a9c09e9117a009bba88ec058f5e1f5c83`;
its logical size is 7.46 GB, including about 1.3 GB of embedded Docling models.
The application dependency declaration was not changed.

The complete ten-case run parsed every selected page without a parser warning.
Nine unchanged cases met all 49 screening checks. Visual inspection showed that
the original LoRA case pointed at page 2 even though the expected adaptation
figure is on page 1; the corrected page and caption terms then met all 5 checks.
The current manifest therefore has 54/54 supported checks across the corpus,
using the full run plus the focused corrected-case rerun. This combined result
does not replace the pending same-configuration repeatability run.

Representative review showed correctly cropped Transformer, FlashAttention,
and Flamingo figures; structured tables; standalone mathematical formula
artifacts on Attention, FlashAttention, and DPO pages; and successful OCR,
table, and figure recovery from the image-only fixture. The simple printed
`E = mc^2` line on that fixture remained a text block rather than a standalone
formula object, so formula classification is useful but not exhaustive.

The corrected full run took 475.3 seconds on the local CPU, averaged 46.4
seconds per case, and reported a 3.33 GiB process peak. Most ordinary figure and
text pages completed in 4-23 seconds. Formula-heavy FlashAttention and DPO pages
took about 196 and 133 seconds. Reusing one initialized converter removed
repeated model cold starts; the remaining cost is content-dependent formula
inference. A production parser should therefore run the inexpensive structural
pass first and invoke formula enrichment only for pages selected by measured
need.

The first adapter revision mislabeled native PDF text as OCR because Docling's
conversion result did not expose parsed native cells in this runtime. The
adapter now uses the already-pinned `pypdf` package once per PDF as the embedded
text baseline. A native LoRA page consequently emitted 11 native-text records
and 2 OCR-only figure labels, while the image-only page emitted 5 OCR-text
records and no native text. This provenance distinction is required before
grounding or citations can describe how evidence was obtained.

The real run also exposed Windows path overflow in the first physical asset-key
layout. Asset keys are now opaque content-addressed paths containing a 128-bit
document storage digest and the full content SHA-256. Full document and artifact
identities remain in the contracts. The longest path in the deep evaluation
worktree fell to 216 characters, and equal bytes within one document are
deduplicated.

### Local GPU feasibility gate

The local NVIDIA GeForce RTX 2050 exposes 4 GiB VRAM and compute capability 8.6
through Docker's NVIDIA runtime. A temporary `docling-gpu` Docker target replaces
the evaluation image's CPU PyTorch wheels with PyTorch 2.13.0 CUDA 13.0 wheels.
It also installs the C compiler required by Triton to build small device launchers
on first use. The first CUDA trial without that compiler produced formula-stage
errors; the repaired target completed the same cases without an out-of-memory
failure.

The repaired candidate is
`sha256:c7539c4c69d489956870edde3a68a97c91b2b5b1ef4844a25aef1f466d1cc14e`
and has a 15.32 GB logical size (14.27 GiB). This is one hybrid evaluation image:
Docling's PyTorch models use CUDA while RapidOCR remains on the CPU through
ONNX Runtime. Adding `onnxruntime-gpu` is deferred because OCR was not the
measured bottleneck and it would add another CUDA compatibility and storage
surface.

Only the two formula-heavy cases were rerun because they accounted for most of
the CPU latency:

| Case | CPU | CUDA | Speedup | Checks | Peak CUDA reserved |
|---|---:|---:|---:|---:|---:|
| FlashAttention algorithm | 196.25 s | 42.15 s | 4.66x | 5/5 | 1,948 MiB |
| DPO objective derivation | 132.68 s | 32.25 s | 4.11x | 5/5 | 1,608 MiB |

The CUDA pair met 10/10 screening checks and stayed below half of the available
VRAM. A second independent FlashAttention CUDA run completed in 34.43 seconds
and reproduced all 31 artifact IDs and all 30 search-record IDs exactly. The
GPU path is therefore repeatable for the tested page. An older CPU result had
one different FlashAttention artifact identity, so `accelerator_device` is now
included in both parser derivation metadata and the retrieval-schema fingerprint.
A V3 index created with the CUDA profile must fail closed if a CPU parser tries
to write into it.

The intended runtime shape is now packaged as one GPU-capable application image,
not one CPU service plus one GPU service. The API alone receives GPU access.
Ordinary Python, PDF text extraction, OCR, storage, and API work continue on CPU
inside that image; supported Docling stages, embedding, and Ettin reranking use
CUDA. A shared process-local lock prevents those three model workloads from
overlapping on the 4 GiB device. The lock does not coordinate separate API or
worker processes, so a future multi-process deployment needs one GPU owner or
an external inference queue.

### Consolidated image evidence

The packaged image is
`sha256:750383503ed60f7944a7d394df97fd37752b67e31ec9b2b53c69196e37adb081`
and reports 15.32 GB logical size. With networking disabled it exposed the RTX
2050, loaded CUDA PyTorch 2.13.0, Docling 2.129.0, the exact MiniLM and Ettin
revisions, and the baked NLTK resources. A packaged FastAPI import also passed
with dummy constructor values and no service access. The complete offline suite
passed 230 tests with one dependency deprecation warning.

The co-residency gate loaded embedding and Ettin first, then parsed the
FlashAttention algorithm page through Docling. The case met 5/5 checks and
peaked at 2,188 MiB CUDA reserved, compared with 234 MiB before Docling loaded.
This is sufficient for the tested serial lifecycle but is not a guarantee for
parallel workers or a future local generative model.

After promotion to `agentic-rag-app:local`, the prior CPU application image and
standalone Docling image were removed. Docker retains only the application image,
`postgres:16`, and the PostgreSQL data volume; 34.69 GB of unused build cache
was reclaimed. Large dependency layers will be downloaded again on the next
clean rebuild, while routine starts reuse the retained image.

### PyMuPDF controls and parser decision

Vanilla PyMuPDF 1.28.2 was first executed as the speed control. Two final runs
each succeeded on all ten cases, met 42/54 screening checks, and reproduced
every artifact and search-record ID. They averaged 1.24 seconds per case and
13.75 seconds wall time, with a maximum 168.2 MiB process peak. Its misses were
the V3 capabilities under investigation: OCR-searchable text, standalone
formula regions, and independently citable vector diagrams.

PyMuPDF4LLM 1.28.2, PyMuPDF Layout 1.28.2, PyMuPDF 1.28.2, and RapidOCR 3.9.2
were then installed in a 280 MB temporary package volume without changing the
application lockfile or image. The adapter consumes ordered Layout JSON,
restores spaces from span geometry, removes line-break hyphens, filters page
headers and footers from retrieval, emits structured tables, crops pictures and
formula regions, and records native, OCR, and table lineage separately.

Two final network-disabled runs each succeeded on all ten cases, met 54/54
screening checks without adapter warnings, and reproduced every artifact and
search-record ID. They averaged 3.33 seconds per case and 36.77 seconds wall
time, with a maximum 800.1 MiB process peak. Against full CPU Docling this is
about 14 times faster per case, about 13 times faster by wall time, and about
4.2 times lighter by reported process peak.

Visual review found and closed an issue the count checks missed. PyMuPDF4LLM
inserts OCR text into its in-memory PDF; rendering that mutated object produced
doubled text in evidence images. Layout/OCR parsing now uses a separate document
object from page and region rendering. Post-fix page images, vector diagrams,
table regions, and formula crops remain pixel-faithful to the source.

PyMuPDF Layout is therefore the selected V3 integration candidate. It replaces
the unproven PyMuPDF-plus-Docling triage layer in the immediate plan. Docling is
retained as a measured fallback for a future reviewed failure, rather than paid
on every page or introduced through speculative reconciliation logic. Formula
regions are citable crops with native searchable text when available; the
adapter does not claim formula-to-LaTeX conversion. Full evidence and limits are
in `docs/V3_PARSER_COMPARISON.md`.

## Implementation sequence

1. **Contracts and lineage** - implemented locally in this work package.
2. **Parser bakeoff** - complete locally for Docling, vanilla PyMuPDF, and
   PyMuPDF Layout; PyMuPDF Layout is the measured integration candidate.
3. **Selected parser packaging** - implemented locally: the exact PyMuPDF
   Layout runtime is locked, version drift fails closed, and Docker/CI perform
   a network-disabled identity check. The AGPL-or-commercial licensing choice
   remains a deployment gate.
4. **V3 text-proxy ingestion** - the durable manifest/asset boundary, isolated
   vector writer, and lazy `POST /v3/ingest` composition are implemented. The
   PostgreSQL boundary passed a disposable real-service smoke, and the isolated
   Pinecone path passed one controlled parser-to-ready ingestion with complete
   cleanup.
5. **Artifact-aware retrieval/API/UI** - implemented locally. The reader scopes
   V3 dense and sparse candidates by document and retrieval schema, reuses the
   existing Ettin reranker and corrective graph, returns additive structured
   evidence for cited records, and renders stored previews while preserving V2
   citation strings. Coordinated V3 deletion remains pending.
5a. **Evidence-aware selection repair** - merged and targeted-live checked.
   Query plans now declare the required source form
   (`text`, `visual`, `table`, `formula`, or `mixed`). Ordinary text questions
   retain the existing path. Structured questions preserve a bounded number of
   matching RRF candidates plus a same-page caption/explanation inside the
   five-record generation budget, with a
   filtered artifact query used only when the ordinary candidate set contains
   no matching source form. Missing required evidence follows the existing
   retry and abstention path.
6. **Shadow evaluation** - ingest the same corpus into v2 and v3 and compare
   text, figure, table, formula, scan, image, audio, citation, latency, and
   negative-query outcomes.
7. **Optional visual retrieval** - create a separate visual index only if the
   reviewed V3 text-proxy results show misses that OCR and captions cannot fix.

## Promotion gates

V3 will not become the default until:

- existing reviewed V2 text-evidence recall does not regress;
- artifact IDs and schema fingerprints are stable across repeated runs;
- every returned asset resolves to the correct source;
- reviewed page, bounding-box, and timestamp locators are correct;
- citation labels map to the expected artifact and grounding basis;
- unanswerable and absent-artifact questions abstain safely;
- parser, index, Docker storage, and latency measurements fit the local budget.

## V3 live-evaluation design

The live evaluation is frozen as 49 source-reviewed cases across seven
documents. Its seven controlled/natural pairs expose wording sensitivity, and
its 18-case core bounds the first provider-backed run. The extended cases cover
ambiguity, follow-ups, routing, false premises, absent evidence, and additional
format requests. The complete contract and execution order are in
`docs/V3_END_TO_END_EVALUATION.md` and
`eval_set/v3_end_to_end_questions.json`.

The broad question set will run retrieval-only first. This records parser
coverage, expected artifact identity, dense/sparse fusion rank, reranker rank,
final-context inclusion, and retrieval latency without spending LLM quota. A
smaller matched subset will then run through the full graph to measure source
routing, answer status, grounding diagnosis, citation-to-artifact precision,
retry/correction behavior, response formatting, and cold/warm latency.

The suite must include text, figure, table, formula, diagram, caption, and OCR
questions only where the reviewed document contains that evidence. It must also
include false premises, absent evidence, and wrong-document questions. Expected
evidence or expected abstention is frozen before execution. Failures are then
assigned to parsing, retrieval/reranking, generation/formatting, or routing and
verdict logic instead of being treated as one undifferentiated answer failure.

## Current boundary

The current V3 schema uses `artifact_search_record_v2` and
`artifact_structure_v3_token_safe_240`. PyMuPDF Layout artifacts remain whole
and keep their stable source locator, asset, and citation identity. Only an
oversized searchable representation is split into zero-overlap segments below
MiniLM's safe 240-token input budget; every segment links back to the same
artifact and records its parent record plus stable segment position. This index
migration is complete for the frozen seven-document corpus.

Ettin and the Groq generation/verifier path remain text-mediated. They rank and
verify native text, OCR, captions, table Markdown, and formula text rather than
raw pixels. The new evidence requirement is therefore a retrieval and
sufficiency constraint, not a claim that the models became natively visual.
Visual or spatial details absent from a searchable representation must still be
qualified or rejected. A separate visual embedding or vision-verification lane
remains conditional on a measured miss after this release candidate.

The consolidated offline release gate exercised token-safe segmentation,
evidence classification and selection, filtered artifact retrieval, graph
wiring, citations, grounding, API contracts, and their existing regressions.
The complete network-disabled suite passed 312 tests with one existing Starlette
dependency warning. It used the existing local application image, mounted this
worktree read-only, and did not load `.env` or contact any service. Image rebuild,
fresh V3 ingestion, and one targeted live comparison subsequently completed;
the broader manual release-candidate run remains pending.

The promoted local image contains the exact selected PyMuPDF Layout runtime,
CUDA MiniLM/Ettin support, and a writable V3 asset-volume target. PostgreSQL DDL
and repository code now preserve schema identity, artifacts, search records,
parent-child page relationships, locators, BM25 state, and fail-closed readiness.
The V3 vector writer is independent of the V2 vector store and prunes stale IDs
only after successful current writes. The opt-in V3 endpoint composes these
dependencies only when called and uses the existing serialized embedding runtime.
The query/API/UI boundary adds version-scoped checkpoint identities, ready-only
document listing and lookup, a safe artifact asset route, and Streamlit PDF-to-V3
routing. The focused V3 query/UI selection passes 45 tests, and the complete
network-disabled offline suite passes 312 tests with the retrieval-only
evaluator contracts and the existing Starlette deprecation warning.

The frozen seven-document corpus was hash verified and re-ingested after a
digest-locked removal of the previous schema. All manifests are ready;
PostgreSQL record counts match both manifest vector counts and Pinecone prefix
counts. The artifact total remains stable while token-safe segmentation raises
the searchable record count from 2,022 to 2,208. Stable artifact IDs still match
the reviewed evidence, while exact search-record labels must be regenerated
where segmentation changed record identity.

One bounded public-API replay targeted the former FlashAttention Figure 1 miss.
The planner required visual evidence, final selection retained the page 2 figure
and its supporting text, the answer cited both, and verification returned
`answered` and `grounded` without exhausting retries. This is direct evidence
for the repaired failure, not a broad accuracy claim. Local evaluation may
continue under the installed package terms, while public deployment remains
gated on an explicit AGPL-or-commercial licensing decision.
