# V3 PDF parser comparison

Date: 2026-10-13

## Decision

Use PyMuPDF4LLM 1.28.2 with PyMuPDF Layout 1.28.2 as the leading V3 PDF
parser candidate. It recovered all evidence required by the current fixed
corpus while remaining much faster and lighter than full Docling. Vanilla
PyMuPDF remains the speed control. Docling remains a measured fallback for a
future case that PyMuPDF Layout cannot recover; it is no longer required in the
default page path by the evidence collected so far.

This selects the parser for the next integration work package. The adapter is
not connected to live ingestion, the packages are not in the application
lockfile, and no V3 index has been created.

## What was compared

All three adapters use the same hash-pinned ten-case manifest in
`eval_set/v3_parser_bakeoff_cases.json`. The six exact PDFs cover native
multi-column text, embedded and vector figures, tables, formulas, algorithms,
and an image-only OCR page. Both adapters emit the same `Artifact`,
`SearchRecord`, `SourceLocator`, and `ParseRun` contracts.

The automated checks answer only these questions:

- Did the expected artifact kinds exist?
- Did required terms remain searchable?
- Did every selected page produce evidence?
- Were artifact and search-record IDs stable on a repeat run?
- How long did the parser take and how much process memory did it report?

They do not prove general reading-order quality, table-cell fidelity, formula
to LaTeX conversion, semantic figure classification, or citation usefulness.
Representative page images and crops were inspected separately. That visual
review caught an OCR image-mutation defect that the 54 count checks could not
detect.

## Runtime identities

| Candidate | Runtime |
|---|---|
| PyMuPDF | PyMuPDF 1.28.2, 144 DPI page and embedded-image renders, native dictionary text, `find_tables()` Markdown, no OCR, no formula extraction |
| PyMuPDF Layout | PyMuPDF 1.28.2, PyMuPDF4LLM 1.28.2, PyMuPDF Layout 1.28.2, RapidOCR 3.9.2, ordered semantic regions, conditional OCR, table grids, picture and formula crops |
| Docling | Docling 2.129.0, Docling Core 2.101.1, Docling Parse 7.22.2, Docling IBM Models 4.0.3, RapidOCR 3.9.2, accurate tables, OCR and formula enrichment |

PyMuPDF Layout was installed in a 280 MB temporary package volume against the
existing `agentic-rag-app:local` image. Parsing then ran with networking
disabled and one CPU thread for the numerical runtimes. The application
lockfile and image were not changed.

## Results

| Metric | Vanilla PyMuPDF | PyMuPDF Layout | Docling evidence |
|---|---:|---:|---:|
| Successful cases | 10/10 | 10/10 | 10/10 |
| Supported screening checks | 42/54 (77.8%) | 54/54 (100%) | 54/54 using the full run plus corrected LoRA rerun |
| Mean time per case | 1.24 s across two runs | 3.33 s across two final runs | 46.4 s on the full CPU run |
| Total wall time | 13.75 s average | 36.77 s average | 475.3 s |
| Maximum process peak | 168.2 MiB | 800.1 MiB | 3.33 GiB |
| Repeat artifact IDs | stable in every case | stable in every case | stable on the repeated bounded GPU case |
| Repeat search-record IDs | stable in every case | stable in every case | stable on the repeated bounded GPU case |

PyMuPDF Layout costs about 2.7 times the vanilla control's mean latency and
about 4.8 times its reported peak memory, but it recovers the 12 checks the
control misses. Against the full Docling CPU run, it is about 14 times faster
per case, about 13 times faster by wall time, and about 4.2 times lighter by
reported process peak.

The two final PyMuPDF Layout runs were independent disposable containers. Both
met 54/54, emitted no adapter warnings, and exactly reproduced every artifact
and search-record ID. Their wall times were 33.52 and 40.02 seconds; their mean
case times were 3.01 and 3.65 seconds.

Twenty-five focused parser tests passed after the final corrections. The
complete repository offline suite passed 247 tests in the existing application
image with networking disabled; the only warning is the existing Starlette
`BlockingPortal` dependency deprecation.

## Case-level PyMuPDF Layout evidence

| Case | Checks | Reviewed evidence |
|---|---:|---|
| Transformer architecture | 6/6 | Correct standalone vector architecture crop plus searchable labels. |
| Attention equations and diagram | 6/6 | Separate diagram and formula crops; required terms searchable. |
| Attention results table | 4/4 | Structured table Markdown and surrounding text present. |
| FlashAttention tiling figure | 6/6 | Vector diagram became an independently citable figure. |
| FlashAttention algorithm | 5/5 | Two formula regions plus searchable algorithm text. |
| LoRA adapter architecture | 5/5 | Figure present; wrapped `reparametrization` reconstructed correctly. |
| DPO objective derivation | 5/5 | Three formula regions plus theorem text. |
| Flamingo cross-attention architecture | 5/5 | Architecture figure and labels searchable. |
| Flamingo result tables | 5/5 | Two structured tables and required text present. |
| Image-only OCR fixture | 7/7 | OCR recovered all required terms and preserved a clean original page image. |

## Vanilla PyMuPDF control evidence

| Case | Checks | Missing or questionable evidence |
|---|---:|---|
| Transformer architecture | 6/6 | Embedded raster figure crop was correct. |
| Attention equations and diagram | 5/6 | Diagram crops were correct; no standalone formula artifact. |
| Attention results table | 4/4 | Required table and searchable terms were present. |
| FlashAttention tiling figure | 5/6 | Vector diagram remained in the page render but was not emitted as a figure. |
| FlashAttention algorithm | 4/5 | Text remained searchable; no standalone formula artifact. |
| LoRA adapter architecture | 3/5 | Vector figure was not emitted; `reparametrization` did not survive as an exact searchable term. |
| DPO derivation | 4/5 | Text remained searchable; no standalone formula artifact. |
| Flamingo cross-attention architecture | 4/5 | Vector architecture diagram was not emitted as a figure. |
| Flamingo result tables | 5/5 | Required table count and terms were present; tiny decorative image blocks were also labeled as figures. |
| Image-only OCR fixture | 2/7 | Full page image was preserved, but no text or required terms were searchable because OCR is absent. |

The adapter also reported table candidates on some diagram-heavy pages. This
shows why a count can pass while semantic table quality remains uncertain.

## Defects found and repaired during the comparison

### Missing spaces and wrapped words

PyMuPDF Layout can return each printed word as a separate span without literal
space characters. Concatenating span strings produced values such as
`Algorithm1` and hid terms that were visibly present. The adapter now restores
spaces from horizontal bounding-box gaps and removes line-break hyphens before
creating search records. This raised the same corpus from 49/54 to 54/54. The
logic is covered by focused regression tests.

### OCR provenance

RapidOCR text uses a filled fallback font in this integration, so PDF rendering
flags alone cannot identify it reliably. The adapter captures native page text
before OCR and combines substantial word overlap with rendering flags. Native
PDF text remains `native_text`; the synthetic scan emits only `ocr_text`.

### Evidence-image mutation

PyMuPDF4LLM inserts OCR text into its in-memory PDF before layout processing.
Rendering that same object caused doubled text in stored page images and figure
crops. The adapter now parses layout in one document object and renders assets
from a separate untouched object. Post-fix visual review confirmed clean page,
diagram, table-region, and formula crops.

## Remaining limits

- A detected formula region is stored as a source crop. Native formula text may
  be searchable, but the adapter does not claim that it is normalized LaTeX.
- OCR makes scanned text searchable but does not guarantee that every scanned
  table or formula receives the correct semantic artifact type.
- The ten-case corpus is intentionally diagnostic rather than statistically
  representative. More document styles must enter the reviewed shadow set
  before V3 becomes the default.
- Table structure and locator precision still need artifact-level reviewed
  labels in the V2-versus-V3 shadow evaluation.
- Installed metadata for PyMuPDF, PyMuPDF4LLM, and PyMuPDF Layout 1.28.2 says
  they are dual licensed under GNU AGPL 3.0 or an Artifex commercial license.
  The project must choose a compliant path before production distribution or
  hosted deployment; temporary evaluation does not settle that requirement.

## Selected next architecture

The next implementation should use one parser identity, `pymupdf_layout`, with
all package versions and parser options in the retrieval-schema fingerprint.

1. PyMuPDF4LLM plus Layout performs ordered text, heading, table, picture,
   formula-region, and conditional-OCR extraction.
2. Original page images and region crops are always rendered from an untouched
   PDF object.
3. Search records preserve whether text came from native PDF content, OCR, or
   table extraction and link back to the exact artifact and locator.
4. Docling is invoked only in a future experiment with a reviewed failure that
   PyMuPDF Layout cannot recover. Adding it would create a new parser revision
   and retrieval-schema identity.
5. After dependency and licensing review, connect the selected adapter to V3
   artifact storage and a new versioned index, then run V2 and V3 in shadow on
   the same documents and questions.

## Production-boundary checkpoint

The selected package set is now pinned in the application dependency graph and
lockfile. The Docker definition installs the same locked parser in both CPU and
GPU targets and runs `scripts/verify_v3_parser_runtime.py` during the build.
That check rejects missing or different package versions and reports only safe
identity, options, and capability metadata.

A disposable Python 3.12 container imported the exact set with networking
disabled. The selected GPU image was then rebuilt and promoted after parser,
CUDA, API, Streamlit, and offline-suite verification. Transactional artifact
and search-record persistence plus an isolated V3 vector writer are now
implemented, bringing the complete offline suite to 265 tests. V3 ingestion is
not live, no V3 index has been created, and the AGPL/commercial licensing choice
remains a deployment gate. The next boundary is an opt-in V3 ingestion entry
point and real local PostgreSQL schema validation.

This removes the unproven composite-parser triage layer from the immediate
critical path. A fallback is added only when measured evidence justifies its
latency, storage, and reconciliation cost.

## Reproducible local evidence

Final reports and review images are retained under ignored local paths:

- `logs/evaluation/v3-parser-bakeoff/pymupdf-final-1/`
- `logs/evaluation/v3-parser-bakeoff/pymupdf-final-2/`
- `logs/evaluation/v3-parser-bakeoff/pymupdf-layout-final-5/`
- `logs/evaluation/v3-parser-bakeoff/pymupdf-layout-final-6/`
- `logs/evaluation/v3-parser-bakeoff/docling-run-20261012-2/`
- `logs/evaluation/v3-parser-bakeoff/docling-run-20261012-3-lora/`

The shared runner is `scripts/run_v3_parser_bakeoff.py`; the previous
`scripts/run_v3_docling_bakeoff.py` name remains as a compatibility entry point.
Reports contain counts, identities, measurements, and review-image paths rather
than extracted passage text.

## API references

- PyMuPDF4LLM: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
- PyMuPDF Layout integration: https://pymupdf.io/blog/introducing-the-new-pymupdf4llm-now-including-layout
- PyMuPDF page extraction and rendering: https://pymupdf.readthedocs.io/en/latest/page.html
