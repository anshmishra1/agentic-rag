# Reranker model upgrade

## Failure being addressed

The broad v2 evaluation found reviewed DPO partition evidence in the 15-item
candidate pool, but the MiniLM cross-encoder placed it at rank 10. The answer
path keeps five content passages, so that evidence never reached generation.
This is a reranking failure, not an embedding-recall failure: the existing
embedding and hybrid search already found the passage.

## Controlled change

Replace only the local reranker:

| Component | Control | Adopted candidate |
|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | unchanged |
| Chunking | `structure_aware_v2` | unchanged |
| Candidate pool | 15 hybrid candidates | unchanged |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | `cross-encoder/ettin-reranker-32m-v1` |
| Candidate revision | n/a | `b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd` |
| Final content budget | 5 | unchanged |

Ettin uses the standard query-passage CrossEncoder interface without a task
instruction. Its raw relevance logits continue through the existing sigmoid
transform so ranking semantics stay monotonic. The default batch is 8 because
that was slightly faster than 32 on the matched local candidate groups.

## Why the embedding is unchanged

Changing the embedding would alter stored vectors and require a new index and
full re-ingestion. It would also mix candidate-recall changes with reranking
changes. Keeping the current embedding lets the comparison answer one question:
does the stronger reranker move reviewed support into the final five passages?

## Acceptance gate

The candidate is not accepted because it has stronger public benchmarks. It
must be compared with MiniLM on the same reviewed candidates and questions.
Record at least:

- reviewed support rank and Hit@5;
- mean reciprocal rank;
- rank regressions on previously successful questions;
- cold and warm latency;
- peak process memory; and
- the post-sigmoid score distribution used by retrieval routing.

The known DPO passage must move into the five-content evidence set, aggregate
reviewed Hit@5 must not regress, and CPU latency and memory must remain usable.
Existing routing thresholds stay provisional until the candidate score
distribution is observed.

## Storage and rollback

The exact candidate revision is pinned for reproducibility. Development keeps
one candidate application image rather than one image per code edit. Docker
can reuse dependency, embedding, and NLTK layers; after acceptance, the old
application tag and rejected model cache can be removed. Runtime rollback is
explicit: select `cross-encoder/ms-marco-MiniLM-L6-v2`, its pinned revision,
and its pinned revision.

## Verification boundaries

Offline tests can prove configuration, pinned identity, instruction handling,
score transformation, batching, and rollback behavior. They cannot prove that
a candidate improves the research-document ranks or that its CPU latency is
acceptable. Those conclusions require one controlled model download followed
by the existing reviewed reranker comparison; no Pinecone write, ingestion, or
LLM provider call is needed for that comparison.

## Rejected Qwen candidate result - 2026-10-11

Built `agentic-rag-app:qwen3-reranker` from this worktree and verified the
packaged embedding and reranker in a disposable container with networking
disabled. The image is 5,891,945,734 bytes, compared with 3,919,514,428 bytes
for `agentic-rag-app:local`. The additional logical image size is about
1.84 GiB. Docker still reported 130.99 GiB free on drive D after the build.

The matched offline replay reconstructed the exact 15 RRF candidates for 15
historical diagnostic cases from the six hash-verified local PDFs. It contained
225 passages and 12 cases with verified supporting passage IDs. MiniLM
completed the full set with 11/12 Hit@5, MRR 0.744, and mean reranking time
0.545 seconds per 15 candidates.

The full Qwen replay was stopped after two cases because those cases took
86.093 and 132.965 seconds. Both retained support at rank 1, but the sustained
CPU cost had already failed the interactive latency gate. A reduced replay then
measured the three cases that directly exposed reranker behavior:

| Case | MiniLM rank | Qwen rank | MiniLM seconds | Qwen seconds |
|---|---:|---:|---:|---:|
| `q061-source` | 11 | 1 | 0.549 | 182.889 |
| `q053-source` | 2 | 5 | 0.557 | 151.155 |
| `q043-source` | 3 | 1 | 0.603 | 180.232 |

Qwen repaired the known rank-11 loss and improved one other case, but it also
moved one supporting passage down to the final context boundary. Its mean time
on these cases was 171.425 seconds versus 0.570 seconds for MiniLM, about 301
times slower. Model loading took 1.445 seconds and the one-pair warmup took
11.567 seconds; the sustained per-case latency came from scoring the passages,
not image startup.

### Decision

Do not merge this unoptimized BF16 Qwen candidate as the default CPU reranker.
The experiment proves that additional model capacity can repair the observed
ranking failure, but this packaging is not usable in an interactive request
path and is not uniformly better. At this checkpoint the Qwen image remained
available for inspection; it was deleted after the follow-up comparison. An
additional replay used the exact saved
`structure_aware_v2` candidates for the current DPO partition question. It
reproduced MiniLM's support rank 10 in 1.367 seconds; Qwen moved the passage
only to rank 7 in 169.349 seconds. The passage therefore remained outside the
five-content budget, so Qwen also failed the primary current-case quality gate.
The next candidate must preserve these fixed passages and labels while testing
a CPU-efficient model or an explicitly optimized runtime such as a measured
quantized/ONNX variant.

All comparison artifacts are ignored under
`logs/evaluation/20261011-qwen-reranker/`. No `.env`, Pinecone, PostgreSQL,
ingestion, LLM provider, or application service was used.

## Ettin 32M follow-up result - 2026-10-11

The rejected Qwen image was deleted and Docker reclaimed 6.55 GB of unused
build cache while retaining the existing application images and volumes. The
next candidate was downloaded into an isolated Docker volume rather than baked
into another application image:

| Component | Candidate |
|---|---|
| Reranker | `cross-encoder/ettin-reranker-32m-v1` |
| Revision | `b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd` |
| Parameters | 32.8 million |
| Instruction prefix | none |
| Evaluation device | CPU |

On the same 15 historical cases and 225 candidates, Ettin placed all 12
reviewed supporting passages inside the top five, improved Hit@1 from MiniLM's
8/12 to 9/12, and improved MRR from 0.744 to 0.840. Mean time was 0.851 seconds
per 15-candidate case with batch 8. A batch-32 check produced the same ranks and
0.942-second mean latency, so the measured local default remains 8.

The decisive `structure_aware_v2` DPO case moved from MiniLM rank 10 and Qwen
rank 7 to Ettin rank 4 in 0.835 seconds. It therefore reaches the five-content
answer budget and passes the primary quality gate that Qwen failed. All scoring
runs used Docker networking disabled after provisioning and made no `.env`,
Pinecone, PostgreSQL, ingestion, provider, or application-service call.

Ettin's sigmoid scores are tightly saturated on these candidates (mean 0.980,
and 0.996 to 1.000 on the DPO case). Its ranks are useful, but these values must
not be treated as calibrated relevance probabilities or used to justify the
existing routing thresholds.

## Source adoption checkpoint

The uncommitted runtime, example configuration, and Docker provisioning now
select the exact Ettin revision above. The Qwen instruction path was removed,
MiniLM's historical model alias remains available for rollback, and batch 8 is
the measured default. The evaluator records model identity, scores, support
ranks, and latency without storing question or passage text in its result.

Ten focused adoption tests and all 185 offline unit tests passed in the existing
application image with networking disabled. Eight changed Python files passed
AST parsing and `git diff --check` passed.

## Packaged image checkpoint

Built `agentic-rag-app:ettin-candidate` and promoted the identical image ID to
the Compose `agentic-rag-app:local` tag. Image digest
`sha256:58b1984b4c404f328425e596b1576c9ff6040385d993f1661db31b4e10cccb8b`
has logical size 3,994,803,911 bytes. Build-time and fresh exported-container
checks loaded the exact embedding and Ettin revisions plus both NLTK resources
with networking disabled. A production-loader smoke also ranked an obviously
relevant passage above an irrelevant one without network access.

After verification, removed six stopped containers, the obsolete WP2 and V2
application tags, the replaced MiniLM local image, the temporary Ettin cache
volume, dangling images, and 4.203 GB of unused build cache. The Ettin candidate
and local tags share one image and therefore do not duplicate its layers.
PostgreSQL and other data volumes were preserved. Docker now reports two images
using 4.637 GB, three preserved volumes using 147.7 MB, and 3.995 GB of build
cache with zero bytes currently reclaimable. Host drive D still reports 130.99
GiB free because deleting data inside Docker's virtual disk does not necessarily
shrink the backing virtual-disk file immediately.

Before the live acceptance run, the effective non-secret runtime model fields
must be checked because a private `.env` value can override image defaults.
Positive and negative routing behavior also remains an explicit live acceptance
criterion because Ettin's saturated score scale is not calibrated.

## Consolidated GPU runtime checkpoint - 2026-10-13

The accepted Ettin source and direct-evidence selection repair were carried into
the V3 worktree before building the Docling-capable runtime. This prevents the
V3 image from silently reverting to the older MiniLM reranker simply because the
two experiments began on separate branches.

The default local Docker target now packages the pinned MiniLM embedding, Ettin,
Docling, CUDA PyTorch, and Docling artifacts in one image. The API receives GPU
access through `scripts/local_docker.py`; Streamlit and PostgreSQL do not. A
shared process-local lock serializes embedding, reranking, and Docling inference
to fit the local 4 GiB VRAM budget. `--cpu` remains a deterministic recovery
mode, and CI builds the smaller `cpu-runtime` target.

The final consolidated image replayed the same 15 cases and 225 candidates on
both devices with batch 8. CPU reproduced Hit@5 12/12, MRR 0.840, and 0.850
seconds mean per case. CUDA preserved every support rank and metric while
reducing the mean to 0.142 seconds, about 6.0x faster. The separate V2 DPO case
remained rank 4 on both devices. Scores differed only at floating-point noise
level and retained the same saturated distribution.

The same image embedded the saved 225 passages in 7.73 seconds on CPU and 1.34
seconds on CUDA, about 5.8x faster. A 15-query batch took 0.054 seconds on CPU
and 0.079 on CUDA, so small query-only embedding does not benefit on this card.
The application still uses CUDA for one local configuration because reranking
dominates interactive model latency and batched ingestion benefits materially.

With embedding and Ettin resident, the FlashAttention Docling case met 5/5
checks and peaked at 2,188 MiB reserved CUDA memory. This validates serial
co-residency on the 4 GiB RTX 2050 for the tested workload. The process-local
lock does not protect multiple API/worker processes; a future multi-process
deployment needs a single GPU owner or external inference queue.
