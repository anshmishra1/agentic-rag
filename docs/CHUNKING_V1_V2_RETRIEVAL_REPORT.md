# V1 versus v2 chunking retrieval report

Date: 2026-10-05

## Question

Does layout-preserving, structure-aware chunking improve document-scoped
retrieval relative to the existing fixed token-window strategy when the PDFs,
questions, embedding model, sparse retrieval, reranker, and routing settings
remain fixed?

## Treatments

| Control | Pinecone index | Chunking strategy | Content chunks |
|---|---|---|---:|
| v1 | `agentic-rag-hybrid` | `token_window_v1` | 242 |
| v2 | `agentic-rag-hybrid-v2` | `structure_aware_v2` | 246 |

Both indexes use 384-dimensional MiniLM embeddings at the same pinned revision,
per-document BM25, weighted RRF, and the same local cross-encoder. The corpus is
the same three exact-hash PDFs. The evaluation contains twelve source-reviewed
questions: nine development and three holdout.

Chunk IDs are hashes of chunk text, so the v1 labels cannot identify v2
passages. V2 evidence was reviewed on the same source pages and exported to
`eval_set/chunking_pilot_v2_reviewed_pairs.json`. Retrieval never supplied the
labels used to judge its own output.

## Index integrity

After ingestion, a read-only Pinecone prefix audit found exactly one overview
plus the expected content vectors for every document:

| Document | v1 content | v2 content | Unexpected IDs |
|---|---:|---:|---:|
| Attention Is All You Need | 51 | 58 | 0 |
| Building Effective AI Agents | 27 | 29 | 0 |
| FlashAttention | 164 | 159 | 0 |

This rules out stale-vector contamination for these six document/index pairs.

## Overall results

Counts below are out of twelve questions. `App selected` is the five-passage
content selection used by the application, including its two-slot RRF recall
reserve.

| Stage | Version | Hit@1 | Hit@3 | Hit@5 | In 15 candidates | MRR |
|---|---|---:|---:|---:|---:|---:|
| RRF | v1 | 5 | 7 | 9 | 10 | 0.531 |
| RRF | v2 | 3 | 9 | 10 | 12 | 0.519 |
| Cross-encoder | v1 | 4 | 9 | 9 | 10 | 0.520 |
| Cross-encoder | v2 | 7 | 9 | 11 | 12 | 0.710 |
| App selected | v1 | 4 | 9 | 9 | 9 | 0.514 |
| App selected | v2 | 7 | 9 | 10 | 10 | 0.688 |

Mean retrieval plus reranking time was 1.601 seconds for v1 and 1.616 seconds
for v2 in these disposable CPU containers.

## Development and holdout

| Split | Version | App Hit@1 | App Hit@3 | App Hit@5 | App MRR |
|---|---|---:|---:|---:|---:|
| Development (9) | v1 | 3 | 7 | 7 | 0.519 |
| Development (9) | v2 | 4 | 6 | 7 | 0.583 |
| Holdout (3) | v1 | 1 | 2 | 2 | 0.500 |
| Holdout (3) | v2 | 3 | 3 | 3 | 1.000 |

The holdout result favors v2, but three questions are too few for a general
quality claim. The development slice also shows that v2 is not uniformly
better at every cutoff.

## Per-question application rank

`Miss` means no reviewed supporting chunk survived into the application's five
content passages.

| Case | Split | v1 | v2 | Direction |
|---|---|---:|---:|---|
| Attention architecture | Development | 3 | Miss | Regression |
| Attention scaling | Development | 3 | 2 | Improved |
| Attention multi-head | Development | 1 | 1 | Same |
| Attention positional encoding | Holdout | Miss | 1 | Improved |
| Workflow versus agent | Development | 1 | 1 | Same |
| Augmented LLM | Development | 1 | 1 | Same |
| Evaluator-optimizer | Development | 2 | 2 | Same |
| Agent tradeoffs | Holdout | 1 | 1 | Same |
| IO awareness | Development | 2 | 1 | Improved |
| Tiling | Development | Miss | 4 | Improved |
| Exact attention | Development | Miss | Miss | Unresolved |
| HBM complexity | Holdout | 2 | 1 | Improved |

## Interpretation

V2 is a promising treatment and should remain available for answer-level
testing. It brought reviewed evidence into the 15-candidate pool for all twelve
questions, compared with ten for v1. The cross-encoder made better use of the
v2 candidates, raising top-one evidence hits from four to seven.

Two selection failures remain visible rather than hidden by the aggregate:

1. The Attention architecture evidence was RRF rank 14 and cross-encoder rank
   5 in v2, but the two-slot RRF reserve displaced it from the final five.
2. The exact-attention evidence was present at RRF rank 14 and cross-encoder
   rank 15, so it still missed the application context.

These failures point to ranking and selection work after the chunking decision.
They do not justify changing thresholds or embeddings during this experiment.

## Operational record and limits

- Six bounded Groq primary calls generated ingestion overviews, one per
  document and index. All succeeded on their first attempt.
- The 24 retrieval evaluations made no LLM calls and no writes.
- PDF extraction reported rotated text that may be incomplete.
- This set contains only answerable, document-scoped questions. Negative and
  adversarial cases remain a separate evaluation slice.
- No full answer, citation, grounding, or abstention comparison has run yet.
- Detailed ID/rank outputs are saved locally under the ignored
  `logs/evaluation/chunking-comparison/` directory without passage text.

## Matched application check

On 2026-10-06, the two diagnostic questions were run once against each index
through the FastAPI endpoint. Groq was the only provider, each provider call
had one attempt, and `MAX_RETRIES=0` disabled retrieval rewrites so the initial
retrieval treatment remained isolated.

| Question | Version | Final status | Grounding | Citations | Graph time |
|---|---|---|---|---:|---:|
| Positional encoding | v1 | `unsupported` | Unsupported after correction | 1 | 11.24s |
| Positional encoding | v2 | `answered` | Grounded | 2 | 8.47s |
| Attention architecture | v1 | `answered` | Grounded | 2 | 5.03s |
| Attention architecture | v2 | `answered` | Grounded | 2 | 4.61s |

The positional-encoding result directly demonstrates the chunk-boundary
effect. The v1 cited passage began after the sentence explaining that the model
has no recurrence or convolution, so generation stated that premise without
cited support. One correction still ended unsupported. V2 kept the section
heading and complete opening sentence together; its answer cited the full
reason for adding positional information and the later relative-position
explanation, and the verifier returned grounded without correction.

The predicted Attention architecture content-rank regression did not become an
answer failure. Both versions answered correctly and were grounded. V2 used a
page-10 conclusion passage for the recurrent-layer claim and its generated
overview for the convolution claim. That succeeds operationally, although a
direct PDF citation is preferable to an overview citation and remains a useful
provenance improvement.

V1 used five successful Groq calls across the two questions because positional
encoding required one correction. V2 used four successful calls and required
no correction. No provider attempt failed, and no other provider was contacted.
Full responses and contexts remain in ignored local files
`logs/evaluation/chunking-comparison/app-v1.json` and `app-v2.json`.

## Decision

Keep `structure_aware_v2` as the chunking treatment for subsequent local
evaluation. It improved aggregate retrieval, fixed the selected chunk-boundary
failure in the full graph, did not break the matched architecture answer, and
added negligible retrieval latency. The evidence is sufficient to stop tuning
chunking in this pilot and move to a broader answer-quality baseline.

V2 is not yet a general production-quality claim: the set has twelve positive
questions, only three holdouts, no scanned PDFs, and one full-answer comparison
used an overview citation. The next evaluation stage should expand answerable,
unanswerable, and adversarial questions across more documents while keeping
this chunking strategy fixed.

That expanded implementation-level evaluation is now complete. See
[`CHUNKING_V2_BROAD_EVALUATION.md`](CHUNKING_V2_BROAD_EVALUATION.md) for the
five-document, 30-question dataset, 50 matched retrieval runs, 16 full API
queries, chunk-level citation analysis, and the resulting repair priorities.
