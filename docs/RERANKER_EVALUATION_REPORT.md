# Clean-index reranker diagnostic — 2026-09-30

## Question and method

Does the local cross-encoder move a known supporting passage higher than
hybrid dense/BM25 retrieval with RRF? This is a passage-ranking check, not an
answer-quality score. We used the cleaned Pinecone index, the saved BM25
state, `HYBRID_CANDIDATE_K=15`, and the existing local cross-encoder. The
collector ran the application's content-retrieval and reranking functions
without ingesting, querying an LLM, or writing Pinecone data. It saved IDs
and ranks, not document text, to ignored
`logs/evaluation/20260930-reranker/reranker_comparison.json` (SHA-256
`6951280cfdc4e5d4e8b1d6b2e2568f57ee33b2396a46c7ad16397347f851f5fe`).

The original 15-case passage review supplied ten known-support cases, two
cases with no support in the passages previously inspected, and three
unresolved cases. Because these cases and passages were selected from earlier
retrieval results, the figures below are **diagnostic sample results**, not
an unbiased estimate for future questions. We did not count absent support
as irrelevance unless the entire candidate set was reviewed.

## Original ten known-support cases

| Case | RRF rank | Cross-encoder rank |
| --- | ---: | ---: |
| q051-source | 2 | 1 |
| q001-source | 5 | 1 |
| q021-source | 1 | 1 |
| q026-source | 1 | 1 |
| q043-source | 2 | 3 |
| q046-source | 1 | 1 |
| q031-source | 1 | 1 |
| q033-source | 5 | 2 |
| q038-source | 2 | 2 |
| q038-other | 1 | 1 |

| Measure | RRF | Cross-encoder |
| --- | ---: | ---: |
| Known support at rank 1 | 5/10 | 7/10 |
| Known support in top 3 | 8/10 | 10/10 |
| Known support in top 5 | 10/10 | 10/10 |
| Mean reciprocal rank | 0.690 | 0.833 |

The cross-encoder improved this selected set overall. It moved the known
support for `q043-source` from rank 2 to rank 3, but retained it in the
application's top-five content budget.

## Two formerly unresolved cases

Offline inspection of current PDF chunks resolved two cases after the clean
index run. The reviewed IDs and evidence notes are in
`eval_set/retrieval_clean_index_review.json`. These labels were added after
seeing the new candidate lists, so they are case studies rather than a new
holdout metric.

- `q053-source`: the page-120 chunk explains why a RAG pipeline divides large
  documents into manageable chunks. RRF put it at rank 6; the cross-encoder
  moved it to rank 2, inside the app's top-five content budget.
- `q061-source`: the page-2 CS229 course description lists supervised,
  unsupervised, learning theory, and reinforcement learning. RRF put it at
  rank 2; the cross-encoder moved it to rank 11, **outside** the app's
  top-five content budget. This is a concrete reranker recall failure.

`q033-other` still has no verified single supporting passage in the inspected
top candidates. It remains unresolved. No broad negative-case or answer
accuracy claim follows from this sample.

## Decision and next check

The cross-encoder should remain: it improved more selected support ranks than
it hurt. A narrow recall safeguard should keep the first two RRF content
candidates within the five content slots while otherwise preserving
cross-encoder order. On these saved lists, that would retain the `q061` page-2
passage without removing any known supporting passage from the original ten
cases. This is a rank-selection rule, not a new score threshold. It still
needs an offline replay, regression tests, and one bounded live app check.

The pilot does not establish generalized hit@k, semantic grounding accuracy,
or a calibrated routing threshold. A larger blind passage-label set remains
valuable, but this specific top-five loss is enough to justify a focused
recall fix before moving on.
