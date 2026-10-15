# Final evidence selection

## Problem

The retrieval node independently selected five content passages and two
whole-document overviews, then globally sorted all seven by cross-encoder
score. Generation and verification later kept only the first five documents.
That second truncation could discard direct content already retained by the
RRF recall safeguard, especially when one or both overview scores were high.

This was a boundary mismatch between retrieval ranking and answer-context
assembly. The cross-encoder score remains useful for retrieval assessment, but
an overview summary is not a substitute for direct passage evidence when the
answer must provide page-level citations.

## Repair

Retrieval results remain globally score ordered for confidence metrics and
semantic grading. At the answer boundary, one shared pure policy now:

1. retains up to five content passages in their existing reranker order;
2. appends at most the highest-ranked overview as supplemental context; and
3. supplies that identical sequence to generation, correction, citation
   labeling, and grounding verification.

The direct-content cap remains `MAX_GENERATION_CONTEXT_DOCUMENTS=5`.
`MAX_GENERATION_CONTEXT_OVERVIEWS=1` controls the separate supplemental
overview budget. The existing 12,000-character prompt cap remains the final
size bound.

## What this changes

- An overview can no longer consume a direct-content slot.
- The five-slot RRF recall safeguard survives through prompt construction.
- Citation labels seen by the answer model and grounding verifier stay aligned.
- At most one overview is exposed, reducing redundant summary evidence.

## What remains unchanged

- candidate retrieval, weighted RRF, and cross-encoder inference;
- the top-five content selector and its two-candidate RRF reserve;
- chunking, embeddings, provider models, routing thresholds, graph topology,
  Pinecone data, and PostgreSQL data.

## Evidence and limit

Offline policy tests use an adversarial ordering in which two overviews score
ahead of five direct passages. They confirm that all five direct passages reach
the answer context, only the best overview is appended, and the verifier maps
the same `[S1]` through `[S6]` labels.

The focused generation, citation, and verifier suite passed 22 tests. The
complete offline suite collected 184 tests, reached 100 percent, and returned
pytest exit code 0 with Hugging Face and Transformers networking disabled. The
ordinary pytest CLI left an existing interpreter shutdown thread alive after
completion; the verified run explicitly exited only after `pytest.main()`
returned its successful result. All source and unit-test Python files passed
AST parsing, and `git diff --check` passed.

This repair prevents downstream loss of already selected evidence. It does not
recover a relevant content passage that the cross-encoder ranks below the
five-content cutoff. The broad v2 DPO case, whose reviewed passage appeared at
cross-encoder rank 10, therefore remains the next measured selector problem.
Changing that cutoff or adding a second-stage expansion requires a separate
comparison because extra marginal passages can reduce answer precision.
