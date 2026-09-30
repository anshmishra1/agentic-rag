# Reranker content recall safeguard — 2026-09-30

The clean-index diagnostic in `docs/RERANKER_EVALUATION_REPORT.md` found that
the local cross-encoder improved the original selected ten support cases
overall, but moved the CS229 syllabus passage for `q061-source` from RRF rank
2 to cross-encoder rank 11. The app kept only five content passages, so the
source passage was lost before generation.

The fix keeps the first two RRF content candidates within the existing five
content slots, replacing the lowest cross-encoder selections when necessary.
The retained passages remain sorted by cross-encoder score. Overview
selection and all numeric retrieval score thresholds are unchanged. The
rank reserve is configurable with `RERANK_CONTENT_RRF_RESERVE` (default `2`,
`0` disables it). The five-passage budget remains unchanged.

An offline replay of the actual selection function on saved clean-index
candidate IDs changed four of 15 selected diagnostic candidate sets. With
the ten original reviewed support labels plus two explicitly reviewed
post-cleanup cases, known support in the top five was 11/12 for pure RRF,
11/12 for pure cross-encoder, and 12/12 with the safeguard. Rank-1 coverage
was 5/12, 7/12, and 7/12 respectively; top-3 coverage was 9/12, 11/12,
and 11/12. The two added labels were inspected after the candidate lists
were known, so these are diagnostic case counts, not a population estimate.
Saved replay: ignored `logs/evaluation/20260930-reranker/guard_replay.json`.

All 97 offline unit tests passed in the existing API image with networking
disabled and no private `.env` mount. A single approved live `q061-source`
query used Groq only with one provider attempt per call and one graph retry.
The page-2 supporting passage appeared as final content result 5 and was
cited as `[S5]`. The answer was **not** verified: Groq returned an empty
verifier response on both checks, and the app correctly returned
`answer_status=verification_uncertain` and `grounded=false`. The answer and
trace are in ignored `logs/evaluation/20260930-reranker/live_probe_result.json`
and `logs/runs/`; they may contain source text. No ingestion or Pinecone
write occurred. PostgreSQL was stopped after the query with its volume
preserved.

This fixes the observed loss of the CS229 passage, but does not prove answer
quality across documents. The empty Groq verifier response is a separate
provider/output-format issue to diagnose before treating the live answer
as grounded. The current response fails closed, so it is safe to review the
recall change without presenting that answer as verified.
