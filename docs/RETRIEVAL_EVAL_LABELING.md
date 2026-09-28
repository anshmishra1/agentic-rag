# Retrieval evaluation labels

The app answers questions about one selected document. Calibrate routing with
question/document pairs rather than the 80-question answer set alone. Cross-
document questions need a separate evaluation because the current query API
accepts only one `document_id`.

## Prepared review set

`eval_set/retrieval_label_candidates.json` contains 48 proposed pairs
from 24 of the existing questions: one source-document pair and one alternate-
document pair per question. Six local PDFs are represented. The original set's
ten cross-document questions and ten questions referencing the missing
`Foundation_of_LLMS_TongXiao(1).pdf` are excluded. The local file is named
`Foundation_of_LLMS_TongXiao.pdf`; do not silently equate their contents.

`eval_set/retrieval_reviewed_pilot.json` contains 46 pairs checked against
the local PDFs, with concise evidence notes. Two remain out of the scored set:
`q068-source` asks for a rationale the CS229 prerequisite list does not state,
and `q031-other` may be answerable from the alternate AI Engineering book.
Two proposed negatives (`q033-other` and `q038-other`) were relabeled positive
because the alternate book also supports those answers. The exported
`eval_set/retrieval_reviewed_pairs.json` has 35 development and 11 holdout
pairs. These document-level labels are ready for score collection, subject to
confirming that the exact PDF bytes are indexed. Chunk-level relevance is
still unreviewed.

The `document_id` is SHA-256 of the exact local PDF bytes, matching the
ingestion ID algorithm. This does **not** prove that the PDF is currently
indexed. Development and holdout pairs share a split for each question. The
holdout checks unseen questions from these six PDFs, not new documents.

## Review each pair

1. Check the `target_document` itself. For source pairs, inspect the cited
   `source_pages` and confirm that the answer is supported there. PDF page
   labels may differ from physical page indexes. For alternate-document pairs,
   check whether that document independently answers the question; topic
   overlap means a proposed negative may actually be positive.
2. Add `should_match: true` or `false` based on the target document, not the
   `candidate_should_match` suggestion. Add a brief `review_notes` explanation
   with page or section evidence. Set `review_status` to `verified` only after
   checking. Remove or replace ambiguous cases rather than guessing.
3. Leave `relevant_chunk_ids` empty until actual indexed chunk identities are
   reviewed. Page-level references are useful for review but do not establish
   retrieval hit@k at chunk level.

The export command refuses pending rows, missing reviewed labels, and missing
review notes. It writes only the fields accepted by the score collector:

```powershell
$env:PYTHONPATH='src'
python -m agentic_rag.evaluation.prepare_retrieval_labels --export-reviewed eval_set/retrieval_reviewed_pilot.json --output eval_set/retrieval_reviewed_pairs.json
```

To regenerate the candidate set after changing the sample size, run the same
module without `--export-reviewed`. Regeneration overwrites the output and
resets review fields, so keep reviewed work in a separate file first.

## Score collection and analysis

Only after labels are reviewed and the exact PDFs are indexed, a separately
authorized `--live` run can collect retrieval-only scores. It reads PostgreSQL
and Pinecone and runs the local embedder/reranker; it makes no LLM calls and
does not ingest. The saved snapshot includes routing metrics and ranked chunk
IDs derived from the current vector-ID algorithm, types, pages, and scores,
without passage text. Confirm those IDs against the live index before using
them as chunk-level gold labels, especially for older ingestions. Treat query
text and document IDs in that snapshot as potentially sensitive.

The offline analyzer picks a provisional top-score floor using **development**
rows only, then reports holdout errors at that fixed floor. If development
scores overlap, it offers no floor. A top-score floor alone cannot calibrate
the current ratio, gap, and overview gates; their routing effects require
evaluation against the same reviewed cases before changing configuration.
No numeric routing setting should change until actual scores and routing
outcomes have been evaluated on the reviewed cases.

## First bounded diagnostic (2026-09-28)

The PostgreSQL registry contained two document IDs, but only one matched the
six exact PDF hashes in this set. Eight reviewed pairs for
`Explainable-AI-for-Practitioners.pdf` were eligible (six development, two
holdout). Retrieval-only collection saved the scores and an offline policy
replay under the ignored `logs/evaluation/20260928T071746Z/` directory.
The API image was reused and PostgreSQL was stopped afterward; no ingestion
or LLM request was made.

On these eight cases, all four negative pairs routed to `rewrite_query`.
Of the four positive pairs, two routed to `generate`, one to `grade`, and
`q001-source` routed to `rewrite_query` with reason
`weak_candidate_separation`. Its top content chunk was on the cited page 6
with score about 0.999; several other candidates scored nearly as high.
The current ratio-and-gap rule treated that close group as weak evidence.
The development sample contains only three positives and three negatives,
so the analyzer correctly made no threshold recommendation. This identifies
a case to test across more documents, not a justified new numeric setting.
