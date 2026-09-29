# Retrieval evidence pilot — 2026-09-28

This report makes the saved six-PDF retrieval run inspectable. It uses the
46 ranked-score rows collected on 2026-09-28 under
`logs/evaluation/20260928T092147Z/all_scores.json` and the local PDFs.
The full question set and document-level review are in
`eval_set/retrieval_reviewed_pilot.json`. The manually reviewed passage
judgments and exact chunk IDs are in
`eval_set/retrieval_chunk_review_pilot.json`.

## How to inspect the evidence

Open the chunk review JSON beside the named PDF. Each `confirmed_support`
entry gives the saved final rank, chunk ID, PDF page label, and cross-encoder
score. The saved score file has the query and all final ranked IDs. A local
reconstruction of passage text is in ignored
`logs/evaluation/20260928T155128Z/reconstructed_chunks.json`; it may contain
copyrighted document text and should remain local.

To reproduce the reconstruction, use the existing API image with Docker
networking disabled. From the repository root in PowerShell:

```powershell
$cache = "$env:USERPROFILE\.cache\huggingface"
docker run --rm --network none -e DEBUG=false -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 `
  -v "${PWD}\src:/app/src:ro" -v "${PWD}\scripts:/app/scripts:ro" -v "${PWD}\PDF:/app/PDF:ro" `
  -v "${PWD}\logs/evaluation/20260928T092147Z:/app/scores:ro" `
  -v "${PWD}\logs/evaluation/20260928T155128Z:/app/output" `
  -v "${cache}:/home/appuser/.cache/huggingface:ro" `
  agentic-rag-api:latest python /app/scripts/reconstruct_eval_chunks.py `
  /app/scores/all_scores.json /app/PDF /app/output/reconstructed_chunks.json
```

The script checks each PDF's SHA-256 against the saved document ID, uses the
application's PDF loader and chunker, and matches content by its stable ID.
It makes no Pinecone or LLM request and does not load `.env`.

## What the review found

The 15 diagnostic cases span all six PDFs. Ten document-positive cases have
at least one confirmed sufficient **content** passage in the saved final
ranking. Six of those have confirmed support at rank 1; four first show it
at ranks 2, 3, or 5. Two document-negative cases have no support in the five
reconstructed content passages inspected. Three cases remain unresolved.
These are selected cases, so `10/10` is **not** an estimate of retrieval
accuracy or recall.

| Case | Document | Review of saved final ranking |
| --- | --- | --- |
| `q051-source` | AI Engineering | RAG purpose supported at content rank 3, page 182; ranks 1–2 are unreconstructed overviews. |
| `q001-source` | Explainable AI | Explanation purpose supported at rank 1, page 6. |
| `q021-source` | ML eBook | Supervised learning supported at rank 1, page 2. |
| `q026-source` | ML eBook | Regularization supported at rank 1, page 5. |
| `q043-source` | DevOps Troubleshooting | Slow-server resource areas supported at rank 5, page 18. |
| `q046-source` | DevOps Troubleshooting | Traceroute purpose supported at rank 1, page 80. |
| `q031-source` | Hands-On LLM | Representation versus generation supported at rank 1, page 9. This is the case whose previous answer had uncited claims. |
| `q033-source` | Hands-On LLM | Semantic search definition supported at rank 2, page 316; rank 1 is an index entry. |
| `q038-source` | Hands-On LLM | Fine-tuning purpose supported at rank 2, page 488; rank 1 is only partial. |
| `q038-other` | AI Engineering | Alternate-document fine-tuning definition supported at rank 1, page 68. |
| `q001-other`, `q003-other` | ML eBook | No sufficient explanation/causality passage among five reconstructed content hits in either case. |
| `q033-other` | AI Engineering | Document-level positive, but the five content hits mention related search concepts without a clear one-passage definition. Unresolved. |
| `q053-source` | AI Engineering | Top content hit discusses using chunks, but the role of chunking is not clear; one hit could not be reconstructed. Unresolved. |
| `q061-source` | CS229 | Three of five content hits and both overview hits lack reconstructed text. Unresolved. |

## Measurement limits and next actions

Across the **whole** saved snapshot, 155 of 230 final ranked content hits
matched chunks reconstructed from the exact local PDFs; 75 did not. The
script fails on PDF hash mismatch, so these are chunk-text/ID differences,
not different PDF bytes. Possible causes include stale indexed vectors or
processing-version differences; this offline reconstruction cannot identify
which. The 84 overview candidates cannot be reconstructed because their
text was generated during ingestion. Missing text stays unjudged.

The snapshot records final reranked results, not the candidate order before
the cross-encoder. It therefore shows useful and weak final passages but
**does not prove whether the reranker improved or harmed ranking**. The
cross-encoder score is a ranking signal, not a probability. No numeric
threshold was changed from this review.

Next, capture pre-rerank candidate IDs and final IDs in a bounded retrieval
diagnostic, then label a larger balanced development set and keep holdout
cases separate. Investigate unmatched IDs before claiming chunk recall.
The earlier offline regression suite passed 87 tests. The authorized
single-question answer check is recorded below; it does not measure general
answer quality or reranker performance.

## Authorized single-question live check — 2026-09-29

After explicit authorization, one `q031-source` `/query` request ran against
the already indexed Hands-On LLM PDF. The API used the merged grounding code,
Pinecone retrieval, Groq only, one attempt per provider call, the normal
two-retry graph setting, and the existing read-only model cache. It performed
no ingestion. The result and container trace are local under ignored
`logs/evaluation/20260929T042310Z/`; the result file contains answer and
passage text and should stay local. PostgreSQL was stopped afterward with
its volume preserved.

The retrieval decision was `grade` because of close candidates; the Groq
grader said `relevant`. The first generation was followed by an **empty,
unparseable verifier response**. The graph treated it as `unsupported` and
used its single corrective regeneration. The second verifier response parsed
as `grounded`. The final API response was `answered`, `grounded=true`, with
citations `[S1]` (page 9) and `[S4]` (overview), and zero retrieval retries.
Manual review found that the final answer's model distinction and examples
were supported by its cited page 9 passage; its factual table cells and
closing statement carried citations. There were five successful Groq calls:
three fast-tier and two primary-tier. No other provider was called.

This confirms that the merged app can produce a cited, supported answer for
this one case and that malformed verification fails closed. Because the first
verifier returned no verdict, this run **did not exercise a live semantic
rejection of uncited claims**. The offline regression probe remains the
evidence for that specific new rule. One question cannot establish general
answer accuracy, reranker quality, or token cost. No threshold changed.

## Follow-up index audit

The later read-only audit established that the unmatched content IDs are
real indexed vectors, not merely IDs derived incorrectly by the score
collector. All current content IDs are present, alongside 5,567 extra
content IDs and 16 extra overview/legacy IDs across five of six PDFs.
The ML eBook is clean. This contamination must be repaired before using
these PDFs to judge reranker quality. See
`docs/RETRIEVAL_EVALUATION_RATIONALE.md` for the per-document counts, test
reasoning, dry-run cleanup plan, and remaining evaluation gate. No Pinecone
vectors were deleted during the audit.
