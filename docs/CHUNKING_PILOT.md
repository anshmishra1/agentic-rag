# Chunking pilot

This pilot measures whether a chunking change improves retrieval before the
application migrates more documents. It separates three concerns:

1. **Document identity** - exact PDF SHA-256 values prevent evaluating different
   file revisions by accident.
2. **Passage labels** - a reviewer records the page and every current chunk that
   directly supports each question.
3. **Retrieval comparison** - the same reviewed questions and source passages
   are run against the current and candidate chunking strategies. Development
   questions guide changes; holdout questions are inspected only for the final
   comparison.

## Pilot corpus

The tracked manifest uses three structurally different research documents:

- a conventional research paper (`AttentionIsAllyouNeed.pdf`);
- a web article exported as PDF (`Building Effective AI Agents _ Anthropic.pdf`);
- a dense systems research paper with equations (`FlashAttention Fast and Memory-Efficient Exact Attention.pdf`).

It contains twelve source-reviewed questions: three development and one holdout
question per document. Each question records the PDF pages, current deterministic
chunk IDs, and a short explanation of the supporting evidence.

## Prepare the local review catalog

Run in the self-contained application image with networking disabled. Mount the
research directory read-only. Do not pass an env file or provider credentials.

```powershell
docker run --rm --network none `
  --mount type=bind,source="D:\Research_Articles",target=/research,readonly `
  --mount type=bind,source="$PWD",target=/workspace `
  --workdir /workspace `
  -e MODEL_LOCAL_FILES_ONLY=true `
  agentic-rag-app:local `
  python scripts/prepare_chunking_pilot.py --pdf-dir /research
```

The generated `logs/evaluation/chunking-pilot-v1/chunk_catalog.json` contains
full extracted text and remains ignored by Git. The command fails if a PDF hash,
model revision, chunk size, or overlap differs from the manifest. The CLI
imports application settings from a temporary empty working directory, so it
does not load the project `.env`.

Generate the v2 structure-aware comparison catalog without exporting v1 labels:

```powershell
docker run --rm --network none `
  --mount type=bind,source="D:\Research_Articles",target=/research,readonly `
  --mount type=bind,source="$PWD",target=/workspace `
  --workdir /workspace `
  -e MODEL_LOCAL_FILES_ONLY=true `
  agentic-rag-app:local `
  python scripts/prepare_chunking_pilot.py --pdf-dir /research `
    --strategy structure_aware_v2 `
    --catalog-output logs/evaluation/chunking-pilot-v2/chunk_catalog.json
```

## Review and export

The initial manifest was reviewed with this process:

1. find the answer in the source PDF independently;
2. record the PDF page label in `source_pages`;
3. record every catalog chunk that directly supports the answer in
   `relevant_chunk_ids`;
4. explain the evidence briefly in `review_notes`;
5. set `review_status` to `verified`.

Do not select a chunk merely because the retriever ranked it. Gold labels must
come from reading the source passage first. Keep holdout labels hidden from
chunking decisions after review.

To repeat validation and export the source-positive retrieval set:

```powershell
docker run --rm --network none `
  --mount type=bind,source="D:\Research_Articles",target=/research,readonly `
  --mount type=bind,source="$PWD",target=/workspace `
  --workdir /workspace `
  -e MODEL_LOCAL_FILES_ONLY=true `
  agentic-rag-app:local `
  python scripts/prepare_chunking_pilot.py --pdf-dir /research `
    --pairs-output eval_set/chunking_pilot_reviewed_pairs.json
```

This pilot establishes source-positive chunk recall. Wrong-document and
unanswerable cases remain a separate evaluation slice because absence must also
be reviewed rather than inferred from a filename.

## What changes between v1 and v2

Only the parsing and chunking treatment changes in the first comparison.

| Property | v1 control | v2 treatment |
|---|---|---|
| PDF extraction | plain text | layout-preserving text |
| Unit | recursive token window | heading, sentence, or bullet |
| Packing | 240 tokens, 40-token overlap | whole units up to 240 tokens, no synthetic overlap |
| Provenance | loader page | section title and page range |
| Embedding model | MiniLM-L6-v2 at the pinned revision | identical |
| Questions and PDFs | reviewed pilot set | identical |

An offline catalog snapshot across the three pilot PDFs produced 242 v1 chunks
and 246 v2 chunks. The v2 catalog attached a section to 243 chunks and allowed
47 chunks to continue across a page boundary within the same section. Both
strategies stayed within the 240-token embedding cap. V2 had 14 chunks below
32 tokens versus one in v1, so short structural units are a measured tradeoff
to examine in retrieval results rather than silently tuning away.

These catalog statistics prove that v2 is a different, reproducible treatment;
they do not prove that it retrieves better. The existing `relevant_chunk_ids`
are v1-specific because chunk IDs are hashes of chunk text. Before comparing
recall, review and export the supporting v2 chunk IDs for the same source pages.
Then ingest the same PDFs into separate v1 and v2 indexes and compare Recall@k,
MRR, reranker retention, and answer citation coverage on the same questions.

The first controlled retrieval comparison is now recorded in
[`CHUNKING_V1_V2_RETRIEVAL_REPORT.md`](CHUNKING_V1_V2_RETRIEVAL_REPORT.md).
Its v2 evidence labels are in
`eval_set/chunking_pilot_v2_reviewed_pairs.json`.
