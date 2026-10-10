# V2 answer-reliability acceptance gate

This gate turns the 2026-10-08 manual Prompt Engineering, LoRA, and
FlashAttention run into repeatable release evidence. It measures the complete
answer path after `structure_aware_v2` retrieval while keeping chunking,
embeddings, reranker thresholds, provider models, and index contents fixed.

## Why this gate exists

The run exposed failures in different layers:

| Evidence | Conclusion |
|---|---|
| The LoRA two-matrix question had a top reranker score of `0.9515` but ended unsupported | The defect was generation or verification, not retrieval |
| A vague FlashAttention standalone query scored `0.0003` | Contextualization lost the subject needed by retrieval |
| The document-diagram request reused the prior architecture question | A broad anaphoric format rule discarded the current information need |
| That diagram path used three retrievals, three grades, two rewrites, generation, correction, and verification in 10.54 seconds | Wrong planning caused avoidable retries and latency |
| Five of eight internally grounded answers required correction | First-pass generation and citation compliance need measurement |
| Groq returned visually valid citations containing alternate brackets, line suffixes, or invisible format characters | Citation parsing needs provider-independent canonicalization |

An internal `grounded` verdict is a runtime result. It is not treated as human
ground truth.

## Runtime repair contract

The query plan separates three independent decisions:

```text
source: document | general | control | clarify
relationship: standalone | follow_up | verify_previous
information_need_source: current_turn | previous_information_need
response_format: requested | prose | bullets | table | ascii_flowchart | math
standalone_query: string | null
needs_clarification: boolean
```

`previous_information_need` is accepted for a presentation-only transformation
only when both the structured plan and a narrow deterministic grammar agree.
The word `that` or `it` cannot replace the current query by itself.

Examples:

| Message | Information need | Presentation |
|---|---|---|
| `Put that in a flowchart.` | Previous | ASCII flowchart |
| `Is there a diagram in the document that explains the architecture?` | Current | Requested/default |
| `Create a diagram explaining FlashAttention.` | Current | ASCII flowchart |
| `Can you explain that diagram?` | Current, resolved with history | Prose/default |

## Citation and correction contract

- Canonical `[S#]`, alternate source brackets, optional line suffixes, and
  invisible Unicode format characters inside a marker normalize to `[S#]`.
- Normalization is limited to citation candidates; unrelated text is unchanged.
- Unknown labels remain invalid after normalization.
- Inside a fenced textual diagram, exact structural labels such as `Start`,
  `End`, `Yes`, and `No` need no citation. Factual nodes still require one.
- Generation cannot infer author motivation, causality, maximality, optimality,
  or exhaustive absence without explicit cited evidence.
- Correction removes or qualifies the listed unsupported claim. It cannot
  invent a replacement rationale or claim that the entire document lacks an
  artifact when only selected excerpts were retrieved.

## Reviewed acceptance set

`eval_set/answer_reliability_v2_acceptance.json` freezes eleven sanitized
questions and their invariants. It stores no document passages or provider
prompts. Each case defines:

- source mode;
- current or previous information need;
- expected presentation;
- allowed final statuses;
- required retrieval-query subject terms;
- citation requirement;
- maximum retrieval, rewrite, and correction counts;
- forbidden unsupported claims.

The observed baseline is:

| Metric | Baseline |
|---|---:|
| Queries | 11 |
| Answered and internally grounded | 8 |
| Unsupported after correction | 2 |
| Abstained | 1 |
| Corrected answered cases | 5 / 8 |
| Document-diagram graph time | 10.54 seconds |

## Verification levels

Reports must name the level they establish:

| Level | What it proves |
|---|---|
| `OFFLINE_CONTRACT` | Pure policy and schema invariants |
| `GRAPH_REPLAY` | Node sequence, query lineage, result status, and call bounds with deterministic fakes |
| `PACKAGED_IMAGE` | Installed imports and required cached assets inside the built image |
| `INFRA_HEALTH` | PostgreSQL, API, and Streamlit start together without runtime downloads |
| `LIVE_E2E` | Real index, provider, citations, answer, and final verdict |

A lower level cannot be reported as proof of a higher one.

## Packaged-image evidence — 2026-10-09

Work Package 2 built `agentic-rag-app:wp2-candidate` from the uncommitted review
worktree. Docker reused the frozen dependency, model, and NLTK layers; only the
source and installed-project layers changed.

| Check | Result |
|---|---|
| Image identity | `sha256:fb3604c75a6789aedc22a86e166ddc1313d1b45105e57421b268e6c7dd54322b` |
| Runtime imports | Pinecone SDK and CPU-only PyTorch `2.13.0+cpu` passed |
| Installed repair | `information_need_source` and citation normalization passed |
| Baked retrieval models | Embedding and cross-encoder revisions loaded with `--network none` |
| Baked BM25 assets | `punkt_tab` and `stopwords` verified with `--network none` |
| FastAPI package import | Passed with placeholder credentials and `--network none` |
| Containerized offline suite | 180 passed, one existing dependency warning |
| Storage impact | 3.92 GB logical, 3.919 GB shared, approximately 643 KB unique |

This establishes `PACKAGED_IMAGE`. It does not establish database connectivity,
HTTP health, Pinecone retrieval, provider behavior, or answer quality. The
candidate tag remains local until review.

## Offline evaluation command

The evaluator consumes one sanitized JSON object per line. Answer text may be
supplied for forbidden-claim checks, but the generated report omits it.

```powershell
$env:PYTHONPATH='src'
python scripts/evaluate_answer_reliability.py `
  --results logs/evaluation/answer-reliability-v2/results.jsonl `
  --output-dir logs/evaluation/answer-reliability-v2/report
```

It writes:

- `run_manifest.json` — contract identity and privacy boundary;
- `query_results.jsonl` — sanitized per-case pass/fail evidence;
- `summary.md` — reviewable release report.

The command exits nonzero when any case fails.

## Release gates

- Every labeled request uses the selected source and document.
- The document-diagram question keeps `current_turn`, includes the visual and
  FlashAttention subject in its retrieval query, and uses at most two
  retrievals and one rewrite.
- The LoRA answer does not introduce the unsupported fixed-budget maximality
  rationale. It either answers from evidence or states the evidence limit.
- Every answered document case has valid citations.
- Every labeled unanswerable case abstains.
- Previously successful factual, flowchart, mathematical, and Auto-mode cases
  do not regress.
- Corrected answered cases should fall from five toward two or fewer in this
  small set. Because provider output can vary, the hard gates are the route,
  status, citation, claim, and call-count invariants.

The next live acceptance run occurs only after offline review and packaged-image
verification. Its results will be appended to this document rather than
overwriting the baseline.
