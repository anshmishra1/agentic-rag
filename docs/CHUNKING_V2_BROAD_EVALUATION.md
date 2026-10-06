# Broad v1 versus v2 implementation evaluation

Date: 2026-10-06

## Purpose

This evaluation tests whether structure-aware v2 chunking improves the running
application rather than only improving an offline chunk statistic. It exercises
the real PDF loader, chunker, BM25 state, Pinecone hybrid retrieval, weighted
RRF, local cross-encoder, final passage selection, Groq generation, citation
extraction, grounding verification, correction, abstention, FastAPI response,
and persistent logging paths.

The experiment keeps the embedding model, sparse encoder, retrieval settings,
reranker, questions, PDFs, and application image fixed. The only intended
retrieval-schema difference is the chunking and PDF extraction strategy:

| Version | Index | Strategy |
|---|---|---|
| v1 control | `agentic-rag-hybrid` | `token_window_v1` |
| v2 treatment | `agentic-rag-hybrid-v2` | `structure_aware_v2` |

## Expanded corpus and question set

Five additional PDFs were selected from the local research collection to add
web-exported prose, mathematical objectives, systems design, numeric claims,
tables, and false-premise questions:

1. `Demystifying evals for AI agents _ Anthropic.pdf`
2. `Direct Preference Optimization.pdf`
3. `Efficient Memory Management for Large Language.pdf`
4. `LoRa.pdf`
5. `Training Compute-Optimal Large Language Models.pdf`

All 22 available research PDFs passed an offline extractability audit. These
five extend the original three-document pilot to eight evaluated document
styles without making the live test unnecessarily large.

The new set contains 30 source-reviewed cases:

- 20 ordinary answerable questions;
- five questions with a false premise that the answer should explicitly reject;
- five genuinely unanswerable questions that should end in abstention.

Every source-supported case was tied to source pages and exact evidence phrases
before retrieval ran. Because chunk IDs depend on chunk text, v1 and v2 have
separate gold chunk IDs. The tracked case and pair files contain no full passage
text; full catalogs and responses remain under ignored `logs/evaluation/`.

## Ingestion and index integrity

The same five exact PDF files were ingested into each versioned index. Groq was
the only configured provider and generated one overview per document.

| Document | v1 content chunks | v2 content chunks |
|---|---:|---:|
| Agent evaluation article | 53 | 41 |
| DPO | 137 | 140 |
| PagedAttention / vLLM | 108 | 94 |
| LoRA | 129 | 134 |
| Chinchilla | 144 | 120 |
| **Total** | **571** | **529** |

A read-only Pinecone prefix audit checked all ten document/index combinations.
Every combination contained exactly the expected content vectors, one overview,
and no unexpected vector IDs.

## Retrieval and reranking results

The 25 source-supported and false-premise questions were run once against each
index. These 50 retrieval evaluations called Pinecone and the local reranker but
did not call an LLM or write vectors.

| Stage | Version | Hit@1 | Hit@3 | Hit@5 / final recall | Candidate recall | MRR | Mean time |
|---|---|---:|---:|---:|---:|---:|---:|
| RRF | v1 | 12/25 | 19/25 | 22/25 | 24/25 | 0.619 | 1.397s |
| RRF | v2 | 11/25 | 18/25 | 20/25 | 25/25 | 0.591 | 1.388s |
| Cross-encoder | v1 | 15/25 | 17/25 | 21/25 | 24/25 | 0.679 | 1.397s |
| Cross-encoder | v2 | 14/25 | 20/25 | 23/25 | 25/25 | 0.714 | 1.388s |
| Application selection | v1 | 15/25 | 17/25 | 19/25 | 19/25 | 0.647 | 1.397s |
| Application selection | v2 | 14/25 | 20/25 | 23/25 | 23/25 | 0.703 | 1.388s |

V2 therefore raised reviewed-evidence recall in the final application context
from 76% to 92%, while mean retrieval time was effectively unchanged. V1 had
one additional top-one hit, but v2 retained four additional supported cases in
the five-passage context and achieved higher MRR.

The most useful case-level changes were:

| Case | v1 final rank | v2 final rank | Interpretation |
|---|---:|---:|---|
| DPO direct objective | Miss | 2 | v2 recovered the direct policy-loss evidence |
| DPO evaluation method | Miss | 4 | v2 recovered the GPT-4 win-rate evidence |
| vLLM block-size tradeoff | Miss | 1 | v2 kept the complete ablation conclusion |
| LoRA mechanism | Miss | 4 | v2 recovered the frozen-weight / low-rank mechanism |
| Chinchilla false premise | 4 | 2 | v2 promoted the corrective evidence |
| DPO partition cancellation | Miss | Miss | v2 put the gold evidence in the candidate pool at cross-encoder rank 10, but final selection still dropped it |
| vLLM contiguous-memory false premise | Miss | Miss | unresolved ranking/selection case |

The DPO partition result is now a passage-selection problem rather than a
chunk-generation problem: v1 never retrieved the reviewed evidence, while v2
retrieved it but ranked it below the five-passage budget.

## Matched end-to-end application results

Six diagnostic questions were run through the complete API against both
indexes. Retrieval rewrites were disabled, provider attempts were limited at
the application layer, and Groq was the only provider.

| Case | v1 final result | v2 final result | v1 gold in context | v2 gold in context | v1 gold cited | v2 gold cited |
|---|---|---|---:|---:|---:|---:|
| DPO direct objective | Answered, grounded | Answered, grounded | No | Yes | No | Yes |
| DPO partition cancellation | Unsupported | Unsupported | No | No | No | No |
| DPO evaluation method | Unsupported | Answered, grounded | No | Yes | No | Yes |
| vLLM block-size tradeoff | Unsupported | Unsupported | No | Yes | No | No |
| LoRA mechanism | Verification uncertain | Answered, grounded | No | Yes | No | Yes |
| Chinchilla false premise | Unsupported | Unsupported | Yes | Yes | Yes | No |

On these six matched cases, v2 increased exact reviewed evidence in the final
context from one case to five and increased exact reviewed evidence cited by the
answer from one case to three. Three v2 cases finished answered and grounded,
compared with one v1 case.

The strict gold metric is conservative. V1 answered the DPO objective correctly
using alternate supporting passages that were not among the exact evidence
anchors. The result remains a valid answer, but v2 additionally cited the
pre-reviewed direct passage.

Two v2 answers were substantively correct and had the reviewed passage in
context, yet finished unsupported because Groq emitted citations such as
`【S1†L7-L9】`. The current normalizer handles `【S1】` but not the line-suffix
form, so citation extraction returned an empty list. This affected the vLLM
block-size and Chinchilla false-premise answers.

Two successfully grounded v2 answers cited both direct content and the generated
overview. Direct content supported the reviewed claim, but overview citations
remain weaker provenance and should be deprioritized when direct evidence is
available.

## Unanswerable-question behavior

Four of the five negative cases were included in the bounded v2 application
run. Three correctly ended with deterministic `insufficient_evidence` /
`abstained` responses and no citations.

The vLLM BLEU question produced safe text beginning with “I don’t know” and
explicitly stated that the document contains no WMT 2014 BLEU result. However,
the API labeled it `answered` and `grounded`. The refusal detector recognizes
the ASCII form `don't` and imposes a short length cap; it missed the curly
apostrophe and longer evidence-limitation explanation. The content was safe,
but the response contract violated the design invariant that a refusal is an
abstention.

## Additional implementation findings

1. **Citation normalization is incomplete.** Normalize Groq's
   `【S#†L#-L#】` form to canonical `[S#]` before extraction and verification.
2. **Refusal status detection is incomplete.** Recognize typographic apostrophes
   and evidence-limitation explanations without treating a substantive hedged
   answer as a refusal.
3. **Final evidence selection still drops difficult evidence.** DPO partition
   evidence reached v2 cross-encoder rank 10 but missed the five-passage budget;
   the vLLM false-premise evidence also missed. This should be addressed in the
   selection policy rather than by changing chunking again.
4. **Provider retry accounting is incomplete.** The v2 and v1 application logs
   recorded 11 and 9 SDK-level retry messages respectively, even though the
   application provider-attempt limit was one. `ChatGroq` should disable hidden
   SDK retries so the application's retry budget is authoritative and visible.
5. **NLTK assets are not self-contained.** Each recreated API container
   downloaded `punkt_tab` and `stopwords` during ingestion. These assets should
   be baked into the image or replaced with a dependency-free tokenizer path.
6. **PDF extraction warnings remain.** Several papers reported rotated text,
   and one long source unit triggered a tokenizer-length warning before being
   split. Neither broke this run, but both motivate the later parser/OCR phase.

## Decision and next gate

The broader test supports adopting `structure_aware_v2` for the next local
application stage. Across eight document styles and 37 total positive questions
when combined with the original pilot, v2 has now shown repeatable improvements
in candidate coverage, final-context recall, direct citation coverage, and
grounded answer outcomes without increasing retrieval latency.

Chunking should remain fixed. The next focused repair should address citation
normalization and refusal-status detection, then rerun only the three affected
queries: vLLM block size, Chinchilla false premise, and the unanswerable vLLM
BLEU question. After that, passage selection can be adjusted against the saved
DPO partition and vLLM false-premise cases. Embedding-model experiments should
begin only after these application correctness defects are closed.

The test left both versioned indexes populated for reproducibility and future
local use. API and PostgreSQL containers were stopped; the PostgreSQL volume was
preserved. No source change was committed or pushed.
