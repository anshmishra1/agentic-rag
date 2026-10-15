# V3 end-to-end evaluation plan

This document turns the V3 requirements into a repeatable acceptance run. The
machine-readable source is `eval_set/v3_end_to_end_questions.json`; this page is
the human review and execution guide.

## What this suite is meant to prove

The suite measures the application as a connected system:

1. the selected parser finds native text, OCR text, figures, tables, formulas,
   and page images;
2. retrieval remains scoped to the selected document;
3. hybrid retrieval and Ettin reranking place the expected evidence near the
   top;
4. routing chooses document evidence, general knowledge, clarification, a
   conversational control response, or prior-turn context correctly;
5. generation follows requested tables, mathematics, bullets, and ASCII
   flowcharts;
6. citations resolve to the evidence that supports the answer;
7. false premises are corrected and absent evidence produces an abstention;
8. stage timings expose parser, retrieval, reranking, provider, and total
   latency instead of hiding them in one duration.

This is not a collection of prompts chosen only because the current system can
answer them. Each document has a controlled question and a natural, less
precise version aimed at the same evidence. Negative and routing cases are part
of the contract as well.

## Why these seven documents

The 22-paper research folder is useful for future breadth testing, but running
every paper through Groq would make the first acceptance run expensive and hard
to diagnose. This seven-document corpus covers the distinct evidence and
behavior classes needed for the V3 decision:

| Document | Main capabilities under test |
|---|---|
| Attention Is All You Need | architecture figure, formula, numeric table, false premise |
| FlashAttention | hardware-memory diagram, algorithm, ASCII flowchart, absent metric |
| LoRA | mechanism figure, quantitative claims, later-method abstention |
| Direct Preference Optimization | equations, proof reasoning, false premise |
| Flamingo | multimodal architecture figure and results table |
| Building Effective AI Agents | multi-page synthesis, workflow language, follow-up |
| Scanned fixture | conditional OCR, scanned table, diagram direction, formula label |

The files and their SHA-256 hashes are frozen in the JSON suite so a changed
PDF cannot silently invalidate the expected evidence.

## Suite shape

- **49 total cases** across all seven documents.
- **18 `e2e_core` cases** for the first bounded provider-backed run.
- **31 `e2e_extended` cases** for ambiguity, routing, conversations, additional
  negative cases, and broader formatting checks.
- **Seven matched language pairs**: one controlled question and one realistic,
  ambiguous question per document, both targeting the same concepts.
- **Seven unanswerable questions**: one per document.
- Required evidence covers text blocks, figures, tables, formulas, and page
  images, with direct, OCR-derived, table, formula, and visual grounding.

The suite is currently marked
`source_reviewed_pending_v3_artifact_ids`. Page-level evidence has been checked,
but the final artifact and search-record IDs can only be attached after these
exact files are ingested into the isolated V3 index.

## Core live run

Run the cases in this order so a failure can be attributed to one capability.
The expected answer is summarized in the JSON file and is not sent to the
application.

### Attention Is All You Need

1. `attn-architecture-controlled` — Explain the main components of the
   Transformer encoder and decoder shown in Figure 1.
2. `attn-scaling-math` — Why is dot-product attention divided by the square
   root of the key dimension? Show the attention equation in valid LaTeX and
   define every symbol.
3. `attn-table-comparison` — From Table 3, compare the base and big Transformer
   configurations in a compact table, including parameters, development BLEU,
   and development perplexity.

### FlashAttention

1. `flash-tiling-controlled` — How does FlashAttention use tiling to reduce
   reads and writes between HBM and on-chip SRAM?
2. `flash-figure-lookup` — Does the paper contain a diagram showing how data
   moves between HBM and SRAM? Use that figure to explain the flow.
3. `flash-algorithm-flowchart` — Turn Algorithm 1 into a concise ASCII flowchart
   that shows the block loops, on-chip computation, and writes back to HBM.
4. `flash-energy-unanswerable` — How many kilowatt-hours does one
   FlashAttention training run consume?

### LoRA

1. `lora-mechanism-controlled` — How does LoRA adapt a pretrained Transformer
   while keeping the original weights frozen?
2. `lora-efficiency-table` — Compare full fine-tuning and LoRA for GPT-3 175B
   in a table using the parameter, memory, and inference-latency claims reported
   by the paper.
3. `lora-qlora-unanswerable` — According to this paper, when should I choose
   QLoRA instead of LoRA?

### Direct Preference Optimization

1. `dpo-partition-math` — Explain why the partition function does not have to
   be estimated in DPO. Include the relevant relation in valid LaTeX and define
   the terms.
2. `dpo-reward-model-false-premise` — Why must DPO train a separate reward model
   before updating the policy?

### Flamingo

1. `flamingo-architecture-controlled` — How do Flamingo's gated cross-attention
   dense layers connect visual features to the frozen language model?
2. `flamingo-results-table` — Using Table 2, list the five tasks where
   fine-tuned Flamingo sets a new state of the art.

### Building Effective AI Agents

1. `agents-workflow-controlled` — How does Anthropic distinguish workflows from
   agents?
2. `agents-evaluator-flowchart` — Explain the evaluator-optimizer workflow as
   an ASCII flowchart with its feedback loop.

### Scanned fixture

1. `ocr-code-controlled` — What document code appears on the scanned page?
2. `ocr-status-table` — Read the status table. Which stage is ready and which
   one is still pending?

## Two-pass execution

### Pass 1: retrieval only

Run all document cases whose `expected.should_retrieve` value is `true`. This
pass must not call Groq. For every case, record:

- retrieved search-record IDs and their artifact IDs;
- rank before and after reranking;
- dense, sparse, fused, and reranker scores when available;
- locator page and artifact type;
- whether the expected page and evidence type appeared in the candidate set;
- retrieval and reranking latency.

This pass separates retrieval defects from generation and verification defects.
A missing expected artifact indicates parsing or indexing. An indexed artifact
that is never retrieved indicates candidate generation. A retrieved artifact
that falls after reranking indicates evidence ranking.

### Pass 2: bounded end to end

Run the 18 core cases through the application with the same documents and
configuration. For every case, record:

- planner and route decisions;
- retrieval and rewrite counts;
- final answer and requested format;
- answer status and grounding diagnosis;
- citation labels plus resolved artifact and page;
- unsupported or forbidden claims;
- parser, retrieval, reranking, provider, and total latency;
- provider call count and token usage when exposed.

Only move to the 31 extended cases after reviewing the core output and provider
quota. The extended set tests natural phrasing, false premises, one-turn
follow-ups, verification of a prior answer, general knowledge, clarification,
and control messages.

## How results are judged

The JSON contains required concepts, forbidden claims, allowed statuses,
expected pages and evidence types, citation requirements, and retry limits.
Those labels allow each stage to be scored independently:

| Layer | Evidence of success | Failure meaning |
|---|---|---|
| Parser/index | Expected artifact and page exist | Parsing, OCR, structure, or write defect |
| Candidate retrieval | Expected evidence appears before reranking | Dense/sparse query or fusion defect |
| Reranking | Relevant evidence remains near the top | Reranker relevance defect |
| Routing | Expected source and relationship selected | Planner/context defect |
| Answer | Required concepts present; forbidden claims absent | Generation or prompt defect |
| Citation | Every factual claim resolves to supporting evidence | Citation selection or rendering defect |
| Verdict | Answered, insufficient, or control status matches evidence | Verification/state defect |
| Format | Requested table, math, bullets, or flowchart is usable | Response-format defect |

V3 should be promoted only when it does not regress the answerable text cases
against V2, improves the visual/OCR cases V2 cannot represent, introduces no
cross-document leakage or unsupported citations, abstains on absent evidence,
and has a measured latency that is acceptable for an interactive demo. A single
aggregate score is insufficient because it can hide a severe citation or
safety regression.

## Current execution boundary

The dataset and source labels are ready, and the complete V3 path has passed one
controlled scanned-fixture smoke. That run exercised parser, asset storage,
manifest readiness, Pinecone dense and sparse retrieval, weighted RRF, Ettin
reranking, Groq generation and verification, citation resolution, structured
OCR evidence, and the preview route. Cleanup left no fixture vectors, manifest,
or assets. The next package must ingest the hash-locked corpus, attach exact
artifact/search-record IDs, and execute the retrieval-only pass before any
broader provider-backed evaluation.

Evaluation concept matching must normalize Unicode dash variants. The first
smoke returned the correct `ORCHID‑73` with a non-breaking hyphen; an exact
ASCII comparison would have reported a false failure.
