# Query routing and answer-source design

## Decision

The application uses a constrained semantic router inside the existing
LangGraph workflow. It does not add a free-running routing agent.

A routing agent with its own planning loop could repeatedly choose tools or
routes, but the available actions here are known in advance. A typed one-step
decision is faster, cheaper, easier to test, and keeps provenance explicit.

## Why the previous classifier was insufficient

The previous runtime classified a turn as `new_question`, `follow_up`, or
`control` using phrase patterns. That conflated several independent decisions:

- whether the answer should use the selected document or model knowledge;
- whether the message depends on conversation history;
- whether the user is asking to verify the previous answer;
- how the answer should be presented.

For example, `Are you sure?` was passed unchanged to document retrieval because
it did not match the follow-up phrase list. A request for a textual flowchart
was a document question with a formatting requirement, not a separate
retrieval intent.

## Routing contract

The Streamlit UI exposes three source modes:

| Mode | Retrieval | Citations and grounding | Semantic source choice |
|---|---|---|---|
| Selected document | Required | Required | Source is fixed |
| General model | Skipped | Not applicable | Source is fixed |
| Auto | Conditional | Only for document answers | Router decides or asks |

The semantic router returns a validated `QueryPlan` with independent fields:

```text
source: document | general | control | clarify
relationship: standalone | follow_up | verify_previous
response_format: requested | prose | bullets | table | ascii_flowchart | math
standalone_query: string | null
needs_clarification: boolean
```

Groq receives the plan as a strict JSON-schema response request. The result is
validated again with Pydantic. Malformed output, unavailable routing providers,
and unresolved source ambiguity fail closed to a clarification turn.

## Runtime flow

```text
user message
    |
    +-- closed control command -----------------------> record_turn
    |
    +-- explicit general mode ------------------------> general generation
    |
    +-- first explicit document question ------------> document retrieval
    |
    `-- semantic query plan
            |
            +-- document -----------------------------> corrective RAG path
            +-- general ------------------------------> general generation
            `-- clarify ------------------------------> deterministic question
```

Document verification follow-ups use the router's standalone form of the prior
information need for retrieval. Generation receives both the refreshed evidence
and an instruction to say what was confirmed, corrected, or unsupported.

General answers use the primary provider without Pinecone, BM25, reranking,
citations, or the grounding verifier. The API returns `answer_source=general`
and `answer_status=general_answer`; Streamlit labels that provenance directly.

## Deterministic boundary

Hardcoded runtime language is limited to the closed command set such as empty
input, `thanks`, `stop`, and `done`. Open-ended follow-up and verification
language is not maintained as a phrase list.

Explicit source selection is also deterministic. The semantic model cannot
override Selected document or General model. Auto mode never silently converts
failed document retrieval into a general answer.

## Answer presentation contract

Document and general generation share formatting rules:

- display equations use valid LaTeX within `$$` delimiters;
- inline mathematics uses single `$` delimiters;
- citations remain outside equations;
- non-trivial formulas define their symbols and include a plain-language
  explanation;
- textual flowcharts use fenced plain-text blocks.

Presentation does not alter the selected evidence source.

## Cost and latency

Closed commands, first explicit document questions, and explicit general turns
need no routing call. Auto requests and document conversations use one bounded
fast-tier call with a 256-token output cap and low Groq reasoning effort.

## Verification evidence

- 43 focused routing, state, graph-edge, provider-option, formatting, and node
  tests pass offline.
- 147 tests from the supported host suite pass.
- Three existing structure-aware chunking tests require the pinned Hugging Face
  tokenizer that is baked into the Docker image but absent from the host cache;
  their host failure is an environment limitation. The branch has not yet been
  tested inside the self-contained application image.
- No `.env`, Pinecone, PostgreSQL, Groq, model download, ingestion, or live query
  was used for this implementation checkpoint.

## Remaining validation

After review and image integration, run a bounded three-turn check:

1. document mode: request a textual Transformer architecture flowchart;
2. same thread: ask `Are you sure?` and confirm the verification route retrieves
   the prior information need;
3. general mode: ask an unrelated general question and confirm retrieval,
   citations, and document grounding are skipped.
