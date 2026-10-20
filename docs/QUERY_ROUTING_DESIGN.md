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
information_need_source: current_turn | previous_information_need
response_format: requested | prose | bullets | table | ascii_flowchart | math
evidence_requirement: text | visual | table | formula | mixed
standalone_query: string | null
needs_clarification: boolean
```

Groq receives an explicit seven-field contract through JSON Object Mode. The
result is validated locally with Pydantic, which preserves the same closed
allowed values without depending on provider-side strict-schema support. Empty
output, invalid JSON, contract violations, unavailable routing providers, and
unresolved source ambiguity fail closed to a clarification turn.

The enforced plan also produces one application-owned `routing_reason_code`.
It records whether the effective route came from explicit source selection,
semantic selection, the selected-document topic safeguard, a missing document,
source ambiguity, or a classified planner failure. The code is reset every turn
and stored in flat graph state; the more detailed parse failure kind is written
only to the transient trace.

For Auto mode, the API looks up the selected document's filename from the
index-scoped PostgreSQL registry and passes that bounded metadata to the
router. This lets a question about prompt engineering match a selected prompt
engineering document without maintaining topic phrase lists. The filename and
conversation remain delimited as untrusted data in the planning prompt.

## Runtime flow

```text
user message
    |
    +-- closed control command -----------------------> record_turn
    |
    +-- first explicit general question -------------> general generation
    |
    +-- first explicit document question ------------> format detection
    |                                                   + document retrieval
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

High-precision presentation instructions such as `as a flowchart`, `compare in
a table`, and `using LaTeX` are detected deterministically because they are a
closed UI contract, not topic classification. Merely mentioning a document
diagram, figure, table, equation, page, or section does not select an answer
format. A trailing format request is removed from retrieval text while
remaining available to generation.

A format-only follow-up such as `represent that as a flowchart` reuses the
latest human information request only when the planner selects
`previous_information_need` and a narrow full-message grammar confirms the
request is purely presentational. An anaphor such as `that` or `it` alone cannot
discard the current information need. Requests such as `explain that diagram`
remain `current_turn`; history may resolve the reference inside the standalone
query without replacing it with the prior question.

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
- textual flowcharts use fenced plain-text blocks and cite every factual node
  or transition label; diagram borders and arrows are presentation only.

Presentation does not alter the selected evidence source.

## Cost and latency

Closed commands, first explicit document questions, and first explicit general
questions need no routing call. Auto requests and conversational follow-ups use
one bounded fast-tier call with a 256-token output cap and low Groq reasoning
effort.
Low-score candidate sets use one structured relevance-grade call rather than
blindly spending retrieval retries. An uncertain grader result is allowed into
generation, but the stricter citation-aware grounding verifier still decides
whether the answer can be accepted.

## Verification evidence

- 65 focused query-planning, contextualization-node, state, graph-routing, and
  API diagnostic checks pass in the existing application image.
- The image ran with networking disabled, no `.env`, and the worktree mounted
  read-only.
- The repository interpreter is unavailable because its uv-managed Python
  executable is missing.
- No Pinecone, PostgreSQL, provider, model download, ingestion, or live query
  was used for this implementation checkpoint.

## Answer-routing reliability repair

The 2026-10-07 live run exposed three false-negative paths after the original
router shipped:

1. non-empty candidate sets below the provisional cross-encoder score floor
   were rewritten before their text was semantically graded;
2. empty Groq grader or rewriter content was accepted as a successful provider
   call and then interpreted as irrelevant or a stalled rewrite;
3. Auto mode exposed only `document_selected=true` to the router, so a prompt
   engineering question could be sent to general knowledge while a prompt
   engineering document was selected.

The repair sends every non-empty below-floor candidate set to the closed
semantic relevance grader. Its verdict is `relevant`, `irrelevant`, or the
local fail-closed state `uncertain`. `uncertain` proceeds to generation so the
stricter citation-aware grounding verifier can make the final decision;
`irrelevant` retains the bounded rewrite/abstention behavior. Empty required
provider content now triggers provider fallback rather than becoming a
semantic decision. Groq SDK retries are disabled so the application retry
budget remains authoritative.

The relevance grader and rewriter now use low Groq GPT-OSS reasoning effort
and dedicated 512-token caps. The grader requests strict JSON on Groq and the
provider-neutral parser accepts exact legacy verdict words for other adapters.
The grounding verifier keeps its existing 1024-token cap and fails closed if
the provider is unavailable.

For answer presentation, first-turn explicit formats are detected without an
LLM call, presentation suffixes are excluded from retrieval text, and an
anaphoric format-only follow-up reuses the previous human information need.
ASCII flowchart nodes and factual transition labels require citations, while
the verifier ignores borders and arrows as presentation characters.

The second repair slice adds provider-independent citation normalization for
alternate brackets, optional line suffixes, and invisible formatting characters.
It exempts only exact structural labels inside fenced textual diagrams; factual
nodes remain citation checked. Generation and correction now prohibit inferred
motivation, causality, maximality, and exhaustive-document claims unless the
cited excerpts state them.

The reviewed eleven-case acceptance contract and its evidence levels are
documented in `docs/V2_ANSWER_RELIABILITY_ACCEPTANCE.md`.

## Remaining validation

After review and image integration, run a bounded three-turn check:

1. document mode: request a textual Transformer architecture flowchart;
2. same thread: ask `Are you sure?` and confirm the verification route retrieves
   the prior information need;
3. general mode: ask an unrelated general question and confirm retrieval,
   citations, and document grounding are skipped.
