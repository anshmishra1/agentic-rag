# Agentic RAG --- Project Reference & Handoff

## 1. Purpose

This file is the **compact, authoritative project handoff** for the
Agentic RAG application.

Use it to understand:

-   the current architecture;
-   the current implementation direction;
-   important state and routing semantics;
-   observability and logging rules;
-   verified environment facts;
-   known gaps;
-   the immediate development sequence.

### Authority rule

When this document conflicts with the current source code:

> **Current source code wins.**

This document summarizes the project; it does not replace source
inspection.

Historical debugging detail is intentionally compressed. Do not
resurrect an old implementation merely because it appears in an earlier
development note.

------------------------------------------------------------------------

# 2. Project Goal

Build a production-oriented **Agentic RAG application** that can:

-   ingest PDFs, images, and audio;
-   transform content into searchable knowledge;
-   retrieve relevant document context;
-   answer questions grounded in ingested knowledge;
-   maintain persistent conversational state;
-   perform contextual follow-up handling;
-   correct poor retrieval;
-   verify generated answers for hallucination;
-   expose the system through FastAPI;
-   provide a conversational Streamlit UI;
-   use PostgreSQL for LangGraph checkpointing;
-   eventually evolve into a true tool-using Agentic RAG system.

Development progression:

``` text
Basic RAG
    ↓
Production-style RAG
    ↓
Corrective / Advanced RAG       ← current
    ↓
Agentic RAG                    ← next major phase
    ↓
Advanced Agentic System
```

------------------------------------------------------------------------

# 3. Current Project Position

## Current maturity

The RAG layer is approximately **7/10 maturity** and is best described
as:

> **Corrective / Advanced RAG**

It is not yet a true tool-using agent.

Current capabilities include:

-   multimodal document ingestion;
-   chunking;
-   embeddings;
-   Pinecone retrieval;
-   document overviews;
-   document-scoped retrieval;
-   contextualized retrieval queries;
-   relevance assessment;
-   query rewriting and re-retrieval;
-   reranking / retrieval scoring infrastructure;
-   grounded generation;
-   hallucination checking;
-   bounded corrective behavior;
-   LangGraph orchestration;
-   FastAPI;
-   Streamlit;
-   PostgreSQL checkpointing;
-   multi-provider LLM abstraction;
-   request-scoped performance tracking;
-   structured diagnostics;
-   per-run logging;
-   CUDA-enabled embedding environment.

## Current development boundary

We are **stabilizing and measuring the RAG system before adding agentic
tool complexity**.

``` text
State correctness
    ↓
Request diagnostics
    ↓
Clean logging
    ↓
GPU-enabled runtime validation
    ↓
Retrieval / reranking profiling
    ↓
Provider / generation profiling
    ↓
Evidence-driven RAG optimization
    ↓
Freeze reliable RAG baseline
    ↓
RAG as agent tool
    ↓
Tool selection + execution loop
    ↓
Multi-step Agentic RAG
```

------------------------------------------------------------------------

# 4. Current Architecture

``` text
                         USER
                           │
                           ▼
                    Streamlit UI
                           │
                           ▼
                        FastAPI
                           │
             ┌─────────────┴─────────────┐
             │                           │
         /ingest                      /query
             │                           │
             ▼                           ▼
       Ingestion pipeline             LangGraph
             │                           │
             ▼                           ▼
          Pinecone              contextualize_question
                                         │
                                         ▼
                                      retrieve
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                         overview               content
                         retrieval              retrieval
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                                  scored / ranked
                                    retrieval
                                         │
                                         ▼
                                  evidence decision
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                           sufficient            insufficient
                              │                     │
                              ▼                     ▼
                          generate             rewrite / retry
                              │                     │
                              ▼                     └──→ retrieve
                       hallucination check
                              │
                              ▼
                         record_turn
                              │
                              ▼
                             END
```

The LLM layer is provider-abstracted and supports configured fallback
order rather than hard-coding a provider into graph nodes.

------------------------------------------------------------------------

# 5. Request State Semantics

The graph must keep the following concepts separate.

## `question`

The user's original wording.

It must remain unchanged because it may contain:

-   requested detail level;
-   formatting instructions;
-   examples;
-   answer-style constraints;
-   multiple information needs.

## `retrieval_query`

The contextualized / rewritten search representation.

It exists for retrieval only.

Example:

``` text
User question:
"Explain this in detail and use bullet points."

retrieval_query:
"architecture of the topic being discussed"
```

The retrieval query must not replace the original question used for
generation.

## `query_intent`

The canonical classification of the current user message:

``` text
new_question
follow_up
control
```

This is the **single source of truth** for control classification.

Do not maintain a second boolean such as:

``` text
query_is_control
```

when the same information is already encoded by:

``` text
query_intent == "control"
```

## Control classification

There should be one shared control-classification definition.

`contextualize_question()` may classify the query and establish:

``` text
query_intent = "control"
```

`record_turn()` should consume that state rather than independently
maintaining a second control classifier.

Avoid duplicate `is_control()` implementations.

------------------------------------------------------------------------

# 6. Graph Nodes

## `contextualize_question`

Responsibilities:

1.  inspect the original question;
2.  inspect conversation history;
3.  classify intent;
4.  identify control messages;
5.  create a standalone retrieval query for genuine follow-ups;
6.  preserve the original question.

A control message should not enter retrieval.

Examples:

``` text
"thank you"
"thanks"
"okay"
"got it"
```

These should follow the control path rather than trigger retrieval.

Follow-up indicators such as:

``` text
"elaborate"
"explain further"
"tell me more"
"expand on that"
"what do you mean"
"can you clarify"
```

must be treated as follow-up candidates, not automatically as control
messages.

Control matching should avoid overly broad substring behavior that can
misclassify normal questions.

------------------------------------------------------------------------

## `retrieve`

Responsibilities:

-   use `retrieval_query` when available;
-   preserve document scope;
-   retrieve overview and content evidence as required;
-   produce retrieval scores/metrics;
-   expose retrieval diagnostics;
-   avoid unnecessary repeated retrieval.

When `document_id` exists, both overview and content retrieval should
use the active document scope.

------------------------------------------------------------------------

## `grade_documents`

Assesses whether retrieved evidence is sufficient/relevant for the
request.

This is not the same as:

-   similarity score;
-   cross-encoder score;
-   hallucination grade.

Those signals must remain conceptually distinct.

------------------------------------------------------------------------

## `rewrite_query`

Runs only when corrective retrieval is justified and retry budget
remains.

The rewrite must be anchored to:

``` text
original user question
+
most recent retrieval attempt
```

It must preserve all information needs in a compound question.

Do not generate multiple rewrite variants by default.

------------------------------------------------------------------------

## `generate`

Generates the answer using:

-   the original user question;
-   conversation context where appropriate;
-   retrieved evidence;
-   the requested answer style/length.

Never impose a global:

``` text
Keep the answer concise.
```

rule.

The user's requested level of detail controls the answer length.

------------------------------------------------------------------------

## `check_hallucination`

Checks whether the generated answer is supported by retrieved context.

Possible outcome:

``` text
grounded
hallucinated
```

Important:

``` text
"I don't know" + grounded
```

means the answer did not introduce unsupported claims.

It does **not** mean that the question was successfully answered.

The system should distinguish:

``` text
answered and grounded
```

from:

``` text
insufficient evidence / refusal
```

------------------------------------------------------------------------

## `record_turn`

Persists the conversation turn.

It should:

-   record the original user question;
-   record the assistant response;
-   preserve conversation state;
-   use the already-established `query_intent`;
-   avoid independently redefining control-message semantics.

------------------------------------------------------------------------

# 7. Graph Routing

## After contextualization

``` text
query_intent == control
        ↓
   record_turn
```

Otherwise:

``` text
new_question / follow_up
        ↓
     retrieve
```

## After relevance assessment

``` text
sufficient / relevant
        ↓
     generate

insufficient / irrelevant + retries available
        ↓
    rewrite_query
        ↓
      retrieve

insufficient / retry limit reached
        ↓
best-effort generation / refusal
```

## After hallucination check

``` text
grounded
    ↓
record_turn

hallucinated + repair available
    ↓
corrective generation/recovery

repair exhausted
    ↓
record_turn / END
```

A future stronger hallucination-repair design should change the evidence
or reasoning path rather than repeatedly asking the same generation step
to try again.

------------------------------------------------------------------------

# 8. Document-Scoped Retrieval

Every ingested document has a deterministic `document_id` derived from
the file contents using SHA-256.

``` text
file bytes
    ↓
SHA-256
    ↓
document_id
```

This is preferred to a temporary upload path.

Chunks and overviews carry document metadata including:

``` text
document_id
filename
source
type
```

with:

``` text
type=content
```

or:

``` text
type=overview
```

The active `document_id` is propagated through the API, Streamlit state,
and LangGraph state.

When a document is switched in the UI, the conversation thread should be
reset so context from one document does not leak into another.

Existing vectors created before document-scoped metadata was introduced
may require re-ingestion before they can participate correctly in scoped
retrieval.

------------------------------------------------------------------------

# 9. Ingestion

Current conceptual pipeline:

``` text
Upload
   ↓
load
   ├── PDF
   ├── image
   └── audio
   ↓
chunk
   ↓
whole-document overview
   ↓
embeddings
   ↓
Pinecone upsert
   ↓
PostgreSQL document registry
```

Supported source types include:

``` text
.pdf
.png
.jpg
.jpeg
.mp3
.wav
.m4a
```

The original uploaded filename should be retained as metadata rather
than exposing a temporary Windows filename.

Document metadata is a registry concern and should not be forced through
semantic RAG retrieval.

------------------------------------------------------------------------

# 10. Retrieval Layer

Primary components:

``` text
HuggingFaceEmbeddings
        ↓
retrieval / ranking
        ↓
Pinecone
```

The configured embedding model is:

``` text
all-MiniLM-L6-v2
```

Current retrieval work includes:

-   overview retrieval;
-   content retrieval;
-   document scoping;
-   dense/BM25 retrieval infrastructure;
-   RRF fusion;
-   cross-encoder reranking;
-   retrieval scoring;
-   confidence/diagnostic signals.

Current source code is authoritative for exact `k`, candidate counts,
filters, thresholds, and reranking configuration.

Do not change retrieval thresholds or candidate counts without
measurements.

------------------------------------------------------------------------

# 11. FastAPI Backend

`src/agentic_rag/api/main.py` is the stable API boundary between the UI
and graph.

Known endpoints:

``` text
POST /query
POST /ingest
GET  /health
```

The current source must be checked before assuming any additional
endpoint exists.

`/query` carries:

``` json
{
  "question": "...",
  "session_id": "...",
  "document_id": "..."
}
```

The session ID is used as the LangGraph thread ID.

`/ingest` returns the generated document ID.

------------------------------------------------------------------------

# 12. PostgreSQL / Checkpointing

PostgreSQL provides persistent LangGraph checkpoint state.

Conceptual lifecycle:

``` text
FastAPI startup
    ↓
create checkpointer
    ↓
setup
    ↓
build graph
    ↓
serve requests
    ↓
shutdown
    ↓
close resources
```

The checkpointer lifecycle belongs to the API/application layer rather
than individual graph nodes.

Local development configuration has been validated around:

``` text
localhost:5432
```

Historical port investigations are not part of the current architecture.

------------------------------------------------------------------------

# 13. Streamlit

`app/streamlit_app.py` is intentionally thin.

Responsibilities:

-   document upload;
-   document selection;
-   chat UI;
-   HTTP communication;
-   visible conversation history;
-   session/document selection state.

The UI should not implement RAG decisions itself.

Conceptual flow:

``` text
Streamlit
    ↓
POST /query
    ↓
FastAPI
    ↓
LangGraph
    ↓
response
    ↓
Streamlit
```

------------------------------------------------------------------------

# 14. Observability Architecture

Observability is deliberately separated into four concerns:

``` text
Persistent graph state
        ≠
Request diagnostics
        ≠
Performance metrics
        ≠
Human-readable logs
```

## Persistent graph state

Contains information required to continue the conversation.

Do not put transient diagnostic traces into `RAGState`.

In particular, do not reintroduce an accumulating:

``` python
trace
```

field merely for convenience.

## Performance

`PerformanceTracker` measures how long stages take.

It is request-scoped and should not be treated as a process-global
accumulator.

Typical stages:

``` text
contextualize_question
retrieve
grade_documents
rewrite_query
generate
check_hallucination
```

Repeated stage executions must remain measurable.

Example:

``` text
retrieve #1
grade #1
rewrite
retrieve #2
grade #2
```

must not overwrite earlier measurements.

## Structured diagnostics

`DiagnosticCollector` records what happened during a request.

Typical information includes:

-   routing;
-   query intent;
-   retrieval;
-   ranking;
-   evidence decisions;
-   LLM activity;
-   grounding;
-   final outcome.

JSONL is appropriate for event records.

JSON is appropriate for a summarized request-level artifact.

------------------------------------------------------------------------

# 15. Per-Run Logging

The application should not depend on PowerShell `Tee-Object` to preserve
logs.

Python logging writes directly to disk.

Each application process creates a unique run directory:

``` text
logs/
└── runs/
    ├── 2026-09-01_14-30-25/
    ├── 2026-09-01_15-12-47/
    └── ...
```

Every artifact generated by that run must remain inside its run
directory.

Expected structure:

``` text
logs/
└── runs/
    └── <run_id>/
        ├── application.log
        ├── detailed_debug.log
        ├── run_metadata.json
        ├── retrieval.jsonl
        ├── llm.jsonl
        ├── performance.jsonl
        └── diagnostics/
            └── query-level artifacts
```

The exact set of files may vary depending on which instrumentation paths
execute.

## File purposes

### `application.log`

Concise application-level events.

Use this for a quick overview.

### `detailed_debug.log`

Detailed execution information including source location and function
context.

Use this first when diagnosing a failure.

### `run_metadata.json`

Environment/reproducibility information such as:

-   run ID;
-   timestamp;
-   platform;
-   Python version;
-   executable;
-   PyTorch version;
-   CUDA availability/version;
-   GPU;
-   CUDA device.

### Structured JSONL

Event-oriented diagnostics such as retrieval, LLM, performance, routing,
or graph events.

### Query diagnostics

Per-request diagnostic artifacts should remain under the current run
directory.

------------------------------------------------------------------------

# 16. Logging API Compatibility

Changing the storage architecture must not remove helpers already used
by application modules.

The logging module must preserve the application's expected helper API,
including:

``` python
get_logger()
get_run_id()
get_run_directory()
get_run_file()
log_enabled()
log_retrieval()
log_llm()
log_performance()
log_graph()
```

If a helper signature changes, every caller must be updated together.

Do not maintain duplicate definitions of the same logging helper in
`logging.py`.

There must be one canonical implementation of each public logging
helper.

------------------------------------------------------------------------

# 17. What to Provide for Debugging

For a normal application failure, the most useful bundle is:

``` text
application.log
+
detailed_debug.log
+
run_metadata.json
+
relevant diagnostics JSON/JSONL
```

If the problem involves a complete request flow, provide the entire run
directory.

### Which system is authoritative?

Use:

``` text
logging.py artifacts
```

for:

-   execution timeline;
-   application events;
-   performance;
-   provider activity;
-   environment;
-   routing/logging context.

Use:

``` text
DiagnosticCollector artifacts
```

for:

-   structured per-request reasoning/decision evidence;
-   retrieval diagnostics;
-   stage-level diagnostic events;
-   query-level summaries.

For difficult failures, **provide both**. They answer different
questions.

------------------------------------------------------------------------

# 18. Console Noise Policy

The console should show useful application events rather than hundreds
of lines of third-party HTTP/infrastructure chatter.

Target signal:

``` text
RUN
ROUTING
RETRIEVAL
RERANKING
RETRIEVAL DECISION
LLM
GROUNDING
PERFORMANCE
ERROR
```

Third-party libraries such as HTTP clients, Hugging Face, Transformers,
file locks, and watch/reload infrastructure should remain at appropriate
warning levels unless their detailed output is specifically required.

Do not solve console noise by deleting diagnostic information from the
file logs.

The goal is:

> **quiet console + complete useful artifacts**

------------------------------------------------------------------------

# 19. GPU / PyTorch Environment

The local GPU environment has been verified.

Current verified baseline:

``` text
GPU:
NVIDIA GeForce RTX 3060

PyTorch:
2.13.0+cu130

CUDA reported by PyTorch:
13.0

torch.cuda.is_available():
True

device:
cuda:0
```

The configured embedding model is:

``` text
all-MiniLM-L6-v2
```

`HuggingFaceEmbeddings` can be initialized with CUDA.

Important rule:

> CUDA availability alone does not prove that the application is
> actually executing embedding/reranking work on the GPU.

Future benchmarks must verify the actual model/device used during normal
application execution.

Cross-encoder device usage must also be explicitly verified before
claiming GPU acceleration for reranking.

------------------------------------------------------------------------

# 20. UV / CUDA Dependency Policy

The project uses `uv`.

CUDA-enabled PyTorch is selected through the configured PyTorch package
index rather than by duplicating dependency declarations.

Conceptually:

``` text
sentence-transformers
        ↓
      torch
        ↓
PyTorch CUDA wheel index
```

Current environment relationship:

``` text
agentic-rag
    ↓
sentence-transformers 5.6.1
    ↓
torch 2.13.0+cu130
    ↓
CUDA 13.0
    ↓
RTX 3060 / cuda:0
```

When dependencies change:

1.  update `pyproject.toml`;
2.  run `uv` resolution/sync;
3.  allow `uv.lock` to be regenerated.

Do not manually edit `uv.lock`.

------------------------------------------------------------------------

# 21. Provider Architecture

The LLM layer is provider-abstracted.

Current configured providers have included:

``` text
Groq
Cerebras
NVIDIA
OpenRouter
Bedrock
```

Provider order is configuration, not graph logic.

Graph nodes should request an LLM capability rather than hard-code a
provider.

Provider reliability should be handled at the provider boundary:

``` text
provider selection
    ↓
timeout
    ↓
retry policy
    ↓
fallback
    ↓
provider diagnostics
```

Important known issue:

> A provider that hangs for a long time can prevent fallback unless
> provider calls have bounded timeouts.

Do not add provider-specific latency workarounds throughout graph nodes.

------------------------------------------------------------------------

# 22. Generation Behavior

The user's requested answer style must control generation length.

Desired behavior:

``` text
definition request
    → concise

detailed explanation
    → detailed

examples requested
    → include examples

advantages + limitations
    → structured detail

one-line request
    → one line
```

Never use a universal brevity instruction that overrides the user.

The generation layer should also preserve the distinction between:

``` text
question answered
```

and:

``` text
evidence unavailable
```

------------------------------------------------------------------------

# 23. Evidence / CRAG Optimization Strategy

The project is moving toward **evidence-driven / selective CRAG**, not
removal of CRAG.

Target concept:

``` text
Retrieve
    ↓
cheap retrieval evidence
    │
    ├── strong evidence → avoid unnecessary expensive grading
    │
    ├── uncertain evidence → use LLM evidence judge
    │
    └── weak evidence → corrective recovery
```

The exact decision mechanism must be calibrated against evaluation data.

Do not replace one arbitrary threshold with another and call it
adaptive.

Do not send every query through an expensive LLM relevance grader if
reliable cheaper evidence can establish confidence.

------------------------------------------------------------------------

# 24. Optimization Rules

Every optimization should be evaluated against:

``` text
LLM calls ↓
Token usage ↓
Latency ↓ or unchanged
Quality ≥ baseline
Groundedness ≥ baseline
```

A change is not successful merely because it reduces LLM calls.

It must also avoid meaningful quality regression and should not increase
user-visible latency without justification.

Optimize in this order:

``` text
1. Correctness
2. Observability
3. Measurement
4. Retrieval quality
5. Provider reliability
6. Cost / latency
7. Agentic complexity
```

Do not change multiple major pipeline layers in the same benchmark
unless required.

------------------------------------------------------------------------

# 25. Evaluation

RAGAS remains an offline evaluation/referee layer.

Relevant metrics include:

-   faithfulness;
-   answer relevancy;
-   context precision;
-   context recall.

Future agentic evaluation should additionally measure:

-   retrieval relevance;
-   tool-selection accuracy;
-   tool-call success;
-   number of tool calls;
-   iterations;
-   latency;
-   hallucination rate;
-   final answer quality.

Do not put RAGAS in the production hot path.

------------------------------------------------------------------------

# 26. Known Gaps

These should not be assumed complete unless current source code proves
otherwise:

-   fully stabilized evidence-driven retrieval routing;
-   final adaptive CRAG thresholds/calibration;
-   stronger hallucination repair that changes evidence;
-   complete ingestion bottleneck profiling;
-   provider timeout/fallback hardening;
-   full provider-level latency diagnostics;
-   fast-vs-quality model separation;
-   generation context-size profiling;
-   complete RAGAS regression suite;
-   agent evaluation;
-   RAG exposed as an agent tool;
-   calculator tool;
-   web-search tool;
-   LLM-driven tool selection;
-   multi-step tool execution;
-   MCP integration;
-   SQL/database tool;
-   production deployment validation;
-   final session-level diagnostic aggregation.

------------------------------------------------------------------------

# 27. Agentic RAG Roadmap

Do not begin the tool layer by replacing the existing RAG.

The existing RAG should become a reliable capability/tool.

## Agentic RAG v1

``` text
User
  ↓
Agent
  ↓
Decide tool
  ├── RAG / document search
  ├── Calculator
  └── Web search
  ↓
Tool result
  ↓
Observe
  ↓
Need another tool?
  ├── YES → tool
  └── NO → answer
```

Required capabilities:

1.  RAG as a tool;
2.  calculator;
3.  web search;
4.  LLM tool selection;
5.  tool execution loop;
6.  conditional routing;
7.  persistent conversation state.

## Agentic RAG v2

Later add:

-   multi-step tool use;
-   tool-result evaluation;
-   planning;
-   stronger failure handling;
-   source-aware answers;
-   agent tracing;
-   tool success/failure metrics;
-   better memory separation;
-   RAG + agent evaluation.

## MCP

MCP should come **after ordinary tool calling is understood and
working**.

``` text
Agent
  ↓
MCP client
  ↓
MCP server
  ├── Search
  ├── SQL
  ├── Files
  └── Other external capabilities
```

Do not introduce MCP merely to make the architecture look more advanced.

------------------------------------------------------------------------

# 28. Important Design Decisions

  -----------------------------------------------------------------------
  Decision                            Reason
  ----------------------------------- -----------------------------------
  LangGraph                           Conditional routing, corrective
                                      loops, verification

  Pinecone                            Persistent/deployable vector
                                      retrieval

  PostgreSQL checkpointing            Persistent conversational state

  FastAPI boundary                    Keeps UI separate from graph
                                      internals

  Thin Streamlit client               Presentation remains separate from
                                      RAG logic

  `question` vs `retrieval_query`     Protects original user intent/style

  Whole-document overview             Supports document-level questions

  Document registry                   Separates metadata from semantic
                                      content

  Document-scoped retrieval           Prevents unrelated-document context

  Corrective retrieval                Allows recovery from poor evidence

  Hallucination checking              Detects unsupported generation

  Request diagnostics outside         Prevents checkpoint/state growth
  RAGState                            

  Per-run logging                     Prevents consecutive runs from
                                      mixing artifacts

  Existing RAG as future tool         Avoids rebuilding a working
                                      subsystem
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 29. Engineering Rules

Do not:

-   rebuild working components without evidence;
-   optimize from intuition when instrumentation can measure the issue;
-   tune retrieval thresholds blindly;
-   interpret `"I don't know"` as hallucination;
-   interpret `"grounded"` as proof that the question was answered;
-   add provider-specific logic to every graph node;
-   add tools randomly;
-   introduce multiple agents before the single-agent tool loop works;
-   introduce MCP before ordinary tool calling works;
-   put transient traces into persistent graph state;
-   maintain duplicate implementations of the same helper;
-   allow each node to invent its own control-message semantics;
-   depend on `Tee-Object` as the application's primary logging
    mechanism;
-   mix artifacts from different application runs;
-   claim GPU acceleration without verifying actual model/device
    execution.

Core principle:

> **Do not optimize symptoms before proving the boundary and the
> measurement are correct.**

------------------------------------------------------------------------

# 30. Current Validation Baseline

Verified during recent development:

``` text
FastAPI backend                              working
Streamlit conversational UI                 working
PostgreSQL checkpointing                    working
Pinecone retrieval                          working
Document overview retrieval                 working
Document-scoped retrieval                   implemented
Contextual follow-up handling               implemented / being refined
Corrective relevance grading                implemented
Query rewriting / re-retrieval              implemented
Grounded generation                         implemented
Hallucination checking                      implemented
Multi-provider LLM abstraction              implemented
Provider fallback                           implemented
Retrieval scoring / metrics                 implemented
Request-scoped PerformanceTracker            implemented
DiagnosticCollector                         implemented/tested
Per-run logging                             implemented
CUDA-enabled PyTorch                        verified
RTX 3060 / cuda:0                           verified
HuggingFace embedding initialization        verified with CUDA
```

The application still requires a **clean end-to-end validation run**
against the latest combined logging, diagnostics, routing, and GPU
changes before further optimization claims are made.

------------------------------------------------------------------------

# 31. Immediate Development Sequence

``` text
1. Stabilize query intent / control routing
        ↓
2. Verify thank-you / control path
        ↓
3. Verify elaborate / follow-up path
        ↓
4. Run one clean end-to-end query
        ↓
5. Confirm one new per-run directory
        ↓
6. Confirm no stray log files outside that directory
        ↓
7. Confirm console is free of excessive HTTP noise
        ↓
8. Inspect application.log
        ↓
9. Inspect detailed_debug.log
        ↓
10. Inspect run_metadata.json
        ↓
11. Inspect structured diagnostics
        ↓
12. Verify embedding execution on cuda:0
        ↓
13. Verify cross-encoder execution on cuda:0
        ↓
14. Measure retrieval / reranking latency
        ↓
15. Measure provider / generation latency
        ↓
16. Profile full ingestion
        ↓
17. Establish quality + latency baseline
        ↓
18. Perform evidence-driven RAG optimization
        ↓
19. Freeze reliable RAG baseline
        ↓
20. Begin Agentic tool layer
```

------------------------------------------------------------------------

# 32. Debugging Protocol

When a new failure appears:

### Step 1 --- Identify the failing boundary

``` text
UI
API
graph routing
retrieval
ranking
generation
verification
logging
diagnostics
provider
environment
```

### Step 2 --- Inspect the current run artifacts

Start with:

``` text
application.log
detailed_debug.log
run_metadata.json
diagnostic JSON/JSONL
```

### Step 3 --- Inspect the relevant source

Do not infer current behavior from old notes.

### Step 4 --- Change one boundary

Avoid simultaneous changes to:

``` text
retrieval + generation + provider + routing
```

unless the issue explicitly spans those boundaries.

### Step 5 --- Re-run a controlled test

Record:

``` text
quality
latency
LLM calls
retrieval behavior
grounding
```

### Step 6 --- Update this document only after the implementation is confirmed

Do not use `PROJECT.md` to declare a change complete before the source
and validation agree.

------------------------------------------------------------------------

# 33. Development Commands

From the project root:

``` powershell
uvicorn agentic_rag.api.main:app --reload
```

Streamlit:

``` powershell
streamlit run app/streamlit_app.py
```

Health check:

``` text
http://127.0.0.1:8000/health
```

Expected:

``` json
{"status": "ok"}
```

Use the project's configured `uv` environment when running tests and
diagnostics.

------------------------------------------------------------------------

# 34. One-Paragraph Handoff

This project is a **FastAPI + Streamlit + LangGraph Corrective/Advanced
RAG application** that ingests PDFs/images/audio, creates searchable
representations and document overviews, stores retrieval data in
Pinecone, maintains conversation state with PostgreSQL, scopes retrieval
by deterministic document IDs, and uses contextualization, retrieval,
relevance assessment, corrective rewriting, generation, and
hallucination checking. The system now has request-scoped performance
tracking, structured diagnostics, and per-run logging so failures can be
analyzed without putting transient traces into persistent graph state.
The local GPU environment is verified with PyTorch `2.13.0+cu130`, CUDA
`13.0`, and an RTX 3060 exposed as `cuda:0`; `all-MiniLM-L6-v2` can be
initialized through `HuggingFaceEmbeddings` on CUDA. The immediate goal
is to stabilize control/follow-up routing, complete a clean end-to-end
baseline, verify actual GPU execution and clean observability, then
optimize retrieval/provider/generation behavior using measured evidence.
Only after a reliable RAG baseline is frozen should the existing RAG be
exposed as a tool and the tool-using Agentic RAG layer be built.

------------------------------------------------------------------------

# 35. Session Startup Instruction

When this file is loaded in a new development session:

1.  Read this document first.
2.  Treat the current source code as authoritative.
3.  Do not ask the user to explain the architecture from scratch.
4.  Separate current implementation from planned work.
5.  Inspect relevant source files before making changes.
6.  Preserve working architecture unless measurements justify change.
7.  Use run artifacts to diagnose failures.
8.  Keep persistent state, diagnostics, performance, and logs
    conceptually separate.
9.  Prefer one canonical implementation of each helper/classification
    rule.
10. Make incremental, measurable changes.

------------------------------------------------------------------------

# 36. Change Log Policy

This document should record **major current architectural changes**, not
every debugging command or failed experiment.

For future updates, use this compact format:

``` text
## YYYY-MM-DD — Short Change Title

Changed:
- ...

Why:
- ...

Validated:
- ...

Remaining:
- ...
```

Detailed experiments belong in dedicated diagnostic artifacts or
development notes, not in the main handoff.

------------------------------------------------------------------------

# 37. Golden Rule

> **Do not rebuild what already works.**

The immediate question is not:

> "How do we make the RAG more complicated?"

It is:

> **"How do we make the existing RAG reliable, measurable, efficient,
> and then give an agent the ability to decide when to use it?"**
