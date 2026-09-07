# Agentic RAG — Project Reference & Handoff

## Purpose

This file is the **single-source project handoff document** for the Agentic RAG project.

It is written so that a new chatbot/LLM — including a local Ollama model — can understand the project's history, current architecture, important design decisions, known issues, completed work, and next steps **without requiring the user to re-explain the project**.

### How to use this file

At the beginning of a new development session:

1. Load this file first.
2. Treat the **Current State** section as authoritative for the latest project direction.
3. Treat **Completed Changes** as historical context; do not repeat those investigations unless a regression occurs.
4. Treat **Known Gaps / TODOs** as work that has not yet been completed.
5. Before changing architecture, inspect the relevant current source files because this document is a handoff summary, not a replacement for source code.

---

# 1. Project Goal

Build a production-oriented **Agentic RAG application** that can:

- ingest PDFs, images, and audio;
- transform them into searchable knowledge;
- retrieve relevant document context;
- answer questions grounded in the ingested knowledge;
- maintain conversational state;
- correct poor retrieval through query rewriting;
- verify generated answers for hallucination;
- expose the RAG pipeline through a FastAPI backend;
- provide a conversational Streamlit frontend;
- persist graph/checkpoint state through PostgreSQL;
- eventually evolve into a true **Agentic RAG system with tool access, tool selection, multi-step reasoning, and external capabilities**.

The project is deliberately being developed in stages:

```text
Basic RAG
    ↓
Production-style RAG
    ↓
Corrective / Advanced RAG       ← current RAG capability
    ↓
Agentic RAG                     ← next major phase
    ↓
Advanced Agentic System
```

---

# 2. Current Maturity

## RAG maturity

**Current assessment: ~7/10 — Intermediate/Advanced, corrective RAG.**

The system is no longer a basic:

```text
Query → Embed → Vector DB → Context → LLM
```

pipeline.

It currently contains:

- document ingestion;
- chunking;
- embeddings;
- Pinecone vector retrieval;
- conversational state;
- query contextualization;
- relevance grading;
- corrective query rewriting;
- re-retrieval;
- grounded generation;
- hallucination checking;
- regeneration;
- LangGraph orchestration;
- FastAPI backend;
- Streamlit UI;
- PostgreSQL checkpointing.

## Agentic maturity

**Current assessment: foundation only; true tool-using agency has not yet been implemented.**

The next major milestone is to make the existing RAG capability itself a tool and then add additional tools such as:

- web search;
- calculator;
- potentially SQL/database access;
- document metadata access;
- later MCP-based tools.

The goal is not to add tools for the sake of complexity. The agent must be able to decide **when a tool is necessary, which tool to use, execute it, inspect the result, and continue or answer**.

---

# 3. Architecture — Current Query Flow

The intended current corrective-RAG request flow is:

```text
User question
    ↓
contextualize_question
    ↓
retrieve
    ↓
grade_documents
    │
    ├── relevant
    │      ↓
    │   generate
    │      ↓
    │   check_hallucination
    │      │
    │      ├── grounded → record_turn → END
    │      │
    │      └── hallucinated → generate
    │
    └── irrelevant
           ↓
       rewrite_query
           ↓
        retrieve
```

### Important state separation

The graph intentionally distinguishes:

```text
question
```

from:

```text
retrieval_query
```

`question` is the user's original wording and must remain untouched so that formatting/style instructions survive.

`retrieval_query` is the standalone/search-optimized version used by retrieval.

Example:

```text
User:
"Explain this in detail and use bullet points."

question:
"Explain this in detail and use bullet points."

retrieval_query:
"topic being discussed"
```

This prevents retrieval-oriented rewriting from destroying the user's answer-formatting instructions.

---

# 4. Ingestion Flow

Ingestion is separate from the query graph.

```text
Upload
  ↓
load
  ├── PDF
  ├── image
  └── audio
  ↓
chunk
  ↓
generate whole-document overview
  ↓
upsert chunks + overview into Pinecone
  ↓
record ingestion metadata in PostgreSQL
```

The whole-document overview exists because a single chunk cannot reliably answer questions such as:

> "What does this document cover?"

The overview is generated once during ingestion rather than reconstructed from arbitrary fragments during every query.

---

# 5. Current Project Structure

Expected high-level structure:

```text
agentic-rag/
│
├── src/
│   └── agentic_rag/
│       ├── config.py
│       │
│       ├── llm/
│       │   ├── provider.py
│       │   └── vision.py
│       │
│       ├── ingestion/
│       │   ├── loaders.py
│       │   ├── chunking.py
│       │   ├── whisper_transcribe.py
│       │   ├── pipeline.py
│       │   └── registry.py
│       │
│       ├── retrieval/
│       │   └── vectorstore.py
│       │
│       ├── graph/
│       │   ├── state.py
│       │   ├── nodes.py
│       │   ├── edges.py
│       │   └── builder.py
│       │
│       └── api/
│           └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── evaluation/
│   └── ragas_eval.py
│
└── tests/
```

---

# 6. File Responsibilities

## `src/agentic_rag/config.py`

Central settings loaded through Pydantic Settings and `.env`.

Current API/provider settings include:

- Groq API key;
- Cerebras API key;
- NVIDIA API key;
- OpenRouter API key;
- HuggingFace token;
- Pinecone API key and index name;
- configurable model name for each LLM provider;
- NVIDIA OpenAI-compatible base URL;
- OpenRouter OpenAI-compatible base URL;
- configurable provider fallback order;
- Bedrock enable/disable flag and region/model settings.

Current local PostgreSQL configuration:

```text
PostgreSQL: localhost:5432
```

The previous configuration used port `5442`, but local PostgreSQL was found listening on `5432`. Docker was not running during the diagnosis. The application configuration was corrected accordingly.

Current provider-order configuration used during development:

```text
NVIDIA → OpenRouter → Groq → Cerebras
```

Cerebras is deliberately placed last for now because the API integration works but the current account returns HTTP `402 payment_required` / quota when inference is attempted. This is an account/quota limitation, not an application integration failure.

Bedrock is currently disabled because AWS access has not yet been configured.

---

## `src/agentic_rag/llm/provider.py`

Provider abstraction / fallback chain. The application interacts with providers only through:

```python
provider_chain.invoke(prompt)
```

Current supported providers:

```text
NVIDIA NIM
    ↓ fallback
OpenRouter
    ↓ fallback
Groq
    ↓ fallback
Cerebras
    ↓ (future)
AWS Bedrock
```

The order is configurable through `settings.provider_order` rather than being hard-coded into graph nodes. This keeps provider-specific details isolated from LangGraph/RAG logic.

### Provider implementation details

- Groq uses `ChatGroq`.
- Cerebras uses `ChatCerebras`.
- NVIDIA uses `ChatOpenAI` against the NVIDIA NIM OpenAI-compatible endpoint.
- OpenRouter uses `ChatOpenAI` against the OpenRouter OpenAI-compatible endpoint.
- Bedrock uses `ChatBedrockConverse` when explicitly enabled.
- Missing API keys cause a provider to be skipped during provider-chain construction.
- A provider failure during `.invoke()` causes the next configured provider to be attempted.
- Provider diagnostics expose the configured providers, primary provider, and most recent successful provider.

### Provider validation completed

Each of the four currently credentialed providers was tested independently:

| Provider | Result |
|---|---|
| Groq | Working |
| Cerebras | Integration reaches API; account returns `402 payment_required` / quota |
| NVIDIA | Working |
| OpenRouter | Working |
| Bedrock | Disabled intentionally |

A provider-chain smoke test also confirmed real fallback behavior:

```text
Cerebras
   ↓ 402 payment_required
Groq
   ↓ success
Answer returned
```

This proves that the fallback abstraction is operational.

### Important generation behavior

The generation prompt previously contained:

```text
Keep the answer concise.
```

This caused answers to remain around 2–4 sentences even when the user explicitly requested elaboration. The generation instruction was changed so that the user's requested level of detail controls answer length:

```text
Follow the user's requested level of detail, structure, and style.
If the user asks for a detailed explanation, provide a detailed explanation.
If the user asks for a concise answer, keep it concise.
If the user asks for examples, include relevant examples.
Do not omit important details needed to properly answer the question.
```

This is now the intended generation behavior.

---

## `src/agentic_rag/llm/vision.py`

Image understanding/captioning component.

The project notes indicate image ingestion uses a vision model rather than the originally considered OCR path.

---

## `src/agentic_rag/ingestion/loaders.py`

Loads supported input types:

```text
.pdf
.png
.jpg
.jpeg
.mp3
.wav
.m4a
```

---

## `src/agentic_rag/ingestion/chunking.py`

Uses recursive text splitting.

Target configuration:

```text
chunk_size = 1500
chunk_overlap = 300
```

---

## `src/agentic_rag/ingestion/whisper_transcribe.py`

Audio → text transcription.

Uses hosted Whisper functionality.

---

## `src/agentic_rag/ingestion/pipeline.py`

Orchestrates:

```text
load → chunk → overview → Pinecone upsert → registry record
```

The function is:

```python
ingest_file(path, display_name=None)
```

Important API-side improvement:

```python
ingest_file(
    tmp_path,
    display_name=upload.filename,
)
```

should be used so the original uploaded filename is retained as metadata rather than the temporary Windows filename.

---

## `src/agentic_rag/ingestion/registry.py`

Tracks ingested documents in PostgreSQL.

Purpose:

```text
"What did I upload?"
```

is a metadata question, not a semantic document-content question.

Therefore it should not be forced through the normal RAG retrieval graph.

---

# 7. Retrieval

## `src/agentic_rag/retrieval/vectorstore.py`

Uses:

```text
HuggingFaceEmbeddings
        ↓
PineconeVectorStore
        ↓
retriever
```

Current embedding model configured through:

```text
settings.embedding_model
```

Default:

```text
all-MiniLM-L6-v2
```

The uploaded source notes describe an MMR-based retriever with k=8 and an overview-specific retriever. The current source code should be treated as authoritative if these values differ.

### Important historical discrepancy

An earlier version of the project used plain:

```text
k = 5
```

retrieval.

Later project notes describe:

```text
MMR
k = 8
overview filtering
```

Do not assume both are simultaneously true. Check the current `vectorstore.py` before modifying retrieval.

---

# 8. Graph State

## `src/agentic_rag/graph/state.py`

The shared `RAGState` contains:

```text
question
retrieval_query
documents
generation
relevance_grade
hallucination_grade
retry_count
messages
```

### Meaning

```text
question
    = raw user input

retrieval_query
    = rewritten/contextualized search query

documents
    = retrieved LangChain Documents

generation
    = final LLM answer

relevance_grade
    = relevant | irrelevant

hallucination_grade
    = grounded | hallucinated

retry_count
    = corrective loop counter

messages
    = persisted conversation history
```

---

# 9. Graph Nodes

## `src/agentic_rag/graph/nodes.py`

The important nodes are:

### `contextualize_question`

Uses conversation history to turn follow-up questions into standalone retrieval queries.

Example:

```text
Previous:
"What is Linux?"

Follow-up:
"Explain its architecture."

Retrieval query:
"Explain the architecture of Linux."
```

For a first question with no history, the original question is used directly.

---

### `retrieve`

Calls the vector retriever using:

```text
retrieval_query
```

or falls back to:

```text
question
```

if no retrieval query exists.

---

### `grade_documents`

Uses the LLM to classify retrieved context:

```text
relevant
```

or:

```text
irrelevant
```

This is a batch-level relevance grade, not an individual per-document relevance score.

---

### `rewrite_query`

Runs when retrieval is considered irrelevant and retry budget remains.

It rewrites the search query for better semantic retrieval.

---

### `generate`

Generates an answer using:

- retrieved context;
- prior conversation;
- original user question.

It must respect the user's requested answer length and style.

It should **not** globally force concise answers.

If the answer cannot be supported by context, the intended behavior is:

```text
I don't know.
```

---

### `check_hallucination`

Checks whether the generated answer is supported by retrieved context:

```text
grounded
```

or:

```text
hallucinated
```

Important interpretation:

```text
"I don't know" + grounded
```

does NOT mean the question was successfully answered.

It only means the answer itself did not introduce unsupported factual claims.

---

### `record_turn`

Adds the user's question and assistant answer to persisted conversation history.

---

# 10. Graph Routing

## `src/agentic_rag/graph/edges.py`

### After relevance grading

```text
relevant
    → generate

irrelevant + retries available
    → rewrite_query

irrelevant + retry limit reached
    → generate best-effort answer
```

### After hallucination check

```text
grounded
    → record_turn

hallucinated + retries available
    → generate again

hallucinated + retry limit reached
    → record_turn / END
```

### Known architectural limitation

The hallucination loop currently regenerates against essentially the same context.

A stronger future design should change something after a hallucination, for example:

```text
hallucinated
    ↓
retrieve additional evidence
or
answer correction / critique
    ↓
generate
```

Simply asking the same generation step to try again is not a strong corrective mechanism.

---

# 11. Graph Builder

## `src/agentic_rag/graph/builder.py`

Builds the LangGraph `StateGraph`.

It receives the checkpointer from the API layer rather than owning the PostgreSQL connection.

The intended query graph is:

```text
contextualize_question
    ↓
retrieve
    ↓
grade_documents
    ├── generate
    └── rewrite_query → retrieve
    ↓
check_hallucination
    ├── record_turn
    └── generate
    ↓
END
```

### Important historical issue

`contextualize_question()` existed in `nodes.py` before it was actually connected to the graph.

This was identified and corrected conceptually by adding:

```text
contextualize_question
```

as the graph entry node.

This is important for follow-up questions.

---

# 12. FastAPI Backend

## `src/agentic_rag/api/main.py`

FastAPI is the stable REST boundary between the UI and LangGraph.

The UI does not directly manipulate LangGraph internals.

### Endpoints

```text
POST /query
POST /ingest
GET  /health
```

Some project notes also mention:

```text
GET /documents
```

If that endpoint is not present in the current source, do not assume it exists.

### `/query`

Receives:

```json
{
  "question": "...",
  "session_id": "..."
}
```

and invokes LangGraph using:

```text
thread_id = session_id
```

This connects the conversation to PostgreSQL checkpoint state.

### `/ingest`

Accepts multiple files, writes temporary files, runs `ingest_file()`, and deletes temporary files afterwards.

Important current improvement:

```python
ingest_file(tmp_path, display_name=upload.filename)
```

to preserve the original document name.

### `/health`

Returns:

```json
{
  "status": "ok"
}
```

---

# 13. PostgreSQL / LangGraph Checkpointing

PostgreSQL is used for persistent graph/checkpoint state.

The FastAPI lifespan owns the checkpointer lifecycle:

```text
Application startup
    ↓
PostgresSaver.from_conn_string(...)
    ↓
checkpointer.setup()
    ↓
build graph
    ↓
serve requests
    ↓
Application shutdown
    ↓
connection closes
```

This is preferred over creating/closing a database connection for every request.

### Local development finding

Local PostgreSQL was confirmed to be listening on:

```text
localhost:5432
```

not:

```text
localhost:5442
```

Docker was not running during diagnosis.

---

# 14. Streamlit Frontend

## `app/streamlit_app.py`

The Streamlit application is intentionally thin.

It should contain UI and HTTP logic only.

### Current UI behavior

The UI now supports:

- document upload;
- ingestion;
- persistent visible chat history;
- multiple questions in the same session;
- follow-up questions;
- entirely new questions;
- assistant responses;
- grounding status;
- reusable HTTP session.

The chat UI uses Streamlit's conversational primitives rather than a single ephemeral text input.

Conceptually:

```text
User message
    ↓
display in chat
    ↓
POST /query
    ↓
FastAPI
    ↓
LangGraph
    ↓
response
    ↓
display assistant message
    ↓
store in Streamlit chat history
```

The same `session_id` is sent for each query, so the backend can maintain the conversation state.

### Important follow-up behavior

The UI supports both:

```text
follow-up:
"Explain that in more detail."
```

and:

```text
new question:
"What is Docker?"
```

However, the graph's contextualization logic must distinguish between genuine follow-ups and independent new questions. This is an area for future improvement.

---

# 15. Debugging / Observability

A major development decision was made to stop debugging the RAG pipeline as a black box.

The desired debug trace is:

```text
QUESTION
    ↓
RETRIEVAL QUERY
    ↓
RETRIEVED DOCUMENTS
    ↓
RELEVANCE GRADE
    ↓
QUERY REWRITE (if needed)
    ↓
GENERATION
    ↓
HALLUCINATION GRADE
    ↓
FINAL ROUTE
```

For each request, development-mode logging should expose:

- original question;
- contextualized retrieval query;
- number of retrieved documents;
- document metadata;
- document previews;
- relevance grader raw output;
- normalized relevance grade;
- retry count;
- rewritten query;
- generation context preview;
- generated answer;
- hallucination grader raw output;
- final route.

### Purpose

This allows us to distinguish:

```text
Retrieval failure
vs
Relevance grader failure
vs
Generation failure
vs
Hallucination grader failure
```

Do not randomly modify multiple layers before inspecting this trace.

---

# 16. Important Debugging Lesson: "I Don't Know"

A test such as:

```text
What is Linux?
```

may correctly produce:

```text
I don't know.
```

if Linux is not present in the uploaded documents.

This is expected behavior for a document-grounded RAG system.

Also:

```text
I don't know.
Grounded in context
```

is not contradictory.

It means:

```text
The answer did not hallucinate.
```

It does NOT mean:

```text
The question was successfully answered.
```

A better UI should eventually distinguish:

```text
Answer supported by context
```

from:

```text
No answer found in context
```

---

# 17. Generation Length Problem

A specific generation-quality issue was identified.

The generation prompt contained:

```text
Keep the answer concise.
```

This was causing answers to remain short even when the user explicitly asked for detail.

### Desired behavior

```text
User asks for definition
    → concise

User asks for detailed explanation
    → detailed

User asks for examples
    → include examples

User asks for advantages + limitations
    → structured detailed answer

User asks for one line
    → one line
```

The user's instruction should determine answer length rather than a hard-coded global brevity rule.

---

# 18. Why This Is Already More Than Basic RAG

Current capabilities include:

```text
Document ingestion
        ↓
Chunking
        ↓
Embeddings
        ↓
Pinecone
        ↓
Contextual retrieval
        ↓
Relevance grading
        ↓
Corrective query rewriting
        ↓
Re-retrieval
        ↓
Grounded generation
        ↓
Hallucination verification
        ↓
Regeneration
        ↓
Persistent conversation state
```

This is best described as:

> **Corrective / Advanced RAG**

It is not yet a full tool-using agent.

---

# 19. Agentic Phase — NEXT MAJOR DEVELOPMENT

The next phase is to transform the current RAG subsystem into a tool available to an agent.

The target architecture is:

```text
                         ┌── RAG / Document Search
                         │
User → Agent/Router ─────┼── Web Search
                         │
                         ├── Calculator
                         │
                         ├── Document Metadata
                         │
                         └── Future Tools
```

The agent should be able to:

1. understand the user request;
2. decide whether a tool is required;
3. select the appropriate tool;
4. execute it;
5. inspect the result;
6. decide whether another tool call is required;
7. produce the final answer.

---

# 20. Agentic RAG Roadmap

## Agentic RAG v1

Implement:

1. Existing RAG as `search_documents` tool.
2. Calculator tool.
3. Web search tool.
4. LLM tool selection.
5. Tool execution loop.
6. Conditional routing.
7. Persistent conversation state.

Target flow:

```text
User
 ↓
Agent
 ↓
Decide tool
 ├── RAG
 ├── Web
 └── Calculator
 ↓
Tool result
 ↓
Observe
 ↓
Need another tool?
 ├── YES → tool
 └── NO → answer
```

---

## Agentic RAG v2

Then add:

- multi-step tool use;
- tool-result evaluation;
- planning;
- better failure handling;
- source-aware answers;
- agent tracing;
- tool-call success/failure metrics;
- stronger memory separation;
- RAG + agent evaluation.

---

## MCP phase

MCP should be introduced **after basic tool calling is understood and working**.

Desired progression:

```text
LLM
 ↓
Native application tool
 ↓
Tool result
 ↓
LLM
```

then:

```text
LLM / Agent
 ↓
MCP client
 ↓
MCP server
 ├── Search
 ├── SQL
 ├── Files
 └── Other external capabilities
```

Do not introduce MCP merely for complexity. First establish the underlying tool-calling loop.

---

# 21. Evaluation

## `evaluation/ragas_eval.py`

An offline RAGAS evaluation harness exists / is planned for:

- faithfulness;
- relevancy;
- context precision;
- context recall.

It should remain separate from the live query path.

### Future evaluation expansion

Agentic evaluation should include:

- retrieval relevance;
- answer faithfulness;
- context utilization;
- tool selection accuracy;
- tool-call success rate;
- number of tool calls;
- number of iterations;
- latency;
- hallucination rate;
- final answer quality.

---

# 22. Important Design Decisions

| Decision | Reason |
|---|---|
| LangGraph instead of a linear chain | Enables conditional routing, corrective retrieval, and verification loops |
| Pinecone instead of local-only vector storage | Supports persistent/deployable retrieval |
| PostgreSQL checkpointer | Conversation state survives process restarts/redeployments |
| FastAPI between UI and graph | Keeps LangGraph internals out of the frontend |
| Streamlit as thin client | Presentation layer remains separate from RAG logic |
| `question` separate from `retrieval_query` | Retrieval rewriting must not destroy user formatting/style instructions |
| Whole-document overview | Broad document-level questions need document-level information |
| Document registry separate from semantic retrieval | Upload metadata is not the same as document content |
| Corrective retrieval | Poor retrieval can trigger query rewriting and another retrieval attempt |
| Hallucination check | Prevents unsupported answers from being accepted silently |
| Tool layer comes after stable RAG | Existing RAG should become a reliable capability rather than being rebuilt unnecessarily |

---

# 23. Known Gaps / Honest TODOs

The following are **not yet complete** unless the current source code explicitly shows otherwise:

- True dense + sparse/BM25 hybrid search.
- Post-retrieval reranking.
- Web search fallback/tool.
- Calculator tool.
- RAG exposed as an agent tool.
- LLM-driven tool selection.
- Multi-step tool execution loop.
- Proper hallucination correction that changes/retrieves better evidence.
- Automatic RAGAS evaluation in development/deployment.
- Agent evaluation.
- Agent tracing/observability dashboard.
- Strong distinction between follow-up questions and unrelated new questions.
- MCP integration.
- SQL/database tool.
- Production deployment validation.
- Consolidation/optimization of PostgreSQL connections across checkpointer and registry.

---

# 24. What NOT to Do

Do not:

- add tools randomly without a routing reason;
- introduce multiple agents before the single-agent tool loop works;
- replace the current RAG architecture without evidence that it is necessary;
- tune retrieval blindly without inspecting retrieved chunks;
- interpret `"I don't know"` as a hallucination;
- interpret `"grounded"` as proof that the question was answered;
- add MCP before understanding ordinary tool calling;
- add hybrid search/reranking merely because they sound advanced;
- change multiple pipeline layers at once during debugging.

The project is being developed deliberately from:

```text
Reliable RAG
    ↓
Observable RAG
    ↓
Tool-enabled agent
    ↓
Advanced agentic system
```

---

# 25. Development Commands

## Start FastAPI

From the project root:

```powershell
uvicorn agentic_rag.api.main:app --reload
```

Expected:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

## Start Streamlit

In another terminal:

```powershell
streamlit run app/streamlit_app.py
```

## Health check

```text
http://127.0.0.1:8000/health
```

Expected:

```json
{"status": "ok"}
```

---

# 26. Validation & Current Baseline

The following validation has now been completed after the multi-provider changes.

## Provider configuration test

The application successfully loads:

```text
Provider order: nvidia,openrouter,groq,cerebras
Groq: True
Cerebras: True
NVIDIA: True
OpenRouter: True
Bedrock: False
```

No API secrets are printed by the diagnostic test; only whether each credential is configured.

## Provider smoke tests

Individual tests for NVIDIA and OpenRouter succeeded. Groq was already working and continues to work. Cerebras was reached successfully but returned `402 payment_required` because the current account does not have usable inference quota.

## End-to-end RAG test

The standalone `tests/test_rag.py` was corrected to match the real graph-builder contract:

```python
build_graph(checkpointer)
```

The test now creates a PostgreSQL `PostgresSaver`, calls `checkpointer.setup()`, builds the real LangGraph, and invokes it using a persistent thread ID. The complete RAG test subsequently completed successfully.

This establishes that the current provider changes did not break the existing graph. The validated path is:

```text
PostgreSQL checkpointer
        ↓
LangGraph construction
        ↓
Contextualization
        ↓
Retrieval
        ↓
Relevance grading
        ↓
Generation
        ↓
Hallucination checking
        ↓
Successful end-to-end RAG execution
```

## Generation performance observation

During debugging, the generation node was observed to be the longest-running stage in at least one end-to-end query. The generation context also contained repeated/overlapping document content. This is an observation, not yet a measured performance conclusion.

Before changing retrieval or model selection for performance, the next debugging step is to instrument:

- generation-node duration;
- provider invocation duration;
- provider actually used;
- prompt character/token size;
- context character/token size;
- total request duration.

Do not blindly reduce retrieval `k` or change the graph until these measurements establish the actual bottleneck.

## Provider diagnostic implementation note

During development, the provider diagnostic properties were temporarily placed outside the `ProviderChain` class because of an indentation mistake. This was identified and corrected. The intended properties are:

```text
provider_names
primary_provider
last_provider
```

They are diagnostic only and do not change the graph interface.

---

# 27. Current Development Order

The intended development order from this point is:

```text
CURRENT BASELINE
│
├── Stable conversational UI                         ✅
├── FastAPI backend                                  ✅
├── PostgreSQL checkpointing                        ✅
├── Pinecone retrieval                               ✅
├── Corrective RAG                                   ✅
├── Contextual follow-up support                    ✅/refine
├── Multi-provider LLM abstraction                   ✅
├── Provider fallback                                ✅
├── End-to-end RAG validation                        ✅
└── Generation performance profiling                 🔧 NEXT
│
▼
NEXT MAJOR PHASE
│
├── RAG as a tool
├── Calculator tool
├── Web search tool
├── Agent tool selection
├── Tool execution loop
├── Multi-step tool use
├── Tool-result reasoning
│
▼
LATER
│
├── Better planning
├── Better memory
├── Evaluation
├── Tracing
├── MCP
├── SQL / external tools
└── Production deployment
```

The immediate task before tool implementation is to establish a performance/latency baseline for the now-stable RAG system.

---

# 28. Current Project Position — One-Paragraph Handoff

**This project is a FastAPI + Streamlit + LangGraph Agentic RAG application. It ingests PDFs/images/audio, chunks and embeds the content, stores vectors in Pinecone, generates document overviews, and uses a corrective LangGraph RAG pipeline with contextualized queries, retrieval, relevance grading, query rewriting/re-retrieval, grounded generation, hallucination checking, and PostgreSQL-backed conversation state. The Streamlit UI now behaves as a persistent conversational interface and supports follow-ups as well as entirely new questions. The RAG layer is approximately 7/10 maturity: advanced/corrective RAG, but not yet a true tool-using agent. The LLM layer now supports configurable Groq, Cerebras, NVIDIA NIM, and OpenRouter providers with automatic fallback; Bedrock is prepared but disabled. NVIDIA, OpenRouter, and Groq are working; Cerebras integration reaches the API but currently returns `402 payment_required` due to account quota. The provider layer and the complete RAG graph have been smoke-tested successfully. The immediate next task is to profile generation latency and prompt/context size, then freeze this RAG baseline before exposing RAG as an agent tool, adding calculator/web tools, and implementing LLM-driven tool selection and execution.**

---

# 29. Session Startup Instruction for Another Chatbot

When this file is loaded into a new session, the assistant should assume:

> The user is continuing development of the Agentic RAG project described in this document. Do not ask the user to explain the project from scratch. First use this file to understand the architecture, completed work, known issues, and next milestone. If source code is provided, inspect the current source before relying on historical values in this document. Clearly distinguish completed features, current implementation, planned features, and unresolved gaps. Prefer incremental changes over unnecessary rewrites.

---

# 30. Source Reference

This document was based on the project's existing `PROJECT_NOTES.md` reference and the development history established during the current project work.

Where historical notes and current source code disagree, **current source code wins**.

This document should be updated whenever a major architectural decision, implementation milestone, debugging discovery, or roadmap change occurs.

---

# 31. Change Log

## Initial project

- Started as a RAG-focused application.
- Established document ingestion, embeddings, vector storage, and LLM generation.
- Moved toward LangGraph to support corrective workflows.

## Retrieval architecture

- Moved to Pinecone for persistent/deployable vector storage.
- Added document overview generation.
- Added corrective relevance grading and query rewriting.

## Persistence

- Replaced in-process memory with PostgreSQL/LangGraph checkpointing.
- Connected FastAPI lifespan to checkpointer lifecycle.

## Backend

- Added FastAPI REST boundary.
- Added `/query`, `/ingest`, and `/health`.

## Frontend

- Streamlit became a thin API client.
- Added persistent visible chat history.
- Added support for multiple questions in one conversation.
- Preserved the same backend session ID for follow-ups.

## Debugging

- Diagnosed Uvicorn import-path issue.
- Diagnosed PostgreSQL port mismatch:
  - application originally attempted `5442`;
  - local PostgreSQL was actually listening on `5432`;
  - Docker had no running containers.
- Identified misleading interpretation of:
  - `I don't know`
  - `grounded`.
- Identified overly restrictive generation instruction:
  - `Keep the answer concise.`
- Identified that `contextualize_question()` existed but needed to be connected into the graph.

## Multi-provider LLM layer — latest milestone

- Expanded the provider abstraction from the earlier Groq/Cerebras/Bedrock design to support Groq, Cerebras, NVIDIA NIM, and OpenRouter, with Bedrock prepared but disabled.
- Added configurable provider order through settings instead of hard-coding provider priority into graph nodes.
- Added OpenAI-compatible `ChatOpenAI` integrations for NVIDIA NIM and OpenRouter.
- Added provider diagnostics for configured providers, primary provider, and last successful provider.
- Verified all four configured credentials are loaded without exposing secrets.
- Independently tested NVIDIA and OpenRouter successfully.
- Confirmed Groq remains operational.
- Confirmed Cerebras connectivity/integration but encountered HTTP `402 payment_required` / quota on inference; the provider is retained as a fallback option.
- Verified automatic fallback by allowing Cerebras to fail and successfully receiving a response from Groq.

## RAG validation — latest milestone

- Corrected the standalone RAG test to pass the real PostgreSQL checkpointer into `build_graph(checkpointer)`.
- Verified PostgreSQL checkpoint initialization and graph construction.
- Completed the end-to-end RAG test successfully through retrieval, grading, generation, and hallucination checking.
- Confirmed the multi-provider changes did not break the existing LangGraph RAG flow.
- Observed that generation appears to be the slowest stage in at least one test and that retrieved context can contain repeated/overlapping text. This is now a performance investigation item rather than an assumed defect.

## Current direction

The project is now moving from:

```text
Corrective RAG
```

to:

```text
Agentic RAG
```

by adding tool access and LLM-driven tool selection while keeping the existing RAG subsystem as a reliable capability.

---

# 32. Golden Rule

**Do not rebuild what already works.**

The current RAG should become the foundation of the agentic system.

The next architectural question is not:

> "How do we make the RAG more complicated?"

It is:

> **"How does an agent decide when to use this RAG capability, when to use another tool, and when it has enough information to answer?"**

---

# 36. Current Optimization Direction — 2026-08-24

The project has deliberately moved away from repeatedly tuning hard-coded retrieval-score thresholds. Recent application logs showed that relative score metrics can classify a weak candidate set as "strong" when the absolute cross-encoder relevance is poor. This is the central weakness in the current CRAG decision layer.

## Decision

Do **not** solve this by adding another document-specific regression/classification model or by adding more arbitrary score thresholds.

The runtime decision layer should become **evidence-driven and multi-signal**, while remaining document-agnostic.

The current retrieval foundation remains:

```text
Dense retrieval
      +
BM25 / sparse retrieval
      ↓
RRF fusion
      ↓
Candidate deduplication
      ↓
Cross-encoder reranking
      ↓
Evidence assessment
```

The cross-encoder remains the primary relevance signal, but its score must not be interpreted in isolation.

Useful evidence signals include:

```text
cross-encoder score
cross-encoder score distribution
score separation / rank separation
dense retrieval evidence
BM25 evidence
RRF rank / score
agreement between retrieval signals
candidate concentration
```

The purpose is to answer:

> "Does the retrieved evidence collectively support answering this question?"

rather than:

> "Did one retrieval score cross an arbitrary threshold?"

---

# 37. Target Evidence-Driven CRAG Flow

The proposed next architecture is:

```text
                         USER QUERY
                              ↓
                   Context / intent handling
                              ↓
                    Original-query retrieval
                              ↓
                    Dense + BM25 + RRF
                              ↓
                     Cross-encoder rerank
                              ↓
                  ┌────────────────────────┐
                  │ Evidence assessment    │
                  │                        │
                  │ CE + dense + BM25 +   │
                  │ RRF + distribution    │
                  └───────────┬────────────┘
                              ↓
                 ┌────────────┼────────────┐
                 ↓            ↓            ↓
              CLEAR        AMBIGUOUS      WEAK
               GOOD            ↓            ↓
                 ↓       Evidence LLM    Rewrite
                 │          Judge           ↓
                 │            ↓         Re-retrieve
                 │       ┌────┴────┐        │
                 │       ↓         ↓        │
                 │     GOOD      BAD ───────┘
                 │       ↓
                 └───────┤
                         ↓
                      GENERATE
                         ↓
               Cheap grounding /
                 answerability gate
                         ↓
                ┌────────┴────────┐
                ↓                 ↓
             SUPPORTED        UNCERTAIN
                ↓                 ↓
               END          One repair / verifier
```

This keeps expensive LLM reasoning conditional rather than sending every request through every stage.

---

# 38. Role of the LLM Evidence Judge

The existing LLM relevance grader should not be treated as the universal first-line classifier.

Instead, it becomes an **ambiguity resolver**.

It should receive the question and the selected evidence and provide structured evidence assessment, conceptually including:

```json
{
  "answerability": "high | partial | low",
  "evidence_quality": "strong | moderate | weak",
  "supported_aspects": [],
  "missing_information": [],
  "contradictions": [],
  "recommendation": "generate | retrieve_more | rewrite"
}
```

The exact schema should be finalized from the current source before implementation.

The objective is to distinguish:

```text
topic similarity
        vs
actual answer-bearing evidence
```

This directly addresses cases where a passage mentions a concept but does not contain enough information to answer the user's question.

---

# 39. Retrieval Failure vs Generation Failure

The system must explicitly distinguish different failure types.

```text
Retrieval failure
    → required evidence is absent or insufficient
    → rewrite / broaden retrieval / eventually abstain

Ranking failure
    → relevant evidence exists but is ranked poorly
    → improve candidate/reranking behavior

Context assembly failure
    → useful evidence exists but is lost, truncated, or buried
    → improve context packing

Generation failure
    → evidence is present but the answer adds unsupported details
    → constrain or repair generation

Verification uncertainty
    → grounding/verification result is ambiguous
    → escalate selectively
```

The generator should not be expected to solve a retrieval failure.

---

# 40. Refusal Is Not the Same as Grounded Success

A response such as:

```text
I don't know.
```

can correctly be classified as **non-hallucinated**, but it must not automatically be treated as a successfully answered question.

The system should eventually distinguish at least:

```text
grounded_answer
refusal
unsupported_answer
verification_uncertain
```

For example:

```text
weak retrieval
      ↓
generation refuses
      ↓
refusal / insufficient evidence
      ↓
corrective retrieval
      ↓
if still insufficient → bounded abstention
```

This avoids both false success states and uncontrolled retry loops.

---

# 41. Grounding / Hallucination Strategy

The hallucination stage should become layered rather than automatically invoking an expensive LLM verifier for every answer.

Target strategy:

```text
Generate
   ↓
Cheap grounding / answerability checks
   ↓
Clearly supported → return
Clearly unsupported → targeted repair / abstain
Ambiguous → LLM verifier
```

Potential cheap checks include:

- evidence/source IDs exist;
- cited evidence was actually retrieved;
- unsupported entities, dates, and numerical claims;
- answer specificity exceeds available evidence;
- obvious mismatch between answer and retrieved context.

The exact implementation will be decided after inspecting the current generation and hallucination nodes.

Retry/correction must remain bounded. The target is **one corrective repair**, not an uncontrolled self-correction loop.

---

# 42. RAGAS Evaluation Strategy

RAGAS is an **evaluation layer**, not a production routing mechanism.

It should be used to compare architecture versions and measure whether changes improve the system.

Useful evaluation dimensions include:

```text
context relevance / precision
answer relevance
faithfulness / groundedness
unsupported-claim rate
retrieval recall where ground truth is available
rewrite rate
repair success rate
abstention rate
latency
LLM call count
```

RAGAS should not be called for every production request merely to decide whether to rewrite or regenerate.

The objective is to use RAGAS as the measurement/referee layer for architectural changes.

---

# 43. What We Are Explicitly NOT Doing

The following approaches are rejected for the current optimization phase:

- training a document-specific regression/classification model for retrieval routing;
- adding another collection of arbitrary cross-encoder score thresholds;
- making raw dense/BM25/RRF scores individually responsible for final routing;
- sending every query through an LLM relevance grader;
- generating multiple rewrite variants by default;
- allowing unlimited regeneration/retrieval loops;
- using RAGAS inside the production hot path;
- redesigning hybrid retrieval again before measuring the current implementation;
- changing ingestion or PDF chunking without evidence from logs/evaluation.

Hard-coded values are still acceptable for **system constraints**, such as a maximum correction count or maximum candidate count. They should not be used as unexplained universal relevance labels.

---

# 44. Cost / Latency / Explainability Objectives

Every optimization must be evaluated against three constraints:

### Cost

```text
LLM calls ↓
Token usage ↓
Expensive verification conditional
```

### Latency

```text
Avoid unnecessary rewrite/retrieval cycles
Avoid unnecessary grading
Parallelize genuinely independent retrieval work
Stream generation when stable
```

### Explainability

Every major decision should expose:

```text
retrieval evidence
retrieval diagnostics
reason for escalation
reason for rewrite
reason for verification
final outcome
```

The system should remain sufficiently white-box that a developer can inspect a failed request and determine whether the failure originated in retrieval, ranking, context assembly, generation, or verification.

---

# 45. Current Development Priority — Updated 2026-08-24

The project is **not ready to move into the tool-using Agentic phase yet**. The immediate objective remains stabilizing the RAG decision and correction layers.

```text
CURRENT
│
├── Hybrid dense + BM25 retrieval                  ✅
├── RRF fusion                                     ✅
├── Cross-encoder reranking                        ✅
├── Retrieval diagnostics                          ✅
├── Query-time candidate deduplication             🔧 / validate current source
├── Evidence-driven retrieval assessment           🔧 NEXT
├── Structured LLM evidence judge                   🔧 NEXT
├── Distinct refusal / insufficient-evidence state 🔧 NEXT
├── Conditional query rewriting                    🔧 NEXT
├── Layered grounding / answerability checks       🔧 NEXT
├── Bounded corrective repair                      🔧 NEXT
├── RAGAS regression evaluation                    🔧 NEXT
├── Generation streaming                           ⏳ AFTER decision layer
├── Latency/cost optimization                      ⏳ AFTER quality baseline
│
▼
AFTER STABLE RAG
│
├── RAG as a tool
├── Calculator
├── Web search
├── LLM tool selection
├── Tool execution loop
└── Multi-step Agentic RAG
```

The current architecture therefore remains **Corrective / Advanced RAG**, not yet a true tool-using agent.

---

# 46. Latest Development Principle

> **Do not replace an unreliable threshold with another threshold and call it adaptive. Make the system reason about evidence, use expensive reasoning only for uncertainty, and use evaluation to measure whether the architecture actually improved.**

The current project should prefer incremental, measurable changes over repeated architectural rewrites.

Where this document conflicts with current source code, **current source code wins**. Ask for permission before directly modifying repository files.

---

# 47. Observability / State Persistence Update — 2026-09-01

This section records the latest observability work and supersedes older tracing notes where they conflict with the current implementation direction.

## 47.1 Primary objective

The project is deliberately separating:

```text
PERSISTENT GRAPH STATE
        vs
TRANSIENT REQUEST DIAGNOSTICS
        vs
PERFORMANCE METRICS
        vs
CONSOLE PRESENTATION
```

Request diagnostics must not be stored inside `RAGState` or PostgreSQL checkpoints. Persistent conversation memory remains part of LangGraph state; transient trace/debug information does not.

## 47.2 RAGState trace removal

The previous state schema contained:

```python
trace: Annotated[list[dict], lambda left, right: left + right]
```

This was identified as a source of state growth across conversation turns because the reducer accumulated trace entries and the state was checkpointed.

Current direction: remove `trace` from graph state rather than attempting to make its reducer more sophisticated. Do not reintroduce `trace` into `RAGState` merely to make diagnostics convenient.

## 47.3 `nodes.py` trace-return removal

The current working `nodes.py` variant has been made free of node return values containing:

```python
"trace": [...]
```

Nodes continue to return only state updates required by the graph. Transient diagnostics belong in the observability layer.

## 47.4 `rewrite_query` correction

A confirmed retrieval-loop bug was identified. The previous rewrite prompt effectively used only the latest `retrieval_query`, meaning a compound user request could lose one of its information needs after the first rewrite.

The corrected design anchors every rewrite to both:

```text
Original user question
+
Most recent search attempt
```

The rewrite instruction explicitly requires preservation of every part of the original information need, including multiple sub-questions.

No retrieval threshold was changed as part of this fix.

---

# 48. Request-Scoped Performance Tracking

`PerformanceTracker` is intended to be request-scoped through `ContextVar` rather than a module-level process-wide accumulator.

Intended lifecycle:

```text
FastAPI /query
      ↓
create PerformanceTracker
      ↓
set_current_tracker(...)
      ↓
graph.invoke(...)
      ↓
nodes measure stages
      ↓
summary_data()
      ↓
reset_current_tracker(...)
```

It answers:

> How long did each stage take?

It is separate from diagnostic evidence and graph decisions.

---

# 49. Structured Diagnostic Collector

A separate `DiagnosticCollector` has been introduced for transient structured request diagnostics.

The collector has been independently tested successfully, including JSONL and JSON output.

Its responsibility is:

```text
DiagnosticCollector
    ↓
structured request events
    ↓
JSONL
    +
JSON summary
```

It is not a replacement for LangGraph state.

---

# 50. Per-Run vs Session-Level Diagnostics

The intended observability model is:

```text
INDIVIDUAL REQUEST
        ↓
per-run diagnostics
+
per-run performance
```

with a separate eventual session-level aggregate:

```text
Session
├── Query 1
├── Query 2
└── Query N
```

The immediate debugging objective is per-run correctness first. Session aggregation comes only after individual runs are trustworthy.

---

# 51. Console Logging Strategy — 2026-09-01

The project is moving away from large `print()`-based diagnostic dumps.

The objective is not to remove useful information. It is to improve diagnostic signal-to-noise ratio.

The console should prioritize:

```text
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

Third-party infrastructure chatter should not dominate the diagnostic transcript.

The key requirement is that logs remain sufficient to distinguish retrieval failure, ranking failure, context assembly failure, generation failure, and verification failure.

---

# 52. JSONL / JSON Diagnostic Artifacts

The preferred diagnostic hierarchy is now:

```text
JSONL
    ↓ event-by-event diagnostic record

JSON
    ↓ per-run structured summary

Console
    ↓ concise human-readable live view

MD transcript
    ↓ optional raw development capture
```

The `.md` transcript remains useful for infrastructure/startup investigation, but JSONL/JSON should become the primary machine-readable evidence supplied to another debugging model.

---

# 53. Main API Observability Direction

The intended `/query` lifecycle is:

```text
POST /query
      ↓
create PerformanceTracker
      ↓
set current tracker
      ↓
create DiagnosticCollector
      ↓
record run_started
      ↓
graph.invoke(...)
      ↓
record run_completed / run_failed
      ↓
collect tracker.summary_data()
      ↓
record performance event
      ↓
finalize diagnostics
      ↓
reset current tracker
```

`PerformanceTracker` and `DiagnosticCollector` are intentionally separate: one measures timing; the other records what happened.

---

# 54. Logging Noise Investigation

The latest Uvicorn startup capture showed substantial third-party noise, including repeated Hugging Face HTTP `HEAD`/`GET` requests, redirects, 404 checks, model-loading progress, and Uvicorn lifecycle messages.

Examples include:

```text
HTTP Request: HEAD https://huggingface.co/...
HTTP/1.1 307 Temporary Redirect
HTTP/1.1 200 OK
HTTP/1.1 404 Not Found
```

and:

```text
Will watch for changes...
Uvicorn running...
Started reloader process...
Application startup complete.
```

These should not be confused with application-level RAG diagnostics.

The cleanup strategy is to suppress repetitive transport/info noise while retaining meaningful warnings/errors and concise model initialization information. Exact logger filtering should be finalized from the actual logger names observed in the running application.

---

# 55. GPU / PyTorch Environment — VERIFIED 2026-09-01

The local development machine has an NVIDIA GPU available and the project environment has now been corrected to use the CUDA-enabled PyTorch build.

## System GPU

```text
GPU:             NVIDIA GeForce RTX 3060
VRAM:            12 GB
NVIDIA driver:   610.88
CUDA UMD:        13.3
```

## Active `uv` environment

```text
Python:          3.13.12
PyTorch:         2.13.0+cu130
CUDA available:  True
PyTorch CUDA:    13.0
Device:          cuda:0
GPU:             NVIDIA GeForce RTX 3060
```

The environment was originally resolving the CPU-only build (`torch 2.13.0+cpu`). The project was corrected by configuring `uv` to source PyTorch from the dedicated CUDA 13.0 wheel index and installing the CUDA-enabled build.

The relevant `pyproject.toml` configuration is:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

`torch` is intentionally not duplicated as a separate normal dependency entry. `sentence-transformers` remains the dependency that requires PyTorch, while `tool.uv.sources` controls where `torch` is resolved from.

## Embedding runtime verification

The configured embedding model is:

```text
all-MiniLM-L6-v2
```

`HuggingFaceEmbeddings` is initialized with the runtime device selected from PyTorch CUDA availability. Verification confirmed that the model can be created with:

```text
model_kwargs={"device": "cuda"}
```

and that the underlying Sentence Transformers client reports the CUDA device.

A CUDA-enabled environment alone is not considered sufficient evidence of GPU utilization. Future performance benchmarks must verify actual model/tensor execution on `cuda:0`.

---

# 56. Current Development Boundary

The current work is deliberately separated into controlled stages:

```text
1. State / trace persistence cleanup
        ↓
2. Request-scoped diagnostics
        ↓
3. Clean structured logging
        ↓
4. GPU environment correction
        ↓
5. Retrieval diagnostics
        ↓
6. LLM performance diagnostics
        ↓
7. Evidence-driven retrieval optimization
        ↓
8. RAG baseline evaluation
        ↓
9. Agentic tool layer
```

Do not combine multiple architectural changes in one benchmark run unless explicitly required.

---

# 57. Current Validation Status — 2026-09-01

## Completed / confirmed

- `rewrite_query` original-question anchoring implemented.
- Current `nodes.py` variant has node-level `"trace"` returns removed.
- Request-scoped `PerformanceTracker` design established.
- `DiagnosticCollector` independently tested successfully.
- JSONL diagnostic output tested.
- JSON diagnostic summary output tested.
- Per-run logging architecture implemented.
- Human-readable logging filenames established.
- `get_logger()` and the existing logging helper API preserved for compatibility.
- NVIDIA GPU and driver confirmed operational at the system level.
- CUDA-enabled PyTorch installed successfully.
- `torch.cuda.is_available()` returns `True`.
- PyTorch reports CUDA `13.0`.
- RTX 3060 is visible as `cuda:0`.
- `sentence-transformers` remains installed and resolves through the project environment.
- `HuggingFaceEmbeddings` can be initialized against CUDA.

## Pending

- One clean live end-to-end RAG query using the final logging implementation.
- Final cleanup of Uvicorn / HTTPX / Hugging Face startup noise.
- Explicit cross-encoder device verification.
- Retrieval-stage structured diagnostics.
- LLM-stage structured diagnostics.
- Per-session diagnostic aggregation.
- Full ingestion stage profiling with GPU-enabled embeddings.
- Retrieval-quality optimization after the observability baseline is trustworthy.
- Benchmark CPU-vs-GPU embedding performance before making further optimization claims.

---

# 58. Immediate Next Steps

```text
1. Run one clean end-to-end RAG query
        ↓
2. Confirm a new timestamped run directory is created
        ↓
3. Inspect application.log
        ↓
4. Inspect detailed_debug.log
        ↓
5. Inspect run_metadata.json
        ↓
6. Inspect JSONL / JSON diagnostics produced by the request
        ↓
7. Verify embedding execution on cuda:0
        ↓
8. Verify cross-encoder execution on cuda:0
        ↓
9. Measure retrieval + reranking + LLM stage latency
        ↓
10. Profile full ingestion pipeline
        ↓
11. Establish CPU/GPU performance baseline
        ↓
12. Begin evidence-driven retrieval optimization
```

No retrieval thresholds should be changed merely because the logging or GPU work is in progress.

---

# 59. Updated Project Principle

> **Persistent state should describe what the application needs to continue the conversation. Diagnostics should describe what happened during a run. Performance metrics should describe how long it took. Logs should present only the information needed by a human to understand the run.**

The goal is not the smallest possible log. The goal is the highest diagnostic signal-to-noise ratio.

---

# 60. Latest Handoff — 2026-09-01

The project remains a **Corrective / Advanced RAG** system and is not yet moving into the tool-using Agentic phase.

The current engineering focus is establishing a trustworthy performance and observability baseline before further retrieval optimization. Request diagnostics are deliberately separated from persistent LangGraph state, and the logging system now creates a distinct timestamped directory for each application run. The logging API retains compatibility helpers such as `get_logger`, `log_llm`, `log_retrieval`, `log_performance`, and `log_graph` so existing modules do not need to be rewritten merely because the storage architecture changed.

The local GPU environment is now verified: an NVIDIA GeForce RTX 3060 is available, PyTorch `2.13.0+cu130` is installed, PyTorch reports CUDA `13.0`, and `torch.cuda.is_available()` returns `True` with the device exposed as `cuda:0`. The `all-MiniLM-L6-v2` embedding model can be initialized through `HuggingFaceEmbeddings` with CUDA. The next validation step is a clean end-to-end application run proving that the embedding and reranking components actually execute on the GPU during normal RAG operation.

Only after this baseline is captured should retrieval-stage latency, reranking, generation, ingestion, and provider behavior be optimized. The eventual objective remains to freeze a reliable optimized RAG baseline and then expose that RAG capability as a tool for the Agentic layer.

---

# 61. Per-Run Logging Architecture — 2026-09-01

The project no longer relies on PowerShell `Tee-Object` as the primary mechanism for preserving application logs. Logging is handled directly by the Python application.

Each application process creates its own run directory beneath `logs/runs/`. This prevents consecutive runs from overwriting or mixing artifacts.

Expected structure:

```text
logs/
└── runs/
    ├── 2026-09-01_14-30-25/
    │   ├── application.log
    │   ├── detailed_debug.log
    │   ├── run_metadata.json
    │   ├── diagnostics.json
    │   └── diagnostics.jsonl
    │
    ├── 2026-09-01_15-12-47/
    │   ├── application.log
    │   ├── detailed_debug.log
    │   ├── run_metadata.json
    │   ├── diagnostics.json
    │   └── diagnostics.jsonl
    │
    └── ...
```

The exact diagnostic files may vary according to which instrumentation paths execute during a run, but all artifacts belonging to a run should remain inside that run's directory.

## Human-readable log files

### `application.log`

Concise application-level events suitable for quickly understanding what happened.

### `detailed_debug.log`

Higher-detail logs including source filename, line number, function name, logger name, level, and message. This is the preferred artifact when diagnosing a failure or unexpected execution path.

## Structured artifacts

### `run_metadata.json`

Captures reproducibility/environment information such as:

- run ID;
- start time;
- platform;
- Python version;
- Python executable;
- PyTorch version;
- CUDA availability;
- PyTorch CUDA version;
- GPU name;
- CUDA device.

### `diagnostics.json` / `diagnostics.jsonl`

Structured request and stage diagnostics. JSONL is preferred for event-oriented records; JSON is preferred for a summarized/request-level representation.

## Logging API compatibility

The logging module must preserve the helper functions already consumed by the application, including:

```python
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

A logging architecture change must not remove these public helpers without first updating every caller. This avoids regressions where the application starts failing because a previously imported logging helper no longer exists.

## Selection of artifacts for debugging

For a normal application failure, provide:

```text
application.log
+
detailed_debug.log
+
run_metadata.json
+
relevant diagnostics.json/jsonl
```

The full run directory should be retained when reproducing a complex issue.

---

# 62. CUDA / Dependency Management Policy — 2026-09-01

The project uses `uv` for dependency resolution and environment management. CUDA-enabled PyTorch is treated as an environment/dependency-source decision rather than an application-level package-name workaround.

The project intentionally keeps:

```text
sentence-transformers
        ↓
      torch
        ↓
PyTorch CUDA wheel index
```

The CUDA source is configured through `tool.uv.sources`, so `torch` does not need to appear as a second duplicate dependency entry merely to select the CUDA build.

Current verified relationship:

```text
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

When modifying dependencies, do not manually edit `uv.lock`. Update `pyproject.toml` and allow `uv` to regenerate the lockfile.

Before trusting GPU performance measurements, verify all of the following from the active environment:

```text
python version
torch version contains +cu
torch.cuda.is_available() == True
torch.version.cuda
actual CUDA device
embedding model device
cross-encoder model device
```

A visible NVIDIA GPU in `nvidia-smi` by itself does not prove that the Python application is using it.

---

# 63. Current Development Boundary — GPU-Enabled Baseline

The controlled development sequence is now:

```text
Reliable RAG
    ↓
Observable RAG
    ↓
CUDA-enabled runtime
    ↓
Clean end-to-end baseline
    ↓
Retrieval / reranking profiling
    ↓
Provider / generation profiling
    ↓
Evidence-driven RAG optimization
    ↓
Freeze optimized RAG baseline
    ↓
RAG as an agent tool
    ↓
Tool selection + execution loop
    ↓
Multi-step Agentic RAG
```

Do not skip directly to agentic tool complexity while the underlying RAG performance and diagnostic baseline is still being established.

