"""Node functions for the optimized corrective RAG pipeline.

Optimization introduced in this version:
- Retrieval now produces an evidence profile from the scores/metadata it already has.
- A deterministic retrieval assessment decides whether an LLM relevance grader is
  actually necessary.
- Strong retrieval results go directly to generation.
- Ambiguous/weak retrieval results keep the existing corrective grading + rewrite loop.
- The LLM grader evaluates the strongest candidates instead of blindly grading every
  retrieved chunk.

The existing document_id-scoped retrieval and provider tiers are preserved.
"""
from __future__ import annotations

import hashlib
import re
import json
from concurrent.futures import ThreadPoolExecutor
from agentic_rag.core.logging import (
    get_logger,
    log_graph,
    log_llm,
    log_performance,
    log_retrieval,
)

from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag.config import settings
from agentic_rag.graph.state import RAGState
from agentic_rag.llm.provider import provider_chain, fast_provider_chain
from agentic_rag.retrieval.vectorstore import (
    build_query_representation,
    retrieve_hybrid_with_scores,
)
from agentic_rag.retrieval.reranker import rerank_many
from agentic_rag.retrieval.sparse import load_bm25_json
from agentic_rag.ingestion.registry import get_bm25_params
from agentic_rag.policies.retrieval import assess_retrieval_confidence
from agentic_rag.policies.generation import apply_generation_limits, is_refusal_answer
from agentic_rag.core.timing import get_current_tracker#, set_current_tracker, reset_current_tracker
from agentic_rag.policies.grounding import parse_grounding_response, GROUNDING_VERDICTS

logger = get_logger(__name__)
# tracker = get_current_tracker()


# =============================================================
# Debug helpers
# =============================================================

def _separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _preview(text: str, length: int = 500) -> str:
    text = text.replace("\n", " ")
    return text[:length]


# =============================================================
# Query Intent
# =============================================================

CONTROL_QUERY_PATTERNS = (
    "thanks",
    "thank you",
    "no more questions",
    "no further questions",
    "nothing else",
    "that's all",
    "that is all",
    "stop",
    "done",
    "end this",
)


def classify_query_intent(
    question: str,
    history: list,
) -> tuple[str, bool]:
    """
    Classify the incoming user message as:

        new_question
        follow_up
        control

    Control messages are only recognized when the complete normalized
    message is a conversational control message. We deliberately avoid
    substring matching because phrases such as "Can you elaborate on that?"
    must never become control messages.
    """

    normalized = " ".join(question.lower().split()).strip()

    # ---------------------------------------------------------
    # Control / conversation-ending input
    # ---------------------------------------------------------

    if normalized in CONTROL_QUERY_PATTERNS:
        return "control", True

    # ---------------------------------------------------------
    # No history -> cannot be a follow-up
    # ---------------------------------------------------------

    if not history:
        return "new_question", False

    # ---------------------------------------------------------
    # Strong follow-up indicators
    # ---------------------------------------------------------

    follow_up_indicators = (
        "elaborate",
        "elaborate further",
        "explain further",
        "explain more",
        "tell me more",
        "expand on",
        "expand upon",
        "go deeper",
        "in more detail",
        "more details",
        "why is that",
        "how so",
        "what about that",
        "can you clarify",
        "clarify that",
        "what do you mean",
    )

    if any(
        indicator in normalized
        for indicator in follow_up_indicators
    ):
        return "follow_up", False

    # ---------------------------------------------------------
    # Referential follow-ups
    # ---------------------------------------------------------

    referential_terms = (
        "that",
        "this",
        "it",
        "those",
        "these",
        "the above",
        "the previous",
        "mentioned earlier",
    )

    if any(
        normalized == term
        or normalized.startswith(term + " ")
        for term in referential_terms
    ):
        return "follow_up", False

    return "new_question", False
# =============================================================
# Contextualize Question
# =============================================================

def contextualize_question(state: RAGState) -> dict:
    """
    Determine whether the incoming query is new, a follow-up, or a
    conversation-control message.

    Only follow-ups requiring conversational resolution invoke the
    fast LLM contextualizer.
    """

    with get_current_tracker().measure("contextualize_question"):

        history = state.get("messages", [])[-6:]
        question = state["question"]

        intent, is_control = classify_query_intent(
            question,
            history,
        )

        # print("\n" + "=" * 70)
        print("QUERY INTENT")
        # print("=" * 70)
        print(f"Question: {question}")
        print(f"Intent: {intent}")
        print(f"Control query: {is_control}")

        # -----------------------------------------------------
        # Control message
        # -----------------------------------------------------

        if intent == 'control':
            return {
                "query_intent": "control",
                "query_is_control": True,
                "contextualization_used": False,
                "retrieval_query": question,
                "hallucination_retry_count": 0,
                "correction_attempted": False,
            }

        # -----------------------------------------------------
        # New standalone question
        # -----------------------------------------------------

        if intent == "new_question":
            return {
                "query_intent": "new_question",
                "query_is_control": False,
                "contextualization_used": False,
                "retrieval_query": question,
            }

        # -----------------------------------------------------
        # Follow-up
        # -----------------------------------------------------

        history_text = "\n".join(
            f"{m.type}: {m.content}"
            for m in history
        )

        prompt = (
            "Rewrite the user's follow-up as a standalone search query "
            "for the same document.\n\n"
            "Rules:\n"
            "1. Preserve the user's actual information need.\n"
            "2. Use the previous conversation only to resolve references.\n"
            "3. Do not introduce new topics.\n"
            "4. Do not answer the question.\n"
            "5. Return only the standalone search query.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Follow-up:\n{question}"
        )

        result = fast_provider_chain.invoke(prompt)

        rewritten = result.content.strip()
        
    return {
        "query_intent": "control",
        "query_is_control": True,
        "contextualization_used": False,
        "retrieval_query": question,

        # Reset fields which belong only to an actual RAG execution.
        "documents": [],
        "generation": "",
        "retrieval_decision": None,
        "retrieval_evidence_strength": None,
        "retrieval_decision_reason": None,
        "relevance_grade": None,
        "hallucination_grade": None,
        "hallucination_retry_count": 0,
        "correction_attempted": False,
        "verification_exhausted": False,
        "retry_count": 0,
    }


# =============================================================
# Retrieval metrics
# =============================================================

def calculate_retrieval_metrics(scores: list[float]) -> dict:
    if not scores:
        return {
            "top_score": 0.0,
            "second_score": 0.0,
            "score_gap": 0.0,
            "mean_score": 0.0,
            "top_to_mean_ratio": 0.0,
            "gap_ratio": 0.0,
        }

    ordered = sorted((float(score) for score in scores), reverse=True)
    top_score = ordered[0]
    second_score = ordered[1] if len(ordered) > 1 else ordered[0]
    mean_score = sum(ordered) / len(ordered)
    score_gap = top_score - second_score

    # Scores are assumed to be similarity scores where larger is better, as
    # returned by PineconeVectorStore.similarity_search_with_score().
    top_to_mean_ratio = top_score / mean_score if mean_score > 0 else 0.0
    gap_ratio = score_gap / abs(top_score) if top_score != 0 else 0.0

    return {
        "top_score": top_score,
        "second_score": second_score,
        "score_gap": score_gap,
        "mean_score": mean_score,
        "top_to_mean_ratio": top_to_mean_ratio,
        "gap_ratio": gap_ratio,
    }


def _top_score_for_type(results: list[tuple], chunk_type: str) -> float | None:
    scores = [
        float(score)
        for doc, score in results
        if doc.metadata.get("type") == chunk_type
    ]
    return max(scores) if scores else None

def _normalize_chunk_text(text: str) -> str:
    """Normalize chunk text for stable query-time deduplication."""

    return " ".join(text.lower().split())


def _candidate_identity(doc) -> tuple:
    """Build a stable identity for a retrieved chunk.

    This intentionally operates only at query time. It does not modify
    ingestion, indexing, chunk IDs, or persisted data.
    """

    metadata = doc.metadata

    document_id = metadata.get("document_id", "")
    chunk_type = metadata.get("type", "")
    page = metadata.get(
        "page_label",
        metadata.get("page", ""),
    )

    normalized_text = _normalize_chunk_text(
        doc.page_content
    )

    content_hash = hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()[:16]

    return (
        document_id,
        chunk_type,
        str(page),
        content_hash,
    )


def _deduplicate_candidates(
    candidates: list[tuple],
) -> tuple[list[tuple], int]:
    """Remove duplicate chunks while preserving ranking order.

    The first occurrence wins because RRF already sorted candidates by
    relevance before this function is called.
    """

    seen = set()
    deduplicated = []
    removed = 0

    for candidate in candidates:
        doc = candidate[0]

        identity = _candidate_identity(doc)

        if identity in seen:
            removed += 1
            continue

        seen.add(identity)
        deduplicated.append(candidate)

    return deduplicated, removed

# =============================================================
# Retrieval
# =============================================================

def retrieve(state: RAGState) -> dict:
    with get_current_tracker().measure("retrieve"):
        # _separator("2. RETRIEVAL")

        query = state.get("retrieval_query") or state["question"]
        # print(f"Query sent to retriever:\n{query}")

        document_id = state.get("document_id")

        if document_id:
            overview_filter = {
                "$and": [
                    {"document_id": {"$eq": document_id}},
                    {"type": {"$eq": "overview"}},
                ]
            }
            content_filter = {
                "$and": [
                    {"document_id": {"$eq": document_id}},
                    {"type": {"$eq": "content"}},
                ]
            }

            bm25_params = get_bm25_params(document_id)
            bm25_encoder = load_bm25_json(bm25_params) if bm25_params else None

            if bm25_encoder is None:
                logger.warning(
                    "RETRIEVAL | BM25 unavailable | document_id=%s | "
                    "falling back to dense-only retrieval",
                    document_id,
                )
        else:
            overview_filter = {"type": {"$eq": "overview"}}
            content_filter = None
            bm25_encoder = None

            logger.info(
                "RETRIEVAL | document_scope=GLOBAL | mode=dense-only"
            )

        # ---------------------------------------------------------
        # Query representation is built exactly once.
        # ---------------------------------------------------------
        # representation = build_query_representation(query, bm25_encoder)
        representation = build_query_representation(query, bm25_encoder)

        log_retrieval(
            logger,
            "query_representation | dense=1 | sparse=%d",
            1 if representation.sparse is not None else 0,
            enabled=settings.retrieval_logs,
        )
        # print("Query representation: dense=1 | sparse=" + ("1" if representation.sparse is not None else "0"))

        # ---------------------------------------------------------
        # Overview/content retrieval are independent I/O operations.
        # Run them concurrently while sharing the same query vectors.
        # ---------------------------------------------------------
        retrieval_args = (
            {
                "query": query,
                "bm25_encoder": bm25_encoder,
                "k": settings.hybrid_candidate_k,
                "filter": overview_filter,
                "query_representation": representation,
            },
            {
                "query": query,
                "bm25_encoder": bm25_encoder,
                "k": settings.hybrid_candidate_k,
                "filter": content_filter,
                "query_representation": representation,
            },
        )

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-retrieval") as executor:
            overview_future = executor.submit(retrieve_hybrid_with_scores, **retrieval_args[0])
            content_future = executor.submit(retrieve_hybrid_with_scores, **retrieval_args[1])
            overview_payload = overview_future.result()
            content_payload = content_future.result()

        overview_candidates, overview_diagnostics = overview_payload
        content_candidates, content_diagnostics = content_payload

        # ---------------------------------------------------------
        # Query-time candidate deduplication
        # ---------------------------------------------------------
        overview_candidates, overview_duplicates_removed = _deduplicate_candidates(overview_candidates)
        content_candidates, content_duplicates_removed = _deduplicate_candidates(content_candidates)

        print("\n" + "-" * 70)
        print("HYBRID CANDIDATE DEDUPLICATION")
        print("-" * 70)
        print(f"Overview: {len(overview_candidates) + overview_duplicates_removed} -> {len(overview_candidates)} ({overview_duplicates_removed} duplicates removed)")
        print(f"Content: {len(content_candidates) + content_duplicates_removed} -> {len(content_candidates)} ({content_duplicates_removed} duplicates removed)")

        # ---------------------------------------------------------
        # One cross-encoder inference batch for both groups.
        # ---------------------------------------------------------
        reranked = rerank_many(
            query,
            {
                "overview": overview_candidates,
                "content": content_candidates,
            },
            {
                "overview": settings.rerank_top_k_overview,
                "content": settings.rerank_top_k_content,
            },
        )

        overview_results = reranked.get("overview", [])
        content_results = reranked.get("content", [])

        results = overview_results + content_results
        results.sort(key=lambda item: float(item[1]), reverse=True)

        docs = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]

        overview_docs_count = len(overview_results)
        content_docs_count = len(content_results)
        metrics = calculate_retrieval_metrics(scores)
        top_score = metrics["top_score"]
        overview_top_score = _top_score_for_type(results, "overview")
        content_top_score = _top_score_for_type(results, "content")

        print(f"\nOverview candidates (hybrid): {len(overview_candidates)} -> reranked to {overview_docs_count}")
        print(f"Content candidates (hybrid): {len(content_candidates)} -> reranked to {content_docs_count}")

        for i, doc in enumerate(docs, start=1):
            print("\n" + "-" * 60)
            print(f"DOCUMENT {i}")
            print(f"Cross-encoder relevance score: {scores[i - 1]}")
            print(f"Metadata: {doc.metadata}")
            print(f"\nContent preview:\n{_preview(doc.page_content)}")

        log_retrieval(
            logger,
            "query=%r | candidates=%d | top=%.4f | second=%.4f | "
            "gap=%.4f | mean=%.4f | top/mean=%.4f | gap_ratio=%.4f",
            query,
            len(scores),
            metrics["top_score"],
            metrics["second_score"],
            metrics["score_gap"],
            metrics["mean_score"],
            metrics["top_to_mean_ratio"],
            metrics["gap_ratio"],
            enabled=settings.retrieval_logs,
        )

        return {
            "documents": docs,
            "retrieval_scores": scores,
            "retrieval_top_score": top_score,
            "retrieval_second_score": metrics["second_score"],
            "retrieval_score_gap": metrics["score_gap"],
            "retrieval_mean_score": metrics["mean_score"],
            "retrieval_top_to_mean_ratio": metrics["top_to_mean_ratio"],
            "retrieval_gap_ratio": metrics["gap_ratio"],
            "retrieval_overview_top_score": overview_top_score,
            "retrieval_content_top_score": content_top_score,
        }


# =============================================================
# Retrieval assessment — NEW optimization layer
# =============================================================

def assess_retrieval(state: RAGState) -> dict:
    """Decide whether an LLM relevance grader is necessary - or whether the
    evidence is weak enough that grading is pointless and a query rewrite is
    the better next step. Decision logic itself lives in
    policies.retrieval.assess_retrieval_confidence(); this node's job is I/O
    (printing, tracing) and reading state, not the decision rule.
    """
    with get_current_tracker().measure("assess_retrieval"):
        _separator("3. RETRIEVAL ASSESSMENT")

        documents = state.get("documents", [])
        scores = [float(score) for score in state.get("retrieval_scores", [])]

        if not documents or not scores:
            decision = "grade"
            reason = "no_retrieval_candidates"
            evidence_strength = "weak"
        else:
            metrics = calculate_retrieval_metrics(scores)
            top_doc = documents[0]
            top_type = top_doc.metadata.get("type")
            overview_top = state.get("retrieval_overview_top_score")
            content_top = state.get("retrieval_content_top_score")
            retry_count = state.get("retry_count", 0)

            result = assess_retrieval_confidence(
                metrics=metrics,
                top_doc_type=top_type,
                overview_top_score=overview_top,
                content_top_score=content_top,
                retry_count=retry_count,
            )
            decision = result["decision"]
            evidence_strength = result["evidence_strength"]
            reason = result["reason"]

            print(f"Top score: {metrics['top_score']:.6f}")
            print(f"Second score: {metrics['second_score']:.6f}")
            print(f"Mean score: {metrics['mean_score']:.6f}")
            print(f"Score gap: {metrics['top_score'] - metrics['second_score']:.6f}")
            print(f"Top/mean ratio: {metrics['top_to_mean_ratio']:.4f}")
            print(f"Gap ratio: {metrics['gap_ratio']:.4f}")
            print(f"Top document type: {top_type}")
            print(
                "Score interpretation: "
                "cross-encoder ranking signal, not calibrated probability"
            )

        print(f"Evidence strength: {evidence_strength}")
        print(f"Retrieval decision: {decision}")
        print(f"Decision reason: {reason}")

        return {
            "retrieval_decision": decision,
            "retrieval_evidence_strength": evidence_strength,
            "retrieval_decision_reason": reason,
        }


# =============================================================
# Document Grading
# =============================================================

def grade_documents(state: RAGState) -> dict:
    """Use the fast LLM only for ambiguous retrieval results."""
    with get_current_tracker().measure("grade_documents"):
        _separator("4. DOCUMENT RELEVANCE GRADING")

        documents = state.get("documents", [])
        scores = state.get("retrieval_scores", [])
        query = state.get("retrieval_query") or state["question"]

        # Retrieval already sorted candidates by score. Keep the grader prompt
        # bounded and focused on the strongest evidence instead of all chunks.
        candidates = list(zip(documents, scores))
        preview = [
            {
                "rank": idx,
                "score": round(float(score), 4),
                "type": doc.metadata.get("type"),
                "source": doc.metadata.get("source"),
                "content": doc.page_content[:500],
            }
            for idx, (doc, score) in enumerate(candidates, start=1)
        ]

        print(f"Question/query being graded:\n{query}")
        print(f"Candidates sent to grader: {len(preview)}")

        prompt = (
            "You are a retrieval relevance grader. Determine whether the retrieved "
            "evidence contains enough information to answer the user's query. "
            "Consider the candidate type metadata: an 'overview' chunk represents "
            "whole-document evidence, while a 'content' chunk represents specific "
            "document content. Return exactly one word: 'relevant' or 'irrelevant'.\n\n"
            f"Question: {query}\n\n"
            f"Retrieved candidates:\n{json.dumps(preview, ensure_ascii=False, default=str)}"
        )

        result = fast_provider_chain.invoke(prompt)
        raw_grade = result.content.strip().lower()
        grade = (
            "relevant"
            if "relevant" in raw_grade and "irrelevant" not in raw_grade
            else "irrelevant"
        )

        print(f"\nRaw LLM grading response: {raw_grade}")
        print(f"\nNormalized relevance grade: {grade}")

        return {
            "relevance_grade": grade,
        }


# =============================================================
# Query Rewrite
# =============================================================

def rewrite_query(state: RAGState) -> dict:
    with get_current_tracker().measure("rewrite_query"):
        _separator("5. QUERY REWRITE")

        current = state.get("retrieval_query") or state["question"]
        original_question = state["question"]
        retry_count = state.get("retry_count", 0)

        prompt = (
            "Rewrite the search query below to be clearer and better suited "
            "for semantic search. The most recent search attempt is shown "
            "for context, but your rewrite must address EVERY part of the "
            "user's original question - not just whichever part the most "
            "recent attempt focused on. If the original question has "
            "multiple parts (e.g. asks for both a definition AND benefits, "
            "or both what and why), make sure your rewrite still covers "
            "all of them, even if the most recent attempt dropped one.\n\n"
            f"Original question: {original_question}\n\n"
            f"Most recent search attempt: {current}\n\n"
            "Return only the rewritten query, nothing else."
        )

        result = fast_provider_chain.invoke(prompt)
        rewritten_query = result.content.strip()
        new_retry_count = retry_count + 1

        print(f"Original question: {original_question}")
        print(f"Previous attempt  : {current}")
        print(f"Rewritten query   : {rewritten_query}")
        print(f"Retry count       : {new_retry_count}")

        return {
            "retrieval_query": rewritten_query,
            "retry_count": new_retry_count,
        }


# =============================================================
# Generation
# =============================================================

def generate(state: RAGState) -> dict:
    with get_current_tracker().measure("generate"):
        _separator("6. GENERATION")

        documents = state.get("documents", [])
        history = state.get("messages", [])

        context, history = apply_generation_limits(documents, history)

        history_text = (
            "\n".join(
                f"{m.type}: {m.content}"
                for m in history
            )
            if history
            else "None"
        )

        prompt = (
            "Answer the question using only the provided context and "
            "prior conversation. "
            "Follow the user's requested level of detail, structure, "
            "and style. "
            "If the user asks for a detailed explanation, provide a "
            "detailed explanation. "
            "If the user asks for a concise answer, keep it concise. "
            "If the answer isn't supported by the context, say you "
            "don't know. "
            "Do not omit important details needed to properly answer "
            "the question.\n\n"
            f"Prior conversation:\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}"
        )

        # ---------------------------------------------------------
        # Generation diagnostics
        # ---------------------------------------------------------

        context_chars = len(context)
        history_chars = len(history_text)
        prompt_chars = len(prompt)

        print(f"Context characters : {context_chars}")
        print(f"History characters : {history_chars}")
        print(f"Prompt characters  : {prompt_chars}")
        print(f"Documents supplied : {len(documents)}")
        print(f"History turns      : {len(history)}")

        # ---------------------------------------------------------
        # LLM generation
        # ---------------------------------------------------------

        result = provider_chain.invoke(prompt)

        generation = result.content.strip()

        output_chars = len(generation)

        print(f"Output characters  : {output_chars}")

        return {
            "generation": generation,
        }

# =============================================================
# Hallucination Check
# =============================================================

def check_hallucination(state: RAGState) -> dict:
    with get_current_tracker().measure("check_hallucination"):
        _separator("7. HALLUCINATION CHECK")

        generation = state["generation"]

        if is_refusal_answer(generation):
            print("Answer is a refusal ('I don't know' or similar) - skipping")
            print("the hallucination check entirely. A refusal makes no")
            print("factual claim, so there's nothing to verify as grounded.")
            return {
                "hallucination_grade": "grounded",
                "hallucination_retry_count": state.get("hallucination_retry_count", 0),
                "verification_exhausted": False,
            }

        documents = state.get("documents", [])
        context = "\n\n".join(d.page_content for d in documents)

        prompt = (
            "Is the following answer fully supported by the context below? "
            "Answer with exactly one word: 'grounded' or 'hallucinated'.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{generation}"
        )

        result = fast_provider_chain.invoke(prompt)
        raw_grade = result.content.strip().lower()
        grade = (
            "grounded"
            if "grounded" in raw_grade and "hallucinated" not in raw_grade
            else "hallucinated"
        )

        hallucination_retry_count = state.get("hallucination_retry_count", 0)
        if grade == "hallucinated":
            hallucination_retry_count += 1

        # Distinguishes "still hallucinated, but a retry will happen" from
        # "still hallucinated, retries exhausted, this is what gets served."
        # Routing (edges.py) already decides retry-vs-end based on this same
        # comparison; this just makes the outcome explicit in state instead
        # of leaving grounded=False as the only (easy-to-miss) signal.
        verification_exhausted = (
            grade == "hallucinated"
            and hallucination_retry_count >= settings.max_retries
        )

        print(f"Raw hallucination grader response: {raw_grade}")
        print(f"Normalized grade: {grade}")
        print(f"Hallucination retry count: {hallucination_retry_count} / {settings.max_retries}")
        print(f"Verification exhausted: {verification_exhausted}")

        return {
            "hallucination_grade": grade,
            "hallucination_retry_count": hallucination_retry_count,
            "verification_exhausted": verification_exhausted,
        }


# =============================================================
# Corrective Regeneration — NEW, single bounded correction
# =============================================================

def correct_generation(state: RAGState) -> dict:
    """Single bounded corrective regeneration, triggered only when the
    grounding diagnosis is 'unsupported' (generation overclaimed) rather than
    'insufficient_evidence' (a retrieval problem, routed to rewrite_query
    instead). Feeds the specific unsupported claims back into the prompt so
    the model corrects the actual failure instead of blindly retrying with
    the same instructions against the same evidence."""
    with get_current_tracker().measure("correct_generation"):
        _separator("6b. CORRECTIVE REGENERATION")

        documents = state.get("documents", [])
        history = state.get("messages", [])
        previous_generation = state.get("generation", "")
        unsupported_claims = state.get("grounding_unsupported_claims", [])

        context, history = apply_generation_limits(documents, history)

        history_text = (
            "\n".join(f"{m.type}: {m.content}" for m in history)
            if history else "None"
        )

        claims_text = (
            "\n".join(f"- {c}" for c in unsupported_claims)
            if unsupported_claims
            else "(no specific claims identified - be more conservative overall)"
        )

        prompt = (
            "Your previous answer contained claims that are not supported "
            "by the provided context:\n\n"
            f"{claims_text}\n\n"
            "Rewrite the answer using ONLY claims directly supported by the "
            "context below. If the context does not establish something, "
            "explicitly say the document does not cover it rather than "
            "omitting it silently. Preserve the user's requested level of "
            "detail and formatting where the context allows it.\n\n"
            f"Prior conversation:\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\n"
            f"Previous (flawed) answer:\n{previous_generation}"
        )

        result = provider_chain.invoke(prompt)
        generation = result.content.strip()

        print(f"Unsupported claims addressed: {unsupported_claims}")
        print(f"Corrected answer characters: {len(generation)}")

        return {
            "generation": generation,
            "correction_attempted": True,
        }

# =============================================================
# Record Turn
# =============================================================

# =============================================================
# Record Turn
# =============================================================

_UNVERIFIED_DISCLAIMER = (
    "Note: this answer could not be fully verified against the retrieved "
    "sources after multiple attempts. Treat it with extra caution and "
    "consider rephrasing your question.\n\n"
)


def record_turn(state: RAGState) -> dict:
    """
    Persist the current conversational turn.

    Control messages such as "thank you", "thanks", "done", etc. bypass
    retrieval and generation. Because LangGraph persists state across turns,
    those control turns must NEVER reuse generation or retrieval/verification
    fields from the previous turn.
    """

    is_control = state.get("query_is_control", False)

    # ---------------------------------------------------------
    # CONTROL TURN
    # ---------------------------------------------------------
    if is_control:
        control_response = (
            "You're welcome! If you'd like to continue, "
            "feel free to ask another question."
        )

        _separator("8. RECORD TURN")

        print(f"Question               : {state['question']}")
        print("Query type             : control")
        print("Retrieval              : skipped")
        print("Generation             : skipped")
        print("Hallucination check    : skipped")
        print(f"Final answer length    : {len(control_response)} chars")

        return {
            # IMPORTANT:
            # Explicitly replace the stale generation from the previous
            # checkpointed turn.
            "generation": control_response,

            # Explicitly clear previous RAG metadata.
            "retrieval_query": state["question"],
            "retrieval_decision": None,
            "retrieval_evidence_strength": None,
            "retrieval_decision_reason": None,
            "relevance_grade": None,
            "hallucination_grade": None,
            "hallucination_retry_count": 0,
            "correction_attempted": False,
            "verification_exhausted": False,

            # Persist THIS conversational turn only.
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=control_response),
            ],

            "trace": [
                {
                    "stage": "record_turn",
                    "final_route": "end",
                    "query_intent": "control",
                    "query_is_control": True,
                    "retrieval_skipped": True,
                    "generation_skipped": True,
                    "hallucination_check_skipped": True,
                }
            ],
        }

    # ---------------------------------------------------------
    # NORMAL RAG TURN
    # ---------------------------------------------------------

    hallucination_grade = state.get("hallucination_grade")
    hallucination_retry_count = state.get(
        "hallucination_retry_count",
        0,
    )

    answer_verified = hallucination_grade == "grounded"

    generation = state.get("generation", "")

    if not answer_verified and hallucination_grade is not None:
        print(
            "\nWARNING: answer reached record_turn without passing "
            "the hallucination check (exhausted retries). "
            "Attaching disclaimer."
        )

        generation = _UNVERIFIED_DISCLAIMER + generation

    final_entry = {
        "stage": "record_turn",
        "final_route": "end",
        "query_intent": state.get("query_intent"),
        "query_is_control": False,
        "retry_count": state.get("retry_count", 0),
        "retrieval_decision": state.get("retrieval_decision"),
        "retrieval_evidence_strength": state.get(
            "retrieval_evidence_strength"
        ),
        "retrieval_decision_reason": state.get(
            "retrieval_decision_reason"
        ),
        "hallucination_final_grade": hallucination_grade,
        "hallucination_retry_count": hallucination_retry_count,
        "answer_verified": answer_verified,
    }

    full_trace = state.get("trace", []) + [final_entry]

    _separator("STRUCTURED TRACE SUMMARY (full request, end to end)")

    print(
        json.dumps(
            full_trace,
            indent=2,
            default=str,
        )
    )

    return {
        "generation": generation,

        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=generation),
        ],

        "trace": [final_entry],
    }