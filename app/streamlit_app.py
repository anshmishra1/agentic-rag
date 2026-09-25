"""
Streamlit frontend.

The RAG pipeline lives behind FastAPI.
This file is responsible only for:

- document upload
- ingestion
- displaying conversation history
- sending questions to the FastAPI backend
"""

import os
import uuid

import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)


# ============================================================
# Session State
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "documents" not in st.session_state:
    st.session_state.documents = []


if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None


# ============================================================
# HTTP Session
# ============================================================

@st.cache_resource
def get_http_session() -> requests.Session:
    """
    Reuse one HTTP connection pool across Streamlit reruns.
    """
    return requests.Session()


http = get_http_session()


def load_documents() -> list[dict]:
    """Fetch the documents registered by the FastAPI backend."""
    response = http.get(f"{API_URL}/documents", timeout=30)
    response.raise_for_status()
    return response.json()


# ============================================================
# Page
# ============================================================

st.title("Agentic RAG assistant")

st.write(
    "Ask questions grounded in your ingested documents."
)


# ============================================================
# Document Upload
# ============================================================

st.subheader("Upload documents")

uploaded_files = st.file_uploader(
    "Add PDFs, images, or audio to the knowledge base",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "mp3",
        "wav",
        "m4a",
    ],
    accept_multiple_files=True,
)


if uploaded_files and st.button("Ingest"):

    with st.spinner("Ingesting documents..."):

        files_payload = [
            (
                "files",
                (
                    f.name,
                    f.getvalue(),
                    f.type or "application/octet-stream",
                ),
            )
            for f in uploaded_files
        ]

        response = http.post(
            f"{API_URL}/ingest",
            files=files_payload,
            timeout=300,
        )

        if not response.ok:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            st.error(
                detail
                if isinstance(detail, str)
                else f"Ingestion failed (HTTP {response.status_code})."
            )
            st.stop()

    ingestion_results = response.json()

    for result in ingestion_results:
        st.success(
            f"{result['filename']}: "
            f"{result['chunks_indexed']} chunks indexed"
        )

    # Refresh the registry after ingestion.
    try:
        st.session_state.documents = load_documents()

        # Automatically select the first newly ingested document when
        # there was no active document before the upload.
        if (
            st.session_state.active_document_id is None
            and ingestion_results
        ):
            st.session_state.active_document_id = ingestion_results[0]["document_id"]

    except requests.RequestException as exc:
        st.warning(
            "Documents were ingested, but I could not refresh the document list."
        )
        st.error(str(exc))


# ============================================================
# Active Document
# ============================================================

st.subheader("Active document")

try:
    st.session_state.documents = load_documents()
except requests.RequestException as exc:
    st.warning("Could not load the document registry from FastAPI.")
    st.error(str(exc))

documents = st.session_state.documents

if documents:
    document_options = {
        f"{doc['filename']} ({doc['chunk_count']} chunks)": doc["document_id"]
        for doc in documents
        if doc.get("document_id")
    }

    if document_options:
        labels = list(document_options.keys())

        current_label = next(
            (
                label
                for label, document_id in document_options.items()
                if document_id == st.session_state.active_document_id
            ),
            labels[0],
        )

        selected_label = st.selectbox(
            "Questions will be retrieved only from this document:",
            labels,
            index=labels.index(current_label),
        )

        selected_document_id = document_options[selected_label]

        if selected_document_id != st.session_state.active_document_id:
            st.session_state.active_document_id = selected_document_id

            # A document switch changes retrieval scope. Keeping the previous
            # chat visible can be misleading, so start a clean scoped thread.
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []

        st.caption(
            f"Document ID: {st.session_state.active_document_id}"
        )

        if st.button("🗑️ Delete this document", type="secondary"):
            with st.spinner("Deleting document..."):
                delete_response = http.delete(
                    f"{API_URL}/documents/{st.session_state.active_document_id}"
                )
                delete_response.raise_for_status()

            st.session_state.active_document_id = None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.documents = load_documents()
            st.rerun()
            
    else:
        st.warning(
            "The registry contains documents without document IDs. "
            "Re-ingest them with the updated pipeline."
        )
        st.session_state.active_document_id = None
else:
    st.info("Upload and ingest a document before asking document-scoped questions.")
    st.session_state.active_document_id = None


# ============================================================
# Divider
# ============================================================

st.divider()


# ============================================================
# Conversation History
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")

        # Show grounding information only for assistant messages
        if message["role"] == "assistant":

            if message.get("verification_exhausted"):
                st.warning(
                    "⚠️ This answer could not be fully verified against the "
                    "source after the available verification attempts. "
                    "Please double-check it against the document."
                )
            elif message.get("is_control"):
                pass
            elif message.get("answer_status") == "insufficient_evidence":
                st.caption("Not enough evidence in the selected document")
            elif message.get("grounded"):
                st.caption("Grounded in context")
            else:
                st.caption("Could not fully verify against context")


# ============================================================
# New Question
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# Process Question
# ============================================================

if question:

    if not st.session_state.active_document_id:
        st.warning(
            "Please upload and select an active document before asking a question."
        )
        st.stop()

    # --------------------------------------------------------
    # Display user question immediately
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Store user question
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # --------------------------------------------------------
    # Query FastAPI backend
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = http.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "document_id": st.session_state.active_document_id,
                    },
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                answer = data["answer"]
                grounded = data["grounded"]
                answer_status = data.get("answer_status", "verification_uncertain")
                verification_exhausted = data.get("verification_exhausted", False)
                is_control = data.get("is_control", False)
                citations = data.get("citations", [])

            except requests.RequestException as exc:

                answer = (
                    "I couldn't connect to the RAG backend. "
                    "Please make sure the FastAPI server is running."
                )

                grounded = False
                answer_status = "request_failed"
                verification_exhausted = False
                is_control = False
                citations = []

                st.error(str(exc))


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(answer)

        if citations:
            with st.expander("Sources"):
                for citation in citations:
                    st.markdown(f"- {citation}")

        if verification_exhausted:
            st.warning(
                "⚠️ This answer could not be fully verified against the "
                "source after the available verification attempts. "
                "Please double-check it against the document."
            )
        elif is_control:
            pass  # no grounding badge for conversational control turns
        elif answer_status == "insufficient_evidence":
            st.caption("Not enough evidence in the selected document")
        elif grounded:
            st.caption("Grounded in context")
        else:
            st.caption("Could not fully verify against context")


    # --------------------------------------------------------
    # Store assistant response
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "grounded": grounded,
            "answer_status": answer_status,
            "verification_exhausted": verification_exhausted,
            "is_control": is_control,
            "citations": citations,
        }
    )
