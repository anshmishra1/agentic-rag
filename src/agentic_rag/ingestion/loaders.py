"""Multimodal document loaders: PDF, images, audio."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path, *, extraction_mode: str = "plain") -> list[Document]:
    """Load one document per PDF page with an explicit extraction mode."""
    documents = PyPDFLoader(str(path), extraction_mode=extraction_mode).load()
    for document in documents:
        document.metadata["extraction_mode"] = extraction_mode
    return documents


def load_image(path: str | Path) -> list[Document]:
    """Captions the image via a multimodal LLM and indexes the caption as searchable text."""
    from agentic_rag.llm.vision import caption_image

    caption = caption_image(path)
    return [Document(page_content=caption, metadata={"source": str(path), "type": "image"})]


def load_audio(path: str | Path) -> list[Document]:
    """Transcribes audio via Groq's hosted Whisper endpoint."""
    from agentic_rag.ingestion.whisper_transcribe import transcribe

    text = transcribe(path)
    return [Document(page_content=text, metadata={"source": str(path), "type": "audio"})]
