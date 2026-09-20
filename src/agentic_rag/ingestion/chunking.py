"""Document chunking with document-level metadata propagation.

Chunks are split by TOKEN count using the embedding model's own tokenizer,
not raw character count. all-MiniLM-L6-v2 (the configured embedding model)
has a 256-token max sequence length and silently truncates anything longer -
the previous character-based chunk_size (1500 chars, ~350-375 tokens for
typical English text) regularly exceeded that limit, meaning the tail of
longer chunks was silently dropped from the DENSE embedding even though the
full text still reached the reranker and the generation LLM. Token-based
splitting guarantees every chunk fits within the model's actual window
regardless of how token-dense the source text is.
"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from agentic_rag.config import settings


@lru_cache(maxsize=1)
def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Built once and cached - loading a tokenizer per call would be wasteful
    given chunk_documents may be invoked once per ingested file."""
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def chunk_documents(
    docs: list[Document],
    *,
    document_id: str | None = None,
    filename: str | None = None,
) -> list[Document]:
    """Split documents while preserving and enriching chunk metadata.

    Existing loader metadata such as page numbers is preserved.
    document_id and filename are added to every resulting chunk when supplied.
    """
    splitter = _get_splitter()

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        if document_id is not None:
            chunk.metadata["document_id"] = document_id

        if filename is not None:
            chunk.metadata["filename"] = filename
            chunk.metadata["source"] = filename

        chunk.metadata.setdefault("type", "content")

    return chunks