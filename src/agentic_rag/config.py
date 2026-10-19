"""Central application settings loaded from environment / .env."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_rag.model_identity import (
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_REVISION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
)


class Settings(BaseSettings):
    """Runtime configuration for the PDF RAG application."""

    # API Keys
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    nvidia_api_key: str = ""
    openrouter_api_key: str = ""
    pinecone_api_key: str = ""
    hf_token: str = ""

    # Primary provider models
    groq_model: str = "openai/gpt-oss-120b"
    cerebras_model: str = "gpt-oss-120b"
    nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    openrouter_model: str = "openai/gpt-4o-mini"
    bedrock_model: str = "meta.llama3-3-70b-instruct-v1:0"

    # Fast-tier models
    groq_fast_model: str = "openai/gpt-oss-20b"
    groq_verifier_reasoning_effort: Literal["low", "medium", "high"] = "low"
    cerebras_fast_model: str = "llama3.1-8b"
    nvidia_fast_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    openrouter_fast_model: str = "qwen/qwen-2.5-7b-instruct"
    bedrock_fast_model: str = "us.meta.llama3-1-8b-instruct-v1:0"

    # Provider endpoints and fallback order
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    provider_order: str = "cerebras,groq,nvidia,openrouter"

    # LLM request timeouts
    fast_llm_timeout: float = 15.0
    primary_llm_timeout: float = 45.0
    fast_llm_max_tokens: int = 512
    semantic_router_max_tokens: int = 256
    relevance_grader_max_tokens: int = 512
    query_rewrite_max_tokens: int = 512
    grounding_verifier_max_tokens: int = 1024
    primary_llm_max_tokens: int = 2048

    # Bedrock
    bedrock_enabled: bool = False
    aws_bedrock_region: str = "us-east-1"

    # Vector database
    # The index name is the retrieval schema identity. V2 is the current text
    # runtime; later parser or embedding migrations use another numeric suffix.
    pinecone_index_name: str = "agentic-rag-hybrid-v2"

    # PostgreSQL / LangGraph checkpointing
    postgres_url: str = (
        "postgresql://postgres:postgres@localhost:5442/postgres"
        "?sslmode=disable"
    )

    # RAG / pipeline
    embedding_model: str = EMBEDDING_MODEL
    embedding_model_revision: str = EMBEDDING_REVISION
    embedding_dimension: int = Field(default=EMBEDDING_DIMENSION, ge=1)
    embedding_device: Literal["auto", "cuda", "cpu"] = "auto"
    model_local_files_only: bool = False
    # NOTE: units are TOKENS (per the embedding model's own tokenizer), not
    # characters - see ingestion/chunking.py. all-MiniLM-L6-v2's max sequence
    # length is 256 tokens; 240 leaves headroom for any tokenizer rounding.
    chunk_size: int = 240
    chunk_overlap: int = 40
    chunking_strategy: Literal["token_window_v1", "structure_aware_v2"] = "structure_aware_v2"
    max_retries: int = 2

    # Multimodal V3 parser. These values are part of the retrieval-schema
    # identity once the V3 index writer is connected.
    v3_parser_backend: Literal["pymupdf_layout"] = "pymupdf_layout"
    v3_parser_render_dpi: int = Field(default=144, ge=72, le=600)
    v3_parser_ocr_dpi: int = Field(default=150, ge=72, le=600)
    v3_parser_ocr_enabled: bool = True
    v3_index_name: str = "agentic-rag-hybrid-v3"
    v3_asset_root: Path = Path("/app/data/v3-assets")
    # Search proxies share MiniLM's 256-token encoder. The smaller payload
    # leaves room for tokenizer special tokens and prevents silent truncation.
    v3_search_record_max_tokens: int = Field(default=240, ge=32, le=240)
    # Context capacity reserved for the source form explicitly requested by
    # the user. This is a slot budget, not a relevance-score threshold.
    v3_required_evidence_slots: int = Field(default=2, ge=1, le=5)
    v3_evidence_companion_slots: int = Field(default=1, ge=0, le=2)

    # Conversation / generation optimization
    max_history_messages_for_generation: int = 2
    # Direct content and whole-document overviews have separate budgets so an
    # overview cannot silently displace a content passage retained by the
    # reranker's recall safeguard.
    max_generation_context_documents: int = 5
    max_generation_context_overviews: int = 1
    max_generation_context_chars: int = 12000
    max_generation_output_chars: int = 12000

    # Retrieval assessment - absolute score gate
    # IMPORTANT: these were calibrated against raw Pinecone cosine similarity
    # scores. Hybrid search + cross-encoder reranking changed the score source
    # entirely (sigmoid-activated cross-encoder relevance, not cosine
    # similarity). Cross-encoder scores are ranking signals, not calibrated
    # relevance probabilities. Ettin's measured sigmoid values are saturated,
    # so these legacy thresholds remain provisional until positive and negative
    # routing labels establish a replacement policy.
    retrieval_min_top_score: float = 0.30
    retrieval_strong_top_score: float = 0.60

    # Retrieval assessment - relative shape gate (used only once the absolute
    # floor above is cleared). Moved here from module-level constants in
    # nodes.py so every retrieval threshold lives in one place.
    retrieval_top_to_mean_ratio: float = 1.10
    retrieval_gap_ratio: float = 0.02
    retrieval_overview_margin: float = 0.02

    # Hybrid search (dense + sparse BM25) and cross-encoder reranking
    hybrid_alpha: float = 0.5              # 1.0 = pure dense, 0.0 = pure sparse
    hybrid_candidate_k: int = 15           # candidates pulled from Pinecone before reranking
    rrf_k: int = 60
    rerank_top_k_content: int = 5
    rerank_top_k_overview: int = 2
    rerank_content_rrf_reserve: int = 2  # retain these top RRF content IDs within the existing content budget
    cross_encoder_model: str = CROSS_ENCODER_MODEL
    cross_encoder_model_revision: str = CROSS_ENCODER_REVISION
    # Local reranker execution. "auto" selects CUDA when available, else CPU.
    cross_encoder_device: Literal["auto", "cuda", "cpu"] = "auto"
    # Batch 8 was faster than 32 on the fixed local 15-candidate evaluation.
    cross_encoder_batch_size: int = 8

    # Development
    debug: bool = False
    retrieval_logs: bool = True
    llm_logs: bool = True
    performance_logs: bool = True
    graph_logs: bool = False

    llm_rate_limit_cooldown_seconds: float = Field(default=60.0, ge=0.0)
    llm_rate_limit_max_cooldown_seconds: float = Field(default=300.0, ge=0.0)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
