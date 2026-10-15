FROM python:3.12-slim AS application-base

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/appuser/.cache/uv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/appuser/.cache/huggingface \
    NLTK_DATA=/home/appuser/nltk_data

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install --yes --no-install-recommends \
      libgl1 \
      libglib2.0-0t64 \
      libgomp1 \
      libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10001 --user-group --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/logs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --no-install-project

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBEDDING_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ARG CROSS_ENCODER_MODEL=cross-encoder/ettin-reranker-32m-v1
ARG CROSS_ENCODER_MODEL_REVISION=b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd

COPY --chown=appuser:appuser scripts/cache_models.py scripts/cache_models.py
RUN python scripts/cache_models.py \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-revision "$EMBEDDING_MODEL_REVISION" \
    --reranker-model "$CROSS_ENCODER_MODEL" \
    --reranker-revision "$CROSS_ENCODER_MODEL_REVISION"

COPY --chown=appuser:appuser scripts/cache_nltk.py scripts/cache_nltk.py
COPY --chown=appuser:appuser scripts/verify_v3_parser_runtime.py scripts/verify_v3_parser_runtime.py
RUN python scripts/cache_nltk.py

ENV EMBEDDING_MODEL=$EMBEDDING_MODEL \
    EMBEDDING_MODEL_REVISION=$EMBEDDING_MODEL_REVISION \
    CROSS_ENCODER_MODEL=$CROSS_ENCODER_MODEL \
    CROSS_ENCODER_MODEL_REVISION=$CROSS_ENCODER_MODEL_REVISION \
    MODEL_LOCAL_FILES_ONLY=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN python scripts/cache_models.py --verify-only \
    && python scripts/cache_nltk.py --verify-only

FROM application-base AS cpu-runtime

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser app/ app/
RUN mkdir -p /app/data/v3-assets \
    && uv pip install --python /app/.venv/bin/python --no-deps --no-build-isolation . \
    && python scripts/verify_v3_parser_runtime.py

EXPOSE 8000
CMD ["uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.opencontainers.image.title="AgenticRAG CPU verification runtime" \
      org.opencontainers.image.description="CPU target used by offline CI and non-CUDA hosts"

FROM application-base AS gpu-runtime

ARG TORCH_BACKEND=cu130

# Ettin's ModernBERT CUDA path uses Triton kernels. Triton compiles a small
# launcher when that path first executes, so the GPU runtime needs a C compiler
# even though the application and parser dependencies are already installed.
USER root
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install --yes --no-install-recommends \
      gcc \
      libc6-dev \
    && rm -rf /var/lib/apt/lists/*
USER appuser

# Replace the Linux CPU PyTorch wheels with CUDA wheels in this target. The
# resulting image runs embedding and Ettin reranking on the GPU while retaining
# CPU execution for PyMuPDF Layout, RapidOCR, BM25, and services.
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=10001,gid=10001 \
    uv pip install \
      --python /app/.venv/bin/python \
      --no-config \
      --no-sources \
      --torch-backend "$TORCH_BACKEND" \
      --reinstall-package torch \
      --reinstall-package torchvision \
      "torch==2.13.0" \
      "torchvision==0.28.0"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN python -c "import torch; assert torch.version.cuda == '13.0'; print('torch_cuda_build=' + torch.version.cuda)"

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser app/ app/
RUN mkdir -p /app/data/v3-assets \
    && uv pip install --python /app/.venv/bin/python --no-deps --no-build-isolation . \
    && python scripts/verify_v3_parser_runtime.py \
    && python -c "import torch; assert torch.version.cuda == '13.0'; print('final_torch_cuda_build=' + torch.version.cuda)"

EXPOSE 8000
CMD ["uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.opencontainers.image.title="AgenticRAG V2/V3 CUDA runtime" \
      org.opencontainers.image.description="One shared image for CUDA embedding, Ettin reranking, and pinned PyMuPDF Layout parsing"
