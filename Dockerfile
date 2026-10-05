FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/appuser/.cache/uv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/appuser/.cache/huggingface

RUN useradd --uid 10001 --user-group --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/logs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --no-install-project

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBEDDING_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ARG CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
ARG CROSS_ENCODER_MODEL_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a

COPY --chown=appuser:appuser scripts/cache_models.py scripts/cache_models.py
RUN python scripts/cache_models.py \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-revision "$EMBEDDING_MODEL_REVISION" \
    --reranker-model "$CROSS_ENCODER_MODEL" \
    --reranker-revision "$CROSS_ENCODER_MODEL_REVISION"

ENV EMBEDDING_MODEL=$EMBEDDING_MODEL \
    EMBEDDING_MODEL_REVISION=$EMBEDDING_MODEL_REVISION \
    CROSS_ENCODER_MODEL=$CROSS_ENCODER_MODEL \
    CROSS_ENCODER_MODEL_REVISION=$CROSS_ENCODER_MODEL_REVISION \
    MODEL_LOCAL_FILES_ONLY=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN python scripts/cache_models.py --verify-only

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser app/ app/
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=10001,gid=10001 \
    uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
