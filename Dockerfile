FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/appuser/.cache/huggingface

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY app/ app/
RUN uv sync --frozen --no-dev \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/logs /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000
CMD ["uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
