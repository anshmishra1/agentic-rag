from __future__ import annotations

import json
import logging
import platform
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# Agentic RAG - Central Logging Module
# =============================================================================
#
# Run layout:
#
# logs/
# └── runs/
#     └── YYYY-MM-DD_HH-MM-SS/
#         ├── application.log
#         ├── detailed_debug.log
#         ├── diagnostics.jsonl
#         ├── diagnostics.json
#         └── run_metadata.json
#
# =============================================================================


_CONFIGURED = False

_RUN_ID: str | None = None
_RUN_DIR: Path | None = None

_DIAGNOSTIC_LOCK = threading.Lock()
_DIAGNOSTIC_EVENTS: list[dict[str, Any]] = []


# =============================================================================
# Log formats
# =============================================================================

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(message)s"
)

_DETAILED_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(filename)s:%(lineno)d | "
    "%(funcName)s() | %(message)s"
)


# =============================================================================
# Noisy third-party libraries
# =============================================================================

_NOISY_LOGGERS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "urllib3": logging.WARNING,
    "filelock": logging.WARNING,
    "watchfiles": logging.WARNING,
    "asyncio": logging.WARNING,
}


# =============================================================================
# Console filter
# =============================================================================

class _ConsoleFilter(logging.Filter):
    """Keep console output focused on useful application events."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO


# =============================================================================
# Utility helpers
# =============================================================================

def _safe_json(value: Any) -> Any:
    """
    Convert arbitrary Python objects into JSON-serializable values.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _safe_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _safe_json(item)
            for item in value
        ]

    try:
        json.dumps(value)
        return value

    except (TypeError, ValueError):
        return str(value)


# =============================================================================
# Run directory management
# =============================================================================

def _create_run_directory(
    log_root: str | Path,
) -> tuple[str, Path]:
    """
    Create a unique directory for the current application run.

    Example:

        logs/
        └── runs/
            └── 2026-09-01_14-30-25/
    """

    root = Path(log_root).resolve() / "runs"

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    run_dir = root / timestamp

    # Protect against multiple processes starting
    # within the same second.
    counter = 1

    while run_dir.exists():

        run_dir = root / (
            f"{timestamp}_{counter:02d}"
        )

        counter += 1

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_dir.name, run_dir


# =============================================================================
# Public run helpers
# =============================================================================

def get_run_id() -> str:
    """
    Return the ID of the current application run.

    Raises:
        RuntimeError:
            If logging has not been configured.
    """

    if _RUN_ID is None:

        raise RuntimeError(
            "Logging has not been configured yet. "
            "Call configure_logging() first."
        )

    return _RUN_ID


def get_run_directory() -> Path:
    """
    Return the directory containing artifacts for
    the current application run.
    """

    if _RUN_DIR is None:

        raise RuntimeError(
            "Logging has not been configured with "
            "file logging."
        )

    return _RUN_DIR


def get_run_file(
    filename: str,
) -> Path:
    """
    Return a path inside the current run directory.

    Example:

        get_run_file("diagnostics.json")
    """

    path = Path(filename)

    if path.is_absolute():

        raise ValueError(
            "filename must be a relative path "
            "inside the run directory"
        )

    if ".." in path.parts:

        raise ValueError(
            "filename cannot contain '..'"
        )

    return get_run_directory() / path

def _append_jsonl(
    filename: str,
    data: dict[str, Any],
) -> None:
    """Append a structured record to a JSONL file."""

    if _RUN_DIR is None:
        return

    path = get_run_file(filename)

    with path.open("a", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            default=str,
        )
        file.write("\n")


# =============================================================================
# Logger access
# =============================================================================

def get_logger(
    name: str | None = None,
) -> logging.Logger:
    """
    Return an application logger.

    Existing application code can therefore continue to use:

        logger = get_logger(__name__)

    or:

        logger = get_logger("agentic_rag.retrieval")
    """

    return logging.getLogger(
        name or "agentic_rag"
    )


# =============================================================================
# Structured diagnostics
# =============================================================================

def _write_diagnostic_event(
    event: dict[str, Any],
) -> None:
    """
    Append one structured event to diagnostics.jsonl
    and update diagnostics.json.

    Logging failures are deliberately swallowed because
    logging must never crash the application.
    """

    if _RUN_DIR is None:
        return

    event = _safe_json(event)

    try:

        with _DIAGNOSTIC_LOCK:

            _DIAGNOSTIC_EVENTS.append(
                event
            )

            # -------------------------------------------------------------
            # JSONL
            # -------------------------------------------------------------

            jsonl_path = (
                _RUN_DIR
                / "diagnostics.jsonl"
            )

            with jsonl_path.open(
                "a",
                encoding="utf-8",
            ) as file:

                file.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

            # -------------------------------------------------------------
            # Consolidated JSON
            # -------------------------------------------------------------

            json_path = (
                _RUN_DIR
                / "diagnostics.json"
            )

            with json_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    {
                        "run_id": _RUN_ID,
                        "event_count": len(
                            _DIAGNOSTIC_EVENTS
                        ),
                        "events": (
                            _DIAGNOSTIC_EVENTS
                        ),
                    },
                    file,
                    indent=4,
                    ensure_ascii=False,
                    default=str,
                )

    except Exception:
        # Never allow logging to crash the application.
        pass


def _log_structured_event(
    event_type: str,
    *,
    message: str | None = None,
    level: int = logging.INFO,
    logger_name: str = "agentic_rag",
    **data: Any,
) -> None:
    """
    Write an event to:

    1. Console
    2. application.log
    3. detailed_debug.log
    4. diagnostics.jsonl
    5. diagnostics.json
    """

    logger = get_logger(
        logger_name
    )

    clean_data = {
        key: _safe_json(value)
        for key, value in data.items()
        if value is not None
    }

    # ---------------------------------------------------------------------
    # Human-readable logging
    # ---------------------------------------------------------------------

    if message:

        if clean_data:

            logger.log(
                level,
                "%s | %s",
                message,
                json.dumps(
                    clean_data,
                    ensure_ascii=False,
                    default=str,
                ),
            )

        else:

            logger.log(
                level,
                message,
            )

    # ---------------------------------------------------------------------
    # Structured event
    # ---------------------------------------------------------------------

    event = {
        "timestamp": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "run_id": _RUN_ID,
        "event_type": event_type,
        "level": logging.getLevelName(
            level
        ),
        "logger": logger_name,
        **clean_data,
    }

    _write_diagnostic_event(
        event
    )


# =============================================================================
# Graph logging
# =============================================================================

def log_graph(
    graph_name: str,
    *,
    event: str = "graph_execution",
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Log a LangGraph lifecycle event.

    Examples:

        log_graph(
            "agentic_rag",
            event="graph_started",
        )

        log_graph(
            "agentic_rag",
            event="graph_completed",
            details={
                "nodes_executed": 5
            },
        )
    """

    _log_structured_event(
        "graph",
        message=(
            f"Graph event | "
            f"{graph_name} | "
            f"{event}"
        ),
        level=level,
        logger_name="agentic_rag.graph",
        graph=graph_name,
        event=event,
        details=details or {},
    )


# =============================================================================
# LLM logging
# =============================================================================

def log_llm(
    model: str,
    *,
    prompt: str | None = None,
    response: str | None = None,
    latency_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    temperature: float | None = None,
    success: bool | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """
    Log an LLM invocation.

    Supports different LLM providers without
    coupling logging to a particular provider.
    """

    _log_structured_event(
        "llm",
        message=(
            f"LLM call | model={model}"
        ),
        level=(
            logging.ERROR
            if error
            else level
        ),
        logger_name="agentic_rag.llm",
        model=model,
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        temperature=temperature,
        success=success,
        error=error,
        details=details,
        **kwargs,
    )


# =============================================================================
# Retrieval logging
# =============================================================================

def log_retrieval(
    logger: logging.Logger,
    message: str,
    *args: Any,
    enabled: bool = True,
) -> None:
    """
    Log a retrieval-related event.

    Supports standard logging-style formatting:

        log_retrieval(
            logger,
            "query=%r | candidates=%d | top=%.4f",
            query,
            len(scores),
            top_score,
        )

    The function also writes the formatted retrieval event to the
    per-run retrieval.jsonl artifact.
    """

    if not enabled:
        return

    # ---------------------------------------------------------------
    # Format the message using standard logging semantics
    # ---------------------------------------------------------------

    try:
        formatted_message = message % args if args else message
    except (TypeError, ValueError):
        # Defensive fallback. We never want logging itself to crash
        # the application.
        formatted_message = (
            f"{message} | formatting_args={args!r}"
        )

    # ---------------------------------------------------------------
    # Standard application logging
    # ---------------------------------------------------------------

    logger.info(
        "RETRIEVAL | %s",
        formatted_message,
    )

    # ---------------------------------------------------------------
    # Structured retrieval artifact
    # ---------------------------------------------------------------

    _append_jsonl(
        "retrieval.jsonl",
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "retrieval",
            "message": formatted_message,
        },
    )


# =============================================================================
# Embedding logging
# =============================================================================

def log_embedding(
    model: str,
    *,
    text_count: int | None = None,
    dimensions: int | None = None,
    device: str | None = None,
    latency_ms: float | None = None,
    success: bool | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """
    Log an embedding-model operation.

    Useful for recording GPU/CPU usage, model name,
    embedding dimensions and latency.
    """

    _log_structured_event(
        "embedding",
        message=(
            f"Embedding | model={model}"
        ),
        level=(
            logging.ERROR
            if error
            else level
        ),
        logger_name="agentic_rag.embedding",
        model=model,
        text_count=text_count,
        dimensions=dimensions,
        device=device,
        latency_ms=latency_ms,
        success=success,
        error=error,
        details=details,
        **kwargs,
    )


# =============================================================================
# Tool logging
# =============================================================================

def log_tool(
    tool_name: str,
    *,
    arguments: Any = None,
    result: Any = None,
    latency_ms: float | None = None,
    success: bool | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """
    Log an agent/tool invocation.
    """

    _log_structured_event(
        "tool",
        message=(
            f"Tool call | {tool_name}"
        ),
        level=(
            logging.ERROR
            if error
            else level
        ),
        logger_name="agentic_rag.tool",
        tool=tool_name,
        arguments=arguments,
        result=result,
        latency_ms=latency_ms,
        success=success,
        error=error,
        details=details,
        **kwargs,
    )


# =============================================================================
# Error logging
# =============================================================================

def log_error(
    error: Exception | str,
    *,
    context: str | None = None,
    details: dict[str, Any] | None = None,
    logger_name: str = "agentic_rag.error",
    exc_info: bool = False,
    **kwargs: Any,
) -> None:
    """
    Log an application error.

    Captures:

    - error message
    - exception type
    - traceback
    - context
    - additional details
    """

    if isinstance(
        error,
        BaseException,
    ):

        error_message = str(
            error
        )

        error_type = type(
            error
        ).__name__

        traceback_text = "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        )

    else:

        error_message = str(
            error
        )

        error_type = type(
            error
        ).__name__

        traceback_text = None

    _log_structured_event(
        "error",
        message=(
            "Error | "
            f"{context + ' | ' if context else ''}"
            f"{error_message}"
        ),
        level=logging.ERROR,
        logger_name=logger_name,
        error=error_message,
        error_type=error_type,
        context=context,
        traceback=traceback_text,
        details=details,
        **kwargs,
    )

    if exc_info:

        get_logger(
            logger_name
        ).exception(
            "Exception details",
            exc_info=True,
        )


# =============================================================================
# Generic event logging
# =============================================================================

def log_event(
    event_type: str,
    *,
    message: str | None = None,
    level: int = logging.INFO,
    logger_name: str = "agentic_rag",
    **data: Any,
) -> None:
    """
    Generic structured logging entry point.

    Useful when an event does not fit one of the
    specialized log_* functions.
    """

    _log_structured_event(
        event_type,
        message=(
            message
            or event_type
        ),
        level=level,
        logger_name=logger_name,
        **data,
    )


# =============================================================================
# Run metadata
# =============================================================================

def _write_run_metadata() -> None:
    """
    Write environment information for reproducibility.
    """

    metadata: dict[str, Any] = {
        "run_id": get_run_id(),
        "started_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": (
            platform.python_version()
        ),
        "python_executable": sys.executable,
    }

    # ---------------------------------------------------------------------
    # PyTorch / CUDA information
    # ---------------------------------------------------------------------

    try:

        import torch

        cuda_available = (
            torch.cuda.is_available()
        )

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_available": (
                cuda_available
            ),
            "cuda_version": (
                torch.version.cuda
            ),
            "device": (
                torch.cuda.get_device_name(0)
                if cuda_available
                else "cpu"
            ),
            "cuda_device": (
                "cuda:0"
                if cuda_available
                else "cpu"
            ),
        }

    except Exception as exc:

        metadata["torch"] = {
            "available": False,
            "error": str(exc),
        }

    # ---------------------------------------------------------------------
    # Application configuration
    # ---------------------------------------------------------------------

    try:

        from src.agentic_rag.config import settings

        metadata["application"] = {
            "embedding_model": getattr(
                settings,
                "embedding_model",
                None,
            ),
        }

    except Exception:
        # Configuration should never prevent
        # logging from being initialized.
        pass

    # ---------------------------------------------------------------------
    # Write metadata
    # ---------------------------------------------------------------------

    metadata_path = get_run_file(
        "run_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


# =============================================================================
# Main logging configuration
# =============================================================================

def configure_logging(
    debug: bool = False,
    *,
    log_dir: str | Path = "logs",
    file_logging: bool = True,
) -> None:
    """
    Configure application logging for one application run.

    A fresh run directory is created once per process.

    Example:

        logs/
        └── runs/
            └── 2026-09-01_14-30-25/
                ├── application.log
                ├── detailed_debug.log
                ├── diagnostics.json
                ├── diagnostics.jsonl
                └── run_metadata.json

    Safe to call multiple times within the same process.
    """

    global _CONFIGURED
    global _RUN_ID
    global _RUN_DIR

    if _CONFIGURED:
        return

    level = (
        logging.DEBUG
        if debug
        else logging.INFO
    )

    # =========================================================================
    # Create run directory
    # =========================================================================

    if file_logging:

        _RUN_ID, _RUN_DIR = (
            _create_run_directory(
                log_dir
            )
        )

        application_handler = (
            logging.FileHandler(
                _RUN_DIR
                / "application.log",
                mode="a",
                encoding="utf-8",
            )
        )

        detailed_handler = (
            logging.FileHandler(
                _RUN_DIR
                / "detailed_debug.log",
                mode="a",
                encoding="utf-8",
            )
        )

        application_handler.setLevel(
            logging.INFO
        )

        detailed_handler.setLevel(
            logging.DEBUG
        )

        application_handler.setFormatter(
            logging.Formatter(
                _LOG_FORMAT
            )
        )

        detailed_handler.setFormatter(
            logging.Formatter(
                _DETAILED_LOG_FORMAT
            )
        )

        # ---------------------------------------------------------------------
        # Create structured diagnostic files immediately
        # ---------------------------------------------------------------------

        (
            _RUN_DIR
            / "diagnostics.jsonl"
        ).touch()

        with (
            _RUN_DIR
            / "diagnostics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "run_id": _RUN_ID,
                    "event_count": 0,
                    "events": [],
                },
                file,
                indent=4,
                ensure_ascii=False,
            )

    else:

        _RUN_ID = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        _RUN_DIR = None

        application_handler = None
        detailed_handler = None

    # =========================================================================
    # Console handler
    # =========================================================================

    console_handler = (
        logging.StreamHandler(
            sys.stdout
        )
    )

    console_handler.setLevel(
        logging.INFO
    )

    console_handler.setFormatter(
        logging.Formatter(
            _LOG_FORMAT
        )
    )

    console_handler.addFilter(
        _ConsoleFilter()
    )

    # =========================================================================
    # Assemble handlers
    # =========================================================================

    handlers: list[
        logging.Handler
    ] = [
        console_handler
    ]

    if application_handler is not None:

        handlers.append(
            application_handler
        )

    if detailed_handler is not None:

        handlers.append(
            detailed_handler
        )

    # =========================================================================
    # Configure root logger
    # =========================================================================

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    # =========================================================================
    # Silence noisy third-party libraries
    # =========================================================================

    for (
        logger_name,
        logger_level,
    ) in _NOISY_LOGGERS.items():

        logging.getLogger(
            logger_name
        ).setLevel(
            logger_level
        )

    # =========================================================================
    # Write run metadata
    # =========================================================================

    if file_logging:

        _write_run_metadata()

    # =========================================================================
    # Mark configured
    # =========================================================================

    _CONFIGURED = True

    logger = get_logger(
        "agentic_rag.logging"
    )

    logger.info(
        "Logging initialized | run_id=%s",
        _RUN_ID,
    )

    if _RUN_DIR is not None:

        logger.info(
            "Run artifacts directory | %s",
            _RUN_DIR,
        )

    # =========================================================================
    # Record run-start event
    # =========================================================================

    _write_diagnostic_event(
        {
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "run_id": _RUN_ID,
            "event_type": "run_started",
            "level": "INFO",
            "logger": (
                "agentic_rag.logging"
            ),
            "debug": debug,
            "file_logging": file_logging,
        }
    )

# =============================================================================
# Performance logging
# =============================================================================

def log_performance(
    stage: str,
    *,
    latency_ms: float | None = None,
    duration_ms: float | None = None,
    success: bool | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """
    Log performance information for an application stage.

    Examples:
        log_performance(
            "retrieval",
            latency_ms=125.4,
        )

        log_performance(
            "reranking",
            duration_ms=84.2,
            details={
                "documents": 10,
                "device": "cuda:0",
            },
        )
    """

    _log_structured_event(
        "performance",
        message=(
            f"Performance | stage={stage}"
        ),
        level=(
            logging.ERROR
            if error
            else level
        ),
        logger_name="agentic_rag.performance",
        stage=stage,
        latency_ms=latency_ms,
        duration_ms=duration_ms,
        success=success,
        error=error,
        details=details,
        **kwargs,
    )
