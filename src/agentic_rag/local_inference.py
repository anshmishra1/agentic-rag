"""Process-local coordination for models that share one accelerator."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Iterator


_LOCAL_INFERENCE_LOCK = RLock()


@contextmanager
def local_inference_slot() -> Iterator[None]:
    """Serialize local model inference within one application process.

    The local RTX 2050 has 4 GiB VRAM. Embedding, reranking, and Docling may
    share it, but overlapping inference can exceed that budget or create
    allocator fragmentation. The reentrant lock also permits warm-up helpers
    to call model loaders that use the same slot.
    """

    with _LOCAL_INFERENCE_LOCK:
        yield


def resolve_torch_device(configured: str, *, cuda_available: bool) -> str:
    """Resolve ``auto`` and fail clearly when requested CUDA is absent."""

    if configured not in {"auto", "cpu", "cuda"}:
        raise ValueError("local inference device must be auto, cpu, or cuda")
    if configured == "auto":
        return "cuda" if cuda_available else "cpu"
    if configured == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested for local inference but is unavailable")
    return configured
