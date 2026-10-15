"""Device-selection contracts for shared local inference."""

import pytest

from agentic_rag.local_inference import resolve_torch_device


def test_auto_uses_cuda_when_available() -> None:
    assert resolve_torch_device("auto", cuda_available=True) == "cuda"


def test_auto_falls_back_to_cpu_without_cuda() -> None:
    assert resolve_torch_device("auto", cuda_available=False) == "cpu"


def test_explicit_cpu_is_portable() -> None:
    assert resolve_torch_device("cpu", cuda_available=True) == "cpu"


def test_explicit_cuda_fails_clearly_when_unavailable() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_torch_device("cuda", cuda_available=False)


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        resolve_torch_device("gpu", cuda_available=True)
