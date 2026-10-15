"""Deployment contracts for self-contained local retrieval models."""

import importlib.util
from pathlib import Path

from agentic_rag.model_identity import canonical_model_name


ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_MODEL = "cross-encoder/ettin-reranker-32m-v1"
RERANKER_REVISION = "b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd"


def test_docker_image_pins_and_verifies_both_models_offline() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert EMBEDDING_REVISION in dockerfile
    assert RERANKER_MODEL in dockerfile
    assert RERANKER_REVISION in dockerfile
    assert "python scripts/cache_models.py --verify-only" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "MODEL_LOCAL_FILES_ONLY=true" in dockerfile


def test_dockerfile_packages_selected_v3_parser_in_both_runtime_targets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/offline.yml").read_text(encoding="utf-8")

    assert "FROM application-base AS cpu-runtime" in dockerfile
    assert "FROM application-base AS gpu-runtime" in dockerfile
    assert "Acquire::Retries=5" in dockerfile
    assert "--torch-backend \"$TORCH_BACKEND\"" in dockerfile
    assert "assert torch.version.cuda == '13.0'" in dockerfile
    assert "python scripts/verify_v3_parser_runtime.py" in dockerfile
    assert "docling-tools models download" not in dockerfile
    assert '"docling[rapidocr]' not in dockerfile
    assert "target: cpu-runtime" in workflow
    assert "--network none agentic-rag-ci:local" in workflow
    assert "python scripts/verify_v3_parser_runtime.py" in workflow


def test_docker_image_bakes_and_verifies_bm25_tokenizer_data() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/offline.yml").read_text(encoding="utf-8")

    assert "NLTK_DATA=/home/appuser/nltk_data" in dockerfile
    assert "python scripts/cache_nltk.py" in dockerfile
    assert "python scripts/cache_nltk.py --verify-only" in dockerfile
    assert "--network none agentic-rag-ci:local" in workflow
    assert "python scripts/cache_nltk.py --verify-only" in workflow


def test_compose_uses_one_built_application_image() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert compose.count("image: agentic-rag-app:local") == 2
    assert compose.count("context: .") == 1
    assert compose.count("target: gpu-runtime") == 1


def test_local_runner_does_not_hide_baked_models_with_host_cache() -> None:
    path = ROOT / "scripts/local_docker.py"
    spec = importlib.util.spec_from_file_location("local_docker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    override = module._override_text(bounded=True, use_gpu=True)

    assert "/app/logs" in override
    assert "/home/appuser/.cache/huggingface" not in override
    assert "HF_HUB_OFFLINE" in override
    assert "gpus: all" in override
    assert 'EMBEDDING_DEVICE: "cuda"' in override
    assert 'CROSS_ENCODER_DEVICE: "cuda"' in override

    cpu_override = module._override_text(bounded=False, use_gpu=False)
    assert "gpus: all" not in cpu_override
    assert 'EMBEDDING_DEVICE: "cpu"' in cpu_override
    assert 'CROSS_ENCODER_DEVICE: "cpu"' in cpu_override


def test_historical_model_names_resolve_to_baked_repositories() -> None:
    assert canonical_model_name("all-MiniLM-L6-v2") == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert canonical_model_name("cross-encoder/ms-marco-MiniLM-L-6-v2") == (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )


def test_example_configuration_selects_pinned_ettin_reranker() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert f"CROSS_ENCODER_MODEL={RERANKER_MODEL}" in example
    assert f"CROSS_ENCODER_MODEL_REVISION={RERANKER_REVISION}" in example
    assert "CROSS_ENCODER_BATCH_SIZE=8" in example
