"""Deployment contracts for self-contained local retrieval models."""

import importlib.util
from pathlib import Path

from agentic_rag.model_identity import canonical_model_name


ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def test_docker_image_pins_and_verifies_both_models_offline() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert EMBEDDING_REVISION in dockerfile
    assert RERANKER_REVISION in dockerfile
    assert "python scripts/cache_models.py --verify-only" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "MODEL_LOCAL_FILES_ONLY=true" in dockerfile


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
    assert compose.count("build: .") == 1


def test_local_runner_does_not_hide_baked_models_with_host_cache() -> None:
    path = ROOT / "scripts/local_docker.py"
    spec = importlib.util.spec_from_file_location("local_docker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    override = module._override_text(bounded=True)

    assert "/app/logs" in override
    assert "/home/appuser/.cache/huggingface" not in override
    assert "HF_HUB_OFFLINE" in override


def test_historical_model_names_resolve_to_baked_repositories() -> None:
    assert canonical_model_name("all-MiniLM-L6-v2") == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert canonical_model_name("cross-encoder/ms-marco-MiniLM-L-6-v2") == (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )
