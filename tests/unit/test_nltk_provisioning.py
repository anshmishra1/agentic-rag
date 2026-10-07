import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/cache_nltk.py"
    spec = importlib.util.spec_from_file_location("cache_nltk_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_resources_match_pinecone_text_runtime_dependencies() -> None:
    module = _module()

    assert module.REQUIRED_RESOURCES == {
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }


def test_verification_fails_when_a_required_resource_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    module = _module()

    def find(resource_path, paths):
        if resource_path == "corpora/stopwords":
            raise LookupError("missing")
        return tmp_path / resource_path

    monkeypatch.setattr(module.nltk.data, "find", find)

    with pytest.raises(RuntimeError, match="stopwords"):
        module.verify_resources(tmp_path)
