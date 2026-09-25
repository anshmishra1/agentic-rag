"""Provider fallback must respect a caller's output budget."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_provider_with_fake_model(monkeypatch):
    models = []

    class FakeChatModel:
        def __init__(self, **kwargs):
            self.bound_options = None
            self.calls = 0
            models.append(self)

        def bind(self, **kwargs):
            self.bound_options = kwargs
            return self

        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content="Overview")

    fake_config = ModuleType("agentic_rag.config")
    fake_config.settings = SimpleNamespace(
        groq_model="test-model",
        cerebras_model="test-model",
        nvidia_model="test-model",
        openrouter_model="test-model",
        bedrock_model="test-model",
        groq_fast_model="test-model",
        cerebras_fast_model="test-model",
        nvidia_fast_model="test-model",
        openrouter_fast_model="test-model",
        bedrock_fast_model="test-model",
        provider_order="groq",
        groq_api_key="fake-key",
        primary_llm_timeout=1,
        fast_llm_timeout=1,
        primary_llm_max_tokens=4096,
        fast_llm_max_tokens=1024,
        llm_max_retries_per_provider=3,
    )
    fake_groq = ModuleType("langchain_groq")
    fake_groq.ChatGroq = FakeChatModel
    monkeypatch.setitem(sys.modules, "agentic_rag.config", fake_config)
    monkeypatch.setitem(sys.modules, "langchain_groq", fake_groq)

    path = Path(__file__).resolve().parents[2] / "src/agentic_rag/llm/provider.py"
    spec = importlib.util.spec_from_file_location("provider_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, models


def test_caller_can_limit_overview_output(monkeypatch):
    provider, models = _load_provider_with_fake_model(monkeypatch)

    response = provider.provider_chain.invoke("Summarize", max_tokens=512)

    assert response.content == "Overview"
    assert models[0].bound_options == {"max_tokens": 512}
    assert models[0].calls == 1


def test_payment_required_is_not_retried_as_a_rate_limit(monkeypatch):
    provider, _ = _load_provider_with_fake_model(monkeypatch)
    error = RuntimeError("payment required: quota")
    error.status_code = 402

    assert provider._is_rate_limit_error(error) is False


def test_primary_calls_have_a_default_output_limit(monkeypatch):
    provider, models = _load_provider_with_fake_model(monkeypatch)

    provider.provider_chain.invoke("Answer the question")

    assert models[0].bound_options == {"max_tokens": 4096}
