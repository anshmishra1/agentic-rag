"""Provider fallback must respect a caller's output budget."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_provider_with_fake_model(monkeypatch, *, response_content="Overview"):
    models = []

    class FakeChatModel:
        def __init__(self, **kwargs):
            self.init_options = kwargs
            self.bound_options = None
            self.calls = 0
            models.append(self)

        def bind(self, **kwargs):
            self.bound_options = kwargs
            return self

        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content=response_content)

    fake_config = ModuleType("agentic_rag.config")
    fake_config.settings = SimpleNamespace(
        groq_model="test-model",
        cerebras_model="test-model",
        nvidia_model="test-model",
        openrouter_model="test-model",
        bedrock_model="test-model",
        groq_fast_model="openai/gpt-oss-20b",
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


def test_groq_sdk_retries_are_disabled_so_the_app_owns_the_budget(monkeypatch):
    _, models = _load_provider_with_fake_model(monkeypatch)

    assert models[0].init_options["max_retries"] == 0
    assert models[1].init_options["max_retries"] == 0


def test_required_empty_content_is_a_provider_failure(monkeypatch):
    provider, _ = _load_provider_with_fake_model(
        monkeypatch,
        response_content="",
    )

    with pytest.raises(provider.ProviderUnavailableError):
        provider.fast_provider_chain.invoke(
            "Grade retrieval",
            require_nonempty_content=True,
        )


def test_payment_required_is_not_retried_as_a_rate_limit(monkeypatch):
    provider, _ = _load_provider_with_fake_model(monkeypatch)
    error = RuntimeError("payment required: quota")
    error.status_code = 402

    assert provider._is_rate_limit_error(error) is False


def test_primary_calls_have_a_default_output_limit(monkeypatch):
    provider, models = _load_provider_with_fake_model(monkeypatch)

    provider.provider_chain.invoke("Answer the question")

    assert models[0].bound_options == {"max_tokens": 4096}


def test_only_fast_groq_gpt_oss_verifier_uses_low_reasoning_effort(monkeypatch):
    provider, models = _load_provider_with_fake_model(monkeypatch)

    provider.fast_provider_chain.invoke(
        "Check citations", max_tokens=1024, groq_reasoning_effort="low"
    )
    assert models[1].bound_options == {
        "max_tokens": 1024,
        "reasoning_effort": "low",
    }

    provider._MODELS["fast"]["groq"] = "other-model"
    provider.fast_provider_chain.invoke(
        "Check citations", max_tokens=1024, groq_reasoning_effort="low"
    )
    assert models[1].bound_options == {"max_tokens": 1024}

    provider.provider_chain.invoke(
        "Answer", groq_reasoning_effort="low"
    )
    assert models[0].bound_options == {"max_tokens": 4096}


def test_fast_groq_router_receives_structured_output_schema(monkeypatch):
    provider, models = _load_provider_with_fake_model(monkeypatch)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "query_plan", "schema": {"type": "object"}},
    }

    provider.fast_provider_chain.invoke(
        "Route this query",
        max_tokens=256,
        groq_reasoning_effort="low",
        groq_response_format=response_format,
    )

    assert models[1].bound_options == {
        "max_tokens": 256,
        "reasoning_effort": "low",
        "response_format": response_format,
    }


def test_installed_chatgroq_forwards_verifier_options_without_network() -> None:
    from langchain_groq import ChatGroq

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": '{"verdict":"grounded"}'},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

    model = ChatGroq(groq_api_key="offline-test-key", model_name="openai/gpt-oss-20b")
    model.client = FakeCompletions()

    response = model.bind(max_tokens=1024, reasoning_effort="low").invoke("Verify")

    assert response.content == '{"verdict":"grounded"}'
    assert captured["max_tokens"] == 1024
    assert captured["reasoning_effort"] == "low"
