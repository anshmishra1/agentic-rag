"""Provider-agnostic LLM layer with two configurable tiers and fallback."""

from __future__ import annotations
import time
import logging
from email.utils import parsedate_to_datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_rag.config import settings
from agentic_rag.core.llm_accounting import record_llm_event

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


_MODELS: dict[str, dict[str, str]] = {
    "primary": {
        "groq": settings.groq_model,
        "cerebras": settings.cerebras_model,
        "nvidia": settings.nvidia_model,
        "openrouter": settings.openrouter_model,
        "bedrock": settings.bedrock_model,
    },
    "fast": {
        "groq": settings.groq_fast_model,
        "cerebras": settings.cerebras_fast_model,
        "nvidia": settings.nvidia_fast_model,
        "openrouter": settings.openrouter_fast_model,
        "bedrock": settings.bedrock_fast_model,
    },
}


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit-shaped errors across heterogeneous provider SDKs.

    Groq, Cerebras, OpenRouter (via ChatOpenAI), and Bedrock each raise
    different exception types under the hood, so we check for the common
    signatures (HTTP 429, or the phrase in the message) rather than
    importing every provider's specific exception class.

    Module-level, not a method - this is a pure function with no need for
    instance state, and stays independently testable/importable that way.
    """
    status_code = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status_code in (401, 402, 403):
        return False
    if status_code == 429:
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("rate limit", "too many requests", "429", "quota")
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read a provider Retry-After header without importing its SDK."""

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if not headers:
        return None

    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                return None
            return max(0.0, retry_at.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


class ProviderUnavailableError(RuntimeError):
    """Every configured provider failed to answer a request."""


class ProviderChain:
    """Configurable LLM provider chain with automatic fallback."""

    def __init__(self, tier: str = "primary", *, clock=None) -> None:
        if tier not in _MODELS:
            raise ValueError(
                f"Invalid tier '{tier}'. Expected one of {list(_MODELS.keys())}."
            )

        self.tier = tier
        self._clock = clock or time.monotonic
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._last_provider: str | None = None
        self._providers = self._build_providers()

        if not self._providers:
            raise RuntimeError(
                f"No LLM provider initialized for tier '{self.tier}'. "
                "Check .env settings for API keys and PROVIDER_ORDER."
            )

        logger.info(
            "LLM provider chain initialized (%s tier): %s",
            self.tier,
            [name for name, _ in self._providers],
        )

    @property
    def timeout(self) -> float:
        return (
            settings.fast_llm_timeout
            if self.tier == "fast"
            else settings.primary_llm_timeout
        )

    def _build_providers(self) -> list[tuple[str, BaseChatModel]]:
        available = {
            "groq": self._build_groq,
            "cerebras": self._build_cerebras,
            "nvidia": self._build_nvidia,
            "openrouter": self._build_openrouter,
            "bedrock": self._build_bedrock,
        }

        configured_order = [
            provider.strip().lower()
            for provider in settings.provider_order.split(",")
            if provider.strip()
        ]

        providers: list[tuple[str, BaseChatModel]] = []

        for provider_name in configured_order:
            builder = available.get(provider_name)
            if builder is None:
                logger.warning(
                    "Unknown provider '%s' in PROVIDER_ORDER. Skipping.",
                    provider_name,
                )
                continue

            try:
                provider = builder()
                if provider is not None:
                    providers.append((provider_name, provider))
            except Exception as exc:
                logger.warning(
                    "Could not initialize provider '%s' (%s tier): %s",
                    provider_name,
                    self.tier,
                    exc,
                )

        return providers

    def _build_groq(self) -> BaseChatModel | None:
        if not settings.groq_api_key:
            logger.info("Groq skipped: GROQ_API_KEY not configured.")
            return None

        from langchain_groq import ChatGroq

        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=_MODELS[self.tier]["groq"],
            timeout=self.timeout,
            max_retries=0,
        )

    def _build_cerebras(self) -> BaseChatModel | None:
        if not settings.cerebras_api_key:
            logger.info("Cerebras skipped: CEREBRAS_API_KEY not configured.")
            return None

        from langchain_cerebras import ChatCerebras

        return ChatCerebras(
            api_key=settings.cerebras_api_key,
            model=_MODELS[self.tier]["cerebras"],
            timeout=self.timeout,
        )

    def _build_nvidia(self) -> BaseChatModel | None:
        if not settings.nvidia_api_key:
            logger.info("NVIDIA skipped: NVIDIA_API_KEY not configured.")
            return None

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            model=_MODELS[self.tier]["nvidia"],
            timeout=self.timeout,
        )

    def _build_openrouter(self) -> BaseChatModel | None:
        if not settings.openrouter_api_key:
            logger.info("OpenRouter skipped: OPENROUTER_API_KEY not configured.")
            return None

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=_MODELS[self.tier]["openrouter"],
            timeout=self.timeout,
        )

    def _build_bedrock(self) -> BaseChatModel | None:
        if not settings.bedrock_enabled or not settings.aws_bedrock_region:
            logger.info("Bedrock skipped: disabled or AWS region not configured.")
            return None

        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            region_name=settings.aws_bedrock_region,
            model=_MODELS[self.tier]["bedrock"],
        )

    def invoke(
        self,
        prompt: Any,
        *,
        max_tokens: int | None = None,
        groq_reasoning_effort: str | None = None,
        groq_response_format: dict | None = None,
        require_nonempty_content: bool = False,
        stage: str = "unspecified",
    ):
        """Invoke each distinct available provider/model at most once.

        A rate-limited provider/model enters a bounded cooldown. The current
        call then falls through to a distinct provider rather than repeating
        an identical request and amplifying quota use.
        """
        last_error: Exception | None = None
        attempted_requests: set[tuple[Any, ...]] = set()
        output_limit = max_tokens if max_tokens is not None else (
            settings.fast_llm_max_tokens
            if self.tier == "fast"
            else settings.primary_llm_max_tokens
        )

        for index, (name, llm) in enumerate(self._providers, start=1):
            model = _MODELS[self.tier][name]
            cooldown_key = (name, model)
            now = self._clock()
            cooldown_until = self._cooldowns.get(cooldown_key, 0.0)
            if cooldown_until > now:
                record_llm_event(
                    stage=stage, event="cooldown_skip", provider=name, tier=self.tier
                )
                logger.info(
                    "[LLM:%s] %s skipped for %.1fs remaining cooldown",
                    self.tier, name, cooldown_until - now,
                )
                continue
            self._cooldowns.pop(cooldown_key, None)

            request_key = (
                name, model, output_limit, groq_reasoning_effort,
                repr(groq_response_format), require_nonempty_content,
            )
            if request_key in attempted_requests:
                record_llm_event(
                    stage=stage, event="duplicate_skip", provider=name, tier=self.tier
                )
                continue
            attempted_requests.add(request_key)
            started = time.perf_counter()
            record_llm_event(
                stage=stage, event="attempt", provider=name, tier=self.tier
            )

            try:
                logger.info(
                    "[LLM:%s] Attempt %d/%d -> %s (stage=%s)",
                    self.tier, index, len(self._providers), name, stage,
                )
                request_options = {"max_tokens": output_limit}
                if (
                    groq_reasoning_effort is not None
                    and name == "groq"
                    and self.tier == "fast"
                    and _MODELS["fast"]["groq"] in {
                        "openai/gpt-oss-20b", "openai/gpt-oss-120b",
                    }
                ):
                    request_options["reasoning_effort"] = groq_reasoning_effort
                if groq_response_format is not None and name == "groq":
                    request_options["response_format"] = groq_response_format

                response = llm.bind(**request_options).invoke(prompt)
                if require_nonempty_content and (
                    not isinstance(response.content, str) or not response.content.strip()
                ):
                    raise ValueError(
                        "Provider returned empty textual content for a task that "
                        "requires a decision."
                    )

                elapsed = time.perf_counter() - started
                self._last_provider = name
                record_llm_event(
                    stage=stage, event="success", provider=name, tier=self.tier
                )
                logger.info(
                    "[LLM:%s] %s -> SUCCESS in %.2fs", self.tier, name, elapsed
                )
                return response

            except Exception as exc:
                elapsed = time.perf_counter() - started
                last_error = exc
                rate_limited = _is_rate_limit_error(exc)
                record_llm_event(
                    stage=stage, event="failure", provider=name, tier=self.tier
                )
                logger.warning(
                    "[LLM:%s] %s -> FAILED after %.2fs (rate_limited=%s): %s",
                    self.tier, name, elapsed, rate_limited, exc,
                )
                if rate_limited:
                    retry_after = _retry_after_seconds(exc)
                    cooldown = retry_after if retry_after is not None else (
                        settings.llm_rate_limit_cooldown_seconds
                    )
                    cooldown = min(
                        cooldown, settings.llm_rate_limit_max_cooldown_seconds
                    )
                    self._cooldowns[cooldown_key] = self._clock() + cooldown
                    record_llm_event(
                        stage=stage, event="rate_limit", provider=name, tier=self.tier
                    )
                    logger.info(
                        "[LLM:%s] %s cooling down for %.1fs",
                        self.tier, name, cooldown,
                    )

        raise ProviderUnavailableError(
            f"All configured LLM providers failed for {self.tier} tier. "
            f"Last error: {last_error}"
        )

    def as_langchain_llm(self) -> BaseChatModel:
        return self._providers[0][1]

    @property
    def primary_provider(self) -> str:
        return self._providers[0][0]

    @property
    def last_provider(self) -> str | None:
        return self._last_provider

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._providers]


provider_chain = ProviderChain(tier="primary")
fast_provider_chain = ProviderChain(tier="fast")
