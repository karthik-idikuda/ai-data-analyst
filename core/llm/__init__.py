"""LLM provider factory."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from ..observability import get_logger
from .base import (
    LLMProvider,
    LLMResponse,
    Message,
    NullProvider,
    ToolCallRequest,
    ToolSpec,
)
from .fallback import FallbackProvider
from .gemini import GeminiProvider
from .openai_compat import GroqProvider, OpenAICompatProvider, OpenAIProvider

log = get_logger(__name__)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "NullProvider",
    "ToolCallRequest",
    "ToolSpec",
    "FallbackProvider",
    "GeminiProvider",
    "GroqProvider",
    "OpenAIProvider",
    "OpenAICompatProvider",
    "get_provider",
    "reset_provider_cache",
]


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    """Build the configured provider once per process.

    Returns :class:`NullProvider` when nothing is configured, so the app boots and
    every non-LLM feature keeps working without credentials.
    """
    settings = get_settings()
    if not settings.llm_configured:
        log.warning("llm.not_configured", provider=settings.llm_provider)
        return NullProvider()

    model = settings.default_model
    if settings.llm_provider == "gemini":
        provider: LLMProvider = GeminiProvider(settings.llm_api_key, default_model=model)
    elif settings.llm_provider == "groq":
        provider = GroqProvider(settings.llm_api_key, default_model=model)
    elif settings.llm_provider == "openai":
        provider = OpenAIProvider(settings.llm_api_key, default_model=model)
    else:  # pragma: no cover - guarded by the Literal type
        return NullProvider()

    chain = settings.model_chain
    if len(chain) > 1:
        provider = FallbackProvider(provider, chain)
        log.info("llm.configured", provider=provider.name, model=chain[0], fallbacks=chain[1:])
    else:
        log.info("llm.configured", provider=provider.name, model=model)
    return provider


def reset_provider_cache() -> None:
    """Drop the cached provider (used by tests and after a settings change)."""
    get_provider.cache_clear()
