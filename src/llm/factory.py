"""Create an :class:`LLMProvider` from an :class:`LLMConfig`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import LLMConfig, LLMProvider
from src.llm.demo_provider import DemoProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.openai_provider import OpenAIProvider
from src.utils.errors import ErrorCode, LLMError

PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "demo": DemoProvider,
}

ProviderFactory = Callable[[LLMConfig], LLMProvider]


def create_provider(config: LLMConfig, **kwargs: Any) -> LLMProvider:
    """Instantiate the adapter for ``config.provider``.

    Raises:
        LLMError: ``LLM_CONFIG`` for an unknown provider, ``LLM_AUTH`` for a missing key.
    """
    cls = PROVIDERS.get(config.provider)
    if cls is None:
        raise LLMError(ErrorCode.LLM_CONFIG, f"Unknown AI provider '{config.provider}'.")
    return cls(config, **kwargs)
