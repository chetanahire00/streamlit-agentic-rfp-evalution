"""Anthropic (Claude) adapter using structured outputs (``output_config.format``).

Sampling parameters such as ``temperature`` were removed from the Messages API
(2026-10-21), so none are sent; reproducibility comes from the LLM cache.
Models that support it get server-side refusal fallbacks, so a safety-classifier
refusal is retried on a suitable model instead of failing the supplier.
"""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider
from src.utils.errors import ErrorCode, LLMError

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-5"}


class AnthropicProvider(LLMProvider):
    """Calls the Claude Messages API with a JSON-schema output format."""

    provider_id = "anthropic"

    def _build_client(self) -> Any:
        from anthropic import Anthropic

        return Anthropic(api_key=self.config.api_key, timeout=self.config.timeout_s, max_retries=0)

    def _complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if self.config.model in FALLBACK_MODELS:
            response = self._client.beta.messages.create(
                **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            response = self._client.messages.create(**kwargs)
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError(ErrorCode.LLM_BAD_REQUEST,
                           "Claude declined to evaluate this document. Try another model.")
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""
