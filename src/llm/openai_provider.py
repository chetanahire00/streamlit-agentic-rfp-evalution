"""OpenAI adapter (Chat Completions with strict JSON-schema structured output)."""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """Calls OpenAI with ``response_format={"type": "json_schema", strict: true}``."""

    provider_id = "openai"

    def _build_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=self.config.api_key, timeout=self.config.timeout_s, max_retries=0)

    def _complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "supplier_evaluation", "schema": schema, "strict": True},
            },
            max_completion_tokens=self.config.max_tokens,
            **self._temperature_kwargs(),
        )
        return response.choices[0].message.content or ""
