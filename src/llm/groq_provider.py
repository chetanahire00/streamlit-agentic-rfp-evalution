"""Groq adapter (JSON object mode, available on every Groq model)."""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    """Calls Groq with ``response_format={"type": "json_object"}``.

    Strict ``json_schema`` mode exists only on some Groq models, so JSON object
    mode is used everywhere and the schema is enforced by the prompt plus the
    validation tool.
    """

    provider_id = "groq"

    def _build_client(self) -> Any:
        from groq import Groq

        return Groq(api_key=self.config.api_key, timeout=self.config.timeout_s, max_retries=0)

    def _complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_completion_tokens=self.config.max_tokens,
            **self._temperature_kwargs(),
        )
        return response.choices[0].message.content or ""
