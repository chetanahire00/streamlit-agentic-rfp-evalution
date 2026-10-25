"""Google Gemini adapter (google-genai SDK, JSON mode with a response schema)."""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Calls Gemini with ``response_mime_type="application/json"`` and a JSON schema."""

    provider_id = "gemini"

    def _build_client(self) -> Any:
        from google import genai
        from google.genai import types

        return genai.Client(
            api_key=self.config.api_key,
            http_options=types.HttpOptions(timeout=int(self.config.timeout_s * 1000)),
        )

    def _complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.config.max_tokens,
            response_mime_type="application/json",
            response_json_schema=schema,
            **self._temperature_kwargs(),
        )
        response = self._client.models.generate_content(
            model=self.config.model, contents=user, config=config
        )
        return response.text or ""
