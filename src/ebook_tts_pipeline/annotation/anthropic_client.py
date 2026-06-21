from __future__ import annotations

import json
import re
from typing import Dict, Protocol


class JsonCompletionClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict:
        ...


class AnnotationModelOutputError(ValueError):
    pass


def parse_json_response_text(text: str, source: str = "model response") -> Dict:
    stripped = text.strip()
    if not stripped:
        raise AnnotationModelOutputError(f"{source} was empty; expected a JSON object.")
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AnnotationModelOutputError(
            f"{source} was not valid JSON: {exc}. Preview: {_preview(stripped)}"
        ) from exc
    if not isinstance(payload, dict):
        raise AnnotationModelOutputError(f"{source} must be a JSON object.")
    return payload


class AnthropicJsonClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        stop_reason = getattr(message, "stop_reason", "unknown")
        return parse_json_response_text(text, source=f"Anthropic response (stop_reason={stop_reason})")


def _preview(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."
