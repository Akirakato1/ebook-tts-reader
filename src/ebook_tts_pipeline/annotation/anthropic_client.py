from __future__ import annotations

import json
import re
from typing import Dict, Optional, Protocol


class JsonCompletionClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict:
        ...


class AnnotationModelOutputError(ValueError):
    def __init__(
        self,
        message: str,
        raw_text: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.source = source


def parse_json_response_text(text: str, source: str = "model response") -> Dict:
    stripped = text.strip()
    if not stripped:
        raise AnnotationModelOutputError(
            f"{source} was empty; expected a JSON object.",
            raw_text=text,
            source=source,
        )
    candidates = _json_response_candidates(stripped)
    last_error: json.JSONDecodeError | None = None
    non_object_text: str | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict):
            return payload
        non_object_text = candidate

    if non_object_text is not None:
        raise AnnotationModelOutputError(
            f"{source} must be a JSON object.",
            raw_text=non_object_text,
            source=source,
        )

    error_text = str(last_error) if last_error is not None else "no JSON object found"
    raise AnnotationModelOutputError(
        f"{source} was not valid JSON: {error_text}. Preview: {_preview(stripped)}",
        raw_text=stripped,
        source=source,
    ) from last_error


def _json_response_candidates(stripped: str) -> list[str]:
    candidates = [stripped]
    full_fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if full_fence:
        return [full_fence.group(1).strip()]
    for fence in re.finditer(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL):
        candidate = fence.group(1).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


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
