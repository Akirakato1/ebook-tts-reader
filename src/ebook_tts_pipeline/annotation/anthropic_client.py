from __future__ import annotations

import json
from typing import Dict, Protocol


class JsonCompletionClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict:
        ...


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
        return json.loads(text)
