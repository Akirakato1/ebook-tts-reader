from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class PipelineConfig:
    book_root: str
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_temperature: float = 0.1
    anthropic_max_tokens: int = 8192
    annotation_repair_retries: int = 1
    qwen_model_choice: str = "1.7B"
    qwen_model_root: str = "models/qwen-tts"
    qwen_device: str = "auto"
    qwen_precision: str = "bf16"
    qwen_attention: str = "auto"
    max_llm_window_chars: int = 48000
    max_tts_window_chars: int = 6000
    max_tts_roles: int = 8
    pause_between_sentences_ms: int = 250

    @classmethod
    def from_env(cls, book_root: str) -> "PipelineConfig":
        return cls(
            book_root=book_root,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            anthropic_model=os.environ.get(
                "EBOOK_TTS_ANTHROPIC_MODEL",
                DEFAULT_ANTHROPIC_MODEL,
            ),
            qwen_model_choice=os.environ.get("EBOOK_TTS_QWEN_MODEL", "1.7B"),
            qwen_model_root=os.environ.get("EBOOK_TTS_QWEN_MODEL_ROOT", "models/qwen-tts"),
            qwen_device=os.environ.get("EBOOK_TTS_QWEN_DEVICE", "auto"),
            qwen_precision=os.environ.get("EBOOK_TTS_QWEN_PRECISION", "bf16"),
            qwen_attention=os.environ.get("EBOOK_TTS_QWEN_ATTENTION", "auto"),
        )

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for annotation.")
        return self.anthropic_api_key
