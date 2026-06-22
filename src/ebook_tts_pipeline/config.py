from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


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
    qwen_batch_size: int = 8
    qwen_max_block_chars: int = 600
    global_registry_window_chars: int = 2000000
    max_llm_window_chars: int = 48000
    max_llm_window_sentences: int = 300
    max_tts_window_chars: int = 6000
    max_tts_roles: int = 8
    pause_between_sentences_ms: int = 250
    intra_sentence_pause_ms: int = 50
    tts_speed: float = 1.0
    debug_log_root: str = "logs/annotation_failures"

    @classmethod
    def from_env(
        cls,
        book_root: str,
        user_env_lookup: Optional[Callable[[str], Optional[str]]] = None,
    ) -> "PipelineConfig":
        lookup = user_env_lookup or _lookup_user_env
        return cls(
            book_root=book_root,
            anthropic_api_key=_env("ANTHROPIC_API_KEY", user_env_lookup=lookup),
            anthropic_model=os.environ.get(
                "EBOOK_TTS_ANTHROPIC_MODEL",
                DEFAULT_ANTHROPIC_MODEL,
            ),
            qwen_model_choice=os.environ.get("EBOOK_TTS_QWEN_MODEL", "1.7B"),
            qwen_model_root=os.environ.get("EBOOK_TTS_QWEN_MODEL_ROOT", "models/qwen-tts"),
            qwen_device=os.environ.get("EBOOK_TTS_QWEN_DEVICE", "auto"),
            qwen_precision=os.environ.get("EBOOK_TTS_QWEN_PRECISION", "bf16"),
            qwen_attention=os.environ.get("EBOOK_TTS_QWEN_ATTENTION", "auto"),
            qwen_batch_size=int(os.environ.get("EBOOK_TTS_QWEN_BATCH_SIZE", "8")),
            qwen_max_block_chars=int(os.environ.get("EBOOK_TTS_QWEN_MAX_BLOCK_CHARS", "600")),
            global_registry_window_chars=int(
                os.environ.get("EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS", "2000000")
            ),
            max_llm_window_sentences=int(os.environ.get("EBOOK_TTS_MAX_LLM_WINDOW_SENTENCES", "300")),
            tts_speed=float(os.environ.get("EBOOK_TTS_SPEED", "1.0")),
            pause_between_sentences_ms=int(os.environ.get("EBOOK_TTS_PAUSE_BETWEEN_SENTENCES_MS", "250")),
            intra_sentence_pause_ms=int(os.environ.get("EBOOK_TTS_INTRA_SENTENCE_PAUSE_MS", "50")),
            debug_log_root=os.environ.get("EBOOK_TTS_DEBUG_LOG_ROOT", "logs/annotation_failures"),
        )

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for annotation.")
        return self.anthropic_api_key


def _env(name: str, user_env_lookup: Callable[[str], Optional[str]]) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    return user_env_lookup(name)


def _lookup_user_env(name: str) -> Optional[str]:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value) if value else None
    except OSError:
        return None
