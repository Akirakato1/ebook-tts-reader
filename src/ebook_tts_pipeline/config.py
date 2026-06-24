from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_VLLM_OMNI_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_VLLM_OMNI_STAGE_CONFIG = str(
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    item = Path(path)
    if item.is_absolute():
        return item.resolve()
    if item.exists():
        return item.resolve()
    return (PROJECT_ROOT / item).resolve()


def resolve_qwen_model_root(model_root: str | Path) -> Path:
    return resolve_project_path(model_root)


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
    qwen_max_new_tokens: int = 2048
    qwen_max_generation_block_chars: int = 0
    qwen_max_generation_blocks_per_call: int = 0
    qwen_cache_clear_interval: int = 8
    qwen_streaming_text_mode: bool = True
    qwen_perf_log_path: Optional[str] = None
    qwen_adaptive_memory_target_gb: Optional[float] = None
    tts_backend: str = "native"
    voice_asset_tts_backend: str = "wsl"
    wsl_distro: str = "Ubuntu-24.04"
    wsl_python: str = "/opt/ebook-tts-venv/bin/python"
    wsl_timeout_seconds: float = 600.0
    read_along_tts_backend: str = "wsl-vllm-omni"
    vllm_omni_model: str = DEFAULT_VLLM_OMNI_MODEL
    vllm_omni_stage_configs_path: str = DEFAULT_VLLM_OMNI_STAGE_CONFIG
    vllm_omni_wsl_python: str = "/opt/ebook-vllm-omni-venv/bin/python"
    global_registry_window_chars: int = 2000000
    max_llm_window_chars: int = 48000
    max_llm_window_sentences: int = 300
    max_tts_window_chars: int = 1100
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
        voice_asset_tts_backend = os.environ.get("EBOOK_TTS_VOICE_ASSET_BACKEND", "wsl")
        if voice_asset_tts_backend in {"wsl-vllm-omni", "vllm-omni"}:
            voice_asset_tts_backend = "wsl"
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
            qwen_max_new_tokens=int(os.environ.get("EBOOK_TTS_QWEN_MAX_NEW_TOKENS", "2048")),
            qwen_max_generation_block_chars=int(
                os.environ.get("EBOOK_TTS_QWEN_MAX_GENERATION_BLOCK_CHARS", "0")
            ),
            qwen_max_generation_blocks_per_call=int(
                os.environ.get("EBOOK_TTS_QWEN_MAX_GENERATION_BLOCKS_PER_CALL", "0")
            ),
            qwen_cache_clear_interval=int(os.environ.get("EBOOK_TTS_QWEN_CACHE_CLEAR_INTERVAL", "8")),
            qwen_streaming_text_mode=_bool_env("EBOOK_TTS_QWEN_STREAMING_TEXT_MODE", True),
            qwen_perf_log_path=os.environ.get("EBOOK_TTS_QWEN_PERF_LOG") or None,
            qwen_adaptive_memory_target_gb=_optional_float_env("EBOOK_TTS_QWEN_ADAPTIVE_TARGET_GB"),
            tts_backend=os.environ.get("EBOOK_TTS_BACKEND", "native"),
            voice_asset_tts_backend=voice_asset_tts_backend,
            wsl_distro=os.environ.get("EBOOK_TTS_WSL_DISTRO", "Ubuntu-24.04"),
            wsl_python=os.environ.get("EBOOK_TTS_WSL_PYTHON", "/opt/ebook-tts-venv/bin/python"),
            wsl_timeout_seconds=float(os.environ.get("EBOOK_TTS_WSL_TIMEOUT_SECONDS", "600")),
            read_along_tts_backend=os.environ.get("EBOOK_TTS_READ_ALONG_BACKEND", "wsl-vllm-omni"),
            vllm_omni_model=os.environ.get("EBOOK_TTS_VLLM_OMNI_MODEL", DEFAULT_VLLM_OMNI_MODEL),
            vllm_omni_stage_configs_path=os.environ.get(
                "EBOOK_TTS_VLLM_OMNI_STAGE_CONFIG",
                DEFAULT_VLLM_OMNI_STAGE_CONFIG,
            ),
            vllm_omni_wsl_python=os.environ.get(
                "EBOOK_TTS_VLLM_OMNI_WSL_PYTHON",
                "/opt/ebook-vllm-omni-venv/bin/python",
            ),
            global_registry_window_chars=int(
                os.environ.get("EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS", "2000000")
            ),
            max_tts_window_chars=int(os.environ.get("EBOOK_TTS_MAX_TTS_WINDOW_CHARS", "1100")),
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


def _optional_float_env(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
