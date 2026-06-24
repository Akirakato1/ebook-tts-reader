from ebook_tts_pipeline.config import PipelineConfig


def test_voice_asset_backend_defaults_to_lightweight_backend(monkeypatch):
    monkeypatch.delenv("EBOOK_TTS_VOICE_ASSET_BACKEND", raising=False)
    monkeypatch.setenv("EBOOK_TTS_BACKEND", "wsl-vllm-omni")

    config = PipelineConfig.from_env("book")

    assert config.voice_asset_tts_backend == "wsl"
    assert config.voice_asset_tts_backend != config.read_along_tts_backend


def test_voice_asset_backend_defaults_to_wsl_without_global_backend(monkeypatch):
    monkeypatch.delenv("EBOOK_TTS_VOICE_ASSET_BACKEND", raising=False)
    monkeypatch.delenv("EBOOK_TTS_BACKEND", raising=False)

    config = PipelineConfig.from_env("book")

    assert config.voice_asset_tts_backend == "wsl"


def test_voice_asset_backend_can_be_overridden(monkeypatch):
    monkeypatch.setenv("EBOOK_TTS_VOICE_ASSET_BACKEND", "native")

    config = PipelineConfig.from_env("book")

    assert config.voice_asset_tts_backend == "native"


def test_default_config_is_ui_friendly_and_overridable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("EBOOK_TTS_ANTHROPIC_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("EBOOK_TTS_QWEN_MODEL_ROOT", "models/qwen-tts")
    monkeypatch.setenv("EBOOK_TTS_QWEN_MAX_NEW_TOKENS", "1536")
    monkeypatch.setenv("EBOOK_TTS_QWEN_MAX_GENERATION_BLOCK_CHARS", "512")
    monkeypatch.setenv("EBOOK_TTS_QWEN_MAX_GENERATION_BLOCKS_PER_CALL", "3")
    monkeypatch.setenv("EBOOK_TTS_QWEN_CACHE_CLEAR_INTERVAL", "16")
    monkeypatch.setenv("EBOOK_TTS_QWEN_STREAMING_TEXT_MODE", "0")
    monkeypatch.setenv("EBOOK_TTS_QWEN_PERF_LOG", "logs/qwen_perf.jsonl")
    monkeypatch.setenv("EBOOK_TTS_QWEN_ADAPTIVE_TARGET_GB", "13")
    monkeypatch.setenv("EBOOK_TTS_BACKEND", "wsl")
    monkeypatch.setenv("EBOOK_TTS_WSL_DISTRO", "Ubuntu-24.04")
    monkeypatch.setenv("EBOOK_TTS_WSL_PYTHON", "/opt/ebook-tts-venv/bin/python")
    monkeypatch.setenv("EBOOK_TTS_WSL_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("EBOOK_TTS_READ_ALONG_BACKEND", "wsl-vllm-omni")
    monkeypatch.setenv("EBOOK_TTS_VLLM_OMNI_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    monkeypatch.setenv("EBOOK_TTS_VLLM_OMNI_STAGE_CONFIG", "scripts/custom-vllm.yaml")
    monkeypatch.setenv("EBOOK_TTS_VLLM_OMNI_WSL_PYTHON", "/opt/ebook-vllm-omni-venv/bin/python")
    monkeypatch.setenv("EBOOK_TTS_DEBUG_LOG_ROOT", "logs/debug")
    monkeypatch.setenv("EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS", "135000")
    monkeypatch.setenv("EBOOK_TTS_MAX_TTS_WINDOW_CHARS", "3000")

    config = PipelineConfig.from_env(book_root="books/demo")

    assert config.book_root == "books/demo"
    assert config.anthropic_api_key == "test-key"
    assert config.anthropic_model == "claude-haiku-4-5"
    assert config.qwen_model_root == "models/qwen-tts"
    assert config.qwen_model_choice == "1.7B"
    assert config.qwen_max_new_tokens == 1536
    assert config.qwen_max_generation_block_chars == 512
    assert config.qwen_max_generation_blocks_per_call == 3
    assert config.qwen_cache_clear_interval == 16
    assert config.qwen_streaming_text_mode is False
    assert config.qwen_perf_log_path == "logs/qwen_perf.jsonl"
    assert config.qwen_adaptive_memory_target_gb == 13
    assert config.tts_backend == "wsl"
    assert config.wsl_distro == "Ubuntu-24.04"
    assert config.wsl_python == "/opt/ebook-tts-venv/bin/python"
    assert config.wsl_timeout_seconds == 600.0
    assert config.read_along_tts_backend == "wsl-vllm-omni"
    assert config.vllm_omni_model == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert config.vllm_omni_stage_configs_path == "scripts/custom-vllm.yaml"
    assert config.vllm_omni_wsl_python == "/opt/ebook-vllm-omni-venv/bin/python"
    assert config.max_tts_window_chars == 3000
    assert config.max_tts_roles == 8
    assert config.debug_log_root == "logs/debug"
    assert config.global_registry_window_chars == 135000


def test_default_anthropic_model_is_sonnet(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("EBOOK_TTS_ANTHROPIC_MODEL", raising=False)

    config = PipelineConfig.from_env(book_root="books/demo", user_env_lookup=lambda name: None)

    assert config.anthropic_model == "claude-sonnet-4-6"


def test_config_falls_back_to_user_env_lookup_for_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = PipelineConfig.from_env(
        book_root="books/demo",
        user_env_lookup=lambda name: "user-key" if name == "ANTHROPIC_API_KEY" else None,
    )

    assert config.anthropic_api_key == "user-key"
    assert config.global_registry_window_chars == 2000000
    assert config.max_tts_window_chars == 1100
    assert config.qwen_max_new_tokens == 2048
    assert config.qwen_max_generation_block_chars == 0
    assert config.qwen_max_generation_blocks_per_call == 0
    assert config.qwen_cache_clear_interval == 8
    assert config.qwen_streaming_text_mode is True
    assert config.qwen_perf_log_path is None
    assert config.qwen_adaptive_memory_target_gb is None
    assert config.read_along_tts_backend == "wsl-vllm-omni"
    assert config.vllm_omni_model == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert config.vllm_omni_stage_configs_path.replace("\\", "/").endswith(
        "scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml"
    )
    assert config.vllm_omni_wsl_python == "/opt/ebook-vllm-omni-venv/bin/python"
    assert not hasattr(config, "qwen_batch_size")
