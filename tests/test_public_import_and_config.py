from ebook_tts_pipeline.config import PipelineConfig


def test_default_config_is_ui_friendly_and_overridable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("EBOOK_TTS_ANTHROPIC_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("EBOOK_TTS_QWEN_MODEL_ROOT", "models/qwen-tts")
    monkeypatch.setenv("EBOOK_TTS_DEBUG_LOG_ROOT", "logs/debug")
    monkeypatch.setenv("EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS", "135000")

    config = PipelineConfig.from_env(book_root="books/demo")

    assert config.book_root == "books/demo"
    assert config.anthropic_api_key == "test-key"
    assert config.anthropic_model == "claude-haiku-4-5"
    assert config.qwen_model_root == "models/qwen-tts"
    assert config.qwen_model_choice == "1.7B"
    assert config.max_tts_window_chars == 1100
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
    assert not hasattr(config, "qwen_batch_size")
