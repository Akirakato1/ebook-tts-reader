from ebook_tts_pipeline.config import PipelineConfig


def test_default_config_is_ui_friendly_and_overridable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("EBOOK_TTS_ANTHROPIC_MODEL", "claude-haiku-4-5")

    config = PipelineConfig.from_env(book_root="books/demo")

    assert config.book_root == "books/demo"
    assert config.anthropic_api_key == "test-key"
    assert config.anthropic_model == "claude-haiku-4-5"
    assert config.qwen_model_choice == "1.7B"
    assert config.max_tts_roles == 8
