from ebook_tts_pipeline.debug_logging import FailureLogger


def test_failure_logger_writes_sanitized_json(tmp_path):
    logger = FailureLogger(tmp_path / "logs", context={"book_root": "books/demo"})

    path = logger.write_failure(
        "annotation_model_output_error",
        {"message": "Bearer secret-token sk-ant-api03-abc", "chapter": "chapter_001"},
    )

    text = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path / "logs"
    assert "annotation_model_output_error" in path.name
    assert "books/demo" in text
    assert "secret-token" not in text
    assert "sk-ant-api03-abc" not in text


def test_failure_logger_attaches_debug_log_path_to_exception(tmp_path):
    logger = FailureLogger(tmp_path / "logs")
    exc = RuntimeError("boom")

    path = logger.write_failure("ui_pipeline_error", {"action": "chapter"}, exc=exc)

    assert exc.debug_log_path == str(path)
