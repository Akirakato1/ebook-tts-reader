from ebook_tts_pipeline.debug_logging import attach_debug_log_path
from ebook_tts_pipeline.ui.errors import pipeline_error_message


def test_pipeline_error_message_uses_existing_debug_log_path(tmp_path):
    log_path = tmp_path / "annotation.json"
    exc = RuntimeError("annotation failed")
    attach_debug_log_path(exc, log_path)

    message = pipeline_error_message(
        exc,
        label="Working on chapter_003...",
        book_root="books/demo",
        log_root=tmp_path / "logs",
    )

    assert "annotation failed" in message
    assert str(log_path) in message
    assert not (tmp_path / "logs").exists()


def test_pipeline_error_message_writes_generic_log_when_no_path_exists(tmp_path):
    exc = RuntimeError("tts failed")

    message = pipeline_error_message(
        exc,
        label="Working on chapter_003...",
        book_root="books/demo",
        log_root=tmp_path / "logs",
    )

    logs = list((tmp_path / "logs").glob("*.json"))
    assert len(logs) == 1
    assert "tts failed" in message
    assert str(logs[0]) in message
    log_text = logs[0].read_text(encoding="utf-8")
    assert "ui_pipeline_error" in log_text
    assert "Working on chapter_003..." in log_text
    assert "books/demo" in log_text
