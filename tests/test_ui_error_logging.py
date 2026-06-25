from ebook_tts_pipeline.debug_logging import attach_debug_log_path
from ebook_tts_pipeline.ui.errors import pipeline_error_message, write_readalong_error_event


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


def test_write_readalong_error_event_appends_jsonl_with_traceback(tmp_path):
    log_path = tmp_path / "book" / "logs" / "readalong_web_errors.jsonl"
    exc = RuntimeError("top-up failed")

    written = write_readalong_error_event(
        tmp_path / "book",
        "readalong_top_up_error",
        {"chapter": "chapter_017", "token": "Bearer secret-token"},
        exc=exc,
    )

    assert written == log_path
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "readalong_top_up_error" in lines[0]
    assert "chapter_017" in lines[0]
    assert "Bearer secret-token" not in lines[0]
    assert "[REDACTED]" in lines[0]
    assert "RuntimeError" in lines[0]
    assert "top-up failed" in lines[0]
