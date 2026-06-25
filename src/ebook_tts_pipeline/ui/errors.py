from __future__ import annotations

from pathlib import Path
from typing import Union

from ebook_tts_pipeline.debug_logging import FailureLogger


def pipeline_error_message(
    exc: BaseException,
    label: str,
    book_root: str,
    log_root: Union[str, Path] = Path("logs") / "annotation_failures",
) -> str:
    log_path = getattr(exc, "debug_log_path", None)
    if not log_path:
        log_path = FailureLogger(
            log_root,
            context={"book_root": book_root, "ui_action": label},
        ).write_failure(
            "ui_pipeline_error",
            {"label": label, "book_root": book_root},
            exc=exc,
        )
    return f"{exc}\n\nDebug log: {log_path}"
