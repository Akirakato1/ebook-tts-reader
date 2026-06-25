from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ebook_tts_pipeline.debug_logging import FailureLogger, _sanitize, attach_debug_log_path


READALONG_ERROR_LOG = "readalong_web_errors.jsonl"
_READALONG_LOG_LOCK = threading.Lock()


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


def write_readalong_error_event(
    book_root: Union[str, Path],
    event_type: str,
    details: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
) -> Path:
    path = Path(book_root) / "logs" / READALONG_ERROR_LOG
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": str(event_type),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "details": dict(details or {}),
    }
    if exc is not None:
        payload["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }

    with _READALONG_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_sanitize(payload), sort_keys=True))
            fh.write("\n")

    if exc is not None:
        attach_debug_log_path(exc, path)
    return path
