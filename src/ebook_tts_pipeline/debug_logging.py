from __future__ import annotations

import re
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ebook_tts_pipeline.json_io import write_json_atomic


JsonLike = Union[Dict[str, Any], list, str, int, float, bool, None]

_REDACTIONS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", flags=re.IGNORECASE),
    re.compile(r"sk-ant-[A-Za-z0-9._\-]+", flags=re.IGNORECASE),
]


class FailureLogger:
    def __init__(
        self,
        log_root: Union[str, Path] = Path("logs") / "annotation_failures",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_root = Path(log_root)
        self.context = dict(context or {})

    def with_context(self, **context: Any) -> "FailureLogger":
        merged = dict(self.context)
        merged.update(context)
        return FailureLogger(self.log_root, context=merged)

    def write_failure(
        self,
        event_type: str,
        details: Dict[str, Any],
        exc: Optional[BaseException] = None,
    ) -> Path:
        payload: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "context": self.context,
            "details": details,
        }
        if exc is not None:
            payload["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }

        path = self.log_root / f"{_timestamp_slug()}_{_slug(event_type)}_{uuid.uuid4().hex[:8]}.json"
        write_json_atomic(path, _sanitize(payload))
        if exc is not None:
            attach_debug_log_path(exc, path)
        return path


def attach_debug_log_path(exc: BaseException, path: Union[str, Path]) -> None:
    try:
        setattr(exc, "debug_log_path", str(path))
    except Exception:
        return


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "failure"


def _sanitize(value: Any) -> JsonLike:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, Path)):
        return _sanitize_text(str(value))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(repr(value))


def _sanitize_text(value: str) -> str:
    sanitized = value
    for pattern in _REDACTIONS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized
