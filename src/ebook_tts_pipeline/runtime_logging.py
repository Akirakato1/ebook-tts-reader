from __future__ import annotations

from typing import Any


def log_runtime_step(event: str, **details: Any) -> None:
    parts = [f"[ebook-tts] {event}"]
    for key, value in details.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)
