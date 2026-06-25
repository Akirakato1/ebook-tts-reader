from __future__ import annotations

import threading
from collections import deque
from typing import Iterable


class SubprocessStderrTail:
    def __init__(self, name: str, max_lines: int = 200) -> None:
        self.name = name
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, stream: Iterable[str] | None) -> None:
        if stream is None:
            return
        self._thread = threading.Thread(
            target=self._drain,
            args=(stream,),
            name=f"{self.name}-stderr-drain",
            daemon=True,
        )
        self._thread.start()

    def tail(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def _drain(self, stream: Iterable[str]) -> None:
        try:
            for line in stream:
                text = str(line).rstrip()
                if text:
                    self._append(text)
        except Exception as exc:  # pragma: no cover - defensive guard for stream teardown.
            self._append(f"[stderr drain failed: {exc}]")

    def _append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
