from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def to_wsl_path(path: Path | str) -> str:
    raw = str(path)
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return raw.replace("\\", "/")


def translate_job_paths(jobs: Iterable[Dict[str, Any]], book_root: Path | str) -> List[Dict[str, Any]]:
    root = Path(book_root)
    translated: List[Dict[str, Any]] = []
    for job in jobs:
        item = dict(job)
        voice_path = str(item.get("voice_config_path") or "").strip()
        if voice_path:
            path = Path(voice_path)
            if not path.is_absolute():
                path = root / path
            item["voice_config_path"] = to_wsl_path(path)
        translated.append(item)
    return translated
