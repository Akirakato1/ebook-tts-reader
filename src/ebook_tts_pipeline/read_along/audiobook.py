from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.windowing import build_tts_windows


AUDIOBOOK_WINDOW_PROFILES: Dict[str, Dict[str, int]] = {
    "precise": {"max_chars": 2400, "max_roles": 4},
    "balanced": {"max_chars": 6000, "max_roles": 8},
    "fast": {"max_chars": 9000, "max_roles": 8},
}


def audiobook_window_profile(mode: str) -> Dict[str, int]:
    key = str(mode or "balanced").strip().lower()
    return dict(AUDIOBOOK_WINDOW_PROFILES.get(key, AUDIOBOOK_WINDOW_PROFILES["balanced"]))


def build_audiobook_windows(
    units: Iterable[Dict[str, Any] | ReadAlongUnit],
    max_chars: int,
    max_roles: int,
) -> List[List[Dict[str, Any]]]:
    jobs = [_unit_to_job(unit) for unit in units]
    return [window.jobs for window in build_tts_windows(jobs, max_chars=int(max_chars), max_roles=int(max_roles))]


def _unit_to_job(unit: Dict[str, Any] | ReadAlongUnit) -> Dict[str, Any]:
    if isinstance(unit, ReadAlongUnit):
        return unit.to_tts_job()
    return ReadAlongUnit.from_dict(dict(unit)).to_tts_job()
