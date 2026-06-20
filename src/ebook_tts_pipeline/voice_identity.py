from __future__ import annotations

import hashlib
from typing import List


DIFFERENTIATORS = [
    "brighter timbre",
    "darker timbre",
    "slightly quicker cadence",
    "slower deliberate cadence",
    "lighter resonance",
    "deeper chest resonance",
    "more breathiness",
    "cleaner crisp articulation",
]


def role_seed(book_slug: str, role_id: str) -> int:
    digest = hashlib.sha256(f"{book_slug}:{role_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def choose_differentiators(book_slug: str, role_id: str, count: int = 3) -> List[str]:
    seed = role_seed(book_slug, role_id)
    start = seed % len(DIFFERENTIATORS)
    return [DIFFERENTIATORS[(start + offset) % len(DIFFERENTIATORS)] for offset in range(count)]


def append_differentiators(qwen_instruct: str, differentiators: List[str]) -> str:
    suffix = ", ".join(differentiators)
    stripped = qwen_instruct.rstrip(". ")
    return f"{stripped}, with {suffix}."
