from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ebook_tts_pipeline.domain import Sentence


@dataclass(frozen=True)
class LlmWindow:
    sentences: List[Sentence]


@dataclass(frozen=True)
class TtsWindow:
    jobs: List[Dict[str, Any]]


def build_llm_windows(sentences: List[Sentence], max_chars: int) -> List[LlmWindow]:
    windows: List[LlmWindow] = []
    current: List[Sentence] = []
    current_chars = 0
    for sentence in sentences:
        sentence_size = len(sentence.text)
        if sentence_size > max_chars:
            raise ValueError(f"sentence {sentence.idx} exceeds max LLM window size")
        if current and current_chars + sentence_size > max_chars:
            windows.append(LlmWindow(sentences=current))
            current = []
            current_chars = 0
        current.append(sentence)
        current_chars += sentence_size
    if current:
        windows.append(LlmWindow(sentences=current))
    return windows


def build_tts_windows(
    jobs: List[Dict[str, Any]],
    max_chars: int,
    max_roles: int,
) -> List[TtsWindow]:
    windows: List[TtsWindow] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    current_roles = set()
    for job in jobs:
        text_size = len(str(job["text"]))
        role = str(job["role"])
        if text_size > max_chars:
            raise ValueError(f"sentence {job['sentence_idx']} exceeds max TTS window size")
        next_roles = current_roles | {role}
        would_exceed_chars = bool(current) and current_chars + text_size > max_chars
        would_exceed_roles = bool(current) and len(next_roles) > max_roles
        if would_exceed_chars or would_exceed_roles:
            windows.append(TtsWindow(jobs=current))
            current = []
            current_chars = 0
            current_roles = set()
        current.append(job)
        current_chars += text_size
        current_roles.add(role)
    if current:
        windows.append(TtsWindow(jobs=current))
    return windows
