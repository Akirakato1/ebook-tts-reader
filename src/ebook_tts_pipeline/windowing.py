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


def build_llm_windows(
    sentences: List[Sentence],
    max_chars: int,
    max_sentences: int = 0,
) -> List[LlmWindow]:
    windows: List[LlmWindow] = []
    current: List[Sentence] = []
    current_chars = 0
    for sentence in sentences:
        sentence_size = len(sentence.text)
        if sentence_size > max_chars:
            raise ValueError(f"sentence {sentence.idx} exceeds max LLM window size")
        would_exceed_chars = current_chars + sentence_size > max_chars
        would_exceed_sentences = max_sentences > 0 and len(current) >= max_sentences
        if current and (would_exceed_chars or would_exceed_sentences):
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
    current_roles = set()
    for job in jobs:
        role = str(job["role"])
        if _qwen_section_char_count([job]) > max_chars:
            raise ValueError(f"sentence {job['sentence_idx']} exceeds max TTS window size")
        next_roles = current_roles | {role}
        would_exceed_chars = bool(current) and _qwen_section_char_count(current + [job]) > max_chars
        would_exceed_roles = bool(current) and len(next_roles) > max_roles
        if would_exceed_chars or would_exceed_roles:
            windows.append(TtsWindow(jobs=current))
            current = []
            current_roles = set()
        current.append(job)
        current_roles.add(role)
    if current:
        windows.append(TtsWindow(jobs=current))
    return windows


def _qwen_section_char_count(jobs: List[Dict[str, Any]]) -> int:
    lines: List[str] = []
    current_role = ""
    current_text: List[str] = []

    for job in jobs:
        role = str(job["role"])
        text = " ".join(str(job["text"]).split())
        if current_text and role != current_role:
            lines.append(f"{current_role}: {' '.join(current_text)}")
            current_text = []
        current_role = role
        if text:
            current_text.append(text)

    if current_text:
        lines.append(f"{current_role}: {' '.join(current_text)}")
    return len("\n".join(lines))
