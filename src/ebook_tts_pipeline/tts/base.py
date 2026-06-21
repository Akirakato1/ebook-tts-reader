from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Protocol

import numpy as np


@dataclass(frozen=True)
class GeneratedSentenceAudio:
    sentence_idx: int
    role: str
    speech_type: str
    samples: np.ndarray
    sample_rate: int


class TtsAdapter(Protocol):
    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        ...

    def generate_sentence_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        ...

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        ...
