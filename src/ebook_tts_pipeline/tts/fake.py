from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


class FakeTtsAdapter:
    def __init__(self, sample_rate: int = 24000, samples_per_character: int = 100) -> None:
        self.sample_rate = sample_rate
        self.samples_per_character = samples_per_character

    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        if not voice_path.exists():
            voice_path.write_bytes(f"fake voice for {role_id}".encode("utf-8"))
        return voice_path

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        generated: List[GeneratedSentenceAudio] = []
        for job in jobs:
            length = len(str(job["text"])) * self.samples_per_character
            samples = np.full(length, 0.05, dtype=np.float32)
            generated.append(
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    role=str(job["role"]),
                    speech_type=str(job["type"]),
                    samples=samples,
                    sample_rate=self.sample_rate,
                )
            )
        return generated
