from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


class FakeTtsAdapter:
    def __init__(self, sample_rate: int = 24000, samples_per_character: int = 100) -> None:
        self.sample_rate = sample_rate
        self.samples_per_character = samples_per_character

    def ensure_voice(
        self,
        role_id: str,
        voice_record: Dict,
        voice_path: Path,
        sample_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
    ) -> Path:
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        sample = Path(sample_path) if sample_path is not None else None
        sample_missing = sample is not None and not sample.exists()
        if voice_record.get("_force_regenerate") or not voice_path.exists() or sample_missing:
            voice_path.write_bytes(f"fake voice for {role_id}".encode("utf-8"))
            if sample is not None:
                _write_fake_wav(
                    sample,
                    samples=np.full(
                        max(1, len(str(reference_text or role_id))) * self.samples_per_character,
                        0.05,
                        dtype=np.float32,
                    ),
                    sample_rate=self.sample_rate,
                )
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
                    unit_idx=int(job.get("unit_idx", job["sentence_idx"])),
                )
            )
        return generated

    def generate_sentence_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        yield self.generate_sentences(jobs)


def _write_fake_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
