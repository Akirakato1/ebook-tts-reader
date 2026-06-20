from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from ebook_tts_pipeline.json_io import write_json_atomic
from ebook_tts_pipeline.tts.base import TtsAdapter


class ChapterAudioBuilder:
    def __init__(self, tts_adapter: TtsAdapter, pause_between_sentences_ms: int) -> None:
        self.tts_adapter = tts_adapter
        self.pause_between_sentences_ms = pause_between_sentences_ms

    def build_chapter_audio(
        self,
        chapter: str,
        jobs: List[Dict],
        audio_path: Union[str, Path],
        timeline_path: Union[str, Path],
    ) -> Dict:
        generated = self.tts_adapter.generate_sentences(jobs)
        if not generated:
            raise ValueError("Cannot build audio without sentence jobs.")

        sample_rate = generated[0].sample_rate
        cursor_samples = 0
        timeline_sentences: List[Dict] = []
        chunks: List[np.ndarray] = []
        pause_samples = int(sample_rate * self.pause_between_sentences_ms / 1000)

        for index, item in enumerate(generated):
            if item.sample_rate != sample_rate:
                raise ValueError("All generated sentence audio must use the same sample rate.")
            start_samples = cursor_samples
            end_samples = start_samples + len(item.samples)
            timeline_sentences.append(
                {
                    "sentence_idx": item.sentence_idx,
                    "role": item.role,
                    "type": item.speech_type,
                    "start_ms": int(round(start_samples * 1000 / sample_rate)),
                    "end_ms": int(round(end_samples * 1000 / sample_rate)),
                }
            )
            chunks.append(item.samples.astype(np.float32))
            cursor_samples = end_samples
            if index + 1 < len(generated) and pause_samples:
                chunks.append(np.zeros(pause_samples, dtype=np.float32))
                cursor_samples += pause_samples

        merged = np.concatenate(chunks)
        self._write_wav(Path(audio_path), merged, sample_rate)
        timeline = {
            "chapter": chapter,
            "audio_path": str(audio_path),
            "sample_rate": sample_rate,
            "sentences": timeline_sentences,
        }
        write_json_atomic(timeline_path, timeline)
        return timeline

    def _write_wav(self, path: Path, samples: np.ndarray, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm16.tobytes())
