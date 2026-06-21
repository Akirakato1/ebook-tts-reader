from __future__ import annotations

import gc
import shutil
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
        return self.build_chapter_audio_from_windows(
            chapter=chapter,
            job_windows=[jobs],
            audio_path=audio_path,
            timeline_path=timeline_path,
        )

    def build_chapter_audio_from_windows(
        self,
        chapter: str,
        job_windows: List[List[Dict]],
        audio_path: Union[str, Path],
        timeline_path: Union[str, Path],
    ) -> Dict:
        audio_target = Path(audio_path)
        chunk_dir = audio_target.with_suffix(".chunks")
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        total_jobs = sum(len(jobs) for jobs in job_windows)
        if total_jobs == 0:
            raise ValueError("Cannot build audio without sentence jobs.")

        sample_rate = None
        cursor_samples = 0
        timeline_sentences: List[Dict] = []
        chunk_paths: List[Path] = []
        emitted_sentences = 0

        try:
            for window_idx, jobs in enumerate(job_windows):
                generated = self.tts_adapter.generate_sentences(jobs)
                if not generated:
                    continue
                if sample_rate is None:
                    sample_rate = generated[0].sample_rate
                pause_samples = int(sample_rate * self.pause_between_sentences_ms / 1000)
                chunks: List[np.ndarray] = []

                for item in generated:
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
                    emitted_sentences += 1
                    if emitted_sentences < total_jobs and pause_samples:
                        chunks.append(np.zeros(pause_samples, dtype=np.float32))
                        cursor_samples += pause_samples

                chunk_path = chunk_dir / f"{window_idx:05d}.wav"
                self._write_wav(chunk_path, np.concatenate(chunks), sample_rate)
                chunk_paths.append(chunk_path)
                del generated
                del chunks
                gc.collect()

            if sample_rate is None or not chunk_paths:
                raise ValueError("Cannot build audio without generated sentence audio.")

            self._stitch_wav_chunks(chunk_paths, audio_target, sample_rate)
            timeline = {
                "chapter": chapter,
                "audio_path": str(audio_path),
                "sample_rate": sample_rate,
                "sentences": timeline_sentences,
            }
            write_json_atomic(timeline_path, timeline)
            return timeline
        finally:
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)

    def _write_wav(self, path: Path, samples: np.ndarray, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm16.tobytes())

    def _stitch_wav_chunks(self, chunk_paths: List[Path], output_path: Path, sample_rate: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            for chunk_path in chunk_paths:
                with wave.open(str(chunk_path), "rb") as chunk:
                    if chunk.getframerate() != sample_rate:
                        raise ValueError("All chunk audio must use the same sample rate.")
                    if chunk.getnchannels() != 1 or chunk.getsampwidth() != 2:
                        raise ValueError("Chunk audio must be mono 16-bit PCM.")
                    output.writeframes(chunk.readframes(chunk.getnframes()))
