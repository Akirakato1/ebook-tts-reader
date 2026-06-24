from __future__ import annotations

import gc
import shutil
import wave
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from ebook_tts_pipeline.json_io import write_json_atomic
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio, TtsAdapter


class ChapterAudioBuilder:
    def __init__(
        self,
        tts_adapter: TtsAdapter,
        pause_between_sentences_ms: int,
        tts_speed: float = 1.0,
        intra_sentence_pause_ms: int = 50,
    ) -> None:
        self.tts_adapter = tts_adapter
        self.pause_between_sentences_ms = pause_between_sentences_ms
        self.intra_sentence_pause_ms = max(0, int(intra_sentence_pause_ms))
        self.tts_speed = max(0.1, float(tts_speed))

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
        ordered_jobs = [job for jobs in job_windows for job in jobs]
        emitted_sentences = 0
        chunk_index = 0

        try:
            for jobs in job_windows:
                for generated in self._iter_generated_batches(jobs):
                    if not generated:
                        continue
                    if sample_rate is None:
                        sample_rate = generated[0].sample_rate
                    chunks: List[np.ndarray] = []

                    for item in generated:
                        if item.sample_rate != sample_rate:
                            raise ValueError("All generated sentence audio must use the same sample rate.")
                        samples = self._apply_speed(item.samples.astype(np.float32))
                        start_samples = cursor_samples
                        end_samples = start_samples + len(samples)
                        timeline_row = {
                            "sentence_idx": item.sentence_idx,
                            "unit_idx": item.unit_idx if item.unit_idx is not None else item.sentence_idx,
                            "role": item.role,
                            "type": item.speech_type,
                            "start_ms": int(round(start_samples * 1000 / sample_rate)),
                            "end_ms": int(round(end_samples * 1000 / sample_rate)),
                        }
                        if item.voice_config_path is not None:
                            timeline_row["voice_config_path"] = item.voice_config_path
                        timeline_sentences.append(timeline_row)
                        chunks.append(samples)
                        cursor_samples = end_samples
                        emitted_sentences += 1
                        pause_ms = (
                            item.pause_after_ms
                            if item.pause_after_ms is not None
                            else self._pause_after_item(
                                current=item,
                                next_job=ordered_jobs[emitted_sentences] if emitted_sentences < total_jobs else None,
                            )
                        )
                        pause_samples = int(sample_rate * pause_ms / 1000)
                        if emitted_sentences < total_jobs and pause_samples:
                            chunks.append(np.zeros(pause_samples, dtype=np.float32))
                            cursor_samples += pause_samples

                    chunk_path = chunk_dir / f"{chunk_index:05d}.wav"
                    chunk_index += 1
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

    def _iter_generated_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        batch_generator = getattr(self.tts_adapter, "generate_sentence_batches", None)
        if batch_generator is not None:
            yield from batch_generator(jobs)
            return
        generated = self.tts_adapter.generate_sentences(jobs)
        if generated:
            yield generated

    def _apply_speed(self, samples: np.ndarray) -> np.ndarray:
        if self.tts_speed == 1.0 or len(samples) <= 1:
            return samples
        target_len = max(1, int(round(len(samples) / self.tts_speed)))
        if target_len == len(samples):
            return samples
        source_positions = np.linspace(0, len(samples) - 1, num=len(samples), dtype=np.float32)
        target_positions = np.linspace(0, len(samples) - 1, num=target_len, dtype=np.float32)
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    def _pause_after_item(self, current: GeneratedSentenceAudio, next_job: Optional[Dict]) -> int:
        if next_job is None:
            return 0
        next_sentence_idx = int(next_job["sentence_idx"])
        if next_sentence_idx == current.sentence_idx:
            return self.intra_sentence_pause_ms
        return self.pause_between_sentences_ms

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
