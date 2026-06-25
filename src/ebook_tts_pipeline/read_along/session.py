from __future__ import annotations

import json
import math
import shutil
import time
import wave
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio, TtsAdapter


@dataclass(frozen=True)
class BufferedAudio:
    unit_id: int
    audio_path: Optional[Path]
    audio_bytes: bytes
    playback_seconds: float


class ReadAlongSession:
    def __init__(
        self,
        session_id: str,
        units: List[ReadAlongUnit],
        tts_adapter: TtsAdapter,
        session_dir: Path,
        timing_log_path: Path,
        buffer_limit: int = 2,
        target_buffer_seconds: float = 20.0,
        start_buffer_seconds: float = 20.0,
        max_buffer_seconds: float = 40.0,
        max_buffer_units: int = 32,
        playback_speed: float = 1.0,
        generation_mode: str = "balanced",
        store_audio_files: bool = True,
    ) -> None:
        self.session_id = session_id
        self.units = list(units)
        self.tts_adapter = tts_adapter
        self.session_dir = Path(session_dir)
        self.timing_log_path = Path(timing_log_path)
        self.buffer_limit = max(1, int(buffer_limit))
        self.target_buffer_seconds = max(0.1, float(target_buffer_seconds))
        self.start_buffer_seconds = max(0.1, min(float(start_buffer_seconds), self.target_buffer_seconds))
        self.max_buffer_seconds = max(self.target_buffer_seconds, float(max_buffer_seconds))
        self.max_buffer_units = max(1, int(max_buffer_units))
        self.playback_speed = max(0.1, float(playback_speed))
        self.generation_mode = str(generation_mode)
        self.store_audio_files = bool(store_audio_files)
        self._next_unit_id = 0
        self._ready: List[BufferedAudio] = []
        self._ended = False

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    @property
    def ready_playback_seconds(self) -> float:
        return sum(item.playback_seconds for item in self._ready)

    @property
    def ready_unit_ids(self) -> List[int]:
        return [item.unit_id for item in self._ready]

    @property
    def ready_items(self) -> List[BufferedAudio]:
        return list(self._ready)

    @property
    def has_more_units(self) -> bool:
        return self._next_unit_id < len(self.units)

    def peek_ready(self) -> Optional[BufferedAudio]:
        if not self._ready:
            return None
        return self._ready[0]

    def fill_buffer(
        self,
        start_unit_id: Optional[int] = None,
        min_buffer_seconds: Optional[float] = None,
        exclude_unit_id: Optional[int] = None,
    ) -> List[BufferedAudio]:
        if self._ended:
            return []
        if start_unit_id is not None:
            self._next_unit_id = int(start_unit_id)
        target_seconds = float(min_buffer_seconds) if min_buffer_seconds is not None else self.target_buffer_seconds
        target_seconds = min(self.max_buffer_seconds, max(0.1, target_seconds))
        generated: List[BufferedAudio] = []
        while (
            self._ready_playback_seconds(exclude_unit_id=exclude_unit_id) < target_seconds
            and len(self._ready) < self.max_buffer_units
            and self._next_unit_id < len(self.units)
        ):
            open_unit_slots = max(1, min(self.max_buffer_units - len(self._ready), self.buffer_limit))
            batch_size = self._next_batch_size(
                open_unit_slots,
                target_seconds,
                self._ready_playback_seconds(exclude_unit_id=exclude_unit_id),
            )
            batch_units = self.units[self._next_unit_id:self._next_unit_id + batch_size]
            if not batch_units:
                break
            generated.extend(self._generate_units(batch_units))
            if self._ready_playback_seconds(exclude_unit_id=exclude_unit_id) >= self.max_buffer_seconds:
                break
        return generated

    def _ready_playback_seconds(self, exclude_unit_id: Optional[int] = None) -> float:
        if exclude_unit_id is None:
            return self.ready_playback_seconds
        excluded = int(exclude_unit_id)
        return sum(item.playback_seconds for item in self._ready if item.unit_id != excluded)

    def _next_batch_size(self, open_unit_slots: int, target_seconds: float, ready_seconds: float) -> int:
        if self.generation_mode == "precise":
            return 1
        if self.generation_mode == "balanced":
            max_batch = max(1, min(2, open_unit_slots))
        else:
            max_batch = max(1, open_unit_slots)
        estimated_unit_seconds = self._estimated_ready_unit_seconds()
        if estimated_unit_seconds <= 0:
            return max_batch
        remaining_seconds = max(0.1, target_seconds - ready_seconds)
        seconds_limited_batch = max(1, math.ceil(remaining_seconds / estimated_unit_seconds))
        return max(1, min(max_batch, seconds_limited_batch))

    def _estimated_ready_unit_seconds(self) -> float:
        if not self._ready:
            return 0.0
        return self.ready_playback_seconds / len(self._ready)

    def consume_ready(self) -> Optional[BufferedAudio]:
        if not self._ready:
            return None
        item = self._ready.pop(0)
        if item.audio_path is not None:
            try:
                item.audio_path.unlink()
            except OSError:
                pass
        return item

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            close = getattr(self.tts_adapter, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        finally:
            shutil.rmtree(self.session_dir, ignore_errors=True)
            self._ready.clear()

    def _generate_units(self, units: List[ReadAlongUnit]) -> List[BufferedAudio]:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        buffer_seconds_before = self.ready_playback_seconds
        started = time.perf_counter()
        generated = self.tts_adapter.generate_sentences([unit.to_tts_job() for unit in units])
        generation_seconds = time.perf_counter() - started
        by_unit: Dict[int, GeneratedSentenceAudio] = {
            int(item.unit_idx if item.unit_idx is not None else item.sentence_idx): item
            for item in generated
        }
        buffered = []
        raw_audio_seconds = 0.0
        for unit in units:
            item = by_unit[unit.unit_id]
            audio_seconds = len(item.samples) / item.sample_rate
            raw_audio_seconds += audio_seconds
            audio_bytes = _wav_bytes(item.samples, item.sample_rate)
            audio_path = None
            if self.store_audio_files:
                audio_path = self.session_dir / f"{unit.unit_id:05d}.wav"
                audio_path.write_bytes(audio_bytes)
            buffered.append(
                BufferedAudio(
                    unit_id=unit.unit_id,
                    audio_path=audio_path,
                    audio_bytes=audio_bytes,
                    playback_seconds=audio_seconds / self.playback_speed,
                )
            )
        playback_seconds = raw_audio_seconds / self.playback_speed
        self._append_timing(
            {
                "session_id": self.session_id,
                "unit_ids": [unit.unit_id for unit in units],
                "roles": [unit.role for unit in units],
                "role_ids": [unit.role_id for unit in units],
                "voice_config_paths": [unit.voice_config_path for unit in units],
                "text_chars": [len(unit.text) for unit in units],
                "source_offsets": [[unit.source_start, unit.source_end] for unit in units],
                "generation_mode": self.generation_mode,
                "buffer_limit": self.buffer_limit,
                "target_buffer_seconds": self.target_buffer_seconds,
                "start_buffer_seconds": self.start_buffer_seconds,
                "max_buffer_seconds": self.max_buffer_seconds,
                "max_buffer_units": self.max_buffer_units,
                "buffer_seconds_before_generation": buffer_seconds_before,
                "buffer_seconds_after_generation": buffer_seconds_before + playback_seconds,
                "unit_audio_seconds": [
                    len(by_unit[unit.unit_id].samples) / by_unit[unit.unit_id].sample_rate
                    for unit in units
                ],
                "playback_speed": self.playback_speed,
                "generation_seconds": generation_seconds,
                "raw_audio_seconds": raw_audio_seconds,
                "playback_seconds": playback_seconds,
                "realtime_factor": generation_seconds / playback_seconds if playback_seconds else None,
                "success": True,
            }
        )
        self._ready.extend(buffered)
        self._next_unit_id += len(units)
        return buffered

    def _append_timing(self, row: Dict) -> None:
        self.timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.timing_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.write_bytes(_wav_bytes(samples, sample_rate))


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    handle = BytesIO()
    with wave.open(handle, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16.tobytes())
    return handle.getvalue()
