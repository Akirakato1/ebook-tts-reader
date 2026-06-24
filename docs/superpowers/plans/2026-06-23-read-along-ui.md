# Read-Along UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Read Along tab that displays ebook-style pages with inline sentence/quote highlighting, locked session settings, a max-2 audio buffer, per-unit timing logs, and chapter 15 validation.

**Architecture:** Add a focused read-along package for offset-preserving units and buffer/session behavior, then wire it into the existing `PrototypeUiController` and Tk app. Keep audiobook generation intact and treat read-along audio as temporary session data.

**Tech Stack:** Python 3.9+, Tkinter/ttk, existing `AudiobookPipeline`, existing Qwen/Fake TTS adapters, `pytest`.

---

## File Structure

- Create `src/ebook_tts_pipeline/read_along/__init__.py`: exports public read-along types and helpers.
- Create `src/ebook_tts_pipeline/read_along/units.py`: builds offset-preserving read-along units from chapter text, quote extraction, quote attribution, registry, and temp registry.
- Create `src/ebook_tts_pipeline/read_along/session.py`: owns buffer-limited generation, temp audio lifecycle, timing logs, session cancellation, and no-op/fake playback hooks for tests.
- Modify `src/ebook_tts_pipeline/paths.py`: add read-along artifact/log/session path helpers.
- Modify `src/ebook_tts_pipeline/pipeline.py`: expose `build_read_along_units(chapter)` and reuse quote annotation artifacts.
- Modify `src/ebook_tts_pipeline/ui/controller.py`: expose book processing, read-along unit loading, narrator voice type settings, and session factory methods.
- Modify `src/ebook_tts_pipeline/ui/tk_app.py`: add notebook tabs and the Read Along tab with natural page text, inline tags, session controls, and locked settings.
- Create `tests/test_read_along_units.py`: unit builder and source-offset tests.
- Create `tests/test_read_along_session.py`: buffer limit, timing logging, and cleanup tests.
- Extend `tests/test_ui_controller.py`: controller methods for read-along units/settings/book processing.

## Task 1: Add Read-Along Paths

**Files:**
- Modify: `src/ebook_tts_pipeline/paths.py`
- Test: `tests/test_read_along_units.py`

- [ ] **Step 1: Write the failing path test**

```python
from ebook_tts_pipeline.paths import BookPaths


def test_book_paths_exposes_read_along_artifacts(tmp_path):
    paths = BookPaths(tmp_path / "book")

    assert paths.read_along_units("chapter_015") == (
        tmp_path / "book" / "read_along" / "chapter_015.units.json"
    )
    assert paths.read_along_session_dir("session-1") == (
        tmp_path / "book" / "read_along_sessions" / "session-1"
    )
    assert paths.read_along_timing_log("session-1") == (
        tmp_path / "book" / "read_along_sessions" / "session-1" / "timings.jsonl"
    )
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_read_along_units.py::test_book_paths_exposes_read_along_artifacts -v`

Expected: FAIL because the new path methods do not exist.

- [ ] **Step 3: Implement path helpers**

Add these methods to `BookPaths`:

```python
    def read_along_units(self, chapter: str) -> Path:
        return self.root / "read_along" / f"{chapter}.units.json"

    def read_along_session_dir(self, session_id: str) -> Path:
        return self.root / "read_along_sessions" / session_id

    def read_along_timing_log(self, session_id: str) -> Path:
        return self.read_along_session_dir(session_id) / "timings.jsonl"
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `pytest tests/test_read_along_units.py::test_book_paths_exposes_read_along_artifacts -v`

Expected: PASS.

## Task 2: Build Offset-Preserving Read-Along Units

**Files:**
- Create: `src/ebook_tts_pipeline/read_along/__init__.py`
- Create: `src/ebook_tts_pipeline/read_along/units.py`
- Test: `tests/test_read_along_units.py`

- [ ] **Step 1: Write failing tests for quote and narrator units**

Append to `tests/test_read_along_units.py`:

```python
from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionResult
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue
from ebook_tts_pipeline.read_along.units import build_read_along_units


def _registry_with_voices():
    return {
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
            "voice_profile": {"qwen_instruct": "male narrator"},
        },
        "characters": {
            "leigh_adult": {
                "role_id": "leigh_adult",
                "display_name": "Leigh",
                "age_stage": "adult",
                "voice_config_path": "voices/leigh_adult.qvp",
                "voice_profile": {"qwen_instruct": "adult female"},
            }
        },
    }


def test_build_read_along_units_preserves_quote_offsets():
    text = 'Leigh said, "Right." Then she left.'
    extraction = extract_quoted_dialogue(text)
    attribution = QuoteAttributionResult(
        roles=["leigh_adult"],
        quotes=[(1, 0, "dialogue")],
    )

    units = build_read_along_units(
        chapter="chapter_015",
        chapter_text=text,
        extraction=extraction,
        attribution=attribution,
        registry=_registry_with_voices(),
        temp_registry={},
    )

    assert [unit.text for unit in units] == ["Leigh said,", '"Right."', "Then she left."]
    assert units[0].role_id == "narrator"
    assert units[1].role_id == "leigh_adult"
    assert text[units[1].source_start:units[1].source_end] == '"Right."'
    assert units[1].voice_config_path == "voices/leigh_adult.qvp"


def test_narrator_quote_uses_narrator_voice():
    text = 'The sign said "Closed" on the door.'
    extraction = extract_quoted_dialogue(text)
    attribution = QuoteAttributionResult(
        roles=["Narrator"],
        quotes=[(1, 0, "narrator_quote")],
    )

    units = build_read_along_units(
        chapter="chapter_015",
        chapter_text=text,
        extraction=extraction,
        attribution=attribution,
        registry=_registry_with_voices(),
        temp_registry={},
    )

    closed = [unit for unit in units if unit.text == '"Closed"'][0]
    assert closed.role_id == "narrator"
    assert closed.type == "narration"
    assert closed.voice_config_path == "voices/narrator.qvp"
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_read_along_units.py -v`

Expected: FAIL because `ebook_tts_pipeline.read_along.units` does not exist.

- [ ] **Step 3: Implement unit dataclass and builder**

Create `src/ebook_tts_pipeline/read_along/units.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionResult
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction
from ebook_tts_pipeline.ingestion import fallback_sentence_tokenize
from ebook_tts_pipeline.registry import resolve_effective_voice
from ebook_tts_pipeline.temp_registry import resolve_temp_voice


@dataclass(frozen=True)
class ReadAlongUnit:
    chapter: str
    unit_id: int
    text: str
    source_start: int
    source_end: int
    role: str
    role_id: str
    type: str
    voice_config_path: Optional[str]
    quote_id: Optional[str] = None
    sentence_idx: Optional[int] = None
    character: Optional[str] = None
    voice_variant: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReadAlongUnit":
        return cls(
            chapter=str(data["chapter"]),
            unit_id=int(data["unit_id"]),
            text=str(data["text"]),
            source_start=int(data["source_start"]),
            source_end=int(data["source_end"]),
            role=str(data["role"]),
            role_id=str(data["role_id"]),
            type=str(data["type"]),
            voice_config_path=data.get("voice_config_path"),
            quote_id=data.get("quote_id"),
            sentence_idx=int(data["sentence_idx"]) if data.get("sentence_idx") is not None else None,
            character=data.get("character"),
            voice_variant=data.get("voice_variant"),
        )

    def to_tts_job(self) -> Dict[str, Any]:
        payload = {
            "sentence_idx": self.unit_id,
            "unit_idx": self.unit_id,
            "role": self.role,
            "role_id": self.role_id,
            "type": self.type,
            "text": self.text,
            "voice_config_path": self.voice_config_path,
            "_read_along_source_start": self.source_start,
            "_read_along_source_end": self.source_end,
        }
        if self.character is not None:
            payload["character"] = self.character
        if self.voice_variant is not None:
            payload["voice_variant"] = self.voice_variant
        return payload


def build_read_along_units(
    chapter: str,
    chapter_text: str,
    extraction: QuoteExtraction,
    attribution: QuoteAttributionResult,
    registry: Dict[str, Any],
    temp_registry: Dict[str, Any],
) -> List[ReadAlongUnit]:
    quote_roles = {
        quote_idx: (attribution.roles[role_idx], quote_type)
        for quote_idx, role_idx, quote_type in attribution.quotes
    }
    narrator_effective = resolve_effective_voice(registry, "Narrator", "narration")
    segments = []

    for span in extraction.narrator_spans:
        segments.extend(
            _split_narrator_span(
                chapter_text=chapter_text,
                start=span.start,
                end=span.end,
                effective=narrator_effective,
            )
        )

    for quote in extraction.quotes:
        role_name, quote_type = quote_roles[quote.idx]
        if quote_type == "narrator_quote":
            effective = narrator_effective
            speech_type = "narration"
        else:
            speech_type = "dialogue"
            try:
                effective = resolve_effective_voice(registry, role_name, speech_type)
            except ValueError:
                effective = resolve_temp_voice(temp_registry, role_name, speech_type)
                if effective is None:
                    raise
        segments.append((quote.start, quote.end, quote.text.strip(), speech_type, effective, quote.quote_id))

    units = []
    for unit_id, (start, end, text, speech_type, effective, quote_id) in enumerate(
        sorted(segments, key=lambda item: (item[0], item[1]))
    ):
        record = effective["voice_record"]
        units.append(
            ReadAlongUnit(
                chapter=chapter,
                unit_id=unit_id,
                text=text,
                source_start=start,
                source_end=end,
                role=str(effective["role"]),
                role_id=str(effective["role_id"]),
                type=speech_type,
                voice_config_path=record.get("voice_config_path"),
                quote_id=quote_id,
                sentence_idx=unit_id,
                character=effective["character"],
                voice_variant=effective["voice_variant"],
            )
        )
    return units


def _split_narrator_span(
    chapter_text: str,
    start: int,
    end: int,
    effective: Dict[str, Any],
) -> List[tuple]:
    raw = chapter_text[start:end]
    stripped = raw.strip()
    if not stripped:
        return []
    leading = len(raw) - len(raw.lstrip())
    search_start = start + leading
    parts = fallback_sentence_tokenize(stripped) or [stripped]
    segments = []
    cursor = search_start
    for part in parts:
        found = chapter_text.find(part, cursor, end)
        if found < 0:
            found = cursor
        part_end = found + len(part)
        segments.append((found, part_end, part, "narration", effective, None))
        cursor = part_end
    return segments
```

Create `src/ebook_tts_pipeline/read_along/__init__.py`:

```python
from ebook_tts_pipeline.read_along.units import ReadAlongUnit, build_read_along_units

__all__ = ["ReadAlongUnit", "build_read_along_units"]
```

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_read_along_units.py -v`

Expected: PASS.

## Task 3: Pipeline And Controller Artifact Methods

**Files:**
- Modify: `src/ebook_tts_pipeline/pipeline.py`
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write failing controller tests**

Append to `tests/test_ui_controller.py`:

```python
def test_controller_builds_read_along_units_from_quote_annotation(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right."', encoding="utf-8")
    _write_callie_registry(paths)
    registry = read_json(paths.registry)
    registry["characters"]["leigh_adult"] = {
        "role_id": "leigh_adult",
        "profile_id": "leigh_adult",
        "person_id": "leigh",
        "display_name": "Leigh",
        "age_stage": "adult",
        "voice_config_path": "voices/leigh_adult.qvp",
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
        "voice_profile": {"qwen_instruct": "adult female"},
    }
    write_json_atomic(paths.registry, registry)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )

    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    units = controller.build_read_along_units("chapter_001")

    assert paths.read_along_units("chapter_001").exists()
    assert any(unit["role_id"] == "leigh_adult" for unit in units)
    assert any(unit["role_id"] == "narrator" for unit in units)
```

- [ ] **Step 2: Run test and verify it fails**

Run: `pytest tests/test_ui_controller.py::test_controller_builds_read_along_units_from_quote_annotation -v`

Expected: FAIL because `build_read_along_units` does not exist on the controller.

- [ ] **Step 3: Implement pipeline method**

Add imports to `pipeline.py`:

```python
from ebook_tts_pipeline.read_along.units import build_read_along_units
```

Add method on `AudiobookPipeline`:

```python
    def build_read_along_units(self, chapter: str) -> List[Dict]:
        raw_annotation = read_json(self.paths.annotation(chapter))
        if not _is_quote_attribution_payload(raw_annotation):
            raise ValueError("Read-along mode requires quote attribution annotation.")
        registry = self.registry.load()
        annotation = AnnotationResult.from_dict(raw_annotation)
        temp_registry = ChapterTempRegistryManager(self.paths).write_for_annotation(
            chapter,
            registry,
            annotation,
        )
        chapter_text = self.paths.chapter_text(chapter).read_text(encoding="utf-8", errors="replace")
        extraction = extract_quoted_dialogue(chapter_text)
        units = build_read_along_units(
            chapter=chapter,
            chapter_text=chapter_text,
            extraction=extraction,
            attribution=QuoteAttributionResult.from_dict(raw_annotation),
            registry=registry,
            temp_registry=temp_registry,
        )
        payload = {"chapter": chapter, "units": [unit.to_dict() for unit in units]}
        write_json_atomic(self.paths.read_along_units(chapter), payload)
        return payload["units"]
```

- [ ] **Step 4: Implement controller methods**

Add to `PrototypeUiController`:

```python
    def build_read_along_units(self, chapter: str) -> List[Dict[str, Any]]:
        pipeline = self._pipeline(needs_llm=False)
        return pipeline.build_read_along_units(chapter)

    def read_along_units(self, chapter: str) -> List[Dict[str, Any]]:
        if not self.paths.read_along_units(chapter).exists():
            return self.build_read_along_units(chapter)
        return list(read_json(self.paths.read_along_units(chapter)).get("units", []))

    def chapter_text(self, chapter: str) -> str:
        return self.paths.chapter_text(chapter).read_text(encoding="utf-8", errors="replace")
```

- [ ] **Step 5: Run controller test**

Run: `pytest tests/test_ui_controller.py::test_controller_builds_read_along_units_from_quote_annotation -v`

Expected: PASS.

## Task 4: Add Buffer-Limited Read-Along Session Runtime

**Files:**
- Create: `src/ebook_tts_pipeline/read_along/session.py`
- Test: `tests/test_read_along_session.py`

- [ ] **Step 1: Write failing runtime tests**

Create `tests/test_read_along_session.py`:

```python
import json
from pathlib import Path

import numpy as np

from ebook_tts_pipeline.read_along.session import ReadAlongSession
from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


class RecordingAdapter:
    def __init__(self):
        self.calls = []

    def ensure_voice(self, role_id, voice_record, voice_path):
        return voice_path

    def generate_sentences(self, jobs):
        self.calls.append([dict(job) for job in jobs])
        return [
            GeneratedSentenceAudio(
                sentence_idx=int(job["sentence_idx"]),
                unit_idx=int(job["unit_idx"]),
                role=str(job["role"]),
                speech_type=str(job["type"]),
                samples=np.ones(2400, dtype=np.float32) * 0.05,
                sample_rate=24000,
                voice_config_path=str(job.get("voice_config_path") or ""),
            )
            for job in jobs
        ]


def _unit(unit_id):
    return ReadAlongUnit(
        chapter="chapter_001",
        unit_id=unit_id,
        text=f"Unit {unit_id}.",
        source_start=unit_id * 10,
        source_end=unit_id * 10 + 7,
        role="Narrator",
        role_id="narrator",
        type="narration",
        voice_config_path="voices/narrator.qvp",
    )


def test_session_never_generates_more_than_buffer_limit(tmp_path):
    adapter = RecordingAdapter()
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=1.0,
        generation_mode="fast",
    )

    generated = session.fill_buffer(start_unit_id=0)

    assert len(generated) == 2
    assert len(adapter.calls) == 1
    assert len(adapter.calls[0]) == 2
    assert session.ready_count == 2


def test_session_logs_realtime_factor_and_cleans_temp_audio(tmp_path):
    adapter = RecordingAdapter()
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=2.0,
        generation_mode="precise",
    )

    session.fill_buffer(start_unit_id=0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "session" / "timings.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["unit_ids"] == [0]
    assert rows[0]["playback_speed"] == 2.0
    assert rows[0]["playback_seconds"] == 0.05
    assert rows[0]["realtime_factor"] >= 0
    assert list((tmp_path / "session").glob("*.wav"))

    session.end()

    assert not (tmp_path / "session").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_read_along_session.py -v`

Expected: FAIL because `ReadAlongSession` does not exist.

- [ ] **Step 3: Implement session runtime**

Create `src/ebook_tts_pipeline/read_along/session.py`:

```python
from __future__ import annotations

import json
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio, TtsAdapter


@dataclass(frozen=True)
class BufferedAudio:
    unit_id: int
    audio_path: Path
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
        playback_speed: float = 1.0,
        generation_mode: str = "balanced",
    ) -> None:
        self.session_id = session_id
        self.units = list(units)
        self.tts_adapter = tts_adapter
        self.session_dir = Path(session_dir)
        self.timing_log_path = Path(timing_log_path)
        self.buffer_limit = max(1, int(buffer_limit))
        self.playback_speed = max(0.1, float(playback_speed))
        self.generation_mode = str(generation_mode)
        self._next_unit_id = 0
        self._ready: List[BufferedAudio] = []
        self._ended = False

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    def fill_buffer(self, start_unit_id: int | None = None) -> List[BufferedAudio]:
        if self._ended:
            return []
        if start_unit_id is not None:
            self._next_unit_id = int(start_unit_id)
        open_slots = self.buffer_limit - len(self._ready)
        if open_slots <= 0:
            return []
        batch_size = 1 if self.generation_mode == "precise" else open_slots
        batch_units = self.units[self._next_unit_id:self._next_unit_id + batch_size]
        if not batch_units:
            return []
        return self._generate_units(batch_units)

    def consume_ready(self) -> BufferedAudio | None:
        if not self._ready:
            return None
        item = self._ready.pop(0)
        try:
            item.audio_path.unlink()
        except OSError:
            pass
        return item

    def end(self) -> None:
        self._ended = True
        shutil.rmtree(self.session_dir, ignore_errors=True)
        self._ready.clear()

    def _generate_units(self, units: List[ReadAlongUnit]) -> List[BufferedAudio]:
        self.session_dir.mkdir(parents=True, exist_ok=True)
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
            audio_path = self.session_dir / f"{unit.unit_id:05d}.wav"
            _write_wav(audio_path, item.samples, item.sample_rate)
            buffered.append(
                BufferedAudio(
                    unit_id=unit.unit_id,
                    audio_path=audio_path,
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
    pcm = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16.tobytes())
```

- [ ] **Step 4: Run runtime tests**

Run: `pytest tests/test_read_along_session.py -v`

Expected: PASS.

## Task 5: Controller Read-Along Processing And Settings

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write failing tests for narrator voice settings and book processing**

Append to `tests/test_ui_controller.py`:

```python
def test_controller_saves_read_along_settings_with_narrator_voice_type(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    controller.save_read_along_settings(
        {
            "playback_speed": "1.25",
            "generation_mode": "fast",
            "buffer_limit": "2",
            "narrator_voice_type": "female",
        }
    )

    settings = controller.read_along_settings()
    assert settings["playback_speed"] == 1.25
    assert settings["generation_mode"] == "fast"
    assert settings["buffer_limit"] == 2
    assert settings["narrator_voice_type"] == "female"
```

- [ ] **Step 2: Run test and verify it fails**

Run: `pytest tests/test_ui_controller.py::test_controller_saves_read_along_settings_with_narrator_voice_type -v`

Expected: FAIL because the settings methods do not exist.

- [ ] **Step 3: Implement controller settings**

Add methods to `PrototypeUiController`:

```python
    def read_along_settings(self) -> Dict[str, Any]:
        defaults = {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "narrator_voice_type": "male",
        }
        path = self.book_root / "read_along" / "settings.json"
        if not path.exists():
            return defaults
        payload = read_json(path)
        return {
            "playback_speed": _positive_float(payload.get("playback_speed"), defaults["playback_speed"]),
            "generation_mode": _choice(
                payload.get("generation_mode"),
                {"precise", "balanced", "fast"},
                defaults["generation_mode"],
            ),
            "buffer_limit": min(8, max(1, _positive_int(payload.get("buffer_limit"), defaults["buffer_limit"]))),
            "narrator_voice_type": _choice(
                payload.get("narrator_voice_type"),
                {"male", "female", "current"},
                defaults["narrator_voice_type"],
            ),
        }

    def save_read_along_settings(self, values: Dict[str, Any]) -> None:
        settings = {
            "playback_speed": _positive_float(values.get("playback_speed"), 1.0),
            "generation_mode": _choice(values.get("generation_mode"), {"precise", "balanced", "fast"}, "balanced"),
            "buffer_limit": min(8, max(1, _positive_int(values.get("buffer_limit"), 2))),
            "narrator_voice_type": _choice(values.get("narrator_voice_type"), {"male", "female", "current"}, "male"),
        }
        write_json_atomic(self.book_root / "read_along" / "settings.json", settings)
```

Add helper:

```python
def _choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value).strip().lower()
    return text if text in allowed else default
```

- [ ] **Step 4: Run settings test**

Run: `pytest tests/test_ui_controller.py::test_controller_saves_read_along_settings_with_narrator_voice_type -v`

Expected: PASS.

## Task 6: Tk Read Along Tab With Natural Page Highlighting

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/tk_app.py`

- [ ] **Step 1: Convert main body to a notebook**

In `_build_layout`, replace direct chapter frame placement with a `ttk.Notebook` that has two tabs:

```python
self.tabs = ttk.Notebook(self.main)
self.tabs.grid(row=0, column=1, sticky="nsew")
self.audiobook_tab = ttk.Frame(self.tabs)
self.read_along_tab = ttk.Frame(self.tabs)
self.tabs.add(self.audiobook_tab, text="Audiobook")
self.tabs.add(self.read_along_tab, text="Read Along")
```

Move the existing `PanedWindow`/chapter/registry UI into `self.audiobook_tab`.

- [ ] **Step 2: Add read-along state variables**

In `__init__`, add:

```python
self.read_along_chapter = tk.StringVar()
self.read_along_page = tk.IntVar(value=0)
self.read_along_selected_unit = tk.IntVar(value=0)
self.read_along_playback_speed = tk.StringVar(value="1.0")
self.read_along_generation_mode = tk.StringVar(value="balanced")
self.read_along_buffer_limit = tk.StringVar(value="2")
self.read_along_narrator_voice_type = tk.StringVar(value="male")
self.read_along_session_active = False
self.read_along_units = []
self.read_along_page_offsets = []
```

- [ ] **Step 3: Build natural page reader controls**

Add `_build_read_along_tab` and call it from `_build_layout`:

```python
def _build_read_along_tab(self) -> None:
    self.read_along_tab.columnconfigure(1, weight=1)
    self.read_along_tab.rowconfigure(0, weight=1)

    toc = ttk.Frame(self.read_along_tab, padding=8)
    toc.grid(row=0, column=0, sticky="ns")
    ttk.Label(toc, text="Table of Contents").pack(anchor="w")
    self.read_along_chapter_list = tk.Listbox(toc, width=26, exportselection=False)
    self.read_along_chapter_list.pack(fill="both", expand=True)
    self.read_along_chapter_list.bind("<<ListboxSelect>>", self.select_read_along_chapter)
    ttk.Button(toc, text="Process Book", command=self.process_read_along_book).pack(fill="x", pady=(6, 0))

    reader = ttk.Frame(self.read_along_tab, padding=8)
    reader.grid(row=0, column=1, sticky="nsew")
    reader.columnconfigure(0, weight=1)
    reader.rowconfigure(0, weight=1)
    self.read_along_text = tk.Text(reader, wrap="word", font=("Georgia", 14), padx=42, pady=36)
    self.read_along_text.grid(row=0, column=0, sticky="nsew")
    self.read_along_text.tag_configure("current_unit", background="#fde68a")
    self.read_along_text.tag_configure("queued_unit", background="#dbeafe")
    self.read_along_text.tag_configure("selected_unit", background="#bbf7d0")
    self.read_along_text.bind("<ButtonRelease-1>", self.select_read_along_unit_at_click)

    controls = ttk.Frame(self.read_along_tab, padding=8)
    controls.grid(row=0, column=2, sticky="ns")
    ttk.Label(controls, text="Playback Speed").pack(anchor="w")
    ttk.Entry(controls, textvariable=self.read_along_playback_speed, width=8).pack(anchor="w")
    ttk.Label(controls, text="Generation").pack(anchor="w", pady=(8, 0))
    ttk.Combobox(
        controls,
        textvariable=self.read_along_generation_mode,
        values=["precise", "balanced", "fast"],
        state="readonly",
        width=12,
    ).pack(anchor="w")
    ttk.Label(controls, text="Narrator").pack(anchor="w", pady=(8, 0))
    ttk.Combobox(
        controls,
        textvariable=self.read_along_narrator_voice_type,
        values=["male", "female", "current"],
        state="readonly",
        width=12,
    ).pack(anchor="w")
    ttk.Label(controls, text="Buffer Limit").pack(anchor="w", pady=(8, 0))
    ttk.Entry(controls, textvariable=self.read_along_buffer_limit, width=8).pack(anchor="w")
    ttk.Button(controls, text="Start Here", command=self.start_read_along_session).pack(fill="x", pady=(12, 0))
    ttk.Button(controls, text="End Session", command=self.end_read_along_session).pack(fill="x", pady=(6, 0))
    self.read_along_status = tk.StringVar(value="No read-along session.")
    ttk.Label(controls, textvariable=self.read_along_status, wraplength=210).pack(anchor="w", pady=(12, 0))
```

- [ ] **Step 4: Render page text and tags**

Add methods:

```python
def load_read_along_chapter(self, chapter: str) -> None:
    self._sync_controller()
    self.read_along_chapter.set(chapter)
    self.read_along_units = self.controller.read_along_units(chapter)
    text = self.controller.chapter_text(chapter)
    self.read_along_text.configure(state="normal")
    self.read_along_text.delete("1.0", "end")
    self.read_along_text.insert("1.0", text)
    self.read_along_text.configure(state="disabled")
    self.read_along_status.set(f"Loaded {chapter}. Click text to choose a start point.")

def select_read_along_unit_at_click(self, event=None) -> None:
    if not self.read_along_units:
        return
    index = self.read_along_text.index("current")
    offset = int(self.read_along_text.count("1.0", index, "chars")[0])
    unit = min(
        self.read_along_units,
        key=lambda item: 0 if int(item["source_start"]) <= offset <= int(item["source_end"]) else min(abs(offset - int(item["source_start"])), abs(offset - int(item["source_end"]))),
    )
    self.read_along_selected_unit.set(int(unit["unit_id"]))
    self._tag_read_along_unit(unit, "selected_unit")

def _tag_read_along_unit(self, unit: Dict[str, Any], tag: str) -> None:
    self.read_along_text.configure(state="normal")
    self.read_along_text.tag_remove(tag, "1.0", "end")
    start = f"1.0 + {int(unit['source_start'])} chars"
    end = f"1.0 + {int(unit['source_end'])} chars"
    self.read_along_text.tag_add(tag, start, end)
    self.read_along_text.see(start)
    self.read_along_text.configure(state="disabled")
```

- [ ] **Step 5: Wire chapter list refresh**

Extend `refresh()` to populate `self.read_along_chapter_list` from `self.controller.chapter_rows()` and keep existing audiobook behavior unchanged.

- [ ] **Step 6: Smoke-check UI syntax**

Run: `python -m py_compile src/ebook_tts_pipeline/ui/tk_app.py`

Expected: no syntax errors.

## Task 7: Hook Session Start/End Into UI

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Modify: `src/ebook_tts_pipeline/ui/tk_app.py`

- [ ] **Step 1: Add controller session factory**

Add to `PrototypeUiController`:

```python
    def create_read_along_session(
        self,
        chapter: str,
        units: List[Dict[str, Any]],
        settings: Dict[str, Any],
    ) -> ReadAlongSession:
        pipeline = self._pipeline(needs_llm=False)
        session_id = f"{chapter}-{int(time.time())}"
        return ReadAlongSession(
            session_id=session_id,
            units=[ReadAlongUnit.from_dict(unit) for unit in units],
            tts_adapter=pipeline.tts_adapter,
            session_dir=self.paths.read_along_session_dir(session_id),
            timing_log_path=self.paths.read_along_timing_log(session_id),
            buffer_limit=int(settings["buffer_limit"]),
            playback_speed=float(settings["playback_speed"]),
            generation_mode=str(settings["generation_mode"]),
        )
```

Add imports:

```python
import time
from ebook_tts_pipeline.read_along.session import ReadAlongSession
from ebook_tts_pipeline.read_along.units import ReadAlongUnit
```

- [ ] **Step 2: Add UI start/end behavior**

Add to `PrototypeTkApp`:

```python
def start_read_along_session(self) -> None:
    if self.read_along_session_active:
        return
    chapter = self.read_along_chapter.get().strip()
    if not chapter:
        messagebox.showerror("Read Along", "Choose a chapter first.")
        return
    self.controller.save_read_along_settings(
        {
            "playback_speed": self.read_along_playback_speed.get(),
            "generation_mode": self.read_along_generation_mode.get(),
            "buffer_limit": self.read_along_buffer_limit.get(),
            "narrator_voice_type": self.read_along_narrator_voice_type.get(),
        }
    )
    settings = self.controller.read_along_settings()
    self.read_along_session_active = True
    self.read_along_status.set("Building buffer...")
    selected = int(self.read_along_selected_unit.get())

    def work() -> str:
        session = self.controller.create_read_along_session(chapter, self.read_along_units, settings)
        self.current_read_along_session = session
        generated = session.fill_buffer(start_unit_id=selected)
        return f"Buffered {len(generated)} read-along units."

    self._run_background("Building read-along buffer...", work)

def end_read_along_session(self) -> None:
    session = getattr(self, "current_read_along_session", None)
    if session is not None:
        session.end()
    self.current_read_along_session = None
    self.read_along_session_active = False
    self.read_along_status.set("Read-along session ended.")
```

- [ ] **Step 3: Disable locked controls during active session**

Add helper:

```python
def _set_read_along_controls_locked(self, locked: bool) -> None:
    state = "disabled" if locked else "normal"
    for widget in self.read_along_locked_widgets:
        widget.configure(state=state)
```

Store the playback speed entry, generation combobox, narrator combobox, and buffer entry in `self.read_along_locked_widgets`.

- [ ] **Step 4: Run syntax checks**

Run:

```powershell
python -m py_compile src/ebook_tts_pipeline/ui/controller.py
python -m py_compile src/ebook_tts_pipeline/ui/tk_app.py
```

Expected: no syntax errors.

## Task 8: Focused Test Run And Chapter 15 Validation

**Files:**
- Test only.

- [ ] **Step 1: Run focused automated tests**

Run:

```powershell
pytest tests/test_read_along_units.py tests/test_read_along_session.py tests/test_ui_controller.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
pytest
```

Expected: PASS.

- [ ] **Step 3: Validate chapter 15 artifacts**

Run a small script that loads chapter 15 read-along units and checks exact source offsets:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -c "from ebook_tts_pipeline.ui.controller import PrototypeUiController; c=PrototypeUiController('test false witness real tts chapter 15', fake_tts=True); units=c.build_read_along_units('chapter_015'); text=c.chapter_text('chapter_015'); bad=[u for u in units if text[int(u['source_start']):int(u['source_end'])].strip()!=str(u['text']).strip()]; print({'units': len(units), 'bad_offsets': len(bad), 'first': units[0] if units else None}); assert units and not bad"
```

Expected: prints unit count with `bad_offsets: 0`.

- [ ] **Step 4: Validate buffer/timing on chapter 15 with fake TTS**

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -c "from ebook_tts_pipeline.ui.controller import PrototypeUiController; c=PrototypeUiController('test false witness real tts chapter 15', fake_tts=True); units=c.read_along_units('chapter_015'); s=c.create_read_along_session('chapter_015', units, {'buffer_limit': 2, 'playback_speed': 1.0, 'generation_mode': 'fast'}); out=s.fill_buffer(start_unit_id=0); print({'buffered': len(out), 'ready': s.ready_count, 'log': str(s.timing_log_path)}); assert len(out)==2 and s.ready_count==2; s.end()"
```

Expected: prints `buffered: 2`, `ready: 2`, and exits cleanly after deleting temp audio.

