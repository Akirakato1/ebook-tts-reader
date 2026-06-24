# WSL Qwen Read-Along Speed Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the fastest practical Qwen3-TTS read-along path: persistent WSL FlashAttention generation, in-memory WAV audio, time-based buffering, and repeatable chapter 15 benchmarks.

**Architecture:** Keep the browser/Windows app as the control surface, but move real Qwen generation into a persistent WSL worker that loads the model once and speaks JSONL over stdin/stdout. The read-along session buffers in-memory WAV bytes by playback seconds, not by a fixed number of units: it generates sequential units until unplayed queued audio reaches the configured seconds target, stops generation at that high-water point, and resumes top-up only after playback consumes queued audio. It logs enough timing, role, voice, and VRAM data to tune smooth playback.

**Tech Stack:** Python 3.9+ on Windows for the app, WSL Ubuntu 24.04 with `/opt/ebook-tts-venv`, PyTorch `2.9.1+cu130`, `flash-attn 2.8.3`, Qwen3-TTS, standard-library JSONL subprocess protocol, existing `ThreadingHTTPServer` web UI.

---

## File Structure

- Modify `src/ebook_tts_pipeline/ui/controller.py`: read, validate, save, and pass time-based buffer settings; select native or WSL TTS backend.
- Modify `src/ebook_tts_pipeline/read_along/session.py`: replace count-only buffering with playback-second targets while preserving in-memory WAV bytes.
- Modify `src/ebook_tts_pipeline/ui/web_app.py`: expose buffer seconds in the UI/API, show buffer health, and request enough initial buffer before playback.
- Create `src/ebook_tts_pipeline/tts/wsl_paths.py`: translate Windows paths to WSL `/mnt/<drive>/...` paths and back for logs.
- Create `src/ebook_tts_pipeline/tts/wsl_worker.py`: worker process run inside WSL; owns `QwenTtsAdapter`, receives JSONL commands, returns JSONL audio payloads.
- Create `src/ebook_tts_pipeline/tts/wsl_adapter.py`: Windows-side `TtsAdapter` implementation that starts and talks to the WSL worker.
- Modify `src/ebook_tts_pipeline/config.py`: add environment switches for WSL backend path, distro, venv python, worker timeout, and default read-along buffer seconds.
- Modify `src/ebook_tts_pipeline/cli.py`: add `benchmark-readalong` command for repeatable chapter/unit speed testing.
- Modify `pyproject.toml`: expose any new console entry point only if needed; prefer existing `ebook-tts` CLI for benchmarks.
- Add or modify tests:
  - `tests/test_read_along_session.py`
  - `tests/test_read_along_web_app.py`
  - `tests/test_ui_controller.py`
  - `tests/test_wsl_paths.py`
  - `tests/test_wsl_adapter.py`
  - `tests/test_cli.py`

---

### Task 1: Add Read-Along Time Buffer Settings

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Modify: `tests/test_ui_controller.py`

- [ ] **Step 1: Write the failing controller settings test**

Append this test near `test_controller_saves_read_along_settings_with_narrator_voice_type` in `tests/test_ui_controller.py`:

```python
def test_controller_saves_read_along_time_buffer_settings(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    controller.save_read_along_settings(
        {
            "playback_speed": "1.25",
            "generation_mode": "fast",
            "buffer_limit": "2",
            "target_buffer_seconds": "12.5",
            "start_buffer_seconds": "4",
            "max_buffer_seconds": "20",
            "max_buffer_units": "9",
            "narrator_voice_type": "female",
        }
    )

    settings = controller.read_along_settings()
    assert settings["playback_speed"] == 1.25
    assert settings["generation_mode"] == "fast"
    assert settings["buffer_limit"] == 2
    assert settings["target_buffer_seconds"] == 12.5
    assert settings["start_buffer_seconds"] == 4.0
    assert settings["max_buffer_seconds"] == 20.0
    assert settings["max_buffer_units"] == 9
    assert settings["narrator_voice_type"] == "female"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_saves_read_along_time_buffer_settings -q
```

Expected: FAIL with a missing `target_buffer_seconds` key.

- [ ] **Step 3: Add settings validation**

In `src/ebook_tts_pipeline/ui/controller.py`, add this helper near the existing `_positive_float` helpers:

```python
def _bounded_positive_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    parsed = _positive_float(value, default)
    return min(float(maximum), max(float(minimum), float(parsed)))
```

Then replace the `defaults` dict in `read_along_settings()` with:

```python
defaults = {
    "playback_speed": 1.0,
    "generation_mode": "balanced",
    "buffer_limit": 2,
    "target_buffer_seconds": 10.0,
    "start_buffer_seconds": 4.0,
    "max_buffer_seconds": 20.0,
    "max_buffer_units": 8,
    "narrator_voice_type": "male",
}
```

Extend the returned dict in `read_along_settings()` after `buffer_limit`:

```python
"target_buffer_seconds": _bounded_positive_float(
    payload.get("target_buffer_seconds"),
    defaults["target_buffer_seconds"],
    1.0,
    60.0,
),
"start_buffer_seconds": _bounded_positive_float(
    payload.get("start_buffer_seconds"),
    defaults["start_buffer_seconds"],
    0.5,
    30.0,
),
"max_buffer_seconds": _bounded_positive_float(
    payload.get("max_buffer_seconds"),
    defaults["max_buffer_seconds"],
    2.0,
    90.0,
),
"max_buffer_units": min(
    32,
    max(1, _positive_int(payload.get("max_buffer_units"), defaults["max_buffer_units"])),
),
```

Extend `save_read_along_settings()` similarly:

```python
target_buffer_seconds = _bounded_positive_float(values.get("target_buffer_seconds"), 10.0, 1.0, 60.0)
start_buffer_seconds = _bounded_positive_float(values.get("start_buffer_seconds"), 4.0, 0.5, 30.0)
max_buffer_seconds = _bounded_positive_float(values.get("max_buffer_seconds"), 20.0, 2.0, 90.0)
if start_buffer_seconds > target_buffer_seconds:
    start_buffer_seconds = target_buffer_seconds
if target_buffer_seconds > max_buffer_seconds:
    max_buffer_seconds = target_buffer_seconds
```

Add these keys to the `settings` dict:

```python
"target_buffer_seconds": target_buffer_seconds,
"start_buffer_seconds": start_buffer_seconds,
"max_buffer_seconds": max_buffer_seconds,
"max_buffer_units": min(32, max(1, _positive_int(values.get("max_buffer_units"), 8))),
```

- [ ] **Step 4: Pass settings to `ReadAlongSession`**

In `PrototypeUiController.create_read_along_session()`, add these keyword arguments to the `ReadAlongSession(...)` call:

```python
target_buffer_seconds=float(settings["target_buffer_seconds"]),
start_buffer_seconds=float(settings["start_buffer_seconds"]),
max_buffer_seconds=float(settings["max_buffer_seconds"]),
max_buffer_units=int(settings["max_buffer_units"]),
```

- [ ] **Step 5: Run the focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_saves_read_along_time_buffer_settings tests\test_ui_controller.py::test_controller_saves_read_along_settings_with_narrator_voice_type -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src\ebook_tts_pipeline\ui\controller.py tests\test_ui_controller.py
git commit -m "feat: add read-along time buffer settings"
```

---

### Task 2: Convert ReadAlongSession To Playback-Seconds Buffering

**Files:**
- Modify: `src/ebook_tts_pipeline/read_along/session.py`
- Modify: `tests/test_read_along_session.py`

- [ ] **Step 1: Write failing time-buffer tests**

Add these tests to `tests/test_read_along_session.py`:

```python
class VariableDurationAdapter(RecordingAdapter):
    def __init__(self, sample_counts):
        super().__init__()
        self.sample_counts = list(sample_counts)

    def generate_sentences(self, jobs):
        self.calls.append([dict(job) for job in jobs])
        result = []
        for job in jobs:
            sample_count = self.sample_counts[int(job["unit_idx"])]
            result.append(
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    unit_idx=int(job["unit_idx"]),
                    role=str(job["role"]),
                    speech_type=str(job["type"]),
                    samples=np.ones(sample_count, dtype=np.float32) * 0.05,
                    sample_rate=24000,
                    voice_config_path=str(job.get("voice_config_path") or ""),
                )
            )
        return result


def test_session_fills_until_target_buffer_seconds(tmp_path):
    adapter = VariableDurationAdapter([24000, 24000, 24000, 24000])
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2), _unit(3)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=1.0,
        generation_mode="fast",
        target_buffer_seconds=3.0,
        start_buffer_seconds=2.0,
        max_buffer_seconds=4.0,
        max_buffer_units=8,
    )

    generated = session.fill_buffer(start_unit_id=0, min_buffer_seconds=3.0)

    assert [item.unit_id for item in generated] == [0, 1, 2]
    assert session.ready_playback_seconds == 3.0
    assert len(adapter.calls) == 2
    assert [len(call) for call in adapter.calls] == [2, 1]


def test_session_does_not_generate_when_buffer_seconds_are_full(tmp_path):
    adapter = VariableDurationAdapter([24000, 24000, 24000])
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=1.0,
        generation_mode="fast",
        target_buffer_seconds=2.0,
        start_buffer_seconds=1.0,
        max_buffer_seconds=2.0,
        max_buffer_units=8,
    )

    session.fill_buffer(start_unit_id=0, min_buffer_seconds=2.0)
    second = session.fill_buffer()

    assert second == []
    assert session.ready_playback_seconds == 2.0
    assert len(adapter.calls) == 1


def test_session_tops_up_after_playback_consumes_buffer(tmp_path):
    adapter = VariableDurationAdapter([24000, 24000, 24000, 24000])
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2), _unit(3)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=1.0,
        generation_mode="fast",
        target_buffer_seconds=2.0,
        start_buffer_seconds=1.0,
        max_buffer_seconds=3.0,
        max_buffer_units=8,
    )

    session.fill_buffer(start_unit_id=0, min_buffer_seconds=2.0)
    consumed = session.consume_ready()
    generated = session.fill_buffer()

    assert consumed is not None
    assert consumed.unit_id == 0
    assert [item.unit_id for item in generated] == [2]
    assert session.ready_playback_seconds == 2.0
    assert session.ready_unit_ids == [1, 2]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_session.py::test_session_fills_until_target_buffer_seconds tests\test_read_along_session.py::test_session_does_not_generate_when_buffer_seconds_are_full tests\test_read_along_session.py::test_session_tops_up_after_playback_consumes_buffer -q
```

Expected: FAIL because `ReadAlongSession` does not accept time-buffer constructor arguments.

- [ ] **Step 3: Add constructor arguments and property**

In `ReadAlongSession.__init__`, add parameters after `buffer_limit`:

```python
target_buffer_seconds: float = 10.0,
start_buffer_seconds: float = 4.0,
max_buffer_seconds: float = 20.0,
max_buffer_units: int = 8,
```

Set them after `self.buffer_limit`:

```python
self.target_buffer_seconds = max(0.1, float(target_buffer_seconds))
self.start_buffer_seconds = max(0.1, min(float(start_buffer_seconds), self.target_buffer_seconds))
self.max_buffer_seconds = max(self.target_buffer_seconds, float(max_buffer_seconds))
self.max_buffer_units = max(1, int(max_buffer_units))
```

Add this property below `ready_count`:

```python
@property
def ready_playback_seconds(self) -> float:
    return sum(item.playback_seconds for item in self._ready)
```

- [ ] **Step 4: Replace `fill_buffer` with seconds-aware logic**

Replace the existing `fill_buffer()` method with:

```python
def fill_buffer(
    self,
    start_unit_id: Optional[int] = None,
    min_buffer_seconds: Optional[float] = None,
) -> List[BufferedAudio]:
    if self._ended:
        return []
    if start_unit_id is not None:
        self._next_unit_id = int(start_unit_id)
    target_seconds = float(min_buffer_seconds) if min_buffer_seconds is not None else self.target_buffer_seconds
    target_seconds = min(self.max_buffer_seconds, max(0.1, target_seconds))
    generated: List[BufferedAudio] = []
    while (
        self.ready_playback_seconds < target_seconds
        and len(self._ready) < self.max_buffer_units
        and self._next_unit_id < len(self.units)
    ):
        open_unit_slots = max(1, min(self.max_buffer_units - len(self._ready), self.buffer_limit))
        batch_size = self._next_batch_size(open_unit_slots)
        batch_units = self.units[self._next_unit_id:self._next_unit_id + batch_size]
        if not batch_units:
            break
        generated.extend(self._generate_units(batch_units))
        if self.ready_playback_seconds >= self.max_buffer_seconds:
            break
    return generated
```

Important invariant: `ready_playback_seconds` counts only unplayed queued audio in `self._ready`. Once `consume_ready()` pops the current audio for playback, that unit is no longer counted as future buffer, so the next `fill_buffer()` call may generate more audio until `ready_playback_seconds >= target_seconds` again.

Add this helper below `fill_buffer()`:

```python
def _next_batch_size(self, open_unit_slots: int) -> int:
    if self.generation_mode == "precise":
        return 1
    if self.generation_mode == "balanced":
        return max(1, min(2, open_unit_slots))
    return max(1, open_unit_slots)
```

- [ ] **Step 5: Extend timing rows**

At the start of `_generate_units()`, after `self.session_dir.mkdir(...)`, add:

```python
buffer_seconds_before = self.ready_playback_seconds
```

In `_append_timing(...)`, add these keys:

```python
"target_buffer_seconds": self.target_buffer_seconds,
"start_buffer_seconds": self.start_buffer_seconds,
"max_buffer_seconds": self.max_buffer_seconds,
"max_buffer_units": self.max_buffer_units,
"buffer_seconds_before_generation": buffer_seconds_before,
"buffer_seconds_after_generation": buffer_seconds_before + playback_seconds,
"unit_audio_seconds": [len(by_unit[unit.unit_id].samples) / by_unit[unit.unit_id].sample_rate for unit in units],
```

- [ ] **Step 6: Run read-along session tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_session.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src\ebook_tts_pipeline\read_along\session.py tests\test_read_along_session.py
git commit -m "feat: buffer read-along audio by playback seconds"
```

---

### Task 3: Update Web UI And API For Time-Based WAV Buffering

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Modify: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write failing web API test**

Update `test_web_api_serves_chapter_and_bounded_session_audio` in `tests/test_read_along_web_app.py` so the start payload includes time settings:

```python
"settings": {
    "playback_speed": 1.25,
    "generation_mode": "balanced",
    "buffer_limit": 2,
    "target_buffer_seconds": 0.16,
    "start_buffer_seconds": 0.16,
    "max_buffer_seconds": 0.32,
    "max_buffer_units": 4,
    "narrator_voice_type": "current",
},
```

Change the assertion after `started` to:

```python
assert started["ready_playback_seconds"] >= 0.16
assert started["target_buffer_seconds"] == 0.16
```

Update `test_web_api_saves_read_along_settings` expected settings:

```python
assert saved["settings"] == {
    "playback_speed": 1.4,
    "generation_mode": "fast",
    "buffer_limit": 3,
    "target_buffer_seconds": 10.0,
    "start_buffer_seconds": 4.0,
    "max_buffer_seconds": 20.0,
    "max_buffer_units": 8,
    "narrator_voice_type": "female",
}
```

- [ ] **Step 2: Run focused web tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_web_api_serves_chapter_and_bounded_session_audio tests\test_read_along_web_app.py::test_web_api_saves_read_along_settings -q
```

Expected: FAIL because payload fields are not returned and UI state ignores new settings.

- [ ] **Step 3: Return buffer health from API**

In `ReadAlongWebState.start_session()`, change:

```python
self.session.fill_buffer(start_unit_id=start_unit_id)
```

to:

```python
self.session.fill_buffer(
    start_unit_id=start_unit_id,
    min_buffer_seconds=self.session.start_buffer_seconds,
)
```

Add these fields to the returned dict:

```python
"ready_playback_seconds": self.session.ready_playback_seconds,
"target_buffer_seconds": self.session.target_buffer_seconds,
"max_buffer_seconds": self.session.max_buffer_seconds,
```

In `advance_session()`, add the same three fields to the returned dict.

In `_ready_payload()`, add:

```python
"ready_index": index,
```

by changing the comprehension to `for index, item in enumerate(session.ready_items)`.

- [ ] **Step 4: Update web controls**

In `INDEX_HTML`, replace the existing buffer label:

```html
<label>Buffer <input id="buffer" type="number" min="1" max="8" step="1"></label>
```

with:

```html
<label>Units <input id="buffer" type="number" min="1" max="8" step="1"></label>
<label>Buffer s <input id="target-buffer" type="number" min="1" max="60" step="0.5"></label>
<label>Start s <input id="start-buffer" type="number" min="0.5" max="30" step="0.5"></label>
<label>Max units <input id="max-units" type="number" min="1" max="32" step="1"></label>
```

Add elements to the `els` object:

```javascript
targetBuffer: document.getElementById("target-buffer"),
startBuffer: document.getElementById("start-buffer"),
maxUnits: document.getElementById("max-units"),
```

Add settings keys in `settings()`:

```javascript
target_buffer_seconds: Number(els.targetBuffer.value || 10),
start_buffer_seconds: Number(els.startBuffer.value || 4),
max_buffer_seconds: Math.max(Number(els.targetBuffer.value || 10), Number(els.targetBuffer.value || 10) * 2),
max_buffer_units: Number(els.maxUnits.value || 8),
```

Update `lockControls()` to include the three new inputs.

Update `loadState()` and save handling to populate:

```javascript
els.targetBuffer.value = payload.settings.target_buffer_seconds;
els.startBuffer.value = payload.settings.start_buffer_seconds;
els.maxUnits.value = payload.settings.max_buffer_units;
```

In `startSession()` after assigning `state.ready`, set:

```javascript
setStatus("Buffered " + payload.ready_playback_seconds.toFixed(1) + "s. Starting playback...");
```

In `advanceSession()` after assigning `state.ready`, set:

```javascript
if (typeof payload.ready_playback_seconds === "number") {
  setStatus("Buffered " + payload.ready_playback_seconds.toFixed(1) + "s");
}
```

- [ ] **Step 5: Run web tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src\ebook_tts_pipeline\ui\web_app.py tests\test_read_along_web_app.py
git commit -m "feat: expose read-along time buffer controls"
```

---

### Task 4: Add WSL Path Translation

**Files:**
- Create: `src/ebook_tts_pipeline/tts/wsl_paths.py`
- Create: `tests/test_wsl_paths.py`

- [ ] **Step 1: Write failing path tests**

Create `tests/test_wsl_paths.py`:

```python
from pathlib import Path

from ebook_tts_pipeline.tts.wsl_paths import to_wsl_path, translate_job_paths


def test_to_wsl_path_translates_windows_drive_path():
    assert (
        to_wsl_path(Path(r"C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\models\qwen-tts"))
        == "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/models/qwen-tts"
    )


def test_to_wsl_path_leaves_relative_path_posix():
    assert to_wsl_path(Path("voices/narrator.qvp")) == "voices/narrator.qvp"


def test_translate_job_paths_resolves_voice_path_against_book_root():
    jobs = [
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "Hello.",
            "voice_config_path": "voices/narrator.qvp",
        }
    ]

    translated = translate_job_paths(
        jobs,
        book_root=Path(r"C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\test book"),
    )

    assert translated[0]["voice_config_path"] == "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/test book/voices/narrator.qvp"
```

- [ ] **Step 2: Run path tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_wsl_paths.py -q
```

Expected: FAIL because `ebook_tts_pipeline.tts.wsl_paths` does not exist.

- [ ] **Step 3: Implement path translation**

Create `src/ebook_tts_pipeline/tts/wsl_paths.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def to_wsl_path(path: Path | str) -> str:
    raw = str(path)
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return raw.replace("\\", "/")


def translate_job_paths(jobs: Iterable[Dict[str, Any]], book_root: Path | str) -> List[Dict[str, Any]]:
    root = Path(book_root)
    translated: List[Dict[str, Any]] = []
    for job in jobs:
        item = dict(job)
        voice_path = str(item.get("voice_config_path") or "").strip()
        if voice_path:
            path = Path(voice_path)
            if not path.is_absolute():
                path = root / path
            item["voice_config_path"] = to_wsl_path(path)
        translated.append(item)
    return translated
```

- [ ] **Step 4: Run path tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_wsl_paths.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\ebook_tts_pipeline\tts\wsl_paths.py tests\test_wsl_paths.py
git commit -m "feat: add WSL path translation for TTS jobs"
```

---

### Task 5: Add The WSL Qwen Worker Process

**Files:**
- Create: `src/ebook_tts_pipeline/tts/wsl_worker.py`
- Create: `tests/test_wsl_worker_protocol.py`

- [ ] **Step 1: Write protocol serialization tests**

Create `tests/test_wsl_worker_protocol.py`:

```python
import base64

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.wsl_worker import decode_audio_item, encode_audio_item


def test_worker_audio_item_round_trips_float32_samples():
    item = GeneratedSentenceAudio(
        sentence_idx=7,
        unit_idx=3,
        role="Narrator",
        speech_type="narration",
        samples=np.array([0.0, 0.25, -0.25], dtype=np.float32),
        sample_rate=24000,
        voice_config_path="/mnt/c/book/voices/narrator.qvp",
    )

    encoded = encode_audio_item(item)

    assert encoded["sentence_idx"] == 7
    assert encoded["unit_idx"] == 3
    assert encoded["dtype"] == "float32"
    assert encoded["shape"] == [3]
    assert base64.b64decode(encoded["samples_b64"])

    decoded = decode_audio_item(encoded)
    assert decoded.sentence_idx == 7
    assert decoded.unit_idx == 3
    assert decoded.role == "Narrator"
    assert decoded.speech_type == "narration"
    assert decoded.sample_rate == 24000
    assert decoded.voice_config_path == "/mnt/c/book/voices/narrator.qvp"
    np.testing.assert_allclose(decoded.samples, item.samples)
```

- [ ] **Step 2: Run protocol test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_wsl_worker_protocol.py -q
```

Expected: FAIL because `wsl_worker.py` does not exist.

- [ ] **Step 3: Implement worker protocol helpers and main loop**

Create `src/ebook_tts_pipeline/tts/wsl_worker.py`:

```python
from __future__ import annotations

import base64
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter


def encode_audio_item(item: GeneratedSentenceAudio) -> Dict[str, Any]:
    samples = np.asarray(item.samples, dtype=np.float32)
    return {
        "sentence_idx": item.sentence_idx,
        "unit_idx": item.unit_idx,
        "role": item.role,
        "speech_type": item.speech_type,
        "sample_rate": item.sample_rate,
        "voice_config_path": item.voice_config_path,
        "pause_after_ms": item.pause_after_ms,
        "dtype": "float32",
        "shape": list(samples.shape),
        "samples_b64": base64.b64encode(samples.tobytes()).decode("ascii"),
    }


def decode_audio_item(payload: Dict[str, Any]) -> GeneratedSentenceAudio:
    samples = np.frombuffer(base64.b64decode(payload["samples_b64"]), dtype=np.float32).copy()
    samples = samples.reshape(tuple(payload.get("shape") or [len(samples)]))
    return GeneratedSentenceAudio(
        sentence_idx=int(payload["sentence_idx"]),
        unit_idx=int(payload["unit_idx"]) if payload.get("unit_idx") is not None else None,
        role=str(payload["role"]),
        speech_type=str(payload["speech_type"]),
        samples=samples,
        sample_rate=int(payload["sample_rate"]),
        pause_after_ms=payload.get("pause_after_ms"),
        voice_config_path=payload.get("voice_config_path"),
    )


class WorkerState:
    def __init__(self) -> None:
        self.adapter: Optional[QwenTtsAdapter] = None

    def init(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.adapter = QwenTtsAdapter(
            model_root=str(payload["model_root"]),
            model_choice=str(payload.get("model_choice", "1.7B")),
            device=str(payload.get("device", "cuda")),
            precision=str(payload.get("precision", "bf16")),
            attention=str(payload.get("attention", "auto")),
            max_new_tokens=int(payload.get("max_new_tokens", 2048)),
            max_generation_block_chars=int(payload.get("max_generation_block_chars", 0)),
            max_generation_blocks_per_call=int(payload.get("max_generation_blocks_per_call", 0)),
            cache_clear_interval=int(payload.get("cache_clear_interval", 8)),
            streaming_text_mode=bool(payload.get("streaming_text_mode", True)),
            performance_log_path=Path(payload["performance_log_path"]) if payload.get("performance_log_path") else None,
            adaptive_memory_target_bytes=(
                int(payload["adaptive_memory_target_bytes"])
                if payload.get("adaptive_memory_target_bytes") is not None
                else None
            ),
        )
        return {"initialized": True}

    def ensure_voice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        adapter = self._require_adapter()
        path = adapter.ensure_voice(
            role_id=str(payload["role_id"]),
            voice_record=dict(payload["voice_record"]),
            voice_path=Path(payload["voice_path"]),
        )
        return {"voice_path": str(path)}

    def generate_sentences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        adapter = self._require_adapter()
        jobs = []
        role_voice_paths = {}
        for job in payload["jobs"]:
            item = dict(job)
            voice_path = str(item.pop("voice_config_path", "") or "")
            role_id = str(item.get("role_id", ""))
            role = str(item.get("role", ""))
            if voice_path:
                path = Path(voice_path)
                if role_id:
                    role_voice_paths[role_id] = path
                if role:
                    role_voice_paths[role] = path
            jobs.append(item)
        adapter.role_voice_paths.update(role_voice_paths)
        return {"items": [encode_audio_item(item) for item in adapter.generate_sentences(jobs)]}

    def shutdown(self) -> Dict[str, Any]:
        self.adapter = None
        return {"shutdown": True}

    def _require_adapter(self) -> QwenTtsAdapter:
        if self.adapter is None:
            raise RuntimeError("Worker has not been initialized.")
        return self.adapter


def handle_command(state: WorkerState, command: Dict[str, Any]) -> Dict[str, Any]:
    name = str(command.get("command", ""))
    payload = dict(command.get("payload", {}))
    if name == "init":
        return state.init(payload)
    if name == "ensure_voice":
        return state.ensure_voice(payload)
    if name == "generate_sentences":
        return state.generate_sentences(payload)
    if name == "shutdown":
        return state.shutdown()
    raise ValueError(f"Unknown worker command: {name}")


def main() -> int:
    state = WorkerState()
    for line in sys.stdin:
        try:
            command = json.loads(line)
            response = {
                "id": command.get("id"),
                "ok": True,
                "payload": handle_command(state, command),
            }
        except Exception as exc:
            response = {
                "id": command.get("id") if isinstance(command, dict) else None,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if isinstance(response.get("payload"), dict) and response["payload"].get("shutdown"):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run protocol tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_wsl_worker_protocol.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify worker imports inside WSL**

Run:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "cd '/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader' && /opt/ebook-tts-venv/bin/python -m ebook_tts_pipeline.tts.wsl_worker < /dev/null"
```

Expected: exit code 0 with no output.

- [ ] **Step 6: Commit**

```powershell
git add src\ebook_tts_pipeline\tts\wsl_worker.py tests\test_wsl_worker_protocol.py
git commit -m "feat: add Qwen WSL worker protocol"
```

---

### Task 6: Add Windows-Side WSL Worker Adapter

**Files:**
- Create: `src/ebook_tts_pipeline/tts/wsl_adapter.py`
- Modify: `src/ebook_tts_pipeline/config.py`
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Modify: `src/ebook_tts_pipeline/cli.py`
- Create: `tests/test_wsl_adapter.py`
- Modify: `tests/test_public_import_and_config.py`

- [ ] **Step 1: Write config test**

In `tests/test_public_import_and_config.py`, extend the environment config test with:

```python
monkeypatch.setenv("EBOOK_TTS_BACKEND", "wsl")
monkeypatch.setenv("EBOOK_TTS_WSL_DISTRO", "Ubuntu-24.04")
monkeypatch.setenv("EBOOK_TTS_WSL_PYTHON", "/opt/ebook-tts-venv/bin/python")
monkeypatch.setenv("EBOOK_TTS_WSL_TIMEOUT_SECONDS", "600")
```

Add assertions:

```python
assert config.tts_backend == "wsl"
assert config.wsl_distro == "Ubuntu-24.04"
assert config.wsl_python == "/opt/ebook-tts-venv/bin/python"
assert config.wsl_timeout_seconds == 600.0
```

- [ ] **Step 2: Write adapter command test**

Create `tests/test_wsl_adapter.py`:

```python
from pathlib import Path

from ebook_tts_pipeline.tts.wsl_adapter import WslQwenWorkerAdapter


def test_wsl_adapter_builds_worker_command():
    adapter = WslQwenWorkerAdapter(
        book_root=Path(r"C:\book"),
        model_root=Path(r"C:\book\models\qwen-tts"),
        distro="Ubuntu-24.04",
        python_path="/opt/ebook-tts-venv/bin/python",
        start_process=False,
    )

    assert adapter.worker_command == [
        "wsl.exe",
        "-d",
        "Ubuntu-24.04",
        "-u",
        "root",
        "--",
        "/opt/ebook-tts-venv/bin/python",
        "-m",
        "ebook_tts_pipeline.tts.wsl_worker",
    ]
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_wsl_adapter.py tests\test_public_import_and_config.py -q
```

Expected: FAIL because config fields and adapter do not exist.

- [ ] **Step 4: Add config fields**

In `src/ebook_tts_pipeline/config.py`, add fields to `PipelineConfig`:

```python
tts_backend: str = "native"
wsl_distro: str = "Ubuntu-24.04"
wsl_python: str = "/opt/ebook-tts-venv/bin/python"
wsl_timeout_seconds: float = 600.0
```

Add parsing in `from_env(...)`:

```python
tts_backend=os.environ.get("EBOOK_TTS_BACKEND", "native"),
wsl_distro=os.environ.get("EBOOK_TTS_WSL_DISTRO", "Ubuntu-24.04"),
wsl_python=os.environ.get("EBOOK_TTS_WSL_PYTHON", "/opt/ebook-tts-venv/bin/python"),
wsl_timeout_seconds=float(os.environ.get("EBOOK_TTS_WSL_TIMEOUT_SECONDS", "600")),
```

- [ ] **Step 5: Implement `WslQwenWorkerAdapter`**

Create `src/ebook_tts_pipeline/tts/wsl_adapter.py`:

```python
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.wsl_paths import to_wsl_path, translate_job_paths
from ebook_tts_pipeline.tts.wsl_worker import decode_audio_item


class WslQwenWorkerAdapter:
    def __init__(
        self,
        book_root: Path | str,
        model_root: Path | str,
        distro: str = "Ubuntu-24.04",
        python_path: str = "/opt/ebook-tts-venv/bin/python",
        model_choice: str = "1.7B",
        device: str = "cuda",
        precision: str = "bf16",
        attention: str = "auto",
        max_new_tokens: int = 2048,
        max_generation_block_chars: int = 0,
        max_generation_blocks_per_call: int = 0,
        cache_clear_interval: int = 8,
        streaming_text_mode: bool = True,
        performance_log_path: Optional[Path] = None,
        adaptive_memory_target_bytes: Optional[int] = None,
        timeout_seconds: float = 600.0,
        start_process: bool = True,
    ) -> None:
        self.book_root = Path(book_root)
        self.model_root = Path(model_root)
        self.distro = str(distro)
        self.python_path = str(python_path)
        self.timeout_seconds = float(timeout_seconds)
        self.role_voice_paths: Dict[str, Path] = {}
        self._lock = threading.RLock()
        self._next_id = 0
        self._process: Optional[subprocess.Popen[str]] = None
        self.worker_command = [
            "wsl.exe",
            "-d",
            self.distro,
            "-u",
            "root",
            "--",
            self.python_path,
            "-m",
            "ebook_tts_pipeline.tts.wsl_worker",
        ]
        self._init_payload = {
            "model_root": to_wsl_path(self.model_root),
            "model_choice": model_choice,
            "device": device,
            "precision": precision,
            "attention": attention,
            "max_new_tokens": int(max_new_tokens),
            "max_generation_block_chars": int(max_generation_block_chars),
            "max_generation_blocks_per_call": int(max_generation_blocks_per_call),
            "cache_clear_interval": int(cache_clear_interval),
            "streaming_text_mode": bool(streaming_text_mode),
            "performance_log_path": to_wsl_path(performance_log_path) if performance_log_path else None,
            "adaptive_memory_target_bytes": adaptive_memory_target_bytes,
        }
        if start_process:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                return
            self._process = subprocess.Popen(
                self.worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            self._request("init", self._init_payload)

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                return
            try:
                self._request("shutdown", {})
            finally:
                self._process.terminate()
                self._process = None

    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        self.start()
        self.role_voice_paths.setdefault(role_id, voice_path)
        payload = {
            "role_id": role_id,
            "voice_record": dict(voice_record),
            "voice_path": to_wsl_path(voice_path),
        }
        self._request("ensure_voice", payload)
        return voice_path

    def generate_sentence_batches(self, jobs: List[Dict]):
        yield self.generate_sentences(jobs)

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        self.start()
        translated = translate_job_paths(jobs, self.book_root)
        payload = self._request("generate_sentences", {"jobs": translated})
        return [decode_audio_item(item) for item in payload["items"]]

    def _request(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("WSL worker is not running.")
        request_id = self._next_id
        self._next_id += 1
        request = {"id": request_id, "command": command, "payload": payload}
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr is not None else ""
            raise RuntimeError(f"WSL worker stopped before responding. stderr={stderr}")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError(f"Unexpected WSL worker response id: {response.get('id')}")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "WSL worker failed"))
        return dict(response.get("payload") or {})
```

- [ ] **Step 6: Wire backend selection into CLI and UI controller**

In both `_build_qwen_adapter` functions in `src/ebook_tts_pipeline/cli.py` and `src/ebook_tts_pipeline/ui/controller.py`, import:

```python
from ebook_tts_pipeline.tts.wsl_adapter import WslQwenWorkerAdapter
```

At the start of `_build_qwen_adapter(config)`, add:

```python
if config.tts_backend == "wsl":
    return WslQwenWorkerAdapter(
        book_root=config.book_root,
        model_root=config.qwen_model_root,
        distro=config.wsl_distro,
        python_path=config.wsl_python,
        model_choice=config.qwen_model_choice,
        device="cuda" if config.qwen_device == "auto" else config.qwen_device,
        precision=config.qwen_precision,
        attention=config.qwen_attention,
        max_new_tokens=config.qwen_max_new_tokens,
        max_generation_block_chars=config.qwen_max_generation_block_chars,
        max_generation_blocks_per_call=config.qwen_max_generation_blocks_per_call,
        cache_clear_interval=config.qwen_cache_clear_interval,
        streaming_text_mode=config.qwen_streaming_text_mode,
        performance_log_path=Path(config.qwen_perf_log_path) if config.qwen_perf_log_path else None,
        adaptive_memory_target_bytes=(
            int(config.qwen_adaptive_memory_target_gb * (1024 ** 3))
            if config.qwen_adaptive_memory_target_gb is not None
            else None
        ),
        timeout_seconds=config.wsl_timeout_seconds,
    )
```

Also add `from pathlib import Path` to files that do not already import it.

- [ ] **Step 7: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_wsl_adapter.py tests\test_public_import_and_config.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src\ebook_tts_pipeline\tts\wsl_adapter.py src\ebook_tts_pipeline\config.py src\ebook_tts_pipeline\ui\controller.py src\ebook_tts_pipeline\cli.py tests\test_wsl_adapter.py tests\test_public_import_and_config.py
git commit -m "feat: add persistent WSL Qwen backend"
```

---

### Task 7: Add Repeatable Read-Along Benchmark CLI

**Files:**
- Modify: `src/ebook_tts_pipeline/cli.py`
- Create: `tests/test_cli_benchmark_readalong.py`

- [ ] **Step 1: Write parser test**

Create `tests/test_cli_benchmark_readalong.py`:

```python
from ebook_tts_pipeline.cli import build_parser


def test_cli_accepts_benchmark_readalong_command():
    args = build_parser().parse_args(
        [
            "benchmark-readalong",
            "--book-root",
            "book",
            "--chapter",
            "chapter_015",
            "--start-unit",
            "0",
            "--unit-count",
            "5",
            "--target-buffer-seconds",
            "10",
        ]
    )

    assert args.command == "benchmark-readalong"
    assert args.book_root == "book"
    assert args.chapter == "chapter_015"
    assert args.start_unit == 0
    assert args.unit_count == 5
    assert args.target_buffer_seconds == 10.0
```

- [ ] **Step 2: Run parser test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli_benchmark_readalong.py -q
```

Expected: FAIL because the command is absent.

- [ ] **Step 3: Add CLI parser command**

In `build_parser()` in `src/ebook_tts_pipeline/cli.py`, add:

```python
benchmark_readalong = subparsers.add_parser("benchmark-readalong")
benchmark_readalong.add_argument("--book-root", required=True)
benchmark_readalong.add_argument("--chapter", required=True)
benchmark_readalong.add_argument("--start-unit", type=int, default=0)
benchmark_readalong.add_argument("--unit-count", type=int, default=20)
benchmark_readalong.add_argument("--target-buffer-seconds", type=float, default=10.0)
benchmark_readalong.add_argument("--playback-speed", type=float, default=1.0)
benchmark_readalong.add_argument("--generation-mode", choices=["precise", "balanced", "fast"], default="balanced")
```

- [ ] **Step 4: Add benchmark implementation**

Add imports:

```python
import json
import time
from pathlib import Path

from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.read_along.session import ReadAlongSession
from ebook_tts_pipeline.read_along.units import ReadAlongUnit
```

In `main()`, before `parser.error(...)`, add:

```python
if args.command == "benchmark-readalong":
    config = PipelineConfig.from_env(book_root=args.book_root)
    pipeline = _build_pipeline(config, needs_llm=False, fake_tts=False)
    annotation = _load_annotation(pipeline, args.chapter)
    pipeline.prepare_voices_for_annotation(annotation, chapter=args.chapter)
    units_payload = pipeline.build_read_along_units(args.chapter)["units"]
    selected = [
        ReadAlongUnit.from_dict(unit)
        for unit in units_payload
        if int(unit["unit_id"]) >= int(args.start_unit)
    ][: int(args.unit_count)]
    session_id = f"benchmark-{args.chapter}-{int(time.time())}"
    paths = BookPaths(args.book_root)
    session = ReadAlongSession(
        session_id=session_id,
        units=selected,
        tts_adapter=pipeline.tts_adapter,
        session_dir=paths.read_along_session_dir(session_id),
        timing_log_path=paths.read_along_timing_log(session_id),
        buffer_limit=2,
        playback_speed=float(args.playback_speed),
        generation_mode=str(args.generation_mode),
        target_buffer_seconds=float(args.target_buffer_seconds),
        start_buffer_seconds=float(args.target_buffer_seconds),
        max_buffer_seconds=float(args.target_buffer_seconds) * 2,
        max_buffer_units=max(1, int(args.unit_count)),
        store_audio_files=False,
    )
    try:
        generated = session.fill_buffer(start_unit_id=0, min_buffer_seconds=float(args.target_buffer_seconds))
        summary = {
            "session_id": session_id,
            "chapter": args.chapter,
            "generated_units": [item.unit_id for item in generated],
            "ready_playback_seconds": session.ready_playback_seconds,
            "timing_log_path": str(paths.read_along_timing_log(session_id)),
        }
        print(json.dumps(summary, sort_keys=True))
    finally:
        session.end()
        close = getattr(pipeline.tts_adapter, "close", None)
        if callable(close):
            close()
    return 0
```

- [ ] **Step 5: Run parser test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli_benchmark_readalong.py -q
```

Expected: PASS.

- [ ] **Step 6: Run CLI help**

Run:

```powershell
.\.venv\Scripts\ebook-tts.exe benchmark-readalong --help
```

Expected: Shows benchmark options.

- [ ] **Step 7: Commit**

```powershell
git add src\ebook_tts_pipeline\cli.py tests\test_cli_benchmark_readalong.py
git commit -m "feat: add read-along benchmark command"
```

---

### Task 8: Run Chapter 15 WSL FlashAttention Benchmark

**Files:**
- No source files changed in this task unless a benchmark exposes a real bug.
- Output logs under: `test false witness real tts chapter 15/read_along_sessions/...`
- Output Qwen perf log: `test false witness real tts chapter 15/logs/qwen_wsl_flash_perf.jsonl`

- [ ] **Step 1: Run a 3-unit smoke benchmark**

Run:

```powershell
$env:EBOOK_TTS_BACKEND="wsl"
$env:EBOOK_TTS_QWEN_MODEL_ROOT="C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\models\qwen-tts"
$env:EBOOK_TTS_QWEN_ATTENTION="auto"
$env:EBOOK_TTS_QWEN_PRECISION="bf16"
$env:EBOOK_TTS_QWEN_PERF_LOG="C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\test false witness real tts chapter 15\logs\qwen_wsl_flash_perf.jsonl"
$env:EBOOK_TTS_QWEN_ADAPTIVE_TARGET_GB="13"
.\.venv\Scripts\ebook-tts.exe benchmark-readalong --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-unit 0 --unit-count 3 --target-buffer-seconds 4 --generation-mode balanced
```

Expected: JSON summary with `ready_playback_seconds` greater than 0 and a timing log path.

- [ ] **Step 2: Inspect benchmark timing**

Run:

```powershell
Get-ChildItem "test false witness real tts chapter 15\read_along_sessions" -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  ForEach-Object { Get-Content (Join-Path $_.FullName "timings.jsonl") }
```

Expected: Each JSON row has `generation_seconds`, `playback_seconds`, `realtime_factor`, and buffer-second fields.

- [ ] **Step 3: Inspect Qwen performance log**

Run:

```powershell
Get-Content "test false witness real tts chapter 15\logs\qwen_wsl_flash_perf.jsonl" -Tail 3
```

Expected: JSON rows include `cuda_after`, `elapsed_seconds`, `audio_seconds`, and `voice_config_paths`.

- [ ] **Step 4: Run a larger 20-unit benchmark**

Run:

```powershell
.\.venv\Scripts\ebook-tts.exe benchmark-readalong --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-unit 0 --unit-count 20 --target-buffer-seconds 12 --generation-mode balanced
```

Expected: completes without OOM; timing log records whether generation is faster than playback.

- [ ] **Step 5: Summarize benchmark**

Run this PowerShell summary:

```powershell
$session = Get-ChildItem "test false witness real tts chapter 15\read_along_sessions" -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$rows = Get-Content (Join-Path $session.FullName "timings.jsonl") | ConvertFrom-Json
$totalGen = ($rows | Measure-Object generation_seconds -Sum).Sum
$totalPlayback = ($rows | Measure-Object playback_seconds -Sum).Sum
[pscustomobject]@{
  session = $session.Name
  calls = $rows.Count
  generation_seconds = [math]::Round($totalGen, 3)
  playback_seconds = [math]::Round($totalPlayback, 3)
  realtime_factor = [math]::Round($totalGen / $totalPlayback, 3)
  max_call_seconds = [math]::Round(($rows | Measure-Object generation_seconds -Maximum).Maximum, 3)
} | ConvertTo-Json
```

Expected: `realtime_factor` is below 1.0 for seamless playback after initial buffer. If it is above 1.0, raise `target_buffer_seconds` and use the Qwen perf log to identify whether the slowdown is model generation or worker overhead.

- [ ] **Step 6: Record result in a benchmark note**

Run this PowerShell command to create `docs/benchmarks/2026-06-23-chapter-15-wsl-flashattention.md` from the newest chapter 15 timing log:

```powershell
New-Item -ItemType Directory -Force docs\benchmarks | Out-Null
$session = Get-ChildItem "test false witness real tts chapter 15\read_along_sessions" -Directory |
  Where-Object { $_.Name -like "benchmark-chapter_015-*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if ($null -eq $session) { throw "No benchmark chapter_015 session found" }
$timingPath = Join-Path $session.FullName "timings.jsonl"
$rows = Get-Content $timingPath | ConvertFrom-Json
if ($rows.Count -eq 0) { throw "Timing log has no rows: $timingPath" }
$totalGen = ($rows | Measure-Object generation_seconds -Sum).Sum
$totalPlayback = ($rows | Measure-Object playback_seconds -Sum).Sum
$maxCall = ($rows | Measure-Object generation_seconds -Maximum).Maximum
$realtime = $totalGen / $totalPlayback
$conclusion = if ($realtime -lt 1.0 -and $maxCall -le 12.0) {
  "Seamless for this sample: realtime_factor is below 1.0 and the 12 second initial buffer covers the slowest call."
} elseif ($realtime -lt 1.0) {
  "Total generation is faster than playback, but increase start_buffer_seconds above the slowest call before using it for interactive read-along."
} else {
  "Not seamless yet: generation is slower than playback for this sample. Inspect the Qwen perf log and WSL worker logs before tuning buffer values."
}
@"
# Chapter 15 WSL FlashAttention Benchmark

Command:
``ebook-tts benchmark-readalong --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-unit 0 --unit-count 20 --target-buffer-seconds 12 --generation-mode balanced``

Environment:
- Backend: WSL Ubuntu-24.04
- Torch: 2.9.1+cu130
- FlashAttention: 2.8.3
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Precision: bf16
- Attention: flash_attention_2

Results:
- generation_seconds: $([math]::Round($totalGen, 3))
- playback_seconds: $([math]::Round($totalPlayback, 3))
- realtime_factor: $([math]::Round($realtime, 3))
- max_call_seconds: $([math]::Round($maxCall, 3))
- timing_log: $timingPath

Conclusion:
$conclusion
"@ | Set-Content -Encoding UTF8 docs\benchmarks\2026-06-23-chapter-15-wsl-flashattention.md
```

Expected: The benchmark note contains numeric `generation_seconds`, `playback_seconds`, `realtime_factor`, and `max_call_seconds` values copied from the newest chapter 15 timing log.

- [ ] **Step 7: Commit benchmark note**

```powershell
git add docs\benchmarks\2026-06-23-chapter-15-wsl-flashattention.md
git commit -m "docs: record chapter 15 WSL flash benchmark"
```

---

### Task 9: Full Verification

**Files:**
- No source files changed unless verification exposes a bug.

- [ ] **Step 1: Run unit test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run WSL import verification**

Run:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- /opt/ebook-tts-venv/bin/python -c "import torch, flash_attn, qwen_tts, ebook_tts_pipeline; from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsRuntime; print(torch.__version__, torch.version.cuda); print(flash_attn.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(QwenTtsRuntime(model_root='/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/models/qwen-tts', model_choice='1.7B', precision='bf16', attention='auto')._attention_param())"
```

Expected output includes:

```text
2.9.1+cu130 13.0
2.8.3
True NVIDIA GeForce RTX 5080 Laptop GPU
flash_attention_2
```

- [ ] **Step 3: Run web UI smoke test**

Run:

```powershell
.\.venv\Scripts\ebook-tts-readalong-web.exe --book-root "test false witness real tts chapter 15" --port 51805
```

Expected: terminal prints `Read-along web UI: http://127.0.0.1:51805/`. Open the browser, start a read-along session, verify that the status shows buffered seconds and that `.wav` audio plays from memory.

- [ ] **Step 4: Stop the web server**

Press `Ctrl+C` in the terminal that is running the web server.

Expected: active session ends and `read_along_sessions/<session_id>` is removed.

- [ ] **Step 5: Final commit**

If all verification is green and no commits were skipped:

```powershell
git status --short
```

Expected: clean working tree.

If verification exposes a bug, stop this verification task and add a focused fix task above Task 9 with exact files, tests, commands, and a commit message for that bug.

---

## Self-Review

Spec coverage:
- WSL FlashAttention backend: Tasks 4, 5, 6, and 8.
- Persistent model worker: Tasks 5 and 6.
- In-memory WAV read-along buffer: Task 2 preserves `store_audio_files=False`; Task 3 verifies no per-unit files in web tests.
- Time-based buffer for smooth playback: Tasks 1, 2, and 3.
- Generation speed and playback comparison: Tasks 2, 7, and 8.
- Chapter 15 benchmark: Task 8.

Placeholder scan:
- The plan avoids deferred-work markers from the "No Placeholders" section.
- The benchmark note in Task 8 is generated from timing data by command, so it records real numeric measurements.

Type consistency:
- Settings keys are consistent: `target_buffer_seconds`, `start_buffer_seconds`, `max_buffer_seconds`, `max_buffer_units`.
- Worker protocol command names are consistent: `init`, `ensure_voice`, `generate_sentences`, `shutdown`.
- Adapter class name is consistent: `WslQwenWorkerAdapter`.
