# Qwen Window Sweep Benchmark Implementation Plan

> **Current invariant:** each character target is independent from other targets. For a target such as 100, 200, or 300 chars, create one fresh backend, generate a one-word `Test` warmup using the first target unit's voice and ignore that result, run the target three sequential times on that backend, average those three measured runs, then close the backend.

**Goal:** Add and run a repeatable Qwen3-TTS window-size sweep that measures generation speed, playback duration, and VRAM from 100-character windows upward until the 10 GB VRAM stop condition is reached.

**Architecture:** Benchmark logic lives in `ebook_tts_pipeline.benchmarks.window_sweep` so unit tests validate window construction, metric math, machine profiling, warmup handling, repeat averaging, VRAM stop behavior, and output writing without loading Qwen. The `ebook-tts benchmark-window-sweep` command prepares annotations and voices once, clears the benchmark perf log, then creates a fresh TTS backend per target length. Each target length first performs an ignored `Test` warmup, then performs three measured generations and averages generation/playback time while taking peak Qwen process VRAM as the maximum observed across measured repeats. The CSV also records device-wide VRAM used before/after each measured run so outside GPU pressure from video, browsers, Codex, Windows, or stale WSL workers is visible separately. The configured VRAM stop limit applies to the maximum of Qwen process reserved/allocated VRAM and device-wide VRAM used.

Because read-along units must stay intact, multiple nominal target sizes can resolve to the exact same unit set. The sweep skips those duplicate unit sets so the benchmark does not waste time retesting identical text windows.

**Outputs:**

- `test false witness real tts chapter 15/logs/window_sweep_chapter_015.csv`
- `test false witness real tts chapter 15/logs/window_sweep_chapter_015.json`
- `docs/benchmarks/<date>-chapter-015-window-sweep.md`

## Tasks

- [x] Add `src/ebook_tts_pipeline/benchmarks/__init__.py`.
- [x] Add `src/ebook_tts_pipeline/benchmarks/window_sweep.py`.
- [x] Add window construction that starts every target at the same chapter position and preserves read-along unit boundaries.
- [x] Add ignored one-word `Test` warmup before measured repeats.
- [x] Add repeat averaging with `repeat_count=3` by default.
- [x] Use one adapter/backend per target length, not one adapter per repeat.
- [x] Skip nominal target sizes that resolve to duplicate read-along unit sets.
- [x] Stop when max Qwen process VRAM or device-wide VRAM across the three repeats reaches the configured limit.
- [x] Record both Qwen process VRAM and device-wide GPU VRAM pressure.
- [x] Add `PerfLogReader`, machine profile collection, CSV/JSON/Markdown writers.
- [x] Add `ebook-tts benchmark-window-sweep`.
- [x] Add parser/unit tests for the benchmark command and sweep module.
- [ ] Run focused verification.
- [ ] Run full unit verification.
- [ ] Run chapter 15 smoke benchmark.
- [ ] Run chapter 15 full sweep if smoke runtime is acceptable.
- [ ] Summarize char count, generation time, playback time, RTF, max smooth speed, and VRAM threshold findings.

## Run Commands

Focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_window_sweep.py tests\test_cli_window_sweep.py tests\test_cli_benchmark_readalong.py -q
```

Smoke sweep:

```powershell
$env:EBOOK_TTS_BACKEND="wsl"
$env:EBOOK_TTS_QWEN_MODEL_ROOT="C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\models\qwen-tts"
$env:EBOOK_TTS_QWEN_ATTENTION="auto"
$env:EBOOK_TTS_QWEN_PRECISION="bf16"
.\.venv\Scripts\ebook-tts.exe benchmark-window-sweep --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-chars 100 --step-chars 100 --max-vram-gb 10 --playback-speed 1.0 --warmup-text Test --repeat-count 3 --max-targets 3
```

Full sweep:

```powershell
$env:EBOOK_TTS_BACKEND="wsl"
$env:EBOOK_TTS_QWEN_MODEL_ROOT="C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\models\qwen-tts"
$env:EBOOK_TTS_QWEN_ATTENTION="auto"
$env:EBOOK_TTS_QWEN_PRECISION="bf16"
.\.venv\Scripts\ebook-tts.exe benchmark-window-sweep --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-chars 100 --step-chars 100 --max-vram-gb 10 --playback-speed 1.0 --warmup-text Test --repeat-count 3
```
