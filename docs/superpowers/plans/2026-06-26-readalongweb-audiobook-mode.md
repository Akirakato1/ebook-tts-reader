# ReadAlongWeb Audiobook Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 12Hz audiobook-generation screen to `readalongweb` that renders persistent per-chapter WAV files from existing read-along annotations and voices.

**Architecture:** Add an offline audiobook renderer beside `ReadAlongSession`. It reuses read-along units, narrator/temp voice preparation, and `ChapterAudioBuilder`, but windows units into larger character-based TTS batches for whole-chapter throughput. Web APIs manage chapter selection, generation jobs, chapter WAV playback, and last-listened timestamp persistence.

**Tech Stack:** Python, pytest, existing `FakeTtsAdapter`, existing WSL vLLM-Omni 12Hz adapter, single-file `readalongweb` HTML/JS.

---

### Task 1: Audiobook Paths And Windowing

**Files:**
- Modify: `src/ebook_tts_pipeline/paths.py`
- Create: `src/ebook_tts_pipeline/read_along/audiobook.py`
- Test: `tests/test_audiobook_generation.py`

- [x] Add paths under `audiobook/` for chapter WAV, timeline, manifest, settings, and playback position.
- [x] Add `build_audiobook_windows(units, max_chars, max_roles)` that converts read-along units to TTS jobs and delegates to `build_tts_windows`.
- [x] Test that windows are larger than read-along live batches, preserve unit order, and respect role limits.

### Task 2: Controller Audiobook Renderer

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [x] Add audiobook settings with default `generation_mode=balanced`, `model_profile=12hz`, and large window sizes.
- [x] Add `generate_audiobook_chapter(chapter, settings, force=False, progress_callback=None)`.
- [x] Reuse runtime narrator qvp generation, legacy narrator-quote normalization, and chapter-local temp voice generation.
- [x] Write `audiobook/chapter_NNN.wav`, `audiobook/chapter_NNN.timeline.json`, and update `audiobook/manifest.json`.
- [x] Test with fake TTS that generation creates the WAV/timeline and calls TTS in character-windowed batches.

### Task 3: Web APIs And Jobs

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [x] Add `GET /api/audiobook?slug=...` for chapter audio status, settings, player position, and any running job.
- [x] Add `POST /api/audiobook/generate` for selected chapters with optional `force`.
- [x] Add background job progress using existing `LibraryJob`.
- [x] Add `GET /api/audiobook/audio/<chapter>.wav`.
- [x] Add `POST /api/audiobook/position` for chapter/timestamp persistence.
- [x] Test API payloads and fake generation.

### Task 4: Web UI Screen

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [x] Add an `Audiobook` button in the book toolbar.
- [x] Add a separate audiobook screen with chapter checkboxes, generate/regenerate action, generated chapter list, embedded player, speed control, and auto-continue toggle.
- [x] Save playback timestamp during audiobook playback.
- [x] Test that HTML exposes the audiobook controls.

### Task 5: Verification

**Files:**
- Modify: `README.md`

- [x] Update README to state audiobook mode is 12Hz-only for now.
- [x] Run `python -m pytest tests/test_audiobook_generation.py tests/test_audiobook_controller.py tests/test_read_along_web_app.py -k "audiobook"`.
- [x] Run `python -m compileall -q src`.
