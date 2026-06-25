# ReadAlong Sharing And Generation Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add library-level ReadAlong package export/import and make read-along generation choices show measured VRAM/RTF tradeoffs.

**Architecture:** Add a focused package module that owns zip construction, validation, and import sanitization. Wire it into the existing `ReadAlongWebState` library API, then add small library UI controls and clearer generation labels without changing the existing read-along session runtime.

**Tech Stack:** Python `zipfile`, existing `readalongweb` HTTP server, existing manifest/registry JSON helpers, vanilla HTML/CSS/JS, pytest.

---

### Task 1: Portable Package Core

**Files:**
- Create: `src/ebook_tts_pipeline/ui/book_package.py`
- Test: `tests/test_readalong_book_package.py`

- [ ] Write failing tests for export contents and import sanitization.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests\test_readalong_book_package.py -q` and confirm failure.
- [ ] Implement package building and importing with safe path validation.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests\test_readalong_book_package.py -q`.

### Task 2: Web API And Library UI

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] Write failing tests for `/api/library/export`, `/api/library/import-package`, and visible UI controls.
- [ ] Run targeted web tests and confirm failure.
- [ ] Add GET zip export, multipart zip import, Share button, and Import Zip input/button.
- [ ] Run targeted web tests.

### Task 3: Generation Mode Clarity

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] Write failing test that the HTML exposes measured VRAM/RTF labels.
- [ ] Update generation dropdown labels and add a short hardware profile hint.
- [ ] Run targeted UI tests.

### Task 4: Verification

**Files:**
- No additional files.

- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q`.
- [ ] Report exact verification output and any remaining limitations.
