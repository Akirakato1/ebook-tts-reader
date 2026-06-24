# Read-Along Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local browser-based read-along app launched from the terminal.

**Architecture:** Add a stdlib HTTP server in `ebook_tts_pipeline.ui.web_app` that wraps `PrototypeUiController` and `ReadAlongSession`. Serve one embedded HTML/CSS/JS app plus JSON/audio endpoints.

**Tech Stack:** Python `http.server`, `ThreadingHTTPServer`, existing pipeline/controller classes, browser-native JavaScript and audio playback.

---

### Task 1: Server API Contract

**Files:**
- Create: `tests/test_read_along_web_app.py`
- Create: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing tests**

Add tests that create a temp read-along book with one chapter, registry, annotation, and fake TTS. Assert that `GET /`, `GET /api/state`, `GET /api/chapter/chapter_001`, `POST /api/session/start`, `GET /api/session/<id>/audio/<unit>.wav`, `POST /api/session/advance`, and `POST /api/session/end` work.

- [ ] **Step 2: Run test and verify failure**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: import failure for `ebook_tts_pipeline.ui.web_app`.

- [ ] **Step 3: Implement minimal server**

Create `ReadAlongWebApp`, `ReadAlongWebState`, and `build_handler`. Use a single active session. Return JSON with `ok`, `chapters`, `settings`, `text`, `units`, `ready`, `audio_url`, and `session_id` fields.

- [ ] **Step 4: Run tests and verify pass**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: all web app tests pass.

### Task 2: Terminal Launcher

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing CLI parser test**

Add a test that `build_parser()` accepts `--book-root`, `--host`, `--port`, `--fake-tts`, and `--no-open`.

- [ ] **Step 2: Run test and verify failure**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py::test_web_app_parser_accepts_notebook_style_options -v`

Expected: missing function or missing script behavior.

- [ ] **Step 3: Implement launcher**

Add `build_parser()`, `run_server()`, and `main()`. Add console script `ebook-tts-readalong-web = "ebook_tts_pipeline.ui.web_app:main"`.

- [ ] **Step 4: Run test and verify pass**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py::test_web_app_parser_accepts_notebook_style_options -v`

Expected: pass.

### Task 3: Browser UI

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing HTML shell test**

Assert that `GET /` includes `Read Along`, `reader-text`, `Start Session`, and the JavaScript boot marker `window.readAlongApp`.

- [ ] **Step 2: Run test and verify failure**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v`

Expected: missing one or more shell markers.

- [ ] **Step 3: Implement embedded HTML/CSS/JS**

Render the clean reader UI with sidebar contents, toolbar controls, continuous text, inline unit spans, status bar, audio element, and fetch calls to the API endpoints.

- [ ] **Step 4: Run test and verify pass**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v`

Expected: pass.

### Task 4: Verification

**Files:**
- No new files.

- [ ] **Step 1: Compile touched Python**

Run: `.\\.venv\\Scripts\\python.exe -m py_compile src\\ebook_tts_pipeline\\ui\\web_app.py`

Expected: exit 0.

- [ ] **Step 2: Run focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py tests\\test_read_along_units.py tests\\test_read_along_session.py -v`

Expected: pass.

- [ ] **Step 3: Run full suite**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests -q`

Expected: pass.

- [ ] **Step 4: Validate chapter 15 temp-copy session**

Copy chapter 15 registry, chapters, and annotations to `C:\\tmp`, start a fake-TTS web app state, build units, start a session with buffer limit 2, fetch audio, advance, and end. Assert 419 units, two ready items, refill to `[1, 2]`, and deleted session directory.

- [ ] **Step 5: Browser smoke**

Start the web server on a local available port with `--no-open`, open it in the in-app browser, and verify the reader shell appears.
