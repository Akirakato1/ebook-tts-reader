# Readalongweb Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Jupyter-style `readalongweb` launcher and library/add-book workflow.

**Architecture:** Keep `PrototypeUiController` as the per-book owner and add a library-aware `ReadAlongWebState` above it. The server resolves a launch root into direct-book or library mode, scans child book folders, and swaps the active controller when the user selects or adds a book.

**Tech Stack:** Python stdlib HTTP server, existing `PrototypeUiController`, existing EPUB extractor protocol, browser-native JavaScript.

---

### Task 1: Root Resolution And Library Discovery

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing tests**

Add tests that create two book-like child folders plus a non-book folder. Assert that direct book roots are selected immediately, library roots list only books, and repo roots containing `books/` resolve to that folder.

- [ ] **Step 2: Run the focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: the new tests fail because library discovery APIs do not exist yet.

- [ ] **Step 3: Implement minimal discovery**

Add helpers that identify a book folder by `chapters/*.txt`, summarize counts, resolve the launch root, and expose `GET /api/library`.

- [ ] **Step 4: Re-run the focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: the new discovery tests pass.

### Task 2: Book Selection And Add Book

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing tests**

Add tests for `POST /api/library/select` and `POST /api/library/add-book`. The add-book test injects a fake extractor through `create_server()` so no real EPUB fixture is required.

- [ ] **Step 2: Run the focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: the new endpoint tests fail with 404 or unsupported parameters.

- [ ] **Step 3: Implement selection and add-book**

Switch the active controller safely, ending any active session first. Add EPUB ingestion into `<library_root>/<slug>/`, initialize the book, register it, and select it.

- [ ] **Step 4: Re-run the focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: all web app tests pass.

### Task 3: CLI Alias And UI

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing tests**

Add tests that `build_parser()` defaults to notebook-style root launch and that `pyproject.toml` exposes both `readalongweb` and the existing long command.

- [ ] **Step 2: Run the focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: alias/default tests fail.

- [ ] **Step 3: Implement CLI and HTML library UI**

Add `readalongweb = "ebook_tts_pipeline.ui.web_app:main"`, update parser defaults, and add a compact library screen plus add-book form to the embedded UI.

- [ ] **Step 4: Re-run the focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: all focused tests pass.

### Task 4: Verification

**Files:**
- No new files.

- [ ] **Step 1: Compile the web app**

Run: `.\\.venv\\Scripts\\python.exe -m py_compile src\\ebook_tts_pipeline\\ui\\web_app.py`

Expected: exit 0.

- [ ] **Step 2: Run focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: pass.

- [ ] **Step 3: Run full suite**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests -q`

Expected: pass.
