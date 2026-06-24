# Readalongweb EPUB Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the visible add-book path field with a browser file picker backed by multipart EPUB upload.

**Architecture:** Keep `ReadAlongWebState.add_book()` for path-based JSON and add `ReadAlongWebState.add_uploaded_book()` for uploaded bytes. The HTTP handler routes multipart `/api/library/add-book` requests to the upload path and JSON requests to the existing path fallback.

**Tech Stack:** Python stdlib HTTP server, `cgi.FieldStorage` for multipart parsing, existing `PrototypeUiController.load_epub()`, browser `FormData`.

---

### Task 1: Multipart Upload API

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing tests**

Add tests for multipart `/api/library/add-book` with fields `epub`, `title`, and `slug`. Assert that the new book is selected and that `_source/original.epub` contains the uploaded bytes. Add a negative multipart test without the `epub` file.

- [ ] **Step 2: Run focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: multipart tests fail because the handler only parses JSON.

- [ ] **Step 3: Implement upload handling**

Add `_read_multipart_body()`, `UploadedBook`, and `ReadAlongWebState.add_uploaded_book()`. Save uploaded bytes under `_source/original.epub`, then call `controller.load_epub()` with that path.

- [ ] **Step 4: Re-run focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: focused tests pass.

### Task 2: Browser File Picker UI

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing HTML test**

Assert that the home page has `id="add-epub-file" type="file"` and `accept=".epub,application/epub+zip"`.

- [ ] **Step 2: Run the HTML test**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v`

Expected: fails because the current UI uses a text path input.

- [ ] **Step 3: Implement file picker JS**

Replace the visible path input with a file input. On selection, fill title and slug from filename. On add, send `FormData` with `epub`, `title`, and `slug`.

- [ ] **Step 4: Re-run focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: pass.

### Task 3: Verification

**Files:**
- No new files.

- [ ] **Step 1: Compile web app**

Run: `.\\.venv\\Scripts\\python.exe -m py_compile src\\ebook_tts_pipeline\\ui\\web_app.py`

Expected: exit 0.

- [ ] **Step 2: Run focused tests**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_read_along_web_app.py -v`

Expected: pass.

- [ ] **Step 3: Run full suite**

Run: `.\\.venv\\Scripts\\python.exe -m pytest tests -q`

Expected: pass.
