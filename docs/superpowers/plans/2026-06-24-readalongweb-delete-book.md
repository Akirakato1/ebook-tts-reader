# Readalongweb Delete Book Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a confirmation-gated delete control for books in the read-along web library.

**Architecture:** The browser shows a left-side `X` button per book row and calls a new JSON endpoint only after native `confirm()` approval. The server resolves the requested slug through discovered books, ends any active session for that book, validates the target remains inside the library root, removes the book directory, and returns a refreshed library payload.

**Tech Stack:** Python `http.server`, `shutil.rmtree`, plain JavaScript, pytest web API tests.

---

### Task 1: Delete Endpoint and UI Affordance

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing API and shell tests**

Add tests that create demo books, call `POST /api/library/delete`, assert the target directory disappears, assert the library refreshes, assert a missing slug returns a JSON error, and assert the served HTML includes delete row styling, `confirm(`, and the delete endpoint string.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -v`

Expected: FAIL because `/api/library/delete` and the delete UI do not exist yet.

- [ ] **Step 3: Implement minimal server and client behavior**

Add `ReadAlongWebState.delete_book(slug)`, route `POST /api/library/delete`, a left-side delete button in `renderLibrary`, a `deleteBook(book)` JavaScript function with native confirmation, and rounded/card-like `.book-row` styling.

- [ ] **Step 4: Run focused tests to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -v`

Expected: PASS.

- [ ] **Step 5: Run full verification**

Run: `.\.venv\Scripts\python.exe -m py_compile src\ebook_tts_pipeline\ui\web_app.py`

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`

Expected: compile exits 0 and all tests pass.
