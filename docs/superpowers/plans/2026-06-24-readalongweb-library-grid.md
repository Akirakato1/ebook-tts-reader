# Readalongweb Library Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Add Book controls above the book list and render each book as a rounded, columnar row with readable stats.

**Architecture:** Keep the existing plain HTML/CSS/JavaScript app shell. Change only the library markup and `renderLibrary()` row construction so the server payload fields are displayed directly in individual columns instead of concatenating `status_detail`.

**Tech Stack:** Python inline HTML string, browser DOM JavaScript, pytest static web shell tests.

---

### Task 1: Library Row Grid

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] **Step 1: Write failing tests**

Add assertions to `test_home_page_serves_clean_reader_shell` that the Add Book controls appear before `book-list`, the page includes column classes for title/status/chapter/annotation/read-along/voice/audio/last-read/action cells, the old concatenated `status_detail + " | Last read: "` rendering is gone, and a title truncation helper caps display text at 40 characters.

- [ ] **Step 2: Verify red**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v`

Expected: FAIL because the current list appears above Add Book and renders stats as a single metadata sentence.

- [ ] **Step 3: Implement layout**

Move the Add Book section above the list in `INDEX_HTML`, add column/header CSS, add `truncateBookTitle(title)` and render row cells from `book.chapter_count`, `book.annotation_count`, `book.read_along_unit_count`, `book.voice_count`, `book.audio_count`, and `book.last_read_label`.

- [ ] **Step 4: Verify green and suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -v`, `.\.venv\Scripts\python.exe -m py_compile src\ebook_tts_pipeline\ui\web_app.py`, and `.\.venv\Scripts\python.exe -m pytest tests -q`.
