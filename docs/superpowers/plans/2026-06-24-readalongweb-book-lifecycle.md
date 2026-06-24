# Readalongweb Book Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manifest-backed staged book processing to the read-along web library.

**Architecture:** Add a per-book manifest at `readalong_book.json`, use it in library discovery/status/action payloads, and expose staged library endpoints for initialize, registry, annotation, and open gating.

**Tech Stack:** Python stdlib HTTP server, existing `PrototypeUiController`, existing pipeline services, browser-native JavaScript.

---

### Task 1: Pending Book Manifest

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] Write failing tests that Add Book creates a manifest-backed `fresh_added` library entry without extracting chapters.
- [ ] Run focused tests and verify failure.
- [ ] Implement manifest read/write helpers, pending book discovery, and pending add-book behavior.
- [ ] Re-run focused tests and verify pass.

### Task 2: Staged Actions

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/controller.py`

- [ ] Write failing tests for initialize, build registry, annotate, and open gating.
- [ ] Run focused tests and verify failure.
- [ ] Add staged endpoints and controller method for annotation-only whole-book processing.
- [ ] Re-run focused tests and verify pass.

### Task 3: Library UI

**Files:**
- Modify: `tests/test_read_along_web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`

- [ ] Write failing tests for row action labels, last-read text, disabled/open gating markers, and spinner text.
- [ ] Run focused tests and verify failure.
- [ ] Render row actions based on `action_key`, show loading status/spinner during actions, and keep the user on the library page until Open.
- [ ] Re-run focused tests and verify pass.

### Task 4: Verification

**Files:**
- No new files.

- [ ] Compile `web_app.py`.
- [ ] Run `tests/test_read_along_web_app.py -v`.
- [ ] Run full `pytest tests -q`.
