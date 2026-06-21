# Annotation Debug Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically save local debug logs for annotation and UI pipeline failures.

**Architecture:** Add a small `FailureLogger` utility that writes sanitized JSON files under `logs/annotation_failures/`. Inject it into `AnnotationService` from CLI/UI pipeline factories, and have the Tkinter background runner show the saved log path in error popups.

**Tech Stack:** Python standard library, existing `write_json_atomic`, pytest.

---

### Task 1: Failure Logger

**Files:**
- Create: `src/ebook_tts_pipeline/debug_logging.py`
- Test: `tests/test_debug_logging.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from ebook_tts_pipeline.debug_logging import FailureLogger


def test_failure_logger_writes_sanitized_json(tmp_path):
    logger = FailureLogger(tmp_path / "logs", context={"book_root": "books/demo"})

    path = logger.write_failure(
        "annotation_model_output_error",
        {"message": "Bearer secret-token sk-ant-api03-abc", "chapter": "chapter_001"},
    )

    text = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path / "logs"
    assert "annotation_model_output_error" in path.name
    assert "books/demo" in text
    assert "secret-token" not in text
    assert "sk-ant-api03-abc" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_debug_logging.py -q`

Expected: FAIL because `ebook_tts_pipeline.debug_logging` does not exist.

- [ ] **Step 3: Implement logger**

Create `FailureLogger` with `write_failure(event_type, details, exc=None)`, recursive JSON sanitization, traceback capture, UTC timestamp filenames, and `attach_debug_log_path(exc, path)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_debug_logging.py -q`

Expected: PASS.

### Task 2: Annotation Logging

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/anthropic_client.py`
- Modify: `src/ebook_tts_pipeline/annotation/service.py`
- Modify: `src/ebook_tts_pipeline/cli.py`
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_annotation_service.py`

- [ ] **Step 1: Write failing tests**

Add tests that provide a `FailureLogger`, trigger an `AnnotationModelOutputError`, and assert a log contains the prompt/raw model text. Add a validation failure followed by repair and assert a log is still written.

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_annotation_service.py -q`

Expected: FAIL because `AnnotationService` does not accept or use a failure logger.

- [ ] **Step 3: Implement annotation logging**

Extend `AnnotationModelOutputError` with optional `raw_text` and `source`. Add optional `failure_logger` and `book_root` context to `AnnotationService`. Log model-output exceptions, malformed payload exceptions, and validation failures before repair. Attach the log path to raised exceptions.

- [ ] **Step 4: Wire factories**

Create a `FailureLogger` in CLI and Tkinter controller pipeline factories using `PipelineConfig.debug_log_root` and context containing `book_root`.

- [ ] **Step 5: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_annotation_service.py tests\test_anthropic_client_parsing.py -q`

Expected: PASS.

### Task 3: UI Error Path

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/tk_app.py`
- Test: `tests/test_ui_error_logging.py`

- [ ] **Step 1: Write failing tests**

Add tests for helper functions that format error messages with an existing debug log path and write a generic UI log when no path is attached.

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_ui_error_logging.py -q`

Expected: FAIL because the helpers do not exist.

- [ ] **Step 3: Implement UI helper and integrate runner**

Add `_pipeline_error_message(exc, label, book_root)` to return a message with `Debug log: <path>`. Update `_run_background` to use it when posting error events.

- [ ] **Step 4: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_ui_error_logging.py -q`

Expected: PASS.

### Task 4: Verification and Delivery

**Files:**
- Modify only files touched by Tasks 1-3 and this plan/spec.

- [ ] **Step 1: Run full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Inspect diff**

Run: `git diff --stat`

Expected: changes are limited to debug logging, annotation service/client wiring, UI error reporting, tests, and docs.

- [ ] **Step 3: Commit and push**

Run: `git add ...`, `git commit -m "Add annotation failure debug logs"`, and `git push origin main`.
