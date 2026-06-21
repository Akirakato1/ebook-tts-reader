# Chunked Global Registry Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Haiku global registry discovery under context limits by processing complete chapter chunks around 130k characters while carrying the updated registry forward.

**Architecture:** `PipelineConfig` owns the default chunk size and environment override. `AudiobookPipeline.build_global_registry` groups chapter text files into whole-chapter windows, calls `GlobalRegistryService` once per window with the current registry, and merges each result before the next call. The prompt tells the model the registry is authoritative.

**Tech Stack:** Python dataclasses, pytest, existing Anthropic/global registry service, existing registry merge logic.

---

### Task 1: Config Knob

**Files:**
- Modify: `src/ebook_tts_pipeline/config.py`
- Test: `tests/test_public_import_and_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_default_config_is_ui_friendly_and_overridable(monkeypatch):
    monkeypatch.setenv("EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS", "130000")
    config = PipelineConfig.from_env(book_root="books/demo")
    assert config.global_registry_window_chars == 130000
```

- [ ] **Step 2: Run the test and see it fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_public_import_and_config.py -q`
Expected: fail because `PipelineConfig` has no `global_registry_window_chars`.

- [ ] **Step 3: Implement the config field**

Add `global_registry_window_chars: int = 130000` and parse `EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS` in `from_env`.

- [ ] **Step 4: Verify the test passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_public_import_and_config.py -q`
Expected: pass.

### Task 2: Chunked Carryover

**Files:**
- Modify: `src/ebook_tts_pipeline/pipeline.py`
- Test: `tests/test_pipeline_facade.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pipeline_builds_global_registry_in_chapter_chunks_with_updated_registry(tmp_path):
    service = QueuedGlobalRegistryService([...])
    pipeline = AudiobookPipeline(config=PipelineConfig(..., global_registry_window_chars=25), global_registry_service=service, ...)
    count = pipeline.build_global_registry(book_title="Demo")
    assert service.calls[0]["chapters"] == ["chapter_001", "chapter_002"]
    assert service.calls[1]["chapters"] == ["chapter_003"]
    assert "akari_adult" in service.calls[1]["registry"]["characters"]
    assert count == 2
```

- [ ] **Step 2: Run the test and see it fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_pipeline_facade.py::test_pipeline_builds_global_registry_in_chapter_chunks_with_updated_registry -q`
Expected: fail because the current code sends all chapters in one call.

- [ ] **Step 3: Implement chunking**

Add a small private chunk helper that keeps whole chapters together and allows one oversized chapter to travel alone. In `build_global_registry`, loop over chunks, reload the registry before each call, merge each result immediately, and accumulate the number of returned characters.

- [ ] **Step 4: Verify targeted test passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_pipeline_facade.py::test_pipeline_builds_global_registry_in_chapter_chunks_with_updated_registry -q`
Expected: pass.

### Task 3: Prompt And Docs

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/global_registry.py`
- Modify: `tests/test_global_registry_service.py`
- Modify: `docs/superpowers/specs/2026-06-21-global-registry-pass-design.md`

- [ ] **Step 1: Write/adjust prompt test**

Assert the rendered prompt includes authoritative-registry language and duplicate-prevention instructions.

- [ ] **Step 2: Implement prompt wording**

Add instructions that the existing registry is authoritative and that the model should not recreate already registered characters by role id, display name, alias, profile id, or person id.

- [ ] **Step 3: Verify full suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all tests pass.
