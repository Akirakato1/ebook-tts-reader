# Global Registry Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Haiku-default global character registry pass and use the locked registry for chapter annotation.

**Architecture:** Add a `global_registry` annotation module that prompts for canonical characters only. Add pipeline/controller methods that run this pass over segmented chapters, merge returned characters into `registry.json`, and mark chapter annotation as locked so new characters become annotation proposals instead of registry mutations.

**Tech Stack:** Python standard library, existing Anthropic JSON client abstraction, pytest, Tkinter prototype UI.

---

### Task 1: Global Registry Service

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/global_registry.py`
- Test: `tests/test_global_registry_service.py`

- [ ] Write tests for prompt rendering and service response parsing.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests\test_global_registry_service.py -q` and confirm failure.
- [ ] Implement `GlobalRegistryService.discover_characters(book_title, registry, chapter_texts)`.
- [ ] Run the focused tests and confirm pass.

### Task 2: Registry Merge Helpers

**Files:**
- Modify: `src/ebook_tts_pipeline/registry.py`
- Test: `tests/test_registry_and_voice_identity.py`

- [ ] Write tests showing a global character adds aliases to an existing record instead of creating a duplicate.
- [ ] Run the focused registry tests and confirm failure.
- [ ] Implement `merge_global_characters(chapter, characters)` on `RegistryManager`.
- [ ] Run the focused tests and confirm pass.

### Task 3: Pipeline and Locked Annotation

**Files:**
- Modify: `src/ebook_tts_pipeline/pipeline.py`
- Modify: `src/ebook_tts_pipeline/annotation/prompts.py`
- Modify: `src/ebook_tts_pipeline/annotation/service.py`
- Test: `tests/test_pipeline_facade.py`
- Test: `tests/test_annotation_prompts.py`

- [ ] Write tests for pipeline global registry generation and locked annotation not mutating registry.
- [ ] Run focused tests and confirm failure.
- [ ] Add `AudiobookPipeline.build_global_registry()` and a `lock_registry` flag for chapter annotation.
- [ ] Update prompts so locked annotation uses `proposed_new_characters` language.
- [ ] Run focused tests and confirm pass.

### Task 4: UI and CLI Flow

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Modify: `src/ebook_tts_pipeline/ui/tk_app.py`
- Modify: `src/ebook_tts_pipeline/cli.py`
- Test: `tests/test_ui_controller.py`
- Test: `tests/test_cli.py`

- [ ] Write tests for controller global registry action and CLI parser command.
- [ ] Run focused tests and confirm failure.
- [ ] Add `build-global-registry` CLI command and `Build Global Registry` UI button.
- [ ] Run focused tests and confirm pass.

### Task 5: Verification

**Files:**
- All files touched above.

- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q`.
- [ ] Inspect `git diff --stat`.
- [ ] Commit with `Add global registry pass`.
- [ ] Push `main`.
