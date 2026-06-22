# Chapter Local Speakers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the global registry limited to named, story-important voiced speakers while supporting chapter-scoped temporary voices for disposable speakers.

**Architecture:** `AnnotationResult` carries persistent `new_characters` plus chapter-local `local_speakers`. Global character voices resolve from `registry.json`; local voices resolve from a per-chapter temp registry under `temp_registries/` and use `.qvp` files under `voices/_temp/<chapter>/`.

**Tech Stack:** Python dataclasses, existing Anthropic annotation prompts, existing registry voice profile helpers, pytest.

---

### Task 1: Annotation Schema And Prompt Boundary

**Files:**
- Modify: `src/ebook_tts_pipeline/domain.py`
- Modify: `src/ebook_tts_pipeline/annotation/prompts.py`
- Modify: `src/ebook_tts_pipeline/annotation/global_registry.py`
- Test: `tests/test_annotation_prompts.py`
- Test: `tests/test_global_registry_service.py`

- [ ] Add `local_speakers` to `AnnotationResult`.
- [ ] Update the annotation prompt to request `local_speakers` for disposable chapter-only speakers.
- [ ] Update the global registry prompt to include only named, story-important characters who speak or think on page.
- [ ] Verify prompt tests fail before implementation and pass after implementation.

### Task 2: Validation And Merge

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/validator.py`
- Modify: `src/ebook_tts_pipeline/annotation/merge.py`
- Test: `tests/test_annotation_validator.py`
- Test: `tests/test_pipeline_facade.py`

- [ ] Allow roles declared by `local_speakers`.
- [ ] Require local speaker profiles to include compact voice fields.
- [ ] Merge local speakers across annotation windows without promoting them globally.
- [ ] Verify locked annotation no longer mutates the registry for disposable speakers.

### Task 3: Temp Registry And TTS Resolution

**Files:**
- Create: `src/ebook_tts_pipeline/temp_registry.py`
- Modify: `src/ebook_tts_pipeline/paths.py`
- Modify: `src/ebook_tts_pipeline/tts/script.py`
- Modify: `src/ebook_tts_pipeline/pipeline.py`
- Test: `tests/test_tts_script.py`
- Test: `tests/test_pipeline_facade.py`

- [ ] Write a chapter temp registry from annotation local speakers.
- [ ] Resolve local speaker roles when building TTS jobs.
- [ ] Prepare `.qvp` voices for local speaker default/internal variants.
- [ ] Keep global registry unchanged unless an explicit global character is added.

### Task 4: UI Behavior

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] Stop auto-promoting proposed/local speakers into `registry.json`.
- [ ] Let annotation approval remain focused on real registry characters and age-stage choices.
- [ ] Ensure already-approved legacy annotations with `proposed_new_characters` generate temp registries instead of failing.

### Task 5: Verification And Push

**Files:**
- Test: all tests

- [ ] Run focused tests for prompts, validator, script generation, pipeline, and UI.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q`.
- [ ] Commit without co-author trailers.
- [ ] Push `main` to `origin`.
