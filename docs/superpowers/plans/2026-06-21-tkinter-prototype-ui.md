# Tkinter Prototype UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a prototype Tkinter UI for loading EPUBs, initializing chapters, navigating previously loaded books, editing the registry, and advancing chapters through annotation, script generation, audio generation, and audio opening.

**Architecture:** Add a testable `ebook_tts_pipeline.ui.controller` module with filesystem/artifact state and workflow methods. Add a thin `ebook_tts_pipeline.ui.tk_app` module that renders Tkinter widgets and delegates all pipeline work to the controller. Keep long-running actions in Tk worker threads.

**Tech Stack:** Python 3.9, Tkinter, existing `AudiobookPipeline`, existing EPUB/chapter/registry/TTS modules, pytest.

---

### Task 1: Controller State and Registry Editing

**Files:**
- Create: `src/ebook_tts_pipeline/ui/__init__.py`
- Create: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write failing tests**

Create tests for chapter state detection and registry JSON save behavior:

```python
from pathlib import Path

import pytest

from ebook_tts_pipeline.json_io import write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.ui.controller import ChapterStage, PrototypeUiController


def test_controller_detects_chapter_stage_from_artifacts(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\n", encoding="utf-8")
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True)
    paths.sentence_artifact("chapter_001").write_text("{}", encoding="utf-8")
    controller = PrototypeUiController(book_root=paths.root)

    assert controller.chapter_rows()[0].stage == ChapterStage.SEGMENTED

    paths.annotation("chapter_001").parent.mkdir(parents=True)
    paths.annotation("chapter_001").write_text("{}", encoding="utf-8")
    assert controller.chapter_rows()[0].stage == ChapterStage.ANNOTATED

    paths.tts_script("chapter_001").parent.mkdir(parents=True)
    paths.tts_script("chapter_001").write_text("{}", encoding="utf-8")
    paths.qwen_script("chapter_001").write_text("Narrator: Hi.\n", encoding="utf-8")
    assert controller.chapter_rows()[0].stage == ChapterStage.SCRIPTED

    paths.chapter_audio("chapter_001").parent.mkdir(parents=True)
    paths.chapter_audio("chapter_001").write_bytes(b"wav")
    assert controller.chapter_rows()[0].stage == ChapterStage.AUDIO
```

Add save validation:

```python
def test_controller_saves_pretty_registry_json(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root)

    controller.save_registry_text('{"book":{"title":"Demo"},"characters":{}}')

    text = paths.registry.read_text(encoding="utf-8")
    assert '"title": "Demo"' in text

    with pytest.raises(ValueError):
        controller.save_registry_text("{bad json")
```

- [ ] **Step 2: Verify tests fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py -q`

Expected: FAIL because `ebook_tts_pipeline.ui.controller` does not exist.

- [ ] **Step 3: Implement controller state and registry methods**

Create `ChapterStage`, `ChapterRow`, and `PrototypeUiController` with:

- `chapter_rows()`
- `registry_text()`
- `save_registry_text(text)`

- [ ] **Step 4: Verify tests pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py -q`

Expected: PASS.

### Task 2: Controller Workflow Actions

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write failing tests**

Add tests using injectable fakes for:

- `load_epub(epub_path, title, slug)` extracts chapters, initializes registry, and segments chapters.
- `run_next_chapter_action(chapter)` annotates when segmented, builds scripts when annotated, synthesizes when scripted, opens audio when audio exists.

- [ ] **Step 2: Verify tests fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py -q`

Expected: FAIL because workflow methods are missing.

- [ ] **Step 3: Implement workflow methods**

Add:

- `load_epub(epub_path, title, slug)`
- `run_next_chapter_action(chapter)`
- injectable `pipeline_factory`, `extractor`, and `audio_opener`

- [ ] **Step 4: Verify tests pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py -q`

Expected: PASS.

### Task 3: Tkinter App and Entry Point

**Files:**
- Create: `src/ebook_tts_pipeline/ui/tk_app.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing entry-point test**

Assert `pyproject.toml` contains `ebook-tts-ui = "ebook_tts_pipeline.ui.tk_app:main"`.

- [ ] **Step 2: Verify test fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_cli.py -q`

Expected: FAIL because the entry point is absent.

- [ ] **Step 3: Implement Tkinter app and entry point**

Add a simple two-column Tkinter UI:

- Top controls for EPUB path, title, slug, book root, fake TTS, load/init, refresh, registry toggle.
- Chapter list with color-coded buttons.
- Registry text panel with save button.
- Background thread wrapper for long controller actions.

- [ ] **Step 4: Verify tests pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_cli.py tests\test_ui_controller.py -q`

Expected: PASS.

### Task 4: Full Verification and Commit

**Files:**
- All modified files.

- [ ] **Step 1: Run full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Commit and push**

Run:

```powershell
git add docs src tests pyproject.toml
git commit -m "Add Tkinter prototype UI"
git push origin main
```
