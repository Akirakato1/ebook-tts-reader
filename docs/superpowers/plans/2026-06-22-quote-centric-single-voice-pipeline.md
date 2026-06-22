# Quote-Centric Single Voice Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sentence-level speaker annotation with quote-centric attribution and simplify every character to one stable voice per role.

**Architecture:** EPUB/chapter ingestion remains deterministic. A new quote attribution layer extracts exact quote spans, asks Anthropic to map quote IDs to global registry roles or chapter-local speakers, then builds TTS jobs by interleaving narrator spans and attributed quote spans by offsets. Registry and temp-registry voice records become single-voice records; old `voice_variants.default` data remains readable as migration compatibility, but new outputs do not write `_default` or `_internal`.

**Tech Stack:** Python dataclasses, existing Anthropic JSON client, existing registry/temp-registry/Qwen adapter abstractions, pytest.

---

### Task 1: Single-Voice Registry Compatibility

**Files:**
- Modify: `src/ebook_tts_pipeline/registry.py`
- Modify: `src/ebook_tts_pipeline/temp_registry.py`
- Modify: `src/ebook_tts_pipeline/paths.py`
- Test: `tests/test_registry_and_voice_identity.py`
- Test: `tests/test_pipeline_facade.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert newly inserted global characters store `voice_profile` and `voice_config_path` directly, do not write `voice_variants`, and resolve to role IDs like `elena_adult`. Add a temp speaker test that expects `voices/_temp/chapter_001/tmp_001.qvp`, not `tmp_001_default.qvp`.

- [ ] **Step 2: Verify tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_registry_and_voice_identity.py::test_inserted_character_uses_single_voice_record tests/test_pipeline_facade.py::test_pipeline_prepares_single_voice_for_local_speaker -q`

Expected: failures mention `voice_variants` or `_default` paths because production code still writes variants.

- [ ] **Step 3: Implement minimal registry migration**

Change `resolve_effective_voice` to return the direct character record for non-narrator roles. If an old record only has `voice_variants.default`, copy its `voice_profile`, `voice_config_path`, and `voice_config_hash` to the character record at load/save time. Change new character insertion and refresh logic to write direct `voice_profile`/`voice_config_path` and remove `voice_variants`. Change temp registry creation/resolution to a direct single-voice record.

- [ ] **Step 4: Verify tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_registry_and_voice_identity.py tests/test_pipeline_facade.py -q`

Expected: all selected tests pass.

### Task 2: Quote Span Extraction

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/quotes.py`
- Test: `tests/test_quote_extraction.py`

- [ ] **Step 1: Write failing tests**

Add tests for adjacent quotes, narrator text before/between/after quotes, smart quotes, straight quotes, and explicit tag preservation. The expected artifact contains quote IDs, text, start/end offsets, and narrator spans.

- [ ] **Step 2: Verify tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quote_extraction.py -q`

Expected: import failure for the new module.

- [ ] **Step 3: Implement extractor**

Create `QuoteSpan`, `NarratorSpan`, and `QuoteExtraction` dataclasses plus `extract_quoted_dialogue(text: str) -> QuoteExtraction`. The scanner treats curly and straight double quotes as direct speech boundaries, preserves quote marks in quote text, emits offsets into the original chapter string, and emits narrator spans for non-empty text outside quotes.

- [ ] **Step 4: Verify tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quote_extraction.py -q`

Expected: all quote extraction tests pass.

### Task 3: Quote Attribution Prompt and Validation

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/quote_attribution.py`
- Modify: `src/ebook_tts_pipeline/annotation/registry_summary.py`
- Test: `tests/test_quote_attribution.py`

- [ ] **Step 1: Write failing tests**

Add tests that render a prompt containing marked quote IDs, compact registry records, and schema rules for `quotes` plus `local_speakers`. Add tests that parse/validate a model response and reject missing quote IDs, narrator dialogue labels, and local roles without profiles.

- [ ] **Step 2: Verify tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quote_attribution.py -q`

Expected: import failure for the new quote attribution module.

- [ ] **Step 3: Implement prompt and service**

Implement `render_quote_attribution_prompt`, `QuoteAttributionResult`, `validate_quote_attribution`, and `QuoteAttributionService`. The output schema is:

```json
{
  "roles": ["callie_child", "local_001"],
  "local_speakers": [
    {
      "local_id": "local_001",
      "label": "Security Guard",
      "profile": {
        "age_stage": "adult",
        "gender": "unknown",
        "race_or_ethnicity": null,
        "accent": null,
        "occupation": "security guard",
        "personality": ["brusque"]
      }
    }
  ],
  "quotes": [[1, 0, "dialogue"], [2, 1, "dialogue"]]
}
```

The prompt forbids creating global registry characters, permits chapter-local speakers for unregistered disposable speakers, and instructs the model to choose age-stage-specific global roles when present.

- [ ] **Step 4: Verify tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quote_attribution.py -q`

Expected: all quote attribution tests pass.

### Task 4: Build TTS Scripts from Quote Attribution

**Files:**
- Modify: `src/ebook_tts_pipeline/tts/script.py`
- Test: `tests/test_tts_script.py`

- [ ] **Step 1: Write failing tests**

Add a test that takes raw chapter text, quote spans, and quote attribution output, then expects narrator jobs for outside text and character jobs for quote text. The rendered Qwen script should use `callie_child:` and never `callie_child_default:`.

- [ ] **Step 2: Verify tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tts_script.py::test_builds_quote_attributed_tts_script_with_single_voice_roles -q`

Expected: failure because no quote-attributed TTS builder exists.

- [ ] **Step 3: Implement quote-attributed TTS builder**

Add `build_tts_script_from_quotes(chapter, chapter_text, extraction, attribution, registry, max_chars, max_roles, language, temp_registry=None)`. It interleaves narrator and quote spans in original offset order, resolves voices through the new single-voice registry/temp-registry path, builds windows with existing `build_tts_windows`, and preserves `sentence_idx`/`unit_idx` as quote/narrator span order for current adapter compatibility.

- [ ] **Step 4: Verify tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tts_script.py -q`

Expected: all TTS script tests pass after legacy expectations are updated for single voice.

### Task 5: Pipeline and CLI/UI Wiring

**Files:**
- Modify: `src/ebook_tts_pipeline/pipeline.py`
- Modify: `src/ebook_tts_pipeline/cli.py`
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Modify: `src/ebook_tts_pipeline/ui/tk_app.py`
- Test: `tests/test_pipeline_facade.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write failing tests**

Add tests that `annotate_chapter` writes quote attribution output, temp registry local speakers, and script generation consumes quote attribution. Add CLI tests for existing `annotate-chapter` and `build-tts-script` using quote attribution. Update UI controller tests so annotation approval reviews global role age-stage choices but no internal/default variants.

- [ ] **Step 2: Verify tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_facade.py tests/test_cli.py tests/test_ui_controller.py -q`

Expected: failures show old `AnnotationResult` contract or `_default` paths.

- [ ] **Step 3: Implement wiring**

Add a quote attribution service to `AudiobookPipeline`. `annotate_chapter` loads raw chapter text, extracts quotes, calls quote attribution, writes a JSON artifact at the existing annotation path for UI compatibility, and writes temp registry records. `build_sentence_jobs` detects quote attribution artifacts and calls `build_tts_script_from_quotes`; legacy sentence annotations stay readable for old data. CLI and UI continue using the same commands/buttons.

- [ ] **Step 4: Verify tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_facade.py tests/test_cli.py tests/test_ui_controller.py -q`

Expected: all selected tests pass.

### Task 6: Final Verification and Push

**Files:**
- Update docs/runbook text if it still describes default/internal voice generation.

- [ ] **Step 1: Run full test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Inspect git diff**

Run: `git status --short` and `git diff --stat`

Expected: source, tests, and docs changed; generated `test example false witness chapter 13/` remains untracked and unstaged.

- [ ] **Step 3: Commit and push**

Run: `git add src tests docs`, then `git commit -m "Implement quote-centric single voice pipeline"`, then `git push origin main`.

Expected: commit on `main` pushed to the SSH remote without `Co-Authored-By`.
