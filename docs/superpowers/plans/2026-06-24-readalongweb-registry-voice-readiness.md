# Readalongweb Registry Voice Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a web registry review and voice preview workflow, then require all narrator, global-character, and chapter-local QVP voices to be prepared before a read-along book can be opened.

**Architecture:** Extend the existing staged `readalongweb` lifecycle rather than adding a parallel flow. The web server owns book status/actions, `PrototypeUiController` owns registry form/sample/voice-prep operations, and `AudiobookPipeline.prepare_voices_for_annotation()` remains the single source of truth for generating global and chapter-local QVP files. Voice preview/sample generation must use a lightweight voice-asset backend, never the `read_along_tts_backend` vLLM stack.

**Tech Stack:** Python `http.server`, existing `PrototypeUiController`, existing `AudiobookPipeline`, Qwen/Fake TTS adapters, plain JavaScript DOM UI, pytest.

---

## File Structure

- Modify `src/ebook_tts_pipeline/config.py`
  - Add an explicit `voice_asset_tts_backend` setting for QVP/sample work.
  - The setting must reject or rewrite `wsl-vllm-omni` / `vllm-omni` so samples do not load the read-along backend.
- Modify `src/ebook_tts_pipeline/ui/controller.py`
  - Add registry review payload methods.
  - Add lightweight voice sample generation.
  - Add whole-book read-along voice preparation after annotation.
  - Add a voice-asset pipeline helper that uses `voice_asset_tts_backend`.
- Modify `src/ebook_tts_pipeline/ui/web_app.py`
  - Add lifecycle stages/actions: `review_registry`, `prepare_voices`, `voices_ready`.
  - Add registry API routes.
  - Add sample audio route.
  - Add registry review panel UI.
  - Gate `Open` on `voices_ready`, not just annotation/read-along units.
- Modify `tests/test_ui_controller.py`
  - Add controller-level tests for registry payload, sample generation backend, save invalidation, and whole-book voice prep.
- Modify `tests/test_read_along_web_app.py`
  - Add web lifecycle/API/UI tests for the new gates and registry panel.
- Modify `tests/test_public_import_and_config.py`
  - Add config test for `EBOOK_TTS_VOICE_ASSET_BACKEND`.

---

### Task 1: Config Guard for Lightweight Voice Assets

**Files:**
- Modify: `src/ebook_tts_pipeline/config.py`
- Test: `tests/test_public_import_and_config.py`

- [ ] **Step 1: Write the failing config test**

Add this test to `tests/test_public_import_and_config.py`:

```python
def test_voice_asset_backend_defaults_to_lightweight_backend(monkeypatch):
    monkeypatch.delenv("EBOOK_TTS_VOICE_ASSET_BACKEND", raising=False)
    monkeypatch.setenv("EBOOK_TTS_BACKEND", "wsl-vllm-omni")

    config = PipelineConfig.from_env("book")

    assert config.voice_asset_tts_backend == "wsl"
    assert config.voice_asset_tts_backend != config.read_along_tts_backend


def test_voice_asset_backend_can_be_overridden(monkeypatch):
    monkeypatch.setenv("EBOOK_TTS_VOICE_ASSET_BACKEND", "native")

    config = PipelineConfig.from_env("book")

    assert config.voice_asset_tts_backend == "native"
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_public_import_and_config.py::test_voice_asset_backend_defaults_to_lightweight_backend tests\test_public_import_and_config.py::test_voice_asset_backend_can_be_overridden -v
```

Expected: FAIL with `AttributeError: 'PipelineConfig' object has no attribute 'voice_asset_tts_backend'`.

- [ ] **Step 3: Implement the config field**

In `PipelineConfig`, add:

```python
voice_asset_tts_backend: str = "native"
```

In `PipelineConfig.from_env()`, compute before `return cls(...)`:

```python
voice_asset_tts_backend = os.environ.get("EBOOK_TTS_VOICE_ASSET_BACKEND") or os.environ.get("EBOOK_TTS_BACKEND", "native")
if voice_asset_tts_backend in {"wsl-vllm-omni", "vllm-omni"}:
    voice_asset_tts_backend = "wsl"
```

Then pass:

```python
voice_asset_tts_backend=voice_asset_tts_backend,
```

- [ ] **Step 4: Verify the config tests pass**

Run the same pytest command from Step 2.

Expected: both tests PASS.

---

### Task 2: Controller Registry Payload Includes Narrator, Characters, and Detected Race/Accent Options

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write the failing controller test**

Add this test near the existing registry form tests in `tests/test_ui_controller.py`:

```python
def test_controller_registry_review_payload_includes_narrator_and_safe_character_fields(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_config_path": "voices/narrator.qvp",
                "voice_config_hash": "narrator-hash",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
            },
            "characters": {
                "callie_adult": {
                    "role_id": "callie_adult",
                    "profile_id": "callie_adult",
                    "person_id": "callie",
                    "display_name": "Callie",
                    "age_stage": "adult",
                    "aliases": ["Callie adult"],
                    "voice_config_path": "voices/callie_adult.qvp",
                    "voice_config_hash": "character-hash",
                    "identity_profile": {
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["guarded"],
                        "race_or_ethnicity": "Japanese",
                        "accent": "Tokyo",
                        "occupation": "lawyer",
                    },
                    "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                }
            },
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    payload = controller.registry_review_payload()

    assert payload["book"] == {"title": "Demo", "slug": "demo"}
    assert [entry["role_id"] for entry in payload["entries"]] == ["narrator", "callie_adult"]
    narrator = payload["entries"][0]
    callie = payload["entries"][1]
    assert narrator["kind"] == "narrator"
    assert narrator["editable"] is False
    assert callie["kind"] == "character"
    assert callie["editable"] is True
    assert callie["fields"]["display_name"] == "Callie"
    assert callie["fields"]["race_or_ethnicity"] == "Japanese"
    assert callie["fields"]["accent"] == "Tokyo"
    assert callie["voice_config_path"] == "voices/callie_adult.qvp"
    assert "voice_config_hash" not in callie
    assert "qwen_instruct" not in callie
    assert "seed" not in callie
    assert "accent_options" in payload
    assert "race_or_ethnicity_options" in payload
    assert "Tokyo" in payload["accent_options"]
    assert "Japanese" in payload["race_or_ethnicity_options"]
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_registry_review_payload_includes_narrator_and_safe_character_fields -v
```

Expected: FAIL with `AttributeError: 'PrototypeUiController' object has no attribute 'registry_review_payload'`.

- [ ] **Step 3: Implement `registry_review_payload()`**

Add this constant near the registry form dataclasses in `controller.py`:

```python
ACCENT_OPTIONS = [
    "",
    "General American",
    "Southern American",
    "British",
    "Irish",
    "Scottish",
    "Australian",
    "Canadian",
    "New York",
    "Tokyo",
    "Custom",
]

RACE_OR_ETHNICITY_OPTIONS = [
    "",
    "African American",
    "British",
    "Chinese",
    "Hispanic / Latino",
    "Indian",
    "Irish",
    "Japanese",
    "Korean",
    "White",
    "Custom",
]
```

Add this method to `PrototypeUiController`:

```python
def registry_review_payload(self) -> Dict[str, Any]:
    registry = read_json(self.paths.registry) if self.paths.registry.exists() else {}
    entries: List[Dict[str, Any]] = []
    narrator = dict(registry.get("narrator", {}))
    if narrator:
        entries.append(
            {
                "kind": "narrator",
                "role_id": str(narrator.get("role_id", "narrator")),
                "title": str(narrator.get("display_name", "Narrator")),
                "editable": False,
                "fields": {
                    "display_name": str(narrator.get("display_name", "Narrator")),
                },
                "voice_config_path": str(narrator.get("voice_config_path", "") or ""),
                "sample_url": self._voice_sample_url("narrator"),
            }
        )
    for form in self.registry_character_forms():
        fields = {field.key: field.value for field in form.editable_fields}
        voice_path = ""
        for field in form.readonly_fields:
            if field.key == "voice_config_path":
                voice_path = field.value
                break
        entries.append(
            {
                "kind": "character",
                "role_id": form.role_id,
                "title": form.title,
                "editable": True,
                "fields": fields,
                "voice_config_path": voice_path,
                "sample_url": self._voice_sample_url(form.role_id),
            }
        )
    return {
        "book": dict(registry.get("book", {})),
        "accent_options": _merged_detected_options(
            ACCENT_OPTIONS,
            _detected_identity_values(self.paths, registry, "accent"),
        ),
        "race_or_ethnicity_options": _merged_detected_options(
            RACE_OR_ETHNICITY_OPTIONS,
            _detected_identity_values(self.paths, registry, "race_or_ethnicity"),
        ),
        "entries": entries,
    }
```

Add helper:

```python
def _voice_sample_url(self, role_id: str) -> str:
    sample_path = self.paths.root / "voices" / "_samples" / f"{role_id}.wav"
    return f"/api/registry/sample/{role_id}.wav" if sample_path.exists() else ""
```

Add module helpers so detected LLM values always appear in the same dropdowns the user edits:

```python
def _merged_detected_options(base_options: List[str], detected_values: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in [*base_options, *detected_values]:
        text = str(value).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def _detected_identity_values(paths: BookPaths, registry: Dict[str, Any], key: str) -> List[str]:
    values: List[str] = []
    for character in registry.get("characters", {}).values():
        if not isinstance(character, dict):
            continue
        identity = character.get("identity_profile", {})
        if isinstance(identity, dict) and identity.get(key):
            values.append(str(identity[key]))
    for annotation_path in sorted((paths.root / "annotations").glob("*.annotation.json")):
        payload = read_json(annotation_path)
        for speaker in payload.get("local_speakers", []):
            profile = speaker.get("profile", {})
            if isinstance(profile, dict) and profile.get(key):
                values.append(str(profile[key]))
        for speaker in payload.get("proposed_new_characters", []):
            profile = speaker.get("profile", {})
            if isinstance(profile, dict) and profile.get(key):
                values.append(str(profile[key]))
    return values
```

Requirement: values detected by the global registry pass or by chapter annotation must not be overwritten or hidden just because they are not in the curated base list. If the model detects `race_or_ethnicity="Japanese"` or `accent="Tokyo"`, those values must appear in `race_or_ethnicity_options` / `accent_options`, remain selected in the character card, and continue into `build_compact_voice_profile()` so Qwen receives the same race/accent descriptor in `qwen_instruct`.

- [ ] **Step 4: Verify the controller test passes**

Run the same pytest command from Step 2.

Expected: PASS.

---

### Task 3: Saving Registry Edits Invalidates Voice Readiness

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write the failing web API test**

Add this test to `tests/test_read_along_web_app.py`:

```python
def test_registry_save_invalidates_voice_readiness_and_returns_updated_review_payload(tmp_path):
    paths = _write_demo_book(tmp_path, name="ready-book", title="Ready Book")
    write_json_atomic(
        paths.root / "readalong_book.json",
        {
            "schema": "readalong_book_v1",
            "title": "Ready Book",
            "slug": "ready-book",
            "source_epub_path": "_source/original.epub",
            "original_filename": "ready.epub",
            "stages": {
                "source_added": True,
                "initialized": True,
                "global_registry": True,
                "annotated": True,
                "registry_reviewed": True,
                "voices_ready": True,
            },
        },
    )
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        before = _get_json(base_url + "/api/library")
        assert before["books"][0]["action_key"] == "open"

        saved = _post_json(
            base_url + "/api/registry/save-character",
            {
                "slug": "ready-book",
                "role_id": "leigh_adult",
                "fields": {
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": "direct, careful",
                    "race_or_ethnicity": "",
            "race_or_ethnicity": "Japanese",
            "accent": "Tokyo",
                    "occupation": "lawyer",
                    "aliases": "Leigh",
                },
            },
        )

        assert saved["ok"] is True
        assert saved["book"]["status_key"] == "registry_review"
        assert saved["book"]["action_key"] == "review_registry"
        assert saved["review"]["entries"][1]["fields"]["race_or_ethnicity"] == "Japanese"
        assert saved["review"]["entries"][1]["fields"]["accent"] == "Tokyo"
        reloaded = read_json(paths.registry)
        assert reloaded["characters"]["leigh_adult"]["voice_config_hash"] is None
        error = _post_json(base_url + "/api/library/select", {"slug": "ready-book"}, expect_status=400)
        assert "Review Voices" in error["error"]
    finally:
        _stop_server(server, thread)
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_registry_save_invalidates_voice_readiness_and_returns_updated_review_payload -v
```

Expected: FAIL because `/api/registry/save-character` does not exist and the lifecycle does not have `registry_review`.

- [ ] **Step 3: Add manifest stage invalidation**

In `web_app.py`, extend `_write_book_manifest()` stages:

```python
"registry_reviewed": False,
"voices_ready": False,
```

Extend `_update_book_stage()` parameters:

```python
registry_reviewed: Optional[bool] = None,
voices_ready: Optional[bool] = None,
```

When `annotated` is set to `False`, also set `registry_reviewed=False` and `voices_ready=False`.

When a registry character is saved, call:

```python
_update_book_stage(summary.book_root, registry_reviewed=False, voices_ready=False)
```

- [ ] **Step 4: Add the save endpoint**

Add `ReadAlongWebState.registry_save_character()`:

```python
def registry_save_character(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    with self.lock:
        summary = self._require_book_summary(str(payload.get("slug", "")))
        controller = self._make_controller(summary.book_root)
        controller.save_registry_character_form(
            str(payload.get("role_id", "")),
            {str(key): str(value) for key, value in dict(payload.get("fields", {})).items()},
        )
        _update_book_stage(summary.book_root, registry_reviewed=False, voices_ready=False)
        summary = summarize_book(summary.book_root)
        return {
            "ok": True,
            "book": summary.to_payload(),
            "review": controller.registry_review_payload(),
            "library": self.library_payload(),
        }
```

Add route:

```python
if path == "/api/registry/save-character":
    self._send_json(app_state.registry_save_character(payload))
    return
```

- [ ] **Step 5: Verify the test passes**

Run the same pytest command from Step 2.

Expected: PASS.

---

### Task 4: Registry Review Gate and Web Panel UI

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write failing lifecycle and shell tests**

Add this lifecycle test to `tests/test_read_along_web_app.py`:

```python
def test_annotated_book_requires_registry_review_before_voice_prep_and_open(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
        pipeline_factory=_fake_lifecycle_pipeline_factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Lifecycle Book", "slug": "lifecycle-book"},
            files={"epub": ("lifecycle-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )
        _post_json(base_url + "/api/library/initialize", {"slug": "lifecycle-book"})
        _post_json(base_url + "/api/library/build-registry", {"slug": "lifecycle-book"})
        annotated = _post_json(base_url + "/api/library/annotate", {"slug": "lifecycle-book"})

        assert annotated["book"]["status_key"] == "registry_review"
        assert annotated["book"]["action_key"] == "review_registry"
        assert annotated["book"]["open_enabled"] is False
        error = _post_json(base_url + "/api/library/select", {"slug": "lifecycle-book"}, expect_status=400)
        assert "Review Voices" in error["error"]

        review = _get_json(base_url + "/api/registry?slug=lifecycle-book")
        assert review["ok"] is True
        assert [entry["role_id"] for entry in review["review"]["entries"]] == ["narrator", "leigh_adult"]

        confirmed = _post_json(base_url + "/api/registry/confirm", {"slug": "lifecycle-book"})

        assert confirmed["book"]["status_key"] == "registry_reviewed"
        assert confirmed["book"]["action_key"] == "prepare_voices"
        assert confirmed["book"]["open_enabled"] is False
    finally:
        _stop_server(server, thread)
```

Add these assertions to `test_home_page_serves_clean_reader_shell`:

```python
assert 'id="registry-panel"' in response
assert "Review Voices" in response
assert "Prepare Voices" in response
assert "/api/registry?slug=" in response
assert "/api/registry/confirm" in response
assert "/api/registry/save-character" in response
assert "renderRegistryPanel" in response
assert "saveRegistryCharacter" in response
assert "playRegistrySample" in response
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_annotated_book_requires_registry_review_before_voice_prep_and_open tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v
```

Expected: FAIL because the status/action and registry routes/UI do not exist.

- [ ] **Step 3: Extend statuses and actions**

In `_book_status()`, for manifest books use:

```python
if not stages.get("initialized"):
    return "fresh_added", "Freshly added"
if not stages.get("global_registry"):
    return "initialized", "Initialized"
if not stages.get("annotated"):
    return "registry_ready", "Registry ready"
if not stages.get("registry_reviewed"):
    return "registry_review", "Review voices"
if not stages.get("voices_ready"):
    return "registry_reviewed", "Registry reviewed"
return "voices_ready", "Voices ready"
```

In `_book_action()` add:

```python
if status_key == "registry_review":
    return "review_registry", "Review Voices", False
if status_key == "registry_reviewed":
    return "prepare_voices", "Prepare Voices", False
if status_key in {"voices_ready", "audio_ready"}:
    return "open", "Open", True
```

Change `annotate_book()` after annotation to:

```python
_update_book_stage(summary.book_root, annotated=True, registry_reviewed=False, voices_ready=False)
```

- [ ] **Step 4: Add registry routes**

In `do_GET()`, parse query with `parse_qs` from `urllib.parse`, then add:

```python
if path == "/api/registry":
    slug = parse_qs(parsed.query).get("slug", [""])[0]
    self._send_json(app_state.registry_payload(slug))
    return
```

Add `ReadAlongWebState.registry_payload()`:

```python
def registry_payload(self, slug: str) -> Dict[str, Any]:
    with self.lock:
        summary = self._require_book_summary(slug)
        controller = self._make_controller(summary.book_root)
        return {"ok": True, "book": summary.to_payload(), "review": controller.registry_review_payload()}
```

Add `registry_confirm()`:

```python
def registry_confirm(self, slug: str) -> Dict[str, Any]:
    with self.lock:
        summary = self._require_book_summary(slug)
        _update_book_stage(summary.book_root, registry_reviewed=True, voices_ready=False)
        summary = summarize_book(summary.book_root)
        return {"ok": True, "book": summary.to_payload(), "library": self.library_payload()}
```

Add route:

```python
if path == "/api/registry/confirm":
    self._send_json(app_state.registry_confirm(str(payload.get("slug", ""))))
    return
```

- [ ] **Step 5: Add registry panel UI**

In `INDEX_HTML`, add a hidden section next to the library list:

```html
<section class="registry-panel" id="registry-panel" hidden>
  <header class="registry-head">
    <h2 id="registry-title">Registry</h2>
    <button id="registry-close">Close</button>
  </header>
  <div class="registry-list" id="registry-list"></div>
  <div class="registry-actions">
    <button class="primary" id="registry-confirm">Confirm Registry Review</button>
  </div>
</section>
```

Add `els.registryPanel`, `els.registryList`, `els.registryTitle`, `els.registryConfirm`.

Add action handling in `runBookAction()`:

```javascript
if (book.action_key === "review_registry") {
  await showRegistry(book.slug);
  return;
}
```

Add JS functions:

```javascript
async function showRegistry(slug) {
  setLibraryStatus("Loading registry...");
  const payload = await api("/api/registry?slug=" + encodeURIComponent(slug));
  renderRegistryPanel(payload.book, payload.review);
  els.registryPanel.hidden = false;
  setLibraryStatus("Review voices before preparing QVP files.");
}

function renderRegistryPanel(book, review) {
  state.registryBook = book;
  state.registryReview = review;
  els.registryTitle.textContent = "Registry - " + book.title;
  els.registryList.textContent = "";
  for (const entry of review.entries) {
    const card = document.createElement("div");
    card.className = "registry-card";
    card.dataset.roleId = entry.role_id;
    card.appendChild(registryField(entry, "display_name", "Name", entry.editable));
    if (entry.kind === "character") {
      card.appendChild(registryField(entry, "age_stage", "Age Stage", true));
      card.appendChild(registryField(entry, "gender", "Gender", true));
      card.appendChild(registryField(entry, "personality", "Personality", true));
      card.appendChild(registryRaceField(entry, review.race_or_ethnicity_options));
      card.appendChild(registryAccentField(entry, review.accent_options));
      card.appendChild(registryField(entry, "occupation", "Occupation", true));
      card.appendChild(registryField(entry, "aliases", "Aliases", true));
    }
    const sample = document.createElement("button");
    sample.textContent = "Play Sample";
    sample.onclick = () => playRegistrySample(entry.role_id);
    card.appendChild(sample);
    if (entry.editable) {
      const save = document.createElement("button");
      save.textContent = "Save";
      save.onclick = () => saveRegistryCharacter(entry.role_id);
      card.appendChild(save);
    }
    els.registryList.appendChild(card);
  }
}
```

- [ ] **Step 6: Verify tests pass**

Run the same pytest command from Step 2.

Expected: PASS.

---

### Task 5: Lightweight Voice Sample Generation

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_ui_controller.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write failing controller test for backend and WAV generation**

Add to `tests/test_ui_controller.py`:

```python
def test_controller_generates_registry_voice_sample_with_voice_asset_backend(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    paths.registry.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator"},
            "characters": {
                "callie_adult": {
                    "role_id": "callie_adult",
                    "profile_id": "callie_adult",
                    "person_id": "callie",
                    "display_name": "Callie",
                    "age_stage": "adult",
                    "aliases": [],
                    "identity_profile": {"age_stage": "adult", "gender": "female", "personality": ["guarded"]},
                    "voice_identity": {"seed": 7, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                    "voice_config_path": None,
                    "voice_config_hash": None,
                }
            },
        },
    )
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        fake_tts=True,
    )

    sample = controller.generate_registry_voice_sample("callie_adult")

    assert sample["role_id"] == "callie_adult"
    assert sample["sample_path"].endswith("voices/_samples/callie_adult.wav")
    assert (paths.root / "voices" / "_samples" / "callie_adult.wav").exists()
    assert paths.voice_qvp("callie_adult").exists()
    assert any(call[0] == "factory" and call[6] != "wsl-vllm-omni" for call in calls)
```

- [ ] **Step 2: Run the failing controller test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_generates_registry_voice_sample_with_voice_asset_backend -v
```

Expected: FAIL with missing `generate_registry_voice_sample`.

- [ ] **Step 3: Implement voice-asset pipeline helper**

In `PrototypeUiController`, add:

```python
def _voice_asset_pipeline(self) -> AudiobookPipeline:
    settings = self.tts_settings()
    base_config = PipelineConfig.from_env(str(self.book_root))
    backend = base_config.voice_asset_tts_backend
    if backend in {"wsl-vllm-omni", "vllm-omni"}:
        backend = "wsl"
    config = replace(
        base_config,
        tts_backend=backend,
        tts_speed=settings["tts_speed"],
        pause_between_sentences_ms=settings["pause_between_sentences_ms"],
        intra_sentence_pause_ms=settings["intra_sentence_pause_ms"],
    )
    return self.pipeline_factory(config, False, self.fake_tts)
```

- [ ] **Step 4: Implement sample generation**

Add:

```python
def generate_registry_voice_sample(self, role_id: str) -> Dict[str, str]:
    role_id = str(role_id).strip()
    if not role_id:
        raise ValueError("role_id is required")
    registry = read_json(self.paths.registry)
    record = self._registry_voice_record(role_id, registry)
    display_name = str(record.get("display_name", role_id)).strip() or role_id
    pipeline = self._voice_asset_pipeline()
    voice_path = pipeline._voice_path_for_record(role_id, record)
    pipeline.tts_adapter.ensure_voice(role_id, record, voice_path)
    record["voice_config_path"] = voice_path.relative_to(self.paths.root).as_posix()
    record["voice_config_hash"] = voice_profile_hash(record)
    pipeline.registry.save(registry)
    text = f"Hi, my name is {display_name}."
    generated = pipeline.tts_adapter.generate_sentences(
        [
            {
                "index": 0,
                "sentence_idx": 0,
                "text": text,
                "role": display_name,
                "role_id": role_id,
                "type": "dialogue",
                "voice_config_path": record["voice_config_path"],
            }
        ]
    )
    if not generated:
        raise RuntimeError("Voice sample generation returned no audio.")
    sample_dir = self.paths.root / "voices" / "_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{role_id}.wav"
    _write_wav_file(sample_path, generated[0].samples, generated[0].sample_rate)
    return {
        "role_id": role_id,
        "sample_path": sample_path.relative_to(self.paths.root).as_posix(),
        "sample_url": f"/api/registry/sample/{role_id}.wav",
    }
```

Add helper:

```python
def _registry_voice_record(self, role_id: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    if role_id == "narrator":
        return registry.setdefault("narrator", {"role_id": "narrator", "display_name": "Narrator"})
    characters = registry.setdefault("characters", {})
    if role_id not in characters:
        raise ValueError(f"Registry character not found: {role_id}")
    return characters[role_id]
```

Add module helper:

```python
def _write_wav_file(path: Path, samples: Any, sample_rate: int) -> None:
    import wave
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16.tobytes())
```

- [ ] **Step 5: Add web sample endpoints**

In `ReadAlongWebState`, add:

```python
def registry_generate_sample(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    with self.lock:
        summary = self._require_book_summary(str(payload.get("slug", "")))
        controller = self._make_controller(summary.book_root)
        sample = controller.generate_registry_voice_sample(str(payload.get("role_id", "")))
        _update_book_stage(summary.book_root, voices_ready=False)
        summary = summarize_book(summary.book_root)
        return {"ok": True, "book": summary.to_payload(), "sample": sample, "review": controller.registry_review_payload()}
```

Add route:

```python
if path == "/api/registry/generate-sample":
    self._send_json(app_state.registry_generate_sample(payload))
    return
```

Add GET route for sample audio:

```python
if path.startswith("/api/registry/sample/"):
    role_file = path.rsplit("/", 1)[-1]
    role_id = role_file[:-4] if role_file.endswith(".wav") else role_file
    audio_bytes = app_state.registry_sample_audio(role_id)
    self._send_bytes(audio_bytes, content_type="audio/wav")
    return
```

Add:

```python
def registry_sample_audio(self, role_id: str) -> bytes:
    with self.lock:
        if self.controller is None:
            raise ValueError("Select or load a book before playing registry samples.")
        safe_role = _safe_slug(role_id)
        sample_path = self.controller.book_root / "voices" / "_samples" / f"{safe_role}.wav"
        root = (self.controller.book_root / "voices" / "_samples").resolve()
        resolved = sample_path.resolve()
        if root not in resolved.parents and resolved != root:
            raise FileNotFoundError("sample path is outside the sample directory")
        return resolved.read_bytes()
```

- [ ] **Step 6: Add JS sample behavior**

In `playRegistrySample(roleId)`:

```javascript
async function playRegistrySample(roleId) {
  if (!state.registryBook) return;
  setLibraryStatus("Regenerating sample...");
  const payload = await api("/api/registry/generate-sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slug: state.registryBook.slug, role_id: roleId })
  });
  state.registryReview = payload.review;
  renderRegistryPanel(payload.book, payload.review);
  const audio = new Audio(payload.sample.sample_url + "?t=" + Date.now());
  await audio.play();
  setLibraryStatus("Sample ready.");
}
```

- [ ] **Step 7: Verify sample tests pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_generates_registry_voice_sample_with_voice_asset_backend tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v
```

Expected: PASS.

---

### Task 6: Whole-Book Voice Preparation After Annotation

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_ui_controller.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write failing controller test for global and local temp speakers**

Add to `tests/test_ui_controller.py`:

```python
def test_controller_prepare_read_along_voices_prepares_global_and_local_temp_speakers(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Narration. "Stop there," the guard said.', encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
                "voice_config_path": None,
            },
            "characters": {
                "leigh_adult": {
                    "role_id": "leigh_adult",
                    "profile_id": "leigh_adult",
                    "person_id": "leigh",
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "aliases": [],
                    "identity_profile": {"age_stage": "adult", "gender": "female", "personality": ["direct"]},
                    "voice_identity": {"seed": 2, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                    "voice_config_path": None,
                }
            },
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "schema": "quote_attribution_v1",
            "roles": ["Security Guard"],
            "quotes": [[1, 0]],
            "local_speakers": [
                {
                    "local_id": "tmp_001",
                    "label": "Security Guard",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "personality": ["authoritative"],
                        "occupation": "security guard",
                    },
                }
            ],
        },
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    result = controller.prepare_read_along_voices()

    assert result["chapters"] == 1
    assert result["voices_ready"] is True
    assert paths.voice_qvp("narrator").exists()
    assert (paths.root / "voices" / "_temp" / "chapter_001" / "tmp_001.qvp").exists()
    units = read_json(paths.read_along_units("chapter_001"))["units"]
    assert any(unit["voice_config_path"] == "voices/_temp/chapter_001/tmp_001.qvp" for unit in units)
```

- [ ] **Step 2: Run the failing controller test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_prepare_read_along_voices_prepares_global_and_local_temp_speakers -v
```

Expected: FAIL with missing `prepare_read_along_voices`.

- [ ] **Step 3: Implement `prepare_read_along_voices()`**

In `PrototypeUiController`, add:

```python
def prepare_read_along_voices(self) -> Dict[str, Any]:
    pipeline = self._voice_asset_pipeline()
    chapters = [row.chapter for row in self.chapter_rows()]
    prepared_chapters = 0
    for chapter in chapters:
        if not self.paths.annotation(chapter).exists():
            raise ValueError(f"Annotation missing for {chapter}.")
        annotation = AnnotationResult.from_dict(read_json(self.paths.annotation(chapter)))
        pipeline.prepare_voices_for_annotation(annotation, chapter=chapter)
        pipeline.build_read_along_units(chapter)
        prepared_chapters += 1
    missing = self._missing_read_along_voice_paths(chapters)
    return {
        "chapters": len(chapters),
        "prepared_chapters": prepared_chapters,
        "missing_voice_paths": missing,
        "voices_ready": not missing,
    }
```

Add:

```python
def _missing_read_along_voice_paths(self, chapters: List[str]) -> List[str]:
    missing: List[str] = []
    for chapter in chapters:
        units_path = self.paths.read_along_units(chapter)
        if not units_path.exists():
            missing.append(f"{chapter}:read_along_units")
            continue
        for unit in read_json(units_path).get("units", []):
            voice_path = str(unit.get("voice_config_path") or "").strip()
            if not voice_path:
                missing.append(f"{chapter}:unit_{unit.get('unit_id')}:empty")
                continue
            if not (self.paths.root / voice_path).exists():
                missing.append(f"{chapter}:unit_{unit.get('unit_id')}:{voice_path}")
    return missing
```

- [ ] **Step 4: Add web route and lifecycle update**

In `ReadAlongWebState`, add:

```python
def prepare_book_voices(self, slug: str) -> Dict[str, Any]:
    with self.lock:
        summary = self._require_book_summary(slug)
        if summary.status_key != "registry_reviewed":
            raise ValueError("Confirm Registry Review is required before preparing voices.")
        controller = self._make_controller(summary.book_root)
        result = controller.prepare_read_along_voices()
        _update_book_stage(summary.book_root, voices_ready=bool(result["voices_ready"]))
        summary = summarize_book(summary.book_root)
        return {"ok": True, "book": summary.to_payload(), **result, "library": self.library_payload()}
```

Add route:

```python
if path == "/api/library/prepare-voices":
    self._send_json(app_state.prepare_book_voices(str(payload.get("slug", ""))))
    return
```

Add endpoint mapping in JS:

```javascript
prepare_voices: "/api/library/prepare-voices"
```

Add pending/status labels:

```javascript
prepare_voices: "Preparing voices..."
```

- [ ] **Step 5: Write/verify web lifecycle test**

Extend `test_annotated_book_requires_registry_review_before_voice_prep_and_open`:

```python
prepared = _post_json(base_url + "/api/library/prepare-voices", {"slug": "lifecycle-book"})
assert prepared["book"]["status_key"] == "voices_ready"
assert prepared["book"]["action_key"] == "open"
assert prepared["book"]["open_enabled"] is True
opened = _post_json(base_url + "/api/library/select", {"slug": "lifecycle-book"})
assert opened["active_book"]["slug"] == "lifecycle-book"
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_annotated_book_requires_registry_review_before_voice_prep_and_open -v
```

Expected: PASS.

---

### Task 7: Remove Start-Session Voice Prep as the Primary Safety Net

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Add regression test that session start does not newly create missing QVPs**

Add to `tests/test_ui_controller.py`:

```python
def test_read_along_session_requires_prepared_voice_paths(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right."', encoding="utf-8")
    _write_callie_registry(paths)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    units = controller.build_read_along_units("chapter_001")

    with pytest.raises(ValueError, match="Prepare Voices"):
        controller.create_read_along_session(
            "chapter_001",
            units,
            {
                "playback_speed": 1.0,
                "generation_mode": "balanced",
                "buffer_limit": 2,
                "target_buffer_seconds": 20,
                "start_buffer_seconds": 20,
                "max_buffer_seconds": 40,
                "max_buffer_units": 32,
                "narrator_voice_type": "current",
            },
        )
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_read_along_session_requires_prepared_voice_paths -v
```

Expected: FAIL because `create_read_along_session()` currently prepares voices itself.

- [ ] **Step 3: Change `create_read_along_session()` to validate, not prepare**

Remove this line from `create_read_along_session()`:

```python
pipeline.prepare_voices_for_annotation(annotation, chapter=chapter)
```

Keep narrator voice type application before validation only if it does not change hashes. If narrator type changes, raise:

```python
if settings.get("narrator_voice_type") != "current":
    raise ValueError("Change narrator voice type before Prepare Voices, then prepare voices again.")
```

After rebuilding units, validate:

```python
missing = self._missing_read_along_voice_paths([chapter])
if missing:
    raise ValueError("Prepare Voices before starting a read-along session.")
```

- [ ] **Step 4: Update web settings behavior**

If the user changes `narrator_voice_type`, save settings but do not allow a session to start until `Prepare Voices` is run again. Add this to `ReadAlongWebState.save_settings()`:

```python
if controller.book_root.exists():
    _update_book_stage(controller.book_root, voices_ready=False)
```

- [ ] **Step 5: Verify regression**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_read_along_session_requires_prepared_voice_paths tests\test_ui_controller.py::test_controller_read_along_session_prepares_voices_before_buffering -v
```

Expected: new test PASS; existing `test_controller_read_along_session_prepares_voices_before_buffering` should be updated to call `prepare_read_along_voices()` before `create_read_along_session()`.

---

### Task 8: Web UI Styling and Interaction Polish

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Add shell assertions**

Extend `test_home_page_serves_clean_reader_shell` with:

```python
assert ".registry-panel" in response
assert ".registry-card" in response
assert ".registry-grid" in response
assert "Regenerating sample..." in response
assert "Sample ready." in response
assert "Confirm Registry Review" in response
```

- [ ] **Step 2: Run failing shell test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v
```

Expected: FAIL until styles and strings are present.

- [ ] **Step 3: Add CSS**

Add:

```css
.registry-panel {
  display: grid;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}
.registry-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.registry-head h2 {
  margin: 0;
  font-size: 18px;
}
.registry-list {
  display: grid;
  gap: 10px;
}
.registry-card {
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f9fafb;
}
.registry-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 8px;
}
.registry-actions {
  display: flex;
  justify-content: end;
  gap: 8px;
}
```

- [ ] **Step 4: Verify shell test passes**

Run the same pytest command from Step 2.

Expected: PASS.

---

### Task 9: Full Verification

**Files:**
- Verify all modified source and tests.

- [ ] **Step 1: Run focused web tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -v
```

Expected: all web tests PASS.

- [ ] **Step 2: Run focused controller/config tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py tests\test_public_import_and_config.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 3: Compile changed Python files**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\ebook_tts_pipeline\config.py src\ebook_tts_pipeline\ui\controller.py src\ebook_tts_pipeline\ui\web_app.py
```

Expected: exit code 0.

- [ ] **Step 4: Run full suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests PASS.

---

## Self-Review

- Spec coverage: Covers registry review UI, editable safe fields, sample playback, save-triggered sample regeneration, lightweight sample backend, whole-book QVP prep after annotation, local temp speaker QVP prep, and `Open` gating.
- Placeholder scan: No `TBD`, `TODO`, or undefined endpoint names remain.
- Type consistency: Stage names are `registry_review`, `registry_reviewed`, and `voices_ready`; actions are `review_registry`, `prepare_voices`, and `open`; endpoints use `/api/registry/*` and `/api/library/prepare-voices`.
- Race/accent handling: The current codebase treats `race_or_ethnicity` and `accent` as prompt text inside `qwen_instruct`, not strict model enums. The plan uses curated dropdowns plus `Custom`, and dynamically appends values detected by the global registry pass or chapter annotation, while preserving arbitrary user-entered text.
