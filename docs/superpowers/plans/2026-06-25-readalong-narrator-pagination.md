# Read-Along Narrator Profile And Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved read-along narrator profile lifecycle and adaptive ebook pagination in `readalongweb`.

**Architecture:** Add a focused narrator profile helper for read-along storage, profile normalization, voice-record construction, and hash-keyed QVP cache paths. Keep global character voice generation unchanged, then update web state/API and browser rendering so session start uses cached narrator QVPs and the reader displays measured logical pages as either a two-page spread or single page based on sidebar state.

**Tech Stack:** Python 3.9+, pytest, the existing stdlib HTTP web app, browser DOM/JavaScript, existing `FakeTtsAdapter`, existing Qwen/WSL adapters.

---

## File Structure

- Modify `src/ebook_tts_pipeline/paths.py`
  - Add `read_along_narrator_profile`.
  - Add `narrator_voice_qvp(profile_hash, role_id)`.
- Create `src/ebook_tts_pipeline/read_along/narrator_profile.py`
  - Own narrator profile defaults, migration, normalization, summary text, voice-record conversion, and functional narrator derivative construction.
- Modify `src/ebook_tts_pipeline/ui/controller.py`
  - Expose narrator profile load/save/review APIs.
  - Replace `narrator_voice_type` session logic with narrator profile hash/cache logic.
  - Keep `Generate Voices` global-character only.
- Modify `src/ebook_tts_pipeline/ui/web_app.py`
  - Add narrator profile API endpoints.
  - Replace narrator select in the reader toolbar with summary + edit button.
  - Add sidebar toggle, page nav buttons, page rendering state, anchor-to-page logic, and visible session-start error reporting.
- Modify `tests/test_ui_controller.py`
  - Add narrator profile storage/cache tests.
  - Update old `narrator_voice_type` tests to profile-based settings.
- Modify `tests/test_read_along_web_app.py`
  - Add web API/UI shell tests for narrator profile and adaptive pagination controls.

---

### Task 1: Narrator Profile Storage Model

**Files:**
- Modify: `src/ebook_tts_pipeline/paths.py`
- Create: `src/ebook_tts_pipeline/read_along/narrator_profile.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write failing tests for default narrator profile and migration**

Add imports near the top of `tests/test_ui_controller.py`:

```python
from ebook_tts_pipeline.read_along.narrator_profile import narrator_profile_hash
```

Add these tests near the existing read-along settings tests:

```python
def test_controller_read_along_narrator_profile_defaults_to_editable_profile(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    profile = controller.read_along_narrator_profile()

    assert profile["role_id"] == "narrator"
    assert profile["display_name"] == "Narrator"
    assert profile["identity_profile"]["age_stage"] == "adult"
    assert profile["identity_profile"]["gender"] == "male"
    assert "audiobook narrator" in profile["identity_profile"]["occupation"]
    assert profile["voice_profile"]["description"]
    assert paths.read_along_narrator_profile.exists()
```

```python
def test_controller_migrates_narrator_profile_from_legacy_registry(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Story Reader",
                "voice_identity": {"seed": 9, "differentiators": ["warm tone"]},
                "voice_profile": {
                    "description": "warm adult female narrator",
                    "qwen_instruct": "A warm adult female narrator.",
                },
            },
            "characters": {},
        },
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    profile = controller.read_along_narrator_profile()

    assert profile["display_name"] == "Story Reader"
    assert profile["voice_identity"]["seed"] == 9
    assert "warm adult female narrator" in profile["voice_profile"]["description"]
    assert paths.read_along_narrator_profile.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_read_along_narrator_profile_defaults_to_editable_profile tests\test_ui_controller.py::test_controller_migrates_narrator_profile_from_legacy_registry -q
```

Expected: fail because `read_along_narrator_profile`, `narrator_profile_hash`, and controller narrator profile methods do not exist.

- [ ] **Step 3: Add path helpers**

Add to `BookPaths` in `src/ebook_tts_pipeline/paths.py`:

```python
    @property
    def read_along_narrator_profile(self) -> Path:
        return self.root / "read_along" / "narrator_profile.json"

    def narrator_voice_qvp(self, profile_hash: str, role_id: str) -> Path:
        safe_hash = str(profile_hash).strip()
        safe_role = str(role_id).strip() or "narrator"
        return self.root / "voices" / "_narrator" / safe_hash / f"{safe_role}.qvp"
```

- [ ] **Step 4: Create narrator profile helper**

Create `src/ebook_tts_pipeline/read_along/narrator_profile.py`:

```python
from __future__ import annotations

from typing import Any, Dict

from ebook_tts_pipeline.registry import (
    build_compact_voice_profile,
    default_narrator_voice_profile,
    role_seed,
    voice_profile_hash,
)


DEFAULT_NARRATOR_IDENTITY = {
    "age_stage": "adult",
    "gender": "male",
    "personality": ["calm", "clear", "measured"],
    "race_or_ethnicity": None,
    "accent": None,
    "occupation": "audiobook narrator",
}


def default_narrator_profile(book_slug: str = "book") -> Dict[str, Any]:
    return normalize_narrator_profile(
        {
            "role_id": "narrator",
            "display_name": "Narrator",
            "identity_profile": dict(DEFAULT_NARRATOR_IDENTITY),
            "voice_identity": {
                "seed": role_seed(book_slug, "narrator"),
                "differentiators": ["calm baseline narrator timbre"],
            },
            "voice_profile": default_narrator_voice_profile(),
        },
        book_slug=book_slug,
    )


def narrator_profile_from_registry(registry: Dict[str, Any], book_slug: str = "book") -> Dict[str, Any]:
    narrator = dict(registry.get("narrator") or {})
    if not narrator:
        return default_narrator_profile(book_slug)
    return normalize_narrator_profile(narrator, book_slug=book_slug)


def normalize_narrator_profile(profile: Dict[str, Any], book_slug: str = "book") -> Dict[str, Any]:
    role_id = "narrator"
    display_name = str(profile.get("display_name") or "Narrator").strip() or "Narrator"
    identity = dict(profile.get("identity_profile") or {})
    for key, value in DEFAULT_NARRATOR_IDENTITY.items():
        identity.setdefault(key, value)
    if isinstance(identity.get("personality"), str):
        identity["personality"] = [
            item.strip()
            for item in str(identity["personality"]).split(",")
            if item.strip()
        ]
    elif not isinstance(identity.get("personality"), list):
        identity["personality"] = list(DEFAULT_NARRATOR_IDENTITY["personality"])
    voice_identity = dict(profile.get("voice_identity") or {})
    voice_identity.setdefault("seed", role_seed(book_slug, role_id))
    voice_identity.setdefault("differentiators", ["calm baseline narrator timbre"])
    voice_profile = dict(profile.get("voice_profile") or {})
    if not voice_profile.get("description") or not voice_profile.get("qwen_instruct"):
        voice_profile = build_compact_voice_profile(display_name, {"identity_profile": identity})
    return {
        "role_id": role_id,
        "display_name": display_name,
        "identity_profile": identity,
        "voice_identity": voice_identity,
        "voice_profile": voice_profile,
    }


def narrator_profile_hash(profile: Dict[str, Any]) -> str:
    return voice_profile_hash(narrator_voice_record(profile))


def narrator_voice_record(profile: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_narrator_profile(profile)
    return {
        "role_id": "narrator",
        "display_name": normalized["display_name"],
        "identity_profile": dict(normalized["identity_profile"]),
        "voice_identity": dict(normalized["voice_identity"]),
        "voice_profile": dict(normalized["voice_profile"]),
    }


def narrator_summary(profile: Dict[str, Any]) -> str:
    normalized = normalize_narrator_profile(profile)
    identity = normalized["identity_profile"]
    parts = [
        str(identity.get("age_stage") or "").replace("_", " "),
        str(identity.get("gender") or ""),
        str(identity.get("accent") or "").replace("_", " "),
    ]
    compact = " ".join(part for part in parts if part and part != "unknown").strip()
    return f"{normalized['display_name']}: {compact or 'custom narrator'}"


def functional_narrator_voice_record(profile: Dict[str, Any]) -> Dict[str, Any]:
    base = narrator_voice_record(profile)
    voice_profile = dict(base["voice_profile"])
    base_description = str(voice_profile.get("description") or "audiobook narrator")
    base_instruction = str(voice_profile.get("qwen_instruct") or base_description)
    return {
        "role_id": "functional_narrator",
        "display_name": "Functional Narrator",
        "identity_profile": dict(base["identity_profile"]),
        "voice_identity": dict(base["voice_identity"]),
        "voice_profile": {
            "description": (
                f"{base_description}; same narrator identity for quoted non-dialogue text, "
                "slightly higher pitch, flatter monotone delivery, crisp and restrained"
            ),
            "qwen_instruct": (
                f"{base_instruction}. Keep the same base narrator identity, but render quoted "
                "non-dialogue text with a slightly higher pitch, flatter monotone cadence, "
                "restrained emotion, and crisp articulation."
            ),
        },
    }
```

- [ ] **Step 5: Add controller load/save profile methods**

In `src/ebook_tts_pipeline/ui/controller.py`, import the helper functions:

```python
from ebook_tts_pipeline.read_along.narrator_profile import (
    functional_narrator_voice_record,
    narrator_profile_from_registry,
    narrator_profile_hash,
    narrator_summary,
    narrator_voice_record,
    normalize_narrator_profile,
)
```

Add methods to `PrototypeUiController` near `read_along_settings`:

```python
    def read_along_narrator_profile(self) -> Dict[str, Any]:
        if self.paths.read_along_narrator_profile.exists():
            return normalize_narrator_profile(read_json(self.paths.read_along_narrator_profile))
        registry = read_json(self.paths.registry) if self.paths.registry.exists() else {}
        book_slug = str(registry.get("book", {}).get("slug", self.book_root.name))
        profile = narrator_profile_from_registry(registry, book_slug=book_slug)
        write_json_atomic(self.paths.read_along_narrator_profile, profile)
        return profile

    def save_read_along_narrator_profile(self, values: Dict[str, Any]) -> Dict[str, Any]:
        current = self.read_along_narrator_profile()
        identity = dict(current.get("identity_profile") or {})
        identity.update(
            {
                "age_stage": str(values.get("age_stage", identity.get("age_stage", "adult"))).strip() or "adult",
                "gender": str(values.get("gender", identity.get("gender", "unknown"))).strip() or "unknown",
                "personality": _split_csv(values.get("personality", ",".join(identity.get("personality", [])))),
                "race_or_ethnicity": _blank_to_none(values.get("race_or_ethnicity", identity.get("race_or_ethnicity", ""))),
                "accent": _blank_to_none(values.get("accent", identity.get("accent", ""))),
                "occupation": _blank_to_none(values.get("occupation", identity.get("occupation", "audiobook narrator"))),
            }
        )
        profile = normalize_narrator_profile(
            {
                "role_id": "narrator",
                "display_name": str(values.get("display_name", current.get("display_name", "Narrator"))).strip() or "Narrator",
                "identity_profile": identity,
                "voice_identity": dict(current.get("voice_identity") or {}),
            },
            book_slug=self.book_root.name,
        )
        write_json_atomic(self.paths.read_along_narrator_profile, profile)
        return profile

    def read_along_narrator_profile_payload(self) -> Dict[str, Any]:
        profile = self.read_along_narrator_profile()
        identity = dict(profile.get("identity_profile") or {})
        return {
            "profile": profile,
            "summary": narrator_summary(profile),
            "hash": narrator_profile_hash(profile),
            "fields": {
                "display_name": str(profile.get("display_name", "Narrator")),
                "age_stage": str(identity.get("age_stage", "")),
                "gender": str(identity.get("gender", "")),
                "personality": ", ".join(identity.get("personality", [])),
                "race_or_ethnicity": str(identity.get("race_or_ethnicity") or ""),
                "accent": str(identity.get("accent") or ""),
                "occupation": str(identity.get("occupation") or ""),
            },
        }
```

- [ ] **Step 6: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_read_along_narrator_profile_defaults_to_editable_profile tests\test_ui_controller.py::test_controller_migrates_narrator_profile_from_legacy_registry -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```powershell
git add src\ebook_tts_pipeline\paths.py src\ebook_tts_pipeline\read_along\narrator_profile.py tests\test_ui_controller.py
git commit -m "Add read-along narrator profile storage"
```

---

### Task 2: Session-Start Narrator QVP Cache

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write failing tests for narrator cache reuse and regeneration**

Replace the old `test_controller_regenerates_narrator_voice_when_session_voice_type_changes` with:

```python
def test_controller_reuses_cached_narrator_voice_when_profile_hash_unchanged(tmp_path, monkeypatch):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Leigh waited.", encoding="utf-8")
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(paths.annotation("chapter_001"), {"schema": "quote_attribution_v1", "roles": [], "quotes": []})
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    profile = controller.read_along_narrator_profile()
    profile_hash = narrator_profile_hash(profile)
    narrator_path = paths.narrator_voice_qvp(profile_hash, "narrator")
    narrator_path.parent.mkdir(parents=True, exist_ok=True)
    narrator_path.write_bytes(b"cached narrator")
    calls = []

    def fail_if_called(self, role_id, voice_record, voice_path):
        calls.append((role_id, Path(voice_path)))
        raise AssertionError("narrator cache should be reused")

    monkeypatch.setattr(FakeTtsAdapter, "ensure_voice", fail_if_called)
    units = controller.build_read_along_units("chapter_001")

    session = controller.create_read_along_session(
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
        },
    )

    assert calls == []
    assert [unit.voice_config_path for unit in session.units] == [narrator_path.relative_to(paths.root).as_posix()]
    session.end()
```

```python
def test_controller_regenerates_narrator_voice_when_profile_changes(tmp_path, monkeypatch):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Leigh waited.", encoding="utf-8")
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(paths.annotation("chapter_001"), {"schema": "quote_attribution_v1", "roles": [], "quotes": []})
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    controller.save_read_along_narrator_profile(
        {
            "display_name": "Narrator",
            "age_stage": "adult",
            "gender": "female",
            "personality": "warm, steady",
            "accent": "American",
            "race_or_ethnicity": "",
            "occupation": "audiobook narrator",
        }
    )
    calls = []
    original_ensure_voice = FakeTtsAdapter.ensure_voice

    def recording_ensure_voice(self, role_id, voice_record, voice_path):
        calls.append((role_id, dict(voice_record), Path(voice_path)))
        return original_ensure_voice(self, role_id, voice_record, voice_path)

    monkeypatch.setattr(FakeTtsAdapter, "ensure_voice", recording_ensure_voice)
    units = controller.build_read_along_units("chapter_001")

    session = controller.create_read_along_session(
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
        },
    )

    narrator_calls = [call for call in calls if call[0] == "narrator"]
    assert narrator_calls
    assert "female" in narrator_calls[-1][1]["voice_profile"]["description"]
    assert "American accent" in narrator_calls[-1][1]["voice_profile"]["description"]
    assert narrator_calls[-1][2].as_posix().endswith("/voices/_narrator/" + narrator_profile_hash(controller.read_along_narrator_profile()) + "/narrator.qvp")
    session.end()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_reuses_cached_narrator_voice_when_profile_hash_unchanged tests\test_ui_controller.py::test_controller_regenerates_narrator_voice_when_profile_changes -q
```

Expected: fail because session start still uses `voices/narrator.qvp` and `narrator_voice_type`.

- [ ] **Step 3: Replace narrator session voice implementation**

In `PrototypeUiController._ensure_read_along_session_narrator_voices`, replace the body with:

```python
        profile = self.read_along_narrator_profile()
        narrator_record = narrator_voice_record(profile)
        profile_hash = voice_profile_hash(narrator_record)
        narrator_path = self.paths.narrator_voice_qvp(profile_hash, "narrator")
        narrator_rel = self._ensure_voice_asset(
            pipeline.tts_adapter,
            "narrator",
            narrator_record,
            narrator_path,
        )

        voice_paths = {"narrator": narrator_rel}
        if any(_is_functional_narrator_unit(unit) for unit in units):
            functional_record = functional_narrator_voice_record(profile)
            functional_hash = voice_profile_hash(functional_record)
            functional_path = self.paths.narrator_voice_qvp(functional_hash, FUNCTIONAL_NARRATOR_ROLE_ID)
            voice_paths[FUNCTIONAL_NARRATOR_ROLE_ID] = self._ensure_voice_asset(
                pipeline.tts_adapter,
                FUNCTIONAL_NARRATOR_ROLE_ID,
                functional_record,
                functional_path,
            )
        return voice_paths
```

Remove `_apply_read_along_narrator_voice_type` and `_validate_read_along_narrator_voice_type` once no call sites remain.

- [ ] **Step 4: Remove narrator type from read-along settings**

In `read_along_settings` and `save_read_along_settings`, remove `narrator_voice_type`. Update tests that expect the old field so saved settings contain only:

```python
{
    "playback_speed": 1.4,
    "generation_mode": "fast",
    "buffer_limit": 3,
    "target_buffer_seconds": 20.0,
    "start_buffer_seconds": 20.0,
    "max_buffer_seconds": 40.0,
    "max_buffer_units": 32,
}
```

- [ ] **Step 5: Update functional narrator test**

Update `test_controller_read_along_session_generates_functional_narrator_voice_at_session_start` assertions:

```python
profile = controller.read_along_narrator_profile()
narrator_hash = narrator_profile_hash(profile)
functional_record = functional_narrator_voice_record(profile)
functional_hash = voice_profile_hash(functional_record)
functional_path = paths.narrator_voice_qvp(functional_hash, "functional_narrator")

assert paths.narrator_voice_qvp(narrator_hash, "narrator").exists()
assert functional_path.exists()
assert patched_quote.voice_config_path == functional_path.relative_to(paths.root).as_posix()
```

- [ ] **Step 6: Run controller read-along tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```powershell
git add src\ebook_tts_pipeline\ui\controller.py tests\test_ui_controller.py
git commit -m "Cache read-along narrator voices by profile hash"
```

---

### Task 3: Narrator Profile Web API And Editor Shell

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write failing web API tests**

Add:

```python
def test_web_api_serves_and_saves_narrator_profile(tmp_path):
    server, thread, base_url = _start_test_server(tmp_path)
    try:
        profile = _get_json(base_url + "/api/narrator-profile")

        assert profile["ok"] is True
        assert profile["profile"]["role_id"] == "narrator"
        assert "summary" in profile

        saved = _post_json(
            base_url + "/api/narrator-profile",
            {
                "display_name": "Narrator",
                "age_stage": "adult",
                "gender": "female",
                "personality": "warm, precise",
                "accent": "American",
                "race_or_ethnicity": "",
                "occupation": "audiobook narrator",
            },
        )

        assert saved["ok"] is True
        assert saved["profile"]["identity_profile"]["gender"] == "female"
        assert saved["fields"]["personality"] == "warm, precise"
        assert "female" in saved["profile"]["voice_profile"]["description"]
    finally:
        _stop_server(server, thread)
```

Update `test_home_page_serves_clean_reader_shell`:

```python
assert 'id="narrator-summary"' in response
assert 'id="edit-narrator"' in response
assert 'id="narrator-panel"' in response
assert "<option value=\"male\">male</option>" not in response
assert "narrator_voice_type" not in response
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_web_api_serves_and_saves_narrator_profile tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -q
```

Expected: fail because endpoints and DOM elements do not exist and the old select still exists.

- [ ] **Step 3: Add web state methods**

Add to `ReadAlongWebState`:

```python
    def narrator_profile_payload(self) -> Dict[str, Any]:
        with self.lock:
            controller = self._require_controller()
            return {"ok": True, **controller.read_along_narrator_profile_payload()}

    def save_narrator_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.session is not None:
                raise ValueError("End the active read-along session before editing narrator.")
            controller = self._require_controller()
            controller.save_read_along_narrator_profile(payload)
            return {"ok": True, **controller.read_along_narrator_profile_payload()}
```

Add GET route:

```python
if path == "/api/narrator-profile":
    self._send_json(app_state.narrator_profile_payload())
    return
```

Add POST route:

```python
if path == "/api/narrator-profile":
    self._send_json(app_state.save_narrator_profile(payload))
    return
```

- [ ] **Step 4: Replace narrator select in HTML**

Replace:

```html
<label>Narrator
  <select id="narrator">
    <option value="male">male</option>
    <option value="female">female</option>
    <option value="current">current</option>
  </select>
</label>
```

with:

```html
<div class="narrator-control">
  <span id="narrator-summary">Narrator: loading</span>
  <button id="edit-narrator" type="button">Edit Narrator</button>
</div>
```

Add panel:

```html
<section class="narrator-panel" id="narrator-panel" hidden>
  <div class="registry-head">
    <h2>Narrator Voice</h2>
    <button id="narrator-close" type="button">Close</button>
  </div>
  <div class="registry-grid">
    <label>Name <input id="narrator-display-name" data-narrator-field="display_name" type="text"></label>
    <label>Age <input id="narrator-age-stage" data-narrator-field="age_stage" type="text"></label>
    <label>Gender <input id="narrator-gender" data-narrator-field="gender" type="text"></label>
    <label>Personality <input id="narrator-personality" data-narrator-field="personality" type="text"></label>
    <label>Race / Ethnicity <input id="narrator-race" data-narrator-field="race_or_ethnicity" type="text"></label>
    <label>Accent <input id="narrator-accent" data-narrator-field="accent" type="text"></label>
    <label>Occupation <input id="narrator-occupation" data-narrator-field="occupation" type="text"></label>
  </div>
  <div class="registry-actions">
    <button id="save-narrator" type="button">Save Narrator</button>
  </div>
</section>
```

- [ ] **Step 5: Update JavaScript settings and profile load/save**

Remove `narrator` from `els`, `settings()`, `lockControls`, `loadState`, and Save settings response handling.

Add:

```javascript
async function loadNarratorProfile() {
  const payload = await api("/api/narrator-profile");
  state.narratorProfile = payload;
  els.narratorSummary.textContent = "Narrator: " + payload.summary;
  for (const input of document.querySelectorAll("[data-narrator-field]")) {
    input.value = payload.fields[input.dataset.narratorField] || "";
  }
}

async function saveNarratorProfile() {
  const fields = {};
  for (const input of document.querySelectorAll("[data-narrator-field]")) {
    fields[input.dataset.narratorField] = input.value;
  }
  const payload = await api("/api/narrator-profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields)
  });
  state.narratorProfile = payload;
  els.narratorSummary.textContent = "Narrator: " + payload.summary;
  setStatus("Narrator profile saved. It will be used when the next session starts.");
}
```

Call `await loadNarratorProfile()` in `loadState()` after settings load.

- [ ] **Step 6: Run web tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```powershell
git add src\ebook_tts_pipeline\ui\web_app.py tests\test_read_along_web_app.py
git commit -m "Add web narrator profile editor"
```

---

### Task 4: Adaptive Reader Pagination UI

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write failing shell test for pagination controls**

Add assertions to `test_home_page_serves_clean_reader_shell`:

```python
assert 'id="toggle-sidebar"' in response
assert 'id="page-prev"' in response
assert 'id="page-next"' in response
assert 'id="page-indicator"' in response
assert 'id="page-measurer"' in response
assert "renderPages()" in response
assert "ensureAnchorPage" in response
assert "state.sidebarOpen" in response
assert "keydown" in response
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -q
```

Expected: fail because pagination controls/functions are absent.

- [ ] **Step 3: Add layout HTML**

In the reader shell:

```html
<button id="toggle-sidebar" type="button" title="Show or hide chapters">Chapters</button>
```

Add navigation around the page area:

```html
<div class="page-shell" id="page-shell">
  <button class="page-nav page-nav-prev" id="page-prev" type="button" aria-label="Previous page">‹</button>
  <div class="page-wrap" id="page-wrap">
    <div class="page-spread" id="page-spread"></div>
  </div>
  <button class="page-nav page-nav-next" id="page-next" type="button" aria-label="Next page">›</button>
</div>
<div class="page-indicator" id="page-indicator">Page 1</div>
<div class="page-measurer" id="page-measurer" aria-hidden="true"></div>
```

Remove the old direct `<article class="page" id="reader-text"></article>` from the visible page.

- [ ] **Step 4: Add CSS**

Add:

```css
.app.sidebar-hidden { grid-template-columns: 0 minmax(0, 1fr); }
.app.sidebar-hidden .sidebar { width: 0; overflow: hidden; border-right: 0; }
.page-shell { position: relative; min-height: calc(100vh - 190px); display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: 12px; padding: 24px clamp(12px, 3vw, 42px); }
.page-wrap { overflow: hidden; padding: 0; }
.page-spread { display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; max-width: 920px; margin: 0 auto; }
.app.sidebar-hidden .page-spread { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); max-width: 1320px; }
.page-nav { width: 38px; min-height: 56px; font-size: 26px; }
.page-indicator { padding: 0 16px 8px; color: var(--muted); font-size: 13px; text-align: center; }
.page-measurer { position: fixed; left: -10000px; top: 0; visibility: hidden; pointer-events: none; }
```

- [ ] **Step 5: Add pagination state and functions**

Extend `state`:

```javascript
pages: [],
unitPage: {},
pageIndex: 0,
sidebarOpen: true,
currentUnitId: null,
```

Add functions:

```javascript
function visiblePageCount() {
  return state.sidebarOpen ? 1 : 2;
}

function ensureAnchorPage(unitId = null) {
  const anchor = unitId ?? state.currentUnitId ?? state.selectedUnitId;
  const page = state.unitPage[String(anchor)];
  if (Number.isFinite(Number(page))) {
    state.pageIndex = Math.max(0, Math.min(Number(page), Math.max(0, state.pages.length - 1)));
    if (!state.sidebarOpen && state.pageIndex % 2 === 1) state.pageIndex -= 1;
  }
}

function turnPage(delta) {
  if (state.sessionActive) return;
  const step = visiblePageCount();
  state.pageIndex = Math.max(0, Math.min(state.pages.length - 1, state.pageIndex + delta * step));
  renderPages();
}
```

- [ ] **Step 6: Implement measured pages**

Replace `renderText()` with:

```javascript
function renderText() {
  paginateChapter();
  ensureAnchorPage();
  renderPages();
}
```

Add `paginateChapter()` that creates logical pages by measuring `.page` height in `els.pageMeasurer`. Keep the algorithm unit-boundary based:

```javascript
function paginateChapter() {
  const pageWidth = Math.max(320, els.pageWrap.clientWidth / visiblePageCount() - 24);
  els.pageMeasurer.style.width = pageWidth + "px";
  state.pages = [];
  state.unitPage = {};
  let current = [];
  let page = document.createElement("article");
  page.className = "page";
  els.pageMeasurer.textContent = "";
  els.pageMeasurer.appendChild(page);
  let cursor = 0;
  for (const unit of [...state.units].sort((a, b) => a.source_start - b.source_start)) {
    const before = state.text.slice(cursor, unit.source_start);
    const unitText = state.text.slice(unit.source_start, unit.source_end);
    const fragment = { before, unit };
    appendMeasuredFragment(page, fragment, unitText);
    if (page.scrollHeight > page.clientHeight && current.length) {
      state.pages.push(current);
      current = [];
      page = document.createElement("article");
      page.className = "page";
      els.pageMeasurer.textContent = "";
      els.pageMeasurer.appendChild(page);
      appendMeasuredFragment(page, fragment, unitText);
    }
    state.unitPage[String(unit.unit_id)] = state.pages.length;
    current.push(fragment);
    cursor = unit.source_end;
  }
  const tail = state.text.slice(cursor);
  if (tail) current.push({ before: tail, unit: null });
  state.pages.push(current);
}
```

Add helpers `appendMeasuredFragment`, `renderPages`, and `renderPageArticle` using the existing unit span click behavior.

- [ ] **Step 7: Wire toggle, arrows, playback anchor**

Add event handlers:

```javascript
els.toggleSidebar.onclick = () => {
  state.sidebarOpen = !state.sidebarOpen;
  document.getElementById("reader-view").classList.toggle("sidebar-hidden", !state.sidebarOpen);
  renderText();
};
els.pagePrev.onclick = () => turnPage(-1);
els.pageNext.onclick = () => turnPage(1);
window.addEventListener("keydown", event => {
  const tag = String(document.activeElement && document.activeElement.tagName || "").toLowerCase();
  if (state.sessionActive || ["input", "select", "button", "textarea"].includes(tag)) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); turnPage(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); turnPage(1); }
});
```

In `playReady()`, before `highlight(item.unit_id)`:

```javascript
state.currentUnitId = item.unit_id;
ensureAnchorPage(item.unit_id);
renderPages();
```

- [ ] **Step 8: Run web tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -q
```

Expected: pass.

- [ ] **Step 9: Commit**

```powershell
git add src\ebook_tts_pipeline\ui\web_app.py tests\test_read_along_web_app.py
git commit -m "Add adaptive read-along pagination"
```

---

### Task 5: Session Error Visibility And Full Verification

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Add shell assertion for visible session errors**

In `test_web_interface_exposes_tts_loading_overlay_and_selection_outline`, add:

```python
assert 'id="session-error"' in response
assert "showSessionError" in response
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_web_interface_exposes_tts_loading_overlay_and_selection_outline -q
```

Expected: fail because visible session error UI is missing.

- [ ] **Step 3: Add error panel and use it in Start Session catch block**

Add near the TTS overlay:

```html
<div class="session-error" id="session-error" hidden></div>
```

Add CSS:

```css
.session-error { margin: 8px 16px; padding: 10px 12px; border: 1px solid #d79090; background: #fff1f1; color: #7a1f1f; border-radius: 6px; white-space: pre-wrap; overflow-wrap: anywhere; }
```

Add JS:

```javascript
function showSessionError(message) {
  const text = String(message || "");
  els.sessionError.hidden = !text;
  els.sessionError.textContent = text;
}
```

Call `showSessionError("")` at session start and `showSessionError(error.message)` in the catch block.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py tests\test_ui_controller.py -q
```

Expected: pass.

- [ ] **Step 5: Run full suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src\ebook_tts_pipeline\ui\web_app.py tests\test_read_along_web_app.py
git commit -m "Show read-along session start errors"
```

---

## Final Manual Check

- [ ] Start the app:

```powershell
readalongweb books --no-open
```

- [ ] Open the printed URL.
- [ ] Select False Witness.
- [ ] Confirm the reader opens without downloading models.
- [ ] Confirm `Edit Narrator` opens the narrator profile editor.
- [ ] Save a narrator profile change and confirm no QVP is generated until Start Session.
- [ ] Start a short fake or real session and confirm narrator QVP paths are under `voices/_narrator/<hash>/`.
- [ ] Hide the sidebar and confirm two-page spread.
- [ ] Show the sidebar and confirm single-page mode.
- [ ] Click a segment, toggle sidebar, and confirm the selected segment remains visible.
- [ ] During playback, confirm the page advances when the current segment crosses a page boundary.

## Self-Review

Spec coverage:

- Narrator profile storage: Task 1.
- Hash-based narrator QVP reuse/regeneration: Task 2.
- Functional narrator derivative cache: Task 2.
- Remove narrator from global voice readiness: Task 2 preserves existing global-only generation tests.
- Narrator web editor: Task 3.
- Adaptive sidebar spread/single page layout: Task 4.
- Active segment page anchoring: Task 4.
- Visible session-start errors: Task 5.

Completeness scan: all implementation sections are concrete.

Type consistency: `read_along_narrator_profile`, `narrator_voice_qvp`, `read_along_narrator_profile_payload`, `save_read_along_narrator_profile`, `narrator_profile_hash`, `narrator_voice_record`, and `functional_narrator_voice_record` are introduced before later tasks use them.
