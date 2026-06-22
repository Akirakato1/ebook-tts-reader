# Sentence Segmentation For Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic sentence segmentation that produces clean role-allocation units for the LLM, so Haiku labels speakers instead of untangling mixed speaker structure.

**Architecture:** NLTK is only the first pass. A deterministic dialogue unit builder runs after NLTK and before annotation. It splits a source sentence whenever it contains multiple non-narrator role sections, usually multiple quoted dialogue turns, while keeping nearby narrator tags/actions attached to the closest quote for context. After annotation, script generation may extract narrator text from an already-labeled unit, but it must not infer or change the quote speaker.

**Tech Stack:** Python 3, existing `Sentence`, `SentenceUnit`, and `SentenceArtifact` dataclasses, pytest, NLTK sentence tokenization plus deterministic quote/role-unit refinement.

---

## Segmentation Contract

The LLM should receive units that are easy to role-allocate:

- A unit can contain narrator context plus one non-narrator role section.
- A unit must not contain two independent non-narrator role sections that might belong to different speakers.
- Multiple adjacent quotes in one NLTK sentence become multiple annotation units.
- Speech tags and short action beats in the same source sentence stay attached to the nearest quote when they provide speaker context.
- Pure narrator sentences remain narrator units.
- Post-annotation/script processing may split narrator text out of an annotated unit for TTS, but it cannot decide or change who spoke the quoted text.

Examples:

```text
Input:
"I found this for you." "Wonderful, thank you." Callie took the book.

Annotation units:
[0] "I found this for you."
[1] "Wonderful, thank you." Callie took the book.

Input:
Walter said, "I like your jacket." "It's from high school." Callie turned around.

Annotation units:
[0] Walter said, "I like your jacket."
[1] "It's from high school." Callie turned around.
```

This keeps context for Haiku while avoiding a unit with two possible speakers.

---

## File Structure

- Modify: `src/ebook_tts_pipeline/ingestion.py`
  - Owns deterministic conversion from NLTK sentences to role-allocation annotation units.
- Modify: `src/ebook_tts_pipeline/annotation/prompts.py`
  - Explain that units may include narrator context attached to one speaker-bearing section.
- Modify: `src/ebook_tts_pipeline/tts/script.py`
  - Replace speaker-inference postprocessing with narrator-text extraction that preserves annotated quote speaker.
- Modify: `src/ebook_tts_pipeline/pipeline.py`
  - Remove annotation postprocess speaker correction.
- Delete: `src/ebook_tts_pipeline/annotation/postprocess.py`
  - This module is the wrong layer because it infers speakers after annotation.
- Modify: `tests/test_ingestion.py`
  - Define the role-allocation unit contract.
- Modify: `tests/test_tts_script.py`
  - Replace speaker-inference tests with narrator-extraction tests.
- Modify: `tests/test_annotation_prompts.py`
  - Lock prompt wording to the new contract.

---

### Task 1: Add Red Tests For Role-Allocation Segmentation

**Files:**
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Add adjacent quote split test**

Append:

```python
def test_sentence_segmenter_splits_adjacent_quotes_into_role_allocation_units(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        '"I found this for you." "Wonderful, thank you." Callie took the book.',
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: ['"I found this for you." "Wonderful, thank you." Callie took the book.']
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": '"I found this for you."'},
        {"idx": 1, "sentence_idx": 0, "text": '"Wonderful, thank you." Callie took the book.'},
    ]
```

- [ ] **Step 2: Add trailing tag context test**

Append:

```python
def test_sentence_segmenter_keeps_trailing_tag_with_quote_context(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        '"Go," Callie said, looking away. "Now."',
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: ['"Go," Callie said, looking away. "Now."']
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": '"Go," Callie said, looking away.'},
        {"idx": 1, "sentence_idx": 0, "text": '"Now."'},
    ]
```

- [ ] **Step 3: Add leading tag and next quote test**

Append:

```python
def test_sentence_segmenter_keeps_leading_tag_with_following_quote(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        'Walter said, "I like your jacket." "It is from high school." Callie turned around.',
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: [
            'Walter said, "I like your jacket." "It is from high school." Callie turned around.'
        ]
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": 'Walter said, "I like your jacket."'},
        {"idx": 1, "sentence_idx": 0, "text": '"It is from high school." Callie turned around.'},
    ]
```

- [ ] **Step 4: Add smart quote version**

Append:

```python
def test_sentence_segmenter_handles_smart_quote_role_units(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        "\u201cWelcome, friend.\u201d Callie smiled. \u201cThank you.\u201d",
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: [
            "\u201cWelcome, friend.\u201d Callie smiled. \u201cThank you.\u201d"
        ]
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": "\u201cWelcome, friend.\u201d Callie smiled."},
        {"idx": 1, "sentence_idx": 0, "text": "\u201cThank you.\u201d"},
    ]
```

- [ ] **Step 5: Add open quote continuation test**

Append:

```python
def test_sentence_segmenter_preserves_open_quote_continuations_as_role_units(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        "\u201cI could not see your face. You never looked up. "
        "You did what she told you to do.\u201d It was almost a relief.",
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: [
            "\u201cI could not see your face.",
            "You never looked up.",
            "You did what she told you to do.\u201d It was almost a relief.",
        ]
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": "\u201cI could not see your face."},
        {"idx": 1, "sentence_idx": 1, "text": "You never looked up."},
        {
            "idx": 2,
            "sentence_idx": 2,
            "text": "You did what she told you to do.\u201d It was almost a relief.",
        },
    ]
```

- [ ] **Step 6: Run the new tests and verify red**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ingestion.py -k "role_allocation or trailing_tag or leading_tag or smart_quote_role_units or open_quote_continuations" -q
```

Expected: FAIL, because current segmentation splits quote and narrator fragments instead of producing role-allocation units.

**Review Gate:** Stop after the red run. Confirm with the user that these expected units are the contract.

---

### Task 2: Implement Role-Allocation Unit Builder

**Files:**
- Modify: `src/ebook_tts_pipeline/ingestion.py`

- [ ] **Step 1: Add internal quote fragment type**

Add near existing splitter helpers:

```python
@dataclass(frozen=True)
class QuoteFragment:
    text: str
    kind: str  # "quote" or "narration"
```

Add constants:

```python
QUOTE_PAIRS = {
    '"': '"',
    "\u201c": "\u201d",
}

CLOSE_QUOTES = {'"', "\u201d"}
```

- [ ] **Step 2: Add quote scanner with open quote state**

Add:

```python
def _scan_quote_fragments(text: str, starts_in_quote: bool = False) -> tuple[List[QuoteFragment], bool]:
    fragments: List[QuoteFragment] = []
    current: List[str] = []
    in_quote = starts_in_quote
    quote_close = ""
    current_kind = "quote" if starts_in_quote else "narration"

    for char in text:
        if not in_quote and char in QUOTE_PAIRS:
            _append_quote_fragment(fragments, current, current_kind)
            current = [char]
            in_quote = True
            quote_close = QUOTE_PAIRS[char]
            current_kind = "quote"
            continue

        current.append(char)
        if in_quote and (char == quote_close or (starts_in_quote and char in CLOSE_QUOTES)):
            _append_quote_fragment(fragments, current, current_kind)
            current = []
            in_quote = False
            quote_close = ""
            current_kind = "narration"

    _append_quote_fragment(fragments, current, current_kind)
    return fragments, in_quote


def _append_quote_fragment(fragments: List[QuoteFragment], current: List[str], kind: str) -> None:
    text = "".join(current).strip()
    if text:
        fragments.append(QuoteFragment(text=text, kind=kind))
```

- [ ] **Step 3: Add role-unit assembler**

Add:

```python
def _role_units_from_fragments(fragments: List[QuoteFragment]) -> List[str]:
    if not fragments:
        return []
    if not any(fragment.kind == "quote" for fragment in fragments):
        return [_join_nonempty(fragment.text for fragment in fragments)]

    units: List[str] = []
    pending_narration = ""

    for fragment in fragments:
        if fragment.kind == "narration":
            pending_narration = _join_unit_text(pending_narration, fragment.text)
            continue

        quote_text = fragment.text
        if pending_narration:
            if units:
                units[-1] = _join_unit_text(units[-1], pending_narration)
            else:
                quote_text = _join_unit_text(pending_narration, quote_text)
            pending_narration = ""
        units.append(quote_text)

    if pending_narration:
        if units:
            units[-1] = _join_unit_text(units[-1], pending_narration)
        else:
            units.append(pending_narration)

    return units


def _join_nonempty(parts: List[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def _join_unit_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"
```

This intentionally attaches narration between quotes to the previous quote when a previous quote exists. It attaches leading narration to the following quote. This keeps context while ensuring one role-bearing quote section per unit.

- [ ] **Step 4: Replace `split_sentence_units()`**

Replace the body with:

```python
def split_sentence_units(sentences: List[Sentence]) -> List[SentenceUnit]:
    units: List[SentenceUnit] = []
    quote_open = False
    for sentence in sentences:
        fragments, quote_open = _scan_quote_fragments(sentence.text, starts_in_quote=quote_open)
        for text in _role_units_from_fragments(fragments):
            units.append(SentenceUnit(idx=len(units), sentence_idx=sentence.idx, text=text))
    return units
```

- [ ] **Step 5: Keep `split_dialogue_embedded_text()` as compatibility wrapper**

Replace it with:

```python
def split_dialogue_embedded_text(text: str) -> List[str]:
    fragments, _ = _scan_quote_fragments(text)
    return _role_units_from_fragments(fragments)
```

- [ ] **Step 6: Delete old quote fragment helpers**

Delete:

```python
_quote_split_fragments
_append_fragment
```

- [ ] **Step 7: Run ingestion tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ingestion.py -q
```

Expected: PASS.

**Review Gate:** Print the five new test outputs and confirm they match the role-allocation contract.

---

### Task 3: Update Annotation Prompt For Role-Allocation Units

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/prompts.py`
- Modify: `tests/test_annotation_prompts.py`

- [ ] **Step 1: Add prompt test**

Add:

```python
def test_annotation_prompt_describes_role_allocation_units():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text='"Hello." Alice smiled.')],
        {"characters": {}},
        lock_registry=True,
    )

    assert "Each annotation unit contains at most one non-narrator speaker section." in prompt
    assert "If a unit contains quoted speech plus narrator context" in prompt
    assert "Do not split or merge unit_idx values in your output." in prompt
```

- [ ] **Step 2: Run prompt test and verify red**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_annotation_prompts.py -k role_allocation_units -q
```

Expected: FAIL.

- [ ] **Step 3: Replace old segmentation bullets**

In `render_annotation_prompt()`, replace the old bullets:

```python
"- The chapter text has already been split into annotation units. A single source sentence can have multiple units.\n"
"- If an input unit is narration around dialogue, such as said-tags or action beats outside quotes, label it Narrator/narration.\n"
"- If an input unit is quoted external speech, label it as the speaking character/dialogue.\n"
```

with:

```python
"- Each annotation unit contains at most one non-narrator speaker section.\n"
"- A unit may include quoted speech plus narrator context, such as said-tags or short action beats.\n"
"- If a unit contains quoted speech plus narrator context, label the unit as the quoted speaker/dialogue.\n"
"- If a unit contains no quoted speech or thought, label it Narrator/narration.\n"
"- Do not split or merge unit_idx values in your output.\n"
```

- [ ] **Step 4: Run prompt tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_annotation_prompts.py -q
```

Expected: PASS.

---

### Task 4: Replace Speaker-Inference Postprocessing With Narrator Extraction

**Files:**
- Modify: `src/ebook_tts_pipeline/pipeline.py`
- Modify: `src/ebook_tts_pipeline/tts/script.py`
- Delete: `src/ebook_tts_pipeline/annotation/postprocess.py`
- Modify: `tests/test_tts_script.py`

- [ ] **Step 1: Remove annotation speaker correction**

In `pipeline.py`, delete:

```python
from ebook_tts_pipeline.annotation.postprocess import normalize_mixed_dialogue_units
```

Delete these calls:

```python
merged = normalize_mixed_dialogue_units(merged, artifact, self.registry.load())
annotation = normalize_mixed_dialogue_units(annotation, artifact, registry)
```

- [ ] **Step 2: Remove postprocess module**

Run:

```powershell
git rm src\ebook_tts_pipeline\annotation\postprocess.py
```

- [ ] **Step 3: Delete speaker-inference tests**

In `tests/test_tts_script.py`, delete:

```python
test_tts_script_normalizes_haiku_mislabeled_mixed_quote_units
test_tts_script_splits_quote_continuations_inside_annotated_units
test_tts_script_uses_local_speaker_for_pronoun_tag_with_intervening_adverb
```

- [ ] **Step 4: Add narrator extraction test**

Add:

```python
def test_tts_script_extracts_narrator_context_after_annotation_without_changing_quote_speaker():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[Sentence(0, 'Walter said, "I like your jacket."')],
        units=[SentenceUnit(0, 0, 'Walter said, "I like your jacket."')],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Walter"],
        types=["narration", "dialogue", "thought"],
        script=[(1, 1, 0)],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            "walter": {
                "role_id": "walter",
                "display_name": "Walter",
                "aliases": [],
                "voice_variants": {
                    "default": {
                        "role_id": "walter_default",
                        "display_name": "Walter_default",
                        "voice_config_path": "voices/walter_default.qvp",
                        "voice_profile": {"qwen_instruct": "Walter aloud."},
                    }
                },
            }
        },
    }

    script = build_tts_script(
        chapter="chapter_001",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [(job.role, job.type, job.text) for job in script.jobs] == [
        ("Narrator", "narration", "Walter said,"),
        ("Walter_default", "dialogue", '"I like your jacket."'),
    ]
```

- [ ] **Step 5: Implement narrator-context extraction in `tts/script.py`**

Keep extraction local to script generation. It may split a resolved `TtsSentenceJob` into multiple jobs, but quoted fragments always use the already annotated role. It must not inspect names, pronouns, gender, registry aliases, or local speaker profiles to change speakers.

Use this helper shape:

```python
def _extract_narrator_context_jobs(
    job: TtsSentenceJob,
    narrator_effective: Dict[str, Any],
) -> List[TtsSentenceJob]:
    fragments, _ = _scan_quote_fragments(job.text)
    if len(fragments) <= 1:
        return [job]
    output: List[TtsSentenceJob] = []
    for fragment in fragments:
        if fragment.kind == "quote":
            output.append(replace(job, text=fragment.text, type="dialogue"))
        else:
            record = narrator_effective["voice_record"]
            output.append(
                replace(
                    job,
                    role=str(narrator_effective["role"]),
                    role_id=str(narrator_effective["role_id"]),
                    character=narrator_effective["character"],
                    voice_variant=narrator_effective["voice_variant"],
                    type="narration",
                    text=fragment.text,
                    voice_config_path=record.get("voice_config_path"),
                )
            )
    return output
```

Implementation can duplicate the small quote scanner in `tts/script.py` or move the scanner to a shared internal module if that keeps files cleaner. Do not depend on `annotation/postprocess.py`.

- [ ] **Step 6: Preserve duplicate `unit_idx` order safely**

If narrator extraction creates multiple jobs with the same `unit_idx`, keep the `_job_order` windowing pattern from `b2e764c` so batching does not collapse duplicate unit IDs. This is ordering only, not speaker inference.

- [ ] **Step 7: Run TTS tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tts_script.py -q
```

Expected: PASS.

**Review Gate:** Confirm `rg -n "normalize_mixed_dialogue|postprocess|pronoun|gender" src\ebook_tts_pipeline\tts src\ebook_tts_pipeline\pipeline.py` returns no speaker-inference postprocess references.

---

### Task 5: Regenerate And Review False Witness Chapter 13 Segments

**Files:**
- Runtime artifact only: `books/.../sentence_segments/chapter_013.sentences.json`

- [ ] **Step 1: Regenerate sentence segments only**

Run:

```powershell
.\.venv\Scripts\python.exe -m ebook_tts_pipeline.cli segment-chapter --book-root "C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\books\false_witness_slaughter_karin_london_2021_harpercollins_publishers_limited_isbn13_9780008303525_1ce09b1b85d25c2685f2802ed288ec8b_anna_s_archive" --chapter chapter_013
```

Expected: command exits `0`.

- [ ] **Step 2: Print critical unit windows**

Run:

```powershell
@'
import json
from pathlib import Path

root = Path(r"C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\books\false_witness_slaughter_karin_london_2021_harpercollins_publishers_limited_isbn13_9780008303525_1ce09b1b85d25c2685f2802ed288ec8b_anna_s_archive")
data = json.loads((root / "sentence_segments" / "chapter_013.sentences.json").read_text(encoding="utf-8"))
units = data["units"]

for needle in [
    "I found this for you in the return bin",
    "Wonderful, thank you",
    "Welcome, friend",
    "I like your jacket",
    "It\u2019s from high school",
    "Are you sure, man",
]:
    print("\nNEEDLE:", needle)
    for unit in units:
        if needle in unit["text"]:
            idx = unit["idx"]
            for neighbor in units[max(0, idx - 2): idx + 3]:
                print(neighbor["idx"], neighbor["sentence_idx"], repr(neighbor["text"]))
            break
'@ | .\.venv\Scripts\python.exe -
```

Expected shape:

```text
unit: "I found this for you in the return bin."
unit: "Wonderful, thank you." Callie took the thick paperback.

unit: He said, "Welcome, friend."
unit: Callie peeled off her mask ...

unit: Walter said, "I like your jacket."
unit: "It's from high school." Callie turned around ...
```

- [ ] **Step 3: Check for units with more than one independent quote**

Run:

```powershell
@'
import json
from pathlib import Path

root = Path(r"C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\books\false_witness_slaughter_karin_london_2021_harpercollins_publishers_limited_isbn13_9780008303525_1ce09b1b85d25c2685f2802ed288ec8b_anna_s_archive")
data = json.loads((root / "sentence_segments" / "chapter_013.sentences.json").read_text(encoding="utf-8"))
bad = []
for unit in data["units"]:
    text = unit["text"]
    opens = text.count('"') // 2 + text.count("\u201c")
    if opens > 1:
        bad.append(unit)
print("multi_quote_units=", len(bad))
for unit in bad[:20]:
    print(unit["idx"], unit["sentence_idx"], repr(unit["text"][:200]))
'@ | .\.venv\Scripts\python.exe -
```

Expected: `multi_quote_units= 0`, except false positives from non-dialogue quotation marks that are manually reviewed.

**Review Gate:** User reviews these segment examples before any new paid annotation run.

---

### Task 6: Full Verification And Commit

**Files:**
- All modified files.

- [ ] **Step 1: Run full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Check wrong-layer code is gone**

Run:

```powershell
rg -n "normalize_mixed_dialogue|annotation.postprocess|pronoun|gender" src\ebook_tts_pipeline\pipeline.py src\ebook_tts_pipeline\tts src\ebook_tts_pipeline\annotation
```

Expected: no references to speaker-inference postprocessing. Registry/profile code may still use `gender`; this check is scoped to pipeline, TTS, and annotation postprocess removal.

- [ ] **Step 3: Commit**

Run:

```powershell
git status --short
git add src\ebook_tts_pipeline\ingestion.py src\ebook_tts_pipeline\annotation\prompts.py src\ebook_tts_pipeline\pipeline.py src\ebook_tts_pipeline\tts\script.py tests\test_ingestion.py tests\test_annotation_prompts.py tests\test_tts_script.py
git commit -m "Segment role allocation units before annotation"
```

If `src\ebook_tts_pipeline\annotation\postprocess.py` was removed with `git rm`, it is already staged.

- [ ] **Step 4: Push**

Run:

```powershell
git push origin main
```

Expected: push succeeds to `main`.

---

## Self-Review

**Spec coverage:** The plan replaces NLTK-only segmentation with deterministic role-allocation units before annotation, removes speaker inference after annotation, keeps narrator text attached for LLM context, and limits post-annotation work to narrator extraction for TTS.

**Placeholder scan:** No placeholder sections remain.

**Type consistency:** The plan keeps the existing `SentenceUnit` serialized shape for token efficiency and compatibility. Duplicate `unit_idx` values are allowed only in generated TTS jobs after narrator extraction, not in annotation units.

**Review requirement:** The critical review point is Task 5. No paid annotation should be run until the regenerated chapter 13 units are shown and approved.
