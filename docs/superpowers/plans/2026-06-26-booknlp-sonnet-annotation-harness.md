# BookNLP Sonnet Annotation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel, test-harnessed quote annotation backend that uses cached BookNLP output plus compact Sonnet consolidation instead of sending whole chapters to Sonnet.

**Architecture:** The existing Sonnet-only `QuoteAttributionService` remains the production path. The new path runs BookNLP once per book into cached artifacts, maps BookNLP quote speakers onto the existing `q001` quote extraction, deterministically resolves registry aliases where safe, and calls Sonnet only for ambiguous or unresolved quote-role consolidation. The final output remains `quote_attribution_v1`, so existing validation, repair diagnostics, read-along unit construction, temp registry handling, and QVP generation remain the gate.

**Tech Stack:** Python 3.9+, pytest, existing `ebook_tts_pipeline` annotation modules, optional external BookNLP environment via `EBOOK_TTS_BOOKNLP_PYTHON`, Anthropic Sonnet for compact consolidation/repair, cached TSV/JSON artifacts under each book root.

---

## File Structure

- Create `src/ebook_tts_pipeline/annotation/booknlp_artifacts.py`
  - Define data classes for stitched text maps, BookNLP quote rows, entity rows, and parsed character clusters.
  - Parse BookNLP `.tokens`, `.quotes`, `.entities`, and `.book` files without importing BookNLP.

- Create `src/ebook_tts_pipeline/annotation/booknlp_runner.py`
  - Build a stitched whole-book input file from existing `chapters/*.txt`.
  - Run BookNLP through a configured external Python executable.
  - Cache outputs and skip rerun when input hash and model settings match.

- Create `src/ebook_tts_pipeline/annotation/booknlp_candidates.py`
  - Map BookNLP quote spans back to our `QuoteExtraction` quote IDs.
  - Produce per-chapter candidate rows with quote text, BookNLP speaker cluster, mention text, aliases, and confidence notes.

- Create `src/ebook_tts_pipeline/annotation/quote_consolidation.py`
  - Resolve candidates to exact global `role_id`, local speaker, or `narrator_quote`.
  - Call Sonnet only with compact candidate data when deterministic mapping is insufficient.
  - Reuse `validate_quote_attribution()` and repair prompt diagnostics.

- Modify `src/ebook_tts_pipeline/pipeline.py`
  - Add an optional alternate annotation service hook used only by harness/tests at first.
  - Keep `annotate_chapter()` default behavior unchanged unless explicitly configured.

- Modify `src/ebook_tts_pipeline/config.py`
  - Add BookNLP config fields and environment variables:
    - `EBOOK_TTS_ANNOTATION_BACKEND=sonnet|booknlp_harness`
    - `EBOOK_TTS_BOOKNLP_PYTHON`
    - `EBOOK_TTS_BOOKNLP_MODEL=small|big`
    - `EBOOK_TTS_BOOKNLP_CACHE_POLICY=reuse|refresh`

- Create `scripts/run_booknlp_annotation_harness.py`
  - Run the alternate backend against selected chapters.
  - Write comparison reports without touching production annotation files unless explicitly requested.

- Create tests:
  - `tests/test_booknlp_artifacts.py`
  - `tests/test_booknlp_candidates.py`
  - `tests/test_quote_consolidation.py`
  - `tests/test_booknlp_annotation_harness.py`

---

## Task 1: Parse Cached BookNLP Artifacts

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/booknlp_artifacts.py`
- Test: `tests/test_booknlp_artifacts.py`

- [ ] **Step 1: Write the failing parser test**

Create `tests/test_booknlp_artifacts.py`:

```python
from pathlib import Path

from ebook_tts_pipeline.annotation.booknlp_artifacts import parse_booknlp_quotes


def test_parse_booknlp_quotes_reads_core_speaker_fields(tmp_path):
    quotes_path = tmp_path / "demo.quotes"
    quotes_path.write_text(
        "quote_start\tquote_end\tmention_start\tmention_end\tmention_phrase\tchar_id\tquote\n"
        "10\t12\t13\t14\tMr. Pounds\t7\tThe apple of my eye.\n",
        encoding="utf-8",
    )

    rows = parse_booknlp_quotes(quotes_path)

    assert len(rows) == 1
    assert rows[0].quote_start_token == 10
    assert rows[0].quote_end_token == 12
    assert rows[0].mention_phrase == "Mr. Pounds"
    assert rows[0].character_id == "7"
    assert rows[0].quote_text == "The apple of my eye."
```

- [ ] **Step 2: Run the failing parser test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_artifacts.py::test_parse_booknlp_quotes_reads_core_speaker_fields -q
```

Expected: FAIL with `ModuleNotFoundError` for `booknlp_artifacts`.

- [ ] **Step 3: Implement the minimal artifact parser**

Create `src/ebook_tts_pipeline/annotation/booknlp_artifacts.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class BookNlpQuoteRow:
    quote_start_token: int
    quote_end_token: int
    mention_start_token: int
    mention_end_token: int
    mention_phrase: str
    character_id: str
    quote_text: str


def parse_booknlp_quotes(path: str | Path) -> List[BookNlpQuoteRow]:
    rows: List[BookNlpQuoteRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                BookNlpQuoteRow(
                    quote_start_token=int(row["quote_start"]),
                    quote_end_token=int(row["quote_end"]),
                    mention_start_token=int(row["mention_start"]),
                    mention_end_token=int(row["mention_end"]),
                    mention_phrase=str(row.get("mention_phrase") or ""),
                    character_id=str(row.get("char_id") or ""),
                    quote_text=str(row.get("quote") or ""),
                )
            )
    return rows
```

- [ ] **Step 4: Verify parser test passes**

Run the same pytest command. Expected: PASS.

---

## Task 2: Build Whole-Book Stitching and Chapter Offset Map

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/booknlp_artifacts.py`
- Test: `tests/test_booknlp_artifacts.py`

- [ ] **Step 1: Write the failing stitch map test**

Append:

```python
from ebook_tts_pipeline.annotation.booknlp_artifacts import stitch_chapters_for_booknlp


def test_stitch_chapters_records_char_offsets(tmp_path):
    chapters = {
        "chapter_001": "First chapter.",
        "chapter_002": "Second chapter.",
    }

    stitched = stitch_chapters_for_booknlp(chapters)

    assert stitched.text.startswith("First chapter.")
    assert "\n\nSecond chapter." in stitched.text
    assert stitched.chapter_offsets["chapter_001"].start == 0
    second_start = stitched.text.index("Second chapter.")
    assert stitched.chapter_offsets["chapter_002"].start == second_start
    assert stitched.chapter_offsets["chapter_002"].end == second_start + len("Second chapter.")
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_artifacts.py::test_stitch_chapters_records_char_offsets -q
```

Expected: FAIL because `stitch_chapters_for_booknlp` is missing.

- [ ] **Step 3: Implement stitch map**

Add to `booknlp_artifacts.py`:

```python
from typing import Dict, Mapping


@dataclass(frozen=True)
class ChapterCharOffset:
    chapter: str
    start: int
    end: int


@dataclass(frozen=True)
class StitchedBookText:
    text: str
    chapter_offsets: Dict[str, ChapterCharOffset]


def stitch_chapters_for_booknlp(chapters: Mapping[str, str]) -> StitchedBookText:
    pieces: List[str] = []
    offsets: Dict[str, ChapterCharOffset] = {}
    cursor = 0
    for chapter, text in chapters.items():
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        offsets[chapter] = ChapterCharOffset(chapter=chapter, start=start, end=cursor)
    return StitchedBookText(text="".join(pieces), chapter_offsets=offsets)
```

- [ ] **Step 4: Verify artifact tests**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_artifacts.py -q
```

Expected: all parser/stitch tests PASS.

---

## Task 3: Add Cached BookNLP Runner

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/booknlp_runner.py`
- Modify: `src/ebook_tts_pipeline/paths.py`
- Test: `tests/test_booknlp_annotation_harness.py`

- [ ] **Step 1: Write the failing cache-skip test**

Create `tests/test_booknlp_annotation_harness.py`:

```python
from ebook_tts_pipeline.annotation.booknlp_runner import BookNlpRunner, BookNlpRunnerConfig
from ebook_tts_pipeline.json_io import read_json
from ebook_tts_pipeline.paths import BookPaths


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def __call__(self, command, cwd):
        self.calls.append((list(command), cwd))


def test_booknlp_runner_reuses_cache_when_input_hash_matches(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text("chapter_001").write_text('"Hello," she said.', encoding="utf-8")
    executor = RecordingExecutor()
    runner = BookNlpRunner(
        BookNlpRunnerConfig(python_path="python", model="small", cache_policy="reuse"),
        executor=executor,
    )

    first = runner.run(paths)
    second = runner.run(paths)

    assert first.output_dir == second.output_dir
    assert len(executor.calls) == 1
    manifest = read_json(paths.booknlp_manifest)
    assert manifest["model"] == "small"
    assert manifest["input_hash"]
```

- [ ] **Step 2: Run the failing cache test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_annotation_harness.py::test_booknlp_runner_reuses_cache_when_input_hash_matches -q
```

Expected: FAIL because the runner and paths are missing.

- [ ] **Step 3: Add BookNLP paths**

Modify `src/ebook_tts_pipeline/paths.py`:

```python
@property
def booknlp_dir(self) -> Path:
    return self.root / "booknlp"

@property
def booknlp_input(self) -> Path:
    return self.booknlp_dir / "input.txt"

@property
def booknlp_manifest(self) -> Path:
    return self.booknlp_dir / "manifest.json"

@property
def booknlp_output_dir(self) -> Path:
    return self.booknlp_dir / "output"
```

- [ ] **Step 4: Implement the cached runner**

Create `src/ebook_tts_pipeline/annotation/booknlp_runner.py`:

```python
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from ebook_tts_pipeline.annotation.booknlp_artifacts import StitchedBookText, stitch_chapters_for_booknlp
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths


@dataclass(frozen=True)
class BookNlpRunnerConfig:
    python_path: str
    model: str = "small"
    cache_policy: str = "reuse"


@dataclass(frozen=True)
class BookNlpRunResult:
    output_dir: Path
    input_path: Path
    manifest_path: Path
    input_hash: str
    reused_cache: bool


Executor = Callable[[List[str], Path], None]


def _default_executor(command: List[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


class BookNlpRunner:
    def __init__(self, config: BookNlpRunnerConfig, executor: Executor = _default_executor) -> None:
        self.config = config
        self.executor = executor

    def run(self, paths: BookPaths) -> BookNlpRunResult:
        chapters = {
            chapter_path.stem: chapter_path.read_text(encoding="utf-8", errors="replace")
            for chapter_path in sorted((paths.root / "chapters").glob("*.txt"))
        }
        stitched = stitch_chapters_for_booknlp(chapters)
        input_hash = _hash_text(stitched.text)
        paths.booknlp_dir.mkdir(parents=True, exist_ok=True)
        paths.booknlp_output_dir.mkdir(parents=True, exist_ok=True)
        if self._cache_valid(paths, input_hash):
            return BookNlpRunResult(paths.booknlp_output_dir, paths.booknlp_input, paths.booknlp_manifest, input_hash, True)

        paths.booknlp_input.write_text(stitched.text, encoding="utf-8")
        write_json_atomic(
            paths.booknlp_manifest,
            {
                "input_hash": input_hash,
                "model": self.config.model,
                "cache_policy": self.config.cache_policy,
                "chapter_offsets": {
                    key: {"start": value.start, "end": value.end}
                    for key, value in stitched.chapter_offsets.items()
                },
            },
        )
        self.executor(self._command(paths), paths.root)
        return BookNlpRunResult(paths.booknlp_output_dir, paths.booknlp_input, paths.booknlp_manifest, input_hash, False)

    def _cache_valid(self, paths: BookPaths, input_hash: str) -> bool:
        if self.config.cache_policy != "reuse" or not paths.booknlp_manifest.exists():
            return False
        manifest = read_json(paths.booknlp_manifest)
        return manifest.get("input_hash") == input_hash and manifest.get("model") == self.config.model

    def _command(self, paths: BookPaths) -> List[str]:
        return [
            self.config.python_path,
            "-c",
            (
                "from booknlp.booknlp import BookNLP; "
                f"BookNLP('en', {{'pipeline':'entity,quote,coref','model':'{self.config.model}'}})"
                f".process(r'{paths.booknlp_input}', r'{paths.booknlp_output_dir}', 'book')"
            ),
        ]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

- [ ] **Step 5: Verify cache test passes**

Run the same pytest command. Expected: PASS.

---

## Task 4: Map BookNLP Quotes to Our Quote IDs

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/booknlp_candidates.py`
- Test: `tests/test_booknlp_candidates.py`

- [ ] **Step 1: Write failing candidate mapping test**

Create `tests/test_booknlp_candidates.py`:

```python
from ebook_tts_pipeline.annotation.booknlp_artifacts import BookNlpQuoteRow
from ebook_tts_pipeline.annotation.booknlp_candidates import map_booknlp_quotes_to_extraction
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue


def test_map_booknlp_quote_to_matching_extracted_quote_id():
    text = 'Mary paused. "The apple of my eye," Mr. Pounds said.'
    extraction = extract_quoted_dialogue(text)
    rows = [
        BookNlpQuoteRow(
            quote_start_token=3,
            quote_end_token=9,
            mention_start_token=10,
            mention_end_token=12,
            mention_phrase="Mr. Pounds",
            character_id="7",
            quote_text="The apple of my eye,",
        )
    ]

    candidates = map_booknlp_quotes_to_extraction("chapter_017", extraction, rows)

    assert len(candidates) == 1
    assert candidates[0].quote_idx == 1
    assert candidates[0].quote_id == "q001"
    assert candidates[0].booknlp_character_id == "7"
    assert candidates[0].mention_phrase == "Mr. Pounds"
```

- [ ] **Step 2: Run failing candidate mapping test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_candidates.py::test_map_booknlp_quote_to_matching_extracted_quote_id -q
```

Expected: FAIL because `booknlp_candidates` is missing.

- [ ] **Step 3: Implement text-normalized mapping**

Create `src/ebook_tts_pipeline/annotation/booknlp_candidates.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ebook_tts_pipeline.annotation.booknlp_artifacts import BookNlpQuoteRow
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction
from ebook_tts_pipeline.registry import normalize_name


@dataclass(frozen=True)
class QuoteAttributionCandidate:
    chapter: str
    quote_idx: int
    quote_id: str
    quote_text: str
    booknlp_character_id: str
    mention_phrase: str
    source: str = "booknlp"


def map_booknlp_quotes_to_extraction(
    chapter: str,
    extraction: QuoteExtraction,
    rows: List[BookNlpQuoteRow],
) -> List[QuoteAttributionCandidate]:
    unmatched = list(extraction.quotes)
    candidates: List[QuoteAttributionCandidate] = []
    for row in rows:
        row_key = _quote_key(row.quote_text)
        match = next((quote for quote in unmatched if _quote_key(quote.text) == row_key), None)
        if match is None:
            continue
        unmatched.remove(match)
        candidates.append(
            QuoteAttributionCandidate(
                chapter=chapter,
                quote_idx=match.idx,
                quote_id=match.quote_id,
                quote_text=match.text,
                booknlp_character_id=row.character_id,
                mention_phrase=row.mention_phrase,
            )
        )
    return candidates


def _quote_key(text: str) -> str:
    stripped = str(text).strip().strip("\"'\"\u201c\u201d\u2018\u2019")
    return normalize_name(stripped)
```

- [ ] **Step 4: Verify candidate mapping test passes**

Run the same pytest command. Expected: PASS.

---

## Task 5: Deterministic Registry Consolidation

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/quote_consolidation.py`
- Test: `tests/test_quote_consolidation.py`

- [ ] **Step 1: Write failing deterministic consolidation tests**

Create `tests/test_quote_consolidation.py`:

```python
from ebook_tts_pipeline.annotation.booknlp_candidates import QuoteAttributionCandidate
from ebook_tts_pipeline.annotation.quote_consolidation import consolidate_candidates_deterministically


def test_consolidation_maps_unique_short_honorific_to_registry_role():
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": ["Mr John Pounds adult"],
            }
        }
    }
    candidates = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye."',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]

    result = consolidate_candidates_deterministically(candidates, registry)

    assert result.resolved_quotes == {1: "mr_john_pounds_adult"}
    assert result.unresolved == []


def test_consolidation_leaves_ambiguous_short_honorific_unresolved():
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": [],
            },
            "mr_james_pounds_adult": {
                "role_id": "mr_james_pounds_adult",
                "display_name": "Mr James Pounds",
                "age_stage": "adult",
                "aliases": [],
            },
        }
    }
    candidates = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye."',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]

    result = consolidate_candidates_deterministically(candidates, registry)

    assert result.resolved_quotes == {}
    assert result.unresolved[0].quote_id == "q001"
    assert sorted(result.unresolved[0].candidate_role_ids) == ["mr_james_pounds_adult", "mr_john_pounds_adult"]
```

- [ ] **Step 2: Run failing consolidation tests**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_quote_consolidation.py -q
```

Expected: FAIL because `quote_consolidation` is missing.

- [ ] **Step 3: Implement deterministic consolidation**

Create `src/ebook_tts_pipeline/annotation/quote_consolidation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ebook_tts_pipeline.annotation.booknlp_candidates import QuoteAttributionCandidate
from ebook_tts_pipeline.annotation.quote_attribution import _registry_role_alias_candidates
from ebook_tts_pipeline.registry import normalize_name


@dataclass(frozen=True)
class UnresolvedQuoteCandidate:
    quote_idx: int
    quote_id: str
    quote_text: str
    mention_phrase: str
    candidate_role_ids: List[str]


@dataclass(frozen=True)
class DeterministicConsolidationResult:
    resolved_quotes: Dict[int, str]
    unresolved: List[UnresolvedQuoteCandidate]


def consolidate_candidates_deterministically(
    candidates: List[QuoteAttributionCandidate],
    registry: Dict,
) -> DeterministicConsolidationResult:
    alias_candidates = _registry_role_alias_candidates(registry)
    resolved: Dict[int, str] = {}
    unresolved: List[UnresolvedQuoteCandidate] = []
    for candidate in candidates:
        possible = sorted(alias_candidates.get(normalize_name(candidate.mention_phrase), set()))
        if len(possible) == 1:
            resolved[candidate.quote_idx] = possible[0]
            continue
        unresolved.append(
            UnresolvedQuoteCandidate(
                quote_idx=candidate.quote_idx,
                quote_id=candidate.quote_id,
                quote_text=candidate.quote_text,
                mention_phrase=candidate.mention_phrase,
                candidate_role_ids=possible,
            )
        )
    return DeterministicConsolidationResult(resolved_quotes=resolved, unresolved=unresolved)
```

- [ ] **Step 4: Verify consolidation tests pass**

Run the same pytest command. Expected: PASS.

---

## Task 6: Compact Sonnet Consolidation for Unresolved Quotes

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/quote_consolidation.py`
- Test: `tests/test_quote_consolidation.py`

- [ ] **Step 1: Write failing compact prompt test**

Append:

```python
from ebook_tts_pipeline.annotation.quote_consolidation import render_consolidation_prompt


def test_render_consolidation_prompt_uses_compact_quote_table_not_full_chapter():
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": ["Mr John Pounds adult"],
            }
        }
    }
    unresolved = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye."',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]

    prompt = render_consolidation_prompt("chapter_017", unresolved, registry)

    assert "Quote candidates to consolidate" in prompt
    assert "q001" in prompt
    assert "Mr. Pounds" in prompt
    assert "mr_john_pounds_adult" in prompt
    assert "Return JSON only" in prompt
    assert "Chapter text with marked quotes" not in prompt
```

- [ ] **Step 2: Run failing prompt test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_quote_consolidation.py::test_render_consolidation_prompt_uses_compact_quote_table_not_full_chapter -q
```

Expected: FAIL because `render_consolidation_prompt` is missing.

- [ ] **Step 3: Implement compact consolidation prompt**

Add:

```python
import json

from ebook_tts_pipeline.annotation.registry_summary import compact_registry_for_prompt


def render_consolidation_prompt(
    chapter: str,
    candidates: List[QuoteAttributionCandidate],
    registry: Dict,
) -> str:
    compact_candidates = [
        {
            "quote_idx": candidate.quote_idx,
            "quote_id": candidate.quote_id,
            "quote_text": candidate.quote_text,
            "booknlp_character_id": candidate.booknlp_character_id,
            "booknlp_mention_phrase": candidate.mention_phrase,
        }
        for candidate in candidates
    ]
    return (
        "You consolidate local BookNLP quote-speaker candidates into exact audiobook registry roles.\n\n"
        f"Chapter: {chapter}\n\n"
        "Global registry role_ids are authoritative:\n"
        f"{json.dumps(compact_registry_for_prompt(registry, include_aliases=True), ensure_ascii=False, indent=2)}\n\n"
        "Quote candidates to consolidate:\n"
        f"{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "- Use an exact global registry role_id when the BookNLP mention refers to a registry character.\n"
        "- If the speaker is not in the global registry, create a chapter-local local_speakers entry.\n"
        "- If the quote is not character dialogue, mark it narrator_quote.\n"
        "- Do not invent global role_ids.\n"
        "- Return JSON only using quote_attribution_v1 fields: roles, local_speakers, quotes.\n"
    )
```

- [ ] **Step 4: Verify prompt test passes**

Run the same pytest command. Expected: PASS.

---

## Task 7: Harness Service Produces Valid `quote_attribution_v1`

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/quote_consolidation.py`
- Test: `tests/test_booknlp_annotation_harness.py`

- [ ] **Step 1: Write failing harness service test**

Append:

```python
from ebook_tts_pipeline.annotation.booknlp_candidates import QuoteAttributionCandidate
from ebook_tts_pipeline.annotation.quote_consolidation import BookNlpSonnetConsolidationService
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue


class NoCallClient:
    def complete_json(self, system_prompt, user_prompt):
        raise AssertionError("Sonnet should not be called for deterministic mappings")


def test_harness_service_writes_valid_annotation_without_sonnet_for_unique_match(tmp_path):
    chapter_text = 'Mary paused. "The apple of my eye," Mr. Pounds said.'
    extraction = extract_quoted_dialogue(chapter_text)
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": ["Mr John Pounds adult"],
            }
        }
    }
    candidates = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye,"',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]
    service = BookNlpSonnetConsolidationService(NoCallClient())

    result = service.consolidate("chapter_017", extraction, candidates, registry)

    assert result.to_dict() == {
        "roles": ["mr_john_pounds_adult"],
        "quotes": [[1, 0]],
    }
```

- [ ] **Step 2: Run failing harness service test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_annotation_harness.py::test_harness_service_writes_valid_annotation_without_sonnet_for_unique_match -q
```

Expected: FAIL because `BookNlpSonnetConsolidationService` is missing.

- [ ] **Step 3: Implement deterministic service path**

Add:

```python
from ebook_tts_pipeline.annotation.quote_attribution import (
    QuoteAttributionResult,
    validate_quote_attribution,
)


class BookNlpSonnetConsolidationService:
    def __init__(self, client) -> None:
        self.client = client

    def consolidate(
        self,
        chapter: str,
        extraction,
        candidates: List[QuoteAttributionCandidate],
        registry: Dict,
    ) -> QuoteAttributionResult:
        deterministic = consolidate_candidates_deterministically(candidates, registry)
        if not deterministic.unresolved and len(deterministic.resolved_quotes) == len(extraction.quotes):
            roles = []
            quotes = []
            for quote in extraction.quotes:
                role_id = deterministic.resolved_quotes[quote.idx]
                if role_id not in roles:
                    roles.append(role_id)
                quotes.append((quote.idx, roles.index(role_id), "dialogue"))
            result = QuoteAttributionResult(roles=roles, quotes=quotes)
            validate_quote_attribution(
                result,
                quote_indices=[quote.idx for quote in extraction.quotes],
                known_role_ids={str(role_id) for role_id in registry.get("characters", {})},
            )
            return result
        payload = self.client.complete_json(
            "Return valid JSON only.",
            render_consolidation_prompt(chapter, candidates, registry),
        )
        result = QuoteAttributionResult.from_dict(payload)
        validate_quote_attribution(
            result,
            quote_indices=[quote.idx for quote in extraction.quotes],
            known_role_ids={str(role_id) for role_id in registry.get("characters", {})},
        )
        return result
```

- [ ] **Step 4: Verify harness service test passes**

Run the same pytest command. Expected: PASS.

---

## Task 8: Add Comparison Harness Script

**Files:**
- Create: `scripts/run_booknlp_annotation_harness.py`
- Test: `tests/test_booknlp_annotation_harness.py`

- [ ] **Step 1: Write failing report writer test**

Append:

```python
from scripts.run_booknlp_annotation_harness import build_harness_report


def test_harness_report_records_cost_reduction_metrics():
    report = build_harness_report(
        book_slug="victorian_psycho",
        chapters=["chapter_017"],
        deterministic_quotes=8,
        sonnet_quotes=2,
        failed_quotes=0,
        sonnet_prompt_chars=2400,
        old_full_prompt_chars=48000,
    )

    assert report["book_slug"] == "victorian_psycho"
    assert report["chapters"] == ["chapter_017"]
    assert report["deterministic_quotes"] == 8
    assert report["sonnet_quotes"] == 2
    assert report["estimated_prompt_char_savings"] == 45600
```

- [ ] **Step 2: Run failing report test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_annotation_harness.py::test_harness_report_records_cost_reduction_metrics -q
```

Expected: FAIL because the script is missing.

- [ ] **Step 3: Implement minimal report helper and CLI shell**

Create `scripts/run_booknlp_annotation_harness.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def build_harness_report(
    book_slug: str,
    chapters: List[str],
    deterministic_quotes: int,
    sonnet_quotes: int,
    failed_quotes: int,
    sonnet_prompt_chars: int,
    old_full_prompt_chars: int,
) -> Dict:
    return {
        "book_slug": book_slug,
        "chapters": list(chapters),
        "deterministic_quotes": deterministic_quotes,
        "sonnet_quotes": sonnet_quotes,
        "failed_quotes": failed_quotes,
        "sonnet_prompt_chars": sonnet_prompt_chars,
        "old_full_prompt_chars": old_full_prompt_chars,
        "estimated_prompt_char_savings": max(0, old_full_prompt_chars - sonnet_prompt_chars),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-root", required=True)
    parser.add_argument("--chapters", nargs="+", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = build_harness_report(
        book_slug=Path(args.book_root).name,
        chapters=args.chapters,
        deterministic_quotes=0,
        sonnet_quotes=0,
        failed_quotes=0,
        sonnet_prompt_chars=0,
        old_full_prompt_chars=0,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify report test passes**

Run the same pytest command. Expected: PASS.

---

## Task 9: Wire Pipeline Only Behind Harness Backend

**Files:**
- Modify: `src/ebook_tts_pipeline/config.py`
- Modify: `src/ebook_tts_pipeline/pipeline.py`
- Test: `tests/test_booknlp_annotation_harness.py`

- [ ] **Step 1: Write failing config test**

Append:

```python
from ebook_tts_pipeline.config import PipelineConfig


def test_booknlp_harness_config_is_opt_in(monkeypatch):
    monkeypatch.delenv("EBOOK_TTS_ANNOTATION_BACKEND", raising=False)
    default_config = PipelineConfig.from_env("book")
    assert default_config.annotation_backend == "sonnet"

    monkeypatch.setenv("EBOOK_TTS_ANNOTATION_BACKEND", "booknlp_harness")
    harness_config = PipelineConfig.from_env("book")
    assert harness_config.annotation_backend == "booknlp_harness"
```

- [ ] **Step 2: Run failing config test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_annotation_harness.py::test_booknlp_harness_config_is_opt_in -q
```

Expected: FAIL because `annotation_backend` is missing.

- [ ] **Step 3: Add config fields**

Modify `PipelineConfig`:

```python
annotation_backend: str = "sonnet"
booknlp_python: str = "python"
booknlp_model: str = "small"
booknlp_cache_policy: str = "reuse"
```

In `from_env()`:

```python
annotation_backend=os.environ.get("EBOOK_TTS_ANNOTATION_BACKEND", "sonnet"),
booknlp_python=os.environ.get("EBOOK_TTS_BOOKNLP_PYTHON", "python"),
booknlp_model=os.environ.get("EBOOK_TTS_BOOKNLP_MODEL", "small"),
booknlp_cache_policy=os.environ.get("EBOOK_TTS_BOOKNLP_CACHE_POLICY", "reuse"),
```

- [ ] **Step 4: Verify config test passes**

Run the same pytest command. Expected: PASS.

---

## Task 10: Full Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run new harness tests**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_booknlp_artifacts.py tests\test_booknlp_candidates.py tests\test_quote_consolidation.py tests\test_booknlp_annotation_harness.py -q
```

Expected: all new harness tests PASS.

- [ ] **Step 2: Run existing quote and web regression tests**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests\test_quote_attribution.py tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -q
```

Expected: all selected tests PASS.

- [ ] **Step 3: Compile source**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src
```

Expected: exit code 0.

---

## Manual Harness Run

After implementation, test Victorian Psycho chapter 17 without touching production annotations:

```powershell
$env:PYTHONPATH='src'
$env:EBOOK_TTS_ANNOTATION_BACKEND='booknlp_harness'
.\.venv\Scripts\python.exe scripts\run_booknlp_annotation_harness.py --book-root "books\victorian_psycho---virginia_feito" --chapters chapter_017 --output "books\victorian_psycho---virginia_feito\booknlp\chapter_017_harness_report.json"
```

Expected report fields:

```json
{
  "book_slug": "victorian_psycho---virginia_feito",
  "chapters": ["chapter_017"],
  "deterministic_quotes": 0,
  "sonnet_quotes": 0,
  "failed_quotes": 0,
  "sonnet_prompt_chars": 0,
  "old_full_prompt_chars": 0,
  "estimated_prompt_char_savings": 0
}
```

As tasks mature, replace zero counts with actual measured counts and add a comparison field for whether the final annotation passed `validate_quote_attribution()`.

---

## Self-Review

- Spec coverage: The plan creates a parallel BookNLP plus Sonnet consolidation pipeline, keeps current production annotation intact, saves BookNLP artifacts per book, maps local NLP candidates into the existing `quote_attribution_v1` schema, and adds a harness report for quality/cost testing.
- Placeholder scan: No `TBD`, `TODO`, or undefined task names remain. Each task has files, test code, implementation code, commands, and expected result.
- Type consistency: The planned types are consistently named `BookNlpQuoteRow`, `QuoteAttributionCandidate`, `DeterministicConsolidationResult`, and `BookNlpSonnetConsolidationService`.
- Scope check: This is intentionally not a default UI annotation backend yet. The first deliverable is a harnessed alternate path for measuring Victorian Psycho and False Witness before production replacement.
