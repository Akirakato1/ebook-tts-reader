# Ebook Audiobook Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-first audiobook pipeline that turns deterministic chapter sentence artifacts into validated speaker annotations, persistent Qwen voices, stitched chapter audio, and sentence-level timeline metadata.

**Architecture:** Create a reusable `ebook_tts_pipeline` package with clean domain models, repository classes, service classes, and adapter interfaces. The CLI remains a thin entry point over these services so a future reader UI can import the same classes without shelling out.

**Tech Stack:** Python 3.9+, standard-library `argparse`, `dataclasses`, `json`, `wave`, `pathlib`; `pytest` for tests; `nltk` for sentence segmentation; `anthropic` for LLM annotation; `numpy` for waveform handling; optional Qwen/torch imports isolated inside the Qwen adapter.

---

## Scope Check

This plan implements one testable vertical slice: ingestion, annotation, registry updates, voice preparation, TTS adapter boundary, audio stitching, timeline output, and CLI orchestration. It does not build the reader UI, EPUB parsing, a FastAPI service, or ComfyUI workflow execution.

The code must keep these UI-ready boundaries:

- Domain models are pure data and contain no CLI behavior.
- Repositories own file paths and persistence.
- Services own pipeline behavior and return structured results.
- Adapters isolate external systems: Anthropic and Qwen.
- CLI only parses arguments, constructs dependencies, and calls services.

## File Structure

- Create: `pyproject.toml`
  - Package metadata, dependencies, test settings, and CLI script.
- Create: `src/ebook_tts_pipeline/__init__.py`
  - Public package exports.
- Create: `src/ebook_tts_pipeline/config.py`
  - Runtime configuration dataclass and environment loading.
- Create: `src/ebook_tts_pipeline/domain.py`
  - Dataclasses for sentences, annotations, registry records, voice records, TTS jobs, generated audio, and timelines.
- Create: `src/ebook_tts_pipeline/json_io.py`
  - Atomic JSON read/write helpers.
- Create: `src/ebook_tts_pipeline/paths.py`
  - `BookPaths` and path layout helpers.
- Create: `src/ebook_tts_pipeline/ingestion.py`
  - Deterministic chapter splitting and sentence segmentation.
- Create: `src/ebook_tts_pipeline/windowing.py`
  - Atomic sentence window builders for LLM and TTS.
- Create: `src/ebook_tts_pipeline/annotation/__init__.py`
  - Annotation package exports.
- Create: `src/ebook_tts_pipeline/annotation/prompts.py`
  - Anthropic prompt rendering.
- Create: `src/ebook_tts_pipeline/annotation/validator.py`
  - Compact annotation JSON validation.
- Create: `src/ebook_tts_pipeline/annotation/anthropic_client.py`
  - Anthropic client adapter and fake-friendly protocol.
- Create: `src/ebook_tts_pipeline/annotation/service.py`
  - Annotation orchestration with repair retry.
- Create: `src/ebook_tts_pipeline/registry.py`
  - Registry loading, updates, alias collision detection, and atomic persistence.
- Create: `src/ebook_tts_pipeline/voice_identity.py`
  - Stable voice differentiators and seeds for similar characters.
- Create: `src/ebook_tts_pipeline/tts/__init__.py`
  - TTS package exports.
- Create: `src/ebook_tts_pipeline/tts/base.py`
  - TTS adapter protocol and shared dataclasses.
- Create: `src/ebook_tts_pipeline/tts/fake.py`
  - Deterministic fake adapter for tests.
- Create: `src/ebook_tts_pipeline/tts/qwen_adapter.py`
  - Optional direct wrapper around `qwen_tts`.
- Create: `src/ebook_tts_pipeline/audio.py`
  - WAV writing, waveform concatenation, pause insertion, and timeline construction.
- Create: `src/ebook_tts_pipeline/pipeline.py`
  - UI-ready `AudiobookPipeline` facade.
- Create: `src/ebook_tts_pipeline/cli.py`
  - Thin CLI entry point.
- Create: `config.example.toml`
  - Example local configuration.
- Create: `tests/fixtures/tiny_book/source/book.txt`
  - Small fixture book.
- Create: focused test files under `tests/`.
- Modify: `docs/runbooks/manual-tts-stack.txt`
  - Add final CLI commands after CLI exists.

---

### Task 1: Project Skeleton And Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/ebook_tts_pipeline/__init__.py`
- Create: `src/ebook_tts_pipeline/config.py`
- Test: `tests/test_public_import_and_config.py`

- [ ] **Step 1: Write the failing import/config test**

Create `tests/test_public_import_and_config.py`:

```python
import os

from ebook_tts_pipeline.config import PipelineConfig


def test_default_config_is_ui_friendly_and_overridable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("EBOOK_TTS_ANTHROPIC_MODEL", "claude-haiku-4-5")

    config = PipelineConfig.from_env(book_root="books/demo")

    assert config.book_root == "books/demo"
    assert config.anthropic_api_key == "test-key"
    assert config.anthropic_model == "claude-haiku-4-5"
    assert config.qwen_model_choice == "1.7B"
    assert config.max_tts_roles == 8
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_public_import_and_config.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'ebook_tts_pipeline'`.

- [ ] **Step 3: Add package metadata and config implementation**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ebook-tts-pipeline"
version = "0.1.0"
description = "Python-first ebook audiobook generation pipeline"
requires-python = ">=3.9"
dependencies = [
  "anthropic>=0.55",
  "nltk>=3.9",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
qwen = ["torch", "torchaudio", "transformers", "librosa", "accelerate"]

[project.scripts]
ebook-tts = "ebook_tts_pipeline.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `src/ebook_tts_pipeline/__init__.py`:

```python
"""Reusable services for deterministic ebook audiobook generation."""

from ebook_tts_pipeline.config import PipelineConfig

__all__ = ["PipelineConfig"]
```

Create `src/ebook_tts_pipeline/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class PipelineConfig:
    book_root: str
    anthropic_api_key: str | None = None
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_temperature: float = 0.1
    anthropic_max_tokens: int = 8192
    annotation_repair_retries: int = 1
    qwen_model_choice: str = "1.7B"
    qwen_device: str = "auto"
    qwen_precision: str = "bf16"
    qwen_attention: str = "auto"
    max_llm_window_chars: int = 48000
    max_tts_window_chars: int = 6000
    max_tts_roles: int = 8
    pause_between_sentences_ms: int = 250

    @classmethod
    def from_env(cls, book_root: str) -> "PipelineConfig":
        return cls(
            book_root=book_root,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            anthropic_model=os.environ.get(
                "EBOOK_TTS_ANTHROPIC_MODEL",
                DEFAULT_ANTHROPIC_MODEL,
            ),
            qwen_model_choice=os.environ.get("EBOOK_TTS_QWEN_MODEL", "1.7B"),
            qwen_device=os.environ.get("EBOOK_TTS_QWEN_DEVICE", "auto"),
            qwen_precision=os.environ.get("EBOOK_TTS_QWEN_PRECISION", "bf16"),
            qwen_attention=os.environ.get("EBOOK_TTS_QWEN_ATTENTION", "auto"),
        )

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for annotation.")
        return self.anthropic_api_key
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_public_import_and_config.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ebook_tts_pipeline tests/test_public_import_and_config.py
git commit -m "Add Python pipeline package skeleton"
```

---

### Task 2: Domain Models, Paths, And Atomic JSON

**Files:**
- Create: `src/ebook_tts_pipeline/domain.py`
- Create: `src/ebook_tts_pipeline/json_io.py`
- Create: `src/ebook_tts_pipeline/paths.py`
- Test: `tests/test_domain_paths_json.py`

- [ ] **Step 1: Write failing tests for models, paths, and atomic JSON**

Create `tests/test_domain_paths_json.py`:

```python
import json

from ebook_tts_pipeline.domain import Sentence, SentenceArtifact
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths


def test_book_paths_match_spec_layout(tmp_path):
    paths = BookPaths(tmp_path / "books" / "demo")

    assert paths.registry.name == "registry.json"
    assert paths.source_book.as_posix().endswith("source/book.txt")
    assert paths.chapter_text("chapter_001").as_posix().endswith("chapters/chapter_001.txt")
    assert paths.sentence_artifact("chapter_001").as_posix().endswith(
        "sentence_segments/chapter_001.sentences.json"
    )
    assert paths.annotation("chapter_001").as_posix().endswith(
        "annotations/chapter_001.annotation.json"
    )
    assert paths.chapter_audio("chapter_001").as_posix().endswith("audio/chapter_001.wav")
    assert paths.chapter_timeline("chapter_001").as_posix().endswith(
        "audio/chapter_001.timeline.json"
    )
    assert paths.voice_qvp("elena").as_posix().endswith("voices/elena.qvp")


def test_sentence_artifact_serializes_with_stable_sentence_indexes():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test", "language": "english", "version": "1"},
        sentences=[Sentence(idx=0, text="Hello."), Sentence(idx=1, text="Goodbye.")],
    )

    data = artifact.to_dict()
    restored = SentenceArtifact.from_dict(data)

    assert [s.idx for s in restored.sentences] == [0, 1]
    assert restored.sentences[1].text == "Goodbye."


def test_write_json_atomic_creates_parent_and_valid_json(tmp_path):
    path = tmp_path / "nested" / "data.json"

    write_json_atomic(path, {"ok": True})

    assert read_json(path) == {"ok": True}
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_domain_paths_json.py -v
```

Expected: fail because `domain.py`, `json_io.py`, and `paths.py` do not exist.

- [ ] **Step 3: Implement domain serialization, paths, and atomic writes**

Create `src/ebook_tts_pipeline/domain.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Sentence:
    idx: int
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sentence":
        return cls(idx=int(data["idx"]), text=str(data["text"]))


@dataclass(frozen=True)
class SentenceArtifact:
    chapter: str
    source_path: str
    segmenter: dict[str, str]
    sentences: list[Sentence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter": self.chapter,
            "source_path": self.source_path,
            "segmenter": self.segmenter,
            "sentences": [asdict(sentence) for sentence in self.sentences],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceArtifact":
        return cls(
            chapter=str(data["chapter"]),
            source_path=str(data["source_path"]),
            segmenter=dict(data["segmenter"]),
            sentences=[Sentence.from_dict(item) for item in data["sentences"]],
        )


@dataclass(frozen=True)
class AnnotationResult:
    new_characters: list[dict[str, Any]]
    roles: list[str]
    types: list[str]
    script: list[tuple[int, int, int]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationResult":
        return cls(
            new_characters=list(data.get("new_characters", [])),
            roles=[str(role) for role in data["roles"]],
            types=[str(item) for item in data["types"]],
            script=[tuple(int(value) for value in row) for row in data["script"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_characters": self.new_characters,
            "roles": self.roles,
            "types": self.types,
            "script": [list(row) for row in self.script],
        }
```

Create `src/ebook_tts_pipeline/json_io.py`:

```python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()
```

Create `src/ebook_tts_pipeline/paths.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BookPaths:
    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root))

    @property
    def source_book(self) -> Path:
        return self.root / "source" / "book.txt"

    @property
    def registry(self) -> Path:
        return self.root / "registry.json"

    def chapter_text(self, chapter: str) -> Path:
        return self.root / "chapters" / f"{chapter}.txt"

    def sentence_artifact(self, chapter: str) -> Path:
        return self.root / "sentence_segments" / f"{chapter}.sentences.json"

    def annotation(self, chapter: str) -> Path:
        return self.root / "annotations" / f"{chapter}.annotation.json"

    def chapter_audio(self, chapter: str) -> Path:
        return self.root / "audio" / f"{chapter}.wav"

    def chapter_timeline(self, chapter: str) -> Path:
        return self.root / "audio" / f"{chapter}.timeline.json"

    def voice_qvp(self, role_id: str) -> Path:
        return self.root / "voices" / f"{role_id}.qvp"

    def voice_metadata(self, role_id: str) -> Path:
        return self.root / "voices" / f"{role_id}.json"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_domain_paths_json.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ebook_tts_pipeline/domain.py src/ebook_tts_pipeline/json_io.py src/ebook_tts_pipeline/paths.py tests/test_domain_paths_json.py
git commit -m "Add pipeline domain models and persistence helpers"
```

---

### Task 3: Deterministic Ingestion And Sentence Segmentation

**Files:**
- Create: `src/ebook_tts_pipeline/ingestion.py`
- Test: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing ingestion tests**

Create `tests/test_ingestion.py`:

```python
from ebook_tts_pipeline.domain import SentenceArtifact
from ebook_tts_pipeline.ingestion import ChapterSplitter, SentenceSegmenter
from ebook_tts_pipeline.json_io import read_json
from ebook_tts_pipeline.paths import BookPaths


def test_chapter_splitter_writes_confident_chapters(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.source_book.parent.mkdir(parents=True)
    paths.source_book.write_text(
        "Chapter 1\nThe first room was silent.\n\n"
        "Chapter 2\nThe second room was loud.\n",
        encoding="utf-8",
    )

    result = ChapterSplitter().split_source_book(paths)

    assert result.chapters == ["chapter_001", "chapter_002"]
    assert paths.chapter_text("chapter_001").read_text(encoding="utf-8").startswith("The first")
    assert paths.chapter_text("chapter_002").read_text(encoding="utf-8").startswith("The second")


def test_chapter_splitter_rejects_low_confidence_source(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.source_book.parent.mkdir(parents=True)
    paths.source_book.write_text("A book with no clear headings.", encoding="utf-8")

    result = ChapterSplitter().split_source_book(paths)

    assert result.chapters == []
    assert result.reason == "low_confidence_chapter_split"


def test_sentence_segmenter_writes_canonical_artifact(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Hello there. She waved.", encoding="utf-8")
    segmenter = SentenceSegmenter(tokenizer=lambda text: ["Hello there.", "She waved."])

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert isinstance(artifact, SentenceArtifact)
    data = read_json(paths.sentence_artifact("chapter_001"))
    assert data["sentences"] == [
        {"idx": 0, "text": "Hello there."},
        {"idx": 1, "text": "She waved."},
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ingestion.py -v
```

Expected: fail because `ingestion.py` does not exist.

- [ ] **Step 3: Implement chapter splitting and segmentation services**

Create `src/ebook_tts_pipeline/ingestion.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ebook_tts_pipeline.domain import Sentence, SentenceArtifact
from ebook_tts_pipeline.json_io import write_json_atomic
from ebook_tts_pipeline.paths import BookPaths


CHAPTER_HEADING_RE = re.compile(
    r"(?im)^\s*(chapter\s+([0-9]+|[ivxlcdm]+|[a-z]+)|prologue|epilogue|part\s+[ivxlcdm]+)\s*$"
)


@dataclass(frozen=True)
class ChapterSplitResult:
    chapters: list[str]
    reason: str | None = None


class ChapterSplitter:
    def split_source_book(self, paths: BookPaths) -> ChapterSplitResult:
        text = paths.source_book.read_text(encoding="utf-8")
        matches = list(CHAPTER_HEADING_RE.finditer(text))
        if len(matches) < 2:
            return ChapterSplitResult(chapters=[], reason="low_confidence_chapter_split")

        chapters: list[str] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if len(body) < 20:
                return ChapterSplitResult(chapters=[], reason="low_confidence_chapter_split")
            chapter_id = f"chapter_{index + 1:03d}"
            output_path = paths.chapter_text(chapter_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(body + "\n", encoding="utf-8")
            chapters.append(chapter_id)

        return ChapterSplitResult(chapters=chapters)


class SentenceSegmenter:
    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self._tokenizer = tokenizer

    def segment_chapter(self, paths: BookPaths, chapter: str) -> SentenceArtifact:
        text = paths.chapter_text(chapter).read_text(encoding="utf-8")
        raw_sentences = self._tokenize(text)
        sentences = [
            Sentence(idx=index, text=sentence.strip())
            for index, sentence in enumerate(raw_sentences)
            if sentence.strip()
        ]
        artifact = SentenceArtifact(
            chapter=chapter,
            source_path=f"chapters/{chapter}.txt",
            segmenter={
                "name": "nltk.sent_tokenize" if self._tokenizer is None else "custom",
                "language": "english",
                "version": self._segmenter_version(),
            },
            sentences=sentences,
        )
        write_json_atomic(paths.sentence_artifact(chapter), artifact.to_dict())
        return artifact

    def _tokenize(self, text: str) -> list[str]:
        if self._tokenizer is not None:
            return self._tokenizer(text)
        import nltk

        return nltk.sent_tokenize(text)

    def _segmenter_version(self) -> str:
        if self._tokenizer is not None:
            return "test"
        try:
            import nltk

            return nltk.__version__
        except Exception:
            return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ingestion.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ebook_tts_pipeline/ingestion.py tests/test_ingestion.py
git commit -m "Add deterministic book ingestion"
```

---

### Task 4: Windowing And Annotation Validation

**Files:**
- Create: `src/ebook_tts_pipeline/windowing.py`
- Create: `src/ebook_tts_pipeline/annotation/__init__.py`
- Create: `src/ebook_tts_pipeline/annotation/validator.py`
- Test: `tests/test_windowing_and_annotation_validation.py`

- [ ] **Step 1: Write failing windowing and validation tests**

Create `tests/test_windowing_and_annotation_validation.py`:

```python
import pytest

from ebook_tts_pipeline.annotation.validator import AnnotationValidationError, validate_annotation
from ebook_tts_pipeline.domain import AnnotationResult, Sentence
from ebook_tts_pipeline.windowing import build_llm_windows, build_tts_windows


def test_llm_windowing_moves_sentence_to_next_window_when_it_would_exceed_limit():
    sentences = [
        Sentence(idx=0, text="aaaa"),
        Sentence(idx=1, text="bbbb"),
        Sentence(idx=2, text="cccc"),
    ]

    windows = build_llm_windows(sentences, max_chars=9)

    assert [[sentence.idx for sentence in window.sentences] for window in windows] == [[0, 1], [2]]


def test_tts_windowing_respects_eight_role_limit_and_sentence_atomicity():
    jobs = [
        {"sentence_idx": idx, "role": f"Role{idx}", "text": "Hi."}
        for idx in range(9)
    ]

    windows = build_tts_windows(jobs, max_chars=1000, max_roles=8)

    assert [len({job["role"] for job in window.jobs}) for window in windows] == [8, 1]
    assert [window.jobs[0]["sentence_idx"] for window in windows] == [0, 8]


def test_annotation_validator_accepts_complete_compact_script():
    result = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Elena"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1)],
    )

    validate_annotation(result, expected_sentence_indices=[0, 1], known_names={"Elena"})


def test_annotation_validator_rejects_duplicate_sentence_ids():
    result = AnnotationResult(
        new_characters=[],
        roles=["Narrator"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (0, 0, 0)],
    )

    with pytest.raises(AnnotationValidationError) as exc:
        validate_annotation(result, expected_sentence_indices=[0, 1], known_names=set())

    assert "missing sentence indexes: [1]" in str(exc.value)
    assert "duplicate sentence indexes: [0]" in str(exc.value)


def test_annotation_validator_rejects_new_character_alias_collision():
    result = AnnotationResult(
        new_characters=[{"name": "Elena", "profile": {}, "voice": {}}],
        roles=["Narrator", "Elena"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1)],
    )

    with pytest.raises(AnnotationValidationError) as exc:
        validate_annotation(result, expected_sentence_indices=[0, 1], known_names={"elena"})

    assert "collides with existing character or alias: Elena" in str(exc.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_windowing_and_annotation_validation.py -v
```

Expected: fail because `windowing.py` and annotation validator do not exist.

- [ ] **Step 3: Implement atomic windowing and annotation validation**

Create `src/ebook_tts_pipeline/windowing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ebook_tts_pipeline.domain import Sentence


@dataclass(frozen=True)
class LlmWindow:
    sentences: list[Sentence]


@dataclass(frozen=True)
class TtsWindow:
    jobs: list[dict[str, Any]]


def build_llm_windows(sentences: list[Sentence], max_chars: int) -> list[LlmWindow]:
    windows: list[LlmWindow] = []
    current: list[Sentence] = []
    current_chars = 0
    for sentence in sentences:
        sentence_size = len(sentence.text)
        if sentence_size > max_chars:
            raise ValueError(f"sentence {sentence.idx} exceeds max LLM window size")
        if current and current_chars + sentence_size > max_chars:
            windows.append(LlmWindow(sentences=current))
            current = []
            current_chars = 0
        current.append(sentence)
        current_chars += sentence_size
    if current:
        windows.append(LlmWindow(sentences=current))
    return windows


def build_tts_windows(
    jobs: list[dict[str, Any]],
    max_chars: int,
    max_roles: int,
) -> list[TtsWindow]:
    windows: list[TtsWindow] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    current_roles: set[str] = set()
    for job in jobs:
        text_size = len(str(job["text"]))
        role = str(job["role"])
        if text_size > max_chars:
            raise ValueError(f"sentence {job['sentence_idx']} exceeds max TTS window size")
        next_roles = current_roles | {role}
        would_exceed_chars = current and current_chars + text_size > max_chars
        would_exceed_roles = current and len(next_roles) > max_roles
        if would_exceed_chars or would_exceed_roles:
            windows.append(TtsWindow(jobs=current))
            current = []
            current_chars = 0
            current_roles = set()
        current.append(job)
        current_chars += text_size
        current_roles.add(role)
    if current:
        windows.append(TtsWindow(jobs=current))
    return windows
```

Create `src/ebook_tts_pipeline/annotation/__init__.py`:

```python
"""Annotation prompting, validation, and Anthropic adapter code."""
```

Create `src/ebook_tts_pipeline/annotation/validator.py`:

```python
from __future__ import annotations

from collections import Counter

from ebook_tts_pipeline.domain import AnnotationResult


class AnnotationValidationError(ValueError):
    pass


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def validate_annotation(
    result: AnnotationResult,
    expected_sentence_indices: list[int],
    known_names: set[str],
) -> None:
    errors: list[str] = []
    expected = set(expected_sentence_indices)
    actual = [row[2] for row in result.script]
    actual_set = set(actual)

    missing = sorted(expected - actual_set)
    extra = sorted(actual_set - expected)
    duplicates = sorted(index for index, count in Counter(actual).items() if count > 1)

    if missing:
        errors.append(f"missing sentence indexes: {missing}")
    if extra:
        errors.append(f"unknown sentence indexes: {extra}")
    if duplicates:
        errors.append(f"duplicate sentence indexes: {duplicates}")

    for row in result.script:
        if len(row) != 3:
            errors.append(f"script row must have 3 items: {row}")
            continue
        role_idx, type_idx, _ = row
        if role_idx < 0 or role_idx >= len(result.roles):
            errors.append(f"role index out of range: {role_idx}")
        if type_idx < 0 or type_idx >= len(result.types):
            errors.append(f"type index out of range: {type_idx}")

    if "narration" in result.types:
        narration_idx = result.types.index("narration")
        if any(row[1] == narration_idx for row in result.script) and "Narrator" not in result.roles:
            errors.append("roles must include Narrator when narration appears")

    normalized_known = {_normalize_name(name) for name in known_names}
    for character in result.new_characters:
        name = str(character.get("name", "")).strip()
        if not name:
            errors.append("new character is missing name")
        elif _normalize_name(name) in normalized_known:
            errors.append(f"collides with existing character or alias: {name}")

    if errors:
        raise AnnotationValidationError("; ".join(errors))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_windowing_and_annotation_validation.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ebook_tts_pipeline/windowing.py src/ebook_tts_pipeline/annotation tests/test_windowing_and_annotation_validation.py
git commit -m "Add annotation validation and windowing"
```

---

### Task 5: Registry Management And Voice Identity Differentiation

**Files:**
- Create: `src/ebook_tts_pipeline/registry.py`
- Create: `src/ebook_tts_pipeline/voice_identity.py`
- Test: `tests/test_registry_and_voice_identity.py`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_registry_and_voice_identity.py`:

```python
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager


def test_registry_adds_new_character_with_stable_voice_identity(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")

    manager.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Elena",
                "profile": {"age_range": "young adult", "gender": "female"},
                "voice": {
                    "description": "young woman, soft",
                    "qwen_instruct": "A soft young adult female voice.",
                },
            }
        ],
    )

    registry = read_json(paths.registry)
    elena = registry["characters"]["elena"]
    assert elena["display_name"] == "Elena"
    assert elena["first_seen"] == "chapter_001"
    assert elena["voice_config_path"] is None
    assert isinstance(elena["voice_identity"]["seed"], int)
    assert elena["voice_identity"]["differentiators"]


def test_similar_character_receives_different_voice_differentiator(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")
    repeated = {
        "profile": {"age_range": "young adult", "gender": "female"},
        "voice": {
            "description": "young woman, soft",
            "qwen_instruct": "A soft young adult female voice.",
        },
    }

    manager.add_new_characters(chapter="chapter_001", new_characters=[{"name": "Elena", **repeated}])
    manager.add_new_characters(chapter="chapter_002", new_characters=[{"name": "Mira", **repeated}])

    registry = read_json(paths.registry)
    elena_voice = registry["characters"]["elena"]["voice_profile"]["qwen_instruct"]
    mira_voice = registry["characters"]["mira"]["voice_profile"]["qwen_instruct"]
    assert elena_voice != mira_voice


def test_registry_rejects_alias_collision(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"display_name": "Narrator"},
            "characters": {
                "elena": {
                    "display_name": "Elena",
                    "aliases": ["Lena"],
                    "voice_profile": {"description": "soft", "qwen_instruct": "soft"},
                }
            },
        },
    )
    manager = RegistryManager(paths)

    try:
        manager.add_new_characters(
            chapter="chapter_002",
            new_characters=[{"name": "Lena", "profile": {}, "voice": {"description": "x", "qwen_instruct": "x"}}],
        )
    except ValueError as exc:
        assert "collides with existing character or alias" in str(exc)
    else:
        raise AssertionError("Expected alias collision")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_registry_and_voice_identity.py -v
```

Expected: fail because registry and voice identity modules do not exist.

- [ ] **Step 3: Implement registry manager and voice identity assignment**

Create `src/ebook_tts_pipeline/voice_identity.py`:

```python
from __future__ import annotations

import hashlib


DIFFERENTIATORS = [
    "brighter timbre",
    "darker timbre",
    "slightly quicker cadence",
    "slower deliberate cadence",
    "lighter resonance",
    "deeper chest resonance",
    "more breathiness",
    "cleaner crisp articulation",
]


def role_seed(book_slug: str, role_id: str) -> int:
    digest = hashlib.sha256(f"{book_slug}:{role_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def choose_differentiators(book_slug: str, role_id: str, count: int = 3) -> list[str]:
    seed = role_seed(book_slug, role_id)
    start = seed % len(DIFFERENTIATORS)
    return [DIFFERENTIATORS[(start + offset) % len(DIFFERENTIATORS)] for offset in range(count)]


def append_differentiators(qwen_instruct: str, differentiators: list[str]) -> str:
    suffix = ", ".join(differentiators)
    stripped = qwen_instruct.rstrip(". ")
    return f"{stripped}, with {suffix}."
```

Create `src/ebook_tts_pipeline/registry.py`:

```python
from __future__ import annotations

import re
from typing import Any

from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.voice_identity import append_differentiators, choose_differentiators, role_seed


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError(f"Cannot create role_id from empty name: {name!r}")
    return slug


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


class RegistryManager:
    def __init__(self, paths: BookPaths) -> None:
        self.paths = paths

    def initialize_if_missing(self, book_title: str, book_slug: str) -> None:
        if self.paths.registry.exists():
            return
        registry = {
            "book": {"title": book_title, "slug": book_slug},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "character_profile": {"role": "narrator"},
                "voice_identity": {
                    "seed": role_seed(book_slug, "narrator"),
                    "differentiators": ["calm baseline narrator timbre"],
                },
                "voice_profile": {
                    "description": "calm literary narrator, clear pacing",
                    "qwen_instruct": "A calm literary narrator voice with clear pacing.",
                },
                "voice_config_path": None,
            },
            "characters": {},
        }
        write_json_atomic(self.paths.registry, registry)

    def load(self) -> dict[str, Any]:
        return read_json(self.paths.registry)

    def save(self, registry: dict[str, Any]) -> None:
        write_json_atomic(self.paths.registry, registry)

    def known_names(self) -> set[str]:
        registry = self.load()
        names = {"Narrator"}
        for character in registry.get("characters", {}).values():
            names.add(str(character.get("display_name", "")))
            names.update(str(alias) for alias in character.get("aliases", []))
        return names

    def add_new_characters(self, chapter: str, new_characters: list[dict[str, Any]]) -> None:
        registry = self.load()
        normalized_known = {normalize_name(name) for name in self.known_names()}
        book_slug = str(registry["book"]["slug"])

        for character in new_characters:
            name = str(character["name"]).strip()
            normalized = normalize_name(name)
            if normalized in normalized_known:
                raise ValueError(f"collides with existing character or alias: {name}")
            role_id = slugify_name(name)
            differentiators = choose_differentiators(book_slug, role_id)
            voice = dict(character["voice"])
            voice["qwen_instruct"] = append_differentiators(
                str(voice["qwen_instruct"]),
                differentiators,
            )
            registry["characters"][role_id] = {
                "role_id": role_id,
                "display_name": name,
                "aliases": [],
                "character_profile": character.get("profile", {}),
                "voice_identity": {
                    "seed": role_seed(book_slug, role_id),
                    "differentiators": differentiators,
                },
                "voice_profile": voice,
                "voice_config_path": None,
                "first_seen": chapter,
            }
            normalized_known.add(normalized)

        self.save(registry)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_registry_and_voice_identity.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ebook_tts_pipeline/registry.py src/ebook_tts_pipeline/voice_identity.py tests/test_registry_and_voice_identity.py
git commit -m "Add registry and voice identity management"
```

---

### Task 6: Anthropic Annotation Service With Repair Retry

**Files:**
- Create: `src/ebook_tts_pipeline/annotation/prompts.py`
- Create: `src/ebook_tts_pipeline/annotation/anthropic_client.py`
- Create: `src/ebook_tts_pipeline/annotation/service.py`
- Test: `tests/test_annotation_service.py`

- [ ] **Step 1: Write failing annotation service tests**

Create `tests/test_annotation_service.py`:

```python
import json

from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.domain import Sentence


class FakeLlmClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        payload = self.payloads.pop(0)
        if isinstance(payload, str):
            return json.loads(payload)
        return payload


def test_annotation_service_returns_valid_result_without_repair():
    client = FakeLlmClient(
        [
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0], [0, 0, 1]],
            }
        ]
    )
    service = AnnotationService(client=client, repair_retries=1)

    result = service.annotate_window(
        chapter="chapter_001",
        sentences=[Sentence(0, "It rained."), Sentence(1, "The road shone.")],
        registry={"characters": {}},
    )

    assert result.roles == ["Narrator"]
    assert result.script == [(0, 0, 0), (0, 0, 1)]
    assert len(client.calls) == 1


def test_annotation_service_repairs_invalid_result_once():
    client = FakeLlmClient(
        [
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0]],
            },
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0], [0, 0, 1]],
            },
        ]
    )
    service = AnnotationService(client=client, repair_retries=1)

    result = service.annotate_window(
        chapter="chapter_001",
        sentences=[Sentence(0, "It rained."), Sentence(1, "The road shone.")],
        registry={"characters": {}},
    )

    assert result.script == [(0, 0, 0), (0, 0, 1)]
    assert "missing sentence indexes: [1]" in client.calls[1]["user"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_annotation_service.py -v
```

Expected: fail because annotation prompt/client/service modules do not exist.

- [ ] **Step 3: Implement prompt rendering, client protocol, and repair flow**

Create `src/ebook_tts_pipeline/annotation/prompts.py`:

```python
from __future__ import annotations

import json

from ebook_tts_pipeline.domain import Sentence


SYSTEM_PROMPT = (
    "You label ebook sentences for audiobook generation. "
    "Return only valid JSON matching the requested compact schema."
)


def render_annotation_prompt(chapter: str, sentences: list[Sentence], registry: dict) -> str:
    rendered_sentences = "\n".join(f"[{sentence.idx}] {sentence.text}" for sentence in sentences)
    known_characters = registry.get("characters", {})
    return (
        f"Known characters: {json.dumps(known_characters, ensure_ascii=False)}\n\n"
        f"Chapter: {chapter}\n\n"
        f"Chapter text:\n{rendered_sentences}\n\n"
        "Return JSON with these keys:\n"
        "- new_characters: list of {name, profile, voice}\n"
        "- roles: list of role names appearing in this window\n"
        "- types: exactly [\"narration\", \"dialogue\", \"thought\"]\n"
        "- script: list of [role_idx, type_idx, sentence_idx]\n"
        "Every sentence index in the input must appear exactly once."
    )


def render_repair_prompt(original_prompt: str, invalid_output: dict, errors: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous JSON failed validation.\n"
        f"Validation errors: {errors}\n"
        f"Invalid JSON: {json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Return corrected JSON only."
    )
```

Create `src/ebook_tts_pipeline/annotation/anthropic_client.py`:

```python
from __future__ import annotations

import json
from typing import Protocol


class JsonCompletionClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        ...


class AnthropicJsonClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        return json.loads(text)
```

Create `src/ebook_tts_pipeline/annotation/service.py`:

```python
from __future__ import annotations

from ebook_tts_pipeline.annotation.anthropic_client import JsonCompletionClient
from ebook_tts_pipeline.annotation.prompts import SYSTEM_PROMPT, render_annotation_prompt, render_repair_prompt
from ebook_tts_pipeline.annotation.validator import AnnotationValidationError, validate_annotation
from ebook_tts_pipeline.domain import AnnotationResult, Sentence


class AnnotationService:
    def __init__(self, client: JsonCompletionClient, repair_retries: int) -> None:
        self.client = client
        self.repair_retries = repair_retries

    def annotate_window(
        self,
        chapter: str,
        sentences: list[Sentence],
        registry: dict,
    ) -> AnnotationResult:
        prompt = render_annotation_prompt(chapter, sentences, registry)
        payload = self.client.complete_json(SYSTEM_PROMPT, prompt)
        result = AnnotationResult.from_dict(payload)
        expected = [sentence.idx for sentence in sentences]
        known_names = _known_names(registry)

        for attempt in range(self.repair_retries + 1):
            try:
                validate_annotation(result, expected_sentence_indices=expected, known_names=known_names)
                return result
            except AnnotationValidationError as exc:
                if attempt >= self.repair_retries:
                    raise
                repair_prompt = render_repair_prompt(prompt, result.to_dict(), str(exc))
                payload = self.client.complete_json(SYSTEM_PROMPT, repair_prompt)
                result = AnnotationResult.from_dict(payload)

        return result


def _known_names(registry: dict) -> set[str]:
    names = {"Narrator"}
    for character in registry.get("characters", {}).values():
        names.add(str(character.get("display_name", "")))
        names.update(str(alias) for alias in character.get("aliases", []))
    return names
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_annotation_service.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ebook_tts_pipeline/annotation/prompts.py src/ebook_tts_pipeline/annotation/anthropic_client.py src/ebook_tts_pipeline/annotation/service.py tests/test_annotation_service.py
git commit -m "Add Anthropic annotation service"
```

---

### Task 7: TTS Adapter Contract, Fake Adapter, Audio Stitching, And Timeline

**Files:**
- Create: `src/ebook_tts_pipeline/tts/__init__.py`
- Create: `src/ebook_tts_pipeline/tts/base.py`
- Create: `src/ebook_tts_pipeline/tts/fake.py`
- Create: `src/ebook_tts_pipeline/audio.py`
- Test: `tests/test_audio_and_tts.py`

- [ ] **Step 1: Write failing tests for fake TTS and timeline math**

Create `tests/test_audio_and_tts.py`:

```python
import wave

from ebook_tts_pipeline.audio import ChapterAudioBuilder
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter


def test_fake_tts_and_audio_builder_write_sentence_timeline(tmp_path):
    adapter = FakeTtsAdapter(sample_rate=1000, samples_per_character=10)
    builder = ChapterAudioBuilder(tts_adapter=adapter, pause_between_sentences_ms=100)
    jobs = [
        {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "Hello."},
        {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Hi."},
    ]

    result = builder.build_chapter_audio(
        chapter="chapter_001",
        jobs=jobs,
        audio_path=tmp_path / "chapter_001.wav",
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    assert result["sentences"][0]["start_ms"] == 0
    assert result["sentences"][0]["end_ms"] == 60
    assert result["sentences"][1]["start_ms"] == 160
    assert result["sentences"][1]["end_ms"] == 190

    with wave.open(str(tmp_path / "chapter_001.wav"), "rb") as wav:
        assert wav.getframerate() == 1000
        assert wav.getnchannels() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_audio_and_tts.py -v
```

Expected: fail because TTS and audio modules do not exist.

- [ ] **Step 3: Implement TTS protocol, fake adapter, and WAV/timeline builder**

Create `src/ebook_tts_pipeline/tts/__init__.py`:

```python
"""TTS adapter interfaces and implementations."""
```

Create `src/ebook_tts_pipeline/tts/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class GeneratedSentenceAudio:
    sentence_idx: int
    role: str
    speech_type: str
    samples: np.ndarray
    sample_rate: int


class TtsAdapter(Protocol):
    def ensure_voice(self, role_id: str, voice_record: dict, voice_path: Path) -> Path:
        ...

    def generate_sentences(self, jobs: list[dict]) -> list[GeneratedSentenceAudio]:
        ...
```

Create `src/ebook_tts_pipeline/tts/fake.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


class FakeTtsAdapter:
    def __init__(self, sample_rate: int = 24000, samples_per_character: int = 100) -> None:
        self.sample_rate = sample_rate
        self.samples_per_character = samples_per_character

    def ensure_voice(self, role_id: str, voice_record: dict, voice_path: Path) -> Path:
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        if not voice_path.exists():
            voice_path.write_bytes(f"fake voice for {role_id}".encode("utf-8"))
        return voice_path

    def generate_sentences(self, jobs: list[dict]) -> list[GeneratedSentenceAudio]:
        generated: list[GeneratedSentenceAudio] = []
        for job in jobs:
            length = len(str(job["text"])) * self.samples_per_character
            samples = np.full(length, 0.05, dtype=np.float32)
            generated.append(
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    role=str(job["role"]),
                    speech_type=str(job["type"]),
                    samples=samples,
                    sample_rate=self.sample_rate,
                )
            )
        return generated
```

Create `src/ebook_tts_pipeline/audio.py`:

```python
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ebook_tts_pipeline.json_io import write_json_atomic
from ebook_tts_pipeline.tts.base import TtsAdapter


class ChapterAudioBuilder:
    def __init__(self, tts_adapter: TtsAdapter, pause_between_sentences_ms: int) -> None:
        self.tts_adapter = tts_adapter
        self.pause_between_sentences_ms = pause_between_sentences_ms

    def build_chapter_audio(
        self,
        chapter: str,
        jobs: list[dict],
        audio_path: str | Path,
        timeline_path: str | Path,
    ) -> dict:
        generated = self.tts_adapter.generate_sentences(jobs)
        if not generated:
            raise ValueError("Cannot build audio without sentence jobs.")

        sample_rate = generated[0].sample_rate
        cursor_samples = 0
        timeline_sentences: list[dict] = []
        chunks: list[np.ndarray] = []
        pause_samples = int(sample_rate * self.pause_between_sentences_ms / 1000)

        for index, item in enumerate(generated):
            if item.sample_rate != sample_rate:
                raise ValueError("All generated sentence audio must use the same sample rate.")
            start_samples = cursor_samples
            end_samples = start_samples + len(item.samples)
            timeline_sentences.append(
                {
                    "sentence_idx": item.sentence_idx,
                    "role": item.role,
                    "type": item.speech_type,
                    "start_ms": int(round(start_samples * 1000 / sample_rate)),
                    "end_ms": int(round(end_samples * 1000 / sample_rate)),
                }
            )
            chunks.append(item.samples.astype(np.float32))
            cursor_samples = end_samples
            if index + 1 < len(generated) and pause_samples:
                chunks.append(np.zeros(pause_samples, dtype=np.float32))
                cursor_samples += pause_samples

        merged = np.concatenate(chunks)
        self._write_wav(Path(audio_path), merged, sample_rate)
        timeline = {
            "chapter": chapter,
            "audio_path": str(audio_path),
            "sample_rate": sample_rate,
            "sentences": timeline_sentences,
        }
        write_json_atomic(timeline_path, timeline)
        return timeline

    def _write_wav(self, path: Path, samples: np.ndarray, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm16.tobytes())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_audio_and_tts.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ebook_tts_pipeline/tts src/ebook_tts_pipeline/audio.py tests/test_audio_and_tts.py
git commit -m "Add TTS adapter contract and audio timeline builder"
```

---

### Task 8: Qwen TTS Adapter Boundary

**Files:**
- Create: `src/ebook_tts_pipeline/tts/qwen_adapter.py`
- Test: `tests/test_qwen_adapter.py`

- [ ] **Step 1: Write failing tests with a fake Qwen runtime**

Create `tests/test_qwen_adapter.py`:

```python
import numpy as np

from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter


class FakeQwenModel:
    def __init__(self):
        self.voice_design_calls = []
        self.voice_clone_calls = []

    def generate_voice_design(self, text, instruct, language, **kwargs):
        self.voice_design_calls.append({"text": text, "instruct": instruct, "language": language})
        return [np.ones(100, dtype=np.float32) * 0.1], 24000

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        return [{"ref_code": None, "ref_spk_embedding": "embedding", "x_vector_only_mode": True, "icl_mode": False}]

    def generate_voice_clone(self, text, language, voice_clone_prompt, **kwargs):
        self.voice_clone_calls.append({"text": text, "language": language, "prompt": voice_clone_prompt})
        return [np.ones(50, dtype=np.float32), np.ones(25, dtype=np.float32)], 24000


class FakeTorchStore:
    def __init__(self):
        self.saved = {}

    def save(self, value, path):
        self.saved[str(path)] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"qvp")

    def load(self, path, map_location="cpu", weights_only=False):
        return self.saved[str(path)]


def test_qwen_adapter_creates_qvp_once_and_reuses_existing_file(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    adapter = QwenTtsAdapter(model=model, torch_module=torch_store)
    voice_path = tmp_path / "voices" / "elena.qvp"
    voice_record = {
        "voice_profile": {"qwen_instruct": "A soft voice."},
        "voice_identity": {"seed": 42},
    }

    first = adapter.ensure_voice("elena", voice_record, voice_path)
    second = adapter.ensure_voice("elena", voice_record, voice_path)

    assert first == voice_path
    assert second == voice_path
    assert len(model.voice_design_calls) == 1


def test_qwen_adapter_generates_sentence_audio_in_order(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "elena.qvp"
    torch_store.save({"prompt": "saved"}, voice_path)
    adapter = QwenTtsAdapter(model=model, torch_module=torch_store, role_voice_paths={"Elena": voice_path})

    generated = adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Elena", "type": "dialogue", "text": "Hello."},
            {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Again."},
        ]
    )

    assert [item.sentence_idx for item in generated] == [0, 1]
    assert [len(item.samples) for item in generated] == [50, 25]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_qwen_adapter.py -v
```

Expected: fail because `qwen_adapter.py` does not exist.

- [ ] **Step 3: Implement optional Qwen adapter with injected model and torch for tests**

Create `src/ebook_tts_pipeline/tts/qwen_adapter.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


class QwenTtsAdapter:
    def __init__(
        self,
        model: Any | None = None,
        torch_module: Any | None = None,
        role_voice_paths: dict[str, Path] | None = None,
        language: str = "auto",
    ) -> None:
        self.model = model if model is not None else self._load_default_model()
        self.torch = torch_module if torch_module is not None else self._load_torch()
        self.role_voice_paths = role_voice_paths or {}
        self.language = language

    def ensure_voice(self, role_id: str, voice_record: dict, voice_path: Path) -> Path:
        if voice_path.exists():
            return voice_path
        seed = int(voice_record.get("voice_identity", {}).get("seed", 0))
        instruct = str(voice_record["voice_profile"]["qwen_instruct"])
        text = "This is the reference voice for this character."

        self._set_seed(seed)
        wavs, sample_rate = self.model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=self.language,
        )
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=(wavs[0], sample_rate),
            ref_text=text,
            x_vector_only_mode=True,
        )
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(prompt, voice_path)
        return voice_path

    def generate_sentences(self, jobs: list[dict]) -> list[GeneratedSentenceAudio]:
        generated: list[GeneratedSentenceAudio] = []
        for job in jobs:
            role = str(job["role"])
            voice_path = self.role_voice_paths[role]
            prompt = self.torch.load(voice_path, map_location="cpu", weights_only=False)
            wavs, sample_rate = self.model.generate_voice_clone(
                text=[str(job["text"])],
                language=[self.language],
                voice_clone_prompt=prompt,
            )
            generated.append(
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    role=role,
                    speech_type=str(job["type"]),
                    samples=np.asarray(wavs[0], dtype=np.float32),
                    sample_rate=sample_rate,
                )
            )
        return generated

    def _set_seed(self, seed: int) -> None:
        if hasattr(self.torch, "manual_seed"):
            self.torch.manual_seed(seed)

    def _load_torch(self) -> Any:
        import torch

        return torch

    def _load_default_model(self) -> Any:
        raise RuntimeError(
            "Qwen model loading must be wired with local qwen_tts runtime paths before real TTS use."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_qwen_adapter.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ebook_tts_pipeline/tts/qwen_adapter.py tests/test_qwen_adapter.py
git commit -m "Add Qwen TTS adapter boundary"
```

---

### Task 9: Pipeline Facade And Thin CLI

**Files:**
- Create: `src/ebook_tts_pipeline/pipeline.py`
- Create: `src/ebook_tts_pipeline/cli.py`
- Test: `tests/test_pipeline_facade.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing pipeline facade test**

Create `tests/test_pipeline_facade.py`:

```python
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter


class FakeLlmClient:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "new_characters": [
                {
                    "name": "Elena",
                    "profile": {"age_range": "young adult"},
                    "voice": {
                        "description": "young woman, soft",
                        "qwen_instruct": "A soft young adult female voice.",
                    },
                }
            ],
            "roles": ["Narrator", "Elena"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0], [1, 1, 1]],
        }


def test_pipeline_runs_tiny_chapter_with_fake_adapters(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text("It rained. \"Hello,\" Elena said.", encoding="utf-8")

    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(FakeLlmClient(), repair_retries=1),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["It rained.", "\"Hello,\" Elena said."],
    )

    result = pipeline.run_chapter("chapter_001", book_title="Demo", book_slug="demo")

    assert result["chapter"] == "chapter_001"
    assert (book_root / "sentence_segments" / "chapter_001.sentences.json").exists()
    assert (book_root / "annotations" / "chapter_001.annotation.json").exists()
    assert (book_root / "audio" / "chapter_001.wav").exists()
    assert (book_root / "audio" / "chapter_001.timeline.json").exists()
    assert (book_root / "voices" / "elena.qvp").exists()
```

- [ ] **Step 2: Write failing CLI argument test**

Create `tests/test_cli.py`:

```python
from ebook_tts_pipeline.cli import build_parser


def test_cli_has_run_chapter_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-chapter",
            "--book-root",
            "books/demo",
            "--book-title",
            "Demo",
            "--book-slug",
            "demo",
            "--chapter",
            "chapter_001",
            "--fake-tts",
        ]
    )

    assert args.command == "run-chapter"
    assert args.book_root == "books/demo"
    assert args.chapter == "chapter_001"
    assert args.fake_tts is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_pipeline_facade.py tests/test_cli.py -v
```

Expected: fail because pipeline and CLI modules do not exist.

- [ ] **Step 4: Implement UI-ready pipeline facade and CLI parser**

Create `src/ebook_tts_pipeline/pipeline.py`:

```python
from __future__ import annotations

from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.audio import ChapterAudioBuilder
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.domain import AnnotationResult, SentenceArtifact
from ebook_tts_pipeline.ingestion import SentenceSegmenter
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager, slugify_name
from ebook_tts_pipeline.tts.base import TtsAdapter


class AudiobookPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        annotation_service: AnnotationService,
        tts_adapter: TtsAdapter,
        tokenizer=None,
    ) -> None:
        self.config = config
        self.paths = BookPaths(config.book_root)
        self.registry = RegistryManager(self.paths)
        self.segmenter = SentenceSegmenter(tokenizer=tokenizer)
        self.annotation_service = annotation_service
        self.tts_adapter = tts_adapter

    def segment_chapter(self, chapter: str) -> SentenceArtifact:
        return self.segmenter.segment_chapter(self.paths, chapter)

    def annotate_chapter(self, chapter: str) -> AnnotationResult:
        artifact = SentenceArtifact.from_dict(read_json(self.paths.sentence_artifact(chapter)))
        registry = self.registry.load()
        result = self.annotation_service.annotate_window(
            chapter=chapter,
            sentences=artifact.sentences,
            registry=registry,
        )
        write_json_atomic(self.paths.annotation(chapter), result.to_dict())
        self.registry.add_new_characters(chapter, result.new_characters)
        return result

    def prepare_voices_for_annotation(self, annotation: AnnotationResult) -> None:
        registry = self.registry.load()
        role_records = {"Narrator": registry["narrator"]}
        for record in registry.get("characters", {}).values():
            role_records[str(record["display_name"])] = record

        for role_name in annotation.roles:
            record = role_records.get(role_name)
            if record is None:
                continue
            role_id = record.get("role_id", slugify_name(role_name))
            voice_path = self.paths.voice_qvp(role_id)
            self.tts_adapter.ensure_voice(role_id, record, voice_path)
            record["voice_config_path"] = f"voices/{role_id}.qvp"

        self.registry.save(registry)

    def build_sentence_jobs(self, chapter: str, annotation: AnnotationResult) -> list[dict]:
        artifact = SentenceArtifact.from_dict(read_json(self.paths.sentence_artifact(chapter)))
        sentence_by_idx = {sentence.idx: sentence.text for sentence in artifact.sentences}
        jobs: list[dict] = []
        for role_idx, type_idx, sentence_idx in annotation.script:
            jobs.append(
                {
                    "sentence_idx": sentence_idx,
                    "role": annotation.roles[role_idx],
                    "type": annotation.types[type_idx],
                    "text": sentence_by_idx[sentence_idx],
                }
            )
        return jobs

    def synthesize_chapter(self, chapter: str, annotation: AnnotationResult) -> dict:
        jobs = self.build_sentence_jobs(chapter, annotation)
        builder = ChapterAudioBuilder(
            tts_adapter=self.tts_adapter,
            pause_between_sentences_ms=self.config.pause_between_sentences_ms,
        )
        return builder.build_chapter_audio(
            chapter=chapter,
            jobs=jobs,
            audio_path=self.paths.chapter_audio(chapter),
            timeline_path=self.paths.chapter_timeline(chapter),
        )

    def run_chapter(self, chapter: str, book_title: str, book_slug: str) -> dict:
        self.registry.initialize_if_missing(book_title=book_title, book_slug=book_slug)
        self.segment_chapter(chapter)
        annotation = self.annotate_chapter(chapter)
        self.prepare_voices_for_annotation(annotation)
        return self.synthesize_chapter(chapter, annotation)
```

Create `src/ebook_tts_pipeline/cli.py`:

```python
from __future__ import annotations

import argparse

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ebook-tts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_chapter = subparsers.add_parser("run-chapter")
    run_chapter.add_argument("--book-root", required=True)
    run_chapter.add_argument("--book-title", required=True)
    run_chapter.add_argument("--book-slug", required=True)
    run_chapter.add_argument("--chapter", required=True)
    run_chapter.add_argument("--fake-tts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-chapter":
        config = PipelineConfig.from_env(book_root=args.book_root)
        llm_client = AnthropicJsonClient(
            api_key=config.require_anthropic_key(),
            model=config.anthropic_model,
            temperature=config.anthropic_temperature,
            max_tokens=config.anthropic_max_tokens,
        )
        annotation_service = AnnotationService(
            client=llm_client,
            repair_retries=config.annotation_repair_retries,
        )
        tts_adapter = FakeTtsAdapter() if args.fake_tts else QwenTtsAdapter()
        pipeline = AudiobookPipeline(
            config=config,
            annotation_service=annotation_service,
            tts_adapter=tts_adapter,
        )
        pipeline.run_chapter(args.chapter, book_title=args.book_title, book_slug=args.book_slug)
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_pipeline_facade.py tests/test_cli.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/ebook_tts_pipeline/pipeline.py src/ebook_tts_pipeline/cli.py tests/test_pipeline_facade.py tests/test_cli.py
git commit -m "Add pipeline facade and CLI"
```

---

### Task 10: Fixture End-To-End Test, Example Config, And Runbook Commands

**Files:**
- Create: `tests/fixtures/tiny_book/source/book.txt`
- Create: `tests/test_end_to_end_fake_pipeline.py`
- Create: `config.example.toml`
- Modify: `docs/runbooks/manual-tts-stack.txt`

- [ ] **Step 1: Add the tiny fixture book**

Create `tests/fixtures/tiny_book/source/book.txt`:

```text
Chapter 1
It rained on the old road.
"Hello," Elena said.

Chapter 2
The sun rose.
Marcus closed the gate.
```

- [ ] **Step 2: Write the end-to-end fake pipeline test**

Create `tests/test_end_to_end_fake_pipeline.py`:

```python
import shutil

from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.ingestion import ChapterSplitter
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter


class TinyBookLlm:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "new_characters": [
                {
                    "name": "Elena",
                    "profile": {"age_range": "young adult", "gender": "female"},
                    "voice": {
                        "description": "young woman, soft",
                        "qwen_instruct": "A soft young adult female voice.",
                    },
                }
            ],
            "roles": ["Narrator", "Elena"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0], [1, 1, 1]],
        }


def test_fake_pipeline_from_whole_book_to_audio_outputs(tmp_path):
    book_root = tmp_path / "tiny_book"
    shutil.copytree("tests/fixtures/tiny_book", book_root)
    paths = BookPaths(book_root)

    split = ChapterSplitter().split_source_book(paths)
    assert split.chapters == ["chapter_001", "chapter_002"]

    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(TinyBookLlm(), repair_retries=1),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["It rained on the old road.", "\"Hello,\" Elena said."],
    )

    timeline = pipeline.run_chapter("chapter_001", book_title="Tiny Book", book_slug="tiny_book")

    assert timeline["sentences"][0]["sentence_idx"] == 0
    assert (book_root / "registry.json").exists()
    assert (book_root / "voices" / "elena.qvp").exists()
    assert (book_root / "audio" / "chapter_001.wav").exists()
```

- [ ] **Step 3: Create example config**

Create `config.example.toml`:

```toml
book_root = "books/demo"
anthropic_model = "claude-haiku-4-5-20251001"
qwen_model_choice = "1.7B"
qwen_device = "auto"
qwen_precision = "bf16"
qwen_attention = "auto"
max_llm_window_chars = 48000
max_tts_window_chars = 6000
max_tts_roles = 8
pause_between_sentences_ms = 250
```

- [ ] **Step 4: Update runbook with exact v1 commands**

Append this section to `docs/runbooks/manual-tts-stack.txt`:

```text
V1 CLI Commands
---------------
Install local package for development:

  python -m pip install -e ".[dev]"

Run all tests:

  python -m pytest -v

Run a chapter with fake TTS for pipeline validation:

  ebook-tts run-chapter --book-root books/demo --book-title "Demo" --book-slug demo --chapter chapter_001 --fake-tts

Run a chapter with real Qwen TTS after local Qwen wiring is complete:

  ebook-tts run-chapter --book-root books/demo --book-title "Demo" --book-slug demo --chapter chapter_001
```

- [ ] **Step 5: Run the full fake test suite**

Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/tiny_book tests/test_end_to_end_fake_pipeline.py config.example.toml docs/runbooks/manual-tts-stack.txt
git commit -m "Add fake end-to-end pipeline coverage"
```

---

## Final Verification

- [ ] Run the full test suite:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] Verify importable CLI script metadata:

```bash
python -c "from ebook_tts_pipeline.cli import build_parser; print(build_parser().prog)"
```

Expected output:

```text
ebook-tts
```

- [ ] Verify no accidental LM Studio or FastAPI dependency was added:

```bash
rg -n "LM Studio|localhost:1234|FastAPI|ComfyUI workflow" src tests docs
```

Expected: only historical notes in design/runbook docs, no source or test dependency on those systems.

---

## Implementation Notes

- Keep `AudiobookPipeline` methods small and return dictionaries/dataclasses that a future UI can display.
- Do not put UI assumptions in domain models.
- Do not import Anthropic or Qwen at module import time except inside their adapter classes.
- Keep code compatible with Python 3.9 in this workspace; avoid Python 3.10-only union syntax in production files.
- Preserve sentence indexes from `sentence_segments/*.sentences.json` through every later artifact.
- Reuse `.qvp` files whenever present; regenerate voices only when a voice path is missing or intentionally replaced.
- Keep generated book data under `books/<book_slug>/`, not under `src/` or `tests/`.

## Self-Review

Spec coverage:

- Deterministic chapter splitting and sentence segmentation: Tasks 3 and 10.
- Sentence artifacts as canonical downstream input: Tasks 2, 3, and 9.
- Anthropic annotation with Haiku default and repair retry: Tasks 1 and 6.
- Strict compact JSON validation: Task 4.
- Registry with character profile, voice profile, and voice identity: Task 5.
- Distinct voices for similar characters: Task 5.
- `.qvp` reuse for consistency: Tasks 7, 8, and 9.
- TTS max 8 roles and sentence atomicity: Task 4.
- Per-sentence audio and timeline metadata: Task 7.
- Direct Qwen wrapper boundary: Task 8.
- UI-ready class boundaries: Tasks 1 through 9, especially Task 9.

Placeholder scan:

- The plan uses concrete file paths, tests, code snippets, commands, and expected outcomes.
- The Qwen adapter intentionally raises a runtime error for default model loading until local model wiring is supplied; tests inject a model. This is a defined boundary, not an unfinished task.

Type consistency:

- `SentenceArtifact`, `AnnotationResult`, `TtsAdapter`, `GeneratedSentenceAudio`, and `AudiobookPipeline` signatures are introduced before later tasks reference them.
- `sentence_idx`, `role`, `type`, and `text` job keys are consistent across windowing, TTS, audio, and pipeline tasks.
