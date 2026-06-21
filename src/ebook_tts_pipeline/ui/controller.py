from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.epub_ingestion import EpubChapterExtractor, EpubExtractResult
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter


class ChapterStage(str, Enum):
    RAW = "raw"
    SEGMENTED = "segmented"
    ANNOTATED = "annotated"
    SCRIPTED = "scripted"
    AUDIO = "audio"


@dataclass(frozen=True)
class ChapterRow:
    chapter: str
    index: int
    title: str
    stage: ChapterStage
    audio_path: Path


@dataclass(frozen=True)
class ChapterActionResult:
    chapter: str
    stage: ChapterStage
    message: str


class ChapterExtractor(Protocol):
    def extract(self, epub_path: Union[str, Path], paths: BookPaths) -> EpubExtractResult:
        ...


PipelineFactory = Callable[[PipelineConfig, bool, bool], AudiobookPipeline]
AudioOpener = Callable[[Path], None]


class PrototypeUiController:
    def __init__(
        self,
        book_root: Union[str, Path],
        pipeline_factory: Optional[PipelineFactory] = None,
        extractor: Optional[ChapterExtractor] = None,
        audio_opener: Optional[AudioOpener] = None,
        fake_tts: bool = False,
    ) -> None:
        self.book_root = Path(book_root)
        self.paths = BookPaths(self.book_root)
        self.pipeline_factory = pipeline_factory or _default_pipeline_factory
        self.extractor = extractor or EpubChapterExtractor()
        self.audio_opener = audio_opener or _open_audio_file
        self.fake_tts = fake_tts

    def set_book_root(self, book_root: Union[str, Path]) -> None:
        self.book_root = Path(book_root)
        self.paths = BookPaths(self.book_root)

    def chapter_rows(self) -> List[ChapterRow]:
        chapters_dir = self.book_root / "chapters"
        toc = self._load_toc()
        if not chapters_dir.exists():
            return []
        rows: List[ChapterRow] = []
        for index, chapter_file in enumerate(sorted(chapters_dir.glob("*.txt")), start=1):
            chapter = chapter_file.stem
            rows.append(
                ChapterRow(
                    chapter=chapter,
                    index=index,
                    title=toc.get(chapter, self._chapter_title(chapter_file, chapter)),
                    stage=self.chapter_stage(chapter),
                    audio_path=self.paths.chapter_audio(chapter),
                )
            )
        return rows

    def chapter_stage(self, chapter: str) -> ChapterStage:
        if self.paths.chapter_audio(chapter).exists():
            return ChapterStage.AUDIO
        if self.paths.tts_script(chapter).exists() and self.paths.qwen_script(chapter).exists():
            return ChapterStage.SCRIPTED
        if self.paths.annotation(chapter).exists():
            return ChapterStage.ANNOTATED
        if self.paths.sentence_artifact(chapter).exists():
            return ChapterStage.SEGMENTED
        return ChapterStage.RAW

    def registry_text(self) -> str:
        if not self.paths.registry.exists():
            return "{}\n"
        return json.dumps(read_json(self.paths.registry), indent=2, ensure_ascii=False) + "\n"

    def save_registry_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Registry JSON is invalid: {exc}") from exc
        write_json_atomic(self.paths.registry, payload)

    def load_epub(self, epub_path: Union[str, Path], title: str, slug: str) -> EpubExtractResult:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        result = self.extractor.extract(epub_path, self.paths)
        pipeline = self._pipeline(needs_llm=False)
        pipeline.registry.initialize_if_missing(book_title=title, book_slug=slug)
        toc = []
        for index, chapter in enumerate(result.chapters, start=1):
            pipeline.segment_chapter(chapter)
            source = result.sources[index - 1] if index - 1 < len(result.sources) else ""
            toc.append(
                {
                    "index": index,
                    "chapter": chapter,
                    "title": self._chapter_title(self.paths.chapter_text(chapter), chapter),
                    "source": source,
                }
            )
        write_json_atomic(self.book_root / "toc.json", {"chapters": toc})
        return result

    def run_next_chapter_action(self, chapter: str) -> ChapterActionResult:
        stage = self.chapter_stage(chapter)
        if stage == ChapterStage.AUDIO:
            self.audio_opener(self.paths.chapter_audio(chapter))
            return ChapterActionResult(chapter=chapter, stage=stage, message="Opened audio file.")

        if stage == ChapterStage.SCRIPTED:
            pipeline = self._pipeline(needs_llm=False)
            annotation = self._load_annotation(chapter)
            pipeline.prepare_voices_for_annotation(annotation)
            pipeline.synthesize_chapter_from_tts_script(chapter)
            return ChapterActionResult(chapter=chapter, stage=ChapterStage.AUDIO, message="Generated audio.")

        if stage == ChapterStage.ANNOTATED:
            pipeline = self._pipeline(needs_llm=False)
            pipeline.build_sentence_jobs(chapter, self._load_annotation(chapter))
            return ChapterActionResult(chapter=chapter, stage=ChapterStage.SCRIPTED, message="Generated scripts.")

        pipeline = self._pipeline(needs_llm=True)
        if stage == ChapterStage.RAW:
            pipeline.segment_chapter(chapter)
        pipeline.annotate_chapter(chapter)
        return ChapterActionResult(chapter=chapter, stage=ChapterStage.ANNOTATED, message="Annotated chapter.")

    def _pipeline(self, needs_llm: bool) -> AudiobookPipeline:
        return self.pipeline_factory(PipelineConfig.from_env(str(self.book_root)), needs_llm, self.fake_tts)

    def _load_annotation(self, chapter: str) -> AnnotationResult:
        return AnnotationResult.from_dict(read_json(self.paths.annotation(chapter)))

    def _load_toc(self) -> Dict[str, str]:
        toc_path = self.book_root / "toc.json"
        if not toc_path.exists():
            return {}
        payload = read_json(toc_path)
        return {
            str(item.get("chapter")): str(item.get("title"))
            for item in payload.get("chapters", [])
            if item.get("chapter") and item.get("title")
        }

    def _chapter_title(self, chapter_file: Path, fallback: str) -> str:
        if not chapter_file.exists():
            return fallback
        for line in chapter_file.read_text(encoding="utf-8", errors="replace").splitlines():
            title = " ".join(line.split())
            if title:
                return title[:120]
        return fallback


def _default_pipeline_factory(config: PipelineConfig, needs_llm: bool, fake_tts: bool) -> AudiobookPipeline:
    return AudiobookPipeline(
        config=config,
        annotation_service=AnnotationService(
            client=_build_llm_client(config) if needs_llm else _UnavailableJsonClient(),
            repair_retries=config.annotation_repair_retries,
        ),
        tts_adapter=FakeTtsAdapter() if fake_tts else _build_qwen_adapter(config),
    )


def _build_llm_client(config: PipelineConfig) -> AnthropicJsonClient:
    return AnthropicJsonClient(
        api_key=config.require_anthropic_key(),
        model=config.anthropic_model,
        temperature=config.anthropic_temperature,
        max_tokens=config.anthropic_max_tokens,
    )


def _build_qwen_adapter(config: PipelineConfig) -> QwenTtsAdapter:
    return QwenTtsAdapter(
        model_root=config.qwen_model_root,
        model_choice=config.qwen_model_choice,
        device=config.qwen_device,
        precision=config.qwen_precision,
        attention=config.qwen_attention,
        max_batch_size=config.qwen_batch_size,
    )


def _open_audio_file(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())


class _UnavailableJsonClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        raise RuntimeError("This UI action does not perform LLM annotation.")
