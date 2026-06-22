from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from ebook_tts_pipeline.annotation.anthropic_client import AnnotationModelOutputError
from ebook_tts_pipeline.annotation.global_registry import (
    GlobalRegistryChapter,
    GlobalRegistryService,
)
from ebook_tts_pipeline.annotation.merge import merge_annotation_windows
from ebook_tts_pipeline.annotation.service import AnnotationService, known_annotation_role_names
from ebook_tts_pipeline.annotation.validator import AnnotationValidationError, validate_annotation
from ebook_tts_pipeline.audio import ChapterAudioBuilder
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.domain import AnnotationResult, SentenceArtifact
from ebook_tts_pipeline.ingestion import SentenceSegmenter
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import (
    RegistryManager,
    resolve_effective_voice,
    voice_profile_hash,
)
from ebook_tts_pipeline.temp_registry import (
    ChapterTempRegistryManager,
    normalize_annotation_local_speakers,
    resolve_temp_voice,
)
from ebook_tts_pipeline.tts.base import TtsAdapter
from ebook_tts_pipeline.tts.script import build_tts_script
from ebook_tts_pipeline.windowing import build_llm_windows, build_tts_windows


class AudiobookPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        annotation_service: AnnotationService,
        tts_adapter: TtsAdapter,
        global_registry_service: Optional[GlobalRegistryService] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.config = config
        self.paths = BookPaths(config.book_root)
        self.registry = RegistryManager(self.paths)
        self.segmenter = SentenceSegmenter(tokenizer=tokenizer)
        self.annotation_service = annotation_service
        self.global_registry_service = global_registry_service
        self.tts_adapter = tts_adapter

    def segment_chapter(self, chapter: str) -> SentenceArtifact:
        return self.segmenter.segment_chapter(self.paths, chapter)

    def build_global_registry(
        self,
        book_title: Optional[str] = None,
        book_slug: Optional[str] = None,
    ) -> int:
        if self.global_registry_service is None:
            raise RuntimeError("Global registry service is not configured.")
        if not self.paths.registry.exists():
            self.registry.initialize_if_missing(
                book_title=book_title or "Untitled Book",
                book_slug=book_slug or self.paths.root.name or "book",
            )
        registry = self.registry.load()
        title = book_title or str(registry.get("book", {}).get("title", "Untitled Book"))
        chapters = [
            GlobalRegistryChapter(
                chapter=chapter_file.stem,
                title=self._chapter_title(chapter_file),
                text=chapter_file.read_text(encoding="utf-8", errors="replace"),
            )
            for chapter_file in sorted((self.paths.root / "chapters").glob("*.txt"))
        ]
        discovered_count = 0
        for chunk in _chunk_global_registry_chapters(
            chapters,
            max_chars=self.config.global_registry_window_chars,
        ):
            result = self.global_registry_service.discover_characters(
                book_title=title,
                registry=self.registry.load(),
                chapters=chunk,
            )
            self.registry.merge_global_characters("global_registry", result.characters)
            discovered_count += len(result.characters)
        return discovered_count

    def annotate_chapter(self, chapter: str, lock_registry: bool = True) -> AnnotationResult:
        artifact = SentenceArtifact.from_dict(read_json(self.paths.sentence_artifact(chapter)))
        initial_known_names = known_annotation_role_names(self.registry.load())
        window_results: List[AnnotationResult] = []

        for window in build_llm_windows(
            artifact.annotation_units,
            self.config.max_llm_window_chars,
            max_sentences=self.config.max_llm_window_sentences,
        ):
            window_results.extend(
                self._annotate_sentences_with_fallback(
                    chapter,
                    window.sentences,
                    lock_registry=lock_registry,
                )
            )

        merged = merge_annotation_windows(window_results, self.registry.load())
        validate_annotation(
            merged,
            expected_sentence_indices=[unit.idx for unit in artifact.annotation_units],
            known_names=initial_known_names,
        )
        write_json_atomic(self.paths.annotation(chapter), merged.to_dict())
        return merged

    def _annotate_sentences_with_fallback(
        self,
        chapter: str,
        sentences: List[Sentence],
        lock_registry: bool = True,
    ) -> List[AnnotationResult]:
        try:
            result = self.annotation_service.annotate_window(
                chapter=chapter,
                sentences=sentences,
                registry=self.registry.load(),
                lock_registry=lock_registry,
            )
        except (AnnotationModelOutputError, AnnotationValidationError):
            if len(sentences) <= 1:
                raise
            midpoint = len(sentences) // 2
            return (
                self._annotate_sentences_with_fallback(
                    chapter,
                    sentences[:midpoint],
                    lock_registry=lock_registry,
                )
                + self._annotate_sentences_with_fallback(
                    chapter,
                    sentences[midpoint:],
                    lock_registry=lock_registry,
                )
            )

        return [result]

    def prepare_voices_for_annotation(
        self,
        annotation: AnnotationResult,
        chapter: Optional[str] = None,
        force_regenerate: bool = False,
    ) -> None:
        registry = self.registry.load()
        annotation = normalize_annotation_local_speakers(annotation)
        temp_manager = ChapterTempRegistryManager(self.paths)
        temp_registry = (
            temp_manager.write_for_annotation(chapter, registry, annotation)
            if chapter is not None
            else {"chapter": "", "speakers": {}}
        )
        prepared_role_ids = set()

        for role_idx, type_idx, _ in annotation.script:
            role_name = annotation.roles[role_idx]
            type_name = annotation.types[type_idx]
            try:
                effective = resolve_effective_voice(registry, role_name, type_name)
            except ValueError:
                effective = resolve_temp_voice(temp_registry, role_name, type_name)
                if effective is None:
                    raise
            record = effective["voice_record"]
            role_id = str(effective["role_id"])
            role_display = str(effective["role"])
            if role_id in prepared_role_ids:
                continue
            prepared_role_ids.add(role_id)

            voice_path = self._voice_path_for_record(role_id, record)
            current_hash = voice_profile_hash(record)
            cached_hash = record.get("voice_config_hash")
            should_generate = force_regenerate or not voice_path.exists() or cached_hash != current_hash
            if should_generate:
                adapter_record = dict(record)
                adapter_record["_force_regenerate"] = force_regenerate or (
                    voice_path.exists() and cached_hash != current_hash
                )
                self.tts_adapter.ensure_voice(role_id, adapter_record, voice_path)
                record["voice_config_hash"] = current_hash

            if hasattr(self.tts_adapter, "role_voice_paths"):
                self.tts_adapter.role_voice_paths[role_display] = voice_path
                self.tts_adapter.role_voice_paths[role_id] = voice_path
            record["voice_config_path"] = voice_path.relative_to(self.paths.root).as_posix()

        self.registry.save(registry)
        if chapter is not None:
            temp_manager.save(chapter, temp_registry)

    def build_sentence_jobs(self, chapter: str, annotation: AnnotationResult) -> List[Dict]:
        artifact = SentenceArtifact.from_dict(read_json(self.paths.sentence_artifact(chapter)))
        registry = self.registry.load()
        annotation = normalize_annotation_local_speakers(annotation)
        if self.paths.annotation(chapter).exists():
            write_json_atomic(self.paths.annotation(chapter), annotation.to_dict())
        temp_registry = ChapterTempRegistryManager(self.paths).write_for_annotation(chapter, registry, annotation)
        script = build_tts_script(
            chapter=chapter,
            annotation=annotation,
            artifact=artifact,
            registry=registry,
            max_chars=self.config.max_tts_window_chars,
            max_roles=self.config.max_tts_roles,
            language="auto",
            temp_registry=temp_registry,
        )
        write_json_atomic(self.paths.tts_script(chapter), script.to_dict())
        qwen_script_path = self.paths.qwen_script(chapter)
        qwen_script_path.parent.mkdir(parents=True, exist_ok=True)
        qwen_script_path.write_text(script.qwen_dialogue_text + "\n", encoding="utf-8")
        return [job.to_adapter_job() for job in script.jobs]

    def _voice_path_for_record(self, role_id: str, record: Dict[str, object]) -> Path:
        configured = str(record.get("voice_config_path") or "").strip()
        if configured.startswith("voices/_temp/"):
            return self.paths.root / configured
        return self.paths.voice_qvp(role_id)

    def synthesize_chapter(self, chapter: str, annotation: AnnotationResult) -> Dict:
        jobs = self.build_sentence_jobs(chapter, annotation)
        return self.synthesize_jobs(chapter, jobs)

    def synthesize_chapter_from_tts_script(self, chapter: str) -> Dict:
        script = read_json(self.paths.tts_script(chapter))
        return self.synthesize_jobs(chapter, [dict(job) for job in script["jobs"]])

    def synthesize_jobs(self, chapter: str, jobs: List[Dict]) -> Dict:
        windows = build_tts_windows(
            jobs,
            max_chars=self.config.max_tts_window_chars,
            max_roles=self.config.max_tts_roles,
        )
        builder = ChapterAudioBuilder(
            tts_adapter=self.tts_adapter,
            pause_between_sentences_ms=self.config.pause_between_sentences_ms,
            intra_sentence_pause_ms=self.config.intra_sentence_pause_ms,
            tts_speed=self.config.tts_speed,
        )
        return builder.build_chapter_audio_from_windows(
            chapter=chapter,
            job_windows=[window.jobs for window in windows],
            audio_path=self.paths.chapter_audio(chapter),
            timeline_path=self.paths.chapter_timeline(chapter),
        )

    def run_chapter(self, chapter: str, book_title: str, book_slug: str) -> Dict:
        self.registry.initialize_if_missing(book_title=book_title, book_slug=book_slug)
        self.segment_chapter(chapter)
        annotation = self.annotate_chapter(chapter)
        self.prepare_voices_for_annotation(annotation, chapter=chapter)
        return self.synthesize_chapter(chapter, annotation)

    def _chapter_title(self, chapter_file) -> str:
        for line in chapter_file.read_text(encoding="utf-8", errors="replace").splitlines():
            title = " ".join(line.split())
            if title:
                return title[:120]
        return chapter_file.stem


def _chunk_global_registry_chapters(
    chapters: List[GlobalRegistryChapter],
    max_chars: int,
) -> List[List[GlobalRegistryChapter]]:
    if max_chars <= 0:
        return [chapters] if chapters else []

    chunks: List[List[GlobalRegistryChapter]] = []
    current: List[GlobalRegistryChapter] = []
    current_chars = 0

    for chapter in chapters:
        chapter_chars = len(chapter.text)
        if current and current_chars + chapter_chars > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(chapter)
        current_chars += chapter_chars

    if current:
        chunks.append(current)

    return chunks
