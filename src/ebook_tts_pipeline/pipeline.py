from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.audio import ChapterAudioBuilder
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.domain import AnnotationResult, SentenceArtifact
from ebook_tts_pipeline.ingestion import SentenceSegmenter
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager, slugify_name
from ebook_tts_pipeline.tts.base import TtsAdapter
from ebook_tts_pipeline.windowing import build_tts_windows


class AudiobookPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        annotation_service: AnnotationService,
        tts_adapter: TtsAdapter,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
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
        role_records: Dict[str, Dict] = {"Narrator": registry["narrator"]}
        for record in registry.get("characters", {}).values():
            role_records[str(record["display_name"])] = record

        for role_name in annotation.roles:
            record = role_records.get(role_name)
            if record is None:
                raise ValueError(f"No registry record exists for annotated role: {role_name}")
            role_id = str(record.get("role_id", slugify_name(role_name)))
            voice_path = self.paths.voice_qvp(role_id)
            self.tts_adapter.ensure_voice(role_id, record, voice_path)
            if hasattr(self.tts_adapter, "role_voice_paths"):
                self.tts_adapter.role_voice_paths[role_name] = voice_path
                self.tts_adapter.role_voice_paths[role_id] = voice_path
            record["voice_config_path"] = f"voices/{role_id}.qvp"

        self.registry.save(registry)

    def build_sentence_jobs(self, chapter: str, annotation: AnnotationResult) -> List[Dict]:
        artifact = SentenceArtifact.from_dict(read_json(self.paths.sentence_artifact(chapter)))
        sentence_by_idx = {sentence.idx: sentence.text for sentence in artifact.sentences}
        jobs: List[Dict] = []
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

    def synthesize_chapter(self, chapter: str, annotation: AnnotationResult) -> Dict:
        jobs = self.build_sentence_jobs(chapter, annotation)
        windows = build_tts_windows(
            jobs,
            max_chars=self.config.max_tts_window_chars,
            max_roles=self.config.max_tts_roles,
        )
        ordered_jobs = [job for window in windows for job in window.jobs]
        builder = ChapterAudioBuilder(
            tts_adapter=self.tts_adapter,
            pause_between_sentences_ms=self.config.pause_between_sentences_ms,
        )
        return builder.build_chapter_audio(
            chapter=chapter,
            jobs=ordered_jobs,
            audio_path=self.paths.chapter_audio(chapter),
            timeline_path=self.paths.chapter_timeline(chapter),
        )

    def run_chapter(self, chapter: str, book_title: str, book_slug: str) -> Dict:
        self.registry.initialize_if_missing(book_title=book_title, book_slug=book_slug)
        self.segment_chapter(chapter)
        annotation = self.annotate_chapter(chapter)
        self.prepare_voices_for_annotation(annotation)
        return self.synthesize_chapter(chapter, annotation)
