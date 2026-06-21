from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ebook_tts_pipeline.annotation.merge import merge_annotation_windows
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.annotation.validator import validate_annotation
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
from ebook_tts_pipeline.tts.base import TtsAdapter
from ebook_tts_pipeline.tts.script import build_tts_script
from ebook_tts_pipeline.windowing import build_llm_windows, build_tts_windows


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
        initial_known_names = self.registry.known_names()
        window_results: List[AnnotationResult] = []

        for window in build_llm_windows(artifact.sentences, self.config.max_llm_window_chars):
            result = self.annotation_service.annotate_window(
                chapter=chapter,
                sentences=window.sentences,
                registry=self.registry.load(),
            )
            window_results.append(result)
            if result.new_characters:
                self.registry.add_new_characters(chapter, result.new_characters)

        merged = merge_annotation_windows(window_results, self.registry.load())
        validate_annotation(
            merged,
            expected_sentence_indices=[sentence.idx for sentence in artifact.sentences],
            known_names=initial_known_names,
        )
        write_json_atomic(self.paths.annotation(chapter), merged.to_dict())
        return merged

    def prepare_voices_for_annotation(self, annotation: AnnotationResult) -> None:
        registry = self.registry.load()
        prepared_role_ids = set()

        for role_idx, type_idx, _ in annotation.script:
            role_name = annotation.roles[role_idx]
            type_name = annotation.types[type_idx]
            effective = resolve_effective_voice(registry, role_name, type_name)
            record = effective["voice_record"]
            role_id = str(effective["role_id"])
            role_display = str(effective["role"])
            if role_id in prepared_role_ids:
                continue
            prepared_role_ids.add(role_id)

            voice_path = self.paths.voice_qvp(role_id)
            current_hash = voice_profile_hash(record)
            cached_hash = record.get("voice_config_hash")
            should_generate = not voice_path.exists() or cached_hash != current_hash
            if should_generate:
                adapter_record = dict(record)
                adapter_record["_force_regenerate"] = voice_path.exists() and cached_hash != current_hash
                self.tts_adapter.ensure_voice(role_id, adapter_record, voice_path)
                record["voice_config_hash"] = current_hash

            if hasattr(self.tts_adapter, "role_voice_paths"):
                self.tts_adapter.role_voice_paths[role_display] = voice_path
                self.tts_adapter.role_voice_paths[role_id] = voice_path
            record["voice_config_path"] = f"voices/{role_id}.qvp"

        self.registry.save(registry)

    def build_sentence_jobs(self, chapter: str, annotation: AnnotationResult) -> List[Dict]:
        artifact = SentenceArtifact.from_dict(read_json(self.paths.sentence_artifact(chapter)))
        script = build_tts_script(
            chapter=chapter,
            annotation=annotation,
            artifact=artifact,
            registry=self.registry.load(),
            max_chars=self.config.max_tts_window_chars,
            max_roles=self.config.max_tts_roles,
            language="auto",
        )
        write_json_atomic(self.paths.tts_script(chapter), script.to_dict())
        qwen_script_path = self.paths.qwen_script(chapter)
        qwen_script_path.parent.mkdir(parents=True, exist_ok=True)
        qwen_script_path.write_text(script.qwen_dialogue_text + "\n", encoding="utf-8")
        return [job.to_adapter_job() for job in script.jobs]

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
