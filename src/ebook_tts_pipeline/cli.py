from __future__ import annotations

import argparse
from typing import Any
from typing import List, Optional

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.json_io import read_json
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

    segment_chapter = subparsers.add_parser("segment-chapter")
    _add_book_chapter_args(segment_chapter)

    annotate_chapter = subparsers.add_parser("annotate-chapter")
    annotate_chapter.add_argument("--book-root", required=True)
    annotate_chapter.add_argument("--book-title", required=True)
    annotate_chapter.add_argument("--book-slug", required=True)
    annotate_chapter.add_argument("--chapter", required=True)

    build_tts_script = subparsers.add_parser("build-tts-script")
    _add_book_chapter_args(build_tts_script)

    prepare_voices = subparsers.add_parser("prepare-voices")
    _add_book_chapter_args(prepare_voices)
    prepare_voices.add_argument("--fake-tts", action="store_true")
    prepare_voices.add_argument("--regenerate-voices", action="store_true")

    synthesize_chapter = subparsers.add_parser("synthesize-chapter")
    _add_book_chapter_args(synthesize_chapter)
    synthesize_chapter.add_argument("--fake-tts", action="store_true")
    synthesize_chapter.add_argument("--rebuild-tts-script", action="store_true")
    synthesize_chapter.add_argument("--regenerate-voices", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-chapter":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=True, fake_tts=args.fake_tts)
        pipeline.run_chapter(args.chapter, book_title=args.book_title, book_slug=args.book_slug)
        return 0
    if args.command == "segment-chapter":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=False, fake_tts=True)
        pipeline.segment_chapter(args.chapter)
        return 0
    if args.command == "annotate-chapter":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=True, fake_tts=True)
        pipeline.registry.initialize_if_missing(book_title=args.book_title, book_slug=args.book_slug)
        pipeline.annotate_chapter(args.chapter)
        return 0
    if args.command == "build-tts-script":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=False, fake_tts=True)
        pipeline.build_sentence_jobs(args.chapter, _load_annotation(pipeline, args.chapter))
        return 0
    if args.command == "prepare-voices":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=False, fake_tts=args.fake_tts)
        pipeline.prepare_voices_for_annotation(
            _load_annotation(pipeline, args.chapter),
            force_regenerate=args.regenerate_voices,
        )
        return 0
    if args.command == "synthesize-chapter":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=False, fake_tts=args.fake_tts)
        annotation = _load_annotation(pipeline, args.chapter)
        pipeline.prepare_voices_for_annotation(
            annotation,
            force_regenerate=args.regenerate_voices,
        )
        if args.rebuild_tts_script or not pipeline.paths.tts_script(args.chapter).exists():
            pipeline.synthesize_chapter(args.chapter, annotation)
        else:
            pipeline.synthesize_chapter_from_tts_script(args.chapter)
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2


def _add_book_chapter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--book-root", required=True)
    parser.add_argument("--chapter", required=True)


def _build_pipeline(config: PipelineConfig, needs_llm: bool, fake_tts: bool) -> AudiobookPipeline:
    return AudiobookPipeline(
        config=config,
        annotation_service=AnnotationService(
            client=_build_llm_client(config) if needs_llm else _UnavailableJsonClient(),
            repair_retries=config.annotation_repair_retries,
            failure_logger=FailureLogger(
                config.debug_log_root,
                context={"book_root": config.book_root},
            ),
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


def _load_annotation(pipeline: AudiobookPipeline, chapter: str) -> AnnotationResult:
    return AnnotationResult.from_dict(read_json(pipeline.paths.annotation(chapter)))


class _UnavailableJsonClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        raise RuntimeError("This CLI command does not perform LLM annotation.")
