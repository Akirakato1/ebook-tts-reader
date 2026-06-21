from __future__ import annotations

import argparse
from typing import List, Optional

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


def main(argv: Optional[List[str]] = None) -> int:
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
        tts_adapter = (
            FakeTtsAdapter()
            if args.fake_tts
            else QwenTtsAdapter(
                model_root=config.qwen_model_root,
                model_choice=config.qwen_model_choice,
                device=config.qwen_device,
                precision=config.qwen_precision,
                attention=config.qwen_attention,
            )
        )
        pipeline = AudiobookPipeline(
            config=config,
            annotation_service=annotation_service,
            tts_adapter=tts_adapter,
        )
        pipeline.run_chapter(args.chapter, book_title=args.book_title, book_slug=args.book_slug)
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2
