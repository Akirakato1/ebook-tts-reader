from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any
from typing import List, Optional

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.global_registry import GlobalRegistryService
from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionService
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.benchmarks.window_sweep import (
    PerfLogReader,
    SweepConfig,
    collect_machine_profile,
    run_sweep,
    write_outputs,
)
from ebook_tts_pipeline.config import PipelineConfig, resolve_project_path, resolve_qwen_model_root
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.json_io import read_json
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.read_along.session import ReadAlongSession
from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.runtime_logging import log_runtime_step
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter
from ebook_tts_pipeline.tts.vllm_omni_adapter import WslVllmOmniQwenAdapter
from ebook_tts_pipeline.tts.wsl_adapter import WslQwenWorkerAdapter


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

    build_global_registry = subparsers.add_parser("build-global-registry")
    build_global_registry.add_argument("--book-root", required=True)
    build_global_registry.add_argument("--book-title", required=True)
    build_global_registry.add_argument("--book-slug", required=True)

    prepare_voices = subparsers.add_parser("prepare-voices")
    _add_book_chapter_args(prepare_voices)
    prepare_voices.add_argument("--fake-tts", action="store_true")
    prepare_voices.add_argument("--regenerate-voices", action="store_true")

    synthesize_chapter = subparsers.add_parser("synthesize-chapter")
    _add_book_chapter_args(synthesize_chapter)
    synthesize_chapter.add_argument("--fake-tts", action="store_true")
    synthesize_chapter.add_argument("--rebuild-tts-script", action="store_true")
    synthesize_chapter.add_argument("--regenerate-voices", action="store_true")

    benchmark_readalong = subparsers.add_parser("benchmark-readalong")
    benchmark_readalong.add_argument("--book-root", required=True)
    benchmark_readalong.add_argument("--chapter", required=True)
    benchmark_readalong.add_argument("--start-unit", type=int, default=0)
    benchmark_readalong.add_argument("--unit-count", type=int, default=20)
    benchmark_readalong.add_argument("--target-buffer-seconds", type=float, default=20.0)
    benchmark_readalong.add_argument("--playback-speed", type=float, default=1.0)
    benchmark_readalong.add_argument("--generation-mode", choices=["precise", "balanced", "fast"], default="balanced")

    benchmark_window_sweep = subparsers.add_parser("benchmark-window-sweep")
    benchmark_window_sweep.add_argument("--book-root", required=True)
    benchmark_window_sweep.add_argument("--chapter", required=True)
    benchmark_window_sweep.add_argument("--start-chars", type=int, default=100)
    benchmark_window_sweep.add_argument("--step-chars", type=int, default=100)
    benchmark_window_sweep.add_argument("--max-vram-gb", type=float, default=10.0)
    benchmark_window_sweep.add_argument("--playback-speed", type=float, default=1.0)
    benchmark_window_sweep.add_argument("--generation-mode", choices=["precise", "balanced", "fast"], default="balanced")
    benchmark_window_sweep.add_argument("--warmup-text", default="Test")
    benchmark_window_sweep.add_argument("--repeat-count", type=int, default=3)
    benchmark_window_sweep.add_argument("--max-targets", type=int)
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
        pipeline.annotate_chapter(args.chapter, lock_registry=True)
        return 0
    if args.command == "build-global-registry":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=True, fake_tts=True)
        pipeline.registry.initialize_if_missing(book_title=args.book_title, book_slug=args.book_slug)
        pipeline.build_global_registry(book_title=args.book_title)
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
            chapter=args.chapter,
            force_regenerate=args.regenerate_voices,
        )
        return 0
    if args.command == "synthesize-chapter":
        config = PipelineConfig.from_env(book_root=args.book_root)
        pipeline = _build_pipeline(config, needs_llm=False, fake_tts=args.fake_tts)
        annotation = _load_annotation(pipeline, args.chapter)
        pipeline.prepare_voices_for_annotation(
            annotation,
            chapter=args.chapter,
            force_regenerate=args.regenerate_voices,
        )
        if args.rebuild_tts_script or not pipeline.paths.tts_script(args.chapter).exists():
            pipeline.synthesize_chapter(args.chapter, annotation)
        else:
            pipeline.synthesize_chapter_from_tts_script(args.chapter)
        return 0
    if args.command == "benchmark-readalong":
        config = PipelineConfig.from_env(book_root=args.book_root)
        config = replace(config, tts_backend=config.read_along_tts_backend)
        pipeline = _build_pipeline(config, needs_llm=False, fake_tts=False)
        annotation = _load_annotation(pipeline, args.chapter)
        pipeline.prepare_voices_for_annotation(annotation, chapter=args.chapter)
        units_payload = pipeline.build_read_along_units(args.chapter)
        selected = [
            ReadAlongUnit.from_dict(unit)
            for unit in units_payload
            if int(unit["unit_id"]) >= int(args.start_unit)
        ][: int(args.unit_count)]
        session_id = f"benchmark-{args.chapter}-{int(time.time())}"
        paths = BookPaths(args.book_root)
        session = ReadAlongSession(
            session_id=session_id,
            units=selected,
            tts_adapter=pipeline.tts_adapter,
            session_dir=paths.read_along_session_dir(session_id),
            timing_log_path=paths.read_along_timing_log(session_id),
            buffer_limit=2,
            playback_speed=float(args.playback_speed),
            generation_mode=str(args.generation_mode),
            target_buffer_seconds=float(args.target_buffer_seconds),
            start_buffer_seconds=float(args.target_buffer_seconds),
            max_buffer_seconds=float(args.target_buffer_seconds) * 2,
            max_buffer_units=max(1, int(args.unit_count)),
            store_audio_files=False,
        )
        try:
            generated = session.fill_buffer(start_unit_id=0, min_buffer_seconds=float(args.target_buffer_seconds))
            generated_unit_ids = [item.unit_id for item in generated]
            consumed_unit_ids = []
            while len(generated_unit_ids) < len(selected):
                ready = session.consume_ready()
                if ready is not None:
                    consumed_unit_ids.append(ready.unit_id)
                more = session.fill_buffer()
                if not more and ready is None:
                    break
                generated_unit_ids.extend(item.unit_id for item in more)
            summary = {
                "session_id": session_id,
                "chapter": args.chapter,
                "generated_units": generated_unit_ids,
                "simulated_consumed_units": consumed_unit_ids,
                "ready_playback_seconds": session.ready_playback_seconds,
                "timing_log_path": str(paths.read_along_timing_log(session_id)),
            }
            print(json.dumps(summary, sort_keys=True))
        finally:
            close = getattr(pipeline.tts_adapter, "close", None)
            if callable(close):
                close()
        return 0
    if args.command == "benchmark-window-sweep":
        paths = BookPaths(args.book_root)
        perf_log_path = paths.root / "logs" / f"window_sweep_{args.chapter}_qwen_perf.jsonl"
        config = PipelineConfig.from_env(book_root=args.book_root)
        config = replace(
            config,
            qwen_perf_log_path=str(perf_log_path),
            qwen_adaptive_memory_target_gb=float(args.max_vram_gb),
        )

        setup_pipeline = _build_pipeline(config, needs_llm=False, fake_tts=False)
        try:
            annotation = _load_annotation(setup_pipeline, args.chapter)
            setup_pipeline.prepare_voices_for_annotation(annotation, chapter=args.chapter)
            units = [
                ReadAlongUnit.from_dict(unit)
                for unit in setup_pipeline.build_read_along_units(args.chapter)
            ]
        finally:
            setup_close = getattr(setup_pipeline.tts_adapter, "close", None)
            if callable(setup_close):
                setup_close()

        perf_log_path.parent.mkdir(parents=True, exist_ok=True)
        if perf_log_path.exists():
            perf_log_path.unlink()

        def adapter_factory():
            return _build_qwen_adapter(config)

        sweep_config = SweepConfig(
            chapter=args.chapter,
            start_chars=int(args.start_chars),
            step_chars=int(args.step_chars),
            max_vram_gb=float(args.max_vram_gb),
            playback_speed=float(args.playback_speed),
            generation_mode=str(args.generation_mode),
            warmup_text=str(args.warmup_text),
            repeat_count=int(args.repeat_count),
            max_targets=args.max_targets,
        )
        perf_reader = PerfLogReader(perf_log_path)
        result = run_sweep(
            units=units,
            adapter_factory=adapter_factory,
            config=sweep_config,
            machine_profile=collect_machine_profile(config),
            perf_event_reader=perf_reader.read_next,
        )
        output_paths = write_outputs(
            result=result,
            logs_dir=paths.root / "logs",
            docs_dir=Path("docs") / "benchmarks",
            date_slug=date.today().isoformat(),
        )
        print(
            json.dumps(
                {
                    "stop_reason": result.stop_reason,
                    "rows": len(result.rows),
                    "csv": str(output_paths["csv"]),
                    "json": str(output_paths["json"]),
                    "markdown": str(output_paths["markdown"]),
                },
                sort_keys=True,
            )
        )
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2


def _add_book_chapter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--book-root", required=True)
    parser.add_argument("--chapter", required=True)


def _build_pipeline(config: PipelineConfig, needs_llm: bool, fake_tts: bool) -> AudiobookPipeline:
    quote_client = _build_llm_client(config) if needs_llm else _UnavailableJsonClient()
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
        global_registry_service=GlobalRegistryService(
            client=_build_llm_client(config) if needs_llm else _UnavailableJsonClient(),
            failure_logger=FailureLogger(
                config.debug_log_root,
                context={"book_root": config.book_root},
            ),
        ),
        quote_attribution_service=QuoteAttributionService(quote_client) if needs_llm else None,
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
    qwen_model_root = resolve_qwen_model_root(config.qwen_model_root)
    log_runtime_step(
        "build_tts_adapter",
        backend=config.tts_backend,
        book_root=config.book_root,
        qwen_model_root=qwen_model_root,
    )
    if config.tts_backend in {"wsl-vllm-omni", "vllm-omni"}:
        stage_configs_path = resolve_project_path(config.vllm_omni_stage_configs_path)
        log_runtime_step(
            "build_tts_adapter_vllm_omni",
            model=_resolve_vllm_omni_model(config),
            stage_config=stage_configs_path,
            voice_model_root=qwen_model_root,
            wsl_distro=config.wsl_distro,
            wsl_python=config.vllm_omni_wsl_python,
        )
        return WslVllmOmniQwenAdapter(
            book_root=config.book_root,
            distro=config.wsl_distro,
            python_path=config.vllm_omni_wsl_python,
            model=_resolve_vllm_omni_model(config),
            stage_configs_path=stage_configs_path,
            voice_model_root=qwen_model_root,
            voice_python_path=config.wsl_python,
            voice_model_choice=config.qwen_model_choice,
            voice_device="cuda" if config.qwen_device == "auto" else config.qwen_device,
            voice_precision=config.qwen_precision,
            voice_attention=config.qwen_attention,
            timeout_seconds=config.wsl_timeout_seconds,
        )
    if config.tts_backend == "wsl":
        log_runtime_step(
            "build_tts_adapter_wsl_qwen",
            model_root=qwen_model_root,
            wsl_distro=config.wsl_distro,
            wsl_python=config.wsl_python,
            attention=config.qwen_attention,
            precision=config.qwen_precision,
        )
        return WslQwenWorkerAdapter(
            book_root=config.book_root,
            model_root=qwen_model_root,
            distro=config.wsl_distro,
            python_path=config.wsl_python,
            model_choice=config.qwen_model_choice,
            device="cuda" if config.qwen_device == "auto" else config.qwen_device,
            precision=config.qwen_precision,
            attention=config.qwen_attention,
            max_new_tokens=config.qwen_max_new_tokens,
            max_generation_block_chars=config.qwen_max_generation_block_chars,
            max_generation_blocks_per_call=config.qwen_max_generation_blocks_per_call,
            cache_clear_interval=config.qwen_cache_clear_interval,
            streaming_text_mode=config.qwen_streaming_text_mode,
            performance_log_path=Path(config.qwen_perf_log_path) if config.qwen_perf_log_path else None,
            adaptive_memory_target_bytes=(
                int(config.qwen_adaptive_memory_target_gb * (1024 ** 3))
                if config.qwen_adaptive_memory_target_gb is not None
                else None
            ),
            timeout_seconds=config.wsl_timeout_seconds,
        )
    return QwenTtsAdapter(
        model_root=str(qwen_model_root),
        model_choice=config.qwen_model_choice,
        device=config.qwen_device,
        precision=config.qwen_precision,
        attention=config.qwen_attention,
        max_new_tokens=config.qwen_max_new_tokens,
        max_generation_block_chars=config.qwen_max_generation_block_chars,
        max_generation_blocks_per_call=config.qwen_max_generation_blocks_per_call,
        cache_clear_interval=config.qwen_cache_clear_interval,
        streaming_text_mode=config.qwen_streaming_text_mode,
        performance_log_path=config.qwen_perf_log_path,
        adaptive_memory_target_bytes=(
            int(config.qwen_adaptive_memory_target_gb * (1024 ** 3))
            if config.qwen_adaptive_memory_target_gb is not None
            else None
        ),
    )


def _resolve_vllm_omni_model(config: PipelineConfig) -> Path | str:
    local_model = resolve_qwen_model_root(config.qwen_model_root) / "Qwen3-TTS-12Hz-1.7B-Base"
    if local_model.exists():
        return local_model.resolve()
    return config.vllm_omni_model


def _load_annotation(pipeline: AudiobookPipeline, chapter: str) -> AnnotationResult:
    return AnnotationResult.from_dict(read_json(pipeline.paths.annotation(chapter)))


class _UnavailableJsonClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        raise RuntimeError("This CLI command does not perform LLM annotation.")
