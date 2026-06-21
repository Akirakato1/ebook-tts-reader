from __future__ import annotations

import argparse
from pathlib import Path

from ebook_tts_pipeline.audio import ChapterAudioBuilder
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager, voice_profile_hash
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal real Qwen TTS smoke test.")
    parser.add_argument("--book-root", default="books/qwen_smoke")
    parser.add_argument("--book-title", default="Qwen Smoke")
    parser.add_argument("--book-slug", default="qwen_smoke")
    parser.add_argument("--chapter", default="chapter_001")
    parser.add_argument("--text", default="This is a tiny Qwen TTS smoke test.")
    parser.add_argument("--regenerate-voice", action="store_true")
    args = parser.parse_args()

    book_root = Path(args.book_root)
    config = PipelineConfig.from_env(str(book_root))
    paths = BookPaths(book_root)
    paths.chapter_text(args.chapter).parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text(args.chapter).write_text(args.text + "\n", encoding="utf-8")

    registry_manager = RegistryManager(paths)
    registry_manager.initialize_if_missing(book_title=args.book_title, book_slug=args.book_slug)
    registry = registry_manager.load()
    narrator = registry["narrator"]
    role_id = str(narrator["role_id"])
    voice_path = paths.voice_qvp(role_id)

    if args.regenerate_voice and voice_path.exists():
        voice_path.unlink()

    adapter = QwenTtsAdapter(
        model_root=config.qwen_model_root,
        model_choice=config.qwen_model_choice,
        device=config.qwen_device,
        precision=config.qwen_precision,
        attention=config.qwen_attention,
    )
    current_hash = voice_profile_hash(narrator)
    if not voice_path.exists() or narrator.get("voice_config_hash") != current_hash:
        adapter.ensure_voice(role_id, narrator, voice_path)
        narrator["voice_config_hash"] = current_hash
    narrator["voice_config_path"] = f"voices/{role_id}.qvp"
    registry_manager.save(registry)

    adapter.role_voice_paths["Narrator"] = voice_path
    adapter.role_voice_paths[role_id] = voice_path
    jobs = [
        {
            "sentence_idx": 0,
            "role": "Narrator",
            "type": "narration",
            "text": args.text,
        }
    ]
    timeline = ChapterAudioBuilder(
        tts_adapter=adapter,
        pause_between_sentences_ms=config.pause_between_sentences_ms,
    ).build_chapter_audio(
        chapter=args.chapter,
        jobs=jobs,
        audio_path=paths.chapter_audio(args.chapter),
        timeline_path=paths.chapter_timeline(args.chapter),
    )
    print(f"Wrote {timeline['audio_path']}")
    print(f"Wrote {paths.chapter_timeline(args.chapter)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
