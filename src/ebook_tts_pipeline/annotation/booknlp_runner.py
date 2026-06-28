from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from ebook_tts_pipeline.annotation.booknlp_artifacts import StitchedBookText, stitch_chapters_for_booknlp
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths


@dataclass(frozen=True)
class BookNlpRunnerConfig:
    python: str = "python"
    model: str = "small"
    cache_policy: str = "reuse"


@dataclass(frozen=True)
class BookNlpRunResult:
    output_dir: Path
    input_path: Path
    manifest_path: Path
    input_hash: str
    cache_hit: bool


class BookNlpRunner:
    def __init__(
        self,
        config: BookNlpRunnerConfig,
        run_command: Callable[[List[str]], None] | None = None,
    ) -> None:
        self.config = config
        self.run_command = run_command or _run_command

    def ensure_booknlp_artifacts(self, paths: BookPaths, chapters: List[str]) -> BookNlpRunResult:
        stitched = self._stitch_chapter_files(paths, chapters)
        input_hash = _text_hash(stitched.text)
        paths.booknlp_dir.mkdir(parents=True, exist_ok=True)
        paths.booknlp_output_dir.mkdir(parents=True, exist_ok=True)
        if self._cache_hit(paths, input_hash, chapters):
            return BookNlpRunResult(
                output_dir=paths.booknlp_output_dir,
                input_path=paths.booknlp_input,
                manifest_path=paths.booknlp_manifest,
                input_hash=input_hash,
                cache_hit=True,
            )
        paths.booknlp_input.write_text(stitched.text, encoding="utf-8")
        self.run_command(self._booknlp_command(paths))
        write_json_atomic(
            paths.booknlp_manifest,
            {
                "input_hash": input_hash,
                "model": self.config.model,
                "cache_policy": self.config.cache_policy,
                "chapters": list(chapters),
                "chapter_count": len(chapters),
                "chapter_offsets": {
                    chapter: {
                        "marker_start": offset.marker_start,
                        "marker_end": offset.marker_end,
                        "content_start": offset.content_start,
                        "content_end": offset.content_end,
                    }
                    for chapter, offset in stitched.chapter_offsets.items()
                },
            },
        )
        return BookNlpRunResult(
            output_dir=paths.booknlp_output_dir,
            input_path=paths.booknlp_input,
            manifest_path=paths.booknlp_manifest,
            input_hash=input_hash,
            cache_hit=False,
        )

    def _stitch_chapter_files(self, paths: BookPaths, chapters: List[str]) -> StitchedBookText:
        chapter_texts = {
            chapter: paths.chapter_text(chapter).read_text(encoding="utf-8", errors="replace")
            for chapter in chapters
        }
        return stitch_chapters_for_booknlp(chapter_texts)

    def _cache_hit(self, paths: BookPaths, input_hash: str, chapters: List[str]) -> bool:
        if self.config.cache_policy != "reuse" or not paths.booknlp_manifest.exists():
            return False
        manifest = read_json(paths.booknlp_manifest)
        return (
            manifest.get("input_hash") == input_hash
            and manifest.get("model") == self.config.model
            and manifest.get("chapters") == list(chapters)
        )

    def _booknlp_command(self, paths: BookPaths) -> List[str]:
        script = (
            "from booknlp.booknlp import BookNLP; "
            f"BookNLP('en', {{'pipeline':'entity,quote,coref','model':'{self.config.model}'}})"
            f".process(r'{paths.booknlp_input}', r'{paths.booknlp_output_dir}', 'book')"
        )
        return [self.config.python, "-c", script]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_command(command: List[str]) -> None:
    subprocess.check_call(command)
