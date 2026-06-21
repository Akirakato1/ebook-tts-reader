from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class BookPaths:
    root: Path

    def __init__(self, root: Union[str, Path]) -> None:
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

    def tts_script(self, chapter: str) -> Path:
        return self.root / "tts_scripts" / f"{chapter}.tts_script.json"

    def qwen_script(self, chapter: str) -> Path:
        return self.root / "tts_scripts" / f"{chapter}.qwen_script.txt"

    def chapter_audio(self, chapter: str) -> Path:
        return self.root / "audio" / f"{chapter}.wav"

    def chapter_timeline(self, chapter: str) -> Path:
        return self.root / "audio" / f"{chapter}.timeline.json"

    def voice_qvp(self, role_id: str) -> Path:
        return self.root / "voices" / f"{role_id}.qvp"

    def voice_metadata(self, role_id: str) -> Path:
        return self.root / "voices" / f"{role_id}.json"
