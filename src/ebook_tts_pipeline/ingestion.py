from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from ebook_tts_pipeline.domain import Sentence, SentenceArtifact
from ebook_tts_pipeline.json_io import write_json_atomic
from ebook_tts_pipeline.paths import BookPaths


CHAPTER_HEADING_RE = re.compile(
    r"(?im)^\s*(chapter\s+([0-9]+|[ivxlcdm]+|[a-z]+)|prologue|epilogue|part\s+[ivxlcdm]+)\s*$"
)


@dataclass(frozen=True)
class ChapterSplitResult:
    chapters: List[str]
    reason: Optional[str] = None


class ChapterSplitter:
    def split_source_book(self, paths: BookPaths) -> ChapterSplitResult:
        text = paths.source_book.read_text(encoding="utf-8")
        matches = list(CHAPTER_HEADING_RE.finditer(text))
        if len(matches) < 2:
            return ChapterSplitResult(chapters=[], reason="low_confidence_chapter_split")

        chapters: List[str] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if len(body) < 20:
                return ChapterSplitResult(chapters=[], reason="low_confidence_chapter_split")
            chapter_id = f"chapter_{index + 1:03d}"
            output_path = paths.chapter_text(chapter_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(body + "\n", encoding="utf-8")
            chapters.append(chapter_id)

        return ChapterSplitResult(chapters=chapters)


class SentenceSegmenter:
    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None) -> None:
        self._tokenizer = tokenizer

    def segment_chapter(self, paths: BookPaths, chapter: str) -> SentenceArtifact:
        text = paths.chapter_text(chapter).read_text(encoding="utf-8")
        raw_sentences = self._tokenize(text)
        sentences = [
            Sentence(idx=index, text=sentence.strip())
            for index, sentence in enumerate(raw_sentences)
            if sentence.strip()
        ]
        artifact = SentenceArtifact(
            chapter=chapter,
            source_path=f"chapters/{chapter}.txt",
            segmenter={
                "name": "nltk.sent_tokenize" if self._tokenizer is None else "custom",
                "language": "english",
                "version": self._segmenter_version(),
            },
            sentences=sentences,
        )
        write_json_atomic(paths.sentence_artifact(chapter), artifact.to_dict())
        return artifact

    def _tokenize(self, text: str) -> List[str]:
        if self._tokenizer is not None:
            return self._tokenizer(text)
        import nltk

        return nltk.sent_tokenize(text)

    def _segmenter_version(self) -> str:
        if self._tokenizer is not None:
            return "test"
        try:
            import nltk

            return nltk.__version__
        except Exception:
            return "unknown"
