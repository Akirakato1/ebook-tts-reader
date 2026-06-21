from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from ebook_tts_pipeline.domain import Sentence, SentenceArtifact, SentenceUnit
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
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        allow_nltk_download: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._allow_nltk_download = allow_nltk_download

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
            units=split_sentence_units(sentences),
        )
        write_json_atomic(paths.sentence_artifact(chapter), artifact.to_dict())
        return artifact

    def _tokenize(self, text: str) -> List[str]:
        if self._tokenizer is not None:
            return self._tokenizer(text)
        import nltk

        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            if self._allow_nltk_download:
                nltk.download("punkt", quiet=True)
                return nltk.sent_tokenize(text)
            return fallback_sentence_tokenize(text)

    def _segmenter_version(self) -> str:
        if self._tokenizer is not None:
            return "test"
        try:
            import nltk

            return nltk.__version__
        except Exception:
            return "unknown"


def fallback_sentence_tokenize(text: str) -> List[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'“‘A-Z0-9])", normalized)
    return [part.strip() for part in parts if part.strip()]


def split_sentence_units(sentences: List[Sentence]) -> List[SentenceUnit]:
    units: List[SentenceUnit] = []
    for sentence in sentences:
        for fragment in split_dialogue_embedded_text(sentence.text):
            units.append(SentenceUnit(idx=len(units), sentence_idx=sentence.idx, text=fragment))
    return units


def split_dialogue_embedded_text(text: str) -> List[str]:
    fragments = _quote_split_fragments(text)
    if len(fragments) <= 1:
        return [text.strip()] if text.strip() else []
    return fragments


def _quote_split_fragments(text: str) -> List[str]:
    fragments: List[str] = []
    current: List[str] = []
    in_quote = False
    quote_close = ""
    opened_quote = False
    closed_quote = False

    for char in text:
        is_open_quote = char in {'"', "“"}
        is_close_quote = in_quote and char == quote_close
        if is_open_quote and not in_quote:
            _append_fragment(fragments, current)
            current = [char]
            in_quote = True
            quote_close = "”" if char == "“" else char
            opened_quote = True
            continue

        current.append(char)
        if is_close_quote:
            _append_fragment(fragments, current)
            current = []
            in_quote = False
            quote_close = ""
            closed_quote = True

    if in_quote or (opened_quote and not closed_quote):
        return [text.strip()] if text.strip() else []

    _append_fragment(fragments, current)
    return fragments


def _append_fragment(fragments: List[str], current: List[str]) -> None:
    fragment = "".join(current).strip()
    if fragment:
        fragments.append(fragment)
