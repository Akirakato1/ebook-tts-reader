from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

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
    quote_open = False
    for sentence in sentences:
        fragments, quote_open = _scan_quote_fragments(sentence.text, starts_in_quote=quote_open)
        for text in _role_units_from_fragments(fragments):
            units.append(SentenceUnit(idx=len(units), sentence_idx=sentence.idx, text=text))
    return units


def split_dialogue_embedded_text(text: str) -> List[str]:
    fragments, _ = _scan_quote_fragments(text)
    return _role_units_from_fragments(fragments)


@dataclass(frozen=True)
class QuoteFragment:
    text: str
    kind: str


QUOTE_PAIRS = {
    '"': '"',
    "\u201c": "\u201d",
}
CLOSE_QUOTES = {'"', "\u201d"}


def _scan_quote_fragments(text: str, starts_in_quote: bool = False) -> Tuple[List[QuoteFragment], bool]:
    fragments: List[QuoteFragment] = []
    current: List[str] = []
    in_quote = starts_in_quote
    quote_close = ""
    current_kind = "quote" if starts_in_quote else "narration"
    for char in text:
        if not in_quote and char in QUOTE_PAIRS:
            _append_quote_fragment(fragments, current, current_kind)
            current = [char]
            in_quote = True
            quote_close = QUOTE_PAIRS[char]
            current_kind = "quote"
            continue

        current.append(char)
        is_close = char == quote_close or (starts_in_quote and char in CLOSE_QUOTES)
        if in_quote and is_close:
            _append_quote_fragment(fragments, current, current_kind)
            current = []
            in_quote = False
            quote_close = ""
            current_kind = "narration"

    _append_quote_fragment(fragments, current, current_kind)
    return fragments, in_quote


def _append_quote_fragment(
    fragments: List[QuoteFragment],
    current: List[str],
    kind: str,
) -> None:
    text = "".join(current).strip()
    if text:
        fragments.append(QuoteFragment(text=text, kind=kind))


def _role_units_from_fragments(fragments: List[QuoteFragment]) -> List[str]:
    if not fragments:
        return []
    if not any(fragment.kind == "quote" for fragment in fragments):
        joined = _join_nonempty(fragment.text for fragment in fragments)
        return [joined] if joined else []

    units: List[str] = []
    pending_narration = ""
    for fragment in fragments:
        if fragment.kind == "narration":
            pending_narration = _join_unit_text(pending_narration, fragment.text)
            continue

        quote_text = fragment.text
        if pending_narration:
            if units:
                units[-1] = _join_unit_text(units[-1], pending_narration)
            else:
                quote_text = _join_unit_text(pending_narration, quote_text)
            pending_narration = ""
        units.append(quote_text)

    if pending_narration:
        if units:
            units[-1] = _join_unit_text(units[-1], pending_narration)
        else:
            units.append(pending_narration)

    return [unit for unit in units if unit.strip()]


def _join_nonempty(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def _join_unit_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"
