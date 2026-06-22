from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


QUOTE_PAIRS = {
    "\u201c": "\u201d",
    '"': '"',
}


@dataclass(frozen=True)
class QuoteSpan:
    idx: int
    quote_id: str
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NarratorSpan:
    idx: int
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuoteExtraction:
    text: str
    quotes: List[QuoteSpan]
    narrator_spans: List[NarratorSpan]

    def to_dict(self) -> Dict[str, object]:
        return {
            "quotes": [quote.to_dict() for quote in self.quotes],
            "narrator_spans": [span.to_dict() for span in self.narrator_spans],
        }

    def to_marked_text(self) -> str:
        if not self.quotes:
            return self.text

        pieces: List[str] = []
        cursor = 0
        for quote in self.quotes:
            pieces.append(self.text[cursor:quote.start])
            pieces.append(f"|{quote.quote_id}| ")
            pieces.append(self.text[quote.start:quote.end])
            pieces.append(f" ||{quote.quote_id}||")
            cursor = quote.end
        pieces.append(self.text[cursor:])
        return "".join(pieces)


def extract_quoted_dialogue(text: str) -> QuoteExtraction:
    quotes: List[QuoteSpan] = []
    narrator_spans: List[NarratorSpan] = []
    quote_start: Optional[int] = None
    quote_close = ""
    narrator_start = 0
    index = 0

    while index < len(text):
        char = text[index]
        if quote_start is None:
            if char in QUOTE_PAIRS:
                _append_narrator_span(narrator_spans, text, narrator_start, index)
                quote_start = index
                quote_close = QUOTE_PAIRS[char]
            index += 1
            continue

        if char == quote_close:
            quote_end = index + 1
            _append_quote_span(quotes, text, quote_start, quote_end)
            quote_start = None
            quote_close = ""
            narrator_start = quote_end
        index += 1

    if quote_start is not None:
        _append_quote_span(quotes, text, quote_start, len(text))
    else:
        _append_narrator_span(narrator_spans, text, narrator_start, len(text))

    return QuoteExtraction(text=text, quotes=quotes, narrator_spans=narrator_spans)


def _append_quote_span(quotes: List[QuoteSpan], text: str, start: int, end: int) -> None:
    quote_text = text[start:end]
    if not quote_text.strip():
        return
    idx = len(quotes) + 1
    quotes.append(
        QuoteSpan(
            idx=idx,
            quote_id=f"q{idx:03d}",
            start=start,
            end=end,
            text=quote_text,
        )
    )


def _append_narrator_span(spans: List[NarratorSpan], text: str, start: int, end: int) -> None:
    if start >= end:
        return
    span_text = text[start:end].strip()
    if not span_text:
        return
    spans.append(
        NarratorSpan(
            idx=len(spans) + 1,
            start=start,
            end=end,
            text=span_text,
        )
    )
