from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ebook_tts_pipeline.annotation.booknlp_artifacts import BookNlpQuoteRow
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction
from ebook_tts_pipeline.registry import normalize_name


@dataclass(frozen=True)
class QuoteAttributionCandidate:
    chapter: str
    quote_idx: int
    quote_id: str
    quote_text: str
    booknlp_character_id: str
    mention_phrase: str
    source: str = "booknlp"


def map_booknlp_quotes_to_extraction(
    chapter: str,
    extraction: QuoteExtraction,
    rows: List[BookNlpQuoteRow],
) -> List[QuoteAttributionCandidate]:
    unmatched = list(extraction.quotes)
    candidates: List[QuoteAttributionCandidate] = []
    for row in rows:
        row_key = _quote_key(row.quote_text)
        match = next((quote for quote in unmatched if _quote_key(quote.text) == row_key), None)
        if match is None:
            continue
        unmatched.remove(match)
        candidates.append(
            QuoteAttributionCandidate(
                chapter=chapter,
                quote_idx=match.idx,
                quote_id=match.quote_id,
                quote_text=match.text,
                booknlp_character_id=row.character_id,
                mention_phrase=row.mention_phrase,
            )
        )
    return candidates


def _quote_key(text: str) -> str:
    stripped = str(text).strip().strip("\"'\u201c\u201d\u2018\u2019")
    return normalize_name(stripped)
