from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping


@dataclass(frozen=True)
class BookNlpQuoteRow:
    quote_start_token: int
    quote_end_token: int
    mention_start_token: int
    mention_end_token: int
    mention_phrase: str
    character_id: str
    quote_text: str


@dataclass(frozen=True)
class ChapterOffset:
    chapter: str
    marker_start: int
    marker_end: int
    content_start: int
    content_end: int


@dataclass(frozen=True)
class StitchedBookText:
    text: str
    chapter_offsets: Dict[str, ChapterOffset]


def parse_booknlp_quotes(path: str | Path) -> List[BookNlpQuoteRow]:
    rows: List[BookNlpQuoteRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                BookNlpQuoteRow(
                    quote_start_token=int(row["quote_start"]),
                    quote_end_token=int(row["quote_end"]),
                    mention_start_token=int(row["mention_start"]),
                    mention_end_token=int(row["mention_end"]),
                    mention_phrase=str(row.get("mention_phrase") or ""),
                    character_id=str(row.get("char_id") or ""),
                    quote_text=str(row.get("quote") or ""),
                )
            )
    return rows


def stitch_chapters_for_booknlp(chapters: Mapping[str, str]) -> StitchedBookText:
    parts: List[str] = []
    offsets: Dict[str, ChapterOffset] = {}
    cursor = 0
    for chapter, text in chapters.items():
        if parts:
            parts.append("\n\n")
            cursor += 2
        marker = f"[{chapter}]\n"
        marker_start = cursor
        marker_end = marker_start + len(marker)
        parts.append(marker)
        cursor = marker_end
        content_start = cursor
        chapter_text = str(text)
        parts.append(chapter_text)
        cursor += len(chapter_text)
        offsets[str(chapter)] = ChapterOffset(
            chapter=str(chapter),
            marker_start=marker_start,
            marker_end=marker_end,
            content_start=content_start,
            content_end=cursor,
        )
    return StitchedBookText(text="".join(parts), chapter_offsets=offsets)
