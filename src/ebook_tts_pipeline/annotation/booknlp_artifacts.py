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
class BookNlpEntityRow:
    character_id: str
    mention_text: str
    start_token: int = -1
    end_token: int = -1
    category: str = ""
    property_type: str = ""


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


def parse_booknlp_entities(path: str | Path) -> List[BookNlpEntityRow]:
    rows: List[BookNlpEntityRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            character_id = _first_text(row, "COREF", "coref", "char_id", "character_id", "characterId")
            mention = _first_text(row, "text", "mention", "mention_phrase", "phrase")
            if not character_id or not mention:
                continue
            rows.append(
                BookNlpEntityRow(
                    character_id=character_id,
                    mention_text=mention,
                    start_token=_int_field(row, "start_token", "start", "token_start"),
                    end_token=_int_field(row, "end_token", "end", "token_end"),
                    category=_first_text(row, "cat", "category", "ner"),
                    property_type=_first_text(row, "prop", "property", "type"),
                )
            )
    return rows


def character_aliases_from_entities(rows: List[BookNlpEntityRow]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}
    for row in rows:
        normalized = _alias_key(row.mention_text)
        if not normalized:
            continue
        character_seen = seen.setdefault(row.character_id, set())
        if normalized in character_seen:
            continue
        aliases.setdefault(row.character_id, []).append(row.mention_text)
        character_seen.add(normalized)
    return aliases


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


def _first_text(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _int_field(row: Dict[str, str], *keys: str) -> int:
    text = _first_text(row, *keys)
    if not text:
        return -1
    return int(text)


def _alias_key(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())
