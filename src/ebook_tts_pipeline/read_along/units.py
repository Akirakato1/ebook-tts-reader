from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionResult
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction
from ebook_tts_pipeline.ingestion import fallback_sentence_tokenize
from ebook_tts_pipeline.registry import resolve_effective_voice
from ebook_tts_pipeline.temp_registry import resolve_temp_voice


FUNCTIONAL_NARRATOR_ROLE = "Functional Narrator"
FUNCTIONAL_NARRATOR_ROLE_ID = "functional_narrator"
FUNCTIONAL_NARRATOR_VARIANT = "functional_narrator"


@dataclass(frozen=True)
class ReadAlongUnit:
    chapter: str
    unit_id: int
    text: str
    source_start: int
    source_end: int
    role: str
    role_id: str
    type: str
    voice_config_path: Optional[str]
    quote_id: Optional[str] = None
    sentence_idx: Optional[int] = None
    character: Optional[str] = None
    voice_variant: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReadAlongUnit":
        return cls(
            chapter=str(data["chapter"]),
            unit_id=int(data["unit_id"]),
            text=str(data["text"]),
            source_start=int(data["source_start"]),
            source_end=int(data["source_end"]),
            role=str(data["role"]),
            role_id=str(data["role_id"]),
            type=str(data["type"]),
            voice_config_path=data.get("voice_config_path"),
            quote_id=data.get("quote_id"),
            sentence_idx=int(data["sentence_idx"]) if data.get("sentence_idx") is not None else None,
            character=data.get("character"),
            voice_variant=data.get("voice_variant"),
        )

    def to_tts_job(self) -> Dict[str, Any]:
        payload = {
            "sentence_idx": self.unit_id,
            "unit_idx": self.unit_id,
            "role": self.role,
            "role_id": self.role_id,
            "type": self.type,
            "text": self.text,
            "voice_config_path": self.voice_config_path,
            "_read_along_source_start": self.source_start,
            "_read_along_source_end": self.source_end,
        }
        if self.character is not None:
            payload["character"] = self.character
        if self.voice_variant is not None:
            payload["voice_variant"] = self.voice_variant
        return payload


def build_read_along_units(
    chapter: str,
    chapter_text: str,
    extraction: QuoteExtraction,
    attribution: QuoteAttributionResult,
    registry: Dict[str, Any],
    temp_registry: Dict[str, Any],
) -> List[ReadAlongUnit]:
    quote_roles = {
        quote_idx: (attribution.roles[role_idx], quote_type)
        for quote_idx, role_idx, quote_type in attribution.quotes
    }
    narrator_effective = resolve_effective_voice(registry, "Narrator", "narration")
    segments: List[Tuple[int, int, str, str, Dict[str, Any], Optional[str]]] = []

    for span in extraction.narrator_spans:
        segments.extend(
            _split_narrator_span(
                chapter_text=chapter_text,
                start=span.start,
                end=span.end,
                effective=narrator_effective,
            )
        )

    for quote in extraction.quotes:
        role_name, quote_type = quote_roles[quote.idx]
        if quote_type == "narrator_quote":
            effective = _functional_narrator_effective(narrator_effective)
            speech_type = "narration"
        else:
            speech_type = "dialogue"
            try:
                effective = resolve_effective_voice(registry, role_name, speech_type)
            except ValueError:
                effective = resolve_temp_voice(temp_registry, role_name, speech_type)
                if effective is None:
                    raise
        segments.append((quote.start, quote.end, quote.text.strip(), speech_type, effective, quote.quote_id))

    units: List[ReadAlongUnit] = []
    for unit_id, (start, end, text, speech_type, effective, quote_id) in enumerate(
        sorted(segments, key=lambda item: (item[0], item[1]))
    ):
        record = effective["voice_record"]
        units.append(
            ReadAlongUnit(
                chapter=chapter,
                unit_id=unit_id,
                text=text,
                source_start=start,
                source_end=end,
                role=str(effective["role"]),
                role_id=str(effective["role_id"]),
                type=speech_type,
                voice_config_path=record.get("voice_config_path"),
                quote_id=quote_id,
                sentence_idx=unit_id,
                character=effective["character"],
                voice_variant=effective["voice_variant"],
            )
        )
    return units


def _functional_narrator_effective(narrator_effective: Dict[str, Any]) -> Dict[str, Any]:
    voice_record = dict(narrator_effective.get("voice_record", {}))
    voice_record["voice_config_path"] = None
    return {
        "role": FUNCTIONAL_NARRATOR_ROLE,
        "role_id": FUNCTIONAL_NARRATOR_ROLE_ID,
        "character": None,
        "voice_variant": FUNCTIONAL_NARRATOR_VARIANT,
        "voice_record": voice_record,
    }


def _split_narrator_span(
    chapter_text: str,
    start: int,
    end: int,
    effective: Dict[str, Any],
) -> List[Tuple[int, int, str, str, Dict[str, Any], Optional[str]]]:
    raw = chapter_text[start:end]
    stripped = raw.strip()
    if not stripped:
        return []
    leading = len(raw) - len(raw.lstrip())
    search_start = start + leading
    parts = fallback_sentence_tokenize(stripped) or [stripped]
    segments: List[Tuple[int, int, str, str, Dict[str, Any], Optional[str]]] = []
    cursor = search_start
    for part in parts:
        found = chapter_text.find(part, cursor, end)
        if found < 0:
            found = cursor
        part_end = found + len(part)
        segments.append((found, part_end, part, "narration", effective, None))
        cursor = part_end
    return segments
