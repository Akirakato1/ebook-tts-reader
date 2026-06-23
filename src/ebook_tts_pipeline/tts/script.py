from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionResult
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction
from ebook_tts_pipeline.domain import AnnotationResult, SentenceArtifact
from ebook_tts_pipeline.ingestion import fallback_sentence_tokenize
from ebook_tts_pipeline.registry import resolve_effective_voice
from ebook_tts_pipeline.temp_registry import resolve_temp_voice
from ebook_tts_pipeline.tts.text_normalization import normalize_tts_text
from ebook_tts_pipeline.windowing import build_tts_windows


def render_qwen_dialogue_script(jobs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    current_role: Optional[str] = None
    current_text: List[str] = []

    for job in jobs:
        role = str(job["role"])
        text = _normalize_script_text(str(job["text"]))
        if current_role is not None and role != current_role:
            lines.append(f"{current_role}: {' '.join(current_text)}")
            current_text = []
        current_role = role
        if text:
            current_text.append(text)

    if current_role is not None:
        lines.append(f"{current_role}: {' '.join(current_text)}")

    return "\n".join(lines)


@dataclass(frozen=True)
class TtsSentenceJob:
    sentence_idx: int
    unit_idx: int
    role: str
    role_id: str
    character: Optional[str]
    voice_variant: Optional[str]
    type: str
    text: str
    voice_config_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "sentence_idx": self.sentence_idx,
            "unit_idx": self.unit_idx,
            "role": self.role,
            "role_id": self.role_id,
            "type": self.type,
            "text": self.text,
            "voice_config_path": self.voice_config_path,
        }
        if self.character is not None:
            payload["character"] = self.character
        if self.voice_variant is not None:
            payload["voice_variant"] = self.voice_variant
        return payload

    def to_adapter_job(self) -> Dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class TtsScriptWindow:
    window_idx: int
    jobs: List[TtsSentenceJob]
    language: str

    @property
    def sentence_indices(self) -> List[int]:
        return [job.sentence_idx for job in self.jobs]

    @property
    def unit_indices(self) -> List[int]:
        return [job.unit_idx for job in self.jobs]

    @property
    def role_count(self) -> int:
        return len({job.role for job in self.jobs})

    @property
    def char_count(self) -> int:
        return len(self.qwen_text)

    @property
    def qwen_text(self) -> str:
        return render_qwen_dialogue_script([job.to_adapter_job() for job in self.jobs])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "section_idx": self.window_idx,
            "sentence_indices": self.sentence_indices,
            "unit_indices": self.unit_indices,
            "role_count": self.role_count,
            "char_count": self.char_count,
            "language": self.language,
            "qwen_text": self.qwen_text,
            "jobs": [job.to_dict() for job in self.jobs],
        }


@dataclass(frozen=True)
class TtsScript:
    chapter: str
    jobs: List[TtsSentenceJob]
    windows: List[TtsScriptWindow]

    @property
    def qwen_dialogue_text(self) -> str:
        return "\n\n".join(window.qwen_text for window in self.windows)

    def to_dict(self) -> Dict[str, Any]:
        serialized_sections = [window.to_dict() for window in self.windows]
        return {
            "chapter": self.chapter,
            "job_count": len(self.jobs),
            "window_count": len(self.windows),
            "section_count": len(self.windows),
            "qwen_dialogue_text": self.qwen_dialogue_text,
            "jobs": [job.to_dict() for job in self.jobs],
            "windows": serialized_sections,
            "sections": serialized_sections,
        }


def build_tts_script(
    chapter: str,
    annotation: AnnotationResult,
    artifact: SentenceArtifact,
    registry: Dict[str, Any],
    max_chars: int,
    max_roles: int,
    language: str,
    temp_registry: Optional[Dict[str, Any]] = None,
) -> TtsScript:
    jobs = _build_sentence_jobs(annotation, artifact, registry, temp_registry or {})
    jobs = _extract_narrator_context(jobs, registry)
    return _build_script_from_jobs(chapter, jobs, max_chars, max_roles, language)


def build_tts_script_from_quotes(
    chapter: str,
    chapter_text: str,
    extraction: QuoteExtraction,
    attribution: QuoteAttributionResult,
    registry: Dict[str, Any],
    max_chars: int,
    max_roles: int,
    language: str,
    temp_registry: Optional[Dict[str, Any]] = None,
) -> TtsScript:
    quote_attribution = {
        quote_idx: (attribution.roles[role_idx], quote_type)
        for quote_idx, role_idx, quote_type in attribution.quotes
    }
    narrator_effective = resolve_effective_voice(registry, "Narrator", "narration")
    segments: List[Tuple[int, int, str, str, Dict[str, Any]]] = []

    for span in extraction.narrator_spans:
        for part_idx, text in enumerate(_split_narrator_span_text(span.text)):
            segments.append((span.start, part_idx, text, "narration", narrator_effective))

    for quote in extraction.quotes:
        role_name, quote_type = quote_attribution[quote.idx]
        if quote_type == "narrator_quote":
            effective = narrator_effective
            speech_type = "narration"
        else:
            speech_type = "dialogue"
            try:
                effective = resolve_effective_voice(registry, role_name, speech_type)
            except ValueError:
                effective = resolve_temp_voice(temp_registry or {}, role_name, speech_type)
                if effective is None:
                    raise
        segments.append((quote.start, 0, quote.text, speech_type, effective))

    jobs: List[TtsSentenceJob] = []
    for order, (_, __, text, speech_type, effective) in enumerate(sorted(segments, key=lambda item: item[0])):
        record = effective["voice_record"]
        jobs.append(
            TtsSentenceJob(
                sentence_idx=order,
                unit_idx=order,
                role=str(effective["role"]),
                role_id=str(effective["role_id"]),
                character=effective["character"],
                voice_variant=effective["voice_variant"],
                type=speech_type,
                text=text,
                voice_config_path=record.get("voice_config_path"),
            )
        )

    return _build_script_from_jobs(chapter, jobs, max_chars, max_roles, language)


def _split_narrator_span_text(text: str) -> List[str]:
    normalized = text.strip()
    if not normalized:
        return []
    parts = fallback_sentence_tokenize(normalized)
    return parts or [normalized]


def _build_script_from_jobs(
    chapter: str,
    jobs: List[TtsSentenceJob],
    max_chars: int,
    max_roles: int,
    language: str,
) -> TtsScript:
    window_dicts = build_tts_windows(
        [_indexed_adapter_job(index, job) for index, job in enumerate(jobs)],
        max_chars=max_chars,
        max_roles=max_roles,
    )
    windows: List[TtsScriptWindow] = []

    for window_idx, window in enumerate(window_dicts):
        window_jobs = [
            jobs[int(job["_job_order"])]
            for job in window.jobs
        ]
        windows.append(
            TtsScriptWindow(
                window_idx=window_idx,
                jobs=window_jobs,
                language=language,
            )
        )

    return TtsScript(chapter=chapter, jobs=jobs, windows=windows)


def _indexed_adapter_job(index: int, job: TtsSentenceJob) -> Dict[str, Any]:
    payload = job.to_adapter_job()
    payload["_job_order"] = index
    return payload


def _build_sentence_jobs(
    annotation: AnnotationResult,
    artifact: SentenceArtifact,
    registry: Dict[str, Any],
    temp_registry: Dict[str, Any],
) -> List[TtsSentenceJob]:
    unit_by_idx = {unit.idx: unit for unit in artifact.annotation_units}
    jobs: List[TtsSentenceJob] = []

    for role_idx, type_idx, unit_idx in annotation.script:
        if unit_idx not in unit_by_idx:
            raise ValueError(f"unit index not found in sentence artifact: {unit_idx}")
        unit = unit_by_idx[unit_idx]
        role_name = annotation.roles[role_idx]
        type_name = annotation.types[type_idx]
        try:
            effective = resolve_effective_voice(registry, role_name, type_name)
        except ValueError:
            effective = resolve_temp_voice(temp_registry, role_name, type_name)
            if effective is None:
                raise
        record = effective["voice_record"]
        jobs.append(
            TtsSentenceJob(
                sentence_idx=unit.sentence_idx,
                unit_idx=unit.idx,
                role=str(effective["role"]),
                role_id=str(effective["role_id"]),
                character=effective["character"],
                voice_variant=effective["voice_variant"],
                type=type_name,
                text=unit.text,
                voice_config_path=record.get("voice_config_path"),
            )
        )

    return sorted(jobs, key=lambda job: job.unit_idx)


def _extract_narrator_context(
    jobs: List[TtsSentenceJob],
    registry: Dict[str, Any],
) -> List[TtsSentenceJob]:
    if not jobs:
        return []

    narrator_effective = resolve_effective_voice(registry, "Narrator", "narration")
    split_jobs: List[TtsSentenceJob] = []
    for job in jobs:
        split_jobs.extend(_extract_narrator_context_jobs(job, narrator_effective))
    return split_jobs


def _extract_narrator_context_jobs(
    job: TtsSentenceJob,
    narrator_effective: Dict[str, Any],
) -> List[TtsSentenceJob]:
    if job.type not in {"dialogue", "thought"}:
        return [job]

    segments = _quote_context_segments(job.text, default_quote=job.type == "dialogue")
    if len(segments) <= 1:
        return [job]

    jobs: List[TtsSentenceJob] = []
    for is_quote, text in segments:
        if is_quote:
            jobs.append(replace(job, text=text))
        else:
            jobs.append(_narrator_job_like(job, text, narrator_effective))
    return jobs


def _quote_context_segments(text: str, default_quote: bool = False) -> List[Tuple[bool, str]]:
    segments: List[Tuple[bool, str]] = []
    current: List[str] = []
    has_open_quote = any(char in text for char in {'"', "\u201c"})
    has_close_quote = any(char in text for char in {'"', "\u201d"})
    starts_as_quote = default_quote and not has_open_quote and has_close_quote
    current_is_quote = starts_as_quote
    in_quote = False

    for char in text:
        if not in_quote and char in {'"', "\u201c"}:
            _append_quote_segment(segments, current_is_quote, current)
            current = [char]
            current_is_quote = True
            in_quote = True
            continue

        current.append(char)
        is_default_close = starts_as_quote and not in_quote and char in {'"', "\u201d"}
        if (in_quote and char in {'"', "\u201d"}) or is_default_close:
            _append_quote_segment(segments, current_is_quote, current)
            current = []
            current_is_quote = False
            in_quote = False

    _append_quote_segment(segments, current_is_quote, current)
    return segments


def _append_quote_segment(
    segments: List[Tuple[bool, str]],
    is_quote: bool,
    current: List[str],
) -> None:
    text = "".join(current).strip()
    if text:
        segments.append((is_quote, text))


def _narrator_job_like(
    job: TtsSentenceJob,
    text: str,
    narrator_effective: Dict[str, Any],
) -> TtsSentenceJob:
    record = narrator_effective["voice_record"]
    return replace(
        job,
        role=str(narrator_effective["role"]),
        role_id=str(narrator_effective["role_id"]),
        character=narrator_effective["character"],
        voice_variant=narrator_effective["voice_variant"],
        type="narration",
        text=text,
        voice_config_path=record.get("voice_config_path"),
    )


def _normalize_script_text(text: str) -> str:
    return normalize_tts_text(text)
