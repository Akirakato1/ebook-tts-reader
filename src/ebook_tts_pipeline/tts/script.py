from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from ebook_tts_pipeline.annotation.postprocess import normalize_mixed_dialogue_units
from ebook_tts_pipeline.domain import AnnotationResult, SentenceArtifact
from ebook_tts_pipeline.registry import resolve_effective_voice
from ebook_tts_pipeline.temp_registry import resolve_temp_voice
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
class QwenTtsBatch:
    batch_idx: int
    role: str
    role_id: str
    voice_config_path: Optional[str]
    language: str
    sentence_indices: List[int]
    unit_indices: List[int]
    types: List[str]
    text: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_idx": self.batch_idx,
            "role": self.role,
            "role_id": self.role_id,
            "voice_config_path": self.voice_config_path,
            "language": self.language,
            "sentence_indices": self.sentence_indices,
            "unit_indices": self.unit_indices,
            "types": self.types,
            "text": self.text,
        }


@dataclass(frozen=True)
class TtsScriptWindow:
    window_idx: int
    jobs: List[TtsSentenceJob]
    batches: List[QwenTtsBatch]

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
        return sum(len(job.text) for job in self.jobs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "sentence_indices": self.sentence_indices,
            "unit_indices": self.unit_indices,
            "role_count": self.role_count,
            "char_count": self.char_count,
            "jobs": [job.to_dict() for job in self.jobs],
            "batches": [batch.to_dict() for batch in self.batches],
        }


@dataclass(frozen=True)
class TtsScript:
    chapter: str
    jobs: List[TtsSentenceJob]
    windows: List[TtsScriptWindow]

    @property
    def qwen_dialogue_text(self) -> str:
        return render_qwen_dialogue_script([job.to_adapter_job() for job in self.jobs])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "job_count": len(self.jobs),
            "window_count": len(self.windows),
            "qwen_dialogue_text": self.qwen_dialogue_text,
            "jobs": [job.to_dict() for job in self.jobs],
            "windows": [window.to_dict() for window in self.windows],
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
    annotation = normalize_mixed_dialogue_units(annotation, artifact, registry)
    jobs = _build_sentence_jobs(annotation, artifact, registry, temp_registry or {})
    if artifact.units:
        jobs = _split_quote_continuation_jobs(jobs, registry)
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
                batches=_build_qwen_batches(window_jobs, language),
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


def _split_quote_continuation_jobs(
    jobs: List[TtsSentenceJob],
    registry: Dict[str, Any],
) -> List[TtsSentenceJob]:
    if not jobs:
        return []

    narrator_effective = resolve_effective_voice(registry, "Narrator", "narration")
    split_jobs: List[TtsSentenceJob] = []
    open_quote_template: Optional[TtsSentenceJob] = None
    in_quote = False

    for job in jobs:
        segments, in_quote = _quote_state_segments(job.text, starts_in_quote=in_quote)
        if len(segments) == 1 and segments[0][0] == (open_quote_template is not None):
            is_quote, text = segments[0]
            if is_quote and open_quote_template is not None:
                split_jobs.append(_quote_job_like(job, open_quote_template, text))
            else:
                split_jobs.append(job)
            if in_quote and segments[0][0]:
                open_quote_template = open_quote_template or job
            elif not in_quote:
                open_quote_template = None
            continue

        for is_quote, text in segments:
            if not text:
                continue
            if is_quote:
                quote_template = open_quote_template or job
                split_jobs.append(_quote_job_like(job, quote_template, text))
                if in_quote or _has_unclosed_quote(text):
                    open_quote_template = quote_template
                else:
                    open_quote_template = None
            else:
                split_jobs.append(_narrator_job_like(job, text, narrator_effective))
                if not in_quote:
                    open_quote_template = None

        if not in_quote:
            open_quote_template = None

    return split_jobs


def _quote_state_segments(text: str, starts_in_quote: bool) -> Tuple[List[Tuple[bool, str]], bool]:
    segments: List[Tuple[bool, str]] = []
    current: List[str] = []
    current_is_quote = starts_in_quote
    in_quote = starts_in_quote

    for char in text:
        if not in_quote and char in {'"', "\u201c"}:
            _append_quote_segment(segments, current_is_quote, current)
            current = [char]
            current_is_quote = True
            in_quote = True
            continue

        current.append(char)
        if in_quote and char in {'"', "\u201d"}:
            _append_quote_segment(segments, current_is_quote, current)
            current = []
            current_is_quote = False
            in_quote = False

    _append_quote_segment(segments, current_is_quote, current)
    return segments, in_quote


def _append_quote_segment(
    segments: List[Tuple[bool, str]],
    is_quote: bool,
    current: List[str],
) -> None:
    text = "".join(current).strip()
    if text:
        segments.append((is_quote, text))


def _has_unclosed_quote(text: str) -> bool:
    opens = text.count('"') + text.count("\u201c")
    closes = text.count('"') + text.count("\u201d")
    return opens > closes


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


def _quote_job_like(
    job: TtsSentenceJob,
    quote_template: TtsSentenceJob,
    text: str,
) -> TtsSentenceJob:
    return replace(
        job,
        role=quote_template.role,
        role_id=quote_template.role_id,
        character=quote_template.character,
        voice_variant=quote_template.voice_variant,
        type="dialogue",
        text=text,
        voice_config_path=quote_template.voice_config_path,
    )


def _build_qwen_batches(jobs: List[TtsSentenceJob], language: str) -> List[QwenTtsBatch]:
    batches: List[QwenTtsBatch] = []
    current: List[TtsSentenceJob] = []

    for job in jobs:
        if current and job.role != current[-1].role:
            batches.append(_batch_from_jobs(len(batches), current, language))
            current = []
        current.append(job)

    if current:
        batches.append(_batch_from_jobs(len(batches), current, language))

    return batches


def _batch_from_jobs(
    batch_idx: int,
    jobs: List[TtsSentenceJob],
    language: str,
) -> QwenTtsBatch:
    first = jobs[0]
    return QwenTtsBatch(
        batch_idx=batch_idx,
        role=first.role,
        role_id=first.role_id,
        voice_config_path=first.voice_config_path,
        language=language,
        sentence_indices=[job.sentence_idx for job in jobs],
        unit_indices=[job.unit_idx for job in jobs],
        types=[job.type for job in jobs],
        text=[job.text for job in jobs],
    )


def _normalize_script_text(text: str) -> str:
    return " ".join(text.split())
