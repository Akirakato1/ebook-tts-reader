from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ebook_tts_pipeline.domain import AnnotationResult, SentenceArtifact
from ebook_tts_pipeline.registry import normalize_name
from ebook_tts_pipeline.windowing import build_tts_windows


@dataclass(frozen=True)
class TtsSentenceJob:
    sentence_idx: int
    role: str
    role_id: str
    type: str
    text: str
    voice_config_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_idx": self.sentence_idx,
            "role": self.role,
            "role_id": self.role_id,
            "type": self.type,
            "text": self.text,
            "voice_config_path": self.voice_config_path,
        }

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
    def role_count(self) -> int:
        return len({job.role for job in self.jobs})

    @property
    def char_count(self) -> int:
        return sum(len(job.text) for job in self.jobs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "sentence_indices": self.sentence_indices,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "job_count": len(self.jobs),
            "window_count": len(self.windows),
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
) -> TtsScript:
    jobs = _build_sentence_jobs(annotation, artifact, registry)
    window_dicts = build_tts_windows(
        [job.to_adapter_job() for job in jobs],
        max_chars=max_chars,
        max_roles=max_roles,
    )
    job_by_sentence_idx = {job.sentence_idx: job for job in jobs}
    windows: List[TtsScriptWindow] = []

    for window_idx, window in enumerate(window_dicts):
        window_jobs = [
            job_by_sentence_idx[int(job["sentence_idx"])]
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


def _build_sentence_jobs(
    annotation: AnnotationResult,
    artifact: SentenceArtifact,
    registry: Dict[str, Any],
) -> List[TtsSentenceJob]:
    sentence_by_idx = {sentence.idx: sentence.text for sentence in artifact.sentences}
    role_records = _role_records(registry)
    jobs: List[TtsSentenceJob] = []

    for role_idx, type_idx, sentence_idx in annotation.script:
        if sentence_idx not in sentence_by_idx:
            raise ValueError(f"sentence index not found in sentence artifact: {sentence_idx}")
        role_name = annotation.roles[role_idx]
        type_name = annotation.types[type_idx]
        record = _lookup_role_record(role_name, role_records)
        jobs.append(
            TtsSentenceJob(
                sentence_idx=sentence_idx,
                role=str(record.get("display_name", role_name)),
                role_id=str(record.get("role_id", role_name)),
                type=type_name,
                text=sentence_by_idx[sentence_idx],
                voice_config_path=record.get("voice_config_path"),
            )
        )

    return sorted(jobs, key=lambda job: job.sentence_idx)


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
        types=[job.type for job in jobs],
        text=[job.text for job in jobs],
    )


def _role_records(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    narrator = dict(registry.get("narrator", {}))
    if narrator:
        _add_role_record(records, narrator)
    for character in registry.get("characters", {}).values():
        _add_role_record(records, dict(character))
    return records


def _add_role_record(records: Dict[str, Dict[str, Any]], record: Dict[str, Any]) -> None:
    names = [
        str(record.get("display_name", "")),
        str(record.get("role_id", "")),
        str(record.get("role_id", "")).replace("_", " "),
    ]
    names.extend(str(alias) for alias in record.get("aliases", []))
    for name in names:
        normalized = normalize_name(name)
        if normalized:
            records[normalized] = record


def _lookup_role_record(
    role_name: str,
    records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = normalize_name(role_name)
    if normalized not in records:
        raise ValueError(f"No registry record exists for annotated role: {role_name}")
    return records[normalized]
