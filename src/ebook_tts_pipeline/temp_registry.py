from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Set

from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import (
    build_compact_voice_profile,
    normalize_character_profile,
    normalize_name,
    slugify_name,
    voice_variant_for_type,
)
from ebook_tts_pipeline.voice_identity import append_differentiators, choose_differentiators, role_seed


class ChapterTempRegistryManager:
    def __init__(self, paths: BookPaths) -> None:
        self.paths = paths

    def build_for_annotation(
        self,
        chapter: str,
        registry: Dict[str, Any],
        annotation: AnnotationResult,
    ) -> Dict[str, Any]:
        speakers = local_speakers_for_annotation(annotation)
        temp_registry = {
            "chapter": chapter,
            "speakers": {},
        }
        if not speakers:
            return temp_registry

        book_slug = str(registry.get("book", {}).get("slug", "book"))
        used_ids: Set[str] = set()
        for index, speaker in enumerate(speakers, start=1):
            record = _normalize_local_speaker_record(
                chapter=chapter,
                speaker=speaker,
                index=index,
                book_slug=book_slug,
                paths=self.paths,
                used_ids=used_ids,
            )
            temp_registry["speakers"][record["local_id"]] = record
        return temp_registry

    def write_for_annotation(
        self,
        chapter: str,
        registry: Dict[str, Any],
        annotation: AnnotationResult,
    ) -> Dict[str, Any]:
        temp_registry = self.build_for_annotation(chapter, registry, annotation)
        if temp_registry.get("speakers"):
            write_json_atomic(self.paths.chapter_temp_registry(chapter), temp_registry)
        return temp_registry

    def load(self, chapter: str) -> Dict[str, Any]:
        path = self.paths.chapter_temp_registry(chapter)
        if not path.exists():
            return {"chapter": chapter, "speakers": {}}
        return read_json(path)

    def save(self, chapter: str, temp_registry: Dict[str, Any]) -> None:
        if temp_registry.get("speakers"):
            write_json_atomic(self.paths.chapter_temp_registry(chapter), temp_registry)


def normalize_annotation_local_speakers(annotation: AnnotationResult) -> AnnotationResult:
    if not annotation.proposed_new_characters:
        return annotation
    existing = list(annotation.local_speakers)
    seen = {_speaker_key(speaker) for speaker in existing}
    converted: List[Dict[str, Any]] = []
    for index, character in enumerate(annotation.proposed_new_characters, start=len(existing) + 1):
        speaker = _legacy_character_to_local_speaker(character, index)
        key = _speaker_key(speaker)
        if key in seen:
            continue
        seen.add(key)
        converted.append(speaker)
    return replace(
        annotation,
        local_speakers=existing + converted,
        proposed_new_characters=[],
    )


def local_speakers_for_annotation(annotation: AnnotationResult) -> List[Dict[str, Any]]:
    normalized = normalize_annotation_local_speakers(annotation)
    return list(normalized.local_speakers)


def resolve_temp_voice(
    temp_registry: Dict[str, Any],
    role_name: str,
    speech_type: str,
) -> Optional[Dict[str, Any]]:
    normalized = normalize_name(role_name)
    for speaker in temp_registry.get("speakers", {}).values():
        if normalized not in _speaker_lookup_names(speaker):
            continue
        variant_key = _matching_variant(speaker, normalized) or voice_variant_for_type(speech_type)
        variant = speaker.get("voice_variants", {}).get(variant_key)
        if not isinstance(variant, dict):
            return None
        return {
            "character": str(speaker.get("label", role_name)),
            "role": str(variant.get("display_name", role_name)),
            "role_id": str(variant.get("role_id", role_name)),
            "voice_variant": variant_key,
            "voice_record": variant,
        }
    return None


def _normalize_local_speaker_record(
    chapter: str,
    speaker: Dict[str, Any],
    index: int,
    book_slug: str,
    paths: BookPaths,
    used_ids: Set[str],
) -> Dict[str, Any]:
    label = str(speaker.get("label") or speaker.get("name") or "").strip() or f"Temporary speaker {index}"
    local_id = _local_id_for_speaker(speaker, index, used_ids)
    profile = normalize_character_profile(label, speaker.get("profile", {}))
    identity_profile = profile["identity_profile"]
    base_voice = build_compact_voice_profile(label, {"identity_profile": identity_profile})

    variants: Dict[str, Dict[str, Any]] = {}
    for variant_key in ("default", "internal"):
        role_id = f"{slugify_name(chapter)}_{local_id}_{variant_key}"
        differentiators = choose_differentiators(book_slug, role_id)
        voice_profile = dict(base_voice)
        if variant_key == "internal":
            voice_profile = _internal_voice_profile(label, base_voice)
        voice_profile["qwen_instruct"] = append_differentiators(
            str(voice_profile["qwen_instruct"]),
            differentiators,
        )
        variants[variant_key] = {
            "role_id": role_id,
            "display_name": f"{label}_{variant_key}",
            "voice_identity": {
                "seed": role_seed(book_slug, role_id),
                "differentiators": differentiators,
            },
            "voice_profile": voice_profile,
            "voice_config_path": _relative_temp_voice_path(paths, chapter, local_id, variant_key),
        }

    return {
        "local_id": local_id,
        "label": label,
        "profile": {
            "age_stage": identity_profile.get("age_stage", "unknown"),
            "gender": identity_profile.get("gender", "unknown"),
            "personality": list(identity_profile.get("personality", [])),
            "race_or_ethnicity": identity_profile.get("race_or_ethnicity"),
            "accent": identity_profile.get("accent"),
            "occupation": identity_profile.get("occupation"),
        },
        "voice_variants": variants,
    }


def _local_id_for_speaker(speaker: Dict[str, Any], index: int, used_ids: Set[str]) -> str:
    raw = str(speaker.get("local_id", "")).strip()
    base = slugify_name(raw) if raw else f"tmp_{index:03d}"
    local_id = base
    suffix = 2
    while local_id in used_ids:
        local_id = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(local_id)
    return local_id


def _legacy_character_to_local_speaker(character: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "local_id": f"tmp_{index:03d}",
        "label": str(character.get("name", "")).strip() or f"Temporary speaker {index}",
        "profile": dict(character.get("profile", {})) if isinstance(character.get("profile"), dict) else {},
    }


def _speaker_key(speaker: Dict[str, Any]) -> str:
    return normalize_name(str(speaker.get("local_id") or speaker.get("label") or speaker.get("name") or ""))


def _relative_temp_voice_path(paths: BookPaths, chapter: str, local_id: str, variant: str) -> str:
    path = paths.temp_voice_qvp(chapter, local_id, variant)
    return path.relative_to(paths.root).as_posix()


def _speaker_lookup_names(speaker: Dict[str, Any]) -> Set[str]:
    names = {
        normalize_name(str(speaker.get("local_id", ""))),
        normalize_name(str(speaker.get("label", ""))),
    }
    for variant in speaker.get("voice_variants", {}).values():
        if not isinstance(variant, dict):
            continue
        names.add(normalize_name(str(variant.get("role_id", ""))))
        names.add(normalize_name(str(variant.get("role_id", "")).replace("_", " ")))
        names.add(normalize_name(str(variant.get("display_name", ""))))
    names.discard("")
    return names


def _matching_variant(speaker: Dict[str, Any], normalized_name: str) -> str:
    for variant_key, variant in speaker.get("voice_variants", {}).items():
        if not isinstance(variant, dict):
            continue
        names = {
            normalize_name(str(variant.get("display_name", ""))),
            normalize_name(str(variant.get("role_id", ""))),
            normalize_name(str(variant.get("role_id", "")).replace("_", " ")),
        }
        if normalized_name in names:
            return str(variant_key)
    return ""


def _internal_voice_profile(label: str, base_voice: Dict[str, Any]) -> Dict[str, str]:
    description = str(base_voice.get("description", "")).rstrip(". ")
    qwen_instruct = str(base_voice.get("qwen_instruct", "")).rstrip(". ")
    return {
        "description": (
            f"{description}; same {label} identity for internal monologue, "
            "closer, softer, reflective, less projected"
        ).strip("; "),
        "qwen_instruct": (
            f"{qwen_instruct}. Keep the same {label} speaker identity and timbre, "
            "but perform this as internal monologue: closer, softer, inward, reflective, "
            "and less projected. Do not whisper unless the text itself implies whispering."
        ).strip(),
    }
