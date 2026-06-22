from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from ebook_tts_pipeline.annotation.validator import ALLOWED_TYPES
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.registry import normalize_name, slugify_name


def merge_annotation_windows(
    results: List[AnnotationResult],
    registry: Dict[str, Any],
) -> AnnotationResult:
    roles: List[str] = []
    script = []
    new_characters = []
    proposed_new_characters = []
    local_speakers = []
    seen_new_character_names = set()
    seen_proposed_character_names = set()
    local_id_by_label = {}
    used_local_ids = set()

    for result in results:
        local_role_map = {}
        for character in result.new_characters:
            name = str(character.get("name", "")).strip()
            normalized = normalize_name(name)
            if normalized and normalized not in seen_new_character_names:
                new_characters.append(character)
                seen_new_character_names.add(normalized)

        for character in result.proposed_new_characters:
            name = str(character.get("name", "")).strip()
            normalized = normalize_name(name)
            if normalized and normalized not in seen_proposed_character_names:
                proposed_new_characters.append(character)
                seen_proposed_character_names.add(normalized)

        for speaker in result.local_speakers:
            label = str(speaker.get("label") or speaker.get("name") or speaker.get("local_id") or "").strip()
            label_key = normalize_name(label)
            if not label_key:
                continue
            assigned_local_id = local_id_by_label.get(label_key)
            if not assigned_local_id:
                assigned_local_id = _next_local_id(speaker, used_local_ids)
                local_id_by_label[label_key] = assigned_local_id
                copied = dict(speaker)
                copied["local_id"] = assigned_local_id
                copied["label"] = label
                local_speakers.append(copied)
            for name in [
                str(speaker.get("local_id", "")),
                str(speaker.get("label", "")),
                str(speaker.get("name", "")),
                assigned_local_id,
            ]:
                if name.strip():
                    local_role_map[normalize_name(name)] = assigned_local_id

        for role_idx, type_idx, sentence_idx in result.script:
            raw_role_name = result.roles[role_idx]
            role_name = local_role_map.get(normalize_name(raw_role_name)) or _canonical_role_name(raw_role_name, registry)
            type_name = result.types[type_idx]
            if role_name not in roles:
                roles.append(role_name)
            script.append((roles.index(role_name), ALLOWED_TYPES.index(type_name), sentence_idx))

    return AnnotationResult(
        new_characters=new_characters,
        roles=roles,
        types=list(ALLOWED_TYPES),
        script=script,
        local_speakers=local_speakers,
        proposed_new_characters=proposed_new_characters,
    )


def _next_local_id(speaker: Dict[str, Any], used_local_ids: set) -> str:
    raw = str(speaker.get("local_id", "")).strip()
    try:
        base = slugify_name(raw) if raw else ""
    except ValueError:
        base = ""
    if not base:
        base = f"tmp_{len(used_local_ids) + 1:03d}"

    local_id = base
    while local_id in used_local_ids:
        local_id = f"tmp_{len(used_local_ids) + 1:03d}"
    used_local_ids.add(local_id)
    return local_id


def _canonical_role_name(role_name: str, registry: Dict[str, Any]) -> str:
    if normalize_name(role_name) == normalize_name("Narrator"):
        return "Narrator"

    normalized = normalize_name(role_name)
    display_counts = Counter(
        normalize_name(str(record.get("display_name", "")))
        for record in registry.get("characters", {}).values()
        if record.get("display_name")
    )
    for record in registry.get("characters", {}).values():
        display_name = str(record.get("display_name", ""))
        names = _role_lookup_names(record)
        if normalized in {normalize_name(name) for name in names if name}:
            if display_counts[normalize_name(display_name)] <= 1:
                return display_name or role_name
            return _disambiguating_role_name(record, role_name)

    return role_name


def _role_lookup_names(record: Dict[str, Any]) -> List[str]:
    names = [
        str(record.get("display_name", "")),
        str(record.get("role_id", "")),
        str(record.get("role_id", "")).replace("_", " "),
    ]
    names.extend(str(alias) for alias in record.get("aliases", []))
    return names


def _disambiguating_role_name(record: Dict[str, Any], fallback: str) -> str:
    normalized_fallback = normalize_name(fallback)
    aliases = [str(alias) for alias in record.get("aliases", []) if str(alias).strip()]
    for alias in aliases:
        if normalize_name(alias) == normalized_fallback:
            return alias

    display_name = str(record.get("display_name", "")).strip()
    age_stage = str(record.get("age_stage", "")).replace("_", " ").strip()
    if display_name and age_stage and age_stage.lower() != "unknown":
        return f"{display_name} {age_stage}"

    role_id = str(record.get("role_id", "")).strip()
    if role_id:
        return role_id.replace("_", " ")
    return fallback
