from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from ebook_tts_pipeline.annotation.validator import ALLOWED_TYPES
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.registry import normalize_name


def merge_annotation_windows(
    results: List[AnnotationResult],
    registry: Dict[str, Any],
) -> AnnotationResult:
    roles: List[str] = []
    script = []
    new_characters = []
    proposed_new_characters = []
    seen_new_character_names = set()
    seen_proposed_character_names = set()

    for result in results:
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

        for role_idx, type_idx, sentence_idx in result.script:
            role_name = _canonical_role_name(result.roles[role_idx], registry)
            type_name = result.types[type_idx]
            if role_name not in roles:
                roles.append(role_name)
            script.append((roles.index(role_name), ALLOWED_TYPES.index(type_name), sentence_idx))

    return AnnotationResult(
        new_characters=new_characters,
        roles=roles,
        types=list(ALLOWED_TYPES),
        script=script,
        proposed_new_characters=proposed_new_characters,
    )


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
