from __future__ import annotations

from collections import Counter
from typing import List, Set

from ebook_tts_pipeline.domain import AnnotationResult


ALLOWED_TYPES = ["narration", "dialogue", "thought"]
REQUIRED_PROFILE_STRING_FIELDS = ["age_stage", "gender"]


class AnnotationValidationError(ValueError):
    pass


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _derived_profile_id(name: str, profile: dict) -> str:
    explicit = str(profile.get("profile_id", "")).strip()
    if explicit:
        return explicit
    person_id = str(profile.get("person_id", "")).strip() or name
    age_stage = str(profile.get("age_stage", "")).strip()
    if age_stage:
        return f"{person_id}_{age_stage}"
    return person_id


def validate_annotation(
    result: AnnotationResult,
    expected_sentence_indices: List[int],
    known_names: Set[str],
) -> None:
    errors: List[str] = []
    expected = set(expected_sentence_indices)
    valid_rows = [row for row in result.script if len(row) == 3]
    actual = [row[2] for row in valid_rows]
    actual_set = set(actual)

    missing = sorted(expected - actual_set)
    extra = sorted(actual_set - expected)
    duplicates = sorted(index for index, count in Counter(actual).items() if count > 1)

    for row in result.script:
        if len(row) != 3:
            errors.append(f"script row must have 3 items: {row}")
            continue
        role_idx, type_idx, _ = row
        if role_idx < 0 or role_idx >= len(result.roles):
            errors.append(f"role index out of range: {role_idx}")
        if type_idx < 0 or type_idx >= len(result.types):
            errors.append(f"type index out of range: {type_idx}")

    if result.types != ALLOWED_TYPES:
        errors.append(f"types must be exactly {ALLOWED_TYPES}")
    if missing:
        errors.append(f"missing sentence indexes: {missing}")
    if extra:
        errors.append(f"unknown sentence indexes: {extra}")
    if duplicates:
        errors.append(f"duplicate sentence indexes: {duplicates}")

    if "narration" in result.types:
        narration_idx = result.types.index("narration")
        if any(row[1] == narration_idx for row in valid_rows) and "Narrator" not in result.roles:
            errors.append("roles must include Narrator when narration appears")

    normalized_known = {_normalize_name(name) for name in known_names}
    for character in result.new_characters:
        name = str(character.get("name", "")).strip()
        if not name:
            errors.append("new character is missing name")
            continue
        if _normalize_name(name) in normalized_known:
            errors.append(f"collides with existing character or alias: {name}")
        profile = character.get("profile")
        voice = character.get("voice")
        if not isinstance(profile, dict):
            errors.append(f"new character profile must be an object: {name}")
        else:
            profile_id = _derived_profile_id(name, profile)
            if _normalize_name(profile_id) in normalized_known:
                errors.append(f"collides with existing character or alias: {profile_id}")
            for field in REQUIRED_PROFILE_STRING_FIELDS:
                if not isinstance(profile.get(field), str) or not profile.get(field, "").strip():
                    errors.append(f"new character profile.{field} must be a non-empty string: {name}")
            personality = profile.get("personality")
            if (
                not isinstance(personality, list)
                or not personality
                or any(not isinstance(item, str) or not item.strip() for item in personality)
            ):
                errors.append(f"new character profile.personality must be a non-empty string list: {name}")
        if voice is not None and not isinstance(voice, dict):
            errors.append(f"new character voice must be an object when provided: {name}")
        elif isinstance(voice, dict):
            for field in ("description", "qwen_instruct"):
                if not isinstance(voice.get(field), str) or not voice.get(field, "").strip():
                    errors.append(f"new character voice.{field} must be a non-empty string: {name}")
            qwen_instruct = str(voice.get("qwen_instruct", ""))
            if len(qwen_instruct) > 240:
                errors.append(f"new character voice.qwen_instruct must be compact, 240 chars or fewer: {name}")

    if errors:
        raise AnnotationValidationError("; ".join(errors))
