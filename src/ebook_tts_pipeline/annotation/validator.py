from __future__ import annotations

from collections import Counter
from typing import List, Set

from ebook_tts_pipeline.domain import AnnotationResult


ALLOWED_TYPES = ["narration", "dialogue", "thought"]


class AnnotationValidationError(ValueError):
    pass


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def validate_annotation(
    result: AnnotationResult,
    expected_sentence_indices: List[int],
    known_names: Set[str],
) -> None:
    errors: List[str] = []
    expected = set(expected_sentence_indices)
    actual = [row[2] for row in result.script]
    actual_set = set(actual)

    missing = sorted(expected - actual_set)
    extra = sorted(actual_set - expected)
    duplicates = sorted(index for index, count in Counter(actual).items() if count > 1)

    if result.types != ALLOWED_TYPES:
        errors.append(f"types must be exactly {ALLOWED_TYPES}")
    if missing:
        errors.append(f"missing sentence indexes: {missing}")
    if extra:
        errors.append(f"unknown sentence indexes: {extra}")
    if duplicates:
        errors.append(f"duplicate sentence indexes: {duplicates}")

    for row in result.script:
        if len(row) != 3:
            errors.append(f"script row must have 3 items: {row}")
            continue
        role_idx, type_idx, _ = row
        if role_idx < 0 or role_idx >= len(result.roles):
            errors.append(f"role index out of range: {role_idx}")
        if type_idx < 0 or type_idx >= len(result.types):
            errors.append(f"type index out of range: {type_idx}")

    if "narration" in result.types:
        narration_idx = result.types.index("narration")
        if any(row[1] == narration_idx for row in result.script) and "Narrator" not in result.roles:
            errors.append("roles must include Narrator when narration appears")

    normalized_known = {_normalize_name(name) for name in known_names}
    for character in result.new_characters:
        name = str(character.get("name", "")).strip()
        if not name:
            errors.append("new character is missing name")
        elif _normalize_name(name) in normalized_known:
            errors.append(f"collides with existing character or alias: {name}")

    if errors:
        raise AnnotationValidationError("; ".join(errors))
